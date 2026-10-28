#!/usr/bin/env python3
"""Stress-test the deployed classifier outside the studio-white-background benchmark.

TrashNet images are objects photographed against a white background, filling the
frame. A photo taken at a point of disposal is none of those things. This script
measures how far the deployed model degrades when that assumption is broken, so
the report can state the limitation with numbers instead of adjectives.

It trains nothing and changes nothing: it loads the shipped model and the saved
held-out split, applies perturbations, and prints a table.

    python stress_test.py

Writes outputs/verified/stress_test.json and stress_test.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ecosort.classifier import CNN_MODEL_FILES

HERE = Path(__file__).resolve().parent

def _find_dir(name: str, start: Path) -> Path:
    """Locate a project folder from wherever this script sits.

    The scripts live in ``scripts/`` in the repository and at the project root in a
    working copy, so a fixed relative path breaks in one of the two. Search this
    folder first, then the parents.
    """
    for base in [start] + list(start.parents):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return start / name


CNN_INPUT = 224
TABLE_BG = (150, 148, 145)


def load_split(data_dir: Path, model_dir: Path):
    names = json.loads((model_dir / "class_names.json").read_text())["class_names"]
    indices = json.loads((model_dir / "split_indices.json").read_text())["test"]

    files = []
    for label, name in enumerate(names):
        for path in sorted((data_dir / name).iterdir()):
            if path.name.startswith("._"):
                continue
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((path, label))

    images = [Image.open(files[i][0]).convert("RGB") for i in indices]
    labels = np.array([files[i][1] for i in indices])
    return names, images, labels


def onto_table(image: Image.Image) -> Image.Image:
    """Replace the studio-white background with a mid-grey surface tone."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    array[array.min(2) > 235] = TABLE_BG
    return Image.fromarray(array.astype("uint8"))


def phone_like(image: Image.Image, background: Image.Image) -> Image.Image:
    """What a phone photo actually looks like: smaller object, cluttered background.

    The object is cut out with the same white-background threshold, scaled to 55%
    and pasted off-centre onto another scene.
    """
    canvas = np.asarray(background.convert("RGB").resize(image.size)).astype(np.float32).copy()
    small = image.resize((int(image.width * 0.55), int(image.height * 0.55)),
                         Image.Resampling.BILINEAR)
    patch = np.asarray(small).astype(np.float32)
    keep = patch.min(2) <= 235
    y0, x0 = int(canvas.shape[0] * 0.30), int(canvas.shape[1] * 0.25)
    region = canvas[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
    region[keep] = patch[keep]
    return Image.fromarray(canvas.astype("uint8"))


def encode(image: Image.Image) -> np.ndarray:
    return np.asarray(image.resize((CNN_INPUT, CNN_INPUT), Image.Resampling.BILINEAR),
                      dtype="float32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "stress test").splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=_find_dir("models", HERE))
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "work" / "dataset-resized")
    parser.add_argument("--out-dir", type=Path, default=_find_dir("outputs", HERE) / "verified")
    args = parser.parse_args(argv)

    try:
        # Keras 3 is imported directly: tensorflow re-exports it lazily, so the
        # attribute is invisible to a static checker even though it exists.
        import keras
        import tensorflow as tf
        from keras import Model
    except Exception as exc:                     # noqa: BLE001 - report, do not crash
        raise SystemExit(f"TensorFlow is needed for the stress test: {exc}") from None

    tf.get_logger().setLevel("ERROR")
    names, images, labels = load_split(args.data_dir, args.model_dir)
    theta = json.loads((args.model_dir / "metrics.json").read_text())["theta"]["chosen"]
    # Follow the same preference order as the deployed classifier, so this can
    # never silently measure a stale artefact sitting next to the current one.
    candidates = [name for name in CNN_MODEL_FILES if (args.model_dir / name).is_file()]
    if not candidates:
        raise SystemExit(f"no CNN artefact in {args.model_dir}: wanted one of {CNN_MODEL_FILES}")
    artefact = args.model_dir / candidates[0]
    print(f"artefact: {artefact.name}")
    model = cast(Model, keras.models.load_model(artefact, compile=False))

    conditions = {
        "baseline (studio white background)": lambda i: images[i],
        "grey surface instead of white": lambda i: onto_table(images[i]),
        "dimmer, as indoors (x0.75)": lambda i: ImageEnhance.Brightness(images[i]).enhance(0.75),
        "soft focus (phone autofocus)": lambda i: images[i].filter(ImageFilter.GaussianBlur(1.2)),
        "phone framing: smaller object, cluttered scene":
            lambda i: phone_like(images[i], images[(i + 97) % len(images)]),
    }

    rows = []
    print(f"theta = {theta}   held-out items = {len(images)}\n")
    print(f"{'condition':46s} {'accuracy':>9s} {'mean confidence':>16s} {'to review':>10s}")
    print("-" * 86)
    for name, perturb in conditions.items():
        batch = np.stack([encode(perturb(i)) for i in range(len(images))])
        # Keras accepts verbose=0 at runtime; its 3.x stub types the parameter as
        # str only, so the numeric value is flagged despite being documented API.
        probs = model.predict(batch, batch_size=32, verbose=0)  # pyright: ignore[reportArgumentType]
        confidence = probs.max(1)
        accuracy = float((probs.argmax(1) == labels).mean())
        review = float((confidence < theta).mean())
        rows.append({"condition": name, "accuracy": round(accuracy, 4),
                     "mean_confidence": round(float(confidence.mean()), 4),
                     "review_rate": round(review, 4)})
        print(f"{name:46s} {accuracy:9.3f} {confidence.mean():16.3f} {review:10.1%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stress_test.json").write_text(
        json.dumps({"theta": theta, "n": len(images), "conditions": rows}, indent=2))

    try:
        import matplotlib  # pyright: ignore[reportMissingImports]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        labels_ = [r["condition"] for r in rows]
        values = [r["accuracy"] for r in rows]
        ax.barh(labels_, values, color="#2E7D32")
        ax.axvline(0.85, ls="--", c="#C62828", lw=1.1, label="85% target")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("accuracy on the same held-out items")
        ax.set_title("Where the model stops working: sensitivity to capture conditions")
        for index, value in enumerate(values):
            ax.text(value + 0.01, index, f"{value:.2f}", va="center", fontsize=8)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(args.out_dir / "stress_test.png", dpi=130)
        plt.close(fig)
        print(f"\nfigure -> {args.out_dir / 'stress_test.png'}")
    except ImportError:
        print("\nmatplotlib not available; numbers written to JSON only")

    print(f"results -> {args.out_dir / 'stress_test.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
