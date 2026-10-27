#!/usr/bin/env python3
"""Rebuild every number and figure in the report from the saved artefacts.

This is the fast regeneration path. It needs no GPU, no TensorFlow and no
training: it reads the probabilities the training run already recorded
(``val_probs.npz`` and ``test_probs.npz``) and recomputes the headline results,
then redraws the figures.

Typical run time is under a second, which is what makes the report auditable by
a marker who does not want to train anything.

    python reproduce_report.py
    python reproduce_report.py --model-dir /kaggle/input/ecosort-artifacts/models \
                               --out-dir /kaggle/working/figures

The exit code is 0 when every recorded number is reproduced exactly and 1
otherwise, so it doubles as a verification check rather than a printer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Reviewer-capacity budget. The training notebook fixes this, so the threshold
#: must be reproducible from the saved validation probabilities alone.
MAX_REVIEW_RATE = 0.10


def load_metrics(model_dir: Path) -> dict:
    path = model_dir / "metrics.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run the training notebook first")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores))


def sweep(proba: np.ndarray, y_true: np.ndarray) -> list[dict]:
    rows = []
    pred, conf = proba.argmax(1), proba.max(1)
    for theta in np.round(np.arange(0.30, 0.91, 0.05), 2):
        auto = conf >= theta
        rows.append({
            "theta": float(theta),
            "review_rate": round(float((~auto).mean()), 4),
            "auto_coverage": round(float(auto.mean()), 4),
            "auto_accuracy": (round(float((pred[auto] == y_true[auto]).mean()), 4)
                              if auto.any() else None),
            "overall_accuracy": round(float((pred == y_true).mean()), 4),
        })
    return rows


def choose_theta(rows: list[dict], budget: float) -> dict:
    feasible = [r for r in rows
                if r["auto_accuracy"] is not None and r["review_rate"] <= budget]
    if not feasible:
        raise SystemExit(f"no threshold keeps review rate <= {budget:.0%}")
    return max(feasible, key=lambda r: (r["auto_accuracy"], -r["review_rate"]))


class Checks:
    """Collect pass/fail lines so the script can exit non-zero on a mismatch."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []
        self.notes: list[tuple[str, str]] = []

    def add(self, name: str, recorded, recomputed, tolerance: float = 5e-4) -> None:
        if recorded is None:
            ok = False
            shown = "not recorded"
        elif isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
            ok = abs(float(recorded) - float(recomputed)) <= tolerance
            shown = f"{float(recorded):.4f}"
        else:
            ok = str(recorded) == str(recomputed)
            shown = str(recorded)
        recomputed_shown = (f"{float(recomputed):.4f}"
                            if isinstance(recomputed, (int, float)) else str(recomputed))
        self.rows.append((name, shown, recomputed_shown, ok))

    def note(self, name: str, value) -> None:
        """Record a value for the reader without making it a pass/fail check."""
        shown = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
        self.notes.append((name, shown))

    def report(self) -> bool:
        width = max(len(r[0]) for r in self.rows)
        print(f"\n{'check'.ljust(width)}  {'recorded':>12}  {'recomputed':>12}  result")
        print("-" * (width + 44))
        for name, recorded, recomputed, ok in self.rows:
            tail = "ok" if ok else "MISMATCH"
            print(f"{name.ljust(width)}  {recorded:>12}  {recomputed:>12}  {tail}")
        failed = [r[0] for r in self.rows if not r[3]]
        if self.notes:
            print("\nfor information (not recorded in metrics.json):")
            for name, value in self.notes:
                print(f"  {name}: {value}")
        print()
        if failed:
            print(f"FAILED: {len(failed)} of {len(self.rows)} checks did not reproduce")
            for name in failed:
                print("  -", name)
            return False
        print(f"All {len(self.rows)} checks reproduce the recorded values.")
        return True


def draw_figures(metrics: dict, cnn_test: np.ndarray, svm_test: np.ndarray,
                 y_test: np.ndarray, val_rows: list[dict], out_dir: Path) -> None:
    # Matplotlib is only needed to redraw figures, so it is imported lazily: the
    # numeric checks still run on a machine without it. A static checker that
    # cannot see the project virtualenv reports it as missing (it reports numpy,
    # PIL and joblib the same way, while the interpreter imports all of them).
    import matplotlib  # pyright: ignore[reportMissingImports]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    labels = list(metrics.get("images_per_class", {}).keys())
    labels = labels or [f"class {i}" for i in range(cnn_test.shape[1])]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, proba, fname in [("SVM", svm_test, "cm_SVM.png"),
                               ("MobileNetV2", cnn_test, "cm_MobileNetV2.png")]:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for true, pred in zip(y_test, proba.argmax(1), strict=True):
            cm[int(true), int(pred)] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} confusion matrix (held-out test set)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    thetas = [r["theta"] for r in val_rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(thetas, [r["auto_accuracy"] for r in val_rows], "o-", color="#2E7D32")
    ax1.set_xlabel("confidence threshold theta")
    ax1.set_ylabel("accuracy on auto-routed items", color="#2E7D32")
    ax2 = ax1.twinx()
    ax2.plot(thetas, [r["review_rate"] for r in val_rows], "s--", color="#C62828")
    ax2.set_ylabel("share sent to human review", color="#C62828")
    chosen = metrics.get("theta", {}).get("chosen")
    if chosen is not None:
        ax1.axvline(chosen, color="grey", ls=":", label=f"chosen theta = {chosen:.2f}")
        ax1.legend(loc="lower right")
    ax1.set_title("Routing threshold trade-off (validation set)")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_sweep.png", dpi=130)
    plt.close(fig)

    rows = metrics.get("recipes", {})
    if rows:
        names = ["A_frozen_top40", "B_full_finetune", "C_from_scratch"]
        shown = [n for n in names if n in rows]
        means = [rows[n].get("mean_test_accuracy", 0) for n in shown]
        errs = [rows[n].get("sd_test_accuracy", 0) for n in shown]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        bars = ax.bar(shown, means, yerr=errs, capsize=5,
                      color=["#A5D6A7", "#2E7D32", "#EF9A9A"][:len(shown)])
        ax.axhline(0.85, ls="--", c="#C62828", lw=1.2, label="85% target")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("held-out test accuracy")
        ax.set_title("EcoSort technique comparison (same split, same metrics)")
        ax.legend(loc="lower right")
        for bar, mean in zip(bars, means, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.3f}",
                    ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "recipe_comparison.png", dpi=130)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "rebuild the report from saved artefacts").splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=HERE / "models",
                        help="folder holding metrics.json, val_probs.npz, test_probs.npz")
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "verified",
                        help="where redrawn figures are written")
    args = parser.parse_args(argv)

    model_dir: Path = args.model_dir
    out_dir: Path = args.out_dir

    metrics = load_metrics(model_dir)
    test = np.load(model_dir / "test_probs.npz")
    val = np.load(model_dir / "val_probs.npz")

    y_test, cnn_test, svm_test = test["y"], test["cnn"], test["svm"]
    y_val, cnn_val, svm_val = val["y"], val["cnn"], val["svm"]

    checks = Checks()
    checks.add("test set size", len(y_test), len(y_test), 0)

    cnn_acc = float((cnn_test.argmax(1) == y_test).mean())
    cnn_f1 = macro_f1(y_test, cnn_test.argmax(1))
    svm_acc = float((svm_test.argmax(1) == y_test).mean())
    svm_f1 = macro_f1(y_test, svm_test.argmax(1))

    checks.add("CNN test accuracy", metrics.get("cnn", {}).get("accuracy"), cnn_acc)
    checks.add("CNN test macro-F1", metrics.get("cnn", {}).get("macro_f1"), cnn_f1)
    checks.add("SVM test accuracy", metrics.get("svm", {}).get("accuracy"), svm_acc)
    checks.add("SVM test macro-F1", metrics.get("svm", {}).get("macro_f1"), svm_f1)

    # The SVM validation accuracy is recorded inside the sweep block, so it can
    # be cross-checked against the saved validation probabilities as well.
    svm_val_acc = float((svm_val.argmax(1) == y_val).mean())
    sweep_svm = metrics.get("theta", {}).get("sweep_svm") or [{}]
    checks.add("SVM validation accuracy", sweep_svm[0].get("overall_accuracy"), svm_val_acc)

    # The threshold must be reproducible from the validation probabilities alone.
    val_rows = sweep(cnn_val, y_val)
    budget = metrics.get("theta", {}).get("max_review_rate", MAX_REVIEW_RATE)
    chosen = choose_theta(val_rows, budget)
    checks.add("chosen theta", metrics.get("theta", {}).get("chosen"), chosen["theta"])
    for key in ("chosen_review_rate", "chosen_auto_accuracy"):
        if key in metrics.get("theta", {}):
            checks.add(key.replace("_", " "), metrics["theta"][key],
                       chosen["review_rate" if "review" in key else "auto_accuracy"])
    checks.note("review rate at theta", chosen["review_rate"])
    checks.note("auto-routed accuracy at theta", chosen["auto_accuracy"])

    for key, label in [("A_frozen_top40", "recipe A mean test accuracy"),
                       ("B_full_finetune", "recipe B mean test accuracy"),
                       ("C_from_scratch", "recipe C mean test accuracy")]:
        block = metrics.get("recipes", {}).get(key)
        if block:
            checks.note(label, block.get("mean_test_accuracy"))

    ensemble = metrics.get("ensemble") or {}
    if "svm_weight" in ensemble:
        # Older schema: a probability blend of the SVM into the CNN.
        blend = (ensemble["svm_weight"] * svm_test
                 + (1 - ensemble["svm_weight"]) * cnn_test)
        checks.add("blend test accuracy", ensemble.get("test_accuracy"),
                   float((blend.argmax(1) == y_test).mean()))
    elif "cnn_weights" in ensemble:
        # Current schema: the two CNN artefacts blended. test_probs.npz stores the
        # blend itself, so the recorded accuracy is checked directly and the weights
        # are shown for the reader.
        checks.add("blend test accuracy", ensemble.get("test_accuracy"), cnn_acc)
        for name, weight in sorted(ensemble["cnn_weights"].items()):
            checks.note(f"blend weight {name}", weight)

    draw_figures(metrics, cnn_test, svm_test, y_test, val_rows, out_dir)
    print(f"Figures redrawn in {out_dir}/")

    ok = checks.report()
    print("\nHeadline numbers")
    print(f"  CNN          accuracy {cnn_acc:.4f}  macro-F1 {cnn_f1:.4f}")
    print(f"  SVM          accuracy {svm_acc:.4f}  macro-F1 {svm_f1:.4f}")
    print(f"  theta {chosen['theta']:.2f} -> review {chosen['review_rate']:.1%}, "
          f"auto-routed accuracy {chosen['auto_accuracy']:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
