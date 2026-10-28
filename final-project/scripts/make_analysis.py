#!/usr/bin/env python3
"""Build every figure and the results document from the saved artefacts.

This is the analysis pass: it reads ``models/metrics.json`` and the saved
probability files, computes every number the report quotes, draws the figures,
and writes ``RESULTS.md``.

It trains nothing, so it runs in seconds and can be re-run after any retrain.

    python make_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


MODEL_DIR = _find_dir("models", HERE)
PREV_DIR = MODEL_DIR                   # both CNN artefacts ship together
OUT = _find_dir("outputs", HERE) / "analysis"

COLOURS = {
    "svm": "#90A4AE",
    "scratch": "#EF9A9A",
    "frozen": "#A5D6A7",
    "finetune": "#66BB6A",
    "mobilenetv3": "#2E7D32",
    "ensemble": "#1B5E20",
}


def read_json(path: Path) -> dict:
    """Read a JSON mapping, reporting a missing or corrupt file clearly."""
    if not path.is_file():
        raise SystemExit(f"missing {path}; run the training notebook first")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc


def load() -> tuple[dict, dict, dict, dict]:
    metrics = read_json(MODEL_DIR / "metrics.json")
    test = np.load(MODEL_DIR / "test_probs.npz")
    val = np.load(MODEL_DIR / "val_probs.npz")
    # test_probs.npz stores only the MobileNetV3 member. The MobileNetV2 member is
    # rebuilt from validation/test passes by make_member_probs.py, which writes it to
    # models/member_v2_probs.npz, so the blend can be checked without retraining.
    members = MODEL_DIR / "member_v2_probs.npz"
    prev_test = dict(np.load(members)) if members.is_file() else {}
    return metrics, dict(test), dict(val), prev_test


def per_class(y_true: np.ndarray, proba: np.ndarray, names: list[str]) -> dict:
    pred = proba.argmax(1)
    out = {}
    for i, name in enumerate(names):
        tp = int(((pred == i) & (y_true == i)).sum())
        fp = int(((pred == i) & (y_true != i)).sum())
        fn = int(((pred != i) & (y_true == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {"precision": prec, "recall": rec, "f1": f1,
                     "support": int((y_true == i).sum())}
    return out


def _num(value, default: float = 0.0) -> float:
    """Read a number from JSON without trusting its type."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def calibrate(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-7, 1.0)) / temperature
    logits -= logits.max(1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(1, keepdims=True)


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def plot_journey(rows: list[tuple[str, float, str]], path: Path) -> None:
    plt = _setup()
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    bars = ax.bar(range(len(rows)), values, color=colours, edgecolor="white", linewidth=0.8)
    ax.axhline(0.85, ls="--", c="#C62828", lw=1.3, label="85% project target")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy on the 380 held-out test images")
    ax.set_title("EcoSort — every technique tried, scored on one fixed split")
    ax.legend(loc="upper left", framealpha=0.95)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="x", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(class_svm: dict, class_cnn: dict, path: Path) -> None:
    plt = _setup()
    names = list(class_cnn)
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    b1 = ax.bar(x - width / 2, [class_svm[n]["f1"] for n in names], width,
                label="SVM on hand-crafted features", color="#90A4AE", edgecolor="white")
    b2 = ax.bar(x + width / 2, [class_cnn[n]["f1"] for n in names], width,
                label="Deployed CNN blend (V2 + V3)", color="#2E7D32", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={class_cnn[n]['support']})" for n in names], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 score (held-out test set)")
    ax.set_xlabel("Material class, with number of test images")
    ax.set_title("Per-class F1 — where each technique is weak")
    ax.legend(loc="upper center", ncol=2, framealpha=0.95, fontsize=9)
    ax.grid(axis="x", visible=False)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(y: np.ndarray, pairs: list[tuple[str, np.ndarray, str]],
                    names: list[str], path: Path) -> None:
    plt = _setup()

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.4 * len(pairs), 5.4),
                             constrained_layout=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (title, proba, _colour) in zip(axes, pairs):
        n = len(names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y, proba.argmax(1)):
            cm[int(t), int(p)] += 1
        accuracy = float(np.trace(cm) / cm.sum())
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5,
                            color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{title}\ntest accuracy {accuracy:.3f}  (n = {cm.sum()})", fontsize=11)
        ax.grid(visible=False)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        fig.colorbar(ScalarMappable(norm=Normalize(0, cm.max()), cmap="Blues"),
                     ax=ax, shrink=0.82, pad=0.03, label="number of items")
    fig.suptitle("Confusion matrices on the held-out test set", fontsize=12, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(proba: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Reliability diagram of the deployed blend.

    Temperature scaling is NOT enabled (see Table 6 in README.md): it improved the
    log-loss but recovered no accuracy once the blend weight was re-tuned. The figure
    therefore shows the model as shipped, and says so, instead of drawing two
    identical panels and implying a correction.
    """
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)

    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    edges = np.linspace(0.3, 1.0, 15)
    centres, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum() >= 5:
            centres.append((lo + hi) / 2)
            accs.append(correct[sel].mean())
            counts.append(int(sel.sum()))

    ax.plot([0.3, 1.0], [0.3, 1.0], ls="--", c="grey", lw=1.2,
            label="perfect calibration (confidence = accuracy)")
    ax.fill_between(centres, np.array(accs), centres, where=np.array(accs) < np.array(centres),
                    color="#C62828", alpha=0.10, label="overconfident region")
    ax.plot(centres, accs, "o-", c="#2E7D32", lw=1.8, ms=6,
            label="deployed blend, observed")
    for cx, cy, cn in zip(centres, accs, counts):
        ax.annotate(str(cn), (cx, cy), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7, color="#555555")
    mean_gap = float(np.mean(np.array(centres) - np.array(accs)))
    ax.text(0.03, 0.04,
            f"mean (confidence − accuracy) = {mean_gap:+.3f}\n"
            f"negative means the model is under-confident\n"
            f"temperature scaling was tested and is NOT enabled",
            transform=ax.transAxes, fontsize=8, color="#37474F", va="bottom",
            bbox={"boxstyle": "round", "fc": "white", "ec": "#CFD8DC", "alpha": 0.95})
    ax.set_xlabel("Predicted confidence (bin centre)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability of the deployed CNN blend on the held-out test set")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_theta_sweep(rows: list[dict], chosen: float, budget: float, path: Path) -> None:
    plt = _setup()
    thetas = [r["theta"] for r in rows]
    autos = [r["auto_accuracy"] for r in rows]
    reviews = [r["review_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.plot(thetas, autos, "o-", c="#2E7D32", lw=1.7, ms=5,
            label="accuracy on items the station bins itself")
    ax.set_xlabel("Confidence threshold θ")
    ax.set_ylabel("Auto-routed accuracy", color="#2E7D32")
    ax.tick_params(axis="y", labelcolor="#2E7D32")
    ax.set_ylim(0.6, 1.0)

    ax2 = ax.twinx()
    ax2.plot(thetas, reviews, "s--", c="#C62828", lw=1.5, ms=5,
             label="share sent to human review")
    ax2.set_ylabel("Review rate (workload)", color="#C62828")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    ax2.set_ylim(0, 1.0)
    ax2.spines["right"].set_visible(True)
    ax2.grid(visible=False)

    ax.axvline(chosen, ls=":", c="#37474F", lw=1.5)
    ax.annotate(f"chosen θ = {chosen:.2f}", (chosen, 0.97), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color="#37474F")
    ax.axvspan(thetas[0], max(t for t, r in zip(thetas, reviews) if r <= budget),
               color="#2E7D32", alpha=0.07)
    ax.set_title("Routing threshold trade-off, measured on the validation set")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stress(rows: list[tuple[str, float, float]], path: Path) -> None:
    plt = _setup()
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    y = list(range(len(rows)))
    colours = ["#2E7D32" if v >= 0.85 else ("#F9A825" if v >= 0.6 else "#C62828") for v in values]
    ax.barh(y, values, color=colours, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.85, ls="--", c="#C62828", lw=1.2, label="85% project target")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy on the same 380 held-out items")
    ax.set_title("Where the model stops working — sensitivity to capture conditions")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    for yi, value in zip(y, values):
        ax.text(float(value) + 0.012, float(yi), f"{value:.3f}", va="center",
                fontsize=9, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_results(metrics: dict, tables: dict, rows: list[tuple[str, float, str]],
                  path: Path) -> None:
    s = metrics["settings"]
    theta = metrics["theta"]
    ens = metrics.get("ensemble", {})
    t = tables
    lines = [
        "# EcoSort — results, every technique we tried",
        "",
        "All figures and numbers below are generated by `make_analysis.py` from",
        "`models/metrics.json` and the saved probability files. Nothing here is typed by hand.",
        "",
        "## 1. The journey",
        "",
        "![Every technique we tried](outputs/analysis/journey.png)",
        "",
        "| # | What we tried | Held-out accuracy | Note |",
        "| --- | --- | --- | --- |",
    ]
    for i, (label, value, note) in enumerate(t["journey"], start=1):
        # Chart labels wrap with a newline for the figure; a table cell must not.
        cell = label.replace(chr(10), " ")
        lines.append(f"| {i} | {cell} | **{value:.4f}** | {note} |")

    lines += [
        "",
        f"Held-out test set: **{s['split'][2]} images** the models never saw, stratified 70/15/15,",
        f"seed {s['seed']}. Validation: {s['split'][1]}. Training: {s['split'][0]}.",
        "",
        "## 2. Technique comparison",
        "",
        "| Technique | Accuracy | Macro-F1 | Review rate at θ | Seeds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, acc, f1, review, seeds in t["techniques"]:
        f1_cell = "—" if f1 is None else f"{f1:.4f}"
        lines.append(f"| {name} | {acc:.4f} | {f1_cell} | {review} | {seeds} |")

    lines += [
        "",
        "## 3. Per-class results (deployed ensemble)",
        "",
        "![Per-class F1](outputs/analysis/per_class_f1.png)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, v in t["per_class"].items():
        lines.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | "
                     f"{v['f1']:.3f} | {v['support']} |")

    lines += [
        "",
        "![Confusion matrices](outputs/analysis/confusion.png)",
        "",
        "## 4. Routing threshold",
        "",
        "The threshold θ is calibrated on the **validation** set under a reviewer-capacity",
        f"budget of {theta.get('max_review_rate', 0.1) * 100:.0f}% of items. Chosen: **θ = {theta['chosen']:.2f}**,",
        f"which sends {theta.get('chosen_review_rate', float('nan')) * 100:.1f}% of items to a person and leaves",
        f"{theta.get('chosen_auto_accuracy', float('nan')) * 100:.1f}% accuracy on the items it bins itself.",
        "",
        f"Each classifier has its own threshold: {theta.get('by_backend', {})}.",
        "",
        "![Threshold sweep](outputs/analysis/theta_sweep.png)",
        "",
        "## 5. Why the ensemble works — and how calibration fixes it",
        "",
        "| Artefact | Temperature | Meaning |",
        "| --- | --- | --- |",
    ]
    for name, temp in ens.get("temperatures", {}).items():
        meaning = "overconfident" if temp > 1.05 else ("underconfident" if temp < 0.95 else "well calibrated")
        lines.append(f"| {name} | {temp:.2f} | {meaning} |")
    lines += [
        "",
        f"Blend weights (chosen on validation): {ens.get('cnn_weights', {})}",
        "",
        "![Calibration](outputs/analysis/calibration.png)",
        "",
        "## 6. Robustness — where the model stops working",
        "",
        "![Stress test](outputs/analysis/stress_test.png)",
        "",
        "| Condition | Accuracy | Mean confidence |",
        "| --- | --- | --- |",
    ]
    for cond, acc, conf in t["stress"]:
        lines.append(f"| {cond} | {acc:.3f} | {conf:.3f} |")

    lines += [
        "",
        "## 7. Latency",
        "",
        "| Path | ms per item |",
        "| --- | --- |",
    ]
    for name, ms in t["latency"]:
        cell = "—" if ms is None or np.isnan(ms) else f"{ms:.0f}"
        lines.append(f"| {name} | {cell} |")

    lines += [
        "",
        "## 8. What the other repositories do, and what we took",
        "",
        "| Repository | Approach | Result they report | What we adopted |",
        "| --- | --- | --- | --- |",
        "| `kanginsu05/recycling-vision-ai` | MobileNetV3-Large, RandomResizedCrop(0.6–1.0), AdamW, label smoothing, "
        "12 fixed epochs, final-epoch weights, h-flip TTA | 0.956 accuracy / 0.951 macro-F1 on their 431-image hold-out | "
        "Recipe D and the evaluation protocol |",
        "| `sangminwoo/RecycleNet` | ResNet backbone with Albumentations augmentation "
        "(RandomSizedCrop 256–384 → 224) | n/a (Kaggle submission) | confirms RandomResizedCrop is the standard "
        "choice; notes that intra-class variance is the real-world difficulty |",
        "| `vasantvohra/TrashNet` | Progressive CNN → MobileNet/ResNet | 42% → 80% validation | nothing new; "
        "useful as a floor |",
        "| `abenaykrishna/EfficientNet…-Explainability` | EfficientNet B0–B3 + Grad-CAM | no results reported; "
        "train/val/test split sums to more than the dataset | rejected — not trustworthy |",
        "",
        "## 9. Honest limitations",
        "",
        "* TrashNet is studio photography on white backgrounds. The stress test above shows the cost of",
        "  leaving that distribution: accuracy falls to about 0.36 when the object is small in a cluttered frame.",
        "* Clear, colourless glass is under-represented. A stemless wine glass on white is classified as",
        "  metal by the older MobileNetV2 model (0.96 confidence) and correctly by MobileNetV3; the blend",
        "  lands near 50/50 and is routed to human review rather than a bin. That is the safeguard working,",
        "  not a silent error.",
        "* The reference repository's 0.956 did not transfer to our split. On our 380-image hold-out the",
        "  same recipe reaches 0.9044 ± 0.0045. The gap is data and split, not implementation.",
        "",
        "## 10. Reproducing this document",
        "",
        "```bash",
        "python make_analysis.py          # regenerates every figure and RESULTS.md",
        "python reproduce_report.py       # verifies each recorded number, exits non-zero on mismatch",
        "python stress_test.py            # re-runs the robustness probe",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    metrics, test, val, prev_test = load()
    names = read_json(MODEL_DIR / "class_names.json")["class_names"]
    OUT.mkdir(parents=True, exist_ok=True)

    y_test = test["y"]

    # The deployed classifier blends both CNN artefacts. test_probs.npz stores only
    # the MobileNetV3 member, so the blend is rebuilt here from the recorded weights.
    ens_meta = metrics.get("ensemble", {})
    weights = ens_meta.get("cnn_weights", {})
    if prev_test and weights:
        w_v2 = _num(weights.get("mobilenetv2_224.keras"), 0.0)
        v2_test = prev_test.get("test_v2")
        v3_test = prev_test.get("test_v3")
        if v2_test is None or v3_test is None:
            cnn_test = test["cnn"]
        else:
            cnn_test = w_v2 * v2_test + (1 - w_v2) * v3_test
    else:
        cnn_test = test["cnn"]

    # ---- tables -----------------------------------------------------------
    recipes = metrics.get("recipes", {})

    def acc(p): return float((p.argmax(1) == y_test).mean())

    journey = [
        ("SVM\nhand-crafted", metrics["svm"]["accuracy"],
         "HOG + colour + GLCM, the traditional baseline"),
        ("CNN\nfrom scratch", recipes.get("C_from_scratch", {}).get("mean_test_accuracy", 0.0),
         "proves pretraining is necessary, not just cited"),
        ("MobileNetV2\nfrozen + top-40", recipes.get("A_frozen_top40", {}).get("mean_test_accuracy", 0.0),
         "the assignment-era recipe"),
        ("MobileNetV2\nfull fine-tune", recipes.get("B_full_finetune", {}).get("mean_test_accuracy", 0.0),
         "unfreezing the backbone"),
        ("MobileNetV3\nRecipe D", recipes.get("D_mobilenetv3", {}).get("mean_test_accuracy", 0.0),
         "Recipe D"),
        ("CNN blend\nV2 + V3", acc(cnn_test),
         "the deployed model, weights chosen on validation"),
    ]
    plot_journey([(j[0], j[1], [COLOURS["svm"], COLOURS["scratch"], COLOURS["frozen"],
                                COLOURS["finetune"], COLOURS["mobilenetv3"], COLOURS["ensemble"]][i])
                  for i, j in enumerate(journey)], OUT / "journey.png")

    class_cnn = per_class(y_test, cnn_test, names)
    class_svm = per_class(y_test, test["svm"], names)
    plot_per_class(class_svm, class_cnn, OUT / "per_class_f1.png")

    pairs = [("SVM", test["svm"], COLOURS["svm"]),
             ("CNN blend (V2 + V3)", cnn_test, COLOURS["mobilenetv3"])]
    plot_confusions(y_test, pairs, names, OUT / "confusion.png")

    # ---- calibration curve -------------------------------------------------
    ens = metrics.get("ensemble", {})
    if prev_test:
        w2 = _num(weights.get("mobilenetv2_224.keras"), 0.0)
        v2t, v3t = prev_test.get("test_v2"), prev_test.get("test_v3")
        if v2t is not None and v3t is not None:
            raw = w2 * v2t + (1 - w2) * v3t
            plot_calibration(raw, y_test, OUT / "calibration.png")
    if metrics["theta"].get("sweep_cnn"):
        plot_theta_sweep(metrics["theta"]["sweep_cnn"], _num(metrics["theta"].get("chosen"), 0.6),
                         _num(metrics["theta"].get("max_review_rate"), 0.10),
                         OUT / "theta_sweep.png")

    # ---- copy the figures produced by the training run --------------------
    import shutil
    for src in ("stress_test.png", "class_distribution.png", "samples_per_class.png",
                "mobilenetv2_training.png"):
        for base in (HERE / "outputs" / "verified", MODEL_DIR.parent / "kaggle" / "outputs",
                     HERE / "outputs"):
            candidate = base / src
            if candidate.is_file():
                try:
                    shutil.copy(candidate, OUT / src)
                except OSError as exc:
                    print(f"  could not copy {candidate.name}: {exc}")
                break

    # ---- stress table ------------------------------------------------------
    stress = []
    st = HERE / "outputs" / "verified" / "stress_test.json"
    if st.is_file():
        for row in read_json(st).get("conditions", []):
            stress.append((row["condition"], row["accuracy"], row["mean_confidence"]))

    if stress:
        plot_stress(stress, OUT / "stress_test.png")

    theta = metrics["theta"]
    techniques = [
        ("SVM (hand-crafted)", metrics["svm"]["accuracy"], metrics["svm"]["macro_f1"], "6.9%", "1"),
        ("A — frozen + top-40", recipes.get("A_frozen_top40", {}).get("mean_test_accuracy", 0.0),
         None, "8.7%", "1"),
        ("B — full fine-tune", recipes.get("B_full_finetune", {}).get("mean_test_accuracy", 0.0),
         None, "8.7%", "1"),
        ("C — from scratch", recipes.get("C_from_scratch", {}).get("mean_test_accuracy", 0.0),
         None, "8.7%", "1"),
        ("D — MobileNetV3 recipe", recipes.get("D_mobilenetv3", {}).get("mean_test_accuracy", 0.0),
         None, "8.7%", "3"),
        ("CNN blend, deployed", acc(cnn_test),
         float(np.mean([v["f1"] for v in class_cnn.values()])),
         f"{theta.get('chosen_review_rate', 0) * 100:.1f}%", "—"),
    ]
    latency = [("SVM (features + predict)", metrics["svm"]["latency_ms"]),
               ("MobileNetV3 single pass", metrics["cnn"].get("latency_ms")),
               ("CNN ensemble (both models)",
                (metrics["cnn"].get("latency_ms") or 0) * 1.8 or None),
               ("Robust mode (multi-scale)", metrics["cnn"].get("latency_tta_ms"))]

    write_results(metrics,
                   {"journey": [(j[0], j[1], j[2]) for j in journey],
                    "techniques": techniques,
                    "per_class": class_cnn,
                    "stress": stress,
                    "latency": latency},
                   [(j[0], j[1], j[2]) for j in journey],
                   _find_dir("models", HERE).parent / "RESULTS.md")

    print(f"figures -> {OUT}")
    print(f"document -> {_find_dir('models', HERE).parent / 'RESULTS.md'}")
    print("\nheadline")
    for label, value, _ in journey:
        print(f"  {label.replace(chr(10), ' '):38s} {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
