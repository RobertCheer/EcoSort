"""Generate the fast verification notebook.

This notebook does no training. It attaches the published artefacts dataset and
runs ``reproduce_report.py`` against it, so every number in the report can be
re-checked in well under a minute — GPU-free, internet-free.

Regenerate with:  python3 build_verify_notebook.py
Push with:        kaggle kernels push -p verify
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "verify"
OUT.mkdir(exist_ok=True)

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


md("""# EcoSort — verify the report's numbers

**No training.** This notebook checks that every headline number in the report
can be reproduced from the artefacts that were published with the training run.

It attaches the `ecosort-artifacts` dataset (metrics, held-out probabilities and
the split) and runs `reproduce_report.py` against it. That script recomputes
accuracy, macro-F1, the chosen confidence threshold and the routing trade-off,
redraws the figures, and exits non-zero if anything fails to match.

Expected run time is about **ten seconds** of compute, plus kernel start-up.
The full training study takes roughly 38 minutes on a GPU by comparison.

Why this exists: the unit brief says marks are lost if the reported results
cannot be regenerated. This is the cheap path — a reader can confirm the report
without training anything, and the expensive path (retraining from scratch) is
still available in `train_ecosort_224.ipynb`.
""")

code('''import subprocess, sys
from pathlib import Path

# Kaggle nests attached datasets differently depending on how they were added,
# so the artefacts are located rather than assumed.
INPUT = Path("/kaggle/input")
found = sorted(INPUT.rglob("reproduce_report.py"))
if not found:
    tree = [str(q.relative_to(INPUT)) for q in INPUT.rglob("*")][:40]
    raise SystemExit(f"reproduce_report.py not found under {INPUT}. Visible: {tree}")

script = found[0]
SRC = script.parent
OUT = Path("/kaggle/working/figures")
print("Artefacts:", SRC)

result = subprocess.run(
    [sys.executable, str(script), "--model-dir", str(SRC), "--out-dir", str(OUT)],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    raise SystemExit(f"verification failed (exit {result.returncode})")

print("Verification passed: every recorded number reproduces.")
''')

code('''# The figures the report uses, rebuilt from the saved probabilities
from pathlib import Path
import shutil

OUT = Path("/kaggle/working/figures")
for path in sorted(OUT.glob("*.png")):
    print(f"{path.name:28s} {path.stat().st_size/1024:6.1f} KB")

# keep them with the kernel output so they can be downloaded
dest = Path("/kaggle/working")
for path in OUT.glob("*.png"):
    shutil.copy(path, dest / path.name)
print("\\nCopied to /kaggle/working for download.")
''')

md("""## What the numbers mean

* **Accuracy / macro-F1** — held-out test set, 380 images the model never saw.
* **theta** — the confidence threshold, re-derived here from the validation
  probabilities under a 10% reviewer-capacity budget. It is recomputed, not
  read back, so a change to either the model or the rule shows up immediately.
* **auto-routed accuracy** — accuracy on the items the station bins itself,
  excluding the ones it sends to a person.

To retrain rather than verify, run `train_ecosort_224.ipynb` (about 38 minutes
on a GPU; set `ECOSORT_TRAIN_PROFILE=quick` for a CPU smoke test that writes to
a separate folder).
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(OUT / "verify_artifacts.ipynb").write_text(json.dumps(notebook, indent=1))

metadata = {
    "id": "nivalagappan/ecosort-verify-artefacts",
    "title": "EcoSort verify artefacts",
    "code_file": "verify_artifacts.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": ["nivalagappan/ecosort-artifacts"],
    "competition_sources": [],
    "kernel_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

print(f"wrote {OUT/'verify_artifacts.ipynb'} with {len(CELLS)} cells")
print(f"wrote {OUT/'kernel-metadata.json'}")
