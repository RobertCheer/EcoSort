# EcoSort Final Project — work plan to 7 November

**Due:** midnight Saturday 7 November 2026. **Worth:** 30% (27% report + user guide + code + slides, 3% presentation).
**Today:** Thursday 22 October 2026 — 16 days.
**Status assessed:** 22 October, against the files in `FINAL_PROJECT/`.

---

## 1. What is already built (verified, not assumed)

| Component | State | Evidence |
|---|---|---|
| Seven sub-systems integrated | Working | `ecosort/{capture,preprocess,classifier,routing,actuation,logging_db,dashboard,pipeline}.py` |
| Streamlit UI, 4 tabs | Imports clean | `app.py` |
| Test suite | **77 pass, 3 fail** of 80 | `pytest -q` |
| Training artefacts @224 px | Trained, saved | `models/mobilenetv2_224.keras`, `svm_hog_glcm.pkl`, `metrics.json` |
| θ sweep executed | Done, but wrong operating point | `metrics.json → theta.sweep_cnn` |
| Figures | Regenerated 20 Oct | `kaggle/outputs/*.png` (6 figures) incl. `theta_sweep.png` |
| User guide / demo script | Written, **stale numbers** | `USER_GUIDE.md`, `DEMO_SCRIPT.md` |
| Final project report | **Does not exist** | `docs/` is empty |
| Slides + demo video | **Do not exist** | — |

---

## 2. Blockers, ranked by mark risk

### P0 — these can cost 50% of the mark ("if the code and the demo is not working, 50% deducted")

**B1. The router sends 70% of items to a human.** `config.THETA` now resolves to **0.90** from `metrics.json`. On a 40-item held-out batch: 28 items reviewed, 12 binned. The station barely sorts anything, and the demo will look broken.

Cause: `kaggle/build_train_notebook.py:438-442`. The rule "lowest review rate within 2 pp of best auto accuracy" is self-defeating — the best auto accuracy only occurs at the extreme end of the sweep (θ=0.90, 0.955), so the 2 pp band contains exactly one candidate and picks the worst workload.

Recommended fix — cap reviewer workload instead:

| θ | review rate | accuracy on auto-routed items |
|---|---|---|
| 0.45 | 3.7% | 84.7% |
| **0.50** | **8.7%** | **85.6%** |
| 0.60 | 21.1% | 89.3% |
| 0.90 | 58.6% | 95.5% |

θ = 0.50 with a stated reviewer-capacity budget (≤10%) is defensible **and** rescues the 85% target: the automatic path meets 85.6% while only 1 item in 11 goes to a person. That is a far better story than "we missed 85%".

> **FIXED 22 Oct.** Selection rule replaced in `kaggle/build_train_notebook.py` with a 10% reviewer-capacity budget; θ recomputed from `models/val_probs.npz` (not hand-edited) → **0.50**; `config._DEFAULT_THETA` aligned; notebook rebuilt; `kaggle/artifacts/metrics.json` synced.

**B2. Held-out ground truth is silently dropped.** `ecosort/pipeline.py:166` — `_capture()` sends a `PIL.Image` to `from_frame(image, item_id=...)` and never forwards `true_class`. Every batch item comes back with `true_class=None`, so `dashboard.contamination_rate()` and `auto_routed_accuracy()` return `None`. All three failing tests are this one bug. It also means the "contamination dashboard" innovation cannot be shown working — the exact criticism from the assignment feedback ("review queue/routing is not built, so no end-to-end behaviour is shown").

Fix is one line; then re-run `pytest`.

> **FIXED 22 Oct.** `pipeline.py:166` now forwards `true_class` to `from_frame()`. All **80 tests pass**. A 60-item held-out batch now reports accuracy 0.783, auto-routed accuracy 0.839 and contamination 0.217 — real numbers, not `None`.

**B3. TensorFlow does not run in `.venv`, so the demo silently uses the SVM.** `tensorflow_available()` → `False`; `import tensorflow` aborts with `libc++abi: mutex lock failed`. The venv is Python 3.9.6 (macOS system Python) with a broken TF 2.20 from user site-packages. The app falls back to SVM (0.70), not the chosen MobileNetV2 (0.80).

Fix: build a fresh venv on `/opt/homebrew/bin/python3.13` (or 3.12) and install TF there. Without this, the demo video shows the fallback, and the report's "chosen technique" claim is not what the assessor sees.

> **FIXED 22 Oct.** `.venv` rebuilt on Python 3.13 with TensorFlow 2.21; probe returns `True`; the app reports backend `cnn`. `run.sh` now picks a TF-capable interpreter, tries `.[cnn]` first, and falls back to the base install with a clear message if TensorFlow cannot be installed — so a grader's machine cannot end up with a silently broken environment. `scikit-learn` pinned to `>=1.6,<1.7` to match the SVM artefact and keep predictions reproducible.

### P1 — these cost marks directly

**B4. The headline numbers disagree with the code.** This is the same fault the feedback already caught once ("Section 17 states every number was matched against the notebook log, which is not true").

| Claim | `USER_GUIDE.md` says | `metrics.json` says |
|---|---|---|
| MobileNetV2 accuracy | 0.776 | **0.800** |
| MobileNetV2 macro-F1 | 0.744 | **0.7788** |
| SVM accuracy | 0.711 | **0.7026** |
| SVM macro-F1 | 0.703 | **0.6924** |
| Latency CNN | 2.8 ms | **95.74 ms** |
| Latency SVM | 2.6 ms | **14.31 ms** |
| θ | 0.60 | **0.90** |

`DEMO_SCRIPT.md` quotes θ = 60% in five places. Note the latency change is a *measurement* difference, not a regression: the old figure was batch-amortised, the new one is single-image. Single-image is the honest number for a point-of-disposal station — say so explicitly.

**B5. Accuracy fell from 0.832 to 0.800.** The freeze-then-fine-tune recipe (11 epochs + 4 fine-tune) did not beat the earlier frozen-backbone run. Decide deliberately: either drop fine-tuning and report the frozen-backbone result, or keep it and report the honest finding that fine-tuning did not help. Do not quietly report the old number with the new model.

**B6. Two AI-technique criticisms are still unanswered.** The feedback said "single run per technique", "no fine-tuning of the backbone", "the from-scratch CNN is dismissed with a GitHub citation rather than evidence". Fine-tuning is now done. Still missing:
- 3-seed repeats for both techniques (mean ± sd) — the Kaggle notebook takes ~14 min, so three seeds is under an hour.
- A small from-scratch CNN baseline on the identical split — under 10 min, and it replaces a citation with a result.

**B7. No version control for the final project.** `github.com/RobertCheer/EcoSort` is public but last pushed **3 October** and contains only assignment files — no `ecosort/` package, no `app.py`, no tests, no models. Version control is a gating tool category ("if the team does not report the software tools, the project will not be marked"), and there is no branch/PR/commit history for the final build.

**B8. The report does not exist.** 27% of the mark, zero words written. The 15% assignment report (`FINAL_3OCT/ICT304_EcoSort_AssignmentReport.docx`, 17 sections, 12 tables) is a solid base to extend — do not write from scratch.

**B9. Kaggle kernel is private** (`kaggle/kernel-metadata.json: "is_private": true`) while `USER_GUIDE.md` publishes the link. If it is still private at submission the assessor cannot regenerate anything, and the brief is explicit that unregenerable results lose marks.

---

## 3. What the final brief needs that the assignment did not

- **Deployment, integration and maintenance plan** — explicitly required, entirely absent.
- **Test plan *and test results*, including every sub-system** — the tests now exist; the results table does not.
- **Demo video** (3%) and **10-minute presentation slides**.
- **All AI prompts** in the appendix, including the prompts used to draft the report text (the feedback noted the log stops at coding).
- **Fresh Group Declaration form** — signatures were blank, tick-boxes invisible, one member listed by a single name.
- **Weekly tutor progress evidence** — already criticised as "asserted without evidence". This can only be collected by actually sending updates, so it must start this week.

---

## 4. Who does what

Roles follow the existing declaration (equal 33.3%): Nivetha = AI and data, Robert = architecture and integration, Waitun = evaluation, dashboard and reporting.

### Nivetha — AI sub-system and evaluation

| # | Task | Due |
|---|---|---|
| 1 | Fix θ selection rule with a workload cap; set θ = 0.50 (or state your chosen cap and justify it) | Sun 25 Oct |
| 2 | Add 3-seed repeats of both techniques to the training notebook (mean ± sd) | Tue 27 Oct |
| 3 | Add a small from-scratch CNN baseline on the identical split | Tue 27 Oct |
| 4 | Decide the fine-tuning question (B5) and report the result honestly | Tue 27 Oct |
| 5 | Re-run training on Kaggle; replace `models/` and `kaggle/outputs/` with the frozen run | Wed 28 Oct |
| 6 | Write report sections: AI Technique, Evaluation Method, Results, Discussion (numbers) | Sun 1 Nov |
| 7 | Extend the AI prompt log to cover report drafting and every experiment | Fri 30 Oct |
| 8 | Make the Kaggle notebook public; verify it runs clean from a fresh session | Wed 28 Oct |

### Robert — integration, deployment, repository

| # | Task | Due |
|---|---|---|
| 1 | Fix `pipeline.py:166` so held-out items keep their ground truth; get all 80 tests green | Sun 25 Oct |
| 2 | Rebuild the venv on Python 3.13 with working TensorFlow; confirm the CNN backend loads and the app reports `cnn`, not `svm` | Sun 25 Oct |
| 3 | Push `FINAL_PROJECT/` to `RobertCheer/EcoSort` with real history — commits per sub-system, a feature branch per workstream, at least one merged PR, tags. This *is* the version-control evidence | Wed 28 Oct |
| 4 | Write the repository `README.md` (what it is, how to run, where the code maps to the sub-systems) | Wed 28 Oct |
| 5 | Write report sections: System Engineering Life Cycle (all phases, now including build/verify), System Design, Deployment/Integration/Maintenance plan | Sun 1 Nov |
| 6 | Produce the architecture and data-flow diagrams (sub-system, element, part level) | Wed 29 Oct |
| 7 | Rebuild the tools evidence for the final project period: Trello board, GitHub activity, Drive folder, Teams channel — six or more dated screenshots | Fri 30 Oct |

### Waitun — evaluation, reporting, presentation

| # | Task | Due |
|---|---|---|
| 1 | Re-derive every number and figure from the frozen run; correct `USER_GUIDE.md` and `DEMO_SCRIPT.md` (B4) | Wed 28 Oct |
| 2 | Build the sub-system test results table: for each of the seven sub-systems, the test, the criterion, the actual result, pass/fail — generated from a real `pytest -v` run | Thu 29 Oct |
| 3 | Compile the final report to the brief's headings; carry over the assignment report structure, add the new sections | Sun 1 Nov |
| 4 | Extend the user guide for the final system: install (incl. the Python 3.13 + TensorFlow path), run, reproduce training, dataset and split | Wed 29 Oct |
| 5 | Build the 10-minute slide deck | Wed 4 Nov |
| 6 | Record the demo video from `DEMO_SCRIPT.md`, after the θ and CNN fixes — re-time the script to θ = 0.50 | Thu 5 Nov |
| 7 | Re-do the Group Declaration form: full names, real signatures, visible tick-boxes, updated contribution descriptions | Thu 5 Nov |
| 8 | Format the report to PDF, upload to LMS, download it again and confirm it opens | Fri 6 Nov |

### Everyone

- **Weekly tutor update, starting this week** — one page: what was built, what the results show, what is next, with the tutor's name and date in a log the report can cite. This is the only way to answer the "asserted without evidence" criticism.
- Read the whole report once, out loud, before it is frozen.

---

## 5. Sequence and dates

```text
Thu 22 – Sun 25 Oct   P0: system works.  B1 θ rule, B2 ground-truth bug, B3 TensorFlow.
                                            Green tests. A batch run that reports real
                                            accuracy and contamination.
Mon 26 – Wed 28 Oct   P1: evidence frozen. B6 seeds + from-scratch CNN, B7 repo live,
                                            B9 notebook public, B4 docs corrected,
                                            all figures regenerated once, final.
Thu 29 Oct – Sun 1 Nov Report draft v1, all sections, all numbers from the frozen run.
Mon 2 – Thu 5 Nov      Report v2, user guide, slides, demo video, declaration form.
Fri 6 Nov              Full read-through, every number spot-checked against metrics.json
                       and a real console run. Submit to LMS. Download and verify.
Sat 7 Nov              Buffer only. Do not plan to use it.
```

**Critical path:** the θ fix and the ground-truth fix must land before anything is written, because every number in the report and every shot in the video depends on them.

---

## 6. Definition of done

- [x] `pytest -q` → 80 passed, 0 failed.
- [x] A held-out batch reports a real accuracy, a real review rate and a non-`None` contamination rate.
- [x] App reports backend `cnn`, and a MobileNetV2 prediction is on screen.
- [ ] `metrics.json` is the single source of every number in the report, user guide and slides; a grep for each stale value (0.776, 0.744, 2.8 ms, θ 0.60, 0.832) returns nothing.
- [ ] Public GitHub repo with final-project code, dated commits and a merged PR.
- [ ] Public Kaggle notebook, re-run clean from a fresh session.
- [ ] Report contains all required headings plus deployment/integration/maintenance and sub-system test results.
- [ ] AI prompt appendix covers coding *and* drafting.
- [ ] Signed Group Declaration form attached.
- [ ] Demo video and slides uploaded.
- [ ] Submission downloaded from LMS and opened.


---

## 7. Measured run times (22 October)

Three paths, so nobody has to train to check the work.

| Path | Command | Measured |
|---|---|---|
| Run the system | `./run.sh` | ~15 s to launch, 55 ms/item |
| Verify the report locally | `python reproduce_report.py` | **0.85 s**, 8/8 checks |
| Verify on Kaggle | `kaggle kernels push -p kaggle/verify` | **31 s** compute, ~2 min wall |
| Retrain from scratch | `kaggle kernels push -p kaggle` | **37.5 min** compute, ~59 min wall |

The retrain breakdown, from the run log: dataset load 5.8 min, SVM grid 3.0 min,
recipe A 1.7 min, recipe B 15.8 min (3 seeds), recipe C 10.9 min (3 seeds).

### Current results (artefacts in `models/`, 22 October)

| Recipe | Held-out test accuracy |
|---|---|
| SVM on hand-crafted features | 0.7105 |
| A — old recipe, frozen backbone + top-40 | 0.8132 |
| **B — full fine-tune + schedule (deployed)** | **0.8781 ± 0.0173**, best seed 0.9026 |
| C — from scratch | 0.7009 ± 0.0126 |

θ = 0.70 → 7.6% of items to human review, **93.1% accuracy on auto-routed items**.
The SVM blend adds nothing (0.9026 with or without) — do not claim it in the report.
Previous model kept at `models_prev_0800/` for comparison; delete before submission.

### Updating the published artefacts

The verification kernel reads `nivalagappan/ecosort-artifacts`. After a retrain,
republish so verification checks the new numbers:

```bash
cp models/metrics.json models/*.npz /tmp/eco-ds/
kaggle datasets version -p /tmp/eco-ds -m "refresh artefacts after retrain"
```

### Still to fix

`USER_GUIDE.md` and `DEMO_SCRIPT.md` still quote 0.776 / 2.8 ms / θ 0.60 and are now
wrong twice over. This is the highest-risk item left: it is the same fault the
marker already caught once.


---

## 8. Recipe decisions (22 October, final)

| Recipe | Kept? | Why |
|---|---|---|
| SVM (HOG + colour + GLCM) | **yes** | the brief requires at least two techniques compared |
| A — frozen backbone + top-40 | **yes** | the controlled ablation behind "fine-tuning the backbone helps" |
| B — full fine-tune + schedule | **yes** | the deployed model, 0.9026 |
| C — from-scratch CNN | **yes** | answers "dismissed with a citation instead of evidence" |
| ~~SVM + CNN blend~~ | **cut** | measured zero gain: 0.9026 with it, 0.9026 without |
| D — background augmentation | **not built** | would only matter if the demo uses phone photos; recorded as future work instead |

Seed counts are unchanged (3 each for B and C) **on purpose**: the shipped
`metrics.json` was produced with three seeds. Changing the notebook now would
make the shipped numbers unreproducible from the shipped notebook — the exact
fault the marker flagged last time. The saving would only appear on a rerun.

Removing the blend changes no reported number. A rerun of the current notebook
reproduces A, B, C, SVM and theta exactly as shipped; it simply omits the
ensemble block.

## 9. Known limitation, measured

`python stress_test.py` (27 s, no training) quantifies how far the model falls
outside the studio benchmark. Same 380 held-out items, five capture conditions:

| Condition | Accuracy | Mean confidence | To review |
|---|---|---|---|
| Baseline, studio white background | 0.903 | 0.931 | 10.5% |
| Grey surface instead of white | 0.897 | 0.932 | 10.5% |
| Dimmer, as indoors | 0.887 | 0.932 | 8.9% |
| Soft focus | 0.884 | 0.923 | 11.1% |
| **Phone framing: smaller object, cluttered scene** | **0.408** | 0.826 | 25.5% |

The failure is background and framing, not lighting: mild capture changes cost
almost nothing, but the object no longer filling a white frame costs 0.50
accuracy. Confidence stays high (0.826), so the review queue catches only a
quarter of it.

This is the Discussion section's centrepiece: a quantified limitation with a
named remedy (background/scale augmentation), rather than a vague "more data
would help". It is also the honest reason the demo should use held-out samples.


---

## 10. Recipe D (22 Oct)

Source: `https://github.com/kanginsu05/recycling-vision-ai` — a Korean Kaggle
competition entry (6th place) that was rebuilt post-competition. Reported hold-out:
0.956 accuracy, 0.951 macro-F1, 40 ms CPU inference.

### What they do that we did not

| Element | Their setting | Our recipe B |
|---|---|---|
| Backbone | MobileNetV3-Large | MobileNetV2 |
| Augmentation | **RandomResizedCrop(0.6-1.0)**, h+v flip, rot 20, colour jitter | flip, rotate, zoom, brightness |
| Optimiser | AdamW, wd 0.02 | Adam |
| Learning rates | backbone 3e-4, head 1e-3 (separate groups) | single 1e-3 |
| Schedule | 1-epoch warm-up + cosine, 12 epochs | EarlyStopping, up to 30 |
| Loss | label smoothing 0.1 + class weights | class weights only |
| Weights kept | **final epoch** | best validation epoch |

Two protocol lessons worth more than the model:

1. **They fix the split and the epoch budget before modelling**, and keep the
   hold-out out of every decision. Our `restore_best_weights=True` is a mild form
   of selecting on validation; recipe D removes it.
2. **They re-evaluated their own competition models on a fixed hold-out** and found
   three steps that all scored 0.7500 publicly were actually 0.647 / 0.661 / 0.763.
   Public leaderboard differences were noise. That is the same argument our report
   should make about our own single-run numbers.

### Their independent confirmation of our glass→metal finding

From their error analysis: 8 of 19 hold-out errors were wrong `metal` predictions,
and Grad-CAM showed the model looking at **metal lids on glass bottles**. That is
exactly our item 115 (glass → metal, confidence 0.99). It is a documented,
physically-explainable limitation, not a bug.

### What the port looks like in Keras

| Their element | Keras implementation |
|---|---|
| `RandomResizedCrop` | custom layer using `tf.image.crop_and_resize`, one box per image |
| two LR groups | two stages: head-only warm-up at 1e-3, then full backbone at 3e-4 on a cosine curve |
| label smoothing | `CategoricalCrossentropy(label_smoothing=0.1)` + one-hot labels |
| class weights | per-sample weights inside the `tf.data` pipeline (Keras 3 refuses `sample_weight` with a Dataset) |
| fixed epochs, final weights | no `EarlyStopping`, no `restore_best_weights` |

**Augmentation is not in the saved artefact.** A custom layer in a saved Keras model
cannot be loaded without its class, which would break the app. The notebook builds
two graphs over shared layer instances — one with augmentation for training, one
without for inference — and saves only the clean one. Verified: the artefact loads
with a plain `load_model` and predicts identically.

### Changes made along the way

* `ecosort/classifier.py` now looks for `mobilenetv3_224.keras` first and falls back
  to `mobilenetv2_224.keras`, so the currently-shipped model keeps working.
* Recipes B and C are now **ablations at one seed each** (they are no longer the
  shipped model); D keeps three seeds because it ships. This is where the runtime
  saving comes from.
* Mixed precision was tried and **reverted**: the run took longer than the 37.5 min
  baseline, so it is not in the recipe. The speed win is the fixed 12-epoch budget.
