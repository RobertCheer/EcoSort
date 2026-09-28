# EcoSort — Intelligent Recycling Sorting System

ICT304 Assignment and Project: Developing an AI System. EcoSort classifies a photo of a
waste item into one of six recycling categories (cardboard, glass, metal, paper, plastic,
trash) and returns a confidence score with the prediction. This repository holds the
**prototype of the AI sub-system**: two trained classifiers, their evaluation on a
held-out test set, and the figures that support the assignment report.

## What is built and what is not

Built for the Assignment (due 3 October 2026):

- Data pipeline: TrashNet load, EDA, stratified 70/15/15 split, random_state=42
- Technique A: SVM on HOG + colour-histogram + GLCM features, inside a scikit-learn
  `Pipeline` with `GridSearchCV` and `probability=True`
- Technique B: MobileNetV2 transfer learning (TensorFlow/Keras, ImageNet weights,
  augmentation, class weights for the minority class)
- Evaluation: accuracy, per-class precision/recall/F1, confusion matrices, inference
  latency, comparison table
- One-image inference demo returning `(class, confidence)`
- An upload box (Step 7 in the notebook) so a reader can classify their own photo and see
  both predictions with the routing decision, plus a file-path fallback where widgets
  cannot run (a saved notebook cannot show live widgets; use Copy & Edit on Kaggle)

Designed but not built yet (Final Project, due 7 November 2026): image capture,
decision and bin-routing, which routes items whose confidence falls below a
threshold θ into a human-review queue, plus logging and
dashboard, user interface, simulated actuation. A static HTML mockup of the dashboard
exists in the team working folder; it is not connected to live data.

## Results

Run on Kaggle (2× T4 GPU), 28 September 2026. Test set: 380 held-out images.
Latency is averaged over the test set.

| Technique | Accuracy | Macro-F1 | Latency |
|---|---|---|---|
| SVM (HOG + colour + GLCM) | 0.711 | 0.703 | 2.6 ms/image |
| MobileNetV2 (transfer learning) | 0.776 | 0.744 | 2.8 ms/image |

Per-class F1 (test set, 380 images):

| Class | SVM F1 | MobileNetV2 F1 |
|---|---|---|
| cardboard | 0.76 | 0.84 |
| glass | 0.69 | 0.77 |
| metal | 0.67 | 0.78 |
| paper | 0.80 | 0.83 |
| plastic | 0.64 | 0.75 |
| trash | 0.67 | 0.50 |

Accuracy 0.776 sits below the REQ-01 target of 0.85. The settings behind this run are
deliberately small (IMG=128, 8 training epochs) so the notebook also finishes on a
laptop CPU. We report it as measured; the report's Discussion covers the improvement
path (larger input size, more epochs, the full hyperparameter grid, and a fine-tuning
stage). Six runs of this exact notebook put MobileNetV2 between 0.755 (a laptop) and 0.792
(Kaggle GPUs), while the SVM returned 0.711 every time. The CNN moves a little between
runs, which is expected and is discussed in the report.

## Reproduce

**Kaggle (recommended).** Open the notebook, press Save & Run All. The dataset and the
MobileNetV2 weights are attached, so the kernel needs no internet. The compute
takes about 6 minutes on GPU; allow a few more for session startup and packaging.
<https://www.kaggle.com/code/nivalagappan/ecosort-ai-sub-system-prototype>

**Colab.** Upload `EcoSort_AI_Prototype_vscode.ipynb`, then Runtime → Run all. It
downloads TrashNet from Hugging Face (about 3.6 GB) on first run.

**Local.** `pip install tensorflow scikit-learn scikit-image huggingface_hub matplotlib
seaborn pillow`, then run the notebook (or the `.py` export). On a laptop CPU expect
roughly 20–40 minutes.

Environment for the recorded run: Python 3.12, TensorFlow 2.20.0, 2× T4. Seeds are
fixed at 42 throughout.

## Data

TrashNet (Thung & Yang, 2016): 2,527 images in six classes: cardboard 403, glass 501,
metal 410, paper 594, plastic 482, trash 137 (the minority). The notebook reads, in
order: a local `dataset-resized/` folder if present, any attached Kaggle dataset
mounted under `/kaggle/input`, and finally the Hugging Face Hub download.

## Repository

- `EcoSort_AI_Prototype_vscode.ipynb`: the source notebook
- `EcoSort_AI_Prototype_Kaggle_executed.ipynb`: the executed notebook with outputs
- `EcoSort_AI_Prototype_vscode.py`: a plain-script version of the same pipeline
- `outputs/`: figures from the executed run
- `kaggle/kernel-metadata.json`: the kernel configuration used for the Kaggle run
- `weights/`: the MobileNetV2 ImageNet weights, bundled so kernels can run offline

## Team

Equal three-way split (Group Declaration weighting 0.33 each). Nivetha leads AI and data and wrote the classifier notebook. Robert leads
architecture and integration and maintains this repository. Waitun leads evaluation,
dashboard and reporting, and compiles the report. AI tool use is declared in the report per Murdoch's APA generative-AI
guide, with prompts in the appendix.

## Links

- Kaggle notebook: <https://www.kaggle.com/code/nivalagappan/ecosort-ai-sub-system-prototype>
- Kaggle dataset (TrashNet resized + MobileNetV2 weights): <https://www.kaggle.com/datasets/nivalagappan/ecosort-trashnet-resized>
- TrashNet upstream: <https://huggingface.co/datasets/garythung/trashnet>


