# EcoSort User Guide

## Overview
EcoSort is an intelligent recycling sorting system for the point of disposal. It uses a camera to capture an image of a waste item, classifies it into one of six recycling categories using a trained AI model, and routes it to the correct bin or to a human-review queue if the model is not confident enough. Every decision is logged and displayed on a live contamination dashboard.

This guide describes how to install, run, and reproduce the results of the EcoSort system as submitted for the ICT304 final project.

## System Requirements
- Operating System: Linux, macOS, or Windows (tested on macOS Ventura 13.6)
- Python 3.9 or higher (tested with Python 3.9.6)
- Internet connection (to download dataset and model weights on first run)
- At least 2 GB of free disk space
- Optional: A webcam for live capture (otherwise, use image upload or held-out samples)

## Installation
1. Clone the repository (if not already done):
   ```bash
   git clone <repository-url>
   cd FINAL_PROJECT
   ```
2. The project includes a virtual environment bootstrap script. To set up the environment:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   This will create a virtual environment (`venv`), install the package and dependencies, and launch the Streamlit app.

   If you prefer to set up manually:
   ```bash
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

## Running the System
To launch the EcoSort web interface:
```bash
./run.sh
```
or, if you have activated the virtual environment:
```bash
streamlit run app.py
```
The application will open in your default web browser (typically at http://localhost:8501).

### Using the Application
The app provides four tabs:
1. **Sort an item**: Capture a photo via upload, webcam, or held-out sample. View the classification, confidence, routing decision, and simulated actuation.
2. **Live dashboard**: View running statistics from the decision log (per-class counts, contamination rate, low-confidence rate, etc.).
3. **Batch evaluation**: Run a batch of held-out items through the full pipeline to evaluate end-to-end performance.
4. **About**: Information about the system components, models, and environment.

## Reproducing Training Results
\n**Public Notebook Links:**\n- Assignment Prototype (AI sub-system): https://www.kaggle.com/code/nivalagappan/ecosort-ai-sub-system-prototype\n- Final Project Training Notebook: https://www.kaggle.com/code/nivalagappan/ecosort-final-project-training (the notebook used to generate the trained models in `models/`)\n
The trained models (`mobilenetv2_224.keras` and `svm_hog_glcm.pkl`) are included in the `models/` directory. To reproduce the training from scratch, follow these steps:

1. Ensure you have a Kaggle account and have installed the Kaggle CLI (see https://www.kaggle.com/docs/api).
2. Place your Kaggle API token in `~/.kaggle/kaggle.json` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
3. Run the training notebook:
   ```bash
   jupyter notebook kaggle/train_ecosort_224.ipynb
   ```
   Alternatively, execute it headlessly:
   ```bash
   jupyter nbconvert --to notebook --execute kaggle/train_ecosort_224.ipynb --output kaggle/train_ecosort_224_executed.ipynb
   ```
4. The notebook will download the TrashNet dataset, train both techniques, evaluate them on the held-out test set, calibrate the confidence threshold, and save the following artifacts to the `models/` directory:
   - `mobilenetv2_224.keras` – the MobileNetV2 model at 224x224 input size
   - `svm_hog_glcm.pkl` – the SVM pipeline on hand-crafted features (HOG + colour histogram + GLCM)
   - `class_names.json` – ordered list of the six class names
   - `split_indices.json` – the train/validation/test split indices (random_state=42)
   - `metrics.json` – training configuration and evaluation results
   - `theta_sweep.png` – visualization of the confidence threshold trade-off
   - Various output figures (confusion matrices, training curves, etc.)

The notebook also creates a zip file `ecosort_artifacts.zip` containing all artifacts.

## Dataset
The system uses the TrashNet dataset (Thung & Yang, 2016), resized to 512x384 pixels, containing 2,527 images across six classes:
- Cardboard: 403
- Glass: 501
- Metal: 410
- Paper: 594
- Plastic: 482
- Trash: 137

The dataset is automatically downloaded from the Hugging Face Hub (`garythung/trashnet`) if not present locally. A local copy is stored in `work/dataset-resized/` (relative to the project root) for reuse.

## Train/Validation/Test Split
A stratified 70/15/15 split is used, with `random_state=42` to ensure reproducibility. This yields:
- Training: 1,768 images
- Validation: 379 images
- Test: 380 images (held-out for final evaluation)

The split indices are saved in `models/split_indices.json`.

## Model Details
### MobileNetV2 Transfer Learning
- Base: MobileNetV2 pre-trained on ImageNet, input size 224x224
- Backbone frozen during initial training; top 40 layers unfrozen for fine-tuning
- Head: Global average pooling → Dropout (0.2) → Dense (6 units, softmax)
- Optimizer: Adam
- Loss: Categorical cross-entropy with label smoothing 0.1
- Metrics: Accuracy
- Class weights: Computed from the training distribution to correct class imbalance
- Augmentation (training only): Random resized crop (scale 0.6 to 1.0), horizontal flip, vertical flip, rotation ±20 degrees, brightness and contrast jitter
- Training: 2 head-only warm-up epochs at learning rate 1e-3, then 10 epochs with the whole backbone trainable at 3e-4 on a cosine curve. The epoch budget is fixed at 12 and the final epoch is kept.
- Optimiser: AdamW with weight decay 0.02
- Test-time augmentation: mean of the plain and mirrored forward pass

### Deployed CNN: a blend of two artefacts
The classifier ships two CNN files and averages their probability vectors. The blend
weight is 0.2 for `mobilenetv2_224.keras` and 0.8 for `mobilenetv3_224.keras`. The
weight is selected on the validation set. Two models make different mistakes, so the
blend is more accurate than either model alone.

### SVM on Hand-Crafted Features
- Features: Histogram of Oriented Gradients (HOG) + 32-bin RGB colour histogram + GLCM texture properties (contrast, dissimilarity, homogeneity, energy, correlation) at distances 1 and 2 and angles 0°, 45°, 90°, 135°
- Feature vector length: 1,900
- Preprocessing: StandardScaler (fitted on training data only)
- Model: Support Vector Machine with RBF kernel, probability=True (to output confidence scores)
- Hyperparameters: C=10, gamma=scale (selected via 5-fold grid search maximizing macro-F1)

## Evaluation Results (held-out test set)

The held-out test set has 380 images. The model did not see these images during training.

| Metric | SVM (HOG+colour+GLCM) | CNN blend (deployed) |
| --- | --- | --- |
| Accuracy | 0.7105 | **0.9211** |
| Macro-F1 | 0.7014 | **0.9066** |
| Latency, one image | 11 ms | 123 ms per model, about 220 ms for the blend |
| Robust mode, one image | not available | about 250 ms |
| Artefact size | about 200 KB | 23 MB per CNN file |

The full pipeline applies a border trim before the resize. With this trim, the
accuracy on the same 380 images is 0.9263.

### Every technique that we tested on the same split

| Technique | Accuracy |
| --- | --- |
| SVM on hand-crafted features | 0.7105 |
| Small CNN from a random start | 0.6921 |
| MobileNetV2, frozen backbone, top 40 layers trained | 0.8132 |
| MobileNetV2, full fine-tune | 0.8868 |
| MobileNetV3-Large, Recipe D | 0.9131 ± 0.0037 |
| **CNN blend, deployed** | **0.9211** |

The figures and the full discussion are in `README.md` and `RESULTS.md`. Every
number above comes from `models/metrics.json`. Run `python reproduce_report.py`
to check each number again.

## Confidence Threshold (θ)
The threshold controls the review queue. We select it on the validation set. The rule is: find the most accurate automatic path that keeps the review queue at or below 10% of the items.

The chosen value is **θ = 0.60** for the CNN. At this value the station sends 9.8% of items to a person. The items that the station bins itself have 95.3% accuracy.

Each classifier has its own threshold. The **SVM uses θ = 0.35**, because its probability values are lower. If you change the backend, the slider moves to that backend's calibrated value.

You can change the slider for a demonstration. A higher value sends more items to a person. A lower value sends more items to a bin, including more wrong items.

## Dependencies
The system declares the following dependencies in `pyproject.toml`:
- numpy>=1.24
- pillow>=9.1
- scikit-learn>=1.3
- scikit-image>=0.21
- pandas>=2.0
- matplotlib>=3.7
- joblib>=1.3
- streamlit>=1.30
- tensorflow>=2.16 (optional, for the CNN backend; the SVM fallback works without it)

## Troubleshooting
- **TensorFlow import errors**: If you see errors related to TensorFlow not being able to load, the system will automatically fall back to the SVM backend. To use the CNN backend, ensure a compatible TensorFlow version is installed (tested with 2.16.1 on Python 3.9). On macOS with Apple Silicon, you may need to install `tensorflow-macos` instead.
- **Missing model files**: If the `models/` directory does not contain `mobilenetv2_224.keras` and `svm_hog_glcm.pkl`, the system will prompt you to run the training notebook or download the artifacts.
- **Webcam not detected**: Ensure your webcam is enabled and not used by another application. You can still use image upload or held-out samples.
- **Streamlit port already in use**: Kill any existing streamlit processes or specify a different port with `streamlit run app.py --server.port 8502`.

## License
This project is provided for educational purposes as part of the ICT304 final project at Murdoch University.

## References
- Thung, G., & Yang, J. (2016). An image-based system for recycling recognition and suggestion. *Resources, Conservation and Recycling*, 117, 1-10.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, Vol. 1, pp. 886-893.
- Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610-621.
