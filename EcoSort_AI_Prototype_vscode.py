"""
EcoSort — AI Sub-system Prototype  (VSCode / local CPU version)
ICT304 Assignment · AI & Data Lead (Nivetha)

This is a plain Python script that runs on your own computer (no Google Colab, no GPU
required). It trains and compares TWO techniques on TrashNet:
  - Technique A: SVM on HOG + colour + GLCM features  (baseline)
  - Technique B: MobileNetV2 transfer learning         (primary)
Both output (predicted class, confidence) — the hook for the bin-routing sub-system.

------------------------------------------------------------------------------
SETUP (run once in the VSCode terminal:  Terminal -> New Terminal):
    pip install tensorflow scikit-learn scikit-image huggingface_hub matplotlib seaborn pandas pillow

  * Apple Silicon (M-series) Mac? CPU TensorFlow works as-is. For a speed boost you can
    also:  pip install tensorflow-metal   (optional, uses the Mac GPU via Metal).
  * Windows/Mac with an NVIDIA GPU? Install CUDA + cuDNN and TensorFlow will use it
    automatically (much faster). Not required — CPU works.

RUN:
    python EcoSort_AI_Prototype_vscode.py

EXPECTED RUNTIME on CPU: ~20-40 min (one-time). Watch the progress prints.
OUTPUT: figures are saved to the  outputs/  folder (open the PNGs to view them).
------------------------------------------------------------------------------
"""

import os
import time
import zipfile
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend -> figures saved to files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from collections import Counter

os.makedirs("outputs", exist_ok=True)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
sns.set_theme(style="whitegrid")

# ---- CPU-friendly settings (edit if you have a GPU and want the full version) ----
IMG = 128  # MobileNetV2 image size (128 is much faster than 224 on CPU)
CNN_EPOCHS = 8  # fewer epochs than the Colab version (CPU is slower)
CNN_PATIENCE = 3
FINE_TUNE = False  # set True to add a short fine-tune step (adds time)

print("=" * 64)
print("EcoSort AI Sub-system Prototype — local run")
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus if gpus else "none -> running on CPU (slower, but fine)")
print("=" * 64)


# ============================================================================
# STEP 1 — Load TrashNet + EDA
# ============================================================================
print("\n[Step 1] Loading TrashNet from the Hugging Face Hub...")
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "garythung/trashnet"
files = list_repo_files(REPO, repo_type="dataset")
zip_name = next((f for f in files if f.lower().endswith(".zip")), None)
if zip_name is None:
    raise FileNotFoundError(f"No .zip found in {REPO}: {files}")
print("  Using zip:", zip_name)

zip_path = hf_hub_download(repo_id=REPO, filename=zip_name, repo_type="dataset")
print("  Downloaded to:", zip_path)

with zipfile.ZipFile(zip_path) as z:
    z.extractall("trashnet")


def find_data_dir(root):
    """Find the folder holding the 6 class subfolders (skips macOS __MACOSX junk)."""
    need = {"glass", "paper", "cardboard", "plastic", "metal", "trash"}
    for dirpath, dirnames, _ in os.walk(root):
        if "__MACOSX" in dirpath.split(os.sep):
            continue
        real = {d.lower() for d in dirnames if not d.startswith("._")}
        if need <= real:
            return dirpath
    raise FileNotFoundError("Could not find the 6 class folders in the extracted zip.")


DATA_DIR = find_data_dir("trashnet")
class_names = sorted(
    [
        d
        for d in os.listdir(DATA_DIR)
        if not d.startswith("._") and os.path.isdir(os.path.join(DATA_DIR, d))
    ]
)
print("  DATA_DIR:", DATA_DIR)

X, y = [], []
for label, cls in enumerate(class_names):
    for fname in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")) and not fname.startswith(
            "._"
        ):
            X.append(Image.open(os.path.join(DATA_DIR, cls, fname)).convert("RGB"))
            y.append(label)
y = np.array(y)

counts = Counter(y.tolist())
print("  Classes:", class_names)
print("  Counts:", {class_names[i]: counts[i] for i in range(len(class_names))})
print("  Total images:", len(X))

# EDA — class distribution + sample images
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(class_names, [counts[i] for i in range(len(class_names))], color="#2E7D32")
ax.set_title("Class distribution (trash is the minority, 137)")
ax.set_ylabel("Number of images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()

fig, axes = plt.subplots(1, len(class_names), figsize=(15, 3))
for i, cls in enumerate(class_names):
    idx = np.where(y == i)[0][0]
    axes[i].imshow(X[idx])
    axes[i].set_title(cls)
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("outputs/samples_per_class.png")
plt.close()
print("  Saved: outputs/class_distribution.png, outputs/samples_per_class.png")


# ============================================================================
# STEP 2 — Stratified 70 / 15 / 15 split (reproducible)
# ============================================================================
print("\n[Step 2] Splitting 70/15/15 (stratified, random_state=42)...")
from sklearn.model_selection import train_test_split

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)
print(f"  Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")


# ============================================================================
# STEP 3 — Technique A: SVM on hand-crafted features
# ============================================================================
print("\n[Step 3] Technique A — SVM on HOG + colour + GLCM features...")
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.color import rgb2gray


def extract_features(img, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    a = np.array(img)
    gray = (rgb2gray(a) * 63).astype(np.uint8)  # quantise to 64 levels -> fast GLCM
    hog_f = hog(
        gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True
    )
    hist = np.concatenate(
        [np.histogram(a[:, :, c].ravel(), bins=32, range=(0, 256))[0] for c in range(3)]
    )
    hist = hist / (hist.sum() + 1e-6)
    g = graycomatrix(
        gray,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    glcm = np.concatenate(
        [
            graycoprops(g, p).ravel()
            for p in [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
            ]
        ]
    )
    return np.concatenate([hog_f, hist, glcm])


print("  Building features (this takes a few minutes)...")
t0 = time.time()
F_train = np.array([extract_features(im) for im in X_train])
F_val = np.array([extract_features(im) for im in X_val])
F_test = np.array([extract_features(im) for im in X_test])
print(f"  Feature matrix — train: {F_train.shape}, done in {time.time() - t0:.1f}s")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(
    [("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True))]
)
# Small grid for CPU speed; expand to [1,10,100] x ['scale',0.01,0.001] if you have time/GPU.
grid = GridSearchCV(
    pipe,
    {"svc__C": [1, 10], "svc__gamma": ["scale", 0.01]},
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)
print("  Training SVM (grid search)...")
t0 = time.time()
grid.fit(F_train, y_train)
svm = grid.best_estimator_
print(f"  Best params: {grid.best_params_}  (in {time.time() - t0:.1f}s)")

svm_pred = svm.predict(F_test)
svm_conf = svm.predict_proba(F_test).max(axis=1)
print(
    "  SVM done. Example (class, confidence):",
    (class_names[svm_pred[0]], round(float(svm_conf[0]), 3)),
)


# ============================================================================
# STEP 4 — Technique B: MobileNetV2 transfer learning
# ============================================================================
print("\n[Step 4] Technique B — MobileNetV2 transfer learning...")
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

base = MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomZoom(0.1),
    ]
)

inp = layers.Input((IMG, IMG, 3))
x = aug(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inp, out)

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), cw))
print(
    "  Class weights:",
    {class_names[k]: round(float(v), 3) for k, v in class_weight.items()},
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
es = tf.keras.callbacks.EarlyStopping(patience=CNN_PATIENCE, restore_best_weights=True)


def make_ds(images, labels, shuffle=False):
    def gen():
        for im, lab in zip(images, labels):
            im = im.convert("RGB").resize((IMG, IMG))
            yield np.array(im, dtype="float32"), int(lab)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((IMG, IMG, 3), tf.float32),
            tf.TensorSpec((), tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds = make_ds(X_val, y_val)
test_ds = make_ds(X_test, y_test)

print(f"  Training head for up to {CNN_EPOCHS} epochs (patience={CNN_PATIENCE})...")
t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CNN_EPOCHS,
    class_weight=class_weight,
    callbacks=[es],
)
print(f"  Training done in {time.time() - t0:.1f}s")

# training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("MobileNetV2 accuracy")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("MobileNetV2 loss")
ax[1].set_xlabel("epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig("outputs/mobilenetv2_training.png")
plt.close()
print("  Saved: outputs/mobilenetv2_training.png")

if FINE_TUNE:
    print("  Fine-tuning (unfreeze top ~20 layers, low LR)...")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        class_weight=class_weight,
        callbacks=[es],
    )

cnn_prob = model.predict(test_ds, verbose=0)
cnn_pred = cnn_prob.argmax(1)
cnn_conf = cnn_prob.max(1)
print(
    "  MobileNetV2 done. Example (class, confidence):",
    (class_names[cnn_pred[0]], round(float(cnn_conf[0]), 3)),
)


# ============================================================================
# STEP 5 — Comparison + justification
# ============================================================================
print("\n[Step 5] Comparison on the test set...")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


def evaluate(name, y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    print(f"\n  {name} — accuracy: {acc:.3f}   macro-F1: {f1m:.3f}")
    print(
        classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_title(f"{name} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"outputs/cm_{name}.png")
    plt.close()
    return acc, f1m


acc_svm, f1_svm = evaluate("SVM", y_test, svm_pred)
acc_cnn, f1_cnn = evaluate("MobileNetV2", y_test, cnn_pred)

# latency (seconds per image)
t0 = time.time()
_ = svm.predict_proba(F_test)
svm_sec = (time.time() - t0) / len(F_test)
t0 = time.time()
_ = model.predict(test_ds, verbose=0)
cnn_sec = (time.time() - t0) / len(X_test)
print(f"\n  SVM latency:         {svm_sec * 1000:.1f} ms/image")
print(f"  MobileNetV2 latency: {cnn_sec * 1000:.1f} ms/image")

import pandas as pd

comparison = pd.DataFrame(
    {
        "Technique": ["SVM (HOG+colour+GLCM)", "MobileNetV2 (transfer learning)"],
        "Accuracy": [round(acc_svm, 3), round(acc_cnn, 3)],
        "Macro-F1": [round(f1_svm, 3), round(f1_cnn, 3)],
        "Latency (ms/img)": [round(svm_sec * 1000, 1), round(cnn_sec * 1000, 1)],
    }
)
print("\n  Comparison table:")
print(comparison.to_string(index=False))
comparison.to_csv("outputs/comparison_table.csv", index=False)


# ============================================================================
# STEP 6 — One-image inference demo
# ============================================================================
print("\n[Step 6] One-image inference demo...")
i = 0
true_label = class_names[y_test[i]]
print(f"\n  True label: {true_label}")

feat = extract_features(X_test[i]).reshape(1, -1)
proba_s = svm.predict_proba(feat)[0]
print(
    f"  SVM          -> class: {class_names[proba_s.argmax()]:<10} confidence: {proba_s.max():.3f}"
)

img_arr = np.array(X_test[i].convert("RGB").resize((IMG, IMG)), dtype="float32")[
    None, ...
]
proba_c = model.predict(img_arr, verbose=0)[0]
print(
    f"  MobileNetV2  -> class: {class_names[proba_c.argmax()]:<10} confidence: {proba_c.max():.3f}"
)

plt.imshow(X_test[i])
plt.title(f"True: {true_label}")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/demo_prediction.png")
plt.close()

print("\n" + "=" * 64)
print("DONE. All figures saved to the outputs/ folder:")
print("  class_distribution.png, samples_per_class.png,")
print("  mobilenetv2_training.png, cm_SVM.png, cm_MobileNetV2.png,")
print("  comparison_table.csv, demo_prediction.png")
print("=" * 64)
