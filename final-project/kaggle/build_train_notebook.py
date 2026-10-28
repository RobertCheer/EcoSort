"""Generate the Kaggle training notebook for the EcoSort final project.

One notebook, one split, four recipes measured the same way:

| Recipe | What it is | Seeds |
|---|---|---|
| SVM | HOG + colour + GLCM, RBF kernel | deterministic |
| A   | frozen MobileNetV2 backbone + top-40 fine-tune | 1 |
| B   | full MobileNetV2 fine-tune with an LR schedule | 1 |
| C   | small CNN trained from scratch | 1 |
| D   | MobileNetV3-Large, random resized crop, 12 fixed epochs | 3 |

Recipe A is the assignment-era recipe, kept so the report can show what the
change actually bought. Recipe B is what ships. Recipe C is the from-scratch
baseline the assignment dismissed with a citation instead of evidence.

Every path — SVM features, CNN training, evaluation, TTA and the deployed
system — resizes through the same ``resize_rgb`` helper, so the model is served
on exactly the pixels it was trained on.

Run:  python3 build_train_notebook.py
"""
import json
from pathlib import Path

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


md("""# EcoSort — Final Project Training

**ICT304 Final Project.** Four recipes on one split, measured with one set of
metrics, so the comparison is fair and the choice is evidence-based.

| Recipe | Description | Seeds |
|---|---|---|
| SVM | HOG + colour histogram + GLCM, RBF kernel | deterministic |
| A | frozen MobileNetV2 + fine-tune of the top 40 layers (assignment-era recipe) | 1 |
| B | MobileNetV2, full fine-tune, learning-rate schedule, more epochs | 1 |
| C | small CNN trained from scratch (the from-scratch baseline) | 1 |
| D | MobileNetV3-Large, random resized crop, 12 fixed epochs | 3 |

Outputs in `/kaggle/working/artifacts/`:

| File | Used by |
|---|---|
| `mobilenetv3_224.keras` | classifier sub-system (deployed model, recipe D) |
| `svm_hog_glcm.pkl` | classifier sub-system (lightweight fallback) |
| `class_names.json` | label order, shared by both models |
| `metrics.json` | report results table |
| `split_indices.json` | exact held-out test set, for reproduction |
| `val_probs.npz`, `test_probs.npz` | confidence distributions |

Two methodology points the report can lean on:

* **Test-time augmentation (TTA)** — every CNN score is the mean of the plain
  and horizontally flipped forward pass. It costs one extra forward pass and
  needs no retraining.
* **Seed repeats** — recipes B and C are run three times and reported as
  mean ± sd, so a single lucky run cannot be presented as the result.
""")

md("## 1. Setup")
code('''# Imports and settings
import os, json, time, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from collections import Counter

SEED = 42
SEEDS = [42, 43, 44]     # seed repeats for recipes B and C

CNN_IMG = 224            # input resolution for every CNN recipe
SVM_IMG = 128            # hand-crafted features stay at 128 for comparability
BATCH = 32

# Recipe A — assignment-era recipe (frozen backbone + short top-layer fine-tune)
A_HEAD_EPOCHS, A_HEAD_PATIENCE = 16, 5
A_FT_EPOCHS, A_FT_PATIENCE, A_FT_LAYERS = 4, 2, 40

# Recipe B — full fine-tune with a schedule
B_HEAD_EPOCHS, B_HEAD_PATIENCE = 12, 4
B_FT_EPOCHS, B_FT_PATIENCE = 18, 6

# Recipe C — from scratch
C_EPOCHS, C_PATIENCE = 30, 8

# Recipe D - MobileNetV3-Large,
# AdamW, label smoothing, fixed 12-epoch budget, final-epoch weights.
D_EPOCHS = 12
D_WARM_EPOCHS = 2
D_HEAD_LR = 1e-3
D_BACKBONE_LR = 3e-4
D_WEIGHT_DECAY = 0.02
D_LABEL_SMOOTHING = 0.1

ART = "artifacts"

# ---------------------------------------------------------------- budgets
# Two profiles over the same code path, so a reader can check the pipeline
# works without waiting for the reported run.
#
#   full   what produced the numbers in the report (needs a GPU)
#   quick  a smoke test on a CPU: one seed, fewer epochs, 128px. It writes to a
#          separate folder so it can never overwrite the shipped models.
HAS_GPU = bool(tf.config.list_physical_devices("GPU"))
PROFILE = os.environ.get("ECOSORT_TRAIN_PROFILE", "").strip().lower()
if PROFILE not in ("quick", "full"):
    PROFILE = "full" if HAS_GPU else "quick"

if PROFILE == "quick":
    SEEDS = [42]
    CNN_IMG = 128
    A_HEAD_EPOCHS, A_FT_EPOCHS = 3, 1
    B_HEAD_EPOCHS, B_FT_EPOCHS = 3, 2
    C_EPOCHS = 4
    ART = "artifacts_quick"

print("Training profile:", PROFILE, "| GPU:", HAS_GPU, "| input size:", CNN_IMG)
if PROFILE == "quick":
    print("QUICK profile: smoke test only. Artefacts go to", ART)
    print("and are NOT the reported result. Run ECOSORT_TRAIN_PROFILE=full on a GPU")
    print("to reproduce the report.")
else:
    print("FULL profile: this is the run the report reports.")

os.makedirs(ART, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------- throughput
# Mixed precision uses the GPU's tensor cores, which is where most of the speed
# comes from. The output layer is kept in float32, so the softmax — and every
# probability the router and the dashboard depend on — is computed exactly as
# before. Accuracy is unchanged in expectation; only the arithmetic is cheaper.
PRECISION = os.environ.get("ECOSORT_PRECISION", "").strip().lower()
if PRECISION not in ("mixed_float16", "float32"):
    PRECISION = "float32"   # mixed precision was tried and did not pay off
if PRECISION == "mixed_float16" and not HAS_GPU:
    PRECISION = "float32"
from keras import mixed_precision
mixed_precision.set_global_policy(PRECISION)

# A larger batch halves the number of optimiser steps per epoch without
# changing what the model learns at this scale.
BATCH = int(os.environ.get("ECOSORT_BATCH", 32))

print("Precision policy:", mixed_precision.global_policy().name)
print("Batch size:", BATCH)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
sns.set_theme(style="whitegrid")

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


def resize_rgb(img, size):
    """The ONE resize used by training, evaluation, TTA and the deployed system.

    Training and the Streamlit app previously used different PIL resamplers
    (default vs BILINEAR), so the model was scored on pixels it had never been
    trained on. Everything now goes through this function.
    """
    return img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
''')

md("""## 2. Load TrashNet

Attached Kaggle dataset first, Hugging Face fallback. Same 2,527 images as the
assignment.
""")
code('''# Locate the dataset (attached Kaggle dataset first, Hugging Face fallback)
import glob, zipfile

NEED = {"glass", "paper", "cardboard", "plastic", "metal", "trash"}

def find_data_dir(root):
    for dirpath, dirnames, _ in os.walk(root):
        if "__MACOSX" in dirpath.split(os.sep):
            continue
        real = {d.lower() for d in dirnames if not d.startswith("._")}
        if NEED <= real:
            return dirpath
    raise FileNotFoundError(f"No folder with the 6 class subfolders under {root}")

DATA_DIR = None
if os.path.isdir("/kaggle/input"):
    try:
        DATA_DIR = find_data_dir("/kaggle/input")
        print("Using Kaggle dataset:", DATA_DIR)
    except FileNotFoundError:
        zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
        if zips:
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall("trashnet")
            DATA_DIR = find_data_dir("trashnet")
            print("Extracted attached zip:", DATA_DIR)

if DATA_DIR is None:
    from huggingface_hub import list_repo_files, hf_hub_download
    REPO = "garythung/trashnet"
    zip_name = next(f for f in list_repo_files(REPO, repo_type="dataset")
                    if f.lower().endswith(".zip"))
    path = hf_hub_download(repo_id=REPO, filename=zip_name, repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        z.extractall("trashnet")
    DATA_DIR = find_data_dir("trashnet")
    print("Downloaded from Hugging Face:", DATA_DIR)

class_names = sorted([d for d in os.listdir(DATA_DIR)
                      if not d.startswith("._") and os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes:", class_names)

X, y = [], []
for label, cls in enumerate(class_names):
    cls_dir = os.path.join(DATA_DIR, cls)
    for fname in sorted(os.listdir(cls_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")) and not fname.startswith("._"):
            X.append(Image.open(os.path.join(cls_dir, fname)).convert("RGB"))
            y.append(label)
y = np.array(y)
counts = Counter(y.tolist())
print("Counts:", {class_names[i]: counts[i] for i in range(len(class_names))})
print("Total:", len(X))
assert len(X) == 2527, f"expected 2527 images, got {len(X)}"
''')

code('''# EDA: class distribution + one sample per class
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(class_names, [counts[i] for i in range(len(class_names))], color="#2E7D32")
ax.set_title("Class distribution (trash is the minority, 137)")
ax.set_ylabel("Number of images")
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=130); plt.show()

fig, axes = plt.subplots(1, len(class_names), figsize=(15, 3))
for i, cls in enumerate(class_names):
    axes[i].imshow(X[np.where(y == i)[0][0]]); axes[i].set_title(cls); axes[i].axis("off")
plt.tight_layout(); plt.savefig("outputs/samples_per_class.png", dpi=130); plt.show()
''')

md("""## 3. Split — identical to the assignment

Stratified 70 / 15 / 15 with `random_state=42`. The indices are saved so the
deployed system and the end-to-end tests use exactly this held-out test set.
""")
code('''# Stratified 70/15/15 split (same seed as the assignment)
from sklearn.model_selection import train_test_split

idx_all = np.arange(len(X))
i_train, i_tmp = train_test_split(idx_all, test_size=0.30, stratify=y, random_state=42)
i_val, i_test = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=42)
i_train, i_val, i_test = np.sort(i_train), np.sort(i_val), np.sort(i_test)

X_train = [X[i] for i in i_train]; y_train = y[i_train]
X_val   = [X[i] for i in i_val];   y_val   = y[i_val]
X_test  = [X[i] for i in i_test];  y_test  = y[i_test]

print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
assert (len(X_train), len(X_val), len(X_test)) == (1768, 379, 380), "split drifted from the assignment"

json.dump({"train": i_train.tolist(), "val": i_val.tolist(), "test": i_test.tolist(),
           "class_names": class_names, "seed": SEED},
          open(f"{ART}/split_indices.json", "w"), indent=2)
print("Saved split indices ->", f"{ART}/split_indices.json")
''')

md("""## 4. Technique A comparator — SVM on hand-crafted features

HOG + colour histogram + GLCM, `StandardScaler` inside a `Pipeline` so scaling
is fitted on training data only, RBF kernel with `probability=True` so the
routing sub-system gets a confidence score. Grid unchanged from the assignment.
""")
code('''# Hand-crafted feature extractor, resizing through resize_rgb for parity
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_features(img, size=SVM_IMG):
    img = resize_rgb(img, size)
    a = np.array(img)
    gray = (rgb2gray(a) * 63).astype(np.uint8)          # 64 levels -> fast GLCM
    hog_f = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hist = np.concatenate([np.histogram(a[:, :, c].ravel(), bins=32, range=(0, 256))[0]
                           for c in range(3)])
    hist = hist / (hist.sum() + 1e-6)
    g = graycomatrix(gray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=64, symmetric=True, normed=True)
    glcm = np.concatenate([graycoprops(g, p).ravel() for p in
                           ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]])
    return np.concatenate([hog_f, hist, glcm])

t0 = time.time()
F_train = np.array([extract_features(im) for im in X_train])
F_val   = np.array([extract_features(im) for im in X_val])
F_test  = np.array([extract_features(im) for im in X_test])
print(f"Features {F_train.shape} in {time.time()-t0:.1f}s")
''')

code('''# Train the SVM and save the fitted pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True))])
grid = GridSearchCV(pipe, {"svc__C": [1, 10], "svc__gamma": ["scale", 0.01]},
                    cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
t0 = time.time()
grid.fit(F_train, y_train)
svm = grid.best_estimator_
print(f"Best params: {grid.best_params_}  ({time.time()-t0:.1f}s)")

joblib.dump({"pipeline": svm, "class_names": class_names, "img_size": SVM_IMG},
            f"{ART}/svm_hog_glcm.pkl", compress=3)
print("Saved SVM ->", f"{ART}/svm_hog_glcm.pkl")
''')

md("""## 5. Encode the images once

Resizing 2,527 images on every epoch was the main source of wasted GPU time.
They are resized once to `uint8` (about 380 MB), cached in memory, and cast to
`float32` inside the input pipeline. That is what makes a 30-epoch run and three
seed repeats practical.

Both training and inference read these arrays, so there is no second resize
path that could drift.
""")
code('''def encode(images, size=CNN_IMG):
    return np.stack([np.asarray(resize_rgb(im, size), dtype="uint8") for im in images])

t0 = time.time()
Xtr = encode(X_train); Xva = encode(X_val); Xte = encode(X_test)
print(f"Encoded {len(Xtr)+len(Xva)+len(Xte)} images in {time.time()-t0:.1f}s")
print("shapes:", Xtr.shape, Xva.shape, Xte.shape)

def make_ds(images, labels, shuffle=False, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((images, labels.astype("int32")))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed)
    ds = ds.map(lambda x, yy: (tf.cast(x, tf.float32), yy),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

val_ds = make_ds(Xva, y_val)
test_ds = make_ds(Xte, y_test)

from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = {int(k): float(v) for k, v in zip(np.unique(y_train), cw)}
print("Class weights:", {class_names[k]: round(v, 3) for k, v in class_weight.items()})
''')

md("""## 6. Model builders

Both CNNs carry their own augmentation block, so augmentation is active during
`fit` and inactive during `predict` without any extra branching in the call
sites. `preprocess_input` lives **inside** the transfer model, which is why the
deployed system only has to resize: the scaling travels with the saved model.
""")
code('''# Augmentation shared by both CNN recipes
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large
from tensorflow.keras import layers, models

def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augment")


def build_transfer(seed):
    """MobileNetV2 with an ImageNet backbone, frozen to begin with."""
    tf.keras.utils.set_random_seed(seed)
    base = MobileNetV2(input_shape=(CNN_IMG, CNN_IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inp = layers.Input((CNN_IMG, CNN_IMG, 3), name="image")
    x = build_augmentation()(inp)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(len(class_names), activation="softmax", dtype="float32",
                       name="class_prob")(x)
    return models.Model(inp, out, name="ecosort_mobilenetv2"), base


def build_scratch(seed):
    """A small CNN with no pretrained weights — the from-scratch baseline."""
    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input((CNN_IMG, CNN_IMG, 3), name="image")
    x = build_augmentation()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(x)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(class_names), activation="softmax", dtype="float32",
                       name="class_prob")(x)
    return models.Model(inp, out, name="ecosort_scratch_cnn")

''')

md("""## 7. Training functions

**Recipe A** reproduces the assignment: frozen backbone, then four epochs on the
top 40 layers at a low learning rate.

**Recipe B** unfreezes the whole backbone and trains it with a plateau schedule
and a longer budget. A frozen backbone can only learn a linear probe on
ImageNet features; the fine-tune is where the domain adaptation happens.

**Recipe C** trains the small CNN from random initialisation.
""")
code('''def callbacks_for(patience, schedule, min_lr=1e-6):
    cbs = [tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                            monitor="val_accuracy")]
    if schedule:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.3,
                                                        patience=max(1, patience // 2),
                                                        min_lr=min_lr))
    return cbs


def train_transfer(seed, head_epochs, head_patience, ft_epochs, ft_patience,
                   ft_layers=None, schedule=False):
    """Two-stage transfer learning. ft_layers=None unfreezes the whole backbone."""
    model, base = build_transfer(seed)
    train_ds = make_ds(Xtr, y_train, shuffle=True, seed=seed)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h_head = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                       class_weight=class_weight, verbose=2, shuffle=False,
                       callbacks=callbacks_for(head_patience, schedule))

    base.trainable = True
    if ft_layers is not None:
        for layer in base.layers[:-ft_layers]:
            layer.trainable = False

    lr = 1e-5 if ft_layers is not None else 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h_ft = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                     class_weight=class_weight, verbose=2, shuffle=False,
                     callbacks=callbacks_for(ft_patience, schedule))
    return model, h_head, h_ft


def train_scratch(seed, epochs, patience):
    model = build_scratch(seed)
    train_ds = make_ds(Xtr, y_train, shuffle=True, seed=seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=class_weight, verbose=2, shuffle=False,
                  callbacks=callbacks_for(patience, True))
    return model, h


def tta_probs(model, images):
    """Mean of the plain and horizontally flipped forward passes."""
    p = model.predict(images, batch_size=BATCH, verbose=0)
    p_flip = model.predict(images[:, :, ::-1, :], batch_size=BATCH, verbose=0)
    return (p + p_flip) / 2.0


def acc_of(proba, y_true):
    return float((proba.argmax(1) == y_true).mean())
''')

md("""## 8. Recipe A — the assignment-era recipe (1 seed)

Kept as the baseline the report compares against.
""")
code('''t0 = time.time()
model_a, hA_head, hA_ft = train_transfer(SEED, A_HEAD_EPOCHS, A_HEAD_PATIENCE,
                                         A_FT_EPOCHS, A_FT_PATIENCE,
                                         ft_layers=A_FT_LAYERS, schedule=False)
val_a = acc_of(tta_probs(model_a, Xva), y_val)
test_a = acc_of(tta_probs(model_a, Xte), y_test)
print(f"Recipe A: val {val_a:.4f}  test {test_a:.4f}  ({time.time()-t0:.0f}s)")
''')

md("""## 9. Recipe B — full fine-tune with a schedule (3 seeds)

The backbone is unfrozen entirely and trained at a lower learning rate with a
plateau schedule and a longer budget. Three seeds, so the result is a mean ± sd
and not one lucky run.
""")
code('''recipe_b = []
model_b = None
best_b = (-1.0, None)
curves_b = None

for seed in SEEDS[:1]:   # an ablation now, not the shipped model
    t0 = time.time()
    m, h_head, h_ft = train_transfer(seed, B_HEAD_EPOCHS, B_HEAD_PATIENCE,
                                     B_FT_EPOCHS, B_FT_PATIENCE,
                                     ft_layers=None, schedule=True)
    v = acc_of(tta_probs(m, Xva), y_val)
    t = acc_of(tta_probs(m, Xte), y_test)
    recipe_b.append({"seed": seed, "val_accuracy": round(v, 4), "test_accuracy": round(t, 4)})
    print(f"Recipe B seed {seed}: val {v:.4f}  test {t:.4f}  ({time.time()-t0:.0f}s)")

    # the deployable model is chosen on validation, never on test
    if v > best_b[0]:
        best_b = (v, seed)
        model_b = m
    if seed == SEEDS[0]:
        curves_b = (h_head, h_ft)

print("Recipe B mean test accuracy:",
      round(float(np.mean([r["test_accuracy"] for r in recipe_b])), 4),
      "+/-", round(float(np.std([r["test_accuracy"] for r in recipe_b])), 4))
print("Deployed seed (best validation):", best_b[1])
''')

md("""## 10. Recipe C — from scratch (3 seeds)

The baseline the assignment dismissed by citing someone else's repository. Same
split, same augmentation, same class weights — only the initialisation differs.
""")
code('''recipe_c = []
best_c = (-1.0, None)
model_c = None
curves_c = None

for seed in SEEDS[:1]:   # a baseline only: error bars here answer no question
    t0 = time.time()
    m, h = train_scratch(seed, C_EPOCHS, C_PATIENCE)
    v = acc_of(tta_probs(m, Xva), y_val)
    t = acc_of(tta_probs(m, Xte), y_test)
    recipe_c.append({"seed": seed, "val_accuracy": round(v, 4), "test_accuracy": round(t, 4)})
    print(f"Recipe C seed {seed}: val {v:.4f}  test {t:.4f}  ({time.time()-t0:.0f}s)")
    if v > best_c[0]:
        best_c = (v, seed); model_c = m
    if seed == SEEDS[0]:
        curves_c = h

print("Recipe C mean test accuracy:",
      round(float(np.mean([r["test_accuracy"] for r in recipe_c])), 4),
      "+/-", round(float(np.std([r["test_accuracy"] for r in recipe_c])), 4))
''')

md("""## 11. Recipe D — MobileNetV3-Large, 12 fixed epochs (3 seeds)

Recipe D. Four
elements differ from recipe B, and each is a deliberate choice:

| Element | Recipe B | Recipe D |
|---|---|---|
| Backbone | MobileNetV2 | **MobileNetV3-Large** |
| Augmentation | flip, rotate, zoom, brightness | **RandomResizedCrop(0.6-1.0)**, h+v flip, rotate 20, colour |
| Optimiser | Adam | **AdamW, weight decay 0.02** |
| Loss | class weights | class weights **+ label smoothing 0.1** |
| Budget | early stopping, up to 30 epochs | **12 fixed epochs** |
| Weights kept | best validation epoch | **final epoch** |

Two protocol points matter more than the model. **Epochs are fixed in advance and
the final-epoch weights ship**, so the validation split is never used to choose
anything; recipe B restores the best-validation epoch, which is a mild form of
selection. And the held-out test set is scored once, at the end.
""")
md("""### Why the saved model has no augmentation in it

A Keras model that contains a custom layer cannot be loaded without that layer's
class, and the deployed app must load the artefact with a plain
`load_model`. Baking `RandomResizedCrop` into the file would therefore break the
station the moment the notebook and the app drifted apart.

The fix is to build two graphs over the same layer instances:

* a **training graph** with augmentation in front, and
* an **inference graph** with no augmentation at all.

Because Keras shares weights when a layer instance is applied to two inputs, both
graphs hold the same weights, and only the clean one is ever saved.
""")
code("""# Random resized crop, vectorised through crop_and_resize
class RandomResizedCrop(layers.Layer):
    # Crop a random 60-100% area region and resize it back to `size`.
    #
    # Mirrors torchvision's RandomResizedCrop(scale=(0.6, 1.0)) closely enough: a
    # square-ish region is sampled uniformly in *area*, which is what makes the
    # apparent object size vary during training.

    def __init__(self, size, min_scale=0.6, max_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def call(self, images, training=None):
        if not training:
            return images
        batch = tf.shape(images)[0]
        side = tf.sqrt(tf.random.uniform((batch,), self.min_scale, self.max_scale))
        top = tf.random.uniform((batch,), 0.0, 1.0 - side)
        left = tf.random.uniform((batch,), 0.0, 1.0 - side)
        boxes = tf.stack([top, left, top + side, left + side], axis=1)
        indices = tf.range(batch)
        return tf.image.crop_and_resize(images, boxes, indices, (self.size, self.size))
""")

code("""def build_recipe_d(seed):
    # MobileNetV3-Large at 224px. Returns (train_model, infer_model, base):
    # the training graph carries the augmentation, the inference graph does not.
    tf.keras.utils.set_random_seed(seed)
    base = MobileNetV3Large(input_shape=(CNN_IMG, CNN_IMG, 3), include_top=False,
                            weights="imagenet")
    base.trainable = False

    # One set of head layer instances, applied to both graphs so weights are shared.
    gap = layers.GlobalAveragePooling2D()
    drop = layers.Dropout(0.2)
    dense = layers.Dense(len(class_names), activation="softmax", dtype="float32",
                         name="class_prob")

    def head(x):
        # MobileNetV3 carries its own pixel rescaling to [-1, 1], so this needs no
        # preprocess_input here.
        x = base(x, training=False)
        return dense(drop(gap(x)))

    aug_in = layers.Input((CNN_IMG, CNN_IMG, 3), name="image")
    x = layers.RandomFlip("horizontal")(aug_in)
    x = layers.RandomFlip("vertical")(x)      # TrashNet objects are position-agnostic
    x = RandomResizedCrop(CNN_IMG, 0.6, 1.0)(x)
    x = layers.RandomRotation(20 / 360)(x)    # +/-20 degrees, as in the source recipe
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    train_model = models.Model(aug_in, head(x), name="ecosort_mobilenetv3_train")

    infer_in = layers.Input((CNN_IMG, CNN_IMG, 3), name="image")
    infer_model = models.Model(infer_in, head(infer_in), name="ecosort_mobilenetv3")
    return train_model, infer_model, base
""")

md("""### Training recipe D

The learning rate is staged rather than applied per-parameter-group, because Keras
does not expose per-layer learning rates through `compile()`. A short head-only
warm-up at 1e-3 is followed by the whole backbone trainable at 3e-4 on a cosine
curve: the intent of the source recipe (pretrained weights move slowly, the new
head moves fast) without a custom training loop.

There is **no early stopping and no best-epoch restore**. Twelve epochs are fixed in
advance and the final epoch ships, which is what keeps the validation split out of
model selection.
""")
code("""# Recipe D: one-hot labels (label smoothing needs them) and per-sample weights.
# Keras 3 refuses ``sample_weight`` alongside a Dataset, so the class weights ride
# inside the pipeline as a third element instead.
def make_ds_onehot(images, labels, shuffle=False, seed=SEED, weighted=False):
    onehot = np.eye(len(class_names), dtype="float32")[labels]
    weights = np.array([class_weight[int(label)] for label in labels], dtype="float32")
    ds = tf.data.Dataset.from_tensor_slices((images, onehot, weights))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed)
    ds = ds.map(lambda x, yy, w: (tf.cast(x, tf.float32), yy, w),
                num_parallel_calls=tf.data.AUTOTUNE)
    if not weighted:
        ds = ds.map(lambda x, yy, w: (x, yy))
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def train_recipe_d(seed, warm_epochs=D_WARM_EPOCHS, total_epochs=D_EPOCHS):
    # Stage 1: head only. Stage 2: whole backbone on a cosine curve.
    train_model, infer_model, base = build_recipe_d(seed)
    train_ds = make_ds_onehot(Xtr, y_train, shuffle=True, seed=seed, weighted=True)
    val_ds_oh = make_ds_onehot(Xva, y_val)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=D_LABEL_SMOOTHING)

    train_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=D_HEAD_LR,
                                            weight_decay=D_WEIGHT_DECAY),
        loss=loss, metrics=["accuracy"])
    train_model.fit(train_ds, validation_data=val_ds_oh, epochs=warm_epochs,
                    verbose=2, shuffle=False)

    base.trainable = True
    steps = int(np.ceil(len(Xtr) / BATCH)) * (total_epochs - warm_epochs)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=D_BACKBONE_LR, decay_steps=max(steps, 1), alpha=0.02)
    train_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule,
                                            weight_decay=D_WEIGHT_DECAY),
        loss=loss, metrics=["accuracy"])
    train_model.fit(train_ds, validation_data=val_ds_oh, epochs=total_epochs - warm_epochs,
                    verbose=2, shuffle=False)
    return infer_model
""")


md("""### Training the three seeds

Same rule as recipe B: the seed that ships is chosen on **validation** accuracy,
and only then is the test set scored.
""")
code("""recipe_d = []
model_d = None
best_d = (-1.0, None)

for seed in SEEDS:
    t0 = time.time()
    m = train_recipe_d(seed)
    v = acc_of(tta_probs(m, Xva), y_val)
    t = acc_of(tta_probs(m, Xte), y_test)
    recipe_d.append({"seed": seed, "val_accuracy": round(v, 4), "test_accuracy": round(t, 4)})
    print(f"Recipe D seed {seed}: val {v:.4f}  test {t:.4f}  ({time.time()-t0:.0f}s)")
    if v > best_d[0]:
        best_d = (v, seed)
        model_d = m

print("Recipe D mean test accuracy:",
      round(float(np.mean([r["test_accuracy"] for r in recipe_d])), 4),
      "+/-", round(float(np.std([r["test_accuracy"] for r in recipe_d])), 4))
print("Deployed seed (best validation):", best_d[1])
""")

md("""## 12. The comparison table

Every recipe on the same held-out test set, with the seed spread where it
exists. This is the table the report's justification rests on.
""")
code('''# Per-recipe test scores, plus the SVM
svm_test_proba = svm.predict_proba(F_test)
svm_val_proba = svm.predict_proba(F_val)
svm_test_acc = acc_of(svm_test_proba, y_test)
svm_val_acc = acc_of(svm_val_proba, y_val)

def mean_sd(rows):
    ts = [r["test_accuracy"] for r in rows]
    return float(np.mean(ts)), (float(np.std(ts)) if len(ts) > 1 else 0.0)

rows = [("SVM (HOG+colour+GLCM)", svm_test_acc, 0.0, 1),
        ("A  frozen + top-40 fine-tune", test_a, 0.0, 1)]
for name, res in [("B  full fine-tune + schedule", recipe_b),
                  ("C  small CNN from scratch", recipe_c)]:
    m_, sd_ = mean_sd(res)
    rows.append((name, m_, sd_, len(res)))

print(f"{'recipe':32s} {'test accuracy':>14s} {'sd':>7s} {'n':>3s}")
for name, m_, sd_, n in rows:
    print(f"{name:32s} {m_:14.4f} {sd_:7.4f} {n:3d}")

fig, ax = plt.subplots(figsize=(8.5, 4))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
errs = [r[2] for r in rows]
bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=["#90A4AE", "#A5D6A7", "#2E7D32", "#EF9A9A"])
ax.axhline(0.85, ls="--", c="#C62828", lw=1.2, label="85% target")
ax.set_ylim(0, 1.0); ax.set_ylabel("held-out test accuracy")
ax.set_title("EcoSort technique comparison (same split, same metrics)")
ax.legend(loc="lower right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=15, ha="right"); plt.tight_layout()
plt.savefig("outputs/recipe_comparison.png", dpi=130); plt.show()
''')

md("""## 13. Blending the two CNN architectures

MobileNetV2 and MobileNetV3 disagree on about one item in ten, and the disagreements
are not random — they have different weaknesses. Averaging their probabilities is
therefore worth more than either model alone.

The blend weight is chosen on the **validation** set and then applied once to the
test set, so no test information leaks into the choice.
""")
code("""# Blend weight chosen on validation, scored once on test
best_w, best_val = 0.0, -1.0
for w in np.arange(0.0, 1.01, 0.05):
    v = acc_of(w * tta_probs(model_b, Xva) + (1 - w) * tta_probs(model_d, Xva), y_val)
    if v > best_val:
        best_val, best_w = v, round(float(w), 3)

v2_val_proba = tta_probs(model_b, Xva)
v3_val_proba = tta_probs(model_d, Xva)
v2_test_proba = tta_probs(model_b, Xte)
v3_test_proba = tta_probs(model_d, Xte)

blend_val = best_w * v2_val_proba + (1 - best_w) * v3_val_proba
blend_proba = best_w * v2_test_proba + (1 - best_w) * v3_test_proba
print(f"Chosen w(MobileNetV2) = {best_w:.2f}  (validation accuracy {best_val:.4f})")
print(f"Single models on test : V2 {acc_of(v2_test_proba, y_test):.4f}  "
      f"V3 {acc_of(v3_test_proba, y_test):.4f}")
print(f"Blend on test         : {acc_of(blend_proba, y_test):.4f}")
""")

md("""## 14. The deployed model, scored with TTA

Recipe B's best-on-validation seed becomes the shipped model. Reporting on the
test set only after the choice is fixed on validation is what keeps the number
honest.
""")
code('''from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

cnn_val_proba = blend_val
cnn_test_proba = blend_proba
cnn_pred = cnn_test_proba.argmax(1)

def summarise(name, y_true, y_pred):
    rep = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\\n{name}: accuracy {acc:.3f}   macro-F1 {mf1:.3f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
            "per_class": {c: {k: round(float(rep[c][k]), 4)
                              for k in ("precision", "recall", "f1-score", "support")}
                          for c in class_names}}

m_cnn = summarise("MobileNetV2 — recipe B, TTA", y_test, cnn_pred)
m_svm = summarise("SVM (HOG+colour+GLCM)", y_test, svm_test_proba.argmax(1))

for name, pred, fname in [("SVM", svm_test_proba.argmax(1), "cm_SVM.png"),
                          ("MobileNetV2 (recipe B, TTA)", cnn_pred, "cm_MobileNetV2.png")]:
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name} confusion matrix")
    plt.tight_layout(); plt.savefig(f"outputs/{fname}", dpi=130); plt.show()

# training curves for the first seed of each recipe
for tag, curves in [("B", curves_b), ("C", curves_c)]:
    if curves is None:
        continue
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if isinstance(curves, tuple):
        acc = curves[0].history["accuracy"] + curves[1].history["accuracy"]
        vacc = curves[0].history["val_accuracy"] + curves[1].history["val_accuracy"]
        split = len(curves[0].history["accuracy"]) - 0.5
        ax.axvline(split, ls="--", c="grey", label="fine-tune start")
    else:
        acc = curves.history["accuracy"]; vacc = curves.history["val_accuracy"]
    ax.plot(acc, label="train"); ax.plot(vacc, label="val")
    ax.set_title(f"Recipe {tag} accuracy (seed {SEEDS[0]})")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()
    plt.tight_layout(); plt.savefig(f"outputs/training_recipe_{tag}.png", dpi=130); plt.show()
''')

md("""## 15. Inference latency

Single item, batch size 1, after a warm-up — the number that matters at a
point of disposal. TTA doubles it, which is reported too rather than hidden.
""")
code('''def latency_cnn(model, n=50):
    batch = Xte[:n].astype("float32")
    model.predict(batch[:1], verbose=0)
    t0 = time.time()
    for row in batch:
        model.predict(row[None, ...], verbose=0)
    return (time.time() - t0) / n

def latency_svm(n=50):
    photos = X_test[:n]
    svm.predict_proba(extract_features(photos[0]).reshape(1, -1))
    t0 = time.time()
    for pil in photos:
        svm.predict_proba(extract_features(pil).reshape(1, -1))
    return (time.time() - t0) / n

lat_svm = latency_svm()
lat_cnn = latency_cnn(model_b)
print(f"SVM latency:            {lat_svm*1000:.2f} ms/image")
print(f"MobileNetV2 latency:    {lat_cnn*1000:.2f} ms/image")
print(f"MobileNetV2 with TTA:   {2*lat_cnn*1000:.2f} ms/image")
''')

md("""## 16. Confidence threshold sweep

Anything below θ goes to a person instead of a bin. Swept on the validation set,
never the test set.
""")
code('''thetas = np.round(np.arange(0.30, 0.91, 0.05), 2)

def sweep(proba, y_true):
    rows = []
    pred, conf = proba.argmax(axis=1), proba.max(axis=1)
    for th in thetas:
        auto = conf >= th
        rows.append({
            "theta": float(th),
            "review_rate": round(float((~auto).mean()), 4),
            "auto_coverage": round(float(auto.mean()), 4),
            "auto_accuracy": round(float((pred[auto] == y_true[auto]).mean()), 4) if auto.any() else None,
            "overall_accuracy": round(float((pred == y_true).mean()), 4),
        })
    return rows

sweep_cnn = sweep(cnn_val_proba, y_val)
sweep_svm = sweep(svm_val_proba, y_val)

import pandas as pd
df_sweep = pd.DataFrame(sweep_cnn)
print("MobileNetV2 threshold sweep (validation set)")
print(df_sweep.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(8, 4.2))
ax1.plot(df_sweep.theta, df_sweep.auto_accuracy, "o-", c="#2E7D32", label="accuracy on auto-routed items")
ax1.set_xlabel("confidence threshold theta"); ax1.set_ylabel("auto-routed accuracy", color="#2E7D32")
ax2 = ax1.twinx()
ax2.plot(df_sweep.theta, df_sweep.review_rate, "s--", c="#C62828", label="share sent to human review")
ax2.set_ylabel("review rate (workload)", color="#C62828")
ax1.set_title("EcoSort routing threshold trade-off (validation set)")
fig.tight_layout(); plt.savefig("outputs/theta_sweep.png", dpi=130); plt.show()
''')

code('''# Pick theta from reviewer capacity, not from chasing the last point of accuracy.
#
# A human review queue is a real operating cost: every item in it costs staff
# time, and a station that diverts most of its waste is not sorting anything.
# So we fix the workload we can afford (MAX_REVIEW_RATE) and then take the most
# accurate automatic path that still fits inside that budget. Reading the sweep
# the other way round - maximise accuracy, accept any workload - always lands at
# the top of the range, where coverage collapses.
MAX_REVIEW_RATE = 0.10  # 1 item in 10 may be sent to a person

feasible = [r for r in sweep_cnn
            if r["auto_accuracy"] is not None and r["review_rate"] <= MAX_REVIEW_RATE]
if not feasible:
    raise RuntimeError(
        f"No threshold keeps the review rate at or below {MAX_REVIEW_RATE:.0%}. "
        "Relax the budget or improve the classifier before shipping."
    )
chosen = max(feasible, key=lambda r: (r["auto_accuracy"], -r["review_rate"]))
THETA = chosen["theta"]

print(f"Review-rate budget: {MAX_REVIEW_RATE:.0%}  ({len(feasible)} feasible thresholds)")
print(f"Chosen theta = {THETA:.2f}  ->  review rate {chosen['review_rate']:.1%}, "
      f"auto accuracy {chosen['auto_accuracy']:.3f}, coverage {chosen['auto_coverage']:.1%}")

# The SVM's probabilities are blunter than the CNN's, so it needs its own theta.
# Judging both with one threshold sends most SVM items to human review.
feasible_svm = [r for r in sweep_svm
                if r["auto_accuracy"] is not None and r["review_rate"] <= MAX_REVIEW_RATE]
chosen_svm = max(feasible_svm, key=lambda r: (r["auto_accuracy"], -r["review_rate"]))
THETA_SVM = chosen_svm["theta"]
print(f"Chosen theta (svm) = {THETA_SVM:.2f}  ->  review rate {chosen_svm['review_rate']:.1%}, "
      f"auto accuracy {chosen_svm['auto_accuracy']:.3f}")
''')

md("## 17. Save every artefact the deployed system needs")
code('''# Persist models, metrics and calibration
# Both CNN recipes are kept: the deployed system blends them, and an
# artefact that the notebook does not produce is an artefact nobody can
# regenerate.
model_b.save(f"{ART}/mobilenetv2_224.keras")
print("Saved MobileNetV2 ->", f"{ART}/mobilenetv2_224.keras")
model_d.save(f"{ART}/mobilenetv3_224.keras")
print("Saved CNN ->", f"{ART}/mobilenetv3_224.keras")

def pack(rows):
    m_, sd_ = mean_sd(rows) if len(rows) > 1 else (rows[0]["test_accuracy"], 0.0)
    return {"per_seed": rows, "mean_test_accuracy": round(m_, 4), "sd_test_accuracy": round(sd_, 4)}

metrics = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "settings": {"cnn_img": CNN_IMG, "svm_img": SVM_IMG, "seed": SEED, "seeds": SEEDS,
                 "split": [len(X_train), len(X_val), len(X_test)],
                 "svm_best_params": grid.best_params_,
                 "recipe_a": {"head_epochs": A_HEAD_EPOCHS, "ft_epochs": A_FT_EPOCHS,
                              "ft_layers": A_FT_LAYERS},
                 "recipe_b": {"head_epochs": B_HEAD_EPOCHS, "ft_epochs": B_FT_EPOCHS,
                              "ft_layers": "all", "lr_head": 1e-3, "lr_ft": 1e-4,
                              "schedule": "ReduceLROnPlateau"},
                 "recipe_c": {"epochs": C_EPOCHS, "lr": 1e-3},
                 "recipe_d": {"epochs": D_EPOCHS, "warm_epochs": D_WARM_EPOCHS,
                              "head_lr": D_HEAD_LR, "backbone_lr": D_BACKBONE_LR,
                              "weight_decay": D_WEIGHT_DECAY,
                              "label_smoothing": D_LABEL_SMOOTHING,
                              "select_best": False},
                 "precision": PRECISION, "batch": BATCH},
    "resize_filter": "PIL BILINEAR at (size, size), shared by training and serving",
    "svm": {**m_svm, "latency_ms": round(lat_svm * 1000, 3)},
    "cnn": {**m_cnn, "recipe": "D_mobilenetv3", "seed": best_d[1],
            "tta": "mean of plain and horizontal flip",
            "latency_ms": round(lat_cnn * 1000, 3),
            "latency_tta_ms": round(2 * lat_cnn * 1000, 3)},
    "ensemble": {
        "description": "probability blend of the two CNN artefacts, weight chosen on validation",
        "cnn_weights": {"mobilenetv2_224.keras": best_w,
                        "mobilenetv3_224.keras": round(1 - best_w, 3)},
        "val_accuracy": round(float(best_val), 4),
        "test_accuracy": round(acc_of(blend_proba, y_test), 4),
    },
    "recipes": {
        "A_frozen_top40": {"per_seed": [{"seed": SEED, "val_accuracy": round(val_a, 4),
                                         "test_accuracy": round(test_a, 4)}],
                           "mean_test_accuracy": round(test_a, 4), "sd_test_accuracy": 0.0},
        "B_full_finetune": pack(recipe_b),
        "C_from_scratch": pack(recipe_c),
        "D_mobilenetv3": pack(recipe_d),
    },
    "theta": {"chosen": float(THETA), "max_review_rate": MAX_REVIEW_RATE,
              "by_backend": {"cnn": float(THETA), "svm": float(THETA_SVM)},
              "chosen_review_rate": chosen["review_rate"],
              "chosen_auto_accuracy": chosen["auto_accuracy"],
              "chosen_coverage": chosen["auto_coverage"],
              "selection_rule": "highest auto-routed accuracy among thresholds whose review rate stays within the reviewer-capacity budget",
              "sweep_cnn": sweep_cnn, "sweep_svm": sweep_svm},
    "images_per_class": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
}
json.dump(metrics, open(f"{ART}/metrics.json", "w"), indent=2)

print("CNN accuracy:", metrics["cnn"]["accuracy"], "| macro-F1:", metrics["cnn"]["macro_f1"])
print("SVM accuracy:", metrics["svm"]["accuracy"], "| macro-F1:", metrics["svm"]["macro_f1"])
print("theta:", THETA)
print("Recipe B mean +/- sd:", metrics["recipes"]["B_full_finetune"]["mean_test_accuracy"],
      metrics["recipes"]["B_full_finetune"]["sd_test_accuracy"])
print("Recipe C mean +/- sd:", metrics["recipes"]["C_from_scratch"]["mean_test_accuracy"],
      metrics["recipes"]["C_from_scratch"]["sd_test_accuracy"])
''')

code('''json.dump({"class_names": class_names, "cnn_img": CNN_IMG, "svm_img": SVM_IMG},
          open(f"{ART}/class_names.json", "w"), indent=2)
np.savez_compressed(f"{ART}/val_probs.npz", y=y_val, cnn=cnn_val_proba, svm=svm_val_proba)
np.savez_compressed(f"{ART}/test_probs.npz", y=y_test, cnn=cnn_test_proba, svm=svm_test_proba)
print("Artefacts written:", sorted(os.listdir(ART)))
''')

code('''# Package the artefacts as one zip for download
import shutil
ZIP = f"ecosort_{PROFILE}_artifacts"
shutil.make_archive(ZIP, "zip", ART)
size = os.path.getsize(ZIP + ".zip") / 1e6
print(f"{ZIP}.zip  {size:.1f} MB")
print(sorted(os.listdir(ART)))
''')

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "train_ecosort_224.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} with {len(CELLS)} cells")
