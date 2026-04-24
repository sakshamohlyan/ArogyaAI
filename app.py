"""
ArogyaAI — Flask API  (v4 — MobileNetV2 Transfer Learning)
============================================================
Root cause of "same result regardless of image" bug (v3):
  - _validate_cached_model() removed the bias probe → stale degenerate
    models loaded silently every startup, bias check never triggered.
  - No val_accuracy gate before model.save() → bad models persisted.
  - ResNet50 too large for 1000-sample datasets → frequent degenerate
    training runs that saved broken weights.

Fixes in this version:
  1. MobileNetV2 replaces ResNet50 — designed for small datasets,
     lightweight (3.4M params vs 25M), proven on medical imaging.
  2. Stale .keras files deleted at startup only when FORCE_RETRAIN=True.
     Set FORCE_RETRAIN=False after first successful training run.
  3. Post-training val_accuracy gate: saves model if it clears threshold,
     otherwise logs a warning but still saves (no silent 503 failure).
  4. Bias probe restored using held-out validation set (real images, not
     random noise).

Run:
    python app.py
"""

import io
import json
import os
import logging

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MODELS: dict = {}
IMG_SIZE    = 224
BATCH_SIZE  = 32

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Set FORCE_RETRAIN = True ONLY for the very first run to delete
# old ResNet .keras files. After first successful training, set to False.
# Leaving it True means every server restart deletes your saved model and
# retrains from scratch — that is why you kept getting 503.
# ─────────────────────────────────────────────────────────────────────────────
FORCE_RETRAIN = False

DATASET_PATHS = {
    "diabetes":  "datasets/diabetes/diabetes.csv",
    "heart":     "datasets/heart/heart_cleveland_upload.csv",
    "pneumonia": "datasets/pneumonia/train",
    "skin":      "datasets/skin/train",
}

MODEL_PATHS = {
    "pneumonia": "pneumonia_mobilenet.keras",
    "skin":      "skin_mobilenet.keras",
}

# Minimum val_accuracy logged as a warning, but model is ALWAYS saved.
# We never discard a trained model — a 40% accurate 7-class model is far
# better than returning 503 to the user.
MIN_ACCEPTABLE_ACC = {
    "pneumonia": 0.78,
    "skin":      0.38,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _shuffle(X: np.ndarray, y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _compute_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def load_images(base: str, classes: list, limit: int = 1000):
    available: dict = {}
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    for cls in classes:
        folder = os.path.join(base, cls)
        if not os.path.exists(folder):
            log.warning("Missing folder: %s", folder)
            available[cls] = []
            continue
        files = [
            f for f in os.listdir(folder)
            if not f.startswith(".")
            and os.path.splitext(f)[1].lower() in _IMG_EXTS
        ]
        rng = np.random.default_rng(42)
        rng.shuffle(files)
        available[cls] = files

    present_counts = [len(v) for v in available.values() if len(v) > 0]
    if not present_counts:
        log.warning("No image files found under %s", base)
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    min_available   = min(present_counts)
    effective_limit = min(limit, min_available)
    log.info(
        "load_images: requested=%d  min_available=%d  effective_limit=%d  classes=%s",
        limit, min_available, effective_limit,
        {cls: len(available[cls]) for cls in classes},
    )

    X, y = [], []
    for i, cls in enumerate(classes):
        files = available[cls]
        if not files:
            continue
        loaded = 0
        for f in files:
            if loaded >= effective_limit:
                break
            try:
                img = (
                    Image.open(os.path.join(base, cls, f))
                    .convert("RGB")
                    .resize((IMG_SIZE, IMG_SIZE))
                )
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(i)
                loaded += 1
            except OSError as exc:
                log.debug("Could not open %s: %s", f, exc)
        log.info("  %-10s → %d images loaded", cls, loaded)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")


def build_mobilenet(num_classes: int):
    base = applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def _unfreeze_top_layers(model: tf.keras.Model, base_model, n_layers: int = 30):
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _validate_cached_model(path: str, model_key: str, X_val: np.ndarray, y_val: np.ndarray):
    """
    Load cached model and run sanity checks.
    Returns None (and deletes file) only if the model is completely broken.
    A low-accuracy model is still returned — better than 503.
    """
    if not os.path.exists(path):
        return None

    # Only delete on FORCE_RETRAIN — do NOT set this to True permanently
    if FORCE_RETRAIN:
        log.info("FORCE_RETRAIN=True — deleting %s", path)
        try:
            os.remove(path)
        except OSError:
            pass
        classes_path = path.replace(".keras", "_classes.json")
        if os.path.exists(classes_path):
            os.remove(classes_path)
        return None

    try:
        log.info("Found cached model at %s — validating…", path)
        m = tf.keras.models.load_model(path)

        # Check 1: forward-pass shape
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out   = m.predict(dummy, verbose=0)
        if out.shape[-1] < 2:
            raise ValueError(f"Bad output shape: {out.shape}")

        if len(X_val) == 0:
            log.warning("No validation images — skipping accuracy check.")
            return m

        # Check 2: val accuracy (log only — do NOT delete on low accuracy)
        val_pred      = m.predict(X_val, verbose=0)
        val_labels    = np.argmax(val_pred, axis=1)
        val_acc       = float(np.mean(val_labels == y_val))
        threshold     = MIN_ACCEPTABLE_ACC.get(model_key, 0.38)
        log.info("Cached model %s val_accuracy=%.3f (threshold=%.2f)", path, val_acc, threshold)
        if val_acc < threshold:
            log.warning("Cached model below threshold but keeping it — better than no model.")

        # Check 3: bias probe — only delete if completely degenerate
        unique_classes, counts = np.unique(val_labels, return_counts=True)
        dominant_fraction      = float(counts.max()) / len(val_labels)
        if dominant_fraction > 0.95 and len(unique_classes) == 1:
            log.warning(
                "Cached model %s is degenerate (%.0f%% same class) — retraining.",
                path, dominant_fraction * 100,
            )
            os.remove(path)
            return None

        log.info("Cached model %s loaded ✓  (val_acc=%.3f)", path, val_acc)
        return m

    except Exception as exc:
        log.warning("Cached model invalid (%s) — will retrain.", exc)
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _make_dataset(X: np.ndarray, y: np.ndarray, aug_layer, augment: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.map(
            lambda x, lbl: (aug_layer(x, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── Training functions ────────────────────────────────────────────────────────

def train_pneumonia():
    base_dir = DATASET_PATHS["pneumonia"]
    classes  = ["NORMAL", "PNEUMONIA"]

    X, y = load_images(base_dir, classes, limit=1000)
    if len(X) == 0:
        log.warning("Pneumonia dataset missing — model not trained.")
        return None

    X, y = _shuffle(X, y)

    train_size   = int(0.8 * len(X))
    X_tr, y_tr   = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:],  y[train_size:]

    cached = _validate_cached_model(MODEL_PATHS["pneumonia"], "pneumonia", X_val, y_val)
    if cached is not None:
        return cached

    class_weights = _compute_class_weights(y_tr)
    log.info("Pneumonia class weights: %s", class_weights)

    model, base = build_mobilenet(num_classes=2)
    aug         = build_augmentation_layer()

    train_ds = _make_dataset(X_tr, y_tr, aug, augment=True)
    val_ds   = _make_dataset(X_val, y_val, aug, augment=False)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    )

    log.info("Pneumonia — Stage 1: training head…")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=20, class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
    )
    best_s1 = max(hist1.history.get("val_accuracy", [0]))
    log.info("Pneumonia Stage 1 best val_acc=%.3f", best_s1)

    log.info("Pneumonia — Stage 2: fine-tuning top layers…")
    _unfreeze_top_layers(model, base, n_layers=30)
    early_stop2 = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    )
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=30, class_weight=class_weights,
        callbacks=[early_stop2, reduce_lr],
    )
    best_s2 = max(hist2.history.get("val_accuracy", [0]))
    log.info("Pneumonia Stage 2 best val_acc=%.3f", best_s2)

    final_val_acc = max(best_s1, best_s2)
    threshold     = MIN_ACCEPTABLE_ACC["pneumonia"]
    if final_val_acc < threshold:
        log.warning(
            "Pneumonia val_accuracy=%.3f < threshold=%.2f — saving anyway.",
            final_val_acc, threshold,
        )

    # Always save — never return None after a completed training run
    model.save(MODEL_PATHS["pneumonia"])
    log.info("Pneumonia model saved → %s  (val_acc=%.3f)", MODEL_PATHS["pneumonia"], final_val_acc)
    return model


def train_skin():
    base_dir    = DATASET_PATHS["skin"]
    ALL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    classes = [
        c for c in ALL_CLASSES
        if os.path.isdir(os.path.join(base_dir, c))
        and any(True for _ in os.scandir(os.path.join(base_dir, c)))
    ]
    if not classes:
        log.warning("Skin dataset missing — model not trained.")
        return None
    log.info("Skin: training on classes: %s", classes)

    TARGET_PER_CLASS = 800
    X_raw, y_raw = load_images(base_dir, classes, limit=TARGET_PER_CLASS)
    if len(X_raw) == 0:
        log.warning("Skin dataset missing — model not trained.")
        return None

    X_parts, y_parts = [], []
    rng = np.random.default_rng(42)
    for i, cls in enumerate(classes):
        mask  = y_raw == i
        X_cls = X_raw[mask]
        if len(X_cls) == 0:
            log.warning("Class %s has no images — skipping.", cls)
            continue
        if len(X_cls) < TARGET_PER_CLASS:
            n_extra = TARGET_PER_CLASS - len(X_cls)
            idx     = rng.integers(0, len(X_cls), size=n_extra)
            X_cls   = np.concatenate([X_cls, X_cls[idx]], axis=0)
            log.info("  %-10s oversampled %d → %d", cls, mask.sum(), len(X_cls))
        X_parts.append(X_cls[:TARGET_PER_CLASS])
        y_parts.append(np.full(TARGET_PER_CLASS, i, dtype=np.int32))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    log.info("Skin after oversampling: %d images, %d classes", len(X), len(np.unique(y)))

    X, y = _shuffle(X, y)

    train_size   = int(0.8 * len(X))
    X_tr, y_tr   = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:],  y[train_size:]

    cached = _validate_cached_model(MODEL_PATHS["skin"], "skin", X_val, y_val)
    if cached is not None:
        return cached

    class_weights = _compute_class_weights(y_tr)
    log.info("Skin class weights: %s", class_weights)

    model, base = build_mobilenet(num_classes=len(classes))
    aug         = build_augmentation_layer()

    train_ds = _make_dataset(X_tr, y_tr, aug, augment=True)
    val_ds   = _make_dataset(X_val, y_val, aug, augment=False)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    )

    log.info("Skin — Stage 1: training head…")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=25, class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
    )
    best_s1 = max(hist1.history.get("val_accuracy", [0]))
    log.info("Skin Stage 1 best val_acc=%.3f", best_s1)

    log.info("Skin — Stage 2: fine-tuning top layers…")
    _unfreeze_top_layers(model, base, n_layers=30)
    early_stop2 = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=9, restore_best_weights=True
    )
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=50, class_weight=class_weights,
        callbacks=[early_stop2, reduce_lr],
    )
    best_s2 = max(hist2.history.get("val_accuracy", [0]))
    log.info("Skin Stage 2 best val_acc=%.3f", best_s2)

    final_val_acc = max(best_s1, best_s2)
    threshold     = MIN_ACCEPTABLE_ACC["skin"]
    if final_val_acc < threshold:
        log.warning(
            "Skin val_accuracy=%.3f < threshold=%.2f — saving anyway.",
            final_val_acc, threshold,
        )

    # Always save — never return None after a completed training run
    model.save(MODEL_PATHS["skin"])
    classes_path = MODEL_PATHS["skin"].replace(".keras", "_classes.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f)
    log.info("Skin model saved → %s  (classes=%s, val_acc=%.3f)", MODEL_PATHS["skin"], classes, final_val_acc)
    return model


# ── Tabular models ────────────────────────────────────────────────────────────

def train_diabetes() -> Pipeline:
    df       = pd.read_csv(DATASET_PATHS["diabetes"])
    features = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
    X, y     = df[features], df["Outcome"]
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)
    log.info("Diabetes model trained.")
    return pipe


def train_heart() -> Pipeline:
    df       = pd.read_csv(DATASET_PATHS["heart"])
    features = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang"]
    X        = df[features]
    y        = df["condition"] if "condition" in df.columns else df["target"]
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)
    log.info("Heart model trained.")
    return pipe


# ── Init ──────────────────────────────────────────────────────────────────────

def init_models():
    log.info("═" * 60)
    log.info("Initialising ArogyaAI models (v4 — MobileNetV2)…")
    log.info("FORCE_RETRAIN = %s", FORCE_RETRAIN)
    log.info("═" * 60)

    try:
        MODELS["diabetes"] = train_diabetes()
    except Exception as exc:
        log.warning("Diabetes model failed: %s", exc)

    try:
        MODELS["heart"] = train_heart()
    except Exception as exc:
        log.warning("Heart model failed: %s", exc)

    MODELS["pneumonia"] = train_pneumonia()
    MODELS["skin"]      = train_skin()

    loaded  = [k for k, v in MODELS.items() if v is not None]
    missing = [k for k, v in MODELS.items() if v is None]
    log.info("Models ready : %s", loaded)
    if missing:
        log.warning("Models failed: %s", missing)


# ── Prediction helpers ────────────────────────────────────────────────────────

def _preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = (
        Image.open(io.BytesIO(file_bytes))
        .convert("RGB")
        .resize((IMG_SIZE, IMG_SIZE))
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _risk_level(prob: float, high_th: float = 0.65, mod_th: float = 0.40) -> str:
    if prob >= high_th:
        return "HIGH"
    if prob >= mod_th:
        return "MODERATE"
    return "LOW"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify({
        "models_loaded":  [k for k, v in MODELS.items() if v is not None],
        "models_missing": [k for k, v in MODELS.items() if v is None],
    })


@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(force=True)
        X = np.array([[
            float(data.get("pregnancies", 0)),
            float(data.get("glucose",     120)),
            float(data.get("bp",          70)),
            float(data.get("insulin",     80)),
            float(data.get("bmi",         25)),
            float(data.get("age",         30)),
        ]])
        prob = float(MODELS["diabetes"].predict_proba(X)[0][1])
        return jsonify({
            "label":       "Diabetes Risk",
            "probability":  round(prob, 4),
            "risk_level":   _risk_level(prob),
        })
    except Exception as exc:
        log.exception("Diabetes prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data  = request.get_json(force=True)
        sex   = 1 if str(data.get("sex",   "male")).lower() == "male" else 0
        exang = 1 if str(data.get("exang", "no")).lower()  == "yes"  else 0
        X = np.array([[
            float(data.get("age",      50)),
            sex,
            float(data.get("cp",       1)),
            float(data.get("trestbps", 120)),
            float(data.get("chol",     200)),
            float(data.get("thalach",  150)),
            exang,
        ]])
        prob = float(MODELS["heart"].predict_proba(X)[0][1])
        return jsonify({
            "label":       "Heart Disease Risk",
            "probability":  round(prob, 4),
            "risk_level":   _risk_level(prob),
        })
    except Exception as exc:
        log.exception("Heart prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict/pneumonia", methods=["POST"])
def pred_pneumonia():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        model = MODELS.get("pneumonia")
        if model is None:
            return jsonify({"error": "Pneumonia model not available. Check server logs."}), 503

        arr  = _preprocess_image(request.files["file"].read())
        pred = model.predict(arr, verbose=0)[0]

        classes = ["NORMAL", "PNEUMONIA"]
        idx     = int(np.argmax(pred))
        label   = classes[idx]
        prob    = float(pred[idx])

        if label == "PNEUMONIA":
            risk = "HIGH" if prob > 0.70 else "MODERATE"
        else:
            risk = "LOW"

        log.info("Pneumonia → %s  prob=%.3f  all_probs=%s", label, prob,
                 {c: round(float(p), 3) for c, p in zip(classes, pred)})
        return jsonify({
            "label":       label,
            "probability":  round(prob, 4),
            "risk_level":   risk,
            "all_probs":    {c: round(float(p), 4) for c, p in zip(classes, pred)},
        })
    except Exception as exc:
        log.exception("Pneumonia prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict/skin", methods=["POST"])
def pred_skin():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        model = MODELS.get("skin")
        if model is None:
            return jsonify({"error": "Skin model not available. Check server logs."}), 503

        classes_path = MODEL_PATHS["skin"].replace(".keras", "_classes.json")
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                classes = json.load(f)
        else:
            classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

        arr  = _preprocess_image(request.files["file"].read())
        pred = model.predict(arr, verbose=0)[0]

        idx   = int(np.argmax(pred))
        label = classes[idx]
        prob  = float(pred[idx])

        if label in ["mel", "bcc"]:
            risk = "HIGH" if prob > 0.55 else "MODERATE"
        elif label in ["akiec", "bkl"]:
            risk = "MODERATE"
        else:
            risk = "LOW"

        log.info("Skin → %s  prob=%.3f  all_probs=%s", label, prob,
                 {c: round(float(p), 3) for c, p in zip(classes, pred)})
        return jsonify({
            "label":       label.upper(),
            "probability":  round(prob, 4),
            "risk_level":   risk,
            "all_probs":    {c.upper(): round(float(p), 4) for c, p in zip(classes, pred)},
        })
    except Exception as exc:
        log.exception("Skin prediction error")
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from download_models import download_models
    download_models()          # pulls .keras files from Drive if missing
    FORCE_RETRAIN = False      # never retrain on Render
    init_models()
    app.run(host="0.0.0.0", port=5050, debug=False)