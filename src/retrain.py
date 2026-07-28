"""
Fine-tunes the already-saved model on newly uploaded images. This is not
training from scratch - it loads models/engagement_model.keras as the
starting point and continues from there, at a low learning rate, on the
new uploads plus a resampled slice of the original train set (otherwise a
handful of new images at a normal learning rate just overwrites what the
model already knew).

Evaluated against the same sealed test set used in the notebook so the
before/after macro F1 numbers are comparable.
"""
import datetime
import os
import random
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src import database
from src.preprocessing import CLASS_NAMES, build_pipeline, list_files_labels, make_dataset_from_files

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "engagement_model.keras")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

RESAMPLE_PER_RETRAIN = 500
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-5


def _macro_f1_on_test(model):
    test_ds = build_pipeline(TEST_DIR, batch_size=64, shuffle=False)
    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(probs.argmax(axis=1))
        y_true.extend(y.numpy())
    return f1_score(y_true, y_pred, average="macro")


def _collect_new_upload_files():
    filepaths, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(UPLOAD_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, fname))
            labels.append(class_idx)
    return filepaths, labels


def run_retrain():
    unused = database.get_unused_uploads()
    new_files = [os.path.join(UPLOAD_DIR, row["label"], row["filename"]) for row in unused]
    new_files = [f for f in new_files if os.path.isfile(f)]
    new_labels = [CLASS_NAMES.index(row["label"]) for row in unused if os.path.isfile(
        os.path.join(UPLOAD_DIR, row["label"], row["filename"]))]

    if not new_files:
        return {"status": "no_new_data"}

    model = tf.keras.models.load_model(MODEL_PATH)
    old_f1 = _macro_f1_on_test(model)

    train_files, train_labels = list_files_labels(TRAIN_DIR)
    sample_idx = random.sample(range(len(train_files)), min(RESAMPLE_PER_RETRAIN, len(train_files)))
    combined_files = [train_files[i] for i in sample_idx] + new_files
    combined_labels = [train_labels[i] for i in sample_idx] + new_labels

    fine_tune_ds = make_dataset_from_files(combined_files, combined_labels, batch_size=32, shuffle=True)

    model.optimizer.learning_rate.assign(FINE_TUNE_LR)
    model.fit(fine_tune_ds, epochs=FINE_TUNE_EPOCHS, verbose=0)

    new_f1 = _macro_f1_on_test(model)

    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(BASE_DIR, "models", f"engagement_model_{version}.keras")
    model.save(versioned_path)
    shutil.copyfile(versioned_path, MODEL_PATH)

    for f, label_idx in zip(new_files, new_labels):
        dest_dir = os.path.join(TRAIN_DIR, CLASS_NAMES[label_idx])
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(f, os.path.join(dest_dir, os.path.basename(f)))

    database.mark_uploads_used([row["id"] for row in unused])
    database.log_retrain_run(len(new_files), old_f1, new_f1, version)

    return {
        "status": "ok",
        "n_new_images": len(new_files),
        "old_macro_f1": old_f1,
        "new_macro_f1": new_f1,
        "model_version": version,
    }
