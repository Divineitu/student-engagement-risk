"""Loads the saved model once and exposes a predict() function for a single
image. Used by the API, and importable directly from the notebook to sanity
check the saved model file."""
import json
import os

import tensorflow as tf

from src.preprocessing import preprocess_single

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "engagement_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

_model = None
_labels = None


def _load():
    global _model, _labels
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH) as f:
            _labels = json.load(f)["classes"]
    return _model, _labels


def predict(image):
    """image: a PIL.Image. returns (label, confidence, probs_dict)"""
    model, labels = _load()
    x = preprocess_single(image)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(probs.argmax())
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return labels[top_idx], float(probs[top_idx]), probs_dict


def reload_model():
    """called after retrain.py swaps the model file so the API picks up the new weights"""
    global _model, _labels
    _model = None
    _labels = None
    return _load()
