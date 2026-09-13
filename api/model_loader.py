"""
Singleton model loader — loads RF, LR, KMeans, and scaler once on startup.
All prediction functions
"""

import json
import os
import pickle
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, _SRC)

from feature_extraction import extract_features_from_text  # noqa: E402
from preprocessing import preprocess  # noqa: E402

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_rf_model      = None
_kmeans_bundle = None
_cluster_labels: dict = {}

VERDICT_THRESHOLD = 0.50   # P(injection) >= this → BLOCK


def _load_models():
    global _rf_model, _kmeans_bundle, _cluster_labels

    rf_path = os.path.join(_MODELS_DIR, 'rf_model.pkl')
    km_path = os.path.join(_MODELS_DIR, 'kmeans_model.pkl')
    cl_path = os.path.join(_MODELS_DIR, 'cluster_labels.json')

    if not os.path.exists(rf_path):
        raise RuntimeError(
            f"RF model not found at {rf_path}. Run `python3 src/train.py` first."
        )

    with open(rf_path, 'rb') as f:
        _rf_model = pickle.load(f)

    if os.path.exists(km_path):
        with open(km_path, 'rb') as f:
            _kmeans_bundle = pickle.load(f)

    if os.path.exists(cl_path):
        with open(cl_path) as f:
            _cluster_labels = json.load(f)


def ensure_loaded():
    if _rf_model is None:
        _load_models()


def predict(text: str) -> dict:
    """
    Full prediction pipeline for a single text.
    Returns a dict matching PredictResponse schema.
    """
    ensure_loaded()

    features, spans = extract_features_from_text(text)
    X = np.array([features.to_list()], dtype=np.float32)

    prob_injection = float(_rf_model.predict_proba(X)[0, 1])
    label = 1 if prob_injection >= VERDICT_THRESHOLD else 0
    verdict = "BLOCK" if label == 1 else "ALLOW"

    cluster_id    = None
    cluster_label = None

    if label == 1 and _kmeans_bundle is not None:
        X_scaled   = _kmeans_bundle['scaler'].transform(X)
        cluster_id = int(_kmeans_bundle['kmeans'].predict(X_scaled)[0])
        cluster_label = _cluster_labels.get(str(cluster_id), {}).get('label', 'unknown')

    decoded = preprocess(text).decoded_text

    return {
        "verdict":       verdict,
        "label":         label,
        "confidence":    round(prob_injection, 4),
        "cluster_id":    cluster_id,
        "cluster_label": cluster_label,
        "spans":         [{"start": s.start, "end": s.end, "label": s.label} for s in spans],
        "decoded_text":  decoded,
    }
