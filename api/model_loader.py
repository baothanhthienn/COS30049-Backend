"""
Singleton model loader — loads RF, LR, KMeans, and scaler once on startup.
Includes deterministic regex pre-filter, single-word pass heuristic, and ML fallback.
"""

import json
import os
import pickle
import re
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

# Broadened high-risk deterministic regex patterns
HIGH_RISK_PATTERNS = [
    r"(?i)\b(transfer|send|wire|move)\b.*?\b(money|funds|balance|cash)\b",
    r"(?i)\baccess\b.*?\b(account|credentials|database)\b",
    r"(?i)\b(override|ignore|bypass)\b.*?\b(instructions|system|rules|prompt)\b",
    r"(?i)\b(exfiltrate|leak|dump)\b",
    r"(?i)\bsystem\s+prompt\b",
    r"(?i)\b(fuck|shit|bitch|asshole)\b"
]


def check_regex_prefilter(text: str) -> bool:
    """Check text against high-risk regex triggers."""
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


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
    Full prediction pipeline for a single text:
    1. Single-word pass heuristic
    2. Deterministic regex pre-filter
    3. Random Forest + KMeans attack clustering fallback
    """
    ensure_loaded()
    cleaned = text.strip()
    decoded = preprocess(cleaned).decoded_text

    # 1. Single-word pass heuristic (prevents false positives on isolated terms)
    tokens = cleaned.split()
    if len(tokens) == 1 and not check_regex_prefilter(cleaned):
        return {
            "verdict":       "ALLOW",
            "label":         0,
            "confidence":    0.0500,
            "cluster_id":    None,
            "cluster_label": None,
            "spans":         [],
            "decoded_text":  decoded,
        }

    # 2. Deterministic Regex Pre-filter (Immediate Block for High-Risk Actions)
    if check_regex_prefilter(cleaned):
        return {
            "verdict":       "BLOCK",
            "label":         1,
            "confidence":    0.9900,
            "cluster_id":    0,
            "cluster_label": "financial_or_override_attempt",
            "spans":         [{"start": 0, "end": len(cleaned), "label": "high_risk_prefilter"}],
            "decoded_text":  decoded,
        }

    # 3. Hybrid ML Inference (Random Forest + Feature Extraction)
    features, spans = extract_features_from_text(cleaned)
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

    return {
        "verdict":       verdict,
        "label":         label,
        "confidence":    round(prob_injection, 4),
        "cluster_id":    cluster_id,
        "cluster_label": cluster_label,
        "spans":         [{"start": s.start, "end": s.end, "label": s.label} for s in spans],
        "decoded_text":  decoded,
    }