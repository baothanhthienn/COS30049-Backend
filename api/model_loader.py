import hashlib
import json
import os
import pickle
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, _SRC)

from feature_extraction import extract_features_from_text, FeatureVector, _RE_ROLE_SWAP_ACTAS, _RE_HARMFUL_ROLE  # noqa: E402
from preprocessing import preprocess  

def _get_normalise_flag() -> bool:
    from api.core.config import settings  
    return settings.normalise

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_best_model    = None
_kmeans_bundle = None
_cluster_labels: dict = {}

VERDICT_THRESHOLD = 0.60  
_NARRATIVE_FRAME_FEATURE_INDEX = FeatureVector.feature_names().index('f21_narrative_frame_count')
_NARRATIVE_FRAME_BOOST_MIN_COUNT = 2.0   
_NARRATIVE_FRAME_BOOST_CEILING   = 0.49 
_NARRATIVE_FRAME_BOOST_PER_MATCH = 0.17                              
_NARRATIVE_FRAME_BOOST_MAX       = 0.40  

_METRICS_PATH = os.path.join(_MODELS_DIR, 'metrics.json')
_HASHES_PATH  = os.path.join(_MODELS_DIR, 'hashes.json')


def _verify_hash(path: str) -> None:
    if not os.path.exists(_HASHES_PATH):
        return  
    with open(_HASHES_PATH) as f:
        expected = json.load(f)
    key = os.path.basename(path)
    if key not in expected:
        return  
    with open(path, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected[key]:
        raise RuntimeError(
            f"Model file integrity check FAILED for {key}: "
            f"expected {expected[key][:16]}…, got {actual[:16]}…. "
            "Re-run training or restore the original file."
        )


def _resolve_best_model_path() -> str:
    if os.path.exists(_METRICS_PATH):
        with open(_METRICS_PATH) as f:
            best = json.load(f).get('best_model', 'rf')
    else:
        best = 'rf'
    return os.path.join(_MODELS_DIR, f'{best}_model.pkl')


def _load_models():
    global _best_model, _kmeans_bundle, _cluster_labels

    model_path = _resolve_best_model_path()
    km_path = os.path.join(_MODELS_DIR, 'kmeans_model.pkl')
    cl_path = os.path.join(_MODELS_DIR, 'cluster_labels.json')

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Best model not found at {model_path}. Run `python3 src/train.py` first."
        )

    _verify_hash(model_path)
    with open(model_path, 'rb') as f:
        _best_model = pickle.load(f)

    if os.path.exists(km_path):
        _verify_hash(km_path)
        with open(km_path, 'rb') as f:
            _kmeans_bundle = pickle.load(f)

    if os.path.exists(cl_path):
        with open(cl_path) as f:
            _cluster_labels = json.load(f)


def get_best_model_name() -> str:
    if os.path.exists(_METRICS_PATH):
        with open(_METRICS_PATH) as f:
            return json.load(f).get('best_model', 'rf').upper()
    return 'RF'


def ensure_loaded():
    if _best_model is None:
        _load_models()


def _apply_narrative_frame_boost(prob_injection: float, features: FeatureVector) -> tuple[float, bool]:
    f21 = features.f21_narrative_frame_count
    if f21 >= _NARRATIVE_FRAME_BOOST_MIN_COUNT and prob_injection < _NARRATIVE_FRAME_BOOST_CEILING:
        amount = min(f21 * _NARRATIVE_FRAME_BOOST_PER_MATCH, _NARRATIVE_FRAME_BOOST_MAX)
        boosted = min(prob_injection + amount, 1.0)
        return boosted, True
    return prob_injection, False


def _apply_keyword_hard_boost(prob: float, features: FeatureVector, decoded_text: str) -> tuple[float, bool]:
    qualified_role_swap = float(len(_RE_ROLE_SWAP_ACTAS.findall(decoded_text)))
    harmful_role = float(len(_RE_HARMFUL_ROLE.findall(decoded_text)))
    keyword_hits = (
        features.f01_override_keyword_count
        + qualified_role_swap
        + harmful_role
        + features.f04_filter_bypass_count
    )
    if keyword_hits >= 1.0 and prob < 0.90:
        return max(prob, 0.90), True
    return prob, False


def predict(text: str) -> dict:
    ensure_loaded()

    normalise = _get_normalise_flag()
    features, spans = extract_features_from_text(text, normalise=normalise)
    X = np.array([features.to_list()], dtype=np.float32)

    prob_injection = float(_best_model.predict_proba(X)[0, 1])
    decoded = preprocess(text, normalise=normalise).decoded_text
    prob_injection, boosted = _apply_narrative_frame_boost(prob_injection, features)
    if not boosted:
        prob_injection, boosted = _apply_keyword_hard_boost(prob_injection, features, decoded)

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
        "rule_boost_applied": boosted,
    }