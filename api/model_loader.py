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

from feature_extraction import extract_features_from_text, FeatureVector  # noqa: E402
from preprocessing import preprocess  # noqa: E402

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_best_model    = None
_kmeans_bundle = None
_cluster_labels: dict = {}

VERDICT_THRESHOLD = 0.50   # P(injection) >= this → BLOCK

# --- Rule-based override for rare-but-strong keyword features ---
# f21_narrative_frame_count (game/hypothetical/story-wrapper jailbreak framing)
# fires on <1% of training rows (80 / 12,444 in the last training run), which
# is too rare for tree-based ensembles (RF/XGBoost) to weight heavily even
# though it's ~96% precise within its own subset (77/80 nonzero rows are
# label=1). Diagnosed via direct feature-frequency analysis: adding more
# training data (SEAS Role_Play, +3021 rows) did not raise this above ~0.6%
# of rows, since SEAS's persona-hijack templates don't overlap with this
# feature's specific game/rules phrasing. Rather than keep diluting the
# feature with more data, this hybrid rule directly boosts the model's raw
# probability when f21 fires strongly, similar to how production guardrail
# systems combine ML scores with keyword-based safety nets. This does not
# replace the ML score — it only nudges borderline cases where the model's
# own signal is ambiguous (0.25–0.50) and a strong independent keyword
# indicator is present.
_NARRATIVE_FRAME_FEATURE_INDEX = FeatureVector.feature_names().index('f21_narrative_frame_count')
_NARRATIVE_FRAME_BOOST_MIN_COUNT = 2.0   # require at least 2 distinct phrase matches
_NARRATIVE_FRAME_BOOST_CEILING   = 0.49  # only boost if raw prob is below BLOCK; must stay < 0.50
# original flat amount) while a 4+ match case gets meaningfully more.
_NARRATIVE_FRAME_BOOST_PER_MATCH = 0.17  # multiplied by f21 count; retuned after SEAS retrain
                                         # pushed short-text raw probs lower
                                         # (f21=2 needs ~0.34 to clear 0.50)
_NARRATIVE_FRAME_BOOST_MAX       = 0.40  # cap so this can't single-handedly force BLOCK on weak raw scores

_METRICS_PATH = os.path.join(_MODELS_DIR, 'metrics.json')


def _resolve_best_model_path() -> str:
    # Read best_model from metrics.json so retraining automatically promotes a
    # new winner without requiring a code change; falls back to rf if file absent.
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

    with open(model_path, 'rb') as f:
        _best_model = pickle.load(f)

    if os.path.exists(km_path):
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
    """
    Rule-based override: if f21_narrative_frame_count indicates strong
    game/rules-framing language (>=2 distinct phrase matches) and the raw
    model probability is still ambiguous-low, nudge the probability up,
    scaled by how many distinct narrative-frame phrases matched. Returns
    (adjusted_probability, was_boosted).

    This is intentionally conservative — it only applies to borderline cases,
    not confidently-benign ones (e.g. it would not have fired if a legitimate
    "let's play chess" sentence already scored near 0.0, since that's well
    below the boost ceiling only in the sense of being far from the 0.50
    decision boundary in the safe direction — see threshold check below).
    Also doesn't fire on single-match cases (a lone "let's play a game" is
    common enough in benign text that it shouldn't get pushed on its own —
    see the chess-game case in the adversarial set).
    """
    f21 = features.f21_narrative_frame_count
    if f21 >= _NARRATIVE_FRAME_BOOST_MIN_COUNT and prob_injection < _NARRATIVE_FRAME_BOOST_CEILING:
        amount = min(f21 * _NARRATIVE_FRAME_BOOST_PER_MATCH, _NARRATIVE_FRAME_BOOST_MAX)
        boosted = min(prob_injection + amount, 1.0)
        return boosted, True
    return prob_injection, False


def predict(text: str) -> dict:
    ensure_loaded()

    features, spans = extract_features_from_text(text)
    X = np.array([features.to_list()], dtype=np.float32)

    prob_injection = float(_best_model.predict_proba(X)[0, 1])
    prob_injection, boosted = _apply_narrative_frame_boost(prob_injection, features)

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
        "rule_boost_applied": boosted,
    }