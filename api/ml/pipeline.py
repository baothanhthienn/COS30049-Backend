import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from feature_extraction import extract_features_from_text

from api.ml.stage_a import predict_binary
from api.ml.stage_b import predict_family
from api.core.config import settings

BLOCK_VERDICT = "BLOCK"


def run(text: str) -> dict:
    result = predict_binary(text)

    attack_family = None
    family_scores: dict[str, float] = {}

    if result["verdict"] == BLOCK_VERDICT:
        features, _ = extract_features_from_text(text)
        attack_family, family_scores = predict_family(
            features.to_list(),
            spans=result.get("spans"),
        )

    return {
        "verdict": result["verdict"],
        "label": result["label"],
        "confidence": result["confidence"],
        "attack_family": attack_family,
        "family_scores": family_scores,
        "cluster_id": result["cluster_id"],
        "cluster_label": result["cluster_label"],
        "spans": result["spans"],
        "decoded_text": result["decoded_text"],
        "rule_boost_applied": result["rule_boost_applied"],
        "model_version": settings.model_version,
    }
