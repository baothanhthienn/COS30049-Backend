import numpy as np
from api import model_loader
from api.ml.taxonomy import family_name, all_families

_SPAN_FAMILY_PRIORITY: list[tuple[str, str]] = [
    ("instruction_override", "instruction_override"),
    ("role_play_jailbreak",  "role_play_jailbreak"),
    ("narrative_frame",      "indirect_injection"),   
    ("role_swap",            "role_play_jailbreak"), 
    ("data_exfiltration",    "context_exfiltration"),
    ("filter_bypass",        "delimiter_escape"),    
]

_ALL_FRONTEND_FAMILIES = [
    "instruction_override",
    "role_play_jailbreak",
    "context_exfiltration",
    "delimiter_escape",
    "encoding_obfuscation",
    "indirect_injection",
]


def _family_from_spans(spans: list[dict]) -> str | None:
    span_labels = {s["label"] for s in spans}
    for span_label, family in _SPAN_FAMILY_PRIORITY:
        if span_label in span_labels:
            return family
    return None


def predict_family(
    features_list: list[float],
    spans: list[dict] | None = None,
) -> tuple[str, dict[str, float]]:
    model_loader.ensure_loaded()
    bundle = model_loader._kmeans_bundle

    if bundle is not None:
        X = np.array([features_list], dtype=np.float32)
        X_scaled = bundle["scaler"].transform(X)
        kmeans = bundle["kmeans"]
        distances = kmeans.transform(X_scaled)[0]
        neg = -distances
        exp = np.exp(neg - neg.max())
        km_scores = exp / exp.sum()
        all_fam = all_families()
        family_scores = {
            all_fam.get(i, f"cluster_{i}"): round(float(s), 4)
            for i, s in enumerate(km_scores)
        }
        km_family = family_name(int(kmeans.predict(X_scaled)[0]))
    else:
        family_scores = {f: 0.0 for f in _ALL_FRONTEND_FAMILIES}
        km_family = "indirect_injection"

    f10_idx = 9  
    if features_list[f10_idx] > 0.3:
        return "encoding_obfuscation", family_scores

    if spans:
        span_family = _family_from_spans(spans)
        if span_family:
            return span_family, family_scores

    return km_family, family_scores
