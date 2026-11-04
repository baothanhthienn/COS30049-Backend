import math

_SPAN_TO_FAMILY: dict[str, str] = {
    "instruction_override": "instruction_override",
    "role_swap":            "role_play_jailbreak",
    "narrative_frame":      "role_play_jailbreak",
    "data_exfiltration":    "context_exfiltration",
    "filter_bypass":        "delimiter_escape",
}


def _signal_scores(f: list[float]) -> dict[str, float]:
    return {
        "encoding_obfuscation": min(10.0,
            f[4] * 8.0             
            + f[9] * 6.0           
            + min(f[5], 5) * 1.5    
            + min(f[6], 5) * 1.5   
            + f[7] * 4.0            
            + f[8] * 3.0           
        ),
        "instruction_override": min(10.0,
            min(f[0], 4) * 2.5     
            + f[17] * 2.0          
            + min(f[3], 2) * 0.5    
        ),
        "role_play_jailbreak": min(10.0,
            min(f[1], 4) * 2.5     
            + min(f[20], 3) * 1.5   
            + f[17] * 0.5          
        ),
        "context_exfiltration": min(10.0,
            min(f[2], 4) * 2.5     
            + min(f[0], 2) * 0.5    
        ),
        "delimiter_escape": min(10.0,
            min(f[3], 4) * 2.5     
            + f[15] * 6.0           
            + min(f[16], 5) * 0.3   
        ),
        "indirect_injection": min(10.0,
            min(f[13], 10) * 0.4    
            + min(f[14], 20) * 0.2  
            + min(f[20], 2) * 1.0  
        ),
    }


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    m = max(vals)
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    s = sum(exps.values())
    return {k: round(v / s, 4) for k, v in exps.items()}


def _best_span_family(spans: list[dict]) -> str | None:
    span_labels = {s["label"] for s in spans}
    for span_label, family in _SPAN_TO_FAMILY.items():
        if span_label in span_labels:
            return family
    return None


def predict_family(
    features_list: list[float],
    spans: list[dict] | None = None,
) -> tuple[str, dict[str, float]]:
    f = features_list
    raw = _signal_scores(f)
    family_scores = _softmax(raw)

    if f[4] > 0 or f[9] > 0.2 or f[5] >= 2 or f[6] >= 1 or f[7] > 0.3:
        return "encoding_obfuscation", family_scores

    if spans:
        span_family = _best_span_family(spans)
        if span_family:
            return span_family, family_scores

    top_family = max(raw, key=lambda k: raw[k])
    if raw[top_family] >= 1.5:
        return top_family, family_scores

    if f[15] > 0.10:
        return "delimiter_escape", family_scores

    return "indirect_injection", family_scores
