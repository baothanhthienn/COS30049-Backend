import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from feature_extraction import extract_features_from_text


def extract_spans(text: str) -> list[dict]:
    _, spans = extract_features_from_text(text)
    return [{"start": s.start, "end": s.end, "label": s.label} for s in spans]
