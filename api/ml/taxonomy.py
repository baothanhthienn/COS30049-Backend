import json
import os

_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "cluster_labels.json")

_ATTACK_FAMILIES: dict[int, str] = {}


def _load():
    global _ATTACK_FAMILIES
    if _ATTACK_FAMILIES:
        return
    with open(_LABELS_PATH) as f:
        raw = json.load(f)
    _ATTACK_FAMILIES = {int(k): v["label"] for k, v in raw.items()}


def family_name(cluster_id: int) -> str:
    _load()
    return _ATTACK_FAMILIES.get(cluster_id, "unknown")


def all_families() -> dict[int, str]:
    _load()
    return dict(_ATTACK_FAMILIES)
