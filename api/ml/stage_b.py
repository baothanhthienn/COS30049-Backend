
import numpy as np
from api import model_loader
from api.ml.taxonomy import family_name, all_families


def predict_family(features_list: list[float]) -> tuple[str, dict[str, float]]:
    model_loader.ensure_loaded()
    bundle = model_loader._kmeans_bundle
    if bundle is None:
        return "unknown", {}

    X = np.array([features_list], dtype=np.float32)
    X_scaled = bundle["scaler"].transform(X)
    kmeans = bundle["kmeans"]

    distances = kmeans.transform(X_scaled)[0]  
    neg = -distances
    exp = np.exp(neg - neg.max())
    scores = exp / exp.sum()

    families = all_families()
    family_scores = {families.get(i, f"cluster_{i}"): round(float(s), 4) for i, s in enumerate(scores)}
    cluster_id = int(kmeans.predict(X_scaled)[0])
    return family_name(cluster_id), family_scores
