"""
K-means clustering on the injection class.

Clusters the feature vectors of prompt injection samples to discover
distinct attack pattern families. Results are saved to:
  models/kmeans_model.pkl
  models/cluster_labels.json  — {cluster_id: label_name, counts, centroid_features}

Run: python3 src/clustering.py
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch, FeatureVector


DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Cluster labels based on dominant feature patterns
_CLUSTER_NAMES = [
    "instruction_override",
    "role_hijack",
    "data_exfiltration",
    "filter_bypass",
    "encoding_evasion",
    "social_engineering",
]


def find_optimal_k(X_scaled, k_range=(2, 9)):
    """Silhouette sweep to find best k."""
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.__getitem__)
    print("Silhouette scores:")
    for k, s in scores.items():
        marker = " <-- best" if k == best_k else ""
        print(f"  k={k}: {s:.4f}{marker}")
    return best_k


def name_cluster(centroid: np.ndarray, feature_names: list[str]) -> str:
    """Heuristic: pick cluster name from the most dominant keyword feature."""
    feat_map = {n: i for i, n in enumerate(feature_names)}
    scores = {
        "instruction_override": centroid[feat_map['f01_override_keyword_count']],
        "role_hijack":          centroid[feat_map['f02_role_swap_keyword_count']],
        "data_exfiltration":    centroid[feat_map['f03_data_exfil_keyword_count']],
        "filter_bypass":        centroid[feat_map['f04_filter_bypass_count']],
        "encoding_evasion":     centroid[feat_map['f10_encoding_anomaly_score']],
    }
    best = max(scores, key=scores.__getitem__)
    if scores[best] < 0.05:
        return "social_engineering"
    return best


def cluster():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    injections = train_df[train_df['label'] == 1].copy()
    print(f"Injection samples for clustering: {len(injections)}")

    feature_names = FeatureVector.feature_names()

    print("Extracting features...")
    X = np.array([
        v.to_list() for v in extract_features_batch(injections['text'].tolist())
    ], dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Silhouette sweep (k=2..8)
    print("\nFinding optimal k via silhouette score...")
    best_k = find_optimal_k(X_scaled, k_range=(2, 8))

    # Fix k=6 if best_k < 4 (we want at least 4 meaningful clusters for reporting)
    k = max(best_k, 4)
    print(f"\nUsing k={k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    # Per-cluster summary
    cluster_info = {}
    cluster_names_used: set[str] = set()
    for cid in range(k):
        mask = labels == cid
        count = int(mask.sum())
        centroid = km.cluster_centers_[cid]

        name = name_cluster(centroid, feature_names)
        # deduplicate names by appending cluster id suffix
        if name in cluster_names_used:
            name = f"{name}_{cid}"
        cluster_names_used.add(name)

        top_features = sorted(
            zip(feature_names, centroid.tolist()),
            key=lambda x: abs(x[1]), reverse=True,
        )[:5]

        cluster_info[cid] = {
            "label": name,
            "count": count,
            "top_features": top_features,
        }
        print(f"  Cluster {cid} ({name}): {count} samples")
        for fn, fv in top_features[:3]:
            print(f"    {fn}: {fv:.3f}")

    # Save
    with open(os.path.join(MODELS_DIR, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump({'kmeans': km, 'scaler': scaler}, f)

    # JSON-serialisable cluster info
    serialisable = {
        str(cid): {
            "label": info["label"],
            "count": info["count"],
            "top_features": [[fn, round(fv, 4)] for fn, fv in info["top_features"]],
        }
        for cid, info in cluster_info.items()
    }
    with open(os.path.join(MODELS_DIR, 'cluster_labels.json'), 'w') as f:
        json.dump(serialisable, f, indent=2)

    sil = silhouette_score(X_scaled, labels)
    print(f"\nFinal silhouette score (k={k}): {sil:.4f}")
    print(f"Cluster model saved to {MODELS_DIR}/kmeans_model.pkl")

    return km, cluster_info


def predict_cluster(text: str) -> dict:
    """
    Predict which injection cluster a text belongs to.
    Returns {'cluster_id': int, 'label': str}.
    Used by the FastAPI /predict endpoint.
    """
    model_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    labels_path = os.path.join(MODELS_DIR, 'cluster_labels.json')

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    with open(labels_path) as f:
        cluster_labels = json.load(f)

    from feature_extraction import extract_features_from_text
    features, _ = extract_features_from_text(text)
    X = np.array([features.to_list()], dtype=np.float32)
    X_scaled = bundle['scaler'].transform(X)
    cid = int(bundle['kmeans'].predict(X_scaled)[0])
    label = cluster_labels[str(cid)]['label']
    return {'cluster_id': cid, 'label': label}


if __name__ == '__main__':
    cluster()
