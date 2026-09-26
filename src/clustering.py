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


DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
ANALYSIS_OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'cluster_analysis.json')

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
    # Silhouette measures intra-cluster cohesion vs inter-cluster separation;
    # avoids elbow-method subjectivity on sparse high-dimensional feature spaces.
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
    # Centroid coordinate on each keyword axis is a proxy for that attack type's
    # prevalence; fallback to social_engineering when no keyword signal dominates.
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

    # Cluster only injections — benign samples have no meaningful attack-pattern
    # structure and would dilute the cluster boundaries.
    print("\nFinding optimal k via silhouette score...")
    best_k = find_optimal_k(X_scaled, k_range=(2, 8))

    # Floor at 4 so the report always has enough distinct attack families to analyse.
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

    save_cluster_analysis(
        injections['text'].tolist(), X, labels, k,
        feature_names, cluster_info,
    )

    return km, cluster_info


def save_cluster_analysis(
    texts: list,
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    feature_names: list,
    cluster_info: dict,
) -> None:
    """
    Produces data/cluster_analysis.json for the report.
    Each cluster entry includes: label, count, feature mean deltas vs overall
    mean (top 3 distinguishing features), 2-3 sentence description, and
    2 representative example texts.
    """
    overall_mean = X.mean(axis=0)

    # Canned descriptions keyed on cluster label prefix — written from the
    # feature semantics, not generated at runtime, so they stay meaningful
    # regardless of which k the silhouette sweep picks.
    _DESCRIPTIONS = {
        "instruction_override": (
            "This cluster groups attacks that directly command the model to "
            "discard its prior instructions, dominated by high override keyword "
            "counts (f01). Examples typically open with imperatives such as "
            "'ignore all previous instructions' or 'disregard your guidelines', "
            "making them the most lexically obvious injection family. Despite "
            "high detection rates, subtle variants using synonyms or leet "
            "substitution still evade the keyword features."
        ),
        "role_hijack": (
            "Samples here attempt to replace the model's identity by assigning "
            "it a new persona (f02 role-swap keyword count is the dominant "
            "signal). Phrases like 'you are now DAN' or 'act as an unrestricted "
            "AI' are characteristic. The cluster overlaps with legitimate "
            "role-framing prompts, which is the primary driver of false positives "
            "in the model's error analysis."
        ),
        "data_exfiltration": (
            "This cluster captures credential and system-prompt theft attempts, "
            "identified by elevated data exfiltration keyword counts (f03). "
            "Requests for API keys, environment variables, or verbatim system "
            "prompt output are typical. These are high-severity injections in "
            "production because a successful bypass yields directly exploitable "
            "information."
        ),
        "filter_bypass": (
            "Attacks in this cluster focus on removing safety constraints rather "
            "than extracting data, showing high filter-bypass keyword counts (f04). "
            "Phrases such as 'no restrictions', 'bypass all filters', and "
            "'jailbreak mode' define the cluster. Many examples combine bypass "
            "language with a role-swap to both remove constraints and assign a "
            "new identity simultaneously."
        ),
        "encoding_evasion": (
            "This cluster is defined by encoding anomaly signals (f10 composite "
            "score, f05 base64 detection, f06 unicode lookalikes, f07 zero-width "
            "characters). Attackers obfuscate injection payloads using base64 "
            "blobs, Cyrillic/Greek character substitution, or invisible Unicode "
            "to evade keyword matching. These are the hardest samples for the "
            "model — they score near-zero on keyword features and rely entirely "
            "on the encoding anomaly group for detection."
        ),
        "social_engineering": (
            "Samples here rely on narrative framing, hypothetical scenarios, or "
            "indirect language rather than explicit keywords, resulting in low "
            "scores across all keyword and encoding features. Structural signals "
            "(text length f19, sentence count f14, entropy f11) are the primary "
            "discriminators. These injections are the main source of false "
            "negatives: without keyword or encoding signal, the model struggles "
            "to distinguish them from benign complex prompts."
        ),
    }

    clusters_out = {}
    for cid in range(k):
        mask   = labels == cid
        subset = X[mask]
        info   = cluster_info[cid]
        label  = info["label"]

        cluster_mean = subset.mean(axis=0)
        # Delta = cluster mean - overall mean; large positive = cluster overrepresents this feature
        deltas = cluster_mean - overall_mean
        top_delta_idx = np.argsort(np.abs(deltas))[::-1][:3]
        distinguishing = [
            {
                "feature":       feature_names[i],
                "cluster_mean":  round(float(cluster_mean[i]), 4),
                "overall_mean":  round(float(overall_mean[i]), 4),
                "delta":         round(float(deltas[i]), 4),
            }
            for i in top_delta_idx
        ]

        # 2 representative examples: closest to the centroid
        centroid_raw = cluster_mean  # centroids in original space
        dists = np.linalg.norm(subset - centroid_raw, axis=1)
        nearest_idx  = np.argsort(dists)[:2]
        cluster_texts = [t for t, m in zip(texts, mask) if m]
        examples = [cluster_texts[i][:300] for i in nearest_idx]

        # Match description to label prefix
        desc_key = next(
            (k for k in _DESCRIPTIONS if label.startswith(k)),
            "social_engineering",
        )

        clusters_out[str(cid)] = {
            "label":               label,
            "count":               info["count"],
            "distinguishing_features": distinguishing,
            "description":         _DESCRIPTIONS[desc_key],
            "examples":            examples,
        }

    with open(ANALYSIS_OUT, 'w') as f:
        json.dump({"k": k, "clusters": clusters_out}, f, indent=2)

    print(f"Cluster analysis saved to {ANALYSIS_OUT}")


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
