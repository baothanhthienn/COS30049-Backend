import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch, FeatureVector


DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
ANALYSIS_OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'cluster_analysis.json')

TARGET_K = 6 


def find_optimal_k(X_scaled, k_range=(2, 9)):
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


_TARGET_FAMILIES = [
    "instruction_override",
    "role_play_jailbreak",
    "context_exfiltration",
    "delimiter_escape",
    "encoding_obfuscation",
    "indirect_injection",
]

_FAMILY_FEATURE = {
    "instruction_override": 'f01_override_keyword_count',
    "role_play_jailbreak":  'f02_role_swap_keyword_count',
    "context_exfiltration": 'f03_data_exfil_keyword_count',
    "delimiter_escape":     'f04_filter_bypass_count',
    "encoding_obfuscation": 'f10_encoding_anomaly_score',
    "indirect_injection":   None,  
}


def assign_cluster_names(centroids: np.ndarray, feature_names: list[str]) -> list[str]:
    feat_map = {n: i for i, n in enumerate(feature_names)}
    k = len(centroids)

    score_matrix = np.zeros((k, len(_TARGET_FAMILIES)))
    indirect_fi = _TARGET_FAMILIES.index("indirect_injection")
    for ci, c in enumerate(centroids):
        for fi, fam in enumerate(_TARGET_FAMILIES):
            feat = _FAMILY_FEATURE[fam]
            if feat is None:
                score_matrix[ci, fi] = 0.001
            else:
                score_matrix[ci, fi] = max(float(c[feat_map[feat]]), 0.0)

    names = [""] * k
    assigned_clusters: set[int] = set()
    assigned_families: set[int] = set()
    order = np.argsort(-score_matrix.flatten())
    for idx in order:
        ci, fi = divmod(int(idx), len(_TARGET_FAMILIES))
        if ci in assigned_clusters or fi in assigned_families:
            continue
        names[ci] = _TARGET_FAMILIES[fi]
        assigned_clusters.add(ci)
        assigned_families.add(fi)
        if len(assigned_clusters) == k:
            break
    return names


def cluster():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), on_bad_lines='skip')
    injections = train_df[train_df['label'] == 1].copy()
    print(f"Injection samples for clustering: {len(injections)}")

    feature_names = FeatureVector.feature_names()

    print("Extracting features...")
    X = np.array([
        v.to_list() for v in extract_features_batch(injections['text'].tolist())
    ], dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = TARGET_K
    print(f"\nUsing k={k} (fixed to match 6 frontend families)")

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    cluster_names = assign_cluster_names(km.cluster_centers_, feature_names)

    cluster_info = {}
    for cid in range(k):
        mask = labels == cid
        count = int(mask.sum())
        name = cluster_names[cid]

        top_features = sorted(
            zip(feature_names, km.cluster_centers_[cid].tolist()),
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
    #Each cluster entry includes: label, count, feature mean deltas vs overal
    
    overall_mean = X.mean(axis=0)
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
        "role_play_jailbreak": (
            "Samples here attempt to replace the model's identity by assigning "
            "it a new persona (f02 role-swap keyword count is the dominant "
            "signal). Phrases like 'you are now DAN', 'act as an unrestricted AI', "
            "or criminal character roleplay ('pretend you are a robber') are "
            "characteristic. Also covers harmful roleplay assigning violent or "
            "criminal roles to the model."
        ),
        "context_exfiltration": (
            "This cluster captures credential and system-prompt theft attempts, "
            "identified by elevated data exfiltration keyword counts (f03). "
            "Requests for API keys, environment variables, or verbatim system "
            "prompt output are typical. These are high-severity injections because "
            "a successful bypass yields directly exploitable information."
        ),
        "delimiter_escape": (
            "Attacks in this cluster exploit formatting tokens, special characters, "
            "or filter-bypass keywords to break out of the input context, showing "
            "high filter-bypass keyword counts (f04). Phrases such as 'no "
            "restrictions', 'bypass all filters', '###SYSTEM###', and 'jailbreak "
            "mode' define the cluster."
        ),
        "encoding_obfuscation": (
            "This cluster is defined by encoding anomaly signals (f10 composite "
            "score, f05 base64 detection, f06 unicode lookalikes, f07 zero-width "
            "characters). Attackers obfuscate injection payloads using base64 "
            "blobs, Cyrillic/Greek character substitution, or invisible Unicode "
            "to evade keyword matching."
        ),
        "indirect_injection": (
            "Samples here rely on narrative framing, hypothetical scenarios, or "
            "indirect language rather than explicit keywords, resulting in low "
            "scores across all keyword and encoding features. Structural signals "
            "(text length f19, sentence count f14, entropy f11) are the primary "
            "discriminators. Includes injections embedded inside documents, emails, "
            "or web content the model is asked to process."
        ),
    }

    clusters_out = {}
    for cid in range(k):
        mask   = labels == cid
        subset = X[mask]
        info   = cluster_info[cid]
        label  = info["label"]

        cluster_mean = subset.mean(axis=0)
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
        centroid_raw = cluster_mean  
        dists = np.linalg.norm(subset - centroid_raw, axis=1)
        nearest_idx  = np.argsort(dists)[:2]
        cluster_texts = [t for t, m in zip(texts, mask) if m]
        examples = [cluster_texts[i][:300] for i in nearest_idx]

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


if __name__ == '__main__':
    cluster()
