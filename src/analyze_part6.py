import os
import pandas as pd
from scipy import stats
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), engine='python', on_bad_lines='skip')

    # 6.1 — class balance (just descriptive, no test needed)
    counts = train['label'].value_counts()
    print("Class counts:", counts.to_dict())
    print(f"Ratio benign:injection = {counts[0]/counts[1]:.2f}:1")

    # 6.1 — is injection text actually longer / differently distributed?
    # Mann-Whitney U rather than a t-test: text length is bimodal/skewed,
    # not normal, so a rank-based test is the right tool here.
    train['length'] = train['text'].astype(str).str.len()
    benign_len    = train.loc[train['label'] == 0, 'length']
    injection_len = train.loc[train['label'] == 1, 'length']
    u_stat, p_val = stats.mannwhitneyu(benign_len, injection_len, alternative='two-sided')
    print(f"\nText length — Mann-Whitney U: U={u_stat:.1f}, p={p_val:.4g}")
    print(f"Benign median length:    {benign_len.median():.0f} (IQR {benign_len.quantile(.25):.0f}-{benign_len.quantile(.75):.0f})")
    print(f"Injection median length: {injection_len.median():.0f} (IQR {injection_len.quantile(.25):.0f}-{injection_len.quantile(.75):.0f})")

    # 6.2 — which feature pairs actually correlate strongest, within vs across groups
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extraction import extract_features_batch, FeatureVector
    X = np.array([v.to_list() for v in extract_features_batch(train['text'].astype(str).tolist())])
    df_feat = pd.DataFrame(X, columns=FeatureVector.feature_names())
    corr = df_feat.corr()

    # top correlated pairs overall (excluding self-correlation)
    pairs = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
    pairs = pairs[pairs.index.map(lambda ij: ij[0] < ij[1])]  # drop mirrored duplicates
    top_pairs = pairs.abs().sort_values(ascending=False).head(10)
    print("\nTop 10 correlated feature pairs:")
    for (a, b), _ in top_pairs.items():
        print(f"  {a} / {b}: r={corr.loc[a,b]:.3f}")

if __name__ == '__main__':
    main()