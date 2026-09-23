"""
Outputs:
  figures/01_class_distribution.png
  figures/02_text_length_distribution.png
  figures/03_feature_correlation_heatmap.png
  figures/04_confusion_matrix.png
  figures/05_roc_curves.png
  figures/06_feature_importance.png
  figures/07_cluster_scatter.png

Run: python3 src/visualize.py
"""

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch, FeatureVector

_ROOT       = os.path.join(os.path.dirname(__file__), '..')
_MODELS_DIR = os.path.join(_ROOT, 'models')
_DATA_DIR   = os.path.join(_ROOT, 'data', 'processed')
_FIG_DIR    = os.path.join(_ROOT, 'figures')
_METRICS    = os.path.join(_MODELS_DIR, 'metrics.json')

PALETTE = {'benign': '#4C9BE8', 'injection': '#E8634C'}
sns.set_theme(style='whitegrid', font_scale=1.1)


def _savefig(name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def _load_model(name: str):
    with open(os.path.join(_MODELS_DIR, f'{name}_model.pkl'), 'rb') as f:
        return pickle.load(f)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(_DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(_DATA_DIR, 'test.csv'))
    return train, test


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    vecs = extract_features_batch(df['text'].astype(str).tolist())
    return np.array([v.to_list() for v in vecs], dtype=np.float32)

# 1. Class distribution
def fig_class_distribution(train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, title in zip(axes, [train, test], ['Train set', 'Test set']):
        counts = df['label'].value_counts().sort_index()
        colors = [PALETTE['benign'], PALETTE['injection']]
        bars = ax.bar(['Benign (0)', 'Injection (1)'], counts.values, color=colors, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    str(val), ha='center', va='bottom', fontsize=11)
        ax.set_title(title)
        ax.set_ylabel('Sample count')
        ax.set_ylim(0, counts.max() * 1.15)
    fig.suptitle('Class Distribution — Benign vs Injection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig('01_class_distribution.png')

# 2. Text length distribution by label
def fig_text_length_distribution(train: pd.DataFrame) -> None:
    df = train.copy()
    df['length'] = df['text'].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    for label, name, color in [(0, 'Benign', PALETTE['benign']),
                                (1, 'Injection', PALETTE['injection'])]:
        subset = df[df['label'] == label]['length']
        axes[0].hist(subset, bins=60, alpha=0.6, color=color, label=name)
    axes[0].set_xlabel('Character count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Text length histogram')
    axes[0].legend()
    axes[0].set_xlim(0, df['length'].quantile(0.99))

    # Box plot
    plot_df = df[df['length'] <= df['length'].quantile(0.99)]
    plot_df = plot_df.copy()
    plot_df['Class'] = plot_df['label'].map({0: 'Benign', 1: 'Injection'})
    sns.boxplot(data=plot_df, x='Class', y='length',
                palette={'Benign': PALETTE['benign'], 'Injection': PALETTE['injection']},
                ax=axes[1], width=0.4)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Character count')
    axes[1].set_title('Text length box plot (99th pct cap)')

    fig.suptitle('Text Length Distribution by Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig('02_text_length_distribution.png')

# 3. Feature correlation heatmap (20 × 20)
def fig_feature_correlation(train: pd.DataFrame, feature_names: list) -> None:
    print("  Extracting features for heatmap (this may take ~30s)…")
    X = _extract_features(train)
    df_feat = pd.DataFrame(X, columns=feature_names)
    # Shorten names for readability
    short = [n.split('_', 1)[1].replace('_', ' ') for n in feature_names]
    df_feat.columns = short

    corr = df_feat.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', linewidths=0.4,
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                annot_kws={'size': 7}, ax=ax)
    ax.set_title('Feature Correlation Heatmap (20 features, lower triangle)',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    _savefig('03_feature_correlation_heatmap.png')
    return X  # reuse in later figures

# 4. Confusion matrix (best model on test set)
def fig_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray,
                          model_name: str) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Injection'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name.upper()} (test set)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig('04_confusion_matrix.png')

# 5. ROC curves — all 4 models on same axes
def fig_roc_curves(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
    colors = {'rf': '#2196F3', 'xgb': '#FF9800', 'svm': '#9C27B0', 'lr': '#4CAF50'}
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=colors.get(name, 'grey'), lw=2,
                label=f'{name.upper()}  (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    _savefig('05_roc_curves.png')

# 6. Feature importance (RF + XGBoost side by side)
def fig_feature_importance(rf_model, xgb_model, feature_names: list) -> None:
    short = [n.split('_', 1)[1].replace('_', ' ') for n in feature_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, model, title in [
        (axes[0], rf_model,  'Random Forest'),
        (axes[1], xgb_model, 'XGBoost'),
    ]:
        importances = model.feature_importances_
        order = np.argsort(importances)
        colors = ['#E8634C' if importances[i] >= np.percentile(importances, 75)
                  else '#4C9BE8' for i in order]
        ax.barh([short[i] for i in order], importances[order], color=colors)
        ax.set_xlabel('Importance score')
        ax.set_title(f'Feature Importance — {title}', fontweight='bold')
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle('Feature Importances (top quartile highlighted)', fontsize=13)
    plt.tight_layout()
    _savefig('06_feature_importance.png')


# 7. Cluster scatter (PCA 2-D, injection samples only)
def fig_cluster_scatter(train: pd.DataFrame, feature_names: list) -> None:
    cluster_path = os.path.join(_MODELS_DIR, 'kmeans_model.pkl')
    labels_path  = os.path.join(_MODELS_DIR, 'cluster_labels.json')

    if not os.path.exists(cluster_path):
        print("  Skipping cluster scatter — kmeans_model.pkl not found.")
        return

    with open(cluster_path, 'rb') as f:
        bundle = pickle.load(f)
    with open(labels_path) as f:
        cluster_labels = json.load(f)

    injections = train[train['label'] == 1].copy()
    print(f"  Extracting features for {len(injections)} injection samples…")
    X = _extract_features(injections)

    X_scaled = bundle['scaler'].transform(X)
    cluster_ids = bundle['kmeans'].predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_

    k = len(cluster_labels)
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(9, 7))

    for cid in range(k):
        mask   = cluster_ids == cid
        label  = cluster_labels[str(cid)]['label'].replace('_', ' ')
        count  = cluster_labels[str(cid)]['count']
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=cmap(cid / k), alpha=0.45, s=18,
                   label=f'C{cid}: {label} (n={count})')

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% variance)')
    ax.set_title('Injection Cluster Scatter (PCA 2-D)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, markerscale=1.5)
    plt.tight_layout()
    _savefig('07_cluster_scatter.png')

# Main
def main():
    os.makedirs(_FIG_DIR, exist_ok=True)

    with open(_METRICS) as f:
        metrics = json.load(f)
    feature_names = metrics.get('feature_names', FeatureVector.feature_names())
    best_model_name = metrics.get('best_model', 'rf')

    print("Loading data…")
    train, test = _load_data()

    print("Loading models…")
    models = {name: _load_model(name) for name in ('rf', 'xgb', 'svm', 'lr')}
    best_model = models[best_model_name]

    print("\n[1/7] Class distribution")
    fig_class_distribution(train, test)

    print("[2/7] Text length distribution")
    fig_text_length_distribution(train)

    print("[3/7] Feature correlation heatmap")
    X_train = fig_feature_correlation(train, feature_names)

    print("[4/7] Confusion matrix")
    print("  Extracting test features…")
    X_test  = _extract_features(test)
    y_test  = test['label'].values
    fig_confusion_matrix(best_model, X_test, y_test, best_model_name)

    print("[5/7] ROC curves")
    fig_roc_curves(models, X_test, y_test)

    print("[6/7] Feature importance")
    fig_feature_importance(models['rf'], models['xgb'], feature_names)

    print("[7/7] Cluster scatter")
    fig_cluster_scatter(train, feature_names)

    print(f"\nAll figures saved to {_FIG_DIR}/")


if __name__ == '__main__':
    main()