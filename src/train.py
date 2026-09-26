"""
Model training pipeline — trains RF, LR, XGBoost, SVM on combined dataset.

Steps:
  1. Load data/processed/train.csv and test.csv (built by src/data_loader.py)
  2. Extract 20 features via extract_features_batch()
  3. Train Logistic Regression baseline (unit-taught, scaled)
  4. Train Random Forest (unit-taught, primary baseline)
  5. Train XGBoost (beyond unit — gradient boosted trees, native imbalance handling)
  6. Train SVM with RBF kernel (beyond unit — effective on small dense feature spaces)
  7. Evaluate all on identical held-out test set; pick best by ROC-AUC
  8. Save all four models + scaler + metrics.json

Run: python3 src/train.py
"""

import os
import sys
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_batch, FeatureVector


DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    print(f"Train: {len(train)} rows  |  Test: {len(test)} rows")
    print(f"Train label dist:\n{train['label'].value_counts()}")
    return train, test


def build_feature_matrix(texts):
    vectors = extract_features_batch(texts)
    return np.array([v.to_list() for v in vectors], dtype=np.float32)


def evaluate(name, model, X_test, y_test, scaler=None):
    """Evaluate a model; return full metrics dict."""
    X = scaler.transform(X_test) if scaler else X_test
    t0 = time.time()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    infer_ms = (time.time() - t0) * 1000

    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['benign', 'injection']))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}")

    return {
        'roc_auc':   roc,
        'pr_auc':    pr,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'infer_ms':  round(infer_ms, 1),
    }


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df, test_df = load_data()

    print("\nExtracting features from train set...")
    X_train = build_feature_matrix(train_df['text'].tolist())
    y_train = train_df['label'].values

    print("Extracting features from test set...")
    X_test = build_feature_matrix(test_df['text'].tolist())
    y_test = test_df['label'].values

    feature_names = FeatureVector.feature_names()
    print(f"\nFeature matrix: train={X_train.shape}, test={X_test.shape}")

    # Scale once — LR and SVM need it; RF/XGBoost use raw
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    results = {}

    # Logistic Regression (unit-taught baseline) 
    # Baseline for comparison; linear decision boundary; needs scaling
    print("\nTraining Logistic Regression baseline...")
    t0 = time.time()
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    train_s = time.time() - t0
    results['lr'] = evaluate("Logistic Regression (baseline)", lr, X_test, y_test, scaler)
    results['lr']['train_s'] = round(train_s, 2)

    # Random Forest (unit-taught primary)
    # Ensemble of decorrelated trees; robust to scale; no normalisation needed
    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_s = time.time() - t0
    results['rf'] = evaluate("Random Forest", rf, X_test, y_test)
    results['rf']['train_s'] = round(train_s, 2)

    # XGBoost (beyond unit)
    # Gradient boosted trees; scale_pos_weight handles imbalance natively;
    # outperforms RF on structured tabular features via additive correction
    print("\nTraining XGBoost...")
    neg, pos = np.bincount(y_train)
    spw = neg / pos  # inverse class frequency for imbalance correction
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,  # compensate class imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    train_s = time.time() - t0
    results['xgb'] = evaluate("XGBoost", xgb, X_test, y_test)
    results['xgb']['train_s'] = round(train_s, 2)

    # SVM with RBF kernel (beyond unit)
    # Maximises margin in kernel-projected space; well-suited to 20-dim dense
    # features; RBF captures non-linear decision boundaries; needs scaling
    print("\nTraining SVM (RBF kernel)...")
    t0 = time.time()
    # CalibratedClassifierCV wraps SVC for predict_proba without deprecation
    _svm_base = SVC(kernel='rbf', class_weight='balanced', random_state=42, C=10.0, gamma='scale')
    svm = CalibratedClassifierCV(_svm_base, ensemble=False)
    svm.fit(X_train_scaled, y_train)
    train_s = time.time() - t0
    results['svm'] = evaluate("SVM (RBF kernel)", svm, X_test, y_test, scaler)
    results['svm']['train_s'] = round(train_s, 2)

    # Feature importance (RF)
    importances = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop-10 RF feature importances:")
    for fname, imp in importances[:10]:
        print(f"  {fname:40s}  {imp:.4f}")

    # Model comparison table
    print("\n" + "="*70)
    print(f"  {'Model':<20} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Train(s)':>9}")
    print("="*70)
    for key, r in results.items():
        print(f"  {key:<20} {r['roc_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['train_s']:>9.2f}")

    best_key = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\nBest model by ROC-AUC: {best_key} ({results[best_key]['roc_auc']:.4f})")

    # Save all models 
    model_map = {'rf': rf, 'lr': lr, 'xgb': xgb, 'svm': svm}
    for name, model in model_map.items():
        with open(os.path.join(MODELS_DIR, f'{name}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        **{k: v for k, v in results.items()},
        'best_model':   best_key,
        'feature_names': feature_names,
        'train_size':   int(len(train_df)),
        'test_size':    int(len(test_df)),
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll models saved to {MODELS_DIR}/")
    return model_map, scaler, results, best_key


if __name__ == '__main__':
    train()
