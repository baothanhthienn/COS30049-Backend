"""
Model training pipeline.

Steps:
  1. Load data/processed/train.csv and data/processed/test.csv
  2. Extract 20 features via extract_features_batch()
  3. Train Logistic Regression baseline
  4. Train Random Forest (primary model)
  5. Evaluate both on held-out test set
  6. Save best model to models/rf_model.pkl and scaler to models/scaler.pkl

Run: python3 src/train.py
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

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
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{'='*50}")
    print(f"  {name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=['benign', 'injection']))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")

    return {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc':  average_precision_score(y_test, y_prob),
    }


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df, test_df = load_data()

    print("\nExtracting features from train set...")
    X_train = build_feature_matrix(train_df['text'].tolist())
    y_train = train_df['label'].values

    print("Extracting features from test set...")
    X_test  = build_feature_matrix(test_df['text'].tolist())
    y_test  = test_df['label'].values

    feature_names = FeatureVector.feature_names()
    print(f"\nFeature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic Regression baseline 
    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate("Logistic Regression (baseline)", lr, X_test, y_test, scaler)

    # Random Forest 
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # Feature importance
    importances = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop-10 feature importances:")
    for fname, imp in importances[:10]:
        print(f"  {fname:40s}  {imp:.4f}")

    # Save models
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        'lr': lr_metrics,
        'rf': rf_metrics,
        'feature_names': feature_names,
        'train_size': int(len(train_df)),
        'test_size':  int(len(test_df)),
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/")
    print(f"RF ROC-AUC: {rf_metrics['roc_auc']:.4f}  |  LR ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    return rf, lr, scaler


if __name__ == '__main__':
    train()
