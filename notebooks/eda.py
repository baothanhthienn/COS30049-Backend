"""
EDA — Prompt Injection Guardrail
Covers: label balance, text length, duplicates, class samples, concat strategy.
Run: python3 notebooks/01_eda.py
"""
import pandas as pd
import numpy as np
import os, re

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

# Load
p_train = pd.read_csv(os.path.join(RAW, "primary_train.csv"))
p_test  = pd.read_csv(os.path.join(RAW, "primary_test.csv"))
s_train = pd.read_csv(os.path.join(RAW, "secondary_train.csv"))
s_test  = pd.read_csv(os.path.join(RAW, "secondary_test.csv"))

print("=" * 60)
print("1. RAW DATASET OVERVIEW")
print("=" * 60)
for name, df in [("primary_train", p_train), ("primary_test", p_test),
                  ("secondary_train", s_train), ("secondary_test", s_test)]:
    inj = (df["label"] == 1).sum()
    ben = (df["label"] == 0).sum()
    print(f"\n{name}: {len(df):,} rows")
    print(f"  benign={ben:,} ({ben/len(df)*100:.1f}%)  injection={inj:,} ({inj/len(df)*100:.1f}%)")
    print(f"  columns: {list(df.columns)}")
    print(f"  null text: {df['text'].isna().sum()}")

# Text length analysis
print("\n" + "=" * 60)
print("2. TEXT LENGTH DISTRIBUTION (chars)")
print("=" * 60)

for name, df in [("primary_train", p_train), ("secondary_train", s_train)]:
    df = df.copy()
    df["length"] = df["text"].str.len()
    for lbl, lbl_name in [(0, "benign"), (1, "injection")]:
        sub = df[df["label"] == lbl]["length"]
        print(f"\n{name} — {lbl_name}:")
        print(f"  min={sub.min()}  median={sub.median():.0f}  mean={sub.mean():.0f}  "
              f"max={sub.max()}  p95={sub.quantile(0.95):.0f}")

# Duplicate analysis
print("\n" + "=" * 60)
print("3. DUPLICATE ANALYSIS")
print("=" * 60)

# Within primary
p_all = pd.concat([p_train, p_test], ignore_index=True)
p_dupes = p_all.duplicated(subset="text", keep=False).sum()
print(f"\nPrimary internal duplicates: {p_dupes}")

# Within secondary
s_all = pd.concat([s_train, s_test], ignore_index=True)
s_dupes = s_all.duplicated(subset="text", keep=False).sum()
print(f"Secondary internal duplicates: {s_dupes}")

# Cross-dataset overlap
primary_texts   = set(p_all["text"].str.strip().str.lower())
secondary_texts = set(s_all["text"].str.strip().str.lower())
overlap = primary_texts & secondary_texts
print(f"Cross-dataset overlap: {len(overlap)} texts appear in both")

# Label conflict check
print("\n" + "=" * 60)
print("4. LABEL CONFLICT CHECK (same text, different label)")
print("=" * 60)

combined_check = pd.concat([
    p_train.assign(source="p_train"),
    s_train.assign(source="s_train")
], ignore_index=True)
combined_check["text_norm"] = combined_check["text"].str.strip().str.lower()
conflicts = (combined_check.groupby("text_norm")["label"]
             .nunique()
             .reset_index()
             .query("label > 1"))
print(f"Label conflicts across train sets: {len(conflicts)}")
if len(conflicts) > 0:
    print(conflicts.head(5))

# Sample examples
print("\n" + "=" * 60)
print("5. SAMPLE EXAMPLES BY CLASS")
print("=" * 60)

rng = np.random.default_rng(42)
for lbl, lbl_name in [(1, "INJECTION"), (0, "BENIGN")]:
    print(f"\n--- {lbl_name} samples (primary_train) ---")
    samples = p_train[p_train["label"] == lbl].sample(3, random_state=42)
    for _, row in samples.iterrows():
        txt = row["text"][:200].replace("\n", " ")
        print(f"  [{len(row['text'])}ch] {txt!r}")

# Concat strategy 
print("\n" + "=" * 60)
print("6. CONCAT STRATEGY — TRAINING SPLIT ONLY")
print("=" * 60)

# Remove cross-dataset duplicates from secondary (keep primary version)
s_train_deduped = s_train[
    ~s_train["text"].str.strip().str.lower().isin(primary_texts)
].copy()
removed = len(s_train) - len(s_train_deduped)
print(f"\nSecondary train rows removed (overlap with primary): {removed}")
print(f"Secondary train rows kept: {len(s_train_deduped)}")

# Concatenate
combined_train = pd.concat([
    p_train[["text", "label"]],
    s_train_deduped[["text", "label"]]
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

inj_c = (combined_train["label"] == 1).sum()
ben_c = (combined_train["label"] == 0).sum()

print(f"\nCombined train: {len(combined_train):,} rows")
print(f"  benign={ben_c:,} ({ben_c/len(combined_train)*100:.1f}%)")
print(f"  injection={inj_c:,} ({inj_c/len(combined_train)*100:.1f}%)")
print(f"  imbalance ratio: {ben_c/inj_c:.2f}:1")

# Keep test set = primary only (clean, controlled)
combined_test = p_test[["text", "label"]].copy()
print(f"\nTest set (primary only): {len(combined_test):,} rows")
print(f"  benign={( combined_test['label']==0).sum():,}  injection={(combined_test['label']==1).sum():,}")

# Saved processed datasets
combined_train.to_csv(os.path.join(OUT, "train.csv"), index=False)
combined_test.to_csv(os.path.join(OUT,  "test.csv"),  index=False)
print(f"\nSaved → data/processed/train.csv ({len(combined_train):,} rows)")
print(f"Saved → data/processed/test.csv  ({len(combined_test):,} rows)")

# Imbalance handling recommendation
print("\n" + "=" * 60)
print("7. IMBALANCE HANDLING RECOMMENDATION")
print("=" * 60)
ratio = ben_c / inj_c
print(f"\nImbalance ratio: {ratio:.2f}:1  (benign:injection)")
if ratio < 2.0:
    print("Mild imbalance. class_weight='balanced' in sklearn is sufficient.")
    print("DECISION: use class_weight='balanced' — no SMOTE or undersampling needed.")
elif ratio < 4.0:
    print("Moderate imbalance. class_weight='balanced' recommended.")
    print("DECISION: use class_weight='balanced' in RF and LR.")
else:
    print("Severe imbalance. Consider SMOTE or undersampling.")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print("\nNext step: run notebooks/02_preprocessing.py")
