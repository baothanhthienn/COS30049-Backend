"""
Merge all three dataset sources into one consistent CSV.

Sources:
  primary   — xTRam1/safe-guard-prompt-injection  (columns: text, label)
  secondary — deepset/prompt-injections            (columns: text, label)
  tertiary  — jackhhao/jailbreak-classification    (columns: prompt, type)

Transformation applied to each source:
  primary:   already {text, label}; add source='primary'
  secondary: already {text, label}; add source='secondary'
  tertiary:  rename prompt→text; map type (jailbreak→1, benign→0); add source='tertiary'

Output: data/processed/combined_dataset.csv  (columns: text, label, source)
        data/raw/tertiary_train.csv
        data/raw/tertiary_test.csv

Run: python3 src/data_loader.py
"""

import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RAW_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def _load_primary() -> pd.DataFrame:
    # xTRam1 already split into raw/primary_*.csv
    train = pd.read_csv(os.path.join(RAW_DIR, 'primary_train.csv'))
    test  = pd.read_csv(os.path.join(RAW_DIR, 'primary_test.csv'))
    df = pd.concat([train, test], ignore_index=True)
    df = df[['text', 'label']].copy()
    df['source'] = 'primary'
    return df


def _load_secondary() -> pd.DataFrame:
    # deepset already split into raw/secondary_*.csv
    train = pd.read_csv(os.path.join(RAW_DIR, 'secondary_train.csv'))
    test  = pd.read_csv(os.path.join(RAW_DIR, 'secondary_test.csv'))
    df = pd.concat([train, test], ignore_index=True)
    df = df[['text', 'label']].copy()
    df['source'] = 'secondary'
    return df


def _load_tertiary() -> pd.DataFrame:
    # jackhhao has 'prompt' column and 'type' (jailbreak | benign)
    # jailbreak attacks overlap with prompt injections — both try to override model behaviour
    ds = load_dataset('jackhhao/jailbreak-classification', split='train')
    df = ds.to_pandas()

    df = df.rename(columns={'prompt': 'text'})
    df['label'] = (df['type'] != 'benign').astype(int)
    df = df[['text', 'label']].copy()
    df['source'] = 'tertiary'
    return df


def build_combined(
    save_raw: bool = True,
    test_size: float = 0.19,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) from the merged three-source dataset."""
    print("Loading primary source (xTRam1/safe-guard-prompt-injection)...")
    primary = _load_primary()
    print(f"  {len(primary)} rows  |  label dist: {primary['label'].value_counts().to_dict()}")

    print("Loading secondary source (deepset/prompt-injections)...")
    secondary = _load_secondary()
    print(f"  {len(secondary)} rows  |  label dist: {secondary['label'].value_counts().to_dict()}")

    print("Loading tertiary source (jackhhao/jailbreak-classification)...")
    tertiary = _load_tertiary()
    print(f"  {len(tertiary)} rows  |  label dist: {tertiary['label'].value_counts().to_dict()}")

    combined = pd.concat([primary, secondary, tertiary], ignore_index=True)

    # Drop empty or whitespace-only texts
    combined = combined[combined['text'].str.strip().ne('')]
    combined = combined.dropna(subset=['text', 'label'])
    combined['label'] = combined['label'].astype(int)

    if save_raw:
        # Save tertiary splits alongside primary/secondary for record
        tertiary_train, tertiary_test = train_test_split(
            tertiary, test_size=test_size, stratify=tertiary['label'], random_state=random_state
        )
        os.makedirs(RAW_DIR, exist_ok=True)
        tertiary_train.to_csv(os.path.join(RAW_DIR, 'tertiary_train.csv'), index=False)
        tertiary_test.to_csv(os.path.join(RAW_DIR, 'tertiary_test.csv'),  index=False)

    # Re-split the whole combined set so train/test proportions are consistent
    train_df, test_df = train_test_split(
        combined, test_size=test_size, stratify=combined['label'], random_state=random_state
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    combined.to_csv(os.path.join(PROCESSED_DIR, 'combined_dataset.csv'), index=False)

    # Replace the existing processed train/test with the new combined split
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'),  index=False)

    print(f"\nCombined: {len(combined)} rows total")
    print(f"  label dist: {combined['label'].value_counts().to_dict()}")
    print(f"  source dist: {combined['source'].value_counts().to_dict()}")
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
    return train_df, test_df


if __name__ == '__main__':
    build_combined()
    print("\nSaved to data/processed/combined_dataset.csv, train.csv, test.csv")
