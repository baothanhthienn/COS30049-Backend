"""
Merge four dataset sources into one consistent CSV.
Output: data/processed/combined_dataset.csv  (columns: text, label, source)
        data/raw/tertiary_train.csv
        data/raw/tertiary_test.csv
        data/raw/quaternary_train.csv   (SEAS, post-dedup/capping, post-split)
        data/raw/quaternary_test.csv
Run: python3 src/data_loader.py
"""

import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAW_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

SEAS_TARGET_ROWS          = 400   
SEAS_MAX_PER_TEMPLATE     = 5     
SEAS_SIMILARITY_THRESHOLD = 0.40  

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
    ds = load_dataset('jackhhao/jailbreak-classification', split='train')
    df = ds.to_pandas()

    df = df.rename(columns={'prompt': 'text'})
    df['label'] = (df['type'] != 'benign').astype(int)
    df = df[['text', 'label']].copy()
    df['source'] = 'tertiary'
    return df


def _cluster_near_duplicate_templates(texts: list, threshold: float = SEAS_SIMILARITY_THRESHOLD) -> list:
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5), binary=True, min_df=1)
    X = vectorizer.fit_transform(texts).astype(float)

    cluster_ids = [-1] * len(texts)
    cluster_reps = []  # list of (cluster_id, row_index_of_representative)
    next_id = 0

    for i in range(len(texts)):
        if not cluster_reps:
            cluster_ids[i] = next_id
            cluster_reps.append((next_id, i))
            next_id += 1
            continue
        rep_indices = [idx for _, idx in cluster_reps]
        sims = cosine_similarity(X[i], X[rep_indices])[0]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            cluster_ids[i] = cluster_reps[best][0]
        else:
            cluster_ids[i] = next_id
            cluster_reps.append((next_id, i))
            next_id += 1

    return cluster_ids


def _select_role_play_subset(
    df: pd.DataFrame,
    target_rows: int = SEAS_TARGET_ROWS,
    max_per_template: int = SEAS_MAX_PER_TEMPLATE,
    similarity_threshold: float = SEAS_SIMILARITY_THRESHOLD,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    cluster_ids = _cluster_near_duplicate_templates(df['text'].tolist(), similarity_threshold)
    df['template_id'] = cluster_ids

    n_templates = df['template_id'].nunique()
    biggest = df['template_id'].value_counts().iloc[0]
    print(f"  SEAS Role_Play: {len(df)} raw rows collapse to {n_templates} distinct "
          f"templates (threshold={similarity_threshold}, largest template={biggest} rows)")

    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    capped = shuffled.groupby('template_id', as_index=False, group_keys=False).head(max_per_template)
    capped = capped.reset_index(drop=True)
    print(f"  After capping at {max_per_template} variants/template: {len(capped)} rows")

    # Round robin across templates up to target_rows, so the selection favours
    # template diversity over just taking whichever templates sort first.
    if len(capped) > target_rows:
        rng = np.random.RandomState(random_state)
        groups = {
            tid: g.sample(frac=1, random_state=random_state).to_dict('records')
            for tid, g in capped.groupby('template_id')
        }
        selected = []
        template_order = list(groups.keys())
        rng.shuffle(template_order)
        while len(selected) < target_rows and groups:
            for tid in list(template_order):
                if tid not in groups:
                    continue
                if not groups[tid]:
                    del groups[tid]
                    template_order.remove(tid)
                    continue
                selected.append(groups[tid].pop())
                if len(selected) >= target_rows:
                    break
        capped = pd.DataFrame(selected)

    print(f"  Final SEAS Role_Play subset: {len(capped)} rows across "
          f"{capped['template_id'].nunique()} templates")
    return capped


def _load_seas(
    target_rows: int = SEAS_TARGET_ROWS,
    max_per_template: int = SEAS_MAX_PER_TEMPLATE,
    similarity_threshold: float = SEAS_SIMILARITY_THRESHOLD,
    random_state: int = 42,
) -> pd.DataFrame:
    
    from huggingface_hub import hf_hub_download

    print("Downloading SEAS-Train.jsonl (diaomuxi/SEAS)...")
    path = hf_hub_download(repo_id='diaomuxi/SEAS', filename='SEAS-Train.jsonl', repo_type='dataset')
    df = pd.read_json(path, lines=True)
    print(f"  {len(df)} total SEAS rows across all 14 categories")

    role_play = df[df['category'] == 'Role_Play'][['prompt']].copy()
    role_play = role_play.rename(columns={'prompt': 'text'})
    role_play = role_play.dropna(subset=['text'])
    role_play = role_play[role_play['text'].str.strip().ne('')]
    role_play = role_play.drop_duplicates(subset='text')
    print(f"  {len(role_play)} Role_Play rows (after exact-duplicate drop)")

    subset = _select_role_play_subset(
        role_play, target_rows=target_rows, max_per_template=max_per_template,
        similarity_threshold=similarity_threshold, random_state=random_state,
    )
    subset['label']  = 1  # SEAS Role_Play rows are all attack prompts — no benign class here
    subset['source'] = 'seas_role_play'
    return subset[['text', 'label', 'source', 'template_id']]


def build_combined(
    save_raw: bool = True,
    test_size: float = 0.19,
    random_state: int = 42,
    include_seas: bool = True,
) -> tuple:
    """Return (train_df, test_df) from the merged four-source dataset."""
    print("Loading primary source (xTRam1/safe-guard-prompt-injection)...")
    primary = _load_primary()
    print(f"  {len(primary)} rows  |  label dist: {primary['label'].value_counts().to_dict()}")

    print("Loading secondary source (deepset/prompt-injections)...")
    secondary = _load_secondary()
    print(f"  {len(secondary)} rows  |  label dist: {secondary['label'].value_counts().to_dict()}")

    print("Loading tertiary source (jackhhao/jailbreak-classification)...")
    tertiary = _load_tertiary()
    print(f"  {len(tertiary)} rows  |  label dist: {tertiary['label'].value_counts().to_dict()}")

    non_seas = pd.concat([primary, secondary, tertiary], ignore_index=True)
    non_seas['template_id'] = None  # every non-SEAS row is its own singleton group

    # Drop empty or whitespace-only texts
    non_seas = non_seas[non_seas['text'].str.strip().ne('')]
    non_seas = non_seas.dropna(subset=['text', 'label'])
    non_seas['label'] = non_seas['label'].astype(int)

    # Save tertiary splits 
    if save_raw:
        tertiary_train, tertiary_test = train_test_split(
            tertiary, test_size=test_size, stratify=tertiary['label'], random_state=random_state
        )
        os.makedirs(RAW_DIR, exist_ok=True)
        tertiary_train.to_csv(os.path.join(RAW_DIR, 'tertiary_train.csv'), index=False)
        tertiary_test.to_csv(os.path.join(RAW_DIR, 'tertiary_test.csv'),  index=False)

    # Non SEAS split: unchanged from before, ordinary stratified random split.
    non_seas_train, non_seas_test = train_test_split(
        non_seas, test_size=test_size, stratify=non_seas['label'], random_state=random_state
    )

    if include_seas:
        print("Loading fourth source (diaomuxi/SEAS, Role_Play only)...")
        seas = _load_seas(random_state=random_state)
        print(f"  {len(seas)} rows  |  label dist: {seas['label'].value_counts().to_dict()}")

        # GROUP aware split keyed on template_id: every variant of a template
        # lands entirely in train or entirely in test. 
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        seas_train_idx, seas_test_idx = next(gss.split(seas, groups=seas['template_id']))
        seas_train = seas.iloc[seas_train_idx]
        seas_test  = seas.iloc[seas_test_idx]

        n_train_templates = seas_train['template_id'].nunique()
        n_test_templates   = seas_test['template_id'].nunique()
        overlap = set(seas_train['template_id']) & set(seas_test['template_id'])
        print(f"  SEAS split: {len(seas_train)} train rows ({n_train_templates} templates), "
              f"{len(seas_test)} test rows ({n_test_templates} templates), "
              f"template overlap between train/test: {len(overlap)} (should be 0)")
        assert len(overlap) == 0, "SEAS template leaked across train/test — group split failed"

        if save_raw:
            seas_train.drop(columns=['template_id'], errors='ignore').to_csv(
                os.path.join(RAW_DIR, 'quaternary_train.csv'), index=False)
            seas_test.drop(columns=['template_id'], errors='ignore').to_csv(
                os.path.join(RAW_DIR, 'quaternary_test.csv'), index=False)
            print(f"  Saved data/raw/quaternary_train.csv ({len(seas_train)} rows), "
                  f"data/raw/quaternary_test.csv ({len(seas_test)} rows)")

        train_df = pd.concat([non_seas_train, seas_train], ignore_index=True)
        test_df  = pd.concat([non_seas_test, seas_test], ignore_index=True)
    else:
        train_df = non_seas_train
        test_df  = non_seas_test

    # Shuffle so SEAS rows aren't all clustered at the end of the file
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    combined = pd.concat([train_df, test_df], ignore_index=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # template_id is internal bookkeeping for the split only — drop before saving
    # so the CSV schema matches what train.py / clustering.py / evaluate.py expect.
    combined.drop(columns=['template_id'], errors='ignore').to_csv(
        os.path.join(PROCESSED_DIR, 'combined_dataset.csv'), index=False)
    train_df.drop(columns=['template_id'], errors='ignore').to_csv(
        os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    test_df.drop(columns=['template_id'], errors='ignore').to_csv(
        os.path.join(PROCESSED_DIR, 'test.csv'), index=False)

    print(f"\nCombined: {len(combined)} rows total")
    print(f"  label dist: {combined['label'].value_counts().to_dict()}")
    print(f"  source dist: {combined['source'].value_counts().to_dict()}")
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
    return train_df, test_df


if __name__ == '__main__':
    build_combined()
    print("\nSaved to data/processed/combined_dataset.csv, train.csv, test.csv")