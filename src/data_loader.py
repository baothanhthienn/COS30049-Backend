"""
Merge four dataset sources into one consistent CSV.

Sources:
  primary   — xTRam1/safe-guard-prompt-injection  (columns: text, label)
  secondary — deepset/prompt-injections            (columns: text, label)
  tertiary  — jackhhao/jailbreak-classification    (columns: prompt, type)
  seas      — diaomuxi/SEAS, Role_Play category only (columns: prompt, category)

Transformation applied to each source:
  primary:   already {text, label}; add source='primary'
  secondary: already {text, label}; add source='secondary'
  tertiary:  rename prompt→text; map type (jailbreak→1, benign→0); add source='tertiary'
  seas:      rename prompt→text; label=1 (all attack prompts); add source='seas_role_play'

--- Why SEAS needs special handling (read before changing SEAS_* constants) ---
SEAS was tried once before at full scale (+3021 rows) and reverted: its
Role_Play prompts are built from a handful of fixed wrapper templates
("Mongo Tom", "HeLLM", "DAN", ...) with only the trailing harmful question
swapped out. A plain random train_test_split let near-identical siblings of
the same template land on both sides of the split, inflating test-set scores
without the model actually generalising (confirmed via manual curl testing
at the time — real-world false positives/negatives got WORSE even though the
test-set number went up).

This version re-adds SEAS at a much smaller, deduplicated scale:
  1. Filter to category == 'Role_Play' only (~3,600-3,700 of the 16k rows,
     based on a spot-check of the source file — the other 13 SEAS categories
     aren't relevant to this project's role-swap/narrative-frame features).
  2. Cluster near-duplicate templates via raw (non-IDF) character n-gram
     overlap — see _cluster_near_duplicate_templates for why TF-IDF itself
     is the WRONG tool here (it down-weights the shared boilerplate that
     defines a template and up-weights the one part we don't want to split
     on, the differing trailing question).
  3. Cap variants per template (SEAS_MAX_PER_TEMPLATE) and round-robin
     across templates up to SEAS_TARGET_ROWS, so the final set favours
     template DIVERSITY rather than whichever template happens to have the
     most raw rows.
  4. Split SEAS rows with a GROUP-aware split keyed on template_id, so every
     variant of a template lands entirely in train or entirely in test —
     this is the actual leakage fix; steps 1-3 just make the residual risk
     smaller even before this guarantee.

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

# --- SEAS Role_Play selection knobs ---
SEAS_TARGET_ROWS          = 400   # final row count after dedup/capping, before split
SEAS_MAX_PER_TEMPLATE     = 5     # cap variants of any one wrapper template
SEAS_SIMILARITY_THRESHOLD = 0.40  # cosine sim on raw char 5-gram overlap; validated
                                   # against a sample of the real data — correctly
                                   # separates 3 known template families (Mongo Tom,
                                   # HeLLM, DAN) from genuinely distinct one-off
                                   # Role_Play prompts (lawyer/hacker/psychotherapist
                                   # scenarios etc). Re-validate if this constant is
                                   # changed — see test at bottom of this docstring's
                                   # accompanying conversation, not reproduced here.


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


def _cluster_near_duplicate_templates(texts: list, threshold: float = SEAS_SIMILARITY_THRESHOLD) -> list:
    """
    Groups near-duplicate SEAS Role_Play prompts (same wrapper template, only
    the trailing harmful question differs) into cluster ids.

    IMPORTANT: uses raw binary character 5-gram overlap (CountVectorizer with
    binary=True), NOT TF-IDF. TF-IDF was tried first and fails for this task
    specifically: the long shared wrapper text (e.g. the ~700-character Mongo
    Tom preamble) is COMMON across many rows, so IDF down-weights it — while
    the short trailing question, which is unique per row, gets weighted
    heavily. That's the exact opposite of what's needed: it made every
    Mongo Tom variant look nearly as different from every other Mongo Tom
    variant as from a completely unrelated prompt. Raw n-gram overlap doesn't
    have this problem — the shared wrapper contributes its full similarity
    weight regardless of how often it recurs in the corpus.

    Greedy single-pass clustering (compare each new text only to existing
    cluster representatives, not full pairwise) is O(n * k) rather than
    O(n^2), which matters at ~3,600+ raw Role_Play rows.
    """
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
    """
    Dedup + cap + round-robin select down to target_rows, keeping a
    'template_id' column so build_combined can do a group-aware split.
    """
    df = df.reset_index(drop=True).copy()
    cluster_ids = _cluster_near_duplicate_templates(df['text'].tolist(), similarity_threshold)
    df['template_id'] = cluster_ids

    n_templates = df['template_id'].nunique()
    biggest = df['template_id'].value_counts().iloc[0]
    print(f"  SEAS Role_Play: {len(df)} raw rows collapse to {n_templates} distinct "
          f"templates (threshold={similarity_threshold}, largest template={biggest} rows)")

    # Cap variants per template.
    # NOTE: this used to be `df.groupby('template_id').apply(lambda g: g.sample(...))`,
    # but pandas (2.2+, confirmed on 3.0) silently DROPS the grouping column
    # from the result of .apply() by default — the resulting frame had no
    # 'template_id' column at all, which crashed the next line with a
    # KeyError. Shuffling once up front and using .groupby().head() instead
    # avoids .apply() entirely and reliably keeps every column.
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    capped = shuffled.groupby('template_id', as_index=False, group_keys=False).head(max_per_template)
    capped = capped.reset_index(drop=True)
    print(f"  After capping at {max_per_template} variants/template: {len(capped)} rows")

    # Round-robin across templates up to target_rows, so the selection favours
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
    """
    Fourth source, reverted per the earlier model_loader.py comment
    ("SEAS Role_Play, +3021 rows") but at a much smaller, deduplicated scale.
    See the module docstring for the full rationale.

    Downloads only SEAS-Train.jsonl (not SEAS-Test.jsonl — this project
    builds its own train/test split downstream, same as the other sources).
    """
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

    if save_raw:
        # Save tertiary splits alongside primary/secondary for record
        tertiary_train, tertiary_test = train_test_split(
            tertiary, test_size=test_size, stratify=tertiary['label'], random_state=random_state
        )
        os.makedirs(RAW_DIR, exist_ok=True)
        tertiary_train.to_csv(os.path.join(RAW_DIR, 'tertiary_train.csv'), index=False)
        tertiary_test.to_csv(os.path.join(RAW_DIR, 'tertiary_test.csv'),  index=False)

    # Non-SEAS split: unchanged from before — ordinary stratified random split.
    non_seas_train, non_seas_test = train_test_split(
        non_seas, test_size=test_size, stratify=non_seas['label'], random_state=random_state
    )

    if include_seas:
        print("Loading fourth source (diaomuxi/SEAS, Role_Play only)...")
        seas = _load_seas(random_state=random_state)
        print(f"  {len(seas)} rows  |  label dist: {seas['label'].value_counts().to_dict()}")

        # GROUP-aware split keyed on template_id: every variant of a template
        # lands entirely in train or entirely in test. This is the actual
        # leakage fix, not just the smaller row count from _select_role_play_subset.
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
            # Fourth source, saved the same way primary/secondary/tertiary
            # are: as its own raw/*_train.csv + raw/*_test.csv pair. Named
            # 'quaternary' to continue the primary/secondary/tertiary
            # ordinal naming already used in this file. template_id is
            # dropped here too — it's split-time bookkeeping only, kept out
            # of every saved CSV so schemas stay consistent (text, label,
            # source) across all four raw sources.
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