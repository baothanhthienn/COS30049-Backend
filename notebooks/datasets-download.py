"""
Download and save raw datasets to data/raw/.
Run once: python3 notebooks/00_download_datasets.py
"""
from datasets import load_dataset
import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_primary():
    print("Downloading xTRam1/safe-guard-prompt-injection ...")
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    print("  Splits:", list(ds.keys()))
    for split, data in ds.items():
        df = data.to_pandas()
        print(f"  {split}: {len(df)} rows, columns: {list(df.columns)}")
        path = os.path.join(RAW_DIR, f"primary_{split}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")
    return ds


def download_secondary():
    print("\nDownloading deepset/prompt-injections ...")
    ds = load_dataset("deepset/prompt-injections")
    print("  Splits:", list(ds.keys()))
    for split, data in ds.items():
        df = data.to_pandas()
        print(f"  {split}: {len(df)} rows, columns: {list(df.columns)}")
        path = os.path.join(RAW_DIR, f"secondary_{split}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")
    return ds


if __name__ == "__main__":
    ds1 = download_primary()
    ds2 = download_secondary()

    print("\n--- Dataset summary ---")
    for split in ds1.keys():
        df = pd.read_csv(os.path.join(RAW_DIR, f"primary_{split}.csv"))
        print(f"Primary {split}: {len(df)} rows")
        print(f"  Label distribution:\n{df.iloc[:, -1].value_counts().to_string()}\n")

    for split in ds2.keys():
        df = pd.read_csv(os.path.join(RAW_DIR, f"secondary_{split}.csv"))
        print(f"Secondary {split}: {len(df)} rows")
        print(f"  Label distribution:\n{df.iloc[:, -1].value_counts().to_string()}\n")

    print("Done. Files in data/raw/:")
    for f in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, f)
        print(f"  {f}  ({os.path.getsize(path):,} bytes)")
