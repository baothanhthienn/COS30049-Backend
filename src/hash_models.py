import hashlib
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUTPUT = os.path.join(MODELS_DIR, 'hashes.json')


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"No .pkl files found in {MODELS_DIR}. Run training first.")
        return

    hashes = {}
    for name in sorted(pkl_files):
        path = os.path.join(MODELS_DIR, name)
        digest = sha256_file(path)
        hashes[name] = digest
        print(f"  {name}: {digest[:16]}…")

    with open(OUTPUT, 'w') as f:
        json.dump(hashes, f, indent=2)

    print(f"\nWrote {len(hashes)} hashes to {OUTPUT}")
    print("Commit models/hashes.json alongside the model files.")


if __name__ == '__main__':
    main()
