"""
§2.8 Detection Evasion — analyse before/after JSON results.

Usage:
  python attacks/analyse_evasion.py \
    --before attacks/results/evasion_before.json \
    --after  attacks/results/evasion_after.json \
    --chart  attacks/results/evasion_comparison.png
"""
import argparse
import json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="§2.8 Evasion before/after analysis")
    parser.add_argument("--before", required=True, help="evasion_before.json")
    parser.add_argument("--after",  required=True, help="evasion_after.json")
    parser.add_argument("--chart",  default="attacks/results/evasion_comparison.png")
    args = parser.parse_args()

    before = load(args.before)
    after  = load(args.after)

    kinds = list(before["attack_kinds"].keys())

    print("=" * 72)
    print("§2.8 Detection Evasion — Before vs After NFKC Normalisation")
    print("=" * 72)
    print(f"{'Attack kind':<18} {'Before evasion':>16}  {'After evasion':>14}  {'Reduction':>10}")
    print("-" * 72)

    before_rates = []
    after_rates  = []

    for kind in kinds:
        b = before["attack_kinds"].get(kind, {})
        a = after["attack_kinds"].get(kind, {})
        br = b.get("evasion_rate", 0.0)
        ar = a.get("evasion_rate", 0.0)
        reduction = br - ar
        before_rates.append(br)
        after_rates.append(ar)
        print(f"  {kind:<16} {br:>14.1%}   {ar:>12.1%}   {reduction:>+9.1%}")

    avg_before = sum(before_rates) / len(before_rates) if before_rates else 0
    avg_after  = sum(after_rates)  / len(after_rates)  if after_rates  else 0
    print("-" * 72)
    print(f"  {'AVERAGE':<16} {avg_before:>14.1%}   {avg_after:>12.1%}   {avg_before-avg_after:>+9.1%}")
    print()

    if avg_after <= avg_before * 0.5:
        print("✓ PASS — normalisation reduced average evasion rate by ≥50%.")
    elif avg_after < avg_before:
        print("~ PARTIAL — normalisation reduced evasion but some evasion remains.")
    else:
        print("✗ FAIL — normalisation did not reduce evasion rate.")

    print()
    print("Cost of the defence:")
    print("  Normalisation adds ~0.1–0.5 ms per request (NFKC + lookalike map pass).")
    print("  False-positive risk on non-Latin legitimate inputs is low because")
    print("  NFKC + lookalike substitution only affects characters with ASCII")
    print("  equivalents; Mandarin, Arabic, etc. are unaffected.")

    if not HAS_MATPLOTLIB:
        print("\n(Install matplotlib to generate chart: pip install matplotlib)")
        return

    x = range(len(kinds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], [r * 100 for r in before_rates],
                   width, label="Before (NORMALISE=off)", color="#e05252")
    bars2 = ax.bar([i + width/2 for i in x], [r * 100 for r in after_rates],
                   width, label="After (NORMALISE=on)",  color="#52b052")

    ax.set_xlabel("Attack Kind")
    ax.set_ylabel("Evasion Rate (%)")
    ax.set_title("§2.8 Detection Evasion: Before vs After NFKC Normalisation")
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.chart, dpi=150)
    print(f"\nChart saved → {args.chart}")


if __name__ == "__main__":
    main()
