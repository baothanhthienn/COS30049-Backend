"""
Text preprocessing pipeline for prompt injection detection.

Pipeline order (must not change — each step feeds the next):
  1. Base64 decode     — reveals hidden payloads in encoded blobs
  2. Unicode NFKC      — collapses compatibility characters (ﬁ→fi, ² →2)
  3. Lookalike map     — Cyrillic/Greek/fullwidth chars → ASCII equivalents
  4. Zero-width strip  — removes invisible Unicode (U+200B, U+FEFF, etc.)
  5. Leetspeak norm    — 4→a, 3→e, 0→o, 1→i/l so keyword signals fire

Returns both original_text (for span highlighting in /predict) and
decoded_text (for feature extraction) as a named tuple.
"""

import base64
import re
import unicodedata
from typing import NamedTuple


class ProcessedText(NamedTuple):
    original_text: str   # unchanged input — used for span/highlight index
    decoded_text: str    # fully normalised — used for feature extraction


# Look alike character substitution map
# Maps visually similar Unicode chars → ASCII. Covers Cyrillic, Greek,
# fullwidth Latin, and common homoglyphs used in prompt injection evasion.
_LOOKALIKE_MAP: dict[int, str] = {
    # Cyrillic → Latin
    ord('а'): 'a', ord('е'): 'e', ord('о'): 'o', ord('р'): 'p',
    ord('с'): 'c', ord('х'): 'x', ord('у'): 'y', ord('і'): 'i',
    ord('В'): 'B', ord('М'): 'M', ord('Н'): 'H', ord('К'): 'K',
    ord('Р'): 'P', ord('С'): 'C', ord('Т'): 'T', ord('Х'): 'X',
    # Greek → Latin
    ord('α'): 'a', ord('β'): 'b', ord('γ'): 'y', ord('ε'): 'e',
    ord('ι'): 'i', ord('κ'): 'k', ord('ν'): 'v', ord('ο'): 'o',
    ord('ρ'): 'p', ord('τ'): 't', ord('υ'): 'u', ord('χ'): 'x',
    # Fullwidth Latin (Ａ–Ｚ, ａ–ｚ, ０–９)
    **{ord('Ａ') + i: chr(ord('A') + i) for i in range(26)},
    **{ord('ａ') + i: chr(ord('a') + i) for i in range(26)},
    **{ord('０') + i: str(i) for i in range(10)},
    # Common punctuation lookalikes
    ord('‘'): "'", ord('’'): "'",   # curly single quotes
    ord('“'): '"', ord('”'): '"',   # curly double quotes
    ord('–'): '-', ord('—'): '-',   # en/em dash
    ord('−'): '-',                        # minus sign
    ord('．'): '.', ord('／'): '/',   # fullwidth period/slash
}

# Zero-width / invisible characters 
_ZERO_WIDTH_RE = re.compile(
    r'[​‌‍‎‏'   # zero-width space/non-joiner/joiner/LRM/RLM
    r'⁠⁡⁢⁣⁤'   # word joiner, function application, etc.
    r'﻿'                             # BOM / zero-width no-break space
    r'­'                             # soft hyphen
    r']'
)

# Leetspeak substitution map 
# Applied LAST — after lookalike normalisation so we don't double-substitute.
# Only substitutes digits/symbols that are unambiguously leet in this context.
_LEET_MAP: dict[str, str] = {
    '4': 'a', '@': 'a',
    '3': 'e',
    '1': 'i',   # also used as 'l' — 'i' is more common in injections
    '0': 'o',
    '5': 's',
    '7': 't',
    '$': 's',
    '!': 'i',
    '+': 't',
    '|': 'i',
}
_LEET_RE = re.compile(r'[4@31057$!+|]')


def _try_base64_decode(text: str) -> str:
    """
    Scan for Base64-looking substrings and replace with decoded UTF-8.
    Conservative: only decodes blobs of 16+ chars that decode cleanly to
    printable ASCII/UTF-8. Leaves the rest of the text intact.
    """
    # Match padded or unpadded base64 blobs (min 16 chars to avoid false positives)
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{16,}={0,2}')

    def try_replace(m: re.Match) -> str:
        blob = m.group(0)
        # Pad to multiple of 4
        pad = (4 - len(blob) % 4) % 4
        try:
            decoded = base64.b64decode(blob + '=' * pad).decode('utf-8', errors='strict')
            # Only accept if result is mostly printable (avoids binary garbage)
            printable = sum(c.isprintable() or c in '\n\t\r' for c in decoded)
            if printable / max(len(decoded), 1) >= 0.90:
                return decoded
        except Exception:
            pass
        return blob

    return b64_pattern.sub(try_replace, text)


def _apply_lookalike_map(text: str) -> str:
    return text.translate(_LOOKALIKE_MAP)


def _strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub('', text)


def _normalise_leet(text: str) -> str:
    return _LEET_RE.sub(lambda m: _LEET_MAP[m.group(0)], text)


def preprocess(text: str) -> ProcessedText:
    """
    Run the full pipeline on a single input string.

    Returns ProcessedText(original_text, decoded_text).
    - original_text: the raw input, unchanged
    - decoded_text:  normalised text ready for feature extraction
    """
    original = text

    # Step 1: Base64 decode
    step = _try_base64_decode(text)

    # Step 2: Unicode NFKC normalisation
    step = unicodedata.normalize('NFKC', step)

    # Step 3: Lookalike character substitution
    step = _apply_lookalike_map(step)

    # Step 4: Strip zero-width / invisible characters
    step = _strip_zero_width(step)

    # Step 5: Leetspeak normalisation
    step = _normalise_leet(step)

    return ProcessedText(original_text=original, decoded_text=step)


def preprocess_series(texts) -> tuple[list[str], list[str]]:
    """
    Batch-process an iterable of texts.
    Returns (original_texts, decoded_texts) as parallel lists.
    Suitable for pandas: originals, decoded = preprocess_series(df['text'])
    """
    results = [preprocess(t) for t in texts]
    originals = [r.original_text for r in results]
    decoded   = [r.decoded_text  for r in results]
    return originals, decoded
