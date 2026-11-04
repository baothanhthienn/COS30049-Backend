import base64
import re
import unicodedata
from typing import NamedTuple


class ProcessedText(NamedTuple):
    original_text: str   
    decoded_text: str   


_LOOKALIKE_MAP: dict[int, str] = {
    ord('а'): 'a', ord('е'): 'e', ord('о'): 'o', ord('р'): 'p',
    ord('с'): 'c', ord('х'): 'x', ord('у'): 'y', ord('і'): 'i',
    ord('В'): 'B', ord('М'): 'M', ord('Н'): 'H', ord('К'): 'K',
    ord('Р'): 'P', ord('С'): 'C', ord('Т'): 'T', ord('Х'): 'X',
    ord('α'): 'a', ord('β'): 'b', ord('γ'): 'y', ord('ε'): 'e',
    ord('ι'): 'i', ord('κ'): 'k', ord('ν'): 'v', ord('ο'): 'o',
    ord('ρ'): 'p', ord('τ'): 't', ord('υ'): 'u', ord('χ'): 'x',
    **{ord('Ａ') + i: chr(ord('A') + i) for i in range(26)},
    **{ord('ａ') + i: chr(ord('a') + i) for i in range(26)},
    **{ord('０') + i: str(i) for i in range(10)},
    ord('‘'): "'", ord('’'): "'",   
    ord('“'): '"', ord('”'): '"',   
    ord('–'): '-', ord('—'): '-',   
    ord('−'): '-',                       
    ord('．'): '.', ord('／'): '/',   
}

_ZERO_WIDTH_RE = re.compile(
    r'[​‌‍‎‏'   
    r'⁠⁡⁢⁣⁤'   
    r'﻿'                            
    r'­'                            
    r']'
)

_LEET_MAP: dict[str, str] = {
    '4': 'a', '@': 'a',
    '3': 'e',
    '1': 'i', 
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
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{16,}={0,2}')

    def try_replace(m: re.Match) -> str:
        blob = m.group(0)
        pad = (4 - len(blob) % 4) % 4
        try:
            decoded = base64.b64decode(blob + '=' * pad).decode('utf-8', errors='strict')
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


def preprocess(text: str, normalise: bool = True) -> ProcessedText:
    original = text
    if not normalise:
        return ProcessedText(original_text=original, decoded_text=text)

    # Base64 decode
    step = _try_base64_decode(text)

    # Unicode NFKC normalisation
    step = unicodedata.normalize('NFKC', step)

    # Lookalike character substitution
    step = _apply_lookalike_map(step)

    # Strip zero width / invisible characters
    step = _strip_zero_width(step)

    # Leetspeak normalisation
    step = _normalise_leet(step)

    return ProcessedText(original_text=original, decoded_text=step)


def preprocess_series(texts) -> tuple[list[str], list[str]]:
    results = [preprocess(t) for t in texts]
    originals = [r.original_text for r in results]
    decoded   = [r.decoded_text  for r in results]
    return originals, decoded
