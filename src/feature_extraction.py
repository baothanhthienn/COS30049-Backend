"""
Feature extraction — 20 features in 3 groups.

Group 1 — Keyword / phrase signals (4 features):
  f01  override_keyword_count    imperative override phrases ("ignore all", "disregard")
  f02  role_swap_keyword_count   persona/role hijack phrases ("act as", "you are now")
  f03  data_exfil_keyword_count  data/credential theft phrases ("give me your", "api key")
  f04  filter_bypass_count       filter-bypass phrases ("bypass", "jailbreak", "no restrictions")

Group 2 — Encoding anomaly signals (6 features):
  f05  has_base64_blob           1 if a valid base64 blob (≥16 chars) was detected
  f06  unicode_lookalike_count   count of Cyrillic/Greek/fullwidth lookalike chars in original
  f07  zero_width_count          count of zero-width / invisible Unicode chars in original
  f08  leetspeak_density         fraction of chars in decoded_text that were leet substitutes
  f09  non_ascii_ratio           fraction of chars in original that are non-ASCII
  f10  encoding_anomaly_score    composite: sum of f05–f09 normalised to 0–1

Group 3 — Structural signals (10 features):
  f11  char_entropy              Shannon entropy of character distribution in decoded_text
  f12  uppercase_ratio           fraction of alpha chars that are uppercase
  f13  avg_word_length           average word length in decoded_text
  f14  sentence_count            number of sentences (split on .!?)
  f15  avg_sentence_length       average words per sentence
  f16  special_char_ratio        fraction of non-alphanumeric, non-space chars
  f17  exclamation_count         count of ! in original
  f18  imperative_opener         1 if decoded_text starts with an imperative verb
  f19  text_length               total char count of original_text
  f20  word_count                total word count of decoded_text

All features are floats (ints cast to float) for scikit-learn compatibility.
Span computation is on original_text so highlight indices survive encoding.
"""

import math
import re
from typing import NamedTuple

from preprocessing import ProcessedText, preprocess


# Keyword lists 

_OVERRIDE_PHRASES = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier|system)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier|system)",
    r"forget\s+(all\s+)?(previous|prior|above|earlier|your)",
    r"override\s+(all\s+)?(previous|prior|above|safety|system)",
    r"bypass\s+(all\s+)?instructions",
    r"do\s+not\s+(follow|obey|respect)\s+(your|the|any|previous)\s+instructions",
    r"you\s+must\s+ignore",
    r"pretend\s+(that\s+)?you\s+(have\s+no|don'?t\s+have)",
]

_ROLE_SWAP_PHRASES = [
    r"\bact\s+as\b",
    r"\byou\s+are\s+now\b",
    r"\byou\s+will\s+now\s+(be|act|pretend|play)",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\bplay\s+(the\s+role|as)\b",
    r"\bswitch\s+(to|your)\s+(role|persona|mode)\b",
    r"\bimagine\s+you\s+(are|were)\b",
    r"\bfrom\s+now\s+on\s+(you\s+are|act)\b",
    r"\byour\s+new\s+(role|persona|identity|name)\b",
]

_DATA_EXFIL_PHRASES = [
    r"\b(give|tell|show|reveal|share|send|output|print|display)\s+me\s+(your|the|all)\b",
    r"\b(api\s*key|secret|password|credential|token|private\s+key)\b",
    r"\bconfidential\b",
    r"\b(access|expose|leak|dump)\s+(all\s+)?(data|information|records|database)\b",
    r"\bsensitive\s+(data|information|records)\b",
    r"\bpersonal\s+(data|information|records|details)\b",
    r"\b(user|customer|patient)\s+(data|information|records)\b",
]

_FILTER_BYPASS_PHRASES = [
    r"\bbypass\b",
    r"\bjailbreak\b",
    r"\bno\s+restrictions?\b",
    r"\bunrestricted\b",
    r"\bno\s+limits?\b",
    r"\bfreedom\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bdan\s+mode\b",
    r"\b(disable|remove|ignore|turn\s+off)\s+(all\s+)?(safety|filter|restriction|guardrail|rule)\b",
    r"\bsafety\s+(is\s+)?(off|disabled|removed)\b",
]

_IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "override", "bypass", "pretend",
    "act", "play", "imagine", "reveal", "give", "show", "tell", "output",
    "print", "access", "expose", "leak", "dump", "disable", "remove",
}

# Pre-compile all regex patterns at import time so the cost is paid once, not
# on every request. Each phrase is wrapped in a non-capturing group so the
# alternation '|' doesn't interfere with inner groups inside individual patterns.
def _compile(phrases: list[str]) -> re.Pattern:
    combined = "|".join(f"(?:{p})" for p in phrases)
    return re.compile(combined, re.IGNORECASE)

_RE_OVERRIDE    = _compile(_OVERRIDE_PHRASES)
_RE_ROLE_SWAP   = _compile(_ROLE_SWAP_PHRASES)
_RE_DATA_EXFIL  = _compile(_DATA_EXFIL_PHRASES)
_RE_FILTER_BYP  = _compile(_FILTER_BYPASS_PHRASES)

# Lookalike char set (same chars as the lookalike map in preprocessing)
_LOOKALIKE_CHARS = set(
    'аеорсхуіВМНКРСТХαβγεικνορτυχ'
    + ''.join(chr(ord('Ａ') + i) for i in range(26))
    + ''.join(chr(ord('ａ') + i) for i in range(26))
    + ''.join(chr(ord('０') + i) for i in range(10))
)

_ZERO_WIDTH_SET = set('​‌‍‎‏⁠⁡⁢⁣⁤﻿­')

_LEET_CHARS = set('4@310057$!+|')   # chars consumed by leet normaliser

_B64_RE = re.compile(r'[A-Za-z0-9+/]{16,}={0,2}')

_SENTENCE_SPLIT_RE = re.compile(r'[.!?]+')

_WORD_RE = re.compile(r'\w+')


# Feature vector 

class FeatureVector(NamedTuple):
    f01_override_keyword_count:    float
    f02_role_swap_keyword_count:   float
    f03_data_exfil_keyword_count:  float
    f04_filter_bypass_count:       float
    f05_has_base64_blob:           float
    f06_unicode_lookalike_count:   float
    f07_zero_width_count:          float
    f08_leetspeak_density:         float
    f09_non_ascii_ratio:           float
    f10_encoding_anomaly_score:    float
    f11_char_entropy:              float
    f12_uppercase_ratio:           float
    f13_avg_word_length:           float
    f14_sentence_count:            float
    f15_avg_sentence_length:       float
    f16_special_char_ratio:        float
    f17_exclamation_count:         float
    f18_imperative_opener:         float
    f19_text_length:               float
    f20_word_count:                float

    def to_list(self) -> list[float]:
        return list(self)

    @staticmethod
    def feature_names() -> list[str]:
        return list(FeatureVector._fields)


# Span helper 

class Span(NamedTuple):
    start: int
    end: int
    label: str   # which keyword group matched


def find_spans(original_text: str, decoded_text: str) -> list[Span]:
    """
    Find character-level spans of trigger phrases in original_text.

    We search decoded_text (normalised) and map offsets back to original_text.
    The offset mapping is approximate — we use char-count delta accumulated
    from deletions (zero-width strips) since insertions (Base64 decode) expand
    the text and offset direct mapping is non-trivial. For the dashboard,
    approximate highlighting on the original is acceptable.

    Strategy: search on decoded_text, return spans relative to decoded_text.
    The API returns both decoded_text and spans so the frontend can highlight
    decoded_text directly.
    """
    spans: list[Span] = []
    for pattern, label in [
        (_RE_OVERRIDE,   "instruction_override"),
        (_RE_ROLE_SWAP,  "role_swap"),
        (_RE_DATA_EXFIL, "data_exfiltration"),
        (_RE_FILTER_BYP, "filter_bypass"),
    ]:
        for m in pattern.finditer(decoded_text):
            spans.append(Span(m.start(), m.end(), label))
    spans.sort(key=lambda s: s.start)
    return spans


# Individual feature functions

def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    return -sum((n / total) * math.log2(n / total) for n in freq.values())


def _uppercase_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _avg_word_length(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _sentence_stats(text: str) -> tuple[int, float]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    count = max(len(sentences), 1)
    word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    avg = sum(word_counts) / max(len(word_counts), 1)
    return count, avg


def _special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def _leetspeak_density(original: str, decoded: str) -> float:
    if not original:
        return 0.0
    leet_count = sum(1 for c in original if c in _LEET_CHARS)
    return leet_count / len(original)


def _encoding_anomaly_score(
    has_b64: float,
    lookalike_count: float,
    zero_width_count: float,
    leet_density: float,
    non_ascii_ratio: float,
) -> float:
    score = (
        has_b64 * 0.30
        + min(lookalike_count / 10.0, 1.0) * 0.25
        + min(zero_width_count / 5.0, 1.0) * 0.20
        + leet_density * 0.15
        + non_ascii_ratio * 0.10
    )
    return min(score, 1.0)


# Main extraction function
def extract_features(processed: ProcessedText) -> FeatureVector:
    """
    Extract all 20 features from a ProcessedText named tuple.
    Call preprocess() first to get a ProcessedText.
    """
    orig = processed.original_text
    dec  = processed.decoded_text

    # Group 1: keyword signals (search decoded_text — normalised)
    f01 = float(len(_RE_OVERRIDE.findall(dec)))
    f02 = float(len(_RE_ROLE_SWAP.findall(dec)))
    f03 = float(len(_RE_DATA_EXFIL.findall(dec)))
    f04 = float(len(_RE_FILTER_BYP.findall(dec)))

    # Group 2: encoding anomaly signals (search original_text)
    f05 = 1.0 if _B64_RE.search(orig) else 0.0
    f06 = float(sum(1 for c in orig if c in _LOOKALIKE_CHARS))
    f07 = float(sum(1 for c in orig if c in _ZERO_WIDTH_SET))
    f08 = _leetspeak_density(orig, dec)
    f09 = (sum(1 for c in orig if ord(c) > 127) / max(len(orig), 1))
    f10 = _encoding_anomaly_score(f05, f06, f07, f08, f09)

    # Group 3: structural signals (mostly on decoded_text)
    f11 = _char_entropy(dec)
    f12 = _uppercase_ratio(dec)
    f13 = _avg_word_length(dec)
    f14_int, f15 = _sentence_stats(dec)
    f14 = float(f14_int)
    f16 = _special_char_ratio(dec)
    f17 = float(orig.count('!'))

    first_word = _WORD_RE.search(dec)
    f18 = 1.0 if (first_word and first_word.group(0).lower() in _IMPERATIVE_VERBS) else 0.0

    f19 = float(len(orig))
    f20 = float(len(_WORD_RE.findall(dec)))

    return FeatureVector(
        f01, f02, f03, f04,
        f05, f06, f07, f08, f09, f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    )


def extract_features_from_text(text: str) -> tuple[FeatureVector, list[Span]]:
    processed = preprocess(text)
    features  = extract_features(processed)
    spans     = find_spans(processed.original_text, processed.decoded_text)
    return features, spans


def extract_features_batch(texts) -> list[FeatureVector]:
    return [extract_features(preprocess(t)) for t in texts]
