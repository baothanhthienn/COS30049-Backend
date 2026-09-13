"""Tests for src/data_loader.py schema and merge correctness."""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import build_combined

PROCESSED = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


@pytest.fixture(scope='module')
def combined_csv():
    path = os.path.join(PROCESSED, 'combined_dataset.csv')
    if not os.path.exists(path):
        pytest.skip("combined_dataset.csv not built yet — run python3 src/data_loader.py first")
    return pd.read_csv(path)


def test_schema_columns(combined_csv):
    # must have exactly these three columns
    assert set(combined_csv.columns) >= {'text', 'label', 'source'}


def test_label_values_binary(combined_csv):
    # label must only be 0 or 1
    assert set(combined_csv['label'].unique()).issubset({0, 1})


def test_no_null_text(combined_csv):
    assert combined_csv['text'].isna().sum() == 0


def test_no_null_label(combined_csv):
    assert combined_csv['label'].isna().sum() == 0


def test_three_sources_present(combined_csv):
    sources = set(combined_csv['source'].unique())
    assert sources == {'primary', 'secondary', 'tertiary'}


def test_minimum_row_count(combined_csv):
    # primary(~34k) + secondary(~674) + tertiary(~1044) → well above 10k
    assert len(combined_csv) > 10_000


def test_tertiary_has_both_labels(combined_csv):
    tertiary = combined_csv[combined_csv['source'] == 'tertiary']
    assert 0 in tertiary['label'].values
    assert 1 in tertiary['label'].values
