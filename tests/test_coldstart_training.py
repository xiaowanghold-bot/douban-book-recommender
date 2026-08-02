"""Tests for leakage-safe cold-start feature generation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.coldstart_training import (
    build_coldstart_feature_frame,
    build_coldstart_stats,
    build_oof_coldstart_feature_frame,
)


def make_frame():
    return pd.DataFrame(
        {
            "Rating": [6.0, 7.0, 8.0, 9.0, 10.0],
            "author": ["A", "B", "C", "D", "E"],
            "publisher": ["P", "P", "P", "P", "P"],
            "binding": ["平装", "平装", "精装", "精装", "平装"],
            "pub_year": [2018, 2019, 2020, 2021, 2022],
            "pages": [200, 210, 220, 230, 240],
            "is_translation": [0, 0, 1, 1, 0],
            "is_series": [0, 1, 0, 1, 0],
        }
    )


def test_oof_unique_author_uses_other_fold_global_mean():
    df = make_frame()
    features = build_oof_coldstart_feature_frame(df, n_splits=5, random_state=42)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    expected = np.empty(len(df))
    for fit_positions, validation_positions in splitter.split(df):
        expected[validation_positions] = df.iloc[fit_positions]["Rating"].mean()

    np.testing.assert_allclose(features["author_avg_rating"], expected)


def test_validation_features_do_not_depend_on_validation_rating():
    train = make_frame().iloc[:4]
    stats = build_coldstart_stats(train)
    validation = make_frame().iloc[[4]].copy()
    changed_target = validation.copy()
    changed_target["Rating"] = 1.0

    original = build_coldstart_feature_frame(validation, stats)
    changed = build_coldstart_feature_frame(changed_target, stats)

    np.testing.assert_allclose(original.values, changed.values)


def test_single_book_publisher_std_falls_back_to_global_std():
    df = make_frame().copy()
    df.loc[4, "publisher"] = "single"
    stats = build_coldstart_stats(df)

    assert stats["publisher"].loc["single", "pub_std_rating"] == stats["global_std"]
