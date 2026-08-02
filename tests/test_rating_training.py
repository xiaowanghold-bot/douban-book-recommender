"""Tests for leakage-safe rating target encoding."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.rating_training import (
    build_oof_rating_feature_frame,
    build_rating_feature_frame,
    fit_target_encoders,
)


def make_frame():
    return pd.DataFrame(
        {
            "Rating": [6.0, 7.0, 8.0, 9.0, 10.0],
            "Votes": [10, 20, 30, 40, 50],
            "price_num": [20, 21, 22, 23, 24],
            "year_num": [2018, 2019, 2020, 2021, 2022],
            "pages_num": [200, 210, 220, 230, 240],
            "author_clean": ["A", "B", "C", "D", "E"],
            "publisher_clean": ["P", "P", "P", "P", "P"],
            "binding_type": ["平装", "平装", "精装", "精装", "平装"],
        }
    )


def test_oof_unique_author_never_uses_its_own_rating():
    df = make_frame()
    features = build_oof_rating_feature_frame(df, n_splits=5, random_state=42)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    expected = np.empty(len(df))
    for fit_positions, validation_positions in splitter.split(df):
        expected[validation_positions] = df.iloc[fit_positions]["Rating"].mean()

    np.testing.assert_allclose(features["author_mean"].to_numpy(), expected)


def test_validation_rows_use_reference_encoders_without_their_targets():
    train = make_frame().iloc[:4]
    validation = make_frame().iloc[[4]]
    encoders = fit_target_encoders(train)

    features = build_rating_feature_frame(validation, encoders)

    assert features.iloc[0]["author_mean"] == train["Rating"].mean()
    assert features.iloc[0]["publisher_mean"] == train["Rating"].mean()
