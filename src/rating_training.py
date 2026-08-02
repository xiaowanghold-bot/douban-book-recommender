"""Shared, leakage-safe training helpers for the rating predictor."""

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.rating_predictor import (
    clean_rating_author,
    clean_rating_binding,
    clean_rating_publisher,
)


FEATURE_NAMES = [
    "price",
    "year",
    "pages",
    "votes_log",
    "author_mean",
    "publisher_mean",
    "binding_mean",
]

CATEGORY_FEATURES = {
    "author_clean": ("author_means", "author_mean"),
    "publisher_clean": ("publisher_means", "publisher_mean"),
    "binding_type": ("binding_means", "binding_mean"),
}


def _parse_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else np.nan


def _parse_year(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group()) if match else np.nan


def prepare_rating_dataframe(detail):
    """Create the canonical training frame from the raw Books_detail schema."""
    df = detail.copy()
    df = df[df["crawl_status"] == "success"].copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
    df["price_num"] = df["price"].apply(_parse_number)
    df["year_num"] = df["pub_year"].apply(_parse_year)
    df["pages_num"] = df["pages"].apply(_parse_number)
    df["author_clean"] = df["author"].apply(
        lambda value: clean_rating_author(value) if pd.notna(value) else "未知"
    )
    df["publisher_clean"] = df["publisher"].apply(
        lambda value: clean_rating_publisher(value) if pd.notna(value) else "未知"
    )
    df["binding_type"] = df["binding"].apply(clean_rating_binding)

    required = ["Rating", "Votes", "price_num", "year_num", "pages_num"]
    df = df.dropna(subset=required).copy()
    df = df[df["Rating"].between(1, 10)]
    df = df[df["year_num"].between(1950, 2025)]
    return df.reset_index(drop=True)


def fit_target_encoders(df):
    """Fit category means on a reference frame for validation/runtime use."""
    encoders = {"global_mean": float(df["Rating"].mean())}
    for column, (encoder_name, _) in CATEGORY_FEATURES.items():
        encoders[encoder_name] = df.groupby(column)["Rating"].mean().to_dict()
    return encoders


def build_rating_feature_frame(df, encoders):
    """Transform rows using encoders fitted on a separate reference frame."""
    global_mean = float(encoders["global_mean"])
    features = pd.DataFrame(
        {
            "price": df["price_num"].astype(float),
            "year": df["year_num"].astype(float),
            "pages": df["pages_num"].astype(float),
            "votes_log": np.log1p(df["Votes"].astype(float)),
        },
        index=df.index,
    )
    for column, (encoder_name, feature_name) in CATEGORY_FEATURES.items():
        features[feature_name] = (
            df[column].map(encoders[encoder_name]).fillna(global_mean).astype(float)
        )
    return features[FEATURE_NAMES]


def build_oof_rating_feature_frame(df, n_splits=5, random_state=42):
    """Build training features whose target means never include their own row."""
    if len(df) < n_splits:
        raise ValueError("OOF target encoding requires at least n_splits rows")

    features = build_rating_feature_frame(df, fit_target_encoders(df))
    positions = np.arange(len(df))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fit_positions, validation_positions in splitter.split(positions):
        fit_df = df.iloc[fit_positions]
        validation_df = df.iloc[validation_positions]
        fold_encoders = fit_target_encoders(fit_df)
        fold_features = build_rating_feature_frame(validation_df, fold_encoders)
        for _, feature_name in CATEGORY_FEATURES.values():
            features.loc[validation_df.index, feature_name] = fold_features[
                feature_name
            ]

    return features[FEATURE_NAMES]


def make_rating_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def train_rating_model(df, test_size=0.2, random_state=42, cv_splits=5):
    """Evaluate honestly, then fit the production model on full OOF features."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_features = build_oof_rating_feature_frame(
        train_df, n_splits=cv_splits, random_state=random_state
    )
    train_encoders = fit_target_encoders(train_df)
    test_features = build_rating_feature_frame(test_df, train_encoders)

    evaluation_model = make_rating_model()
    evaluation_model.fit(train_features.values, train_df["Rating"].values)
    test_predictions = evaluation_model.predict(test_features.values)

    cv_scores = []
    positions = np.arange(len(train_df))
    outer_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for fit_positions, validation_positions in outer_cv.split(positions):
        fold_train = train_df.iloc[fit_positions]
        fold_validation = train_df.iloc[validation_positions]
        fold_features = build_oof_rating_feature_frame(
            fold_train, n_splits=cv_splits - 1, random_state=random_state
        )
        fold_encoders = fit_target_encoders(fold_train)
        validation_features = build_rating_feature_frame(
            fold_validation, fold_encoders
        )
        fold_model = make_rating_model()
        fold_model.fit(fold_features.values, fold_train["Rating"].values)
        fold_predictions = fold_model.predict(validation_features.values)
        cv_scores.append(r2_score(fold_validation["Rating"], fold_predictions))

    full_features = build_oof_rating_feature_frame(
        df, n_splits=cv_splits, random_state=random_state
    )
    full_encoders = fit_target_encoders(df)
    final_model = make_rating_model()
    final_model.fit(full_features.values, df["Rating"].values)

    y_test = test_df["Rating"].values
    author_baseline = (
        test_df["author_clean"]
        .map(train_encoders["author_means"])
        .fillna(train_encoders["global_mean"])
        .values
    )
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "MAE": float(mean_absolute_error(y_test, test_predictions)),
        "R2": float(r2_score(y_test, test_predictions)),
        "CV_R2": float(np.mean(cv_scores)),
        "CV_R2_std": float(np.std(cv_scores)),
        "author_baseline_RMSE": float(
            np.sqrt(mean_squared_error(y_test, author_baseline))
        ),
        "author_baseline_MAE": float(mean_absolute_error(y_test, author_baseline)),
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "encoding": "5-fold OOF target means",
    }

    return {
        "model": final_model,
        "encoders": full_encoders,
        "feature_names": FEATURE_NAMES.copy(),
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "test_predictions": test_predictions,
    }
