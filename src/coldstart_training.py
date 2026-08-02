"""Leakage-safe training helpers for the cold-start rating model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "pub_avg_rating",
    "pub_book_count_log",
    "pub_std_rating",
    "author_avg_rating",
    "author_book_count_log",
    "binding_score",
    "pub_year",
    "pages_log",
    "is_translation",
    "is_series",
]


def prepare_coldstart_dataframe(detail):
    """Create the canonical cold-start frame from Books_detail.csv."""
    df = detail.copy()
    df = df[df["crawl_status"] == "success"].copy()
    for column in ["Rating", "Votes", "pages", "pub_year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Rating", "Votes"])
    df = df[df["Rating"].between(1, 10)]
    df = df[df["Votes"] >= 10]
    df["author"] = df["author"].fillna("未知").astype(str)
    df["publisher"] = df["publisher"].fillna("未知").astype(str)
    df["binding"] = df["binding"].fillna("其他").astype(str)
    df["pages"] = df["pages"].fillna(300).astype(int)
    df["pub_year"] = df["pub_year"].fillna(2010).astype(int)
    df["is_translation"] = (
        df["translator"].notna() | df["original_title"].notna()
    ).astype(int)
    df["is_series"] = df["series"].notna().astype(int)
    return df.reset_index(drop=True)


def build_coldstart_stats(df):
    """Fit aggregate features on a reference frame."""
    global_mean = float(df["Rating"].mean())
    global_std = float(df["Rating"].std())
    publisher = df.groupby("publisher").agg(
        pub_avg_rating=("Rating", "mean"),
        pub_book_count=("Rating", "count"),
        pub_std_rating=("Rating", "std"),
    )
    publisher["pub_std_rating"] = publisher["pub_std_rating"].fillna(global_std)
    author = df.groupby("author").agg(
        author_avg_rating=("Rating", "mean"),
        author_book_count=("Rating", "count"),
    )
    binding = df.groupby("binding")["Rating"].mean().to_dict()
    return {
        "publisher": publisher,
        "author": author,
        "binding": binding,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def build_coldstart_feature_frame(df, stats):
    """Transform rows using stats fitted only on a separate reference frame."""
    global_mean = stats["global_mean"]
    global_std = stats["global_std"]
    publisher = stats["publisher"]
    author = stats["author"]

    features = pd.DataFrame(index=df.index)
    features["pub_avg_rating"] = (
        df["publisher"].map(publisher["pub_avg_rating"]).fillna(global_mean)
    )
    pub_count = df["publisher"].map(publisher["pub_book_count"]).fillna(1)
    features["pub_book_count_log"] = np.log1p(pub_count.clip(lower=1))
    features["pub_std_rating"] = (
        df["publisher"].map(publisher["pub_std_rating"]).fillna(global_std)
    )
    features["author_avg_rating"] = (
        df["author"].map(author["author_avg_rating"]).fillna(global_mean)
    )
    author_count = df["author"].map(author["author_book_count"]).fillna(1)
    features["author_book_count_log"] = np.log1p(author_count.clip(lower=1))
    features["binding_score"] = (
        df["binding"].map(stats["binding"]).fillna(global_mean)
    )
    features["pub_year"] = df["pub_year"].clip(1900, 2030).astype(float)
    features["pages_log"] = np.log1p(df["pages"].clip(10, 5000).astype(float))
    features["is_translation"] = df["is_translation"].astype(float)
    features["is_series"] = df["is_series"].astype(float)
    return features[FEATURE_NAMES].astype(float)


def build_oof_coldstart_feature_frame(df, n_splits=5, random_state=42):
    """Build target-derived training features without using each row's target."""
    if len(df) < n_splits:
        raise ValueError("OOF cold-start encoding requires at least n_splits rows")

    features = pd.DataFrame(index=df.index, columns=FEATURE_NAMES, dtype=float)
    positions = np.arange(len(df))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fit_positions, validation_positions in splitter.split(positions):
        fit_df = df.iloc[fit_positions]
        validation_df = df.iloc[validation_positions]
        fold_stats = build_coldstart_stats(fit_df)
        fold_features = build_coldstart_feature_frame(validation_df, fold_stats)
        features.loc[validation_df.index, FEATURE_NAMES] = fold_features
    return features[FEATURE_NAMES]


def make_coldstart_model(*, loss="squared_error", alpha=0.9):
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        loss=loss,
        alpha=alpha,
    )


def evaluate_coldstart_model(df, test_size=0.2, random_state=42, cv_splits=5):
    """Run an independent holdout and nested CV without target leakage."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_features = build_oof_coldstart_feature_frame(
        train_df, n_splits=cv_splits, random_state=random_state
    )
    train_stats = build_coldstart_stats(train_df)
    test_features = build_coldstart_feature_frame(test_df, train_stats)
    model = make_coldstart_model()
    model.fit(train_features.values, train_df["Rating"].values)
    test_predictions = model.predict(test_features.values)

    cv_scores = []
    positions = np.arange(len(train_df))
    outer_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for fit_positions, validation_positions in outer_cv.split(positions):
        fold_train = train_df.iloc[fit_positions]
        fold_validation = train_df.iloc[validation_positions]
        fold_features = build_oof_coldstart_feature_frame(
            fold_train, n_splits=cv_splits - 1, random_state=random_state
        )
        fold_stats = build_coldstart_stats(fold_train)
        validation_features = build_coldstart_feature_frame(
            fold_validation, fold_stats
        )
        fold_model = make_coldstart_model()
        fold_model.fit(fold_features.values, fold_train["Rating"].values)
        fold_predictions = fold_model.predict(validation_features.values)
        cv_scores.append(r2_score(fold_validation["Rating"], fold_predictions))

    y_test = test_df["Rating"].values
    author_baseline = (
        test_df["author"]
        .map(train_stats["author"]["author_avg_rating"])
        .fillna(train_stats["global_mean"])
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
        "author_baseline_R2": float(r2_score(y_test, author_baseline)),
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "encoding": "5-fold OOF aggregate features",
    }
    return {
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "test_predictions": test_predictions,
    }


def train_coldstart_models(df):
    """Evaluate honestly, then fit all production models on full OOF features."""
    evaluation = evaluate_coldstart_model(df)
    full_features = build_oof_coldstart_feature_frame(df)
    full_stats = build_coldstart_stats(df)

    model = make_coldstart_model()
    model.fit(full_features.values, df["Rating"].values)
    model_lower = make_coldstart_model(loss="quantile", alpha=0.05)
    model_lower.fit(full_features.values, df["Rating"].values)
    model_upper = make_coldstart_model(loss="quantile", alpha=0.95)
    model_upper.fit(full_features.values, df["Rating"].values)

    similarity_scaler = StandardScaler()
    similarity_matrix = similarity_scaler.fit_transform(full_features.values)
    return {
        "model": model,
        "model_lower": model_lower,
        "model_upper": model_upper,
        "feature_names": FEATURE_NAMES.copy(),
        "feature_matrix": full_features.values,
        "similarity_matrix": similarity_matrix,
        "similarity_scaler": similarity_scaler,
        "stats_cache": full_stats,
        "metrics": evaluation["metrics"],
    }

