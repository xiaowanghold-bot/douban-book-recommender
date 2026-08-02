"""Runtime helpers for loading and using the rating prediction artifact."""

from pathlib import Path
import pickle
import re

import numpy as np


def clean_rating_author(value):
    """Match the author cleanup used when the rating model is trained."""
    if value is None:
        return "未知"
    text = re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(value)).strip()
    return text[:30] or "未知"


def clean_rating_publisher(value):
    """Match the publisher cleanup used when the rating model is trained."""
    if value is None:
        return "未知"
    text = str(value).strip()
    return text[:20] or "未知"


def clean_rating_binding(value):
    """Map free-form binding text to the categories used by the model."""
    text = str(value)
    if "平装" in text:
        return "平装"
    if "精装" in text:
        return "精装"
    return "其他"


def build_rating_features(
    encoders,
    *,
    price,
    year,
    pages,
    votes,
    author="未知",
    publisher="未知",
    binding="平装",
):
    """Build one runtime feature row using the artifact's target encodings."""
    global_mean = float(encoders.get("global_mean", 8.0))
    author_clean = clean_rating_author(author)
    publisher_clean = clean_rating_publisher(publisher)
    binding_type = clean_rating_binding(binding)

    return {
        "price": float(price),
        "year": float(year),
        "pages": float(pages),
        "votes_log": float(np.log1p(float(votes))),
        "author_mean": float(
            encoders.get("author_means", {}).get(author_clean, global_mean)
        ),
        "publisher_mean": float(
            encoders.get("publisher_means", {}).get(publisher_clean, global_mean)
        ),
        "binding_mean": float(
            encoders.get("binding_means", {}).get(binding_type, global_mean)
        ),
    }


class RatingPredictorArtifact:
    """Small runtime wrapper around the persisted scikit-learn artifact."""

    def __init__(self, data):
        self.model = data["model"]
        self.encoders = data["encoders"]
        self.feature_names = data["feature_names"]
        self.metrics = data.get("metrics", {})

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return None
        with path.open("rb") as file:
            return cls(pickle.load(file))

    def build_feature_row(self, **inputs):
        features = build_rating_features(self.encoders, **inputs)
        return np.array([[features[name] for name in self.feature_names]], dtype=float)

    def predict(self, **inputs):
        feature_row = self.build_feature_row(**inputs)
        return float(self.model.predict(feature_row)[0])

