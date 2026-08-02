"""Regression tests for the persisted rating predictor runtime wrapper."""

import numpy as np

from src.rating_predictor import (
    RatingPredictorArtifact,
    clean_rating_author,
)


class CapturingModel:
    def __init__(self):
        self.last_row = None

    def predict(self, row):
        self.last_row = row
        return np.array([8.4])


def make_predictor():
    model = CapturingModel()
    predictor = RatingPredictorArtifact(
        {
            "model": model,
            "encoders": {
                "global_mean": 7.5,
                "author_means": {"余华": 9.1},
                "publisher_means": {"人民文学出版社": 8.8},
                "binding_means": {"平装": 8.2, "其他": 7.4},
            },
            "feature_names": [
                "price",
                "year",
                "pages",
                "votes_log",
                "author_mean",
                "publisher_mean",
                "binding_mean",
            ],
            "metrics": {},
        }
    )
    return predictor, model


def test_author_cleanup_preserves_name_and_removes_nationality():
    assert clean_rating_author("余华") == "余华"
    assert clean_rating_author("[美] 雷蒙德·钱德勒") == "雷蒙德·钱德勒"
    assert clean_rating_author("（英）乔治·奥威尔") == "乔治·奥威尔"


def test_runtime_wrapper_uses_known_author_publisher_and_binding_means():
    predictor, model = make_predictor()

    result = predictor.predict(
        price=39.5,
        year=2020,
        pages=300,
        votes=5000,
        author="余华",
        publisher="人民文学出版社",
        binding="平装",
    )

    assert result == 8.4
    assert model.last_row[0, 4] == 9.1
    assert model.last_row[0, 5] == 8.8
    assert model.last_row[0, 6] == 8.2


def test_unknown_categories_fall_back_to_global_mean():
    predictor, model = make_predictor()

    predictor.predict(
        price=39.5,
        year=2020,
        pages=300,
        votes=5000,
        author="未收录作者",
        publisher="未收录出版社",
        binding="线装",
    )

    assert model.last_row[0, 4] == 7.5
    assert model.last_row[0, 5] == 7.5
    assert model.last_row[0, 6] == 7.4
