"""Runtime regression tests for the persisted cold-start predictor."""

import numpy as np

from src.coldstart_predictor import ColdStartPredictor


def test_v4_artifact_metrics_and_similarity_are_valid():
    predictor = ColdStartPredictor.load()

    assert predictor.artifact_version == 4
    assert predictor.metrics["encoding"] == "5-fold OOF aggregate features"

    prediction, lower, upper, _, similar = predictor.predict(
        author="未收录作者",
        publisher="未收录出版社",
        pub_year=2025,
        pages=300,
        binding="其他",
        is_translation=False,
        is_series=False,
    )

    similarities = np.array([book["similarity"] for book in similar])
    assert lower <= prediction <= upper
    assert len(similarities) == 5
    assert np.ptp(similarities) > 1e-5
    assert similarities.max() < 0.99
