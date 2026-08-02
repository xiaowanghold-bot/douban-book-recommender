"""推荐引擎公共搜索接口的行为测试。"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendation import BookRecommender  # noqa: E402


def build_small_recommender():
    books = pd.DataFrame(
        [
            {"id": 1, "title": "三体", "rating": 9.0, "votes": 100, "bayesian_score": 8.8},
            {"id": 2, "title": "三体导读", "rating": 8.5, "votes": 80, "bayesian_score": 8.2},
            {"id": 3, "title": "科幻世界", "rating": 8.0, "votes": 60, "bayesian_score": 7.9},
        ]
    )
    documents = ["科幻", "三体", "三体 科幻"]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S+")

    recommender = BookRecommender(mode="semantic")
    recommender.df = books
    recommender.vectorizer = vectorizer
    recommender.tfidf_matrix = vectorizer.fit_transform(documents)
    recommender._tokenize_semantic = lambda text: text
    recommender._rebuild_lookup_indexes()
    return recommender


def test_exact_title_has_priority_over_higher_semantic_similarity():
    recommender = build_small_recommender()

    results = recommender.recommend_by_title("三体", top_n=3)

    assert results.iloc[0]["id"] == 1
    assert results.iloc[0]["title"] == "三体"


def test_title_search_respects_allowed_book_ids():
    recommender = build_small_recommender()

    results = recommender.recommend_by_title("三体", top_n=3, allowed_ids={2, 3})

    assert set(results["id"]) == {2, 3}
    assert 1 not in set(results["id"])


def test_title_search_returns_empty_when_filter_has_no_model_books():
    recommender = build_small_recommender()

    results = recommender.recommend_by_title("三体", allowed_ids={999})

    assert results.empty
