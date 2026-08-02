"""应用数据访问层与首页选书逻辑测试。"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (  # noqa: E402
    get_cover_path,
    get_description,
    get_detail_info,
    load_app_summary,
)
from app_summary import build_app_summary  # noqa: E402
from home_page import select_featured_books  # noqa: E402


def test_get_detail_info_normalizes_author():
    details = pd.DataFrame(
        [
            {
                "ID": 42,
                "author": "['刘慈欣']",
                "publisher": "重庆出版社",
                "pub_year": 2008,
                "pages": 302,
                "price": "23.00元",
                "binding": "平装",
                "isbn": "9787536692930",
            }
        ]
    )

    info = get_detail_info(details, 42)

    assert info["author"] == "刘慈欣"
    assert info["publisher"] == "重庆出版社"
    assert get_detail_info(details, 999) == {}


def test_get_description_accepts_numeric_book_id():
    assert get_description({"42": "测试简介"}, 42) == "测试简介"
    assert get_description({}, 42) == ""


def test_get_cover_path_only_returns_verified_cover(tmp_path):
    cover = tmp_path / "42.jpg"
    cover.write_bytes(b"cover")

    assert get_cover_path(42, {}, tmp_path, {42}) == str(cover)
    assert get_cover_path(42, {}, tmp_path, set()) is None


def test_app_summary_contains_sidebar_and_home_metrics():
    summary = load_app_summary()

    expected_fields = {
        "book_count",
        "average_rating",
        "detail_count",
        "description_count",
        "cover_count",
        "publisher_count",
        "author_count",
        "recommendation_count",
    }
    assert expected_fields <= summary.keys()
    assert summary["book_count"] >= summary["recommendation_count"] > 0


def test_app_summary_matches_generated_data_sources():
    assert load_app_summary() == build_app_summary()


def test_featured_books_prioritize_verified_quality_covers(tmp_path):
    (tmp_path / "2.jpg").write_bytes(b"x" * 12_001)
    (tmp_path / "3.png").write_bytes(b"x" * 12_001)
    books = pd.DataFrame(
        [
            {"id": 1, "title": "无封面", "votes": 20_000, "bayesian_score": 9.9, "rating": 9.9},
            {"id": 2, "title": "已验证", "votes": 20_000, "bayesian_score": 9.5, "rating": 9.5},
            {"id": 3, "title": "其他格式", "votes": 20_000, "bayesian_score": 9.6, "rating": 9.6},
        ]
    )

    selected = select_featured_books(books, tmp_path, {2}, limit=3)

    assert selected["id"].tolist() == [2, 3, 1]
