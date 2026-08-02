"""Streamlit 应用的数据、索引与模型加载层。"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "reports" / "figures"
COVER_DIR = Path(__file__).parent / "covers"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

FALLBACK_VERIFIED_COVERS = {
    1007305,
    10608468,
    1068337,
    11530078,
    1211572,
    1221512,
    1221514,
    1258136,
    1358243,
    1448820,
    1467519,
    1542939,
    1608298,
    1668197,
    1774227,
    1844794,
    1950809,
    2032898,
    25709685,
    25757313,
    25898626,
    25907864,
    25914783,
    26197294,
    26304954,
    26423502,
    26435630,
    26912767,
    27154246,
    3048059,
    3162991,
    4201317,
    4759840,
    6435891,
    10608472,
    10608473,
    1195905,
    1236999,
    1400833,
    1621418,
    1625657,
    25918941,
    26388289,
    26389897,
    26469245,
}


def _load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_recommender():
    """加载推荐引擎，缺失产物时给出明确恢复命令。"""
    from recommendation import BookRecommender

    required = [
        MODELS_DIR / "tfidf_matrix.npz",
        MODELS_DIR / "nn_neighbors.pkl",
        MODELS_DIR / "vectorizer.pkl",
        MODELS_DIR / "books_for_rec.csv",
    ]
    for path in required:
        if not path.exists():
            st.error(f"模型文件缺失：{path.name}。请先运行 src/recommendation.py 生成模型产物。")
            st.stop()
    recommender = BookRecommender()
    recommender._load_artifacts()
    return recommender


@st.cache_data
def load_scored_data():
    path = PROCESSED_DIR / "books_scored.csv"
    if not path.exists():
        st.error("数据文件缺失：books_scored.csv。请先运行 src/scoring.py 生成评分数据。")
        st.stop()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_price_data():
    path = PROCESSED_DIR / "books_with_price.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_data
def load_publisher_stats():
    path = PROCESSED_DIR / "publisher_stats.csv"
    return pd.read_csv(path, encoding="utf-8-sig", index_col=0) if path.exists() else None


@st.cache_data
def load_author_stats():
    path = PROCESSED_DIR / "author_stats.csv"
    return pd.read_csv(path, encoding="utf-8-sig", index_col=0) if path.exists() else None


@st.cache_resource
def load_rating_predictor():
    from rating_predictor import RatingPredictorArtifact

    return RatingPredictorArtifact.load(MODELS_DIR / "rating_predictor.pkl")


@st.cache_data
def load_rating_model_meta():
    return _load_json(MODELS_DIR / "rating_model_meta.json", {})


@st.cache_data
def load_coldstart_model_meta():
    return _load_json(MODELS_DIR / "coldstart_model_meta.json", {})


@st.cache_resource
def load_coldstart_predictor():
    from coldstart_predictor import ColdStartPredictor

    return ColdStartPredictor.load()


@st.cache_data
def load_detail_data():
    path = DATA_DIR / "raw" / "Books_detail.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


@st.cache_data
def load_descriptions():
    return _load_json(PROCESSED_DIR / "book_descriptions.json", {})


@st.cache_data
def load_cover_map():
    return _load_json(PROCESSED_DIR / "book_covers.json", {})


@st.cache_data
def load_verified_covers():
    values = _load_json(PROCESSED_DIR / "verified_covers.json", None)
    return set(values) if values is not None else set(FALLBACK_VERIFIED_COVERS)


@st.cache_resource
def load_tag_index():
    """从真实豆瓣用户标签构建标签到图书 ID 的倒排索引。"""
    all_tags = _load_json(PROCESSED_DIR / "book_tags.json", {})
    tag_to_ids = {}
    for book_id, tags in all_tags.items():
        for tag in tags:
            tag_to_ids.setdefault(tag, set()).add(int(book_id))

    counts_path = PROCESSED_DIR / "tag_counts.csv"
    top_tags = []
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        top_tags = counts.nlargest(30, "count")["tag"].tolist()
    return tag_to_ids, top_tags


@st.cache_resource
def load_genre_index(_df):
    from genre_search import build_genre_search_index

    descriptions_path = PROCESSED_DIR / "book_descriptions.json"
    return build_genre_search_index(_df, str(descriptions_path))


@st.cache_data
def load_app_summary():
    """加载供导航栏和首页使用的轻量统计，避免为展示计数加载完整模型。"""
    return _load_json(PROCESSED_DIR / "app_summary.json", {})


def get_detail_info(detail_df, book_id):
    """从详情数据中读取单本图书的展示字段。"""
    if detail_df is None:
        return {}
    rows = detail_df[detail_df["ID"] == int(book_id)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "author": str(row.get("author", "")).strip("[]").replace("'", ""),
        "publisher": str(row.get("publisher", "")),
        "pub_year": str(row.get("pub_year", "")),
        "pages": str(row.get("pages", "")),
        "price": str(row.get("price", "")),
        "binding": str(row.get("binding", "")),
        "isbn": str(row.get("isbn", "")),
    }


def get_description(descriptions, book_id):
    return descriptions.get(str(int(book_id)), "")


def get_cover_path(book_id, cover_map, cover_dir=COVER_DIR, verified_covers=None):
    """返回已验证封面的本地路径；未验证或缺失时返回 None。"""
    verified_covers = verified_covers or set()
    if int(book_id) not in verified_covers:
        return None
    filename = cover_map.get(str(int(book_id)), "")
    if filename:
        path = cover_dir / filename
        if path.exists():
            return str(path)
    for extension in ["jpg", "png", "webp"]:
        path = cover_dir / f"{int(book_id)}.{extension}"
        if path.exists():
            return str(path)
    return None
