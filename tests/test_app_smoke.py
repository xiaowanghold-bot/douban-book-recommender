"""UI smoke test for douban-book-recommender Streamlit app."""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "🏠 首页",
    "🏆 排行榜",
    "🔍 搜书推荐",
    "🏢 出版社与作者",
    "🔮 评分预测",
    "🧊 新书预测",
    "💡 更多发现",
    "🏷️ 标签浏览",
    "📋 关于项目",
]

@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP_DIR / "main.py"))
    at.run(timeout=60)
    return at

def test_no_exception(app):
    assert not app.exception

def test_all_nav_pages(app):
    for i, page_name in enumerate(PAGES):
        app.sidebar.radio[0].set_value(page_name).run()
        assert not app.exception, f"Page {page_name} caused exception"

def test_rating_prediction_v2(app):
    """Verify v2 predictor uses dict-based means (not LabelEncoder)."""
    app.sidebar.radio[0].set_value(PAGES[4]).run()
    assert not app.exception
    # Direct verification: pkl uses v2 encoders
    import pickle
    with open("data/models/rating_predictor.pkl", "rb") as f:
        data = pickle.load(f)
    assert "author_means" in data["encoders"], "v2: missing author_means dict"
    assert "global_mean" in data["encoders"], "v2: missing global_mean"
    # Verify prediction works on known author
    import numpy as np
    gm = data["encoders"]["global_mean"]
    am = data["encoders"]["author_means"].get("刘慈欣", gm)
    pm = data["encoders"]["publisher_means"].get("重庆出版社", gm)
    bm = data["encoders"]["binding_means"].get("平装", gm)
    features = {"price": 39.5, "year": 2020.0, "pages": 300.0,
                "votes_log": np.log1p(5000),
                "author_mean": float(am), "publisher_mean": float(pm), "binding_mean": float(bm)}
    X = np.array([[features[n] for n in data["feature_names"]]])
    pred = float(data["model"].predict(X)[0])
    assert 2.0 <= pred <= 10.0, f"Prediction {pred} out of [2,10]"

def test_tag_browse_novel(app):
    app.sidebar.radio[0].set_value(PAGES[7]).run()
    assert not app.exception

def test_search_tag_scifi(app):
    app.sidebar.radio[0].set_value(PAGES[2]).run()
    assert not app.exception
