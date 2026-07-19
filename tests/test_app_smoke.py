"""UI smoke test for douban-book-recommender Streamlit app."""
import sys, os
from pathlib import Path

# Ensure project root and app directory are in path
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

def test_rating_prediction_default(app):
    app.sidebar.radio[0].set_value(PAGES[4]).run()
    assert not app.exception

def test_tag_browse_novel(app):
    app.sidebar.radio[0].set_value(PAGES[7]).run()
    assert not app.exception

def test_search_tag_scifi(app):
    app.sidebar.radio[0].set_value(PAGES[2]).run()
    assert not app.exception
