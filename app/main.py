"""
豆瓣图书评价与推荐系统 - Streamlit Web 应用
江南大学大学生创新训练计划项目
功能：深色模式 | 搜索自动补全+标签筛选 | 图书详情浮窗 | 标签分类浏览 | 图书简介展示 | 评分预测
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, json, pickle, re, base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from recommendation import BookRecommender
from genre_search import build_genre_search_index, search_books_by_genre, GENRE_GROUPS
from coldstart_page import show as show_coldstart

st.set_page_config(
    page_title="豆瓣图书评价与推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
DATA_DIR = BASE_DIR / "data"
COVER_DIR = Path(__file__).parent / "covers"
DESC_FILE = DATA_DIR / "processed" / "book_descriptions.json"
COVER_MAP_FILE = DATA_DIR / "processed" / "book_covers.json"

# ========== 缓存加载 ==========
@st.cache_resource
def load_recommender():
    rec = BookRecommender()
    rec._load_artifacts()
    nn = np.load(str(BASE_DIR / "data" / "models" / "nn_neighbors.npz"))
    rec.nn_distances = nn["distances"]
    rec.nn_indices = nn["indices"]
    return rec

@st.cache_data
def load_scored_data():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "books_scored.csv", encoding="utf-8-sig")

@st.cache_data
def load_price_data():
    p = BASE_DIR / "data" / "processed" / "books_with_price.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_pub_stats():
    p = BASE_DIR / "data" / "processed" / "publisher_stats.csv"
    return pd.read_csv(p, encoding="utf-8-sig", index_col=0) if p.exists() else None

@st.cache_data
def load_author_stats():
    p = BASE_DIR / "data" / "processed" / "author_stats.csv"
    return pd.read_csv(p, encoding="utf-8-sig", index_col=0) if p.exists() else None

@st.cache_resource
def load_predictor():
    p = BASE_DIR / "data" / "models" / "rating_predictor.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    # Wrap dict into a callable object
    class PredictorWrapper:
        def __init__(self, data):
            self.model = data["model"]
            self.encoders = data["encoders"]
            self.feature_names = data["feature_names"]
            self.metrics = data["metrics"]
            # Load detail data for encoder fallback
            import pandas as pd
            det_path = BASE_DIR / "data" / "raw" / "Books_detail.csv"
            self.df = pd.read_csv(det_path, encoding="utf-8-sig") if det_path.exists() else None
        
        def predict(self, price, year, pages, votes, author="未知", publisher="未知", binding="平装"):
            import re, numpy as np
            features = {}
            features["price"] = float(price)
            features["year"] = float(year)
            features["pages"] = float(pages)
            features["votes_log"] = np.log1p(float(votes))
            
            author_clean = re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(author)).strip()[:30]
            publisher_clean = str(publisher).strip()[:20]
            binding_type = "平装" if "平装" in str(binding) else ("精装" if "精装" in str(binding) else "其他")
            
            for col, val in [("author_clean", author_clean),
                             ("publisher_clean", publisher_clean),
                             ("binding_type", binding_type)]:
                encoder = self.encoders[col]
                try:
                    features[col] = float(encoder.transform([val])[0])
                except ValueError:
                    features[col] = 0.0
            
            X = np.array([[features[n] for n in self.feature_names]])
            return float(self.model.predict(X)[0])
    
    return PredictorWrapper(data)

@st.cache_resource
def load_coldstart_predictor():
    from coldstart_predictor import ColdStartPredictor
    return ColdStartPredictor.load()

@st.cache_data
def load_detail_data():
    p = BASE_DIR / "data" / "raw" / "Books_detail.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_descriptions():
    if DESC_FILE.exists():
        with open(DESC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_cover_map():
    if COVER_MAP_FILE.exists():
        with open(COVER_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

rec = load_recommender()

# ===== 流派搜索索引 =====
@st.cache_resource
def load_genre_index(_df):
    return build_genre_search_index(_df, str(DATA_DIR / "processed" / "book_descriptions.json"))

genre_text_index = load_genre_index(df)
df = load_scored_data()
detail_df = load_detail_data()
descriptions = load_descriptions()
cover_map = load_cover_map()

# ========== 工具函数 ==========
def get_detail_info(book_id):
    if detail_df is None:
        return {}
    row = detail_df[detail_df["ID"] == int(book_id)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "author": str(r.get("author", "")).strip("[]").replace("'", ""),
        "publisher": str(r.get("publisher", "")),
        "pub_year": str(r.get("pub_year", "")),
        "pages": str(r.get("pages", "")),
        "price": str(r.get("price", "")),
        "binding": str(r.get("binding", "")),
        "isbn": str(r.get("isbn", "")),
    }

def get_desc(book_id):
    return descriptions.get(str(int(book_id)), "")

# 已验证封面集合（页面爬取，100%准确）
# 动态加载验证封面列表
_vf = BASE_DIR / "data" / "processed" / "verified_covers.json"
if _vf.exists():
    VERIFIED_COVERS = set(json.load(open(_vf, encoding="utf-8")))
else:
    VERIFIED_COVERS = {1007305,10608468,1068337,11530078,1211572,1221512,1221514,1258136,1358243,1448820,1467519,1542939,1608298,1668197,1774227,1844794,1950809,2032898,25709685,25757313,25898626,25907864,25914783,26197294,26304954,26423502,26435630,26912767,27154246,3048059,3162991,4201317,4759840,6435891,10608472,10608473,1195905,1236999,1400833,1621418,1625657,25918941,26388289,26389897,26469245}

def get_cover(book_id):
    """获取封面，仅返回已验证的准确封面"""
    if int(book_id) not in VERIFIED_COVERS:
        return None
    fname = cover_map.get(str(int(book_id)), "")
    if fname:
        full = COVER_DIR / fname
        if full.exists():
            return str(full)
    for ext in ["jpg", "png", "webp"]:
        f = COVER_DIR / "{0}.{1}".format(int(book_id), ext)
        if f.exists():
            return str(f)
    return None
# ========== 深色模式 ==========
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def inject_dark_css():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #1a1a2e; }
        .main .block-container { color: #e0e0e0; }
        h1, h2, h3, h4, h5, h6, p, span, div, label { color: #e0e0e0 !important; }
        .stTextInput>div>div>input, .stSelectbox>div>div { background-color: #2d2d44 !important; color: #e0e0e0 !important; }
        .stat-card { background: #2d2d44 !important; border: 1px solid #3d3d5c !important; }
        .stat-value { color: #fff !important; }
        .stat-label { color: #aaa !important; }
        .stDataFrame { background-color: #2d2d44 !important; }
        section[data-testid="stSidebar"] { background-color: #16213e !important; }
        section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        hr { border-color: #3d3d5c !important; }
        .stProgress > div > div > div { background-color: #667eea !important; }
        .stButton>button { background-color: #3d3d5c !important; color: #e0e0e0 !important; border: 1px solid #555 !important; }
        </style>
        """, unsafe_allow_html=True)

inject_dark_css()

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("# 📚 豆瓣图书评价与推荐系统")
    st.markdown("---")
    
    dm_col1, dm_col2 = st.columns([3, 1])
    with dm_col1:
        st.caption("🌓 显示主题")
    with dm_col2:
        if st.toggle("🌙", value=st.session_state.dark_mode, key="dm_toggle", help="深色模式"):
            if not st.session_state.dark_mode:
                st.session_state.dark_mode = True
                st.rerun()
        else:
            if st.session_state.dark_mode:
                st.session_state.dark_mode = False
                st.rerun()
    
    st.markdown("---")
    
    pages_list = ["🏠 首页", "🏆 排行榜", "🔍 搜书推荐",
                   "🏢 出版社与作者", "🔮 评分预测", "🧊 新书预测", "💡 更多发现",
                   "🏷️ 标签浏览", "📋 关于项目"]
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 首页"
    
    page = st.sidebar.radio("导航菜单", pages_list,
        index=pages_list.index(st.session_state.current_page))
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("收录图书: {0:,} 本".format(len(df)))
    st.sidebar.caption("详细信息: {0:,} 本".format(len(detail_df) if detail_df is not None else 0))
    st.sidebar.caption("图书简介: {0:,} 本".format(len(descriptions)))
    st.sidebar.caption("封面图片: {0:,} 张".format(len(cover_map)))
    st.sidebar.caption("推荐引擎: TF-IDF + Cosine")
    st.sidebar.caption("评分预测: RF MAE=0.40")
    st.sidebar.caption("江南大学 · 大创项目")
    st.sidebar.success("📁 xiaowanghold-bot/douban-book-recommender")
# ======================================================================
#  首页
# ======================================================================
if page == "🏠 首页":
    import plotly.express as px
    import os

    dark_bg = "#1a1a2e" if st.session_state.dark_mode else "#ffffff"
    card_bg = "#2d2d44" if st.session_state.dark_mode else "#ffffff"
    text_color = "#e0e0e0" if st.session_state.dark_mode else "#2c3e50"
    sub_color = "#aaa" if st.session_state.dark_mode else "#888"
    border_color = "#3d3d5c" if st.session_state.dark_mode else "#f0f0f0"

    st.markdown("""
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    .hero-title {
        font-size: 3.2em; font-weight: 900; text-align: center;
        background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite;
        padding: 15px 0 5px 0; letter-spacing: 2px;
    }
    .hero-subtitle {
        text-align: center; font-size: 1.2em; margin-bottom: 30px;
        animation: fadeInUp 0.8s ease;
    }
    .stat-card {
        background: VAR_CARD_BG; border-radius: 16px; padding: 22px 18px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06); transition: all 0.3s ease;
        border: 1px solid VAR_BORDER; animation: fadeInUp 0.6s ease;
    }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(0,0,0,0.12); }
    .stat-icon { font-size: 2.2em; margin-bottom: 8px; animation: float 3s ease-in-out infinite; }
    .stat-value { font-size: 1.8em; font-weight: 800; color: #667eea; }
    .stat-label { font-size: 0.9em; color: VAR_SUB_COLOR; }
    .nav-card {
        background: VAR_CARD_BG; border-radius: 16px; padding: 28px 20px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06); transition: all 0.3s ease;
        border: 1px solid VAR_BORDER; animation: fadeInUp 0.6s ease;
    }
    .nav-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(0,0,0,0.12); }
    .nav-card-icon { font-size: 2.8em; margin-bottom: 10px; animation: float 3s ease-in-out infinite; }
    .nav-card-title { font-size: 1.15em; font-weight: 700; }
    .nav-card-desc { font-size: 0.85em; margin-top: 8px; }
    .cover-placeholder {
        width: 100%; aspect-ratio: 3/4; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; font-size: 2em;
    }
    </style>
    """.replace("VAR_CARD_BG", card_bg).replace("VAR_BORDER", border_color).replace("VAR_SUB_COLOR", sub_color), unsafe_allow_html=True)

    # Hero
    st.markdown('<div class="hero-title">豆瓣图书评价与推荐系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="color:{0};">📖 发现好书 · 智能推荐 · 数据洞察</div>'.format(sub_color), unsafe_allow_html=True)

    # Stats
    stats = [
        ("📕", "{0:,}".format(len(df)), "收录图书", "原始数据 288,824"),
        ("⭐", "8.1", "平均评分", "最高 10.0"),
        ("👥", "{0:,}".format((df["votes"]>=10000).sum()), "评价过万", "过千 {0:,}".format((df["votes"]>=1000).sum())),
        ("🏢", "221", "出版社", "878 位作者"),
        ("📈", "{0:,}".format(len(rec.id_to_idx)), "推荐引擎", "30 近邻/本"),
        ("📝", "{0:,}".format(len(descriptions)), "图书简介", "封面 {0:,}张".format(len(cover_map))),
    ]
    cols = st.columns(len(stats))
    for i, (icon, val, label, sub) in enumerate(stats):
        with cols[i]:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-icon">{0}</div>
                <div class="stat-value">{1}</div>
                <div class="stat-label">{2}</div>
                <div style="font-size:0.7em;color:{3};margin-top:3px;">{4}</div>
            </div>""".format(icon, val, label, sub_color, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ====== Book Cover Showcase (clickable) ======
    st.markdown("### 📖 精选高分图书")
    st.caption("💡 点击「📖 查看」按钮查看图书详细信息")

    # Get ALL books with covers (not just jpg)
    cover_ids_in_dir = set()
    if COVER_DIR.exists():
        for f in COVER_DIR.iterdir():
            if f.suffix.lower() in (".jpg", ".png", ".webp"):
                try:
                    cover_ids_in_dir.add(int(f.stem))
                except ValueError:
                    pass

    df_with_covers = df[df["id"].isin(cover_ids_in_dir) & (df["votes"] >= 100)]
    if len(df_with_covers) < 24:
        # Supplement with top books even without covers
        extra = df[~df["id"].isin(cover_ids_in_dir)].nlargest(24 - len(df_with_covers), "bayesian_score")
        df_with_covers = pd.concat([df_with_covers, extra])
    top_cover_books = df_with_covers.nlargest(60, "bayesian_score")
    # 扩大选池 + 去重同书名 + 封面质量过滤(>12KB防错图)
    # 优先原版页面爬取封面(45张已验证)+CDN优质封面(>12KB)
    _orig_covers = VERIFIED_COVERS
    _pool = df_with_covers.nlargest(200, "bayesian_score")
    _pool = _pool.drop_duplicates(subset="title", keep="first")
    import os as _os_qf
    def _cover_ok(bid):
        p = COVER_DIR / "{}.jpg".format(int(bid))
        return p.exists() and p.stat().st_size > 12000
    _verified = _pool[_pool["id"].isin(_orig_covers) & _pool["id"].apply(_cover_ok)]
    _cdn_ok = _pool[~_pool["id"].isin(_orig_covers) & _pool["id"].apply(_cover_ok)]
    top_cover_books = pd.concat([_verified, _cdn_ok]).head(24)
    if len(top_cover_books) < 24:
        _fallback = _pool[~_pool["id"].isin(top_cover_books["id"])].head(24 - len(top_cover_books))
        top_cover_books = pd.concat([top_cover_books, _fallback])
    # Row-by-row: cards + buttons interleaved
    if "home_detail_bid" not in st.session_state:
        st.session_state.home_detail_bid = None

    card_css = """<style>
    .home-card-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 0; }
    .hc-card { background: VAR_CARD; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: transform 0.2s; }
    .hc-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
    .hc-img { width: 100%; height: 210px; object-fit: cover; display: block; }
    .hc-placeholder { background: linear-gradient(135deg,#667eea,#764ba2); display:flex;align-items:center;justify-content:center;color:white;font-size:3em; }
    .hc-body { padding: 10px 8px 6px 8px; text-align: center; }
    .hc-title { font-weight: 600; font-size: 0.85em; height: 22px; overflow: hidden; line-height: 1.3; }
    .hc-rating { color: #f39c12; font-size: 0.78em; margin-top: 4px; }
    </style>""".replace("VAR_CARD", card_bg)
    st.markdown(card_css, unsafe_allow_html=True)

    for row_idx in range(4):
        # Render this rows cards
        cards_html = '<div class="home-card-grid">'
        for col_idx in range(6):
            i = row_idx * 6 + col_idx
            if i >= len(top_cover_books):
                break
            book = top_cover_books.iloc[i]
            bid = int(book["id"])
            t = str(book["title"])[:10]
            stars = chr(9733) * int(book["rating"]/2) + chr(9734) * (5-int(book["rating"]/2))
            is_sel = (st.session_state.home_detail_bid == bid)
            highlight = 'border: 3px solid #667eea;' if is_sel else ''
            
            img_b64 = ""
            if int(bid) in VERIFIED_COVERS:
                for ext in ["jpg", "png", "webp"]:
                    p = COVER_DIR / "{0}.{1}".format(bid, ext)
                    if p.exists():
                        with open(p, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode()
                        break
            
            if img_b64:
                img_html = '<img src="data:image/jpeg;base64,{0}" class="hc-img">'.format(img_b64)
            else:
                img_html = '<div class="hc-img hc-placeholder">📕</div>'
            
            cards_html += '<div class="hc-card" style="{0}">{1}<div class="hc-body"><div class="hc-title">{2}</div><div class="hc-rating">{3} {4:.1f}</div></div></div>'.format(highlight, img_html, t, stars, book["rating"])
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
        
        # Render this rows buttons directly below
        cols = st.columns(6)
        for col_idx in range(6):
            i = row_idx * 6 + col_idx
            if i >= len(top_cover_books):
                break
            book = top_cover_books.iloc[i]
            bid = int(book["id"])
            with cols[col_idx]:
                is_sel = (st.session_state.home_detail_bid == bid)
                if is_sel:
                    if st.button("🔼 收起", key="hbtn_{0}".format(bid), use_container_width=True):
                        st.session_state.home_detail_bid = None
                        st.rerun()
                else:
                    if st.button("📖 详情", key="hbtn_{0}".format(bid), use_container_width=True):
                        st.session_state.home_detail_bid = bid
                        st.rerun()

    # Full-width detail panel below
    if st.session_state.home_detail_bid is not None:
        bid = st.session_state.home_detail_bid
        sel_book = top_cover_books[top_cover_books["id"] == bid]
        if not sel_book.empty:
            book = sel_book.iloc[0]
            info = get_detail_info(bid)
            desc = get_desc(bid)
            cover = get_cover(bid)
            st.markdown("---")
            st.markdown("### 📖 {0}".format(str(book["title"])))
            dc1, dc2 = st.columns([1, 3])
            with dc1:
                if cover:
                    st.image(cover, width=200)
                else:
                    st.markdown('<div style="width:200px;height:260px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:3em;">📕</div>', unsafe_allow_html=True)
            with dc2:
                st.markdown("⭐ {0:.1f} / 10  |  👥 {1:,} 人评价".format(float(book["rating"]), int(book["votes"])))
                st.markdown("🏅 贝叶斯评分: {0:.4f}".format(float(book.get("bayesian_score", 0))))
                for k in ["author", "publisher", "pub_year", "price", "pages", "binding", "isbn"]:
                    v = info.get(k, "")
                    if v and v != "nan" and v != "None":
                        st.caption("{0}: {1}".format(k, v))
            if desc:
                st.markdown("---")
                st.markdown("**📝 内容简介**")
                st.markdown("> {0}".format(desc[:500]))

    st.markdown("---")

    # Navigation cards
    st.markdown("### 🚀 探索更多功能")
    fc1, fc2, fc3, fc4 = st.columns(4)

    nav_data = [
        (fc1, "🏆", "贝叶斯排行榜", "科学评分排名", "#667eea", "#764ba2", "🏆 排行榜", "前往排行榜", "nav_r"),
        (fc2, "🔍", "智能搜书推荐", "内容相似度匹配", "#f093fb", "#f5576c", "🔍 搜书推荐", "前往搜书", "nav_s"),
        (fc3, "🏢", "出版社与作者", "221社+878位作者", "#4facfe", "#00f2fe", "🏢 出版社与作者", "前往分析", "nav_p"),
        (fc4, "🔮", "评分预测", "MAE=0.40 R²=0.49", "#43e97b", "#38f9d7", "🔮 评分预测", "前往预测", "nav_d"),
    ]
    for col, icon, title, desc, c1, c2, target, btn_text, btn_key in nav_data:
        with col:
            st.markdown("""<div class="nav-card" style="background:linear-gradient(135deg,{0},{1});">
                <div class="nav-card-icon">{2}</div>
                <div class="nav-card-title" style="color:white;">{3}</div>
                <div class="nav-card-desc" style="color:rgba(255,255,255,0.85);">{4}</div>
            </div>""".format(c1, c2, icon, title, desc), unsafe_allow_html=True)
            if st.button(btn_text, key=btn_key, use_container_width=True):
                st.session_state.current_page = target
                st.rerun()

    st.markdown("---")
    st.caption("江南大学 · 大学生创新训练计划项目 | 豆瓣读书公开数据集")

# ======================================================================
#  排行榜
# ======================================================================
elif page == "🏆 排行榜":
    st.title("🏆 贝叶斯加权评分排行榜")
    st.markdown("*IMDb式贝叶斯平均算法 — 平衡评分高低与评价人数*")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        min_votes = st.slider("最少评价人数", 0, 10000, 50, 100, key="lb_min_votes")
    with c2:
        top_n = st.slider("显示数量", 10, 100, 50, 10, key="lb_top_n")
    with c3:
        st.caption("💡 贝叶斯评分 = (v/(v+m))×R + (m/(v+m))×C")
        st.caption("C=全库均值 | m=中位数评价人数")

    top = df[df["votes"] >= min_votes].nlargest(top_n, "bayesian_score")

    tab1, tab2 = st.tabs(["📊 可视化排名", "📋 数据表格"])

    with tab1:
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np

        top_disp = top.head(20).copy()
        top_disp["short_title"] = top_disp["title"].str[:20]
        fig_bar = px.bar(
            top_disp.iloc[::-1],
            x="bayesian_score", y="short_title",
            orientation="h",
            title=f"贝叶斯加权评分 Top {min(20, len(top_disp))}（最少{min_votes}人评价）",
            color="bayesian_score",
            color_continuous_scale="RdYlGn",
            text=top_disp["bayesian_score"].apply(lambda x: f"{x:.3f}")
        )
        fig_bar.update_traces(textposition="outside", textfont_size=11)
        fig_bar.update_layout(
            font_family="Microsoft YaHei",
            height=500,
            xaxis_title="贝叶斯评分", yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=10, r=80, t=40, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            C = df["rating"].mean()
            fig_hist = px.histogram(
                df, x="rating", nbins=50,
                title="全库评分分布与均值 C",
                color_discrete_sequence=["#667eea"],
                opacity=0.7
            )
            fig_hist.add_vline(x=C, line_dash="dash", line_color="red", line_width=2,
                               annotation_text=f"C={C:.2f} (全局均值)")
            fig_hist.update_layout(
                font_family="Microsoft YaHei",
                height=350,
                xaxis_title="评分", yaxis_title="图书数量",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig_scatter = px.scatter(
                sample, x="votes", y="rating",
                title="评分 vs 评价人数（贝叶斯修正）",
                color="bayesian_score" if "bayesian_score" in sample.columns else "rating",
                color_continuous_scale="RdYlGn",
                opacity=0.5, size_max=8,
                log_x=True
            )
            top_sample = top.head(10)
            for _, r in top_sample.iterrows():
                fig_scatter.add_annotation(
                    x=np.log10(r["votes"]), y=r["rating"],
                    text=str(r["title"])[:8],
                    showarrow=True, arrowhead=1, font_size=8
                )
            fig_scatter.update_layout(
                font_family="Microsoft YaHei",
                height=350,
                xaxis_title="评价人数 (log)", yaxis_title="评分",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("全库均值 C", f"{C:.3f}")
        sc2.metric("Top10 平均", f"{top.head(10)['rating'].mean():.1f}")
        sc3.metric("Top10 评价中位数", f"{int(top.head(10)['votes'].median()):,}")
        sc4.metric("符合条件图书", f"{len(top):,} 本")

    with tab2:
        disp = top[["title", "rating", "votes", "bayesian_score"]].copy()
        disp.columns = ["书名", "评分", "评价人数", "贝叶斯评分"]
        disp.index = range(1, len(disp) + 1)
        st.dataframe(
            disp.style.format({"评分": "{:.1f}", "贝叶斯评分": "{:.4f}", "评价人数": "{:,}"})
            .background_gradient(subset=["贝叶斯评分"], cmap="YlOrRd"),
            use_container_width=True, height=600,
        )
        csv = disp.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 下载排行榜 CSV", csv, "book_ranking.csv", "text/csv")
elif page == "🔍 搜书推荐":
    st.title("🔍 搜书 & 智能推荐")

    # Tag filter + search
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        quick_tags = ["全部", "小说", "文学", "历史", "哲学", "科幻", "推理", "爱情",
                      "武侠", "心理", "经济", "漫画", "诗歌", "传记", "散文", "悬疑"]
        tag_sel = st.selectbox("🏷️ 标签筛选", quick_tags, key="search_tag")
    with sc2:
        hint = "输入书名关键词..." if tag_sel == "全部" else "筛选「{0}」类图书，也可输入关键词".format(tag_sel)
        st.caption("💡 {0}".format(hint))

    search_term = st.text_input("🔍 输入书名或关键词",
                                placeholder="例如：三体、活着、百年孤独...",
                                key="search_box")
    if tag_sel != "全部" and not search_term:
        search_term = tag_sel

    if search_term:
        with st.spinner("搜索中..."):
            results = rec.recommend_by_title(search_term, top_n=20)

        if results.empty:
            st.warning("未找到相关图书，请尝试其他关键词")
        else:
            st.caption("🔍 找到 {0} 本匹配图书：".format(len(results)))
            match_cols = st.columns(4)
            for i, (_, m) in enumerate(results.iterrows()):
                ci = i % 4
                with match_cols[ci]:
                    cover = get_cover(m["id"])
                    if cover:
                        st.image(cover, width=110)
                    btn_label = "{0} ⭐{1:.1f}".format(str(m["title"])[:22], m["rating"])
                    if st.button(btn_label, key="suggest_{0}".format(m["id"]), use_container_width=True,
                                 help="{0:,}人评价".format(int(m["votes"]))):
                        st.session_state.selected_book_id = int(m["id"])
                        st.session_state.selected_book_title = m["title"]
                        st.session_state.selected_book_rating = m["rating"]
                        st.session_state.selected_book_votes = m["votes"]
                        st.rerun()

            # Show selected book detail + recommendations
            if "selected_book_id" in st.session_state and st.session_state.selected_book_id:
                bid = st.session_state.selected_book_id
                st.markdown("---")
                st.markdown("### 📖 {0}".format(st.session_state.get("selected_book_title", "")))

                # Detail info
                info = get_detail_info(bid)
                desc = get_desc(bid)
                cover = get_cover(bid)
                dc1, dc2 = st.columns([1, 3])
                with dc1:
                    if cover:
                        st.image(cover, width=200)
                with dc2:
                    st.markdown("⭐ {0:.1f} | 👥 {1:,}人评价".format(
                        st.session_state.get("selected_book_rating", 0),
                        int(st.session_state.get("selected_book_votes", 0))))
                    for k in ["author", "publisher", "pub_year", "price", "pages", "binding", "isbn"]:
                        v = info.get(k, "")
                        if v and v != "nan":
                            st.caption("{0}: {1}".format(k, v))
                if desc:
                    st.markdown("**📝 简介**: {0}".format(desc[:400]))
                if st.button("❌ 关闭详情", key="close_search_detail"):
                    st.session_state.selected_book_id = None
                    st.rerun()

                st.markdown("---")
                # Recommendations
                t1, t2 = st.tabs(["📚 内容推荐", "🔀 混合推荐"])
                with t1:
                    recs = rec.recommend_by_id(bid, top_n=10)
                    for j, (_, r) in enumerate(recs.iterrows()):
                        st.markdown("**{0}. {1}** ⭐{2:.1f} 📊{3:,}人 `{4:.2%}`".format(
                            j+1, r["title"], r["rating"], int(r["votes"]), r["similarity"]))
                        st.progress(float(r["similarity"]))
                with t2:
                    hyb = rec.hybrid_recommend(bid, top_n=10)
                    for j, (_, r) in enumerate(hyb.iterrows()):
                        st.markdown("**{0}. {1}** ⭐{2:.1f} 📊{3:,}人 `{4:.4f}`".format(
                            j+1, r["title"], r["rating"], int(r["votes"]), r["hybrid_score"]))
                        st.progress(float(r["hybrid_score"]))

                # Export
                st.markdown("---")
                st.caption("📥 导出推荐结果")
                ec1, ec2 = st.columns(2)
                with ec1:
                    try:
                        recs_csv = recs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button("📥 内容推荐 CSV", recs_csv, "content_recs.csv", "text/csv", key="dl_c", use_container_width=True)
                    except:
                        pass
                with ec2:
                    try:
                        hyb_csv = hyb.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button("📥 混合推荐 CSV", hyb_csv, "hybrid_recs.csv", "text/csv", key="dl_h", use_container_width=True)
                    except:
                        pass

# ======================================================================
#  出版社与作者
# ======================================================================
elif page == "🏢 出版社与作者":
    st.title("🏢 出版社与作者分析")
    st.markdown("*基于爬虫获取的 6,575 本高分图书详细信息*")

    pub_stats = load_pub_stats()
    if pub_stats is not None:
        st.markdown("### 📚 出版社综合评价")
        c1, c2 = st.columns(2)
        c1.metric("出版社总数", len(pub_stats))
        c2.metric("平均每社图书", "{0:.1f} 本".format(pub_stats["book_count"].mean()))
        tp = pub_stats.head(15)[["book_count", "avg_rating", "pub_score"]]
        tp.columns = ["图书数量", "平均评分", "综合评分"]
        tp.index = ["{0}. {1}".format(i+1, n) for i, n in enumerate(tp.index)]
        st.dataframe(
            tp.style.format({"平均评分": "{:.2f}", "综合评分": "{:.4f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["综合评分"], cmap="YlOrRd"),
            use_container_width=True,
        )
        if (FIG_DIR / "10_publisher_matrix.png").exists():
            st.image(str(FIG_DIR / "10_publisher_matrix.png"), use_container_width=True)

    auth_stats = load_author_stats()
    if auth_stats is not None:
        st.markdown("---")
        st.markdown("### ✍️ 作者影响力")
        c1, c2, c3 = st.columns(3)
        c1.metric("作者总数", len(auth_stats))
        c2.metric("中国作者", int((auth_stats["nationality"] == "中国").sum()))
        c3.metric("外国作者", int((auth_stats["nationality"] != "中国").sum()))
        ta = auth_stats.head(15)[["book_count", "nationality", "avg_rating", "influence"]]
        ta.columns = ["图书数量", "国籍", "平均评分", "影响力"]
        ta.index = ["{0}. {1}".format(i+1, n) for i, n in enumerate(ta.index)]
        st.dataframe(
            ta.style.format({"平均评分": "{:.2f}", "影响力": "{:.1f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["影响力"], cmap="YlOrRd"),
            use_container_width=True,
        )
        if (FIG_DIR / "11_author_influence.png").exists():
            st.image(str(FIG_DIR / "11_author_influence.png"), use_container_width=True)

    if (FIG_DIR / "12_year_trend.png").exists():
        st.markdown("### 📅 出版年份趋势")
        st.image(str(FIG_DIR / "12_year_trend.png"), use_container_width=True)

# ======================================================================
#  评分预测 (REFORMED: prediction-first layout)
# ======================================================================
elif page == "🔮 评分预测":
    st.title("🔮 图书评分预测")
    st.markdown("*输入图书信息，AI 模型预测豆瓣评分*")

    predictor = load_predictor()

    TOP_PUBLISHERS = [
        "上海文艺出版社", "人民文学出版社", "上海出版公司", "新星出版社",
        "吉林出版社", "北京师范大学出版社", "江苏文艺出版社", "中信出版社",
        "生活·读书·新知三联书店", "广西师范大学出版社", "作家出版社",
        "上海译文出版社", "中华书局", "译林出版社", "南海出版公司",
        "北京大学出版社", "商务印书馆", "重庆大学出版社", "上海人民出版社", "浙江文艺出版社"
    ]
    TOP_AUTHORS = [
        "东野圭吾", "村上春树", "金庸", "三毛", "王小波", "鲁迅",
        "阿加莎·克里斯蒂", "莫言", "张爱玲", "余华", "钱钟书",
        "严歌苓", "韩寒", "刘慈欣", "太宰治", "桐华", "杨绛",
        "马尔克斯", "乔治·奥威尔"
    ]

    # Main layout: input form + prediction result side by side
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("### 📝 输入图书信息")

        pub_options = TOP_PUBLISHERS + ["✏️ 其他（手动输入）"]
        pub_choice = st.selectbox("出版社（热门推荐）", pub_options, index=1, key="pred_pub")
        if pub_choice == "✏️ 其他（手动输入）":
            publisher = st.text_input("请输入出版社名称", placeholder="例如：机械工业出版社", key="pred_pub_custom")
        else:
            publisher = pub_choice
            st.caption("已选：{0}".format(publisher))

        author_options = TOP_AUTHORS + ["✏️ 其他（手动输入）"]
        author_choice = st.selectbox("作者（热门推荐）", author_options, index=9, key="pred_author")
        if author_choice == "✏️ 其他（手动输入）":
            author = st.text_input("请输入作者名称", placeholder="例如：陈忠实", key="pred_author_custom")
        else:
            author = author_choice
            st.caption("已选：{0}".format(author))

        price = st.number_input("定价（元）", min_value=0.0, max_value=999.0, value=39.5, step=0.5, key="pred_price")
        year = st.number_input("出版年份", min_value=1900, max_value=2026, value=2014, step=1, key="pred_year")
        pages = st.number_input("页数", min_value=10, max_value=5000, value=300, step=10, key="pred_pages")
        votes = st.number_input("评价人数（预估）", min_value=0, max_value=5000000, value=50000, step=1000, key="pred_votes")

        binding_choice = st.selectbox("装帧", ["平装", "精装", "其他"], key="pred_binding")

        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True, key="pred_btn")

    with rc2:
        st.markdown("### 🎯 预测结果")
        if predict_btn:
            if predictor is not None:
                with st.spinner("模型预测中..."):
                    pred_score = predictor.predict(
                        price=price, year=year, pages=pages, votes=votes,
                        author=author, publisher=publisher, binding=binding_choice
                    )
                if pred_score is not None:
                    st.markdown("""
                    <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;color:white;">
                        <div style="font-size:1.2em;margin-bottom:10px;">预测豆瓣评分</div>
                        <div style="font-size:4em;font-weight:900;">{0:.1f}</div>
                        <div style="font-size:1em;opacity:0.8;">满分 10.0</div>
                    </div>
                    """.format(pred_score), unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**📋 预测详情**")
                    st.markdown("- 作者: {0}".format(author))
                    st.markdown("- 出版社: {0}".format(publisher))
                    st.markdown("- 定价: {0}元 | 年份: {1} | 页数: {2}".format(price, year, pages))
                    st.markdown("- 装帧: {0} | 评价人数: {1:,}".format(binding_choice, int(votes)))
                else:
                    st.error("模型预测失败，请检查输入")
            else:
                st.warning("评分预测模型未加载，请先运行 src/enhancements.py 训练模型")

    # Model info at bottom (less prominent)
    st.markdown("---")
    with st.expander("🧠 模型信息（点击展开）", expanded=False):
        st.markdown("""
        **RandomForest 回归模型**
        - MAE: **0.40** (平均预测误差)
        - R²: **0.49** (拟合度)
        - 5折交叉验证 R²: **0.43**
        - 7 维特征：价格 / 年份 / 页数 / 评价人数 + 作者 / 出版社 / 装帧
        """)

# ======================================================================
#  更多发现
# ======================================================================
elif page == "💡 更多发现":
    st.title("💡 更多发现")
    st.markdown("*词云、价格分析、趋势洞察*")

    if (FIG_DIR / "13_wordcloud.png").exists():
        st.markdown("### ☁️ 高分图书书名词云")
        st.image(str(FIG_DIR / "13_wordcloud.png"), use_container_width=True)

    st.markdown("---")
    if (FIG_DIR / "14_price_analysis.png").exists():
        st.markdown("### 💰 价格分析")
        st.image(str(FIG_DIR / "14_price_analysis.png"), use_container_width=True)

    price_df = load_price_data()
    if price_df is not None:
        st.markdown("### 🏷️ 高性价比图书 (评分>=9, <=50元)")
        vb = price_df[(price_df["Rating"] >= 9) & (price_df["price_num"] <= 50)]
        vb = vb.nlargest(10, "Votes")[["Title", "Rating", "price_num", "author", "Votes"]]
        vb.columns = ["书名", "评分", "价格(元)", "作者", "评价人数"]
        vb.index = range(1, len(vb) + 1)
        st.dataframe(
            vb.style.format({"评分": "{:.1f}", "价格(元)": "{:.1f}", "评价人数": "{:,}"}),
            use_container_width=True,
        )

# ======================================================================
#  标签浏览
# ======================================================================

elif page == "🧊 新书预测":
    csp = load_coldstart_predictor()
    show_coldstart(csp)

elif page == "🏷️ 标签浏览":
    st.title("🏷️ 标签分类浏览")
    st.markdown("*通过关键词匹配 + 评分排序的流派图书探索*")

    # ===== 流派分组标签页 =====
    group_names = list(GENRE_GROUPS.keys())
    tabs = st.tabs(group_names)

    for tab, group_name in zip(tabs, group_names):
        with tab:
            genres = GENRE_GROUPS[group_name]
            cols = st.columns(min(len(genres), 5))
            for i, genre in enumerate(genres):
                with cols[i % 5]:
                    if st.button(
                        genre,
                        key=f"genre_btn_{genre}",
                        use_container_width=True,
                        help=f"浏览{genre}类图书"
                    ):
                        st.session_state["selected_genre"] = genre
                        st.session_state.pop("selected_book_id", None)

    # ===== 搜索结果展示 =====
    if "selected_genre" in st.session_state and st.session_state["selected_genre"]:
        sg = st.session_state["selected_genre"]
        st.markdown("---")
        st.markdown(f"### 📚 「{sg}」相关图书")

        with st.spinner(f"正在搜索「{sg}」类图书..."):
            genre_results = search_books_by_genre(
                sg, df, genre_text_index,
                top_n=30, min_votes=30
            )

        if genre_results.empty:
            st.warning(f"未找到「{sg}」相关的图书，请尝试其他流派")
        else:
            st.success(f"找到 **{len(genre_results)}** 本「{sg}」类图书")
            
            # 6列网格展示
            tag_cols = st.columns(6)
            for bi, (_, tb) in enumerate(genre_results.iterrows()):
                ci = bi % 6
                with tag_cols[ci]:
                    cover = get_cover(tb["id"])
                    if cover:
                        st.image(cover, width=90)
                    st.caption(f"{str(tb['title'])[:16]} ⭐{tb['rating']:.1f}")
                    def _on_click(bid=int(tb["id"]), btitle=tb["title"], brating=tb["rating"], bvotes=tb["votes"]):
                        st.session_state.selected_book_id = bid
                        st.session_state.selected_book_title = btitle
                        st.session_state.selected_book_rating = brating
                        st.session_state.selected_book_votes = int(bvotes)
                    st.button("📖", key=f"genrebk_{tb['id']}", help="查看详情",
                              on_click=_on_click)

            # 图书详情面板
            if "selected_book_id" in st.session_state and st.session_state.selected_book_id:
                st.markdown("---")
                bid2 = st.session_state.selected_book_id
                info2 = get_detail_info(bid2)
                desc2 = get_desc(bid2)
                cov2 = get_cover(bid2)
                dc1, dc2 = st.columns([1, 3])
                with dc1:
                    if cov2:
                        st.image(cov2, width=120)
                with dc2:
                    st.markdown(f"**{st.session_state.get('selected_book_title', '')}**")
                    st.caption(f"⭐{st.session_state.get('selected_book_rating', 0):.1f} | {int(st.session_state.get('selected_book_votes', 0)):,}人")
                    for k in ["author", "publisher", "pub_year", "price", "isbn"]:
                        if info2.get(k) and info2[k] != "nan":
                            st.caption(f"{k}: {info2[k]}")
                if desc2:
                    st.markdown(f"> {desc2[:300]}")
                if st.button("❌ 关闭详情", key="close_gd_genre"):
                    st.session_state.pop("selected_book_id", None)
                    st.rerun()

        # 关闭标签
        if st.button("🔄 返回流派选择", key="back_genre"):
            st.session_state.pop("selected_genre", None)
            st.session_state.pop("selected_book_id", None)
            st.rerun()

    st.markdown("---")
    st.caption("💡 提示：流派搜索通过匹配书名与简介中的关键词来查找相关图书，结果按匹配度和评分综合排序。")

elif page == "📋 关于项目":
    st.title("📋 关于项目")
    st.markdown("""
## 豆瓣图书评价与推荐系统
**江南大学大学生创新训练计划项目**

### 技术方案
- **贝叶斯加权评分**：IMDb 式算法消除评价人数偏差
- **内容推荐引擎**：字符级 N-gram + TF-IDF + 余弦相似度，去重同书名
- **出版社/作者分析**：221 家出版社、878 位作者综合评价矩阵
- **评分预测**：RandomForest 回归，7 维特征，MAE = 0.40

### 数据来源
- 豆瓣读书公开数据集 (yuzhounh/Douban-books-2020)
- 288,824 本基础数据 + 6,575 本爬虫详细信息
- 481 个豆列 + 897 个标签

### 技术栈
Python 3.12 · Streamlit · pandas · scikit-learn · matplotlib · jieba · wordcloud
    """)
