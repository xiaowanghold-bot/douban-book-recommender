"""应用首页及精选图书展示。"""

import base64
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from components import render_cover


SUPPORTED_COVER_SUFFIXES = (".jpg", ".png", ".webp")
MIN_COVER_BYTES = 12_000


def _cover_path(book_id, cover_dir):
    for suffix in SUPPORTED_COVER_SUFFIXES:
        path = cover_dir / f"{int(book_id)}{suffix}"
        if path.exists():
            return path
    return None


def _has_quality_cover(book_id, cover_dir):
    path = _cover_path(book_id, cover_dir)
    return path is not None and path.stat().st_size > MIN_COVER_BYTES


def select_featured_books(df, cover_dir, verified_covers, limit=24):
    """优先选择有高质量封面的高分图书，并按书名去重。"""
    cover_ids = set()
    if cover_dir.exists():
        for path in cover_dir.iterdir():
            if path.suffix.lower() in SUPPORTED_COVER_SUFFIXES:
                try:
                    cover_ids.add(int(path.stem))
                except ValueError:
                    continue

    candidates = df[df["id"].isin(cover_ids) & (df["votes"] >= 5000)]
    if len(candidates) < limit:
        extras = df[~df["id"].isin(cover_ids)].nlargest(
            limit - len(candidates), "bayesian_score"
        )
        candidates = pd.concat([candidates, extras])

    pool = candidates.nlargest(200, "bayesian_score").drop_duplicates(
        subset="title", keep="first"
    )
    quality_mask = pool["id"].apply(lambda book_id: _has_quality_cover(book_id, cover_dir))
    verified = pool[pool["id"].isin(verified_covers) & quality_mask]
    other_quality = pool[~pool["id"].isin(verified_covers) & quality_mask]
    selected = pd.concat([verified, other_quality]).head(limit)
    if len(selected) < limit:
        fallback = pool[~pool["id"].isin(selected["id"])].head(limit - len(selected))
        selected = pd.concat([selected, fallback])
    return selected


@st.cache_data(show_spinner=False)
def _thumbnail_data_uri(path_text, modified_ns):
    """生成卡片缩略图；文件修改时间作为缓存失效键。"""
    del modified_ns
    try:
        with Image.open(path_text) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((240, 320), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=78, optimize=True, progressive=True)
    except (OSError, ValueError):
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _cover_data_uri(book_id, cover_dir, verified_covers):
    if int(book_id) not in verified_covers:
        return None
    path = _cover_path(book_id, cover_dir)
    if path is None:
        return None
    return _thumbnail_data_uri(str(path), path.stat().st_mtime_ns)


def _inject_home_css(card_bg, border_color, sub_color):
    st.markdown(
        """
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
        @media (max-width: 1100px) {
            .st-key-home_stats [data-testid="stHorizontalBlock"],
            .st-key-home_navigation [data-testid="stHorizontalBlock"],
            .st-key-home_featured_grid [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            .stApp .st-key-home_stats [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
            .stApp .st-key-home_featured_grid [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(33.333% - 1rem) !important;
                min-width: calc(33.333% - 1rem) !important;
                width: calc(33.333% - 1rem) !important;
                max-width: calc(33.333% - 1rem) !important;
            }
            .stApp .st-key-home_navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 1rem) !important;
                min-width: calc(50% - 1rem) !important;
                width: calc(50% - 1rem) !important;
                max-width: calc(50% - 1rem) !important;
            }
        }
        @media (max-width: 680px) {
            .hero-title { font-size: clamp(2rem, 10vw, 3.2rem); line-height: 1.15; }
            .hero-subtitle { font-size: 1rem; margin-bottom: 20px; }
            .stat-card { padding: 16px 10px; }
            .stat-icon { font-size: 1.7em; }
            .stat-value { font-size: 1.35em; }
            .stApp .st-key-home_stats [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
            .stApp .st-key-home_featured_grid [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                min-width: calc(50% - 0.75rem) !important;
                width: calc(50% - 0.75rem) !important;
                max-width: calc(50% - 0.75rem) !important;
            }
            .stApp .st-key-home_navigation [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
                max-width: 100% !important;
            }
        }
        </style>
        """.replace("VAR_CARD_BG", card_bg)
        .replace("VAR_BORDER", border_color)
        .replace("VAR_SUB_COLOR", sub_color),
        unsafe_allow_html=True,
    )


def _render_stats(summary, sub_color):
    stats = [
        ("📕", "{0:,}".format(summary.get("book_count", 0)), "收录图书", "原始数据 288,824"),
        ("⭐", "{0:.1f}".format(summary.get("average_rating", 0)), "平均评分", "最高 10.0"),
        (
            "👥",
            "{0:,}".format(summary.get("votes_10000_count", 0)),
            "评价过万",
            "过千 {0:,}".format(summary.get("votes_1000_count", 0)),
        ),
        (
            "🏢",
            str(summary.get("publisher_count", "?")),
            "出版社",
            f"{summary.get('author_count', '?')} 位作者",
        ),
        ("📈", "{0:,}".format(summary.get("recommendation_count", 0)), "推荐引擎", "30 近邻/本"),
        (
            "📝",
            "{0:,}".format(summary.get("description_count", 0)),
            "图书简介",
            "封面 {0:,}张".format(summary.get("cover_count", 0)),
        ),
    ]
    with st.container(key="home_stats"):
        cols = st.columns(len(stats))
        for col, (icon, value, label, sublabel) in zip(cols, stats):
            with col:
                st.markdown(
                    """
                    <div class="stat-card">
                        <div class="stat-icon">{0}</div>
                        <div class="stat-value">{1}</div>
                        <div class="stat-label">{2}</div>
                        <div style="font-size:0.7em;color:{3};margin-top:3px;">{4}</div>
                    </div>
                    """.format(icon, value, label, sub_color, sublabel),
                    unsafe_allow_html=True,
                )


def _render_book_grid(books, cover_dir, verified_covers, card_bg):
    if "home_detail_bid" not in st.session_state:
        st.session_state.home_detail_bid = None

    st.markdown(
        """<style>
        .hc-card { background: VAR_CARD; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: transform 0.2s; }
        .hc-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
        .hc-img { width: 100%; height: 210px; object-fit: cover; display: block; }
        .hc-placeholder { background: linear-gradient(135deg,#667eea,#764ba2); display:flex;align-items:center;justify-content:center;color:white;font-size:3em; }
        .hc-body { padding: 10px 8px 6px 8px; text-align: center; }
        .hc-title { font-weight: 600; font-size: 0.85em; height: 22px; overflow: hidden; line-height: 1.3; }
        .hc-rating { color: #f39c12; font-size: 0.78em; margin-top: 4px; }
        @media (max-width: 680px) { .hc-img { height: 190px; } }
        </style>""".replace("VAR_CARD", card_bg),
        unsafe_allow_html=True,
    )

    with st.container(key="home_featured_grid"):
        for row_start in range(0, len(books), 6):
            row_books = books.iloc[row_start : row_start + 6]
            cols = st.columns(6)
            for position, (col, (_, book)) in enumerate(
                zip(cols, row_books.iterrows()), start=row_start
            ):
                book_id = int(book["id"])
                title = escape(str(book["title"])[:16])
                rating = float(book["rating"])
                stars = chr(9733) * round(rating / 2) + chr(9734) * (5 - round(rating / 2))
                selected = st.session_state.home_detail_bid == book_id
                highlight = "border: 3px solid #667eea;" if selected else ""
                data_uri = _cover_data_uri(book_id, cover_dir, verified_covers)
                loading = "eager" if position < 6 else "lazy"
                if data_uri:
                    image_html = (
                        f'<img src="{data_uri}" class="hc-img" alt="{title}" '
                        f'loading="{loading}">'
                    )
                else:
                    image_html = '<div class="hc-img hc-placeholder">📕</div>'
                with col:
                    st.markdown(
                        (
                            '<div class="hc-card" style="{0}">{1}<div class="hc-body">'
                            '<div class="hc-title" title="{2}">{2}</div>'
                            '<div class="hc-rating">{3} {4:.1f}</div></div></div>'
                        ).format(highlight, image_html, title, stars, rating),
                        unsafe_allow_html=True,
                    )
                    label = "🔼 收起" if selected else "📖 详情"
                    if st.button(label, key=f"hbtn_{book_id}", width="stretch"):
                        st.session_state.home_detail_bid = None if selected else book_id
                        st.rerun()


def _render_selected_book(
    books,
    cover_map,
    cover_dir,
    verified_covers,
    get_detail_info,
    get_desc,
):
    book_id = st.session_state.home_detail_bid
    if book_id is None:
        return
    selected = books[books["id"] == book_id]
    if selected.empty:
        return

    book = selected.iloc[0]
    info = get_detail_info(book_id)
    desc = get_desc(book_id)
    st.markdown("---")
    st.markdown("### 📖 {0}".format(str(book["title"])))
    dc1, dc2 = st.columns([1, 3])
    with dc1:
        render_cover(book_id, cover_map, cover_dir, verified_covers, width=200)
    with dc2:
        st.markdown(
            "⭐ {0:.1f} / 10  |  👥 {1:,} 人评价".format(
                float(book["rating"]), int(book["votes"])
            )
        )
        st.markdown("🏅 贝叶斯评分: {0:.4f}".format(float(book.get("bayesian_score", 0))))
        for key in ["author", "publisher", "pub_year", "price", "pages", "binding", "isbn"]:
            value = info.get(key, "")
            if value and value not in {"nan", "None"}:
                st.caption("{0}: {1}".format(key, value))
    if desc:
        st.markdown("---")
        st.markdown("**📝 内容简介**")
        st.markdown("> {0}".format(desc[:500]))


def _render_navigation(summary, rating_metrics):
    st.markdown("### 🚀 探索更多功能")
    publisher_count = summary.get("publisher_count", "?")
    author_count = summary.get("author_count", "?")
    with st.container(key="home_navigation"):
        columns = st.columns(4)
        nav_data = [
            (columns[0], "🏆", "贝叶斯排行榜", "科学评分排名", "#667eea", "#764ba2", "🏆 排行榜", "前往排行榜", "nav_r"),
            (columns[1], "🔍", "智能搜书推荐", "内容相似度匹配", "#f093fb", "#f5576c", "🔍 搜书推荐", "前往搜书", "nav_s"),
            (columns[2], "🏢", "出版社与作者", f"{publisher_count}社+{author_count}位作者", "#4facfe", "#00f2fe", "🏢 出版社与作者", "前往分析", "nav_p"),
            (columns[3], "🔮", "评分预测", "R²={0:.2f}".format(rating_metrics.get("R2", 0.52)), "#43e97b", "#38f9d7", "🔮 评分预测", "前往预测", "nav_d"),
        ]
        for col, icon, title, description, color1, color2, target, button_text, button_key in nav_data:
            with col:
                st.markdown(
                    """<div class="nav-card" style="background:linear-gradient(135deg,{0},{1});">
                        <div class="nav-card-icon">{2}</div>
                        <div class="nav-card-title" style="color:white;">{3}</div>
                        <div class="nav-card-desc" style="color:rgba(255,255,255,0.85);">{4}</div>
                    </div>""".format(color1, color2, icon, title, description),
                    unsafe_allow_html=True,
                )
                if st.button(button_text, key=button_key, width="stretch"):
                    st.session_state.current_page = target
                    st.rerun()


def show(
    df,
    summary,
    cover_map,
    cover_dir,
    verified_covers,
    get_detail_info,
    get_desc,
    rating_metrics,
):
    """渲染应用首页。"""
    card_bg = "#2d2d44" if st.session_state.dark_mode else "#ffffff"
    sub_color = "#aaa" if st.session_state.dark_mode else "#888"
    border_color = "#3d3d5c" if st.session_state.dark_mode else "#f0f0f0"
    _inject_home_css(card_bg, border_color, sub_color)

    st.markdown('<div class="hero-title">豆瓣图书评价与推荐系统</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle" style="color:{0};">📖 发现好书 · 智能推荐 · 数据洞察</div>'.format(
            sub_color
        ),
        unsafe_allow_html=True,
    )
    _render_stats(summary, sub_color)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📖 精选高分图书")
    st.caption("💡 点击「📖 查看」按钮查看图书详细信息")
    featured_books = select_featured_books(df, cover_dir, verified_covers)
    _render_book_grid(featured_books, cover_dir, verified_covers, card_bg)
    _render_selected_book(
        featured_books,
        cover_map,
        cover_dir,
        verified_covers,
        get_detail_info,
        get_desc,
    )

    st.markdown("---")
    _render_navigation(summary, rating_metrics)
    st.markdown("---")
    st.caption("江南大学 · 大学生创新训练计划项目 | 豆瓣读书公开数据集")
