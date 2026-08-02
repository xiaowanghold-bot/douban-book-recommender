"""
豆瓣图书评价与推荐系统 - Streamlit Web 应用
江南大学大学生创新训练计划项目
功能：深色模式 | 搜索自动补全+标签筛选 | 图书详情浮窗 | 标签分类浏览 | 图书简介展示 | 评分预测
"""
from functools import partial
import importlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import data_loader as data_loader_module


if not hasattr(data_loader_module, "RECOMMENDER_CACHE_VERSION"):
    st.cache_resource.clear()
    data_loader_module = importlib.reload(data_loader_module)

from data_loader import (  # noqa: E402
    COVER_DIR,
    FIG_DIR,
    RECOMMENDER_CACHE_VERSION,
    get_cover_path,
    get_description,
    get_detail_info as find_detail_info,
    load_app_summary,
    load_author_stats,
    load_coldstart_model_meta,
    load_coldstart_predictor,
    load_cover_map,
    load_descriptions,
    load_detail_data,
    load_genre_index,
    load_price_data,
    load_publisher_stats as load_pub_stats,
    load_rating_model_meta,
    load_rating_predictor as load_predictor,
    load_recommender,
    load_scored_data,
    load_tag_index,
    load_verified_covers,
)
st.set_page_config(
    page_title="豆瓣图书评价与推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

app_summary = load_app_summary()
rating_metrics = load_rating_model_meta().get("metrics", {})
coldstart_metrics = load_coldstart_model_meta().get("metrics", {})


def load_book_display_context():
    """仅在需要图书详情或封面的页面加载展示数据。"""
    detail_df = load_detail_data()
    descriptions = load_descriptions()
    cover_map = load_cover_map()
    verified_covers = load_verified_covers()
    get_detail_info = partial(find_detail_info, detail_df)
    get_desc = partial(get_description, descriptions)
    get_cover = partial(
        get_cover_path,
        cover_map=cover_map,
        cover_dir=COVER_DIR,
        verified_covers=verified_covers,
    )
    return cover_map, verified_covers, get_detail_info, get_desc, get_cover

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
    st.sidebar.caption("收录图书: {0:,} 本".format(app_summary.get("book_count", 0)))
    st.sidebar.caption("详细信息: {0:,} 本".format(app_summary.get("detail_count", 0)))
    st.sidebar.caption("图书简介: {0:,} 本".format(app_summary.get("description_count", 0)))
    st.sidebar.caption("封面图片: {0:,} 张".format(app_summary.get("cover_count", 0)))
    st.sidebar.caption("推荐引擎: jieba 语义 TF-IDF + Cosine")
    st.sidebar.caption(
        "评分预测: RF RMSE={0:.2f} | 冷启动: GBR R²={1:.2f}".format(
            rating_metrics.get("RMSE", 0.50),
            coldstart_metrics.get("R2", 0.49),
        )
    )
    st.sidebar.caption("江南大学 · 大创项目")
    st.sidebar.success("📁 xiaowanghold-bot/douban-book-recommender")
# ======================================================================
#  首页
# ======================================================================
if page == "🏠 首页":
    from home_page import show as show_home

    df = load_scored_data()
    cover_map, verified_covers, get_detail_info, get_desc, _ = (
        load_book_display_context()
    )
    show_home(
        df,
        app_summary,
        cover_map,
        COVER_DIR,
        verified_covers,
        get_detail_info,
        get_desc,
        rating_metrics,
    )

# ======================================================================
#  排行榜
# ======================================================================
elif page == "🏆 排行榜":
    from utils import dedup_editions

    df = load_scored_data()
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

    top_raw = df[df["votes"] >= min_votes].nlargest(top_n, "bayesian_score")
    top = dedup_editions(top_raw)

    tab1, tab2 = st.tabs(["📊 可视化排名", "📋 数据表格"])

    with tab1:
        import plotly.express as px
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
        st.plotly_chart(fig_bar, width="stretch")

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
            st.plotly_chart(fig_hist, width="stretch")

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
            st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("全库均值 C", f"{C:.3f}")
        sc2.metric("Top10 平均", f"{top.head(10)['rating'].mean():.1f}")
        sc3.metric("Top10 评价中位数", f"{int(top.head(10)['votes'].median()):,}")
        sc4.metric("符合条件图书 (votes>={min_votes})", f"{len(top):,} 本")

    with tab2:
        disp = top[["title", "rating", "votes", "bayesian_score"]].copy()
        disp.columns = ["书名", "评分", "评价人数", "贝叶斯评分"]
        disp.index = range(1, len(disp) + 1)
        st.dataframe(
            disp.style.format({"评分": "{:.1f}", "贝叶斯评分": "{:.4f}", "评价人数": "{:,}"})
            .background_gradient(subset=["贝叶斯评分"], cmap="YlOrRd"),
            width="stretch", height=600,
        )
        csv = disp.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 下载排行榜 CSV", csv, "book_ranking.csv", "text/csv")
elif page == "🔍 搜书推荐":
    from search_page import show as show_search

    rec = load_recommender(RECOMMENDER_CACHE_VERSION)
    tag_to_ids, _ = load_tag_index()
    cover_map, verified_covers, get_detail_info, get_desc, _ = (
        load_book_display_context()
    )
    show_search(
        rec,
        tag_to_ids,
        cover_map,
        COVER_DIR,
        verified_covers,
        get_detail_info,
        get_desc,
    )

# ======================================================================
#  出版社与作者
# ======================================================================
elif page == "🏢 出版社与作者":
    pub_stats = load_pub_stats()
    auth_stats = load_author_stats()
    st.title("🏢 出版社与作者分析")
    st.markdown("*基于爬虫获取的 6,575 本高分图书详细信息*")
    st.caption("综合评分采用与图书排行榜一致的贝叶斯收缩(m=P75)，避免小样本出版社/作者因少量高分书虚高。")

    if pub_stats is not None:
        st.markdown("### 📚 出版社综合评价")
        c1, c2 = st.columns(2)
        c1.metric("出版社总数", len(pub_stats))
        c2.metric("平均每社图书", "{0:.1f} 本".format(pub_stats["book_count"].mean()))
        tp = pub_stats.head(15)[["book_count", "avg_rating", "pub_score"]]
        tp.columns = ["图书数量", "平均评分", "综合评分(贝叶斯)"]
        tp.index = ["{0}. {1}".format(i+1, n) for i, n in enumerate(tp.index)]
        st.dataframe(
            tp.style.format({"平均评分": "{:.2f}", "综合评分(贝叶斯)": "{:.4f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["综合评分(贝叶斯)"], cmap="YlOrRd"),
            width="stretch",
        )
        if (FIG_DIR / "10_publisher_matrix.png").exists():
            st.image(str(FIG_DIR / "10_publisher_matrix.png"), width="stretch")

    if auth_stats is not None:
        st.markdown("---")
        st.markdown("### ✍️ 作者影响力")
        c1, c2, c3 = st.columns(3)
        c1.metric("作者总数", len(auth_stats))
        c2.metric("中国作者", int((auth_stats["nationality"] == "中国").sum()))
        c3.metric("外国作者", int((auth_stats["nationality"] != "中国").sum()))
        ta = auth_stats.head(15)[["book_count", "nationality", "avg_rating", "influence"]]
        ta.columns = ["图书数量", "国籍", "平均评分", "影响力(贝叶斯)"]
        ta.index = ["{0}. {1}".format(i+1, n) for i, n in enumerate(ta.index)]
        st.dataframe(
            ta.style.format({"平均评分": "{:.2f}", "影响力(贝叶斯)": "{:.1f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["影响力(贝叶斯)"], cmap="YlOrRd"),
            width="stretch",
        )
        if (FIG_DIR / "11_author_influence.png").exists():
            st.image(str(FIG_DIR / "11_author_influence.png"), width="stretch")

    if (FIG_DIR / "12_year_trend.png").exists():
        st.markdown("### 📅 出版年份趋势")
        st.image(str(FIG_DIR / "12_year_trend.png"), width="stretch")

# ======================================================================
#  评分预测 (REFORMED: prediction-first layout)
# ======================================================================
elif page == "🔮 评分预测":
    from rating_page import show as show_rating

    show_rating(load_predictor(), rating_metrics)

# ======================================================================
#  更多发现
# ======================================================================
elif page == "💡 更多发现":
    st.title("💡 更多发现")
    st.markdown("*词云、价格分析、趋势洞察*")

    if (FIG_DIR / "13_wordcloud.png").exists():
        st.markdown("### ☁️ 高分图书书名词云")
        st.image(str(FIG_DIR / "13_wordcloud.png"), width="stretch")

    st.markdown("---")
    if (FIG_DIR / "14_price_analysis.png").exists():
        st.markdown("### 💰 价格分析")
        st.image(str(FIG_DIR / "14_price_analysis.png"), width="stretch")

    price_df = load_price_data()
    if price_df is not None:
        st.markdown("### 🏷️ 高性价比图书 (评分>=9, <=50元)")
        vb = price_df[(price_df["Rating"] >= 9) & (price_df["price_num"] <= 50)]
        vb = vb.nlargest(10, "Votes")[["Title", "Rating", "price_num", "author", "Votes"]]
        vb.columns = ["书名", "评分", "价格(元)", "作者", "评价人数"]
        vb.index = range(1, len(vb) + 1)
        st.dataframe(
            vb.style.format({"评分": "{:.1f}", "价格(元)": "{:.1f}", "评价人数": "{:,}"}),
            width="stretch",
        )

# ======================================================================
#  标签浏览
# ======================================================================

elif page == "🧊 新书预测":
    from coldstart_page import show as show_coldstart

    csp = load_coldstart_predictor()
    show_coldstart(csp)

elif page == "🏷️ 标签浏览":
    from components import render_cover
    from genre_search import GENRE_GROUPS, search_books_by_genre
    from utils import dedup_editions

    df = load_scored_data()
    genre_text_index = load_genre_index(df)
    cover_map, verified_covers, get_detail_info, get_desc, get_cover = (
        load_book_display_context()
    )
    st.title("🏷️ 标签分类浏览")
    st.markdown("*基于真实豆瓣用户标签 + 语义搜索的流派图书探索*")

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
                        width="stretch",
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
        genre_results = dedup_editions(genre_results) if not genre_results.empty else genre_results

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
                dc1, dc2 = st.columns([1, 3])
                with dc1:
                    render_cover(bid2, cover_map, COVER_DIR, verified_covers, width=120)
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
    st.caption("💡 提示：流派搜索基于真实豆瓣用户标签 + 语义搜索来查找相关图书，结果按匹配度和评分综合排序。")

elif page == "📋 关于项目":
    st.title("📋 关于项目")
    st.markdown(f"""
## 豆瓣图书评价与推荐系统
**江南大学大学生创新训练计划项目**

本项目定位为**面向中文图书数据的可解释研究原型**：重点验证数据清洗、评价人数偏差校正、
内容相似推荐和冷启动评分预测能否形成一条可复现的分析流程，不将经典算法包装为前沿原创模型。

### 技术方案
- **贝叶斯加权评分**：IMDb 式收缩公式缓解小样本高分造成的排名偏差
- **内容推荐引擎**：jieba 语义 TF-IDF + 余弦相似度，融合书名/标签/作者/简介，去重同书名
- **出版社/作者分析**：{app_summary.get("publisher_count", "?")} 家出版社、{app_summary.get("author_count", "?")} 位作者综合评价矩阵
- **评分预测**：RandomForest 回归 (RMSE {rating_metrics.get("RMSE", 0):.3f}, OOF目标编码)；冷启动：GradientBoosting v4 (10维OOF特征, 测试R²={coldstart_metrics.get("R2", 0):.3f})

### 证据边界
- 同系列 Recall 只能反映系列检索能力；真实用户 LOO Recall@20 为 **0.0206**，说明个性化推荐仍有明显提升空间
- 评分预测和冷启动指标来自固定随机种子的独立留出测试与嵌套交叉验证，不代表因果关系或线上业务收益
- “新书预测”的 5%–95% 输出是分位数预测区间，未做覆盖率校准，不等同于统计置信区间
- 多版本去重主要按规范化书名处理，同名异书存在被合并的风险

### 数据来源
- 豆瓣读书公开数据集 (yuzhounh/Douban-books-2020)
- 288,824 本基础数据 + {app_summary.get("detail_success_count", "?")} 本爬虫详细信息
- 481 个豆列 + 897 个标签

### 研究用途与责任
数据与模型仅用于非商业的学习、研究和答辩演示；系统不展示用户身份信息，也不替代出版、采购或投资判断。

### 技术栈
Python 3.12 · Streamlit · pandas · scikit-learn · matplotlib · jieba · wordcloud
    """)
