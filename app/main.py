"""
豆瓣读书图书评价与推荐系统 - Streamlit Web 应用
江南大学大学生创新训练计划项目
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from recommendation import BookRecommender

# ========== 页面配置 ==========
st.set_page_config(
    page_title="豆瓣图书评价与推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 数据加载（缓存）==========

@st.cache_resource
def load_recommender():
    """加载推荐引擎（全局单例）"""
    rec = BookRecommender()
    rec._load_artifacts()
    nn = np.load(str(Path(__file__).parent.parent / "data" / "models" / "nn_neighbors.npz"))
    rec.nn_distances = nn["distances"]
    rec.nn_indices = nn["indices"]
    return rec

@st.cache_data
def load_scored_data():
    """加载评分数据"""
    path = Path(__file__).parent.parent / "data" / "processed" / "books_scored.csv"
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data
def load_cleaned_data():
    """加载清洗后数据"""
    path = Path(__file__).parent.parent / "data" / "processed" / "books_cleaned.csv"
    return pd.read_csv(path, encoding="utf-8-sig")

# ========== 初始化 ==========
rec = load_recommender()
df = load_scored_data()

# ========== 侧边栏导航 ==========
st.sidebar.markdown("# 📚 豆瓣图书评价与推荐系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    ["🏠 首页", "🏆 排行榜", "🔍 搜书推荐", "📊 数据洞察", "ℹ️ 关于项目"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"数据总量: {len(df):,} 本图书")
st.sidebar.caption(f"贝叶斯评分模型: 已就绪")
st.sidebar.caption(f"推荐引擎: 已就绪")
st.sidebar.caption("江南大学 · 大学生创新训练计划")

# ========== 首页 ==========
if page == "🏠 首页":
    st.title("📚 豆瓣图书评价与推荐系统")
    st.markdown("### 基于豆瓣读书数据的智能图书评价与推荐平台")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("收录图书", f"{len(df):,}")
    with col2:
        st.metric("平均评分", f"{df['rating'].mean():.1f}")
    with col3:
        st.metric("最高评分", f"{df['rating'].max():.1f}")
    with col4:
        st.metric("评价过万图书", f"{(df['votes']>=10000).sum():,}")

    st.markdown("---")

    st.markdown("### 🎯 项目功能")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **📊 数据洞察**
        - 174,244 本图书的评分分布分析
        - 评价人数对数分布
        - 评分与评价人数关系
        - 相关性热力图
        """)
        st.markdown("""
        **🏆 贝叶斯加权排名**
        - 消除评价人数偏差
        - 小众高质图书也能上榜
        - 参数 m 优化分析
        """)
    with col_b:
        st.markdown("""
        **🔍 智能推荐**
        - 基于中文书名的字符级 N-gram 相似度
        - TF-IDF + 余弦相似度匹配
        - 混合推荐：内容相似度 + 贝叶斯评分
        - 去重同书名多版本
        """)
        st.markdown("""
        **📈 出版社/作者分析**（待爬虫完成）
        - 出版社评价矩阵
        - 作者影响力排行
        - 出版年份趋势
        """)

    st.markdown("---")
    st.markdown("### 🔥 高分图书速览")
    top_books = df.nlargest(8, "bayesian_score")[["title", "rating", "votes", "bayesian_score"]]
    cols = st.columns(4)
    for i, (_, row) in enumerate(top_books.iterrows()):
        with cols[i % 4]:
            score = row["bayesian_score"]
            color = "🟢" if score > 9.5 else "🟡" if score > 9.0 else "🟠"
            st.metric(
                label=f"{color} {row['title'][:16]}",
                value=f"{row['rating']:.1f} 分",
                delta=f"{int(row['votes']):,} 人评价",
            )

# ========== 排行榜 ==========
elif page == "🏆 排行榜":
    st.title("🏆 贝叶斯加权评分排行榜")
    st.markdown("*基于贝叶斯平均算法，平衡评分高低与评价人数的影响*")

    col1, col2 = st.columns([1, 3])
    with col1:
        min_votes = st.slider("最少评价人数", 0, 10000, 50, 100)
        top_n = st.slider("显示数量", 10, 100, 50, 10)

    top = df[df["votes"] >= min_votes].nlargest(top_n, "bayesian_score")

    # 排名变化
    raw_rank = df[df["votes"] >= min_votes].nlargest(top_n, "rating")
    top["raw_rank"] = range(1, len(top) + 1)

    col2.markdown(f"### Top {top_n}（评价人数 ≥ {min_votes}）")

    # 样式化表格
    display_df = top[["title", "rating", "votes", "bayesian_score"]].copy()
    display_df.columns = ["书名", "评分", "评价人数", "贝叶斯评分"]
    display_df.index = range(1, len(display_df) + 1)

    col2.dataframe(
        display_df.style
        .format({"评分": "{:.1f}", "贝叶斯评分": "{:.4f}", "评价人数": "{:,}"})
        .background_gradient(subset=["贝叶斯评分"], cmap="YlOrRd"),
        use_container_width=True,
        height=600,
    )

    # 下载按钮
    csv = display_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("📥 下载排行榜 CSV", csv, "book_ranking.csv", "text/csv")

# ========== 搜书推荐 ==========
elif page == "🔍 搜书推荐":
    st.title("🔍 搜书 & 智能推荐")
    st.markdown("*输入书名关键词，获取相似图书推荐*")

    query = st.text_input("输入书名或关键词", placeholder="例如：三体、活着、百年孤独...")

    if query:
        with st.spinner("搜索中..."):
            results = rec.recommend_by_title(query, top_n=20)

        if results.empty:
            st.warning("未找到相关图书，请尝试其他关键词")
        else:
            # 选择目标图书
            book_titles = [f"{row['title']}（{row['rating']:.1f}分 {int(row['votes']):,}人）"
                           for _, row in results.iterrows()]

            selected_idx = st.selectbox(
                "选择一本图书查看推荐",
                range(len(book_titles)),
                format_func=lambda i: book_titles[i],
            )

            if selected_idx is not None:
                selected = results.iloc[selected_idx]
                book_id = int(selected["id"])

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown("### 📖 选中图书")
                    st.metric("书名", selected["title"])
                    st.metric("评分", f"{selected['rating']:.1f} / 10")
                    st.metric("评价人数", f"{int(selected['votes']):,}")
                    st.metric("贝叶斯评分", f"{selected['bayesian_score']:.4f}")

                with col_b:
                    tab1, tab2 = st.tabs(["📚 内容推荐", "🔀 混合推荐"])

                    with tab1:
                        recs = rec.recommend_by_id(book_id, top_n=10)
                        if not recs.empty:
                            for i, (_, row) in enumerate(recs.iterrows()):
                                sim_bar = "█" * int(row["similarity"] * 20)
                                st.markdown(
                                    f"**{i+1}. {row['title']}**  "
                                    f"⭐{row['rating']:.1f}  "
                                    f"📊{int(row['votes']):,}人  "
                                    f"`相似度 {row['similarity']:.2%}`"
                                )
                                st.progress(float(row["similarity"]))

                    with tab2:
                        hyb_recs = rec.hybrid_recommend(book_id, top_n=10, alpha=0.5)
                        if not hyb_recs.empty:
                            for i, (_, row) in enumerate(hyb_recs.iterrows()):
                                st.markdown(
                                    f"**{i+1}. {row['title']}**  "
                                    f"⭐{row['rating']:.1f}  "
                                    f"📊{int(row['votes']):,}人  "
                                    f"`混合分 {row['hybrid_score']:.4f}`"
                                )
                                st.progress(float(row["hybrid_score"]))

# ========== 数据洞察 ==========
elif page == "📊 数据洞察":
    st.title("📊 探索性数据分析")

    fig_dir = Path(__file__).parent.parent / "reports" / "figures"

    st.markdown("### 评分与评价人数分布")
    col1, col2 = st.columns(2)
    with col1:
        if (fig_dir / "01_rating_distribution.png").exists():
            st.image(str(fig_dir / "01_rating_distribution.png"), caption="评分分布")
    with col2:
        if (fig_dir / "02_votes_distribution.png").exists():
            st.image(str(fig_dir / "02_votes_distribution.png"), caption="评价人数分布")

    st.markdown("### 评分与评价人数关系")
    if (fig_dir / "03_rating_vs_votes.png").exists():
        st.image(str(fig_dir / "03_rating_vs_votes.png"), use_container_width=True)

    st.markdown("### 评分等级与评价人数等级")
    col1, col2 = st.columns(2)
    with col1:
        if (fig_dir / "05_rating_tiers.png").exists():
            st.image(str(fig_dir / "05_rating_tiers.png"))
    with col2:
        if (fig_dir / "06_votes_tiers.png").exists():
            st.image(str(fig_dir / "06_votes_tiers.png"))

    st.markdown("### 相关性分析 & 贝叶斯参数优化")
    col1, col2 = st.columns(2)
    with col1:
        if (fig_dir / "07_correlation_heatmap.png").exists():
            st.image(str(fig_dir / "07_correlation_heatmap.png"))
    with col2:
        if (fig_dir / "08_m_parameter_analysis.png").exists():
            st.image(str(fig_dir / "08_m_parameter_analysis.png"))

    st.markdown("### 贝叶斯 Top 15 vs 原始 Top 15")
    if (fig_dir / "09_bayesian_top15.png").exists():
        st.image(str(fig_dir / "09_bayesian_top15.png"), use_container_width=True)

# ========== 关于 ==========

# ========== 出版社与作者 ==========
elif page == "🏢 出版社与作者":
    st.title("🏢 出版社与作者分析")
    st.markdown("*基于爬虫获取的 6,575 本高分图书详细信息*")

    fig_dir = Path(__file__).parent.parent / "reports" / "figures"
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    # 加载分析数据
    @st.cache_data
    def load_pub_stats():
        path = data_dir / "publisher_stats.csv"
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        return None

    @st.cache_data
    def load_author_stats():
        path = data_dir / "author_stats.csv"
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        return None

    pub_stats = load_pub_stats()
    author_stats = load_author_stats()

    if pub_stats is not None:
        st.markdown("### 📚 出版社综合评价矩阵")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("出版社总数", len(pub_stats))
        with col2:
            st.metric("平均每社图书", f"{pub_stats['book_count'].mean():.1f} 本")

        # Top 出版社表格
        top_pub = pub_stats.head(15)[["book_count", "avg_rating", "avg_bayesian", "pub_score"]]
        top_pub.columns = ["图书数量", "平均评分", "贝叶斯均分", "综合评分"]
        top_pub.index = [f"{i+1}. {n}" for i, n in enumerate(top_pub.index)]
        st.dataframe(
            top_pub.style
            .format({"平均评分": "{:.2f}", "贝叶斯均分": "{:.4f}", "综合评分": "{:.4f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["综合评分"], cmap="YlOrRd"),
            use_container_width=True,
        )

        st.markdown("### 📈 出版社二维评价矩阵")
        if (fig_dir / "10_publisher_matrix.png").exists():
            st.image(str(fig_dir / "10_publisher_matrix.png"), use_container_width=True)

    if author_stats is not None:
        st.markdown("---")
        st.markdown("### ✍️ 作者影响力分析")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("作者总数", len(author_stats))
        with col2:
            st.metric("中国作者", int((author_stats["nationality"] == "中国").sum()))
        with col3:
            st.metric("外国作者", int((author_stats["nationality"] != "中国").sum()))

        # Top 作者表格
        top_author = author_stats.head(15)[["book_count", "nationality", "avg_rating", "author_score", "influence"]]
        top_author.columns = ["图书数量", "国籍", "平均评分", "作者评分", "影响力"]
        top_author.index = [f"{i+1}. {n}" for i, n in enumerate(top_author.index)]
        st.dataframe(
            top_author.style
            .format({"平均评分": "{:.2f}", "作者评分": "{:.4f}", "影响力": "{:.1f}", "图书数量": "{:.0f}"})
            .background_gradient(subset=["影响力"], cmap="YlOrRd"),
            use_container_width=True,
        )

        st.markdown("### 📊 作者影响力与国籍分布")
        if (fig_dir / "11_author_influence.png").exists():
            st.image(str(fig_dir / "11_author_influence.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📅 出版年份趋势")
    if (fig_dir / "12_year_trend.png").exists():
        st.image(str(fig_dir / "12_year_trend.png"), use_container_width=True)

elif page == "ℹ️ 关于项目":
    st.title("ℹ️ 关于项目")

    st.markdown("""
    ## 豆瓣图书评价与推荐系统

    **江南大学大学生创新训练计划项目**

    ### 项目背景
    豆瓣读书是国内最大的图书社区之一，拥有海量的用户评分数据。
    然而，简单的算术平均评分容易受到评价人数偏差的影响——
    一本只有10人评价的9.5分图书，与一本有10万人评价的9.0分图书，哪个更值得推荐？

    ### 技术方案

    **1. 贝叶斯加权评分模型**
    - 采用 IMDb 式的贝叶斯平均算法
    - 公式: `BS = C/(C+m) × 全局平均 + m/(C+m) × 原始评分`
    - 其中 m 为评价人数的 P50 中位数，C 为全局平均评分

    **2. 内容推荐引擎**
    - 对中文书名进行字符级 N-gram 分词
    - TF-IDF 向量化 + L2 归一化
    - 余弦相似度计算
    - 同书名去重 + 混合贝叶斯评分排序

    ### 数据来源
    - 豆瓣读书公开数据集 (yuzhounh/Douban-books-2020)
    - 288,824 本图书基础数据
    - 爬虫补全作者、出版社、价格等详细信息

    ### 技术栈
    - Python 3.12 + Streamlit
    - pandas, numpy, scikit-learn
    - matplotlib, seaborn, plotly
    - jieba 中文分词

    ### 项目结构
    ```
    ├── app/main.py          # Streamlit Web 应用
    ├── src/
    │   ├── data_cleaning.py # 数据清洗
    │   ├── eda.py           # 探索性分析
    │   ├── scoring.py       # 贝叶斯评分
    │   └── recommendation.py # 推荐引擎
    ├── crawler/             # 豆瓣爬虫
    ├── data/                # 数据文件
    ├── reports/             # 图表报告
    └── notebooks/           # Jupyter 分析
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 快速入口")
if st.sidebar.button("🏆 查看排行榜"):
    st.switch_page("app/main.py")  # 通过 rerun 实现
    st.rerun()
if st.sidebar.button("🔍 搜书推荐"):
    st.rerun()
