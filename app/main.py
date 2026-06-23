"""
豆瓣读书图书评价与推荐系统 - Streamlit Web 应用
江南大学大学生创新训练计划项目
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import pickle
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from recommendation import BookRecommender

st.set_page_config(
    page_title="豆瓣图书评价与推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 缓存加载 ==========

@st.cache_resource
def load_recommender():
    rec = BookRecommender()
    rec._load_artifacts()
    nn = np.load(str(Path(__file__).parent.parent / "data" / "models" / "nn_neighbors.npz"))
    rec.nn_distances = nn["distances"]
    rec.nn_indices = nn["indices"]
    return rec

@st.cache_data
def load_scored_data():
    return pd.read_csv(
        Path(__file__).parent.parent / "data" / "processed" / "books_scored.csv",
        encoding="utf-8-sig")

@st.cache_data
def load_price_data():
    p = Path(__file__).parent.parent / "data" / "processed" / "books_with_price.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data
def load_pub_stats():
    p = Path(__file__).parent.parent / "data" / "processed" / "publisher_stats.csv"
    return pd.read_csv(p, encoding="utf-8-sig", index_col=0) if p.exists() else None

@st.cache_data
def load_author_stats():
    p = Path(__file__).parent.parent / "data" / "processed" / "author_stats.csv"
    return pd.read_csv(p, encoding="utf-8-sig", index_col=0) if p.exists() else None

@st.cache_resource
def load_predictor():
    p = Path(__file__).parent.parent / "data" / "models" / "rating_predictor.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

rec = load_recommender()
df = load_scored_data()
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
DATA_DIR = Path(__file__).parent.parent / "data"

# ========== 侧边栏 ==========
st.sidebar.markdown("# 📚 豆瓣图书评价与推荐系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    ["🏠 首页", "🏆 排行榜", "🔍 搜书推荐",
     "🏢 出版社与作者", "🔮 评分预测", "💡 更多发现", "ℹ️ 关于项目"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"数据总量: {len(df):,} 本")
st.sidebar.caption(f"爬虫完成: 6,575 本详细信息")
st.sidebar.caption(f"推荐引擎: TF-IDF + 余弦相似度")
st.sidebar.caption(f"评分预测: RandomForest MAE=0.40")
st.sidebar.caption("江南大学 · 大创项目")

# ======================================================================
#  首页
# ======================================================================
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

    st.markdown("### 🎯 项目功能总览")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        **📊 数据洞察**
        - 174,244 本图书评分分布
        - 评价人数对数分布
        - 相关性热力图
        """)
    with col_b:
        st.markdown("""
        **🏆 贝叶斯排名**
        - 消除评价人数偏差
        - 小众高质图书上榜
        - m 参数优化分析
        """)
    with col_c:
        st.markdown("""
        **🔍 智能推荐**
        - 字符级 N-gram 相似度
        - TF-IDF + 余弦匹配
        - 混合推荐策略
        """)

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        st.markdown("""
        **🏢 出版社/作者**
        - 221 家出版社排名
        - 878 位作者影响力
        - 二维评价矩阵
        """)
    with col_e:
        st.markdown("""
        **🔮 评分预测**
        - RandomForest 回归
        - 7 维特征预测
        - MAE = 0.40
        """)
    with col_f:
        st.markdown("""
        **💡 更多发现**
        - 书名词云
        - 价格分析
        - 高性价比书单
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
                label=f"{color} {str(row['title'])[:16]}",
                value=f"{row['rating']:.1f} 分",
                delta=f"{int(row['votes']):,} 人评价",
            )

# ======================================================================
#  排行榜
# ======================================================================
elif page == "🏆 排行榜":
    st.title("🏆 贝叶斯加权评分排行榜")
    st.markdown("*基于贝叶斯平均算法，平衡评分高低与评价人数*")

    c1, c2 = st.columns([1, 3])
    with c1:
        min_votes = st.slider("最少评价人数", 0, 10000, 50, 100)
        top_n = st.slider("显示数量", 10, 100, 50, 10)
    top = df[df["votes"] >= min_votes].nlargest(top_n, "bayesian_score")
    disp = top[["title", "rating", "votes", "bayesian_score"]].copy()
    disp.columns = ["书名", "评分", "评价人数", "贝叶斯评分"]
    disp.index = range(1, len(disp) + 1)
    c2.dataframe(
        disp.style.format({"评分": "{:.1f}", "贝叶斯评分": "{:.4f}", "评价人数": "{:,}"})
        .background_gradient(subset=["贝叶斯评分"], cmap="YlOrRd"),
        use_container_width=True, height=600,
    )
    csv = disp.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("📥 下载排行榜 CSV", csv, "book_ranking.csv", "text/csv")

# ======================================================================
#  搜书推荐
# ======================================================================
elif page == "🔍 搜书推荐":
    st.title("🔍 搜书 & 智能推荐")
    query = st.text_input("输入书名或关键词", placeholder="例如：三体、活着、百年孤独...")
    if query:
        with st.spinner("搜索中..."):
            results = rec.recommend_by_title(query, top_n=20)
        if results.empty:
            st.warning("未找到相关图书")
        else:
            book_titles = [f"{r['title']}（{r['rating']:.1f}分 {int(r['votes']):,}人）"
                           for _, r in results.iterrows()]
            sel = st.selectbox("选择图书查看推荐", range(len(book_titles)),
                               format_func=lambda i: book_titles[i])
            if sel is not None:
                selected = results.iloc[sel]
                bid = int(selected["id"])
                ca, cb = st.columns([1, 2])
                with ca:
                    st.markdown("### 📖 选中图书")
                    st.metric("书名", selected["title"])
                    st.metric("评分", f"{selected['rating']:.1f} / 10")
                    st.metric("评价人数", f"{int(selected['votes']):,}")
                with cb:
                    t1, t2 = st.tabs(["📚 内容推荐", "🔀 混合推荐"])
                    with t1:
                        recs = rec.recommend_by_id(bid, top_n=10)
                        for i, (_, r) in enumerate(recs.iterrows()):
                            st.markdown(f"**{i+1}. {r['title']}** ⭐{r['rating']:.1f} 📊{int(r['votes']):,}人 `{r['similarity']:.2%}`")
                            st.progress(float(r["similarity"]))
                    with t2:
                        hyb = rec.hybrid_recommend(bid, top_n=10)
                        for i, (_, r) in enumerate(hyb.iterrows()):
                            st.markdown(f"**{i+1}. {r['title']}** ⭐{r['rating']:.1f} 📊{int(r['votes']):,}人 `{r['hybrid_score']:.4f}`")
                            st.progress(float(r["hybrid_score"]))

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
        c2.metric("平均每社图书", f"{pub_stats['book_count'].mean():.1f} 本")
        tp = pub_stats.head(15)[["book_count", "avg_rating", "pub_score"]]
        tp.columns = ["图书数量", "平均评分", "综合评分"]
        tp.index = [f"{i+1}. {n}" for i, n in enumerate(tp.index)]
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
        ta.index = [f"{i+1}. {n}" for i, n in enumerate(ta.index)]
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
#  评分预测
# ======================================================================
elif page == "🔮 评分预测":
    st.title("🔮 图书评分预测")
    st.markdown("*基于作者、出版社、年份、价格等特征，RandomForest 预测评分*")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("### 📝 输入图书信息")
        author = st.text_input("作者", value="余华")
        publisher = st.text_input("出版社", value="人民文学出版社")
        price = st.number_input("价格 (元)", 0.0, 2000.0, 39.5, 0.5)
        year = st.number_input("出版年份", 1950, 2026, 2014)
        pages = st.number_input("页数", 10, 5000, 300)
        votes = st.number_input("评价人数", 0, 10000000, 50000)
        binding = st.selectbox("装帧", ["平装", "精装", "其他"])

        if st.button("🎯 预测评分", type="primary"):
            saved = load_predictor()
            if saved is None:
                st.error("模型文件未找到")
            else:
                try:
                    from enhancements import RatingPredictor
                    rp = RatingPredictor()
                    rp.model = saved["model"]
                    rp.encoders = saved["encoders"]
                    rp.feature_names = saved["feature_names"]
                    ref = pd.read_csv(
                        DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
                    ref = ref[ref["crawl_status"] == "success"].copy()
                    ref["author_clean"] = ref["author"].apply(
                        lambda x: re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(x)).strip()[:30]
                        if pd.notna(x) else "未知")
                    ref["publisher_clean"] = ref["publisher"].fillna("未知").astype(str).str[:20]
                    ref["binding_type"] = ref["binding"].fillna("未知").apply(
                        lambda x: "平装" if "平装" in str(x) else ("精装" if "精装" in str(x) else "其他"))
                    rp.df = ref
                    pred = rp.predict(price, year, pages, votes, author, publisher, binding)
                    if pred:
                        st.success(f"预测评分: **{pred:.2f}** / 10")
                        if pred >= 9.0:
                            st.info("🏆 预测为高分图书！")
                        elif pred >= 8.0:
                            st.info("👍 预测为优良图书")
                        elif pred >= 7.0:
                            st.info("📖 预测为中等评分")
                        else:
                            st.info("📚 预测评分偏低")
                except Exception as e:
                    st.error(f"预测出错: {e}")

    with c2:
        saved = load_predictor()
        if saved:
            st.markdown("### 📊 模型性能")
            m = saved.get("metrics", {})
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE (平均误差)", f"{m.get('MAE', 0):.3f}")
            mc2.metric("R² (拟合度)", f"{m.get('R2', 0):.3f}")
            mc3.metric("CV5 (交叉验证)", f"{m.get('CV_R2', 0):.3f}")
        st.markdown("### 📈 特征重要性")
        if (FIG_DIR / "15_feature_importance.png").exists():
            st.image(str(FIG_DIR / "15_feature_importance.png"), use_container_width=True)
        if (FIG_DIR / "16_prediction_scatter.png").exists():
            st.image(str(FIG_DIR / "16_prediction_scatter.png"), use_container_width=True)

# ======================================================================
#  更多发现
# ======================================================================
elif page == "💡 更多发现":
    st.title("💡 更多发现")

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
#  关于项目
# ======================================================================
elif page == "ℹ️ 关于项目":
    st.title("ℹ️ 关于项目")
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

st.sidebar.markdown("---")
st.sidebar.success("📁 GitHub: xiaowanghold-bot/douban-book-recommender")
