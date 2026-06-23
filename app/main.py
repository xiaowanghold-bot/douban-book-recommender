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

pages_list = ["🏠 首页", "🏆 排行榜", "🔍 搜书推荐",
               "🏢 出版社与作者", "🔮 评分预测", "💡 更多发现", "ℹ️ 关于项目"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 首页"

# 用 index 参数控制 radio 当前选中项
page = st.sidebar.radio(
    "导航菜单", pages_list,
    index=pages_list.index(st.session_state.current_page),
)
# 用户手动点 radio 时同步回 session_state
if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"数据总量: {len(df):,} 本")
st.sidebar.caption(f"爬虫完成: 6,575 本详细信息")
st.sidebar.caption(f"推荐引擎: TF-IDF + 余弦相似度")
st.sidebar.caption(f"评分预测: RandomForest MAE=0.40")
st.sidebar.caption("江南大学 · 大创项目")

# ======================================================================
#  首页
# ======================================================================
if page == "\U0001f3e0 首页":
    import plotly.express as px
    import os

    # ====== Animated CSS ======
    st.markdown("""
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
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
        text-align: center; font-size: 1.2em; color: #888; margin-bottom: 30px;
        animation: fadeInUp 0.8s ease;
    }
    .stat-card {
        background: white; border-radius: 16px; padding: 22px 18px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06); transition: all 0.3s ease;
        border: 1px solid #f0f0f0; animation: fadeInUp 0.6s ease;
    }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(0,0,0,0.12); }
    .stat-icon { font-size: 2.2em; margin-bottom: 8px; animation: float 3s ease-in-out infinite; }
    .stat-value { font-size: 1.8em; font-weight: 800; color: #2c3e50; margin: 5px 0; }
    .stat-label { font-size: 0.9em; color: #999; margin-top: 3px; }
    .stat-delta { font-size: 0.8em; color: #27ae60; margin-top: 5px; }
    .cover-card {
        border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease; background: white;
    }
    .cover-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 12px 30px rgba(0,0,0,0.15); }
    .nav-card {
        border-radius: 14px; padding: 20px 15px; text-align: center; color: white;
        transition: all 0.3s ease;
    }
    .nav-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(0,0,0,0.2); }
    </style>
    <div class="hero-title">Douban Book Recommender</div>
    <div class="hero-subtitle">基于 28 万豆瓣数据 · 智能图书分析与推荐平台</div>
    """, unsafe_allow_html=True)

    # Fix: properly use columns
    c_s1, c_s2, c_s3, c_s4, c_s5 = st.columns(5)
    stat_data = [
        (c_s1, "📕", "收录图书", f"{len(df):,}", f"原始 288K"),
        (c_s2, "⭐", "平均评分", f"{df['rating'].mean():.1f}", f"最高 {df['rating'].max():.1f}"),
        (c_s3, "👥", "评价过万", f"{(df['votes']>=10000).sum():,}", f"过千 {(df['votes']>=1000).sum():,}"),
        (c_s4, "🏢", "出版社", "221", "878 位作者"),
        (c_s5, "📈", "推荐引擎", "163K", "30 近邻/本"),
    ]
    for col, icon, label, value, delta in stat_data:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    # ====== Quick Actions ======
    st.markdown("### ✨ 快速体验")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("🎲 随机推荐", use_container_width=True, type="primary"):
            s = df[df["votes"]>=100].sample(1).iloc[0]
            st.success(f"**{s['title']}** ⭐{s['rating']:.1f}  💬{int(s['votes']):,}人")
    with q2:
        if st.button("🔥 今日热门", use_container_width=True):
            h = df[df["votes"]>=500].nlargest(1, "bayesian_score").iloc[0]
            st.success(f"**{h['title']}**  BS:{h['bayesian_score']:.4f}")
    with q3:
        if st.button("💰 高性价比", use_container_width=True):
            try:
                pp = load_price_data()
                if pp is not None:
                    b = pp[(pp["Rating"]>=9)&(pp["price_num"]<=30)].sample(1).iloc[0]
                    st.success(f"**{b['Title']}** 💰{b['price_num']:.1f}元")
            except: pass
    with q4:
        if st.button("🔮 评分预测", use_container_width=True):
            st.session_state.current_page = "🔮 评分预测"
            st.rerun()

    st.markdown("---")

    # ====== Book Covers (only REAL covers) ======
    st.markdown("### 📖 精选高分图书")

    cover_dir = Path(__file__).parent / "covers"
    cover_ids = set(int(f.stem) for f in cover_dir.glob("*.jpg"))
    df_with_covers = df[df["id"].isin(cover_ids) & (df["votes"] >= 100)]
    top_cover_books = df_with_covers.nlargest(24, "bayesian_score")

    rows = [st.columns(6) for _ in range(4)]
    for i, (_, book) in enumerate(top_cover_books.iterrows()):
        bid = int(book["id"])
        cover_path = cover_dir / f"{bid}.jpg"
        row_idx = i // 6
        col_idx = i % 6
        with rows[row_idx][col_idx]:
            if cover_path.exists():
                st.image(str(cover_path), use_container_width=True)
                t = str(book["title"])[:12]
                stars = "★" * int(book["rating"]/2) + "☆" * (5-int(book["rating"]/2))
                st.caption(f"**{t}** {stars} {book['rating']:.1f}")

    st.markdown("---")

    # ====== Interactive Charts ======
    st.markdown("### 📊 数据一览")
    tab1, tab2, tab3 = st.tabs(["📊 评分分布", "🏆 贝叶斯 Top 50", "💰 价格分布"])

    with tab1:
        fig = px.histogram(df, x="rating", nbins=40,
            title="图书评分分布（悬停交互）",
            color_discrete_sequence=["#636EFA"],
            labels={"rating": "豆瓣评分", "count": "图书数量"})
        fig.update_layout(bargap=0.05, height=350)
        fig.add_vline(x=df["rating"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"均值 {df['rating'].mean():.1f}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top50 = df[df["votes"]>=50].nlargest(50, "bayesian_score")
        fig = px.bar(top50.iloc[::-1], x="bayesian_score", y="title", orientation="h",
            title="贝叶斯加权 Top 50",
            color="bayesian_score", color_continuous_scale="YlOrRd",
            hover_data={"rating": True, "votes": True, "bayesian_score": ":.4f"})
        fig.update_layout(height=700, yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        pp = load_price_data()
        if pp is not None:
            fig = px.histogram(pp[pp["price_num"]<200], x="price_num", nbins=50,
                title="图书价格分布 (<200元)",
                color_discrete_sequence=["#00CC96"],
                labels={"price_num": "价格 (元)", "count": "图书数量"})
            fig.update_layout(height=350)
            fig.add_vline(x=pp["price_num"].median(), line_dash="dash", line_color="red",
                          annotation_text=f"中位数 {pp['price_num'].median():.1f}元")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ====== Navigation Cards ======
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
            st.markdown(f"""<div class="nav-card" style="background:linear-gradient(135deg,{c1},{c2});">
                <div style="font-size:2.5em;">{icon}</div>
                <div style="font-weight:700;font-size:1.1em;">{title}</div>
                <div style="font-size:0.8em;opacity:0.85;margin-top:5px;">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(btn_text, key=btn_key, use_container_width=True):
                st.session_state.current_page = target
                st.rerun()

    st.markdown("---")
    st.caption("江南大学 · 大学生创新训练计划项目 | 豆瓣读书公开数据集 | 288,824 本图书")


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
                    # 导出按钮
                    st.markdown("---")
                    st.caption("📥 导出推荐结果")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        try:
                            recs_csv = recs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                            st.download_button("📥 内容推荐 CSV", recs_csv, "content_recs.csv", "text/csv",
                                              key="dl_c", use_container_width=True)
                        except:
                            pass
                    with ec2:
                        try:
                            hyb_csv = hyb.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                            st.download_button("📥 混合推荐 CSV", hyb_csv, "hybrid_recs.csv", "text/csv",
                                              key="dl_h", use_container_width=True)
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
# ======================================================================
#  评分预测
# ======================================================================
elif page == "🔮 评分预测":
    st.title("🔮 图书评分预测")
    st.markdown("*基于作者、出版社、年份、价格等特征，RandomForest 预测评分*")

    # 热门列表
    TOP_PUBLISHERS = [
        "上海文艺出版社",
        "人民文学出版社",
        "上海出版公司",
        "新星出版社",
        "吉林出版社",
        "北京师范大学出版社",
        "江苏文艺出版社",
        "中信出版社",
        "生活·读书·新知三联书店",
        "广西师范大学出版社",
        "作家出版社",
        "上海译文出版社",
        "中华书局",
        "译林出版社",
        "南海出版公司",
        "北京大学出版社",
        "商务印书馆",
        "重庆大学出版社",
        "上海人民出版社",
        "浙江文艺出版社"
    ]
    TOP_AUTHORS = [
        "东野圭吾",
        "村上春树",
        "金庸",
        "三毛",
        "王小波",
        "鲁迅",
        "阿加莎·克里斯蒂",
        "莫言",
        "张爱玲",
        "余华",
        "钱钟书",
        "严歌苓",
        "韩寒",
        "刘慈欣",
        "太宰治",
        "桐华",
        "杨绛",
        "马尔克斯",
        "乔治·奥威尔"
    ]

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("### 📝 输入图书信息")

        # 出版社：下拉推荐 + 自定义
        pub_options = TOP_PUBLISHERS + ["✏️ 其他（手动输入）"]
        pub_choice = st.selectbox("出版社（热门推荐）", pub_options, index=1)
        if pub_choice == "✏️ 其他（手动输入）":
            publisher = st.text_input("请输入出版社名称", placeholder="例如：机械工业出版社")
        else:
            publisher = pub_choice
            st.caption(f"已选：{publisher}")

        # 作者：下拉推荐 + 自定义
        author_options = TOP_AUTHORS + ["✏️ 其他（手动输入）"]
        author_choice = st.selectbox("作者（热门推荐）", author_options, index=9)
        if author_choice == "✏️ 其他（手动输入）":
            author = st.text_input("请输入作者名称", placeholder="例如：当年明月")
        else:
            author = author_choice
            st.caption(f"已选：{author}")

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

elif page == "💡 更多发现":
    st.title("💡 更多发现")

    if (FIG_DIR / "13_wordcloud.png").exists():
        st.markdown("### ☁️ 高分图书书名词云")
        st.image(str(FIG_DIR / "13_wordcloud.png"), use_container_width=True)


    st.markdown("---")
    st.markdown("### 📚 装帧类型分析")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bd = pd.read_csv(DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
        bd = bd[bd["crawl_status"] == "success"].copy()
        bd["bt"] = bd["binding"].fillna("未知").apply(
            lambda x: "平装" if "平装" in str(x) else ("精装" if "精装" in str(x) else "其他"))
        bc1, bc2, bc3 = st.columns(3)
        for bt, col in zip(["平装", "精装", "其他"], [bc1, bc2, bc3]):
            cnt = int((bd["bt"] == bt).sum())
            avg = bd[bd["bt"] == bt]["Rating"].mean() if cnt > 0 else 0
            col.metric(bt, f"{cnt} 本", delta=f"均分 {avg:.2f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        types = ["平装", "精装", "其他"]
        ratings = [bd[bd["bt"] == t]["Rating"].mean() for t in types]
        counts = [int((bd["bt"] == t).sum()) for t in types]
        colors_b = ["#3498DB", "#E74C3C", "#95A5A6"]
        bars = ax.bar(types, ratings, color=colors_b)
        for bar, r, c in zip(bars, ratings, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{r:.2f}\n({c}本)", ha="center", fontsize=10)
        ax.set_ylabel("平均评分")
        ax.set_title("装帧类型与评分关系")
        ax.set_ylim(0, 10)
        st.pyplot(fig)
        plt.close()
        st.markdown("#### 精装高分图书 Top 5")
        jz = bd[(bd["bt"] == "精装") & (bd["Rating"] >= 9)].nlargest(5, "Votes")
        jz_disp = jz[["Title", "Rating", "price", "publisher"]]
        jz_disp.columns = ["书名", "评分", "价格", "出版社"]
        jz_disp.index = range(1, len(jz_disp) + 1)
        st.dataframe(jz_disp.style.format({"评分": "{:.1f}"}), use_container_width=True)
    except Exception as e:
        st.info("装帧数据加载中...")
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
