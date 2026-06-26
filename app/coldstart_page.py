import streamlit as st
import pandas as pd
import plotly.express as px

def show(csp):
    st.title("🧊 新书评分预测")
    st.info("📌 冷启动预测功能——基于统计特征预测新书评分。结果仅供参考。")

    if csp is None:
        st.warning("模型未加载，请先运行 src/coldstart_predictor.py")
        return

    all_publishers = sorted(csp.get_publisher_list(min_books=5))
    all_authors = sorted(csp.get_author_list(min_books=3))

    st.markdown("### 📝 输入新书信息")
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("书名（可选）", key="cs_title")
        author_input = st.selectbox("作者（热门推荐）", ["✏️ 手动输入"] + all_authors[:30], key="cs_author_sel")
        if author_input == "✏️ 手动输入":
            author = st.text_input("请输入作者", key="cs_author_custom")
        else:
            author = author_input

        pub_input = st.selectbox("出版社（热门推荐）", ["✏️ 手动输入"] + all_publishers[:30], key="cs_pub_sel")
        if pub_input == "✏️ 手动输入":
            publisher = st.text_input("请输入出版社", key="cs_pub_custom")
        else:
            publisher = pub_input

    with col2:
        pub_year = st.slider("出版年份", 1900, 2030, 2025, key="cs_year")
        pages = st.slider("预计页数", 50, 2000, 300, step=10, key="cs_pages")
        binding = st.selectbox("装帧", ["平装", "精装", "其他"], key="cs_binding")
        votes_est = st.slider("预计评价人数", 10, 500000, 5000, step=100, key="cs_votes",
                              help="预估的评分人数")

    is_translation = st.checkbox("📖 翻译作品（有译者或原版书名）", key="cs_trans")
    is_series = st.checkbox("📚 系列作品（属于丛书系列）", key="cs_series")

    predict_btn = st.button("🚀 预测评分", type="primary", use_container_width=True, key="cs_predict_btn")

    if predict_btn:
        if not author or not publisher:
            st.error("请输入作者和出版社")
        else:
            with st.spinner("模型预测中..."):
                pred, lower, upper, X, similar = csp.predict(
                    author=author, publisher=publisher,
                    pub_year=pub_year, pages=pages, binding=binding,
                    is_translation=is_translation, is_series=is_series,
                    votes_estimate=votes_est
                )

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(f"""
                <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#11998e,#38ef7d);border-radius:16px;color:white;">
                    <div style="font-size:1.1em;margin-bottom:10px;">预测豆瓣评分</div>
                    <div style="font-size:4em;font-weight:900;">{pred:.1f}</div>
                    <div style="font-size:1em;opacity:0.9;">满分 10.0</div>
                    <div style="font-size:0.9em;margin-top:8px;">区间 [{lower:.1f} - {upper:.1f}]</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**📋 输入摘要**")
                if title:
                    st.caption(f"书名: {title}")
                st.caption(f"作者: {author}")
                st.caption(f"出版社: {publisher}")
                st.caption(f"年份: {pub_year} | 页数: {pages}")
                st.caption(f"装帧: {binding} | 翻译: {'是' if is_translation else '否'} | 系列: {'是' if is_series else '否'}")

            with res_col2:
                fi = csp.get_feature_importance()
                if fi:
                    st.markdown("**📊 特征重要性**")
                    name_map = {
                        "author_avg_rating": "作者平均评分",
                        "votes_log": "评价人数(log)",
                        "pages_log": "页数(log)",
                        "author_book_count_log": "作者作品数(log)",
                        "pub_avg_rating": "出版社平均评分",
                        "pub_std_rating": "出版社评分波动",
                        "binding_score": "装帧评分",
                        "pub_book_count_log": "出版社作品数(log)",
                        "pub_year": "出版年份",
                        "is_series": "是否系列作品",
                        "is_translation": "是否翻译作品",
                    }
                    fi_df = pd.DataFrame(fi)
                    fi_df["feature_cn"] = fi_df["feature"].map(name_map)
                    fig = px.bar(fi_df, x="importance", y="feature_cn", orientation="h",
                                 title="特征对预测的影响程度",
                                 color="importance", color_continuous_scale="Greens")
                    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**📚 最相似的已有书籍**")
                sim_data = []
                for sb in similar:
                    sim_data.append({
                        "书名": sb["title"][:30],
                        "评分": f"{sb.get('rating', '?'):.1f}" if sb.get("rating") else "?",
                        "作者": sb.get("author", "?")[:15],
                        "相似度": f"{sb['similarity']:.3f}",
                    })
                if sim_data:
                    st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("🧠 模型信息", expanded=False):
        m = csp.metrics
        st.markdown(f"""
        **GradientBoosting 回归模型**
        - 训练样本: **{m.get('n_samples', 0):,}** 本豆瓣图书
        - MAE: **{m.get('MAE', 0):.3f}** (平均预测误差)
        - R²: **{m.get('R2', 0):.3f}** (拟合优度)
        - 5折交叉验证 R²: **{m.get('CV_R2', 0):.3f}**
        - 特征维度: 11 (出版社/作者统计 + 年份/页数/装帧/翻译/系列)
        - 置信区间: 分位数回归 (5%-95%)
        """)
