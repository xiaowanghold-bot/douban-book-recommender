"""评分预测页面。"""

import streamlit as st


TOP_PUBLISHERS = [
    "上海文艺出版社", "人民文学出版社", "上海出版公司", "新星出版社",
    "吉林出版社", "北京师范大学出版社", "江苏文艺出版社", "中信出版社",
    "生活·读书·新知三联书店", "广西师范大学出版社", "作家出版社",
    "上海译文出版社", "中华书局", "译林出版社", "南海出版公司",
    "北京大学出版社", "商务印书馆", "重庆大学出版社", "上海人民出版社", "浙江文艺出版社",
]

TOP_AUTHORS = [
    "东野圭吾", "村上春树", "金庸", "三毛", "王小波", "鲁迅",
    "阿加莎·克里斯蒂", "莫言", "张爱玲", "余华", "钱钟书",
    "严歌苓", "韩寒", "刘慈欣", "太宰治", "桐华", "杨绛",
    "马尔克斯", "乔治·奥威尔",
]


def show(predictor, fallback_metrics):
    """渲染评分预测页，模型未加载时使用元数据中的指标。"""
    st.title("🔮 图书评分预测")
    st.markdown("*输入图书信息，AI 模型预测豆瓣评分*")

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("### 📝 输入图书信息")

        pub_options = TOP_PUBLISHERS + ["✏️ 其他（手动输入）"]
        pub_choice = st.selectbox("出版社（热门推荐）", pub_options, index=1, key="pred_pub")
        if pub_choice == "✏️ 其他（手动输入）":
            publisher = st.text_input(
                "请输入出版社名称",
                placeholder="例如：机械工业出版社",
                key="pred_pub_custom",
            )
        else:
            publisher = pub_choice
            st.caption("已选：{0}".format(publisher))

        author_options = TOP_AUTHORS + ["✏️ 其他（手动输入）"]
        author_choice = st.selectbox("作者（热门推荐）", author_options, index=9, key="pred_author")
        if author_choice == "✏️ 其他（手动输入）":
            author = st.text_input(
                "请输入作者名称",
                placeholder="例如：陈忠实",
                key="pred_author_custom",
            )
        else:
            author = author_choice
            st.caption("已选：{0}".format(author))

        price = st.number_input(
            "定价（元）", min_value=0.0, max_value=999.0, value=39.5, step=0.5, key="pred_price"
        )
        year = st.number_input(
            "出版年份", min_value=1900, max_value=2026, value=2014, step=1, key="pred_year"
        )
        pages = st.number_input(
            "页数", min_value=10, max_value=5000, value=300, step=10, key="pred_pages"
        )
        votes = st.number_input(
            "评价人数（预估）",
            min_value=0,
            max_value=5000000,
            value=50000,
            step=1000,
            key="pred_votes",
        )
        binding_choice = st.selectbox("装帧", ["平装", "精装", "其他"], key="pred_binding")
        predict_btn = st.button(
            "🚀 开始预测", type="primary", use_container_width=True, key="pred_btn"
        )

    with rc2:
        st.markdown("### 🎯 预测结果")
        if predict_btn:
            if predictor is not None:
                with st.spinner("模型预测中..."):
                    pred_score = predictor.predict(
                        price=price,
                        year=year,
                        pages=pages,
                        votes=votes,
                        author=author,
                        publisher=publisher,
                        binding=binding_choice,
                    )
                if pred_score is not None:
                    st.markdown(
                        """
                        <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;color:white;">
                            <div style="font-size:1.2em;margin-bottom:10px;">预测豆瓣评分</div>
                            <div style="font-size:4em;font-weight:900;">{0:.1f}</div>
                            <div style="font-size:1em;opacity:0.8;">满分 10.0</div>
                        </div>
                        """.format(pred_score),
                        unsafe_allow_html=True,
                    )

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

    st.markdown("---")
    with st.expander("🧠 模型信息（点击展开）", expanded=False):
        metrics = predictor.metrics if predictor is not None else fallback_metrics
        st.markdown(f"""
        **RandomForest 回归模型（v3）**
        - 独立测试集 RMSE: **{metrics.get('RMSE', 0):.3f}** (vs 作者均值基线 {metrics.get('author_baseline_RMSE', 0):.3f})
        - 独立测试集 MAE: **{metrics.get('MAE', 0):.3f}** (vs 作者均值基线 {metrics.get('author_baseline_MAE', 0):.3f})
        - 独立测试集 R²: **{metrics.get('R2', 0):.3f}**
        - 嵌套5折 CV R²: **{metrics.get('CV_R2', 0):.3f} ± {metrics.get('CV_R2_std', 0):.3f}**
        - 特征: price, year, pages, votes_log, author_mean, publisher_mean, binding_mean
        - 类别统计: 5折 OOF 目标均值编码
        - 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5)
        """)
