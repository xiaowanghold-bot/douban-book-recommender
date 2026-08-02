"""图书搜索与智能推荐页面。"""

import streamlit as st

from components import render_cover


QUICK_TAGS = [
    "全部",
    "小说",
    "文学",
    "历史",
    "哲学",
    "科幻",
    "推理",
    "爱情",
    "武侠",
    "心理",
    "经济",
    "漫画",
    "诗歌",
    "传记",
    "散文",
    "悬疑",
]


def show(rec, cover_map, cover_dir, verified_covers, get_detail_info, get_desc):
    """渲染图书搜索、详情及两类推荐结果。"""
    st.title("🔍 搜书 & 智能推荐")

    sc1, sc2 = st.columns([1, 3])
    with sc1:
        tag_sel = st.selectbox("🏷️ 标签筛选", QUICK_TAGS, key="search_tag")
    with sc2:
        hint = (
            "输入书名关键词..."
            if tag_sel == "全部"
            else "筛选「{0}」类图书，也可输入关键词".format(tag_sel)
        )
        st.caption("💡 {0}".format(hint))

    search_term = st.text_input(
        "🔍 输入书名或关键词",
        placeholder="例如：三体、活着、百年孤独...",
        key="search_box",
    )
    if tag_sel != "全部" and not search_term:
        search_term = tag_sel

    if not search_term:
        return

    with st.spinner("搜索中..."):
        results = rec.recommend_by_title(search_term, top_n=20)

    if results.empty:
        st.warning("未找到相关图书，请尝试其他关键词")
        return

    st.caption("🔍 找到 {0} 本匹配图书：".format(len(results)))
    match_cols = st.columns(4)
    for i, (_, match) in enumerate(results.iterrows()):
        with match_cols[i % 4]:
            render_cover(match["id"], cover_map, cover_dir, verified_covers, width=110)
            btn_label = "{0} ⭐{1:.1f}".format(str(match["title"])[:22], match["rating"])
            if st.button(
                btn_label,
                key="suggest_{0}".format(match["id"]),
                use_container_width=True,
                help="{0:,}人评价".format(int(match["votes"])),
            ):
                st.session_state.selected_book_id = int(match["id"])
                st.session_state.selected_book_title = match["title"]
                st.session_state.selected_book_rating = match["rating"]
                st.session_state.selected_book_votes = match["votes"]
                st.rerun()

    if not st.session_state.get("selected_book_id"):
        return

    book_id = st.session_state.selected_book_id
    st.markdown("---")
    st.markdown("### 📖 {0}".format(st.session_state.get("selected_book_title", "")))

    info = get_detail_info(book_id)
    desc = get_desc(book_id)
    dc1, dc2 = st.columns([1, 3])
    with dc1:
        render_cover(book_id, cover_map, cover_dir, verified_covers, width=200)
    with dc2:
        st.markdown(
            "⭐ {0:.1f} | 👥 {1:,}人评价".format(
                st.session_state.get("selected_book_rating", 0),
                int(st.session_state.get("selected_book_votes", 0)),
            )
        )
        for key in ["author", "publisher", "pub_year", "price", "pages", "binding", "isbn"]:
            value = info.get(key, "")
            if value and value != "nan":
                st.caption("{0}: {1}".format(key, value))
    if desc:
        st.markdown("**📝 简介**: {0}".format(desc[:400]))
    if st.button("❌ 关闭详情", key="close_search_detail"):
        st.session_state.selected_book_id = None
        st.rerun()

    st.markdown("---")
    t1, t2 = st.tabs(["📚 内容推荐", "🔀 混合推荐"])
    with t1:
        recs = rec.recommend_by_id(book_id, top_n=10)
        for index, (_, recommendation) in enumerate(recs.iterrows(), start=1):
            st.markdown(
                "**{0}. {1}** ⭐{2:.1f} 📊{3:,}人 `{4:.2%}`".format(
                    index,
                    recommendation["title"],
                    recommendation["rating"],
                    int(recommendation["votes"]),
                    recommendation["similarity"],
                )
            )
            st.progress(float(recommendation["similarity"]))
    with t2:
        hybrid_recs = rec.hybrid_recommend(book_id, top_n=10)
        for index, (_, recommendation) in enumerate(hybrid_recs.iterrows(), start=1):
            st.markdown(
                "**{0}. {1}** ⭐{2:.1f} 📊{3:,}人 `{4:.4f}`".format(
                    index,
                    recommendation["title"],
                    recommendation["rating"],
                    int(recommendation["votes"]),
                    recommendation["hybrid_score"],
                )
            )
            st.progress(float(recommendation["hybrid_score"]))

    st.markdown("---")
    st.caption("📥 导出推荐结果")
    ec1, ec2 = st.columns(2)
    with ec1:
        recs_csv = recs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 内容推荐 CSV",
            recs_csv,
            "content_recs.csv",
            "text/csv",
            key="dl_c",
            use_container_width=True,
        )
    with ec2:
        hybrid_csv = hybrid_recs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 混合推荐 CSV",
            hybrid_csv,
            "hybrid_recs.csv",
            "text/csv",
            key="dl_h",
            use_container_width=True,
        )
