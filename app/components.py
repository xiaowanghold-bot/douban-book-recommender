# -*- coding: utf-8 -*-
"""共享 UI 组件 — 封面渲染与图书详情面板"""

import streamlit as st
import base64


def render_cover(book_id, cover_map, cover_dir, verified_covers, width=110):
    """渲染图书封面图片（统一尺寸）。
    
    Args:
        book_id: 图书 ID (int)
        cover_map: {book_id_str: filename} dict
        cover_dir: Path to covers directory
        verified_covers: set of verified book IDs
        width: image width in pixels
    
    Returns:
        str cover path if found, else None
    """
    bid = int(book_id)
    if bid not in verified_covers:
        return None
    
    fname = cover_map.get(str(bid), "")
    if fname:
        full = cover_dir / fname
        if full.exists():
            st.image(str(full), width=width)
            return str(full)
    
    for ext in ("jpg", "png", "webp"):
        f = cover_dir / f"{bid}.{ext}"
        if f.exists():
            st.image(str(f), width=width)
            return str(f)
    
    return None


def cover_to_base64(book_id, cover_map, cover_dir, verified_covers):
    """将封面转为 base64 data URI（用于 HTML 卡片）。
    
    Returns:
        base64 string or empty string if not found
    """
    bid = int(book_id)
    if bid not in verified_covers:
        return ""
    
    fname = cover_map.get(str(bid), "")
    if fname:
        full = cover_dir / fname
        if full.exists():
            with open(full, "rb") as f:
                return base64.b64encode(f.read()).decode()
    
    for ext in ("jpg", "png", "webp"):
        f = cover_dir / f"{bid}.{ext}"
        if f.exists():
            with open(f, "rb") as fh:
                return base64.b64encode(fh.read()).decode()
    
    return ""


def render_book_detail(book_id, book_title, book_rating, book_votes,
                       get_detail_fn, get_desc_fn, cover_map, cover_dir,
                       verified_covers, close_key="close_detail"):
    """渲染图书详情面板（作者/出版社/价格/ISBN/简介）。
    
    在 st.columns 布局中展示封面+元数据+简介，并提供一个关闭按钮。
    
    Args:
        book_id: 图书 ID
        book_title: 书名
        book_rating: 评分
        book_votes: 评价人数
        get_detail_fn: 获取详情的函数，签名 (book_id) -> dict
        get_desc_fn: 获取简介的函数，签名 (book_id) -> str
        cover_map: 封面映射 dict
        cover_dir: 封面目录 Path
        verified_covers: 已验证封面 ID set
        close_key: 关闭按钮的 streamlit key
    """
    import streamlit as st
    
    info = get_detail_fn(book_id)
    desc = get_desc_fn(book_id)
    
    dc1, dc2 = st.columns([1, 3])
    with dc1:
        render_cover(book_id, cover_map, cover_dir, verified_covers, width=120)
    with dc2:
        st.markdown(f"**{book_title}**")
        st.caption(f"⭐{book_rating:.1f} | {int(book_votes):,}人")
        for k in ("author", "publisher", "pub_year", "price", "isbn"):
            v = info.get(k, "")
            if v and v != "nan":
                st.caption(f"{k}: {v}")
    if desc:
        st.markdown(f"> {desc[:300]}")
    if st.button("❌ 关闭详情", key=close_key):
        st.session_state.pop("selected_book_id", None)
        st.rerun()
