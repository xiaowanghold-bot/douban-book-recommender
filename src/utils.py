# -*- coding: utf-8 -*-
"""共享工具函数"""

import warnings
from pathlib import Path
import matplotlib
import matplotlib.font_manager as fm

# 中文字体候选路径（按优先级排列）
_CJK_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    # Linux (WQY)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Linux (Noto)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def setup_chinese_font():
    """跨平台设置 matplotlib 中文字体。找到任一可用字体即返回其名称，
    否则发出警告并退回默认字体。"""
    found = None
    for candidate in _CJK_FONT_CANDIDATES:
        if Path(candidate).exists():
            found = candidate
            break

    if found:
        try:
            fm.fontManager.addfont(found)
            prop = fm.FontProperties(fname=found)
            font_name = prop.get_name()
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font_name
        except Exception as e:
            warnings.warn(f"字体加载失败 ({found}): {e}")

    # 回退
    warnings.warn("未找到中文字体，图表中文可能无法正常显示。"
                  "请安装中文字体（如 WenQuanYi Micro Hei 或 SimHei）。")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


def dedup_editions(df):
    """Collapse clear edition variants while preserving distinct volumes/parts."""
    import re
    if df is None or df.empty:
        return df
    d = df.copy()

    edition_markers = re.compile(
        r"原书第[^）)]*版|第[一二三四五六七八九十0-9]+版|"
        r"修订版|增订版|新版|再版|精装|平装|典藏|珍藏|纪念|套装"
    )

    # Only remove explicit edition/format markers. Volumes and parts are content.
    def _norm(s):
        s = str(s).strip()
        s = re.sub(
            r"[（(]([^）)]*)[）)]",
            lambda match: "" if edition_markers.search(match.group(1)) else match.group(1),
            s,
        )
        s = re.sub(r"[,\.\uff0c\u3001\uff1b;:!\uff01\?\uff1f\u300a\u300b\u300c\u300d]", "", s)
        s = re.sub(r"\s+", "", s)  # spaces
        s = re.sub(
            r"(原书第[^）)]*版|第[一二三四五六七八九十0-9]+版|"
            r"修订版|增订版|新版|再版|精装|平装|典藏|珍藏|纪念|套装)$",
            "",
            s,
        )
        return s[:30]
    d["_ntitle"] = d["title"].apply(_norm)
    # Keep highest votes per normalized title
    d = d.sort_values("votes", ascending=False).drop_duplicates(subset=["_ntitle"], keep="first")
    d = d.drop(columns=["_ntitle"])
    return d

def bayesian_shrink(avg, n, C, m):
    """Bayesian shrinkage: pull group mean toward global mean.

    Formula (same structure as book Bayesian score):
        score = (n/(n+m))*avg + (m/(n+m))*C

    n = group sample size (e.g., publisher book count)
    m = P75 of all group sizes (stronger shrinkage than median)
    C = global mean rating
    """
    import numpy as np
    n = np.asarray(n, dtype=float)
    avg = np.asarray(avg, dtype=float)
    return (n / (n + m)) * avg + (m / (n + m)) * C

