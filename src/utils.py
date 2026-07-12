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
