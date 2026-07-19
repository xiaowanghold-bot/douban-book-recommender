# -*- coding: utf-8 -*-
import pandas as pd
"""
流派搜索模块 - 关键词匹配 + 评分排序
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

GENRE_KEYWORDS = {
    "小说":     ["小说", "故事"],
    "外国文学":  ["外国文学", "欧美", "英国", "法国", "俄国", "日本文学", "诺贝尔"],
    "中国文学":  ["中国文学", "古典", "当代", "鲁迅", "诗词", "散文"],
    "古典文学":  ["古典", "古代", "唐宋", "明清", "文言", "诗经", "楚辞", "名著"],
    "散文随笔":  ["随笔", "散文", "日记", "游记", "随想"],
    "科幻":     ["科幻", "三体", "刘慈欣", "基地", "阿西莫夫", "星际", "外星", "太空", "宇宙航行"],
    "奇幻":     ["奇幻", "魔法", "哈利波特", "魔戒", "冰与火", "龙枪", "地海"],
    "推理":     ["推理", "侦探", "东野圭吾", "阿加莎", "福尔摩斯", "凶案", "密室", "破案"],
    "悬疑":     ["悬疑", "惊悚", "恐怖", "鬼吹灯", "盗墓"],
    "历史":     ["历史", "中国史", "世界史", "朝代", "战争", "罗马", "明史"],
    "传记":     ["传记", "回忆录", "自传", "生平"],
    "哲学":     ["哲学", "尼采", "康德", "柏拉图", "存在主义"],
    "心理学":   ["心理学", "弗洛伊德", "自卑", "焦虑", "内向"],
    "经济学":   ["经济学", "博弈论", "资本论", "国富论", "货币"],
    "管理":     ["管理", "领导力", "高效", "从优秀到卓越"],
    "个人成长":  ["成长", "自律", "习惯", "励志", "成功"],
    "亲子教育":  ["教育", "育儿", "孩子", "父母", "家教"],
    "健康":     ["健康", "养生", "中医", "睡眠", "饮食"],
    "旅行":     ["旅行", "旅游", "游记"],
    "艺术":     ["艺术", "绘画", "美学", "画家"],
    "设计":     ["设计", "配色", "版式"],
    "摄影":     ["摄影", "照片", "相机"],
    "音乐":     ["音乐", "乐理", "钢琴", "吉他", "贝多芬"],
    "科普":     ["科普", "科学", "量子", "相对论", "进化论", "时间简史"],
    "计算机":   ["计算机", "编程", "Python", "Java", "算法", "人工智能", "代码"],
    "数学":     ["数学", "几何", "微积分", "概率论"],
}

GENRE_GROUPS = {
    "📖 文学":    ["小说", "外国文学", "中国文学", "古典文学", "散文随笔"],
    "🚀 科幻奇幻": ["科幻", "奇幻"],
    "🔍 悬疑推理": ["推理", "悬疑"],
    "📜 历史人文": ["历史", "传记"],
    "🎓 社科学术": ["哲学", "心理学", "经济学"],
    "💼 经管励志": ["管理", "个人成长"],
    "🌱 生活":     ["亲子教育", "健康", "旅行"],
    "🎨 艺术":     ["艺术", "设计", "摄影", "音乐"],
    "🔬 科技":     ["科普", "计算机", "数学"],
}


def build_genre_search_index(df, desc_json_path="data/processed/book_descriptions.json"):
    """预构建文本索引"""
    import json
    descs = {}
    if Path(desc_json_path).exists():
        with open(desc_json_path, "r", encoding="utf-8") as f:
            descs = json.load(f)
    
    texts = {}
    for _, row in df.iterrows():
        bid = str(int(row["id"]))
        title = str(row.get("title", ""))
        desc = descs.get(bid, "")
        texts[int(row["id"])] = title + " " + desc
    
    print(f"[GenreSearch] Index: {len(texts)} books")
    return texts


def search_books_by_genre(genre_name, df, text_index, top_n=30, min_votes=10):
    """语义搜索优先 + 关键词匹配 fallback"""
    results = []
    
    # 优先使用语义搜索 (recommend_by_title)
    try:
        from src.recommendation import BookRecommender
        rec = BookRecommender()
        rec._load_artifacts()
        semantic_results = rec.recommend_by_title(genre_name, top_n=top_n)
        if not semantic_results.empty and len(semantic_results) >= 5:
            # 语义结果足够，直接返回
            for _, row in semantic_results.iterrows():
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "rating": row["rating"],
                    "votes": row["votes"],
                    "bayesian_score": row.get("bayesian_score", 0),
                    "similarity": row["similarity"],
                })
            return pd.DataFrame(results[:top_n]) if results else pd.DataFrame()
        elif not semantic_results.empty:
            # 语义结果 <5 条，保留并补充
            for _, row in semantic_results.iterrows():
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "rating": row["rating"],
                    "votes": row["votes"],
                    "bayesian_score": row.get("bayesian_score", 0),
                    "similarity": row["similarity"],
                })
            set(r["id"] for r in results)
    except Exception as e:
        print(f"[genre_search] 语义搜索失败, 回退到关键词: {e}")
    
    # Fallback: GENRE_KEYWORDS 关键词匹配 (仅在语义结果不足时补充)
    GENRE_KEYWORDS.get(genre_name, [genre_name])
