# -*- coding: utf-8 -*-
"""
IJCAI 豆瓣公开数据集整合
=========================
数据集来源: DTCDR (CIKM 2019) / GA-DTCDR (IJCAI 2020) 跨域推荐论文
- books_cleaned.txt:  豆瓣 book_id ↔ 内部 UID 映射 (95,872 本)
- bookreviews_cleaned.txt: user_id, UID, rating 1-5, labels(竖线分隔), comment, time

输出:
- data/processed/book_tags.json:   {豆瓣id: [标签]}
- data/processed/tag_counts.csv:  标签频次表
- data/processed/user_ratings.csv: user_id, douban_book_id, rating, time
"""
import pandas as pd
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_IJCAI = DATA_DIR / "raw" / "douban_ijcai" / "douban_dataset(text information)"
PROCESSED = DATA_DIR / "processed"

# ============================================================
# 1. 加载 IJCAI 数据
# ============================================================
print("=" * 60)
print("  IJCAI 豆瓣数据集整合")
print("=" * 60)

map_df = pd.read_csv(
    RAW_IJCAI / "books_cleaned.txt",
    sep="\t", quoting=1, encoding="utf-8"
)
map_df["book_id"] = map_df["book_id"].astype(str).str.strip('"').str.strip()
map_df["UID"] = map_df["UID"].astype(int)
uid_to_douban = dict(zip(map_df["UID"], map_df["book_id"]))
print(f"[映射] {len(uid_to_douban):,} UID -> 豆瓣 ID")

rev_df = pd.read_csv(
    RAW_IJCAI / "bookreviews_cleaned.txt",
    sep="\t", quoting=1, encoding="utf-8"
)
# 清理引用号
for col in rev_df.columns:
    rev_df[col] = rev_df[col].astype(str).str.strip('"').str.strip()
rev_df["rating"] = pd.to_numeric(rev_df["rating"], errors="coerce")
print(f"[书评] {len(rev_df):,} 条, {rev_df['user_id'].nunique():,} 用户")

# ============================================================
# 2. 还原豆瓣 ID
# ============================================================
rev_df["book_uid"] = pd.to_numeric(rev_df["book_id"], errors="coerce").astype("Int64")
rev_df["douban_id"] = rev_df["book_uid"].map(uid_to_douban)
rev_valid = rev_df[rev_df["douban_id"].notna()].copy()
print(f"[还原] {len(rev_valid):,} 条书评有有效豆瓣 ID")

# ============================================================
# 3. 重合计算
# ============================================================
our_df = pd.read_csv(PROCESSED / "books_scored.csv", encoding="utf-8-sig")
our_ids = set(str(int(i)) for i in our_df["id"])

ijcai_ids = set(rev_valid["douban_id"].unique())
overlap = our_ids & ijcai_ids

print("\n--- 重合统计 ---")
print(f"  本项目图书数: {len(our_ids):,}")
print(f"  IJCAI 图书数: {len(ijcai_ids):,}")
print(f"  重合数:       {len(overlap):,}")
print(f"  重合/本项目:   {len(overlap)/len(our_ids)*100:.1f}%")
print(f"  重合/IJCAI:   {len(overlap)/len(ijcai_ids)*100:.1f}%")

# ============================================================
# 4. 生成 book_tags.json
# ============================================================
rev_overlap = rev_valid[rev_valid["douban_id"].isin(overlap)].copy()

# 解析 labels（竖线分隔），按书聚合去重
book_tags = {}
all_tags = []
for _, row in rev_overlap.iterrows():
    did = row["douban_id"]
    labels_raw = row["labels"]
    if pd.isna(labels_raw) or not labels_raw or labels_raw == "nan":
        continue
    tags = [t.strip() for t in str(labels_raw).split("|") if t.strip()]
    all_tags.extend(tags)
    if did not in book_tags:
        book_tags[did] = set()
    book_tags[did].update(tags)

# 转 list
book_tags_list = {k: sorted(v) for k, v in book_tags.items()}
with open(PROCESSED / "book_tags.json", "w", encoding="utf-8") as f:
    json.dump(book_tags_list, f, ensure_ascii=False, indent=2)
print(f"\n[book_tags.json] {len(book_tags_list)} 本书, {len(set(all_tags)):,} 个独立标签")

# ============================================================
# 5. 生成 tag_counts.csv
# ============================================================
tag_counter = Counter(all_tags)
tag_df = pd.DataFrame(
    [{"tag": t, "count": c} for t, c in tag_counter.most_common()],
)
tag_df.to_csv(PROCESSED / "tag_counts.csv", index=False, encoding="utf-8-sig")
print(f"[tag_counts.csv] {len(tag_df)} 个标签")
print(f"  Top 10: {tag_df.head(10).to_dict('records')}")

# ============================================================
# 6. 生成 user_ratings.csv
# ============================================================
user_ratings = rev_overlap[["user_id", "douban_id", "rating", "time"]].copy()
user_ratings.columns = ["user_id", "douban_book_id", "rating", "time"]
user_ratings.to_csv(PROCESSED / "user_ratings.csv", index=False, encoding="utf-8-sig")
print(f"[user_ratings.csv] {len(user_ratings):,} 条记录, {user_ratings['user_id'].nunique():,} 用户")

# ============================================================
# 7. 用户统计（供实验E参考）
# ============================================================
user_counts = user_ratings.groupby("user_id").size()
active_users = user_counts[user_counts >= 10]
print("\n--- 用户活跃度 ---")
print(f"  总用户: {len(user_counts):,}")
print(f"  >=10 条评分: {len(active_users):,}")
print(f"  >=20 条评分: {(user_counts >= 20).sum():,}")
print(f"  >=50 条评分: {(user_counts >= 50).sum():,}")

print(f"\n{'='*60}")
print("  整合完成!")
print(f"{'='*60}")
