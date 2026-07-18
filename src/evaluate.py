"""
离线评估脚本 — 豆瓣图书推荐系统
================================
实验 A: 同系列 Recall@K
实验 B: 评分预测正规评估
实验 C: 冷启动模型泄露检查

用法: python -m src.evaluate --experiment all|rec|rating|coldstart
"""
import argparse
import sys
import warnings
import time
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
#  共享工具
# ============================================================================

def load_recommender():
    """加载已构建的 BookRecommender（含预计算最近邻）"""
    sys.path.insert(0, str(ROOT))
    from src.recommendation import BookRecommender
    rec = BookRecommender()
    rec._load_artifacts()
    # 加载预计算的最近邻
    nn_path = rec.model_dir / "nn_neighbors.npz"
    if nn_path.exists():
        data = np.load(nn_path)
        rec.nn_distances = data["distances"]
        rec.nn_indices = data["indices"]
        print(f"[加载] 预计算最近邻 ({rec.nn_indices.shape[1]} 近邻)")
    else:
        rec.build_nn_index(n_neighbors=30)
    return rec


def load_books_detail():
    """加载 Books_detail.csv"""
    df = pd.read_csv(RAW_DIR / "Books_detail.csv", encoding="utf-8-sig")
    # 统一列名
    df = df.rename(columns={"ID": "id", "Rating": "rating", "Votes": "votes",
                             "Title": "title"})
    df = df[df["crawl_status"] == "success"].copy()
    for col in ["rating", "votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================================
#  实验 A: 同系列 Recall@K
# ============================================================================

def run_experiment_a():
    """同系列 Recall@K 评估（使用预计算 NN，避免逐次调用 API）"""
    print("\n" + "=" * 70)
    print("  实验 A: 同系列 Recall@K")
    print("=" * 70)

    detail = load_books_detail()
    rec_df = pd.read_csv(MODEL_DIR / "books_for_rec.csv", encoding="utf-8-sig")
    rec_ids = set(rec_df["id"])

    detail_in_rec = detail[detail["id"].isin(rec_ids)].copy()
    has_series = detail_in_rec[detail_in_rec["series"].notna() & (detail_in_rec["series"].str.strip() != "")]

    series_counts = has_series.groupby("series")["id"].count()
    valid_series = series_counts[series_counts >= 2].index
    eval_books = has_series[has_series["series"].isin(valid_series)].copy()

    n_queries = len(eval_books)
    n_series = len(valid_series)
    print(f"  评估查询书数: {n_queries}", flush=True)
    print(f"  系列数: {n_series}", flush=True)

    if n_queries < 100:
        print(f"  [WARN] 样本量 < 100，结果仅供参考")

    # 构建系列内成员映射
    series_members = {}
    for _, row in eval_books.iterrows():
        s = row["series"]
        if s not in series_members:
            series_members[s] = set()
        series_members[s].add(int(row["id"]))

    # 加载推荐引擎 + 预计算 NN
    rec = load_recommender()
    nn_indices = rec.nn_indices  # shape: (n_books, n_neighbors)
    nn_distances = rec.nn_distances

    # book_id -> matrix_idx 映射
    id_to_idx = rec.id_to_idx
    idx_to_id = rec.idx_to_id

    # 预取标题数组加速
    titles_arr = rec.df["title"].values

    # 按 bayesian_score 排序（用于 Popular baseline）
    popular_ids = rec_df.nlargest(len(rec_df), "bayesian_score")["id"].tolist()
    all_ids_set = set(int(x) for x in rec_df["id"])

    K_values = [10, 20]
    results = {}

    # 预计算每个查询在 NN 中的索引和它在 popular 中的排名
    # 构建 series book -> 该系列其他成员的 set 映射（加速）
    book_to_series_others = {}
    for _, row in eval_books.iterrows():
        bid = int(row["id"])
        book_to_series_others[bid] = series_members[row["series"]] - {bid}

    for K in K_values:
        print(f"\n  --- K = {K} ---")

        # 方法1: recommend_by_id（直接使用预计算 NN）
        recall1 = []
        n_evaled = 0
        for _, row in eval_books.iterrows():
            bid = int(row["id"])
            series_others = book_to_series_others.get(bid, set())
            if not series_others:
                continue
            n_evaled += 1
            denominator = min(K, len(series_others))
            try:
                midx = id_to_idx.get(bid)
                if midx is None:
                    recall1.append(0.0)
                    continue
                # 取前 search_range 个邻居（排除自己，去重同书名）
                neighbors = nn_indices[midx]
                seen_titles = set()
                hits = 0
                count = 0
                for nidx in neighbors:
                    nid = idx_to_id.get(nidx)
                    if nid == bid:
                        continue
                    if nid is None:
                        continue
                    title = str(titles_arr[nidx]) if nidx < len(titles_arr) else ""
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    if int(nid) in series_others:
                        hits += 1
                    count += 1
                    if count >= K:
                        break
                recall1.append(hits / denominator)
            except Exception:
                recall1.append(0.0)
        mean_recall1 = np.mean(recall1) if recall1 else 0.0
        print(f"  recommend_by_id    : Recall@{K} = {mean_recall1:.4f}  (n={len(recall1)})")

        # 方法2: Random baseline (使用 numpy 直接从全库ID数组采样，避免重复分配大列表)
        recall_random_trials = []
        all_ids_arr = np.array(sorted(all_ids_set), dtype=np.int64)
        for trial in range(5):
            rng = np.random.RandomState(42 + trial)
            trial_recalls = []
            for _, row in eval_books.iterrows():
                bid = int(row["id"])
                series_others = book_to_series_others.get(bid, set())
                if not series_others:
                    continue
                denominator = min(K, len(series_others))
                # 从全库随机取 K+1 本（万一抽到自身则多取），去掉自身后取 K 本
                sample_size = min(K + 1, len(all_ids_arr))
                sampled = rng.choice(all_ids_arr, size=sample_size, replace=False)
                sampled = [int(s) for s in sampled if int(s) != bid][:K]
                hits = len(set(sampled) & series_others)
                trial_recalls.append(hits / denominator)
            recall_random_trials.append(np.mean(trial_recalls) if trial_recalls else 0.0)
        mean_random = np.mean(recall_random_trials)
        print(f"  Random baseline    : Recall@{K} = {mean_random:.4f}  (avg of 5 trials)")

        # 方法3: Popular baseline
        recall_pop = []
        for _, row in eval_books.iterrows():
            bid = int(row["id"])
            series_others = book_to_series_others.get(bid, set())
            if not series_others:
                continue
            denominator = min(K, len(series_others))
            top = [i for i in popular_ids if i != bid][:K]
            hits = len(set(int(s) for s in top) & series_others)
            recall_pop.append(hits / denominator)
        mean_pop = np.mean(recall_pop) if recall_pop else 0.0
        print(f"  Popular baseline   : Recall@{K} = {mean_pop:.4f}  (n={len(recall_pop)})")

        results[K] = {
            "recommend_by_id": mean_recall1,
            "random": mean_random,
            "popular": mean_pop,
            "n_queries": n_evaled,
        }

    # 生成报告
    lines = [
        "## 实验 A: 同系列 Recall@K",
        "",
        "> ⚠ **方法局限性说明**: 本项目推荐引擎基于字符级 n-gram TF-IDF + 余弦相似度，",
        "> 同系列书名（如《三体》/《三体II》）天然字符重叠度高，Recall 指标会高估实际语义推荐能力。",
        "",
        f"- 查询书数: {n_queries}",
        f"- 系列数: {n_series}",
        f"- 系列平均规模: {n_queries / n_series:.1f} 本",
        "",
        "| 方法 | Recall@10 | Recall@20 | 有效查询数 |",
        "|------|-----------|-----------|-----------|",
    ]
    r10 = results[10]
    r20 = results[20]
    lines.append(
        f"| recommend_by_id | {r10['recommend_by_id']:.4f} | {r20['recommend_by_id']:.4f} | {r10['n_queries']} |"
    )
    lines.append(
        f"| Random | {r10['random']:.4f} | {r20['random']:.4f} | {r10['n_queries']} |"
    )
    lines.append(
        f"| Popular | {r10['popular']:.4f} | {r20['popular']:.4f} | {r10['n_queries']} |"
    )
    lines.append("")

    return "\n".join(lines), results


# ============================================================================
#  实验 B: 评分预测正规评估
# ============================================================================

def run_experiment_b():
    """评分预测（RandomForest）正规评估"""
    print("\n" + "=" * 70)
    print("  实验 B: 评分预测正规评估")
    print("=" * 70)

    detail = load_books_detail()

    # ===== 特征工程（复制自 enhancements.py RatingPredictor） =====
    def parse_price(text):
        import re
        if pd.isna(text):
            return np.nan
        m = re.search(r"[\d.]+", str(text))
        return float(m.group()) if m else np.nan

    def parse_year(text):
        import re
        if pd.isna(text):
            return np.nan
        m = re.search(r"(19|20)\d{2}", str(text))
        return int(m.group()) if m else np.nan

    df = detail.copy()
    df["price_num"] = df["price"].apply(parse_price)
    df["year_num"] = df["pub_year"].apply(parse_year)
    df["pages_num"] = pd.to_numeric(df["pages"], errors="coerce")

    df["author_clean"] = df["author"].apply(
        lambda x: __import__("re").sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(x)).strip()[:30]
        if pd.notna(x) else "未知")
    df["publisher_clean"] = df["publisher"].fillna("未知").astype(str).str[:20]
    df["binding_type"] = df["binding"].fillna("未知").apply(
        lambda x: "平装" if "平装" in str(x) else ("精装" if "精装" in str(x) else "其他"))

    # 筛选有效记录
    df = df.dropna(subset=["rating", "price_num", "year_num", "pages_num"]).copy()
    df = df[df["year_num"].between(1950, 2025)]
    df = df[df["rating"].between(1, 10)]
    df["pages_num"] = df["pages_num"].fillna(df["pages_num"].median())

    print(f"  有效数据: {len(df)} 条")

    # 编码低频类别
    for col in ["author_clean", "publisher_clean", "binding_type"]:
        counts = df[col].value_counts()
        df[f"{col}_enc"] = df[col].apply(
            lambda x, c=counts: x if c.get(x, 0) >= 3 else "其他")

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}")

    y_train = train_df["rating"].values
    y_test = test_df["rating"].values

    # Baseline 1: 全局均值
    global_mean = y_train.mean()
    pred_global = np.full_like(y_test, global_mean)
    rmse_global = np.sqrt(mean_squared_error(y_test, pred_global))
    mae_global = mean_absolute_error(y_test, pred_global)
    print(f"  全局均值 baseline : RMSE={rmse_global:.4f}, MAE={mae_global:.4f}")

    # Baseline 2: 出版社均值（未见→全局均值）
    pub_means = train_df.groupby("publisher_clean_enc")["rating"].mean().to_dict()
    test_pub_preds = test_df["publisher_clean_enc"].map(pub_means).fillna(global_mean).values
    rmse_pub = np.sqrt(mean_squared_error(y_test, test_pub_preds))
    mae_pub = mean_absolute_error(y_test, test_pub_preds)
    print(f"  出版社均值 baseline: RMSE={rmse_pub:.4f}, MAE={mae_pub:.4f}")

    # Baseline 3: 作者均值（未见→全局均值）
    auth_means = train_df.groupby("author_clean_enc")["rating"].mean().to_dict()
    test_auth_preds = test_df["author_clean_enc"].map(auth_means).fillna(global_mean).values
    rmse_auth = np.sqrt(mean_squared_error(y_test, test_auth_preds))
    mae_auth = mean_absolute_error(y_test, test_auth_preds)
    print(f"  作者均值 baseline  : RMSE={rmse_auth:.4f}, MAE={mae_auth:.4f}")

    # RandomForest（仅用训练集）
    le_author = LabelEncoder()
    le_pub = LabelEncoder()
    le_bind = LabelEncoder()

    X_train = pd.DataFrame({
        "price": train_df["price_num"],
        "year": train_df["year_num"],
        "pages": train_df["pages_num"],
        "votes_log": np.log1p(train_df["votes"]),
        "author_clean": le_author.fit_transform(train_df["author_clean_enc"]),
        "publisher_clean": le_pub.fit_transform(train_df["publisher_clean_enc"]),
        "binding_type": le_bind.fit_transform(train_df["binding_type_enc"]),
    }).values

    # 处理测试集未见类别
    def safe_transform(le, series):
        result = []
        for v in series:
            try:
                result.append(le.transform([v])[0])
            except ValueError:
                result.append(-1)
        return result

    X_test = pd.DataFrame({
        "price": test_df["price_num"],
        "year": test_df["year_num"],
        "pages": test_df["pages_num"],
        "votes_log": np.log1p(test_df["votes"]),
        "author_clean": safe_transform(le_author, test_df["author_clean_enc"]),
        "publisher_clean": safe_transform(le_pub, test_df["publisher_clean_enc"]),
        "binding_type": safe_transform(le_bind, test_df["binding_type_enc"]),
    }).values

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print(f"  RandomForest       : RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}")

    lines = [
        "## 实验 B: 评分预测正规评估",
        "",
        f"- 训练集: {len(train_df):,} 条, 测试集: {len(test_df):,} 条",
        "- 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)",
        "- 特征: price, year, pages, votes_log, author_clean, publisher_clean, binding_type",
        "",
        "| 方法 | RMSE | MAE |",
        "|------|------|-----|",
        f"| 全局均值 | {rmse_global:.4f} | {mae_global:.4f} |",
        f"| 出版社均值 | {rmse_pub:.4f} | {mae_pub:.4f} |",
        f"| 作者均值 | {rmse_auth:.4f} | {mae_auth:.4f} |",
        f"| RandomForest | {rmse_rf:.4f} | {mae_rf:.4f} |",
        "",
    ]

    return "\n".join(lines), {
        "global": (rmse_global, mae_global),
        "publisher": (rmse_pub, mae_pub),
        "author": (rmse_auth, mae_auth),
        "rf": (rmse_rf, mae_rf),
    }


# ============================================================================
#  实验 C: 冷启动模型泄露检查
# ============================================================================

def run_experiment_c():
    """冷启动模型泄露检查"""
    print("\n" + "=" * 70)
    print("  实验 C: 冷启动模型泄露检查")
    print("=" * 70)

    detail = load_books_detail()

    # 清洗
    df = detail.copy()
    df["pub_year_num"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["pages_num"] = pd.to_numeric(df["pages"], errors="coerce")
    df["pub_year_num"] = df["pub_year_num"].fillna(2010).astype(int)
    df["pages_num"] = df["pages_num"].fillna(300).astype(int)
    df["author"] = df["author"].fillna("未知").astype(str)
    df["publisher"] = df["publisher"].fillna("未知").astype(str)
    df["binding"] = df["binding"].fillna("其他").astype(str)
    df = df.dropna(subset=["rating", "votes"])
    df = df[(df["rating"] >= 1) & (df["rating"] <= 10)]
    df = df[df["votes"] >= 10]
    df["is_translation"] = (df["translator"].notna() | df["original_title"].notna()).astype(int)
    df["is_series"] = df["series"].notna().astype(int)

    print(f"  有效数据: {len(df)} 条")

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df["rating"].values
    y_test = test_df["rating"].values
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}")

    global_mean = y_train.mean()
    global_std = y_train.std()

    # ===== 1. 现状检查 =====
    print("\n  --- 现状检查 ---")
    print("  当前 coldstart_predictor.py 的 build_stats() 对全量 self.df 计算统计量，")
    print("  然后 build_features() 映射回每行。因此 pub_avg_rating 和 author_avg_rating")
    print("  都包含目标书本身的评级 → **存在数据泄露**。")
    print("  votes_log 使用训练集全体书的真实投票数 → 对新书(votes≈0)不可用 → **特征泄露**。")

    # ===== 辅助函数：train-only stats（排除自身） =====
    def build_train_stats(train_subset):
        pub = train_subset.groupby("publisher")["rating"].agg(["mean", "count", "std"]).fillna(0)
        pub.columns = ["pub_avg", "pub_cnt", "pub_std"]
        auth = train_subset.groupby("author")["rating"].agg(["mean", "count"]).fillna(0)
        auth.columns = ["auth_avg", "auth_cnt"]
        bind = train_subset.groupby("binding")["rating"].mean().to_dict()
        return pub, auth, bind

    pub_stats, auth_stats, bind_stats = build_train_stats(train_df)

    def featurize_v1(subset, pub_s, auth_s, bind_s, is_train, include_votes):
        """原始 11 特征 (全量统计，含 votes_log) 或去掉 votes_log"""
        feats = {}
        feats["pub_avg_rating"] = subset["publisher"].map(pub_s["pub_avg"]).fillna(global_mean)
        feats["pub_book_count_log"] = np.log1p(subset["publisher"].map(pub_s["pub_cnt"]).fillna(1))
        feats["pub_std_rating"] = subset["publisher"].map(pub_s["pub_std"]).fillna(global_std)
        feats["author_avg_rating"] = subset["author"].map(auth_s["auth_avg"]).fillna(global_mean)
        feats["author_book_count_log"] = np.log1p(subset["author"].map(auth_s["auth_cnt"]).fillna(1))
        feats["binding_score"] = subset["binding"].map(lambda x: bind_s.get(x, global_mean))
        feats["pub_year"] = subset["pub_year_num"].clip(1900, 2030)
        feats["pages_log"] = np.log1p(subset["pages_num"].clip(10, 5000))
        feats["is_translation"] = subset["is_translation"]
        feats["is_series"] = subset["is_series"]
        if include_votes:
            feats["votes_log"] = np.log1p(subset["votes"])
        feature_names = list(feats.keys())
        X = np.column_stack([feats[n] for n in feature_names])
        return X, feature_names

    def featurize_loo(subset, train_full, pub_s_train, auth_s_train, bind_s_train):
        """Train-only统计 + leave-one-out（排除自身）"""
        # 对训练集：构建排除自身的统计量
        # 对每本书，临时从训练统计中减去自身
        feats = {}

        # 出版社统计 - LOO
        pub_avg_map = {}
        pub_cnt_map = {}
        pub_std_map = {}
        for pub_name in subset["publisher"].unique():
            if pub_name not in pub_s_train.index:
                pub_avg_map[pub_name] = global_mean
                pub_cnt_map[pub_name] = 1.0
                pub_std_map[pub_name] = global_std
                continue
            n = pub_s_train.loc[pub_name, "pub_cnt"]
            mean_all = pub_s_train.loc[pub_name, "pub_avg"]
            subset_pub = subset[subset["publisher"] == pub_name]
            for idx, row in subset_pub.iterrows():
                if n > 1:
                    loo_mean = (mean_all * n - row["rating"]) / (n - 1)
                    pub_avg_map[idx] = loo_mean
                    pub_cnt_map[idx] = n - 1
                else:
                    pub_avg_map[idx] = global_mean
                    pub_cnt_map[idx] = 1.0
                    pub_std_map[idx] = global_std

        # Author stats - LOO
        auth_avg_map = {}
        auth_cnt_map = {}
        for auth_name in subset["author"].unique():
            if auth_name not in auth_s_train.index:
                auth_avg_map[auth_name] = global_mean
                auth_cnt_map[auth_name] = 1.0
                continue
            n = auth_s_train.loc[auth_name, "auth_cnt"]
            mean_all = auth_s_train.loc[auth_name, "auth_avg"]
            subset_auth = subset[subset["author"] == auth_name]
            for idx, row in subset_auth.iterrows():
                if n > 1:
                    loo_mean = (mean_all * n - row["rating"]) / (n - 1)
                    auth_avg_map[idx] = loo_mean
                    auth_cnt_map[idx] = n - 1
                else:
                    auth_avg_map[idx] = global_mean
                    auth_cnt_map[idx] = 1.0

        # Build features
        X_rows = []
        for _, row in subset.iterrows():
            idx_val = row.name
            row_feats = [
                pub_avg_map.get(idx_val, global_mean),
                np.log1p(max(1, pub_cnt_map.get(idx_val, 1))),
                pub_std_map.get(idx_val, global_std),
                auth_avg_map.get(idx_val, global_mean),
                np.log1p(max(1, auth_cnt_map.get(idx_val, 1))),
                bind_s_train.get(row["binding"], global_mean),
                np.clip(row["pub_year_num"], 1900, 2030),
                np.log1p(max(10, min(row["pages_num"], 5000))),
                int(row["is_translation"]),
                int(row["is_series"]),
            ]
            X_rows.append(row_feats)

        feature_names = [
            "pub_avg_rating", "pub_book_count_log", "pub_std_rating",
            "author_avg_rating", "author_book_count_log",
            "binding_score", "pub_year", "pages_log",
            "is_translation", "is_series",
        ]
        return np.array(X_rows), feature_names

    # ===== 版本1: 原始11特征（全量统计 + votes_log）——注意：这里stats也是train-only算的，但映射用全量=有泄露 =====
    pub_stats_full, auth_stats_full, bind_stats_full = build_train_stats(df)  # 全量！
    X_train_v1, feat_v1 = featurize_v1(train_df, pub_stats_full, auth_stats_full, bind_stats_full, True, True)
    X_test_v1, _ = featurize_v1(test_df, pub_stats_full, auth_stats_full, bind_stats_full, False, True)

    # ===== 版本2: 10特征（去掉 votes_log） =====
    X_train_v2, feat_v2 = featurize_v1(train_df, pub_stats_full, auth_stats_full, bind_stats_full, True, False)
    X_test_v2, _ = featurize_v1(test_df, pub_stats_full, auth_stats_full, bind_stats_full, False, False)

    # ===== 版本3: 10特征 + train-only + LOO 统计 =====
    X_train_v3, feat_v3 = featurize_loo(train_df, train_df, pub_stats, auth_stats, bind_stats)
    X_test_v3, _ = featurize_loo(test_df, train_df, pub_stats, auth_stats, bind_stats)

    # ===== Baseline: 作者均值 =====
    auth_means_test = test_df["author"].map(auth_stats["auth_avg"]).fillna(global_mean).values
    rmse_auth_base = np.sqrt(mean_squared_error(y_test, auth_means_test))
    r2_auth_base = r2_score(y_test, auth_means_test)
    print(f"\n  Baseline (作者均值)    : RMSE={rmse_auth_base:.4f}, R2={r2_auth_base:.4f}")

    # ===== 训练三个版本 =====
    results_c = {}

    for version_name, X_tr, X_te in [
        ("v1_11feat_leaked", X_train_v1, X_test_v1),
        ("v2_10feat_no_votes", X_train_v2, X_test_v2),
        ("v3_10feat_trainonly", X_train_v3, X_test_v3),
    ]:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results_c[version_name] = (rmse, r2)
        print(f"  {version_name:<25s}: RMSE={rmse:.4f}, R2={r2:.4f}")

    lines = [
        "## 实验 C: 冷启动模型泄露检查",
        "",
        "### 现状分析",
        "",
        "`coldstart_predictor.py` 的 `build_stats()` 在**全量 `self.df`** 上计算 stats，",
        "然后 `build_features()` 逐行映射。这意味着 `pub_avg_rating` 和 `author_avg_rating`",
        "都**包含目标书本身的评分** → 标签泄露。",
        "",
        "`votes_log` 特征对真实新书不可用（votes≈0），属于特征泄露。",
        "",
        "### 对比实验",
        "",
        f"- 训练集: {len(train_df):,} 条, 测试集: {len(test_df):,} 条",
        "- 模型: RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)",
        "",
        "| 版本 | RMSE | R² |",
        "|------|------|----|",
        f"| Baseline (作者均值) | {rmse_auth_base:.4f} | {r2_auth_base:.4f} |",
        f"| v1: 11特征(含泄露) | {results_c['v1_11feat_leaked'][0]:.4f} | {results_c['v1_11feat_leaked'][1]:.4f} |",
        f"| v2: 10特征(去votes_log) | {results_c['v2_10feat_no_votes'][0]:.4f} | {results_c['v2_10feat_no_votes'][1]:.4f} |",
        f"| v3: 10特征(去votes+LOO统计) | {results_c['v3_10feat_trainonly'][0]:.4f} | {results_c['v3_10feat_trainonly'][1]:.4f} |",
        "",
        "### 结论",
        "",
        f"- v3（严谨版）R²={results_c['v3_10feat_trainonly'][1]:.4f}，对比作者均值 baseline R²={r2_auth_base:.4f}",
        f"- 差距 = {results_c['v3_10feat_trainonly'][1] - r2_auth_base:+.4f}",
    ]

    if results_c['v3_10feat_trainonly'][1] <= r2_auth_base:
        lines.append("- ⚠ **模型并未显著优于直接查作者均分**，建议审视特征工程的有效性。")
    else:
        lines.append("- ✅ 模型优于作者均值 baseline。")

    lines.append("")
    return "\n".join(lines), results_c


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="豆瓣图书推荐系统 - 离线评估")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "rec", "rating", "coldstart"],
                        help="选择实验: all, rec, rating, coldstart")
    args = parser.parse_args()

    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_parts = [
        f"# 豆瓣图书推荐系统 — 离线评估报告",
        f"",
        f"**生成时间**: {now}",
        f"**数据规模**: books_for_rec.csv ≈ 164K 行, Books_detail.csv = 6,584 行",
        f"",
        "---",
        "",
    ]

    if args.experiment in ("all", "rec"):
        part_a, _ = run_experiment_a()
        report_parts.append(part_a)

    if args.experiment in ("all", "rating"):
        part_b, _ = run_experiment_b()
        report_parts.append(part_b)

    if args.experiment in ("all", "coldstart"):
        part_c, _ = run_experiment_c()
        report_parts.append(part_c)

    elapsed = time.time() - start_time
    report_parts.append(f"---")
    report_parts.append(f"")
    report_parts.append(f"_总耗时: {elapsed:.1f}s_")
    report_parts.append(f"_random_state=42 用于所有随机过程_")

    report_text = "\n".join(report_parts)

    # 输出到终端（容错 GBK 编码）
    print("\n" + "=" * 70)
    print("  评估报告")
    print("=" * 70)
    # 安全输出：替换无法编码的字符
    for line in report_text.split("\n"):
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"))

    # 写入文件
    report_path = REPORT_DIR / "evaluation_results.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[报告保存] {report_path}")


if __name__ == "__main__":
    main()
