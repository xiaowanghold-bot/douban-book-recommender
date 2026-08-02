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

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from .coldstart_training import evaluate_coldstart_model, prepare_coldstart_dataframe
    from .rating_training import fit_target_encoders, prepare_rating_dataframe, train_rating_model
except ImportError:  # Support direct script execution.
    from coldstart_training import evaluate_coldstart_model, prepare_coldstart_dataframe
    from rating_training import fit_target_encoders, prepare_rating_dataframe, train_rating_model

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
    """Load BookRecommender with pre-built NN index"""
    sys.path.insert(0, str(ROOT))
    from src.recommendation import BookRecommender
    npz_path = ROOT / "data" / "models" / "nn_neighbors.npz"
    if not npz_path.exists():
        print("[提示] nn_neighbors.npz 不存在，将在内存中重新计算。可运行 python -m src.recommendation 预生成。")
    rec = BookRecommender()
    rec._load_artifacts()
    if not hasattr(rec, 'nn_indices') or rec.nn_indices is None:
        rec.build_nn_index(n_neighbors=30)
        rec.nn_distances, rec.nn_indices = rec.nn_model.kneighbors(rec.tfidf_matrix)
    print(f"[OK] NN ready ({rec.nn_indices.shape[1]} neighbors)")
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
        print("  [WARN] 样本量 < 100，结果仅供参考")

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

    # book_id -> matrix_idx 映射
    id_to_idx = rec.id_to_idx
    idx_to_id = rec.idx_to_id

    # 预取标题数组加速
    titles_arr = rec.df["title"].values

    # 按 bayesian_score 排序（用于 Popular baseline）
    popular_ids = rec_df.nlargest(len(rec_df), "bayesian_score")["id"].tolist()
    all_ids_set = set(int(x) for x in rec_df["id"])

    K_values = [10, 20, 50]
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
        "> ⚠ **方法局限性说明**: 当前推荐引擎基于 jieba 语义 TF-IDF + 余弦相似度，",
        "> 同系列图书通常共享作者、标签和书名词元，因此同系列 Recall 会高估真实个性化推荐能力。",
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
    """评分预测（RandomForest + OOF 目标编码）正规评估"""
    print("\n" + "=" * 70)
    print("  实验 B: 评分预测正规评估（OOF 目标编码）")
    print("=" * 70)

    raw_detail = pd.read_csv(RAW_DIR / "Books_detail.csv", encoding="utf-8-sig")
    df = prepare_rating_dataframe(raw_detail)
    print(f"  有效数据: {len(df)} 条")

    result = train_rating_model(df)
    train_df = result["train_df"]
    test_df = result["test_df"]
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}")

    y_train = train_df["Rating"].values
    y_test = test_df["Rating"].values

    # Baseline 1: 全局均值
    global_mean = y_train.mean()
    pred_global = np.full_like(y_test, global_mean)
    rmse_global = np.sqrt(mean_squared_error(y_test, pred_global))
    mae_global = mean_absolute_error(y_test, pred_global)
    print(f"  全局均值 baseline : RMSE={rmse_global:.4f}, MAE={mae_global:.4f}")

    # Baseline 2: 出版社均值（未见→全局均值）
    train_encoders = fit_target_encoders(train_df)
    pub_means = train_encoders["publisher_means"]
    test_pub_preds = (
        test_df["publisher_clean"].map(pub_means).fillna(global_mean).values
    )
    rmse_pub = np.sqrt(mean_squared_error(y_test, test_pub_preds))
    mae_pub = mean_absolute_error(y_test, test_pub_preds)
    print(f"  出版社均值 baseline: RMSE={rmse_pub:.4f}, MAE={mae_pub:.4f}")

    # Baseline 3: 作者均值（未见→全局均值）
    auth_means = train_encoders["author_means"]
    test_auth_preds = (
        test_df["author_clean"].map(auth_means).fillna(global_mean).values
    )
    rmse_auth = np.sqrt(mean_squared_error(y_test, test_auth_preds))
    mae_auth = mean_absolute_error(y_test, test_auth_preds)
    print(f"  作者均值 baseline  : RMSE={rmse_auth:.4f}, MAE={mae_auth:.4f}")

    metrics = result["metrics"]
    rmse_rf = metrics["RMSE"]
    mae_rf = metrics["MAE"]
    print(f"  RandomForest v3 OOF: RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}")
    print(
        f"  Nested CV R2       : {metrics['CV_R2']:.4f} "
        f"± {metrics['CV_R2_std']:.4f}"
    )

    lines = [
        "## 实验 B: 评分预测正规评估（OOF 目标编码）",
        "",
        f"- 训练集: {len(train_df):,} 条, 测试集: {len(test_df):,} 条",
        "- 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)",
        "- 训练类别统计: 5折 OOF 目标均值，验证/测试类别统计仅来自对应训练集",
        f"- 嵌套5折 CV R²: {metrics['CV_R2']:.4f} ± {metrics['CV_R2_std']:.4f}",
        "- 特征: price, year, pages, votes_log, author_mean, publisher_mean, binding_mean",
        "",
        "| 方法 | RMSE | MAE |",
        "|------|------|-----|",
        f"| 全局均值 | {rmse_global:.4f} | {mae_global:.4f} |",
        f"| 出版社均值 | {rmse_pub:.4f} | {mae_pub:.4f} |",
        f"| 作者均值 | {rmse_auth:.4f} | {mae_auth:.4f} |",
        f"| RandomForest v3 (OOF target means) | {rmse_rf:.4f} | {mae_rf:.4f} |",
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
    """冷启动模型独立测试与嵌套交叉验证。"""
    print("\n" + "=" * 70)
    print("  实验 C: 冷启动模型 v4 严格评估")
    print("=" * 70)

    raw_detail = pd.read_csv(RAW_DIR / "Books_detail.csv", encoding="utf-8-sig")
    df = prepare_coldstart_dataframe(raw_detail)
    print(f"  有效数据: {len(df)} 条")
    result = evaluate_coldstart_model(df)
    metrics = result["metrics"]
    print(f"  训练集: {metrics['n_train']}, 测试集: {metrics['n_test']}")
    print(
        f"  作者均值 baseline: RMSE={metrics['author_baseline_RMSE']:.4f}, "
        f"R2={metrics['author_baseline_R2']:.4f}"
    )
    print(
        f"  GradientBoosting v4: RMSE={metrics['RMSE']:.4f}, "
        f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}"
    )

    lines = [
        "## 实验 C: 冷启动模型 v4 严格评估",
        "",
        "- 训练目标统计: 5 折 OOF；每行特征不包含自身评分",
        "- 独立测试特征: 只使用训练集作者、出版社和装帧统计",
        "- 最终生产模型: 全量 5 折 OOF 特征训练，不使用 votes_log",
        f"- 训练集: {metrics['n_train']:,} 条, 测试集: {metrics['n_test']:,} 条",
        f"- 嵌套 5 折 CV R²: {metrics['CV_R2']:.4f} ± {metrics['CV_R2_std']:.4f}",
        "",
        "| 方法 | RMSE | MAE | R² |",
        "|------|------|-----|----|",
        f"| 作者均值 baseline | {metrics['author_baseline_RMSE']:.4f} | — | {metrics['author_baseline_R2']:.4f} |",
        f"| GradientBoosting v4 | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | {metrics['R2']:.4f} |",
        "",
        "### 结论",
        "",
        f"- v4 独立测试 R²={metrics['R2']:.4f}，作者均值 baseline R²={metrics['author_baseline_R2']:.4f}",
        f"- R² 提升 = {metrics['R2'] - metrics['author_baseline_R2']:+.4f}",
        "- 旧 v3 评估会从训练统计中错误减去测试行评分，已废弃。",
    ]
    return "\n".join(lines), metrics


# ============================================================================
#  主入口
# ============================================================================

def run_experiment_e():
    """Experiment E: Real user Leave-One-Out evaluation (IJCAI dataset)"""
    print("\n" + "=" * 70)
    print("  Experiment E: Real User Leave-One-Out (IJCAI)")
    print("=" * 70)

    user_ratings_path = DATA_DIR / "processed" / "user_ratings.csv"
    if not user_ratings_path.exists():
        msg = "[SKIP] user_ratings.csv not found. Run src/integrate_ijcai.py first."
        print(msg)
        return "## Experiment E: SKIPPED\n\n" + msg + "\n", {}

    ur = pd.read_csv(user_ratings_path, encoding="utf-8-sig")
    rec_df = pd.read_csv(MODEL_DIR / "books_for_rec.csv", encoding="utf-8-sig")
    rec_ids = set(str(int(i)) for i in rec_df["id"])

    ur["in_rec"] = ur["douban_book_id"].astype(str).isin(rec_ids)

    user_stats = ur.groupby("user_id").agg(
        n_total=("rating", "count"),
        n_high=("rating", lambda x: (x >= 4).sum()),
    )
    # Count high-rated books in rec index per user
    user_high_rec = ur[(ur["rating"] >= 4) & ur["in_rec"]].groupby("user_id").size()
    user_stats["n_high_rec"] = user_high_rec.reindex(user_stats.index).fillna(0)

    valid_users = user_stats[(user_stats["n_total"] >= 10) & (user_stats["n_high_rec"] >= 5)]
    valid_user_ids = set(valid_users.index)

    avg_total = valid_users["n_total"].mean()
    avg_high = valid_users["n_high_rec"].mean()
    print(f"  Valid users: {len(valid_user_ids)} | avg ratings: {avg_total:.0f} | avg high-in-rec: {avg_high:.0f}")
    if len(valid_user_ids) < 50:
        print("  [WARN] Too few users (< 50)")

    rec = load_recommender()
    nn_indices = rec.nn_indices
    id_to_idx = rec.id_to_idx
    idx_to_id = rec.idx_to_id

    # Popular baseline: sort by votes
    popular_ids = rec_df.nlargest(len(rec_df), "votes")["id"].tolist()
    all_ids_arr = np.array(sorted(set(int(i) for i in rec_df["id"])), dtype=np.int64)

    K_values = [10, 20, 50]
    results = {}

    for K in K_values:
        print(f"\n  --- K = {K} ---")
        recall_rec = []
        recall_random = []
        recall_pop = []
        n_valid = 0

        for uid in sorted(valid_user_ids):
            user_data = ur[ur["user_id"] == uid]
            high_rated = user_data[(user_data["rating"] >= 4) & user_data["in_rec"]]
            if len(high_rated) < 2:
                continue
            high_rated = high_rated.sort_values("time")
            target = high_rated.iloc[-1]
            history = high_rated.iloc[:-1]
            target_id = int(float(target["douban_book_id"]))

            # All books rated by this user (to exclude from baselines)
            all_rated = set(int(float(x)) for x in user_data[user_data["in_rec"]]["douban_book_id"])
            n_valid += 1

            # ----- Recommend (max-pooled rank-weighted) -----
            candidate_scores = {}
            for _, hb in history.iterrows():
                hid = int(float(hb["douban_book_id"]))
                midx = id_to_idx.get(hid)
                if midx is None:
                    continue
                neighbors = nn_indices[midx]
                seen_titles = set()
                cnt = 0
                for nidx in neighbors:
                    nid = idx_to_id.get(nidx)
                    if nid is None or nid == hid:
                        continue
                    title = rec.df.iloc[nidx]["title"]
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    cnt += 1
                    score = 1.0 / cnt  # rank-weighted
                    if nid not in candidate_scores or score > candidate_scores[nid]:
                        candidate_scores[nid] = score
                    if cnt >= K * 3:
                        break
            # Sort by score, take top-K
            top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:K]
            rec_top_set = set(cid for cid, _ in top_candidates)
            recall_rec.append(1.0 if target_id in rec_top_set else 0.0)

            # ----- Random baseline -----
            # Random baseline: 5 trials average
            eligible = [i for i in all_ids_arr if i not in all_rated]
            rand_hits = 0
            if len(eligible) >= K:
                for _ in range(5):
                    sampled = np.random.choice(eligible, size=K, replace=False)
                    if target_id in sampled:
                        rand_hits += 1
                recall_random.append(rand_hits / 5.0)
            else:
                recall_random.append(0.0)

            # ----- Popular baseline -----
            pop_candidates = [pid for pid in popular_ids if pid not in all_rated][:K]
            recall_pop.append(1.0 if target_id in pop_candidates else 0.0)

        avg_rec = np.mean(recall_rec) if recall_rec else 0
        avg_rand = np.mean(recall_random) if recall_random else 0
        avg_pop = np.mean(recall_pop) if recall_pop else 0
        results[K] = (avg_rec, avg_rand, avg_pop)
        print(f"  recommend_by_id : Recall@{K} = {avg_rec:.4f}  (n={n_valid})")
        print(f"  Random          : Recall@{K} = {avg_rand:.4f}")
        print(f"  Popular         : Recall@{K} = {avg_pop:.4f}")

    K10 = results.get(10, (0, 0, 0))
    K20 = results.get(20, (0, 0, 0))

    lines = [
        "## 实验 E: 真实用户 Leave-One-Out 评估 (IJCAI 数据集)",
        "",
        "> **数据来源**: DTCDR (CIKM 2019) / GA-DTCDR (IJCAI 2020) 跨域推荐公开数据集",
        "> 引用: Zhu et al., CIKM 2019; Zhu et al., IJCAI 2020",
        "",
        "### 方法学",
        "",
        f"- **用户筛选**: 总评分 >=10 条且 rating>=4 的高分书中至少 5 本在推荐索引内 (最终: {len(valid_user_ids)} 人)",
        "- **目标书选取**: 取该用户 rating>=4 的书, 按时间排序, 留出最后一本作为 ground-truth",
        "- **候选集构造**: 对每本历史高分种子书各取 top-(K*3) NN 邻居, max-pooled rank-weighted 分数聚合后取全局 top-K",
        "- **Random 基线**: 从全库随机抽 K 本 (排除该用户所有已评分书), 5 次平均",
        "- **Popular 基线**: 按 votes 降序取前 K 本 (排除该用户所有已评分书)",
        "  (注: Popular 基线为 0 是正常的——Top20 永远是《活着》《红楼梦》等国民级畅销书, 个性化目标几乎不可能命中)",
        "",
        "| 方法 | Recall@10 | Recall@20 | 用户数 |",
        "|------|-----------|-----------|--------|",
        f"| recommend_by_id | {K10[0]:.4f} | {K20[0]:.4f} | {n_valid} |",
        f"| Random | {K10[1]:.4f} | {K20[1]:.4f} | {n_valid} |",
        f"| Popular | {K10[2]:.4f} | {K20[2]:.4f} | {n_valid} |",
        "",
    ]
    return "\n".join(lines), {"rec10": K10[0], "rec20": K20[0], "rand10": K10[1], "pop20": K20[2]}

def main():
    parser = argparse.ArgumentParser(description="豆瓣图书推荐系统 - 离线评估")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "rec", "rating", "coldstart", "user"],
                        help="Select experiment: all, rec, rating, coldstart, user")
    args = parser.parse_args()

    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_parts = [
        "# 豆瓣图书推荐系统 — 离线评估报告",
        "",
        f"**生成时间**: {now}",
        "**数据规模**: books_for_rec.csv ≈ 164K 行, Books_detail.csv = 6,584 行",
        "",
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

    if args.experiment in ("all", "user", "e"):
        part_e, _ = run_experiment_e()
        report_parts.append(part_e)

    elapsed = time.time() - start_time
    report_parts.append("---")
    report_parts.append("")
    report_parts.append(f"_总耗时: {elapsed:.1f}s_")
    report_parts.append("_random_state=42 用于所有随机过程_")

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

    # 单实验运行时备份原文件（防止覆盖手动维护的章节如语义化对比）
    if args.experiment != "all" and report_path.exists():
        import shutil
        bak_path = report_path.with_suffix(".md.bak")
        shutil.copy2(report_path, bak_path)
        print(f"\n[备份] 单实验模式: 原报告已备份至 {bak_path}")
        print("[警告] 仅生成了所选实验章节，报告其余部分（如语义化对比）需手动恢复。")
        print("[建议] 运行 python -m src.evaluate --experiment all 生成完整报告。")

    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[报告保存] {report_path}")


if __name__ == "__main__":
    main()
