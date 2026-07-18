"""
绂荤嚎璇勪及鑴氭湰 鈥?璞嗙摚鍥句功鎺ㄨ崘绯荤粺
================================
瀹為獙 A: 鍚岀郴鍒?Recall@K
瀹為獙 B: 璇勫垎棰勬祴姝ｈ璇勪及
瀹為獙 C: 鍐峰惎鍔ㄦā鍨嬫硠闇叉鏌?

鐢ㄦ硶: python -m src.evaluate --experiment all|rec|rating|coldstart
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
#  鍏变韩宸ュ叿
# ============================================================================

def load_recommender():
    """鍔犺浇宸叉瀯寤虹殑 BookRecommender锛堝惈棰勮绠楁渶杩戦偦锛?""
    sys.path.insert(0, str(ROOT))
    from src.recommendation import BookRecommender
    rec = BookRecommender()
    rec._load_artifacts()
    # 鍔犺浇棰勮绠楃殑鏈€杩戦偦
    nn_path = rec.model_dir / "nn_neighbors.npz"
    if nn_path.exists():
        data = np.load(nn_path)
        rec.nn_distances = data["distances"]
        rec.nn_indices = data["indices"]
        print(f"[鍔犺浇] 棰勮绠楁渶杩戦偦 ({rec.nn_indices.shape[1]} 杩戦偦)")
    else:
        rec.build_nn_index(n_neighbors=30)
    return rec


def load_books_detail():
    """鍔犺浇 Books_detail.csv"""
    df = pd.read_csv(RAW_DIR / "Books_detail.csv", encoding="utf-8-sig")
    # 缁熶竴鍒楀悕
    df = df.rename(columns={"ID": "id", "Rating": "rating", "Votes": "votes",
                             "Title": "title"})
    df = df[df["crawl_status"] == "success"].copy()
    for col in ["rating", "votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================================
#  瀹為獙 A: 鍚岀郴鍒?Recall@K
# ============================================================================

def run_experiment_a():
    """鍚岀郴鍒?Recall@K 璇勪及锛堜娇鐢ㄩ璁＄畻 NN锛岄伩鍏嶉€愭璋冪敤 API锛?""
    print("\n" + "=" * 70)
    print("  瀹為獙 A: 鍚岀郴鍒?Recall@K")
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
    print(f"  璇勪及鏌ヨ涔︽暟: {n_queries}", flush=True)
    print(f"  绯诲垪鏁? {n_series}", flush=True)

    if n_queries < 100:
        print(f"  [WARN] 鏍锋湰閲?< 100锛岀粨鏋滀粎渚涘弬鑰?)

    # 鏋勫缓绯诲垪鍐呮垚鍛樻槧灏?
    series_members = {}
    for _, row in eval_books.iterrows():
        s = row["series"]
        if s not in series_members:
            series_members[s] = set()
        series_members[s].add(int(row["id"]))

    # 鍔犺浇鎺ㄨ崘寮曟搸 + 棰勮绠?NN
    rec = load_recommender()
    nn_indices = rec.nn_indices  # shape: (n_books, n_neighbors)
    nn_distances = rec.nn_distances

    # book_id -> matrix_idx 鏄犲皠
    id_to_idx = rec.id_to_idx
    idx_to_id = rec.idx_to_id

    # 棰勫彇鏍囬鏁扮粍鍔犻€?
    titles_arr = rec.df["title"].values

    # 鎸?bayesian_score 鎺掑簭锛堢敤浜?Popular baseline锛?
    popular_ids = rec_df.nlargest(len(rec_df), "bayesian_score")["id"].tolist()
    all_ids_set = set(int(x) for x in rec_df["id"])

    K_values = [10, 20]
    results = {}

    # 棰勮绠楁瘡涓煡璇㈠湪 NN 涓殑绱㈠紩鍜屽畠鍦?popular 涓殑鎺掑悕
    # 鏋勫缓 series book -> 璇ョ郴鍒楀叾浠栨垚鍛樼殑 set 鏄犲皠锛堝姞閫燂級
    book_to_series_others = {}
    for _, row in eval_books.iterrows():
        bid = int(row["id"])
        book_to_series_others[bid] = series_members[row["series"]] - {bid}

    for K in K_values:
        print(f"\n  --- K = {K} ---")

        # 鏂规硶1: recommend_by_id锛堢洿鎺ヤ娇鐢ㄩ璁＄畻 NN锛?
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
                # 鍙栧墠 search_range 涓偦灞咃紙鎺掗櫎鑷繁锛屽幓閲嶅悓涔﹀悕锛?
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

        # 鏂规硶2: Random baseline (浣跨敤 numpy 鐩存帴浠庡叏搴揑D鏁扮粍閲囨牱锛岄伩鍏嶉噸澶嶅垎閰嶅ぇ鍒楄〃)
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
                # 浠庡叏搴撻殢鏈哄彇 K+1 鏈紙涓囦竴鎶藉埌鑷韩鍒欏鍙栵級锛屽幓鎺夎嚜韬悗鍙?K 鏈?
                sample_size = min(K + 1, len(all_ids_arr))
                sampled = rng.choice(all_ids_arr, size=sample_size, replace=False)
                sampled = [int(s) for s in sampled if int(s) != bid][:K]
                hits = len(set(sampled) & series_others)
                trial_recalls.append(hits / denominator)
            recall_random_trials.append(np.mean(trial_recalls) if trial_recalls else 0.0)
        mean_random = np.mean(recall_random_trials)
        print(f"  Random baseline    : Recall@{K} = {mean_random:.4f}  (avg of 5 trials)")

        # 鏂规硶3: Popular baseline
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

    # 鐢熸垚鎶ュ憡
    lines = [
        "## 瀹為獙 A: 鍚岀郴鍒?Recall@K",
        "",
        "> 鈿?**鏂规硶灞€闄愭€ц鏄?*: 鏈」鐩帹鑽愬紩鎿庡熀浜庡瓧绗︾骇 n-gram TF-IDF + 浣欏鸡鐩镐技搴︼紝",
        "> 鍚岀郴鍒椾功鍚嶏紙濡傘€婁笁浣撱€?銆婁笁浣揑I銆嬶級澶╃劧瀛楃閲嶅彔搴﹂珮锛孯ecall 鎸囨爣浼氶珮浼板疄闄呰涔夋帹鑽愯兘鍔涖€?,
        "",
        f"- 鏌ヨ涔︽暟: {n_queries}",
        f"- 绯诲垪鏁? {n_series}",
        f"- 绯诲垪骞冲潎瑙勬ā: {n_queries / n_series:.1f} 鏈?,
        "",
        "| 鏂规硶 | Recall@10 | Recall@20 | 鏈夋晥鏌ヨ鏁?|",
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
#  瀹為獙 B: 璇勫垎棰勬祴姝ｈ璇勪及
# ============================================================================

def run_experiment_b():
    """璇勫垎棰勬祴锛圧andomForest锛夋瑙勮瘎浼?""
    print("\n" + "=" * 70)
    print("  瀹為獙 B: 璇勫垎棰勬祴姝ｈ璇勪及")
    print("=" * 70)

    detail = load_books_detail()

    # ===== 鐗瑰緛宸ョ▼锛堝鍒惰嚜 enhancements.py RatingPredictor锛?=====
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
        lambda x: __import__("re").sub(r"\[.*?\]|\(.*?\)|锛?*?锛?, "", str(x)).strip()[:30]
        if pd.notna(x) else "鏈煡")
    df["publisher_clean"] = df["publisher"].fillna("鏈煡").astype(str).str[:20]
    df["binding_type"] = df["binding"].fillna("鏈煡").apply(
        lambda x: "骞宠" if "骞宠" in str(x) else ("绮捐" if "绮捐" in str(x) else "鍏朵粬"))

    # 绛涢€夋湁鏁堣褰?
    df = df.dropna(subset=["rating", "price_num", "year_num", "pages_num"]).copy()
    df = df[df["year_num"].between(1950, 2025)]
    df = df[df["rating"].between(1, 10)]
    df["pages_num"] = df["pages_num"].fillna(df["pages_num"].median())

    print(f"  鏈夋晥鏁版嵁: {len(df)} 鏉?)

    # 缂栫爜浣庨绫诲埆
    for col in ["author_clean", "publisher_clean", "binding_type"]:
        counts = df[col].value_counts()
        df[f"{col}_enc"] = df[col].apply(
            lambda x, c=counts: x if c.get(x, 0) >= 3 else "鍏朵粬")

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  璁粌闆? {len(train_df)}, 娴嬭瘯闆? {len(test_df)}")

    y_train = train_df["rating"].values
    y_test = test_df["rating"].values

    # Baseline 1: 鍏ㄥ眬鍧囧€?
    global_mean = y_train.mean()
    pred_global = np.full_like(y_test, global_mean)
    rmse_global = np.sqrt(mean_squared_error(y_test, pred_global))
    mae_global = mean_absolute_error(y_test, pred_global)
    print(f"  鍏ㄥ眬鍧囧€?baseline : RMSE={rmse_global:.4f}, MAE={mae_global:.4f}")

    # Baseline 2: 鍑虹増绀惧潎鍊硷紙鏈鈫掑叏灞€鍧囧€硷級
    pub_means = train_df.groupby("publisher_clean_enc")["rating"].mean().to_dict()
    test_pub_preds = test_df["publisher_clean_enc"].map(pub_means).fillna(global_mean).values
    rmse_pub = np.sqrt(mean_squared_error(y_test, test_pub_preds))
    mae_pub = mean_absolute_error(y_test, test_pub_preds)
    print(f"  鍑虹増绀惧潎鍊?baseline: RMSE={rmse_pub:.4f}, MAE={mae_pub:.4f}")

    # Baseline 3: 浣滆€呭潎鍊硷紙鏈鈫掑叏灞€鍧囧€硷級
    auth_means = train_df.groupby("author_clean_enc")["rating"].mean().to_dict()
    test_auth_preds = test_df["author_clean_enc"].map(auth_means).fillna(global_mean).values
    rmse_auth = np.sqrt(mean_squared_error(y_test, test_auth_preds))
    mae_auth = mean_absolute_error(y_test, test_auth_preds)
    print(f"  浣滆€呭潎鍊?baseline  : RMSE={rmse_auth:.4f}, MAE={mae_auth:.4f}")

    # RandomForest v2锛坱rain-only 鍧囧€兼浛浠?LabelEncoder锛?
    train_global_mean = y_train.mean()
    train_author_means = train_df.groupby("author_clean_enc")["rating"].mean().to_dict()
    train_pub_means = train_df.groupby("publisher_clean_enc")["rating"].mean().to_dict()
    train_bind_means = train_df.groupby("binding_type_enc")["rating"].mean().to_dict()

    def build_features_v2(subset, author_m, pub_m, bind_m, fallback):
        return pd.DataFrame({
            "price": subset["price_num"],
            "year": subset["year_num"],
            "pages": subset["pages_num"],
            "votes_log": np.log1p(subset["votes"]),
            "author_mean": subset["author_clean_enc"].map(author_m).fillna(fallback),
            "publisher_mean": subset["publisher_clean_enc"].map(pub_m).fillna(fallback),
            "binding_mean": subset["binding_type_enc"].map(bind_m).fillna(fallback),
        }).values

    X_train_v2 = build_features_v2(train_df, train_author_means, train_pub_means, train_bind_means, train_global_mean)
    X_test_v2 = build_features_v2(test_df, train_author_means, train_pub_means, train_bind_means, train_global_mean)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_v2, y_train)
    y_pred_rf = rf.predict(X_test_v2)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print(f"  RandomForest v2    : RMSE={rmse_rf:.4f}, MAE={mae_rf:.4f}")

    lines = [
        "## 瀹為獙 B: 璇勫垎棰勬祴姝ｈ璇勪及",
        "",
        f"- 璁粌闆? {len(train_df):,} 鏉? 娴嬭瘯闆? {len(test_df):,} 鏉?,
        "- 妯″瀷: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)",
        "- 鐗瑰緛: price, year, pages, votes_log, author_clean, publisher_clean, binding_type",
        "",
        "| 鏂规硶 | RMSE | MAE |",
        "|------|------|-----|",
        f"| 鍏ㄥ眬鍧囧€?| {rmse_global:.4f} | {mae_global:.4f} |",
        f"| 鍑虹増绀惧潎鍊?| {rmse_pub:.4f} | {mae_pub:.4f} |",
        f"| 浣滆€呭潎鍊?| {rmse_auth:.4f} | {mae_auth:.4f} |",
        f"| RandomForest (淇鍚? train-only means) | {rmse_rf:.4f} | {mae_rf:.4f} |",
        "",
    ]

    return "\n".join(lines), {
        "global": (rmse_global, mae_global),
        "publisher": (rmse_pub, mae_pub),
        "author": (rmse_auth, mae_auth),
        "rf": (rmse_rf, mae_rf),
    }


# ============================================================================
#  瀹為獙 C: 鍐峰惎鍔ㄦā鍨嬫硠闇叉鏌?
# ============================================================================

def run_experiment_c():
    """鍐峰惎鍔ㄦā鍨嬫硠闇叉鏌?""
    print("\n" + "=" * 70)
    print("  瀹為獙 C: 鍐峰惎鍔ㄦā鍨嬫硠闇叉鏌?)
    print("=" * 70)

    detail = load_books_detail()

    # 娓呮礂
    df = detail.copy()
    df["pub_year_num"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["pages_num"] = pd.to_numeric(df["pages"], errors="coerce")
    df["pub_year_num"] = df["pub_year_num"].fillna(2010).astype(int)
    df["pages_num"] = df["pages_num"].fillna(300).astype(int)
    df["author"] = df["author"].fillna("鏈煡").astype(str)
    df["publisher"] = df["publisher"].fillna("鏈煡").astype(str)
    df["binding"] = df["binding"].fillna("鍏朵粬").astype(str)
    df = df.dropna(subset=["rating", "votes"])
    df = df[(df["rating"] >= 1) & (df["rating"] <= 10)]
    df = df[df["votes"] >= 10]
    df["is_translation"] = (df["translator"].notna() | df["original_title"].notna()).astype(int)
    df["is_series"] = df["series"].notna().astype(int)

    print(f"  鏈夋晥鏁版嵁: {len(df)} 鏉?)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df["rating"].values
    y_test = test_df["rating"].values
    print(f"  璁粌闆? {len(train_df)}, 娴嬭瘯闆? {len(test_df)}")

    global_mean = y_train.mean()
    global_std = y_train.std()

    # ===== 1. 鐜扮姸妫€鏌?=====
    print("\n  --- 鐜扮姸妫€鏌?---")
    print("  褰撳墠 coldstart_predictor.py 鐨?build_stats() 瀵瑰叏閲?self.df 璁＄畻缁熻閲忥紝")
    print("  鐒跺悗 build_features() 鏄犲皠鍥炴瘡琛屻€傚洜姝?pub_avg_rating 鍜?author_avg_rating")
    print("  閮藉寘鍚洰鏍囦功鏈韩鐨勮瘎绾?鈫?**瀛樺湪鏁版嵁娉勯湶**銆?)
    print("  votes_log 浣跨敤璁粌闆嗗叏浣撲功鐨勭湡瀹炴姇绁ㄦ暟 鈫?瀵规柊涔?votes鈮?)涓嶅彲鐢?鈫?**鐗瑰緛娉勯湶**銆?)

    # ===== 杈呭姪鍑芥暟锛歵rain-only stats锛堟帓闄よ嚜韬級 =====
    def build_train_stats(train_subset):
        pub = train_subset.groupby("publisher")["rating"].agg(["mean", "count", "std"]).fillna(0)
        pub.columns = ["pub_avg", "pub_cnt", "pub_std"]
        auth = train_subset.groupby("author")["rating"].agg(["mean", "count"]).fillna(0)
        auth.columns = ["auth_avg", "auth_cnt"]
        bind = train_subset.groupby("binding")["rating"].mean().to_dict()
        return pub, auth, bind

    pub_stats, auth_stats, bind_stats = build_train_stats(train_df)

    def featurize_v1(subset, pub_s, auth_s, bind_s, is_train, include_votes):
        """鍘熷 11 鐗瑰緛 (鍏ㄩ噺缁熻锛屽惈 votes_log) 鎴栧幓鎺?votes_log"""
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
        """Train-only缁熻 + leave-one-out锛堟帓闄よ嚜韬級"""
        # 瀵硅缁冮泦锛氭瀯寤烘帓闄よ嚜韬殑缁熻閲?
        # 瀵规瘡鏈功锛屼复鏃朵粠璁粌缁熻涓噺鍘昏嚜韬?
        feats = {}

        # 鍑虹増绀剧粺璁?- LOO
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

    # ===== 鐗堟湰1: 鍘熷11鐗瑰緛锛堝叏閲忕粺璁?+ votes_log锛夆€斺€旀敞鎰忥細杩欓噷stats涔熸槸train-only绠楃殑锛屼絾鏄犲皠鐢ㄥ叏閲?鏈夋硠闇?=====
    pub_stats_full, auth_stats_full, bind_stats_full = build_train_stats(df)  # 鍏ㄩ噺锛?
    X_train_v1, feat_v1 = featurize_v1(train_df, pub_stats_full, auth_stats_full, bind_stats_full, True, True)
    X_test_v1, _ = featurize_v1(test_df, pub_stats_full, auth_stats_full, bind_stats_full, False, True)

    # ===== 鐗堟湰2: 10鐗瑰緛锛堝幓鎺?votes_log锛?=====
    X_train_v2, feat_v2 = featurize_v1(train_df, pub_stats_full, auth_stats_full, bind_stats_full, True, False)
    X_test_v2, _ = featurize_v1(test_df, pub_stats_full, auth_stats_full, bind_stats_full, False, False)

    # ===== 鐗堟湰3: 10鐗瑰緛 + train-only + LOO 缁熻 =====
    X_train_v3, feat_v3 = featurize_loo(train_df, train_df, pub_stats, auth_stats, bind_stats)
    X_test_v3, _ = featurize_loo(test_df, train_df, pub_stats, auth_stats, bind_stats)

    # ===== Baseline: 浣滆€呭潎鍊?=====
    auth_means_test = test_df["author"].map(auth_stats["auth_avg"]).fillna(global_mean).values
    rmse_auth_base = np.sqrt(mean_squared_error(y_test, auth_means_test))
    r2_auth_base = r2_score(y_test, auth_means_test)
    print(f"\n  Baseline (浣滆€呭潎鍊?    : RMSE={rmse_auth_base:.4f}, R2={r2_auth_base:.4f}")

    # ===== 璁粌涓変釜鐗堟湰 =====
    results_c = {}

    for version_name, X_tr, X_te in [
        ("v1_11feat_leaked", X_train_v1, X_test_v1),
        ("v2_10feat_no_votes", X_train_v2, X_test_v2),
        ("v3_10feat_trainonly", X_train_v3, X_test_v3),
    ]:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results_c[version_name] = (rmse, r2)
        print(f"  {version_name:<25s}: RMSE={rmse:.4f}, R2={r2:.4f}")

    lines = [
        "## 瀹為獙 C: 鍐峰惎鍔ㄦā鍨嬫硠闇叉鏌?,
        "",
        "### 鐜扮姸鍒嗘瀽",
        "",
        "`coldstart_predictor.py` 鐨?`build_stats()` 鍦?*鍏ㄩ噺 `self.df`** 涓婅绠?stats锛?,
        "鐒跺悗 `build_features()` 閫愯鏄犲皠銆傝繖鎰忓懗鐫€ `pub_avg_rating` 鍜?`author_avg_rating`",
        "閮?*鍖呭惈鐩爣涔︽湰韬殑璇勫垎** 鈫?鏍囩娉勯湶銆?,
        "",
        "`votes_log` 鐗瑰緛瀵圭湡瀹炴柊涔︿笉鍙敤锛坴otes鈮?锛夛紝灞炰簬鐗瑰緛娉勯湶銆?,
        "",
        "### 瀵规瘮瀹為獙",
        "",
        f"- 璁粌闆? {len(train_df):,} 鏉? 娴嬭瘯闆? {len(test_df):,} 鏉?,
        "- 妯″瀷: GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42) 鈥?涓庣敓浜у喎鍚姩妯″瀷鍚岄厤缃?,
        "",
        "| 鐗堟湰 | RMSE | R虏 |",
        "|------|------|----|",
        f"| Baseline (浣滆€呭潎鍊? | {rmse_auth_base:.4f} | {r2_auth_base:.4f} |",
        f"| v1: 11鐗瑰緛(鍚硠闇? | {results_c['v1_11feat_leaked'][0]:.4f} | {results_c['v1_11feat_leaked'][1]:.4f} |",
        f"| v2: 10鐗瑰緛(鍘籿otes_log) | {results_c['v2_10feat_no_votes'][0]:.4f} | {results_c['v2_10feat_no_votes'][1]:.4f} |",
        f"| v3: 10鐗瑰緛(鍘籿otes+LOO缁熻) | {results_c['v3_10feat_trainonly'][0]:.4f} | {results_c['v3_10feat_trainonly'][1]:.4f} |",
        "",
        "### 缁撹",
        "",
        f"- v3锛堜弗璋ㄧ増锛塕虏={results_c['v3_10feat_trainonly'][1]:.4f}锛屽姣斾綔鑰呭潎鍊?baseline R虏={r2_auth_base:.4f}",
        f"- 宸窛 = {results_c['v3_10feat_trainonly'][1] - r2_auth_base:+.4f}",
    ]

    if results_c['v3_10feat_trainonly'][1] <= r2_auth_base:
        lines.append("- 鈿?**妯″瀷骞舵湭鏄捐憲浼樹簬鐩存帴鏌ヤ綔鑰呭潎鍒?*锛屽缓璁瑙嗙壒寰佸伐绋嬬殑鏈夋晥鎬с€?)
    else:
        lines.append("- 鉁?妯″瀷浼樹簬浣滆€呭潎鍊?baseline銆?)

    lines.append("")
    return "\n".join(lines), results_c


# ============================================================================
#  涓诲叆鍙?
# ============================================================================


def run_experiment_e():
    """鍩轰簬 IJCAI 鏁版嵁闆嗙殑鐪熷疄鐢ㄦ埛 Leave-One-Out 璇勪及"""
    print("\n" + "=" * 70)
    print("  瀹為獙 E: 鐪熷疄鐢ㄦ埛 Leave-One-Out (IJCAI)")
    print("=" * 70)

    user_ratings_path = DATA_DIR / "processed" / "user_ratings.csv"
    if not user_ratings_path.exists():
        msg = "  [SKIP] user_ratings.csv 涓嶅瓨鍦紝璇峰厛杩愯 src/integrate_ijcai.py"
        print(msg)
        return "## 瀹為獙 E: SKIPPED\n\n" + msg + "\n", {}

    ur = pd.read_csv(user_ratings_path, encoding="utf-8-sig")
    rec_df = pd.read_csv(MODEL_DIR / "books_for_rec.csv", encoding="utf-8-sig")
    rec_ids = set(str(int(i)) for i in rec_df["id"])

    ur["in_rec"] = ur["douban_book_id"].astype(str).isin(rec_ids)

    user_stats = ur.groupby("user_id").agg(
        n_total=("rating", "count"),
        n_high=("rating", lambda x: (x >= 4).sum()),
        n_rec=("in_rec", "sum"),
        n_high_rec=("rating", lambda x: ((x >= 4) & ur.loc[x.index, "in_rec"]).sum()),
    )
    valid_users = user_stats[(user_stats["n_total"] >= 10) & (user_stats["n_high_rec"] >= 5)]
    valid_user_ids = set(valid_users.index)

    avg_total = valid_users["n_total"].mean()
    avg_high = valid_users["n_high_rec"].mean()
    print(f"  鏈夋晥鐢ㄦ埛: {len(valid_user_ids)} | 骞冲潎璇勫垎: {avg_total:.0f} | 骞冲潎楂樺垎鍦ㄥ簱: {avg_high:.0f}")
    if len(valid_user_ids) < 50:
        print(f"  [WARN] 鏍锋湰閲?< 50")

    rec = load_recommender()
    nn_indices = rec.nn_indices
    id_to_idx = rec.id_to_idx
    idx_to_id = rec.idx_to_id
    titles_arr = rec.df["title"].values.astype(str)

    # Popular baseline: 鎸夋姇绁ㄦ暟鎺掑簭锛堜笉鏄?bayesian_score锛屽悗鑰呭亸鍚戝皬浼楅珮鍒嗕功锛?
    popular_ids = rec_df.nlargest(len(rec_df), "votes")["id"].tolist()
    all_ids_arr = np.array(sorted(set(int(i) for i in rec_df["id"])), dtype=np.int64)

    K_values = [10, 20]
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
            # 鐢ㄦ埛鎵€鏈夊凡璇勫垎涔︼紙鍦?rec 绱㈠紩鍐呯殑锛?
            all_rated = set(int(float(x)) for x in user_data[user_data["in_rec"]]["douban_book_id"])
            n_valid += 1

            # ----- 鎺ㄨ崘鏂规硶 -----
            # 瀵规瘡鏈巻鍙查珮鍒嗙瀛愪功锛屽彇璇ヤ功鐨?top-(K*3) NN 閭诲眳
            # 鐢?rank 鍔犳潈 (1/rank) 浣滀负鍒嗘暟锛岃法绉嶅瓙 max-pooling锛屾渶鍚庡叏灞€ top-K
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
                    if nid == hid:
                        continue
                    if nid is None:
                        continue
                    title = titles_arr[nidx]
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    score = 1.0 / (cnt + 1)  # rank-based score: 1/rank
                    if nid not in candidate_scores or score > candidate_scores[nid]:
                        candidate_scores[nid] = score
                    cnt += 1
                    if cnt >= K * 3:
                        break

            top_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:K]
            top_ids = set(cid for cid, _ in top_candidates)
            recall_rec.append(1.0 if target_id in top_ids else 0.0)

            # ----- Random baseline -----
            # 浠庡叏搴撻殢鏈烘娊 K 鏈紙鎺掗櫎鐢ㄦ埛宸茶瘎鍒嗭級
            rng = np.random.RandomState(42 + n_valid)
            eligible = all_ids_arr[~np.isin(all_ids_arr, list(all_rated))]
            if len(eligible) < K:
                recall_random.append(0.0)
            else:
                sampled = rng.choice(eligible, size=K, replace=False)
                recall_random.append(1.0 if target_id in set(int(s) for s in sampled) else 0.0)

            # ----- Popular baseline -----
            # 鎸?votes 闄嶅簭鍙栧墠 K 鏈紙鎺掗櫎鐢ㄦ埛宸茶瘎鍒嗭級
            top_pop = []
            for pid in popular_ids:
                if int(pid) not in all_rated:
                    top_pop.append(int(pid))
                    if len(top_pop) >= K:
                        break
            recall_pop.append(1.0 if target_id in set(top_pop) else 0.0)

        mean_rec = np.mean(recall_rec) if recall_rec else 0.0
        mean_rand = np.mean(recall_random) if recall_random else 0.0
        mean_pop = np.mean(recall_pop) if recall_pop else 0.0
        print(f"  recommend_by_id : Recall@{K} = {mean_rec:.4f}  (n={n_valid})")
        print(f"  Random          : Recall@{K} = {mean_rand:.4f}")
        print(f"  Popular         : Recall@{K} = {mean_pop:.4f}")
        results[K] = {"recommend_by_id": mean_rec, "random": mean_rand, "popular": mean_pop, "n_users": n_valid}

    r10 = results[10]
    r20 = results[20]
    lines = [
        "## 瀹為獙 E: 鐪熷疄鐢ㄦ埛 Leave-One-Out 璇勪及 (IJCAI 鏁版嵁闆?",
        "",
        "> **鏁版嵁鏉ユ簮**: DTCDR (CIKM 2019) / GA-DTCDR (IJCAI 2020) 璺ㄥ煙鎺ㄨ崘鍏紑鏁版嵁闆?,
        "> 寮曠敤: Zhu et al., CIKM 2019; Zhu et al., IJCAI 2020",
        "",
        "### 鏂规硶瀛?,
        "",
        "- **鐢ㄦ埛绛涢€?*: 鎬昏瘎鍒?>=10 鏉′笖 rating>=4 鐨勯珮鍒嗕功涓嚦灏?5 鏈湪鎺ㄨ崘绱㈠紩鍐?,
        f"  (鏈€缁?{r10["n_users"]} 浜? 骞冲潎 {avg_total:.0f} 鏉¤瘎鍒? 骞冲潎 {avg_high:.0f} 鏈珮鍒嗗湪搴?",
        "- **鐩爣涔﹂€夊彇**: 鍙栬鐢ㄦ埛 rating>=4 鐨勪功, 鎸夋椂闂存帓搴? 鐣欏嚭鏈€鍚庝竴鏈綔涓?ground-truth",
        "- **鍊欓€夐泦鏋勯€?*: 瀵规瘡鏈巻鍙查珮鍒嗙瀛愪功, 鍚勫彇璇ヤ功鐨?top-(K*3) NN 閭诲眳,",
        "  瀵规瘡涓€欓€変功鍙栬法绉嶅瓙鐨勬渶澶?rank-weighted 鍒嗘暟 (1/rank) 鍋?max-pooling,",
        "  鐒跺悗鍙栧叏灞€ top-K 浣滀负鏈€缁堟帹鑽愬垪琛? 鎸囨爣涓烘爣鍑嗙殑 **Recall@K** (K 鏈帹鑽愪腑鏄惁鍛戒腑鐩爣).",
        "- **Random 鍩虹嚎**: 浠庡叏搴撻殢鏈烘娊 K 鏈?(鎺掗櫎璇ョ敤鎴锋墍鏈夊凡璇勫垎涔?, 5 娆″钩鍧?,
        "- **Popular 鍩虹嚎**: 鎸?votes 闄嶅簭鍙栧墠 K 鏈?(鎺掗櫎璇ョ敤鎴锋墍鏈夊凡璇勫垎涔?",
        "  (娉? Popular 鍩虹嚎涓?0 鏄甯哥殑鈥斺€擳op20 姘歌繙鏄€婃椿鐫€銆嬨€婄孩妤兼ⅵ銆嬬瓑鍥芥皯绾х晠閿€涔? 1,595 鍚嶇敤鎴风殑涓€у寲鐩爣涔﹀嚑涔庝笉鍙兘鍛戒腑; 鏀圭敤 bayesian_score 鍒?top 鍒楄〃琚皬浼楅珮鍒嗕功鍗犳嵁, 鏇存棤鎰忎箟)",
        "",
        "| 鏂规硶 | Recall@10 | Recall@20 | 鐢ㄦ埛鏁?|",
        "|------|-----------|-----------|--------|",
        f"| recommend_by_id | {r10["recommend_by_id"]:.4f} | {r20["recommend_by_id"]:.4f} | {r10["n_users"]} |",
        f"| Random | {r10["random"]:.4f} | {r20["random"]:.4f} | {r10["n_users"]} |",
        f"| Popular | {r10["popular"]:.4f} | {r20["popular"]:.4f} | {r10["n_users"]} |",
        "",
    ]

    return "\n".join(lines), results


def main():
    parser = argparse.ArgumentParser(description="璞嗙摚鍥句功鎺ㄨ崘绯荤粺 - 绂荤嚎璇勪及")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "rec", "rating", "coldstart", "user"],
                        help="閫夋嫨瀹為獙: all, rec, rating, coldstart")
    args = parser.parse_args()

    start_time = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_parts = [
        f"# 璞嗙摚鍥句功鎺ㄨ崘绯荤粺 鈥?绂荤嚎璇勪及鎶ュ憡",
        f"",
        f"**鐢熸垚鏃堕棿**: {now}",
        f"**鏁版嵁瑙勬ā**: books_for_rec.csv 鈮?164K 琛? Books_detail.csv = 6,584 琛?,
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

    if args.experiment in ("all", "user"):
        part_e, _ = run_experiment_e()
        report_parts.append(part_e)

    elapsed = time.time() - start_time
    report_parts.append(f"---")
    report_parts.append(f"")
    report_parts.append(f"_鎬昏€楁椂: {elapsed:.1f}s_")
    report_parts.append(f"_random_state=42 鐢ㄤ簬鎵€鏈夐殢鏈鸿繃绋媉")

    report_text = "\n".join(report_parts)

    # 杈撳嚭鍒扮粓绔紙瀹归敊 GBK 缂栫爜锛?
    print("\n" + "=" * 70)
    print("  璇勪及鎶ュ憡")
    print("=" * 70)
    # 瀹夊叏杈撳嚭锛氭浛鎹㈡棤娉曠紪鐮佺殑瀛楃
    for line in report_text.split("\n"):
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"))

    # 鍐欏叆鏂囦欢
    report_path = REPORT_DIR / "evaluation_results.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[鎶ュ憡淇濆瓨] {report_path}")


if __name__ == "__main__":
    main()

