"""
冷启动评分预测模块 (Cold-Start Rating Predictor)
用于预测数据集中不存在的书籍的豆瓣评分

Features:
- Publisher平均评分、作者平均评分等统计特征
- GradientBoostingRegressor + 分位数回归置信区间
- 余弦相似度检索最相似书籍
"""
import pandas as pd
import numpy as np
import re
import pickle
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ColdStartPredictor:
    """冷启动评分预测器"""

    def __init__(self):
        self.df = None
        self.model = None
        self.model_lower = None
        self.model_upper = None
        self.feature_names = None
        self.feature_matrix = None
        self.book_ids = None
        self.titles = None
        self.metrics = {}
        self._stats_cache = {}  # publisher/author avg ratings

    def load_data(self):
        """加载并清洗数据"""
        print("[ColdStart] Loading data...")
        df = pd.read_csv(DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
        df = df[df["crawl_status"] == "success"].copy()
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
        df["pages"] = pd.to_numeric(df["pages"], errors="coerce")
        df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")

        df = df.dropna(subset=["Rating", "Votes"])
        df = df[(df["Rating"] >= 1) & (df["Rating"] <= 10)]
        df = df[df["Votes"] >= 10]

        df["author"] = df["author"].fillna("未知").astype(str)
        df["publisher"] = df["publisher"].fillna("未知").astype(str)
        df["binding"] = df["binding"].fillna("其他").astype(str)
        df["pages"] = df["pages"].fillna(300).astype(int)
        df["pub_year"] = df["pub_year"].fillna(2010).astype(int)

        # Derived features
        df["is_translation"] = (df["translator"].notna() | df["original_title"].notna()).astype(int)
        df["is_series"] = df["series"].notna().astype(int)

        self.df = df
        print(f"  Loaded {len(df):,} books for training")
        return self

    def build_stats(self):
        """v3: 对全量数据构建统计量（用于预测时的查询），训练时内部用 LOO 避免泄露"""
        print("[ColdStart] Building statistical features (v3: train-only, no votes_log)...")
        df = self.df

        # Publisher stats
        pub_stats = df.groupby("publisher").agg(
            pub_avg_rating=("Rating", "mean"),
            pub_book_count=("Rating", "count"),
            pub_std_rating=("Rating", "std"),
        ).fillna(0)
        self._stats_cache["publisher"] = pub_stats
        print(f"  Publishers: {len(pub_stats)}")

        # Author stats
        auth_stats = df.groupby("author").agg(
            author_avg_rating=("Rating", "mean"),
            author_book_count=("Rating", "count"),
        ).fillna(0)
        self._stats_cache["author"] = auth_stats
        print(f"  Authors: {len(auth_stats)}")

        # Binding stats
        binding_stats = df.groupby("binding")["Rating"].mean().to_dict()
        self._stats_cache["binding"] = binding_stats
        print(f"  Binding types: {len(binding_stats)}")

        # Year stats (binned)
        df["year_bin"] = pd.cut(df["pub_year"], bins=range(1900, 2031, 10), labels=False)
        year_stats = df.groupby("year_bin")["Rating"].mean().to_dict()
        self._stats_cache["year_bin"] = year_stats

        # Global stats
        self._stats_cache["global_mean"] = float(df["Rating"].mean())
        self._stats_cache["global_std"] = float(df["Rating"].std())
        print(f"  Global mean rating: {self._stats_cache['global_mean']:.2f}")

        return self

    def build_features(self):
        """v3: 构建 10 维特征矩阵（去 votes_log，LOO 统计）"""
        print("[ColdStart] Building feature matrix (v3, 10 features, LOO stats)...")
        from sklearn.model_selection import train_test_split
        df = self.df.copy()

        # Split: 80% for computing LOO stats
        train_idx_arr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
        train_idx = set(train_idx_arr.tolist())

        # Train-only aggregate stats
        train_subset = df.iloc[list(train_idx)]
        pub_train = train_subset.groupby("publisher").agg(
            avg=("Rating", "mean"), cnt=("Rating", "count"), std=("Rating", "std")
        ).fillna(0)
        auth_train = train_subset.groupby("author").agg(
            avg=("Rating", "mean"), cnt=("Rating", "count")
        ).fillna(0)
        bind_train = train_subset.groupby("binding")["Rating"].mean().to_dict()
        gm = float(df["Rating"].mean())
        gs = float(df["Rating"].std())

        N = len(df)
        pub_avg = np.zeros(N)
        pub_cnt = np.zeros(N)
        pub_std = np.zeros(N)
        auth_avg = np.zeros(N)
        auth_cnt = np.zeros(N)

        pub_map = df["publisher"].values
        auth_map = df["author"].values
        ratings = df["Rating"].values

        for i in range(N):
            p = pub_map[i]
            a = auth_map[i]
            r = ratings[i]

            if p in pub_train.index:
                pn = pub_train.loc[p, "cnt"]
                pm = pub_train.loc[p, "avg"]
                ps_val = pub_train.loc[p, "std"]
                if i in train_idx and pn > 1:
                    pub_avg[i] = (pm * pn - r) / (pn - 1)
                    pub_cnt[i] = pn - 1
                else:
                    pub_avg[i] = pm
                    pub_cnt[i] = pn
                pub_std[i] = ps_val if not np.isnan(ps_val) else gs
            else:
                pub_avg[i] = gm; pub_cnt[i] = 1; pub_std[i] = gs

            if a in auth_train.index:
                an = auth_train.loc[a, "cnt"]
                am = auth_train.loc[a, "avg"]
                if i in train_idx and an > 1:
                    auth_avg[i] = (am * an - r) / (an - 1)
                    auth_cnt[i] = an - 1
                else:
                    auth_avg[i] = am
                    auth_cnt[i] = an
            else:
                auth_avg[i] = gm; auth_cnt[i] = 1

        features = {}
        features["pub_avg_rating"] = pub_avg
        features["pub_book_count_log"] = np.log1p(np.maximum(pub_cnt, 1))
        features["pub_std_rating"] = pub_std
        features["author_avg_rating"] = auth_avg
        features["author_book_count_log"] = np.log1p(np.maximum(auth_cnt, 1))
        features["binding_score"] = np.array([bind_train.get(b, gm) for b in df["binding"].values])
        features["pub_year"] = df["pub_year"].clip(1900, 2030).values.astype(float)
        features["pages_log"] = np.log1p(df["pages"].clip(10, 5000).values)
        features["is_translation"] = df["is_translation"].values.astype(float)
        features["is_series"] = df["is_series"].values.astype(float)

        self.feature_names = [
            "pub_avg_rating", "pub_book_count_log", "pub_std_rating",
            "author_avg_rating", "author_book_count_log",
            "binding_score", "pub_year", "pages_log",
            "is_translation", "is_series"
        ]

        X_list = [features[name] for name in self.feature_names]
        X = np.column_stack(X_list)
        self.feature_matrix = X
        self.book_ids = df["ID"].values
        self.titles = df["Title"].values

        # Store book meta
        self._book_meta = {}
        for _, row in df.iterrows():
            self._book_meta[int(row["ID"])] = {
                "title": str(row["Title"]), "rating": float(row["Rating"]),
                "author": str(row["author"]), "publisher": str(row["publisher"]),
            }

        print(f"  Feature matrix: {X.shape} (10 features, LOO stats)")
        return self

    def train(self):
        """训练 GradientBoostingRegressor + 分位数回归"""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        print("[ColdStart] Training models...")
        y = self.df["Rating"].values
        X = self.feature_matrix

        # Main model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(X, y)

        # Prediction interval models (quantile regression)
        self.model_lower = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, loss="quantile", alpha=0.05,
        )
        self.model_lower.fit(X, y)

        self.model_upper = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, loss="quantile", alpha=0.95,
        )
        self.model_upper.fit(X, y)

        # Metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        y_pred = self.model.predict(X)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")
        self.metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred),
            "CV_R2": float(np.mean(cv_scores)),
            "CV_R2_std": float(np.std(cv_scores)),
            "n_samples": len(y),
            "n_samples": len(y),
        }
        print(f"  MAE: {self.metrics['MAE']:.3f}")
        print(f"  R2:  {self.metrics['R2']:.3f}")
        print(f"  CV5: {self.metrics['CV_R2']:.3f}")

        # Feature importance
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
            print(f"    {name:<25s}: {imp:.4f}")

        return self

    def save(self):
        """???? (joblib for sklearn models, pickle for metadata)"""
        import joblib
        # sklearn models via joblib (better cross-version compatibility)
        joblib.dump(self.model, MODEL_DIR / "coldstart_model.joblib")
        joblib.dump(self.model_lower, MODEL_DIR / "coldstart_model_lower.joblib")
        joblib.dump(self.model_upper, MODEL_DIR / "coldstart_model_upper.joblib")
        # metadata via pickle
        meta_path = MODEL_DIR / "coldstart_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "feature_matrix": self.feature_matrix,
                "book_ids": self.book_ids,
                "titles": self.titles,
                "metrics": self.metrics,
                "stats_cache": self._stats_cache,
                "book_meta": {
                    int(row.ID): {
                        "title": str(row.Title),
                        "rating": float(row.Rating),
                        "author": str(row.author),
                        "publisher": str(row.publisher),
                    }
                    for _, row in self.df.iterrows()
                } if self.df is not None else {},
            }, f)
        print(f"  [Saved] {MODEL_DIR}")
        return self

    @staticmethod
    def load(path=None):
        """???? (joblib for sklearn, pickle for metadata)"""
        import joblib
        if path is None:
            path = MODEL_DIR / "coldstart_meta.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = ColdStartPredictor()
        predictor.model = joblib.load(MODEL_DIR / "coldstart_model.joblib")
        predictor.model_lower = joblib.load(MODEL_DIR / "coldstart_model_lower.joblib")
        predictor.model_upper = joblib.load(MODEL_DIR / "coldstart_model_upper.joblib")
        predictor.feature_names = data["feature_names"]
        predictor.feature_matrix = data["feature_matrix"]
        predictor.book_ids = data["book_ids"]
        predictor.titles = data["titles"]
        predictor.metrics = data["metrics"]
        predictor._stats_cache = data["stats_cache"]
        predictor._book_meta = data.get("book_meta", {})
        return predictor

    def predict(self, author, publisher, pub_year, pages, binding, is_translation, is_series, votes_estimate=None):
        """v3: 预测单本书评分（10 维特征，无 votes_log）
        返回 (prediction, lower_bound, upper_bound, feature_vector, similar_books)
        votes_estimate 参数保留兼容但不再使用"""
        from sklearn.metrics.pairwise import cosine_similarity

        pub_stats = self._stats_cache["publisher"]
        auth_stats = self._stats_cache["author"]
        binding_stats = self._stats_cache["binding"]
        global_mean = self._stats_cache["global_mean"]
        global_std = self._stats_cache["global_std"]

        # Build feature vector (10 features, no votes_log)
        features = {}

        if publisher in pub_stats.index:
            features["pub_avg_rating"] = pub_stats.loc[publisher, "pub_avg_rating"]
            features["pub_book_count_log"] = np.log1p(pub_stats.loc[publisher, "pub_book_count"])
            features["pub_std_rating"] = pub_stats.loc[publisher, "pub_std_rating"]
        else:
            features["pub_avg_rating"] = global_mean
            features["pub_book_count_log"] = np.log1p(1)
            features["pub_std_rating"] = global_std

        if author in auth_stats.index:
            features["author_avg_rating"] = auth_stats.loc[author, "author_avg_rating"]
            features["author_book_count_log"] = np.log1p(auth_stats.loc[author, "author_book_count"])
        else:
            features["author_avg_rating"] = global_mean
            features["author_book_count_log"] = np.log1p(1)

        features["binding_score"] = binding_stats.get(binding, global_mean)
        features["pub_year"] = np.clip(pub_year, 1900, 2030)
        features["pages_log"] = np.log1p(max(10, min(pages, 5000)))
        features["is_translation"] = int(is_translation)
        features["is_series"] = int(is_series)

        X = np.array([[features[name] for name in self.feature_names]])

        # Predict
        pred = float(self.model.predict(X)[0])
        lower = float(self.model_lower.predict(X)[0])
        upper = float(self.model_upper.predict(X)[0])

        # Find similar books
        sims = cosine_similarity(X, self.feature_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        _meta = getattr(self, "_book_meta", {})
        similar_books = []
        for idx in top_indices:
            bid = int(self.book_ids[idx])
            info = _meta.get(bid, {})
            similar_books.append({
                "id": bid,
                "title": info.get("title", str(self.titles[idx])),
                "rating": info.get("rating"),
                "author": info.get("author", ""),
                "publisher": info.get("publisher", ""),
                "similarity": float(sims[idx]),
            })

        return pred, lower, upper, X, similar_books

    def get_feature_importance(self):
        """返回特征重要性列表"""
        if self.model is None:
            return []
        importances = self.model.feature_importances_
        return sorted(
            [{"feature": name, "importance": float(imp)} for name, imp in zip(self.feature_names, importances)],
            key=lambda x: -x["importance"]
        )

    def get_publisher_list(self, min_books=5):
        """返回常见出版社列表"""
        pub_stats = self._stats_cache["publisher"]
        return pub_stats[pub_stats["pub_book_count"] >= min_books].index.tolist()

    def get_author_list(self, min_books=3):
        """返回常见作者列表"""
        auth_stats = self._stats_cache["author"]
        return auth_stats[auth_stats["author_book_count"] >= min_books].index.tolist()


if __name__ == "__main__":
    print("=" * 60)
    print("  Cold-Start Rating Predictor Training")
    print("=" * 60)

    csp = ColdStartPredictor()
    csp.load_data()
    csp.build_stats()
    csp.build_features()
    csp.train()
    csp.save()

    # Quick test
    pred, low, up, _, similar = csp.predict(
        author="余华", publisher="人民文学出版社",
        pub_year=2025, pages=350, binding="平装",
        is_translation=False, is_series=False,
    )
    print(f"\n[Test] 余华 / 人民文学出版社 / 2025 / 350p")
    print(f"  Predicted: {pred:.2f}  [{low:.2f} - {up:.2f}]")
    print(f"  Similar books:")
    for sb in similar:
        r = sb.get('rating')
        rs = f"{r:.1f}" if r is not None else "?"
        print(f"    {sb['title'][:30]:<32s} {rs}分 sim={sb['similarity']:.3f}")

    print("\n[Done]")
