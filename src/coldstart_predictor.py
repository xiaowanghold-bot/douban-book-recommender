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
        """构建出版社、作者等统计特征"""
        print("[ColdStart] Building statistical features...")
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
        """从统计数据构建特征矩阵"""
        print("[ColdStart] Building feature matrix...")
        df = self.df
        pub_stats = self._stats_cache["publisher"]
        auth_stats = self._stats_cache["author"]
        binding_stats = self._stats_cache["binding"]

        features = {}

        # Statistical features
        features["pub_avg_rating"] = df["publisher"].map(pub_stats["pub_avg_rating"]).fillna(self._stats_cache["global_mean"])
        features["pub_book_count_log"] = np.log1p(df["publisher"].map(pub_stats["pub_book_count"]).fillna(1))
        features["pub_std_rating"] = df["publisher"].map(pub_stats["pub_std_rating"]).fillna(self._stats_cache["global_std"])

        features["author_avg_rating"] = df["author"].map(auth_stats["author_avg_rating"]).fillna(self._stats_cache["global_mean"])
        features["author_book_count_log"] = np.log1p(df["author"].map(auth_stats["author_book_count"]).fillna(1))

        features["binding_score"] = df["binding"].map(lambda x: binding_stats.get(x, self._stats_cache["global_mean"]))

        # Raw features
        features["pub_year"] = df["pub_year"].clip(1900, 2030)
        features["pages_log"] = np.log1p(df["pages"].clip(10, 5000))
        features["is_translation"] = df["is_translation"]
        features["is_series"] = df["is_series"]
        features["votes_log"] = np.log1p(df["Votes"])

        self.feature_names = [
            "pub_avg_rating", "pub_book_count_log", "pub_std_rating",
            "author_avg_rating", "author_book_count_log",
            "binding_score", "pub_year", "pages_log",
            "is_translation", "is_series", "votes_log"
        ]

        # Build matrix
        X_list = [features[name].values for name in self.feature_names]
        X = np.column_stack(X_list)
        self.feature_matrix = X
        self.book_ids = df["ID"].values
        self.titles = df["Title"].values

        print(f"  Feature matrix: {X.shape}")
        print(f"  Features: {self.feature_names}")
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
        """保存模型"""
        path = MODEL_DIR / "coldstart_predictor.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_lower": self.model_lower,
                "model_upper": self.model_upper,
                "feature_names": self.feature_names,
                "feature_matrix": self.feature_matrix,
                "book_ids": self.book_ids,
                "titles": self.titles,
                "metrics": self.metrics,
                "stats_cache": self._stats_cache,
            }, f)
        print(f"  [Saved] {path}")
        return self

    @staticmethod
    def load(path=None):
        """加载模型"""
        if path is None:
            path = MODEL_DIR / "coldstart_predictor.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = ColdStartPredictor()
        predictor.model = data["model"]
        predictor.model_lower = data["model_lower"]
        predictor.model_upper = data["model_upper"]
        predictor.feature_names = data["feature_names"]
        predictor.feature_matrix = data["feature_matrix"]
        predictor.book_ids = data["book_ids"]
        predictor.titles = data["titles"]
        predictor.metrics = data["metrics"]
        predictor._stats_cache = data["stats_cache"]
        return predictor

    def predict(self, author, publisher, pub_year, pages, binding, is_translation, is_series, votes_estimate=1000):
        """预测单本书的评分，返回 (prediction, lower_bound, upper_bound, feature_vector)"""
        from sklearn.metrics.pairwise import cosine_similarity

        pub_stats = self._stats_cache["publisher"]
        auth_stats = self._stats_cache["author"]
        binding_stats = self._stats_cache["binding"]
        global_mean = self._stats_cache["global_mean"]
        global_std = self._stats_cache["global_std"]

        # Build feature vector (same order as feature_names)
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
        features["votes_log"] = np.log1p(max(1, votes_estimate))

        X = np.array([[features[name] for name in self.feature_names]])

        # Predict
        pred = float(self.model.predict(X)[0])
        lower = float(self.model_lower.predict(X)[0])
        upper = float(self.model_upper.predict(X)[0])

        # Find similar books
        sims = cosine_similarity(X, self.feature_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        similar_books = []
        for idx in top_indices:
            similar_books.append({
                "id": int(self.book_ids[idx]),
                "title": str(self.titles[idx]),
                "rating": float(self.df["Rating"].iloc[idx]) if self.df is not None else None,
                "similarity": float(sims[idx]),
            })

        if self.df is not None:
            for sb in similar_books:
                row = self.df[self.df["ID"] == sb["id"]]
                if len(row) > 0:
                    sb["rating"] = float(row["Rating"].iloc[0])
                    sb["author"] = str(row["author"].iloc[0])
                    sb["publisher"] = str(row["publisher"].iloc[0])

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
        is_translation=False, is_series=False, votes_estimate=5000
    )
    print(f"\n[Test] 余华 / 人民文学出版社 / 2025 / 350p")
    print(f"  Predicted: {pred:.2f}  [{low:.2f} - {up:.2f}]")
    print(f"  Similar books:")
    for sb in similar:
        print(f"    {sb['title'][:30]:<32s} {sb['rating']:.1f}分 sim={sb['similarity']:.3f}")

    print("\n[Done]")
