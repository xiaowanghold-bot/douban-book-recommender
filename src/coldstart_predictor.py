"""
冷启动评分预测模块 (Cold-Start Rating Predictor)
用于预测数据集中不存在的书籍的豆瓣评分

Features:
- Publisher平均评分、作者平均评分等统计特征
- GradientBoostingRegressor + 分位数回归预测区间
- 余弦相似度检索最相似书籍
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path

try:
    from .coldstart_training import (
        build_coldstart_feature_frame,
        build_coldstart_stats,
        build_oof_coldstart_feature_frame,
        prepare_coldstart_dataframe,
        train_coldstart_models,
    )
except ImportError:  # Streamlit imports src modules as top-level modules.
    from coldstart_training import (
        build_coldstart_feature_frame,
        build_coldstart_stats,
        build_oof_coldstart_feature_frame,
        prepare_coldstart_dataframe,
        train_coldstart_models,
    )

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
        self.artifact_version = None
        self._stats_cache = {}  # publisher/author avg ratings

    def load_data(self):
        """加载并清洗数据"""
        print("[ColdStart] Loading data...")
        detail = pd.read_csv(
            DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig"
        )
        self.df = prepare_coldstart_dataframe(detail)
        print(f"  Loaded {len(self.df):,} books for training")
        return self

    def build_stats(self):
        """构建预测时使用的全量参考统计。"""
        print("[ColdStart] Building runtime reference statistics...")
        self._stats_cache = build_coldstart_stats(self.df)
        pub_stats = self._stats_cache["publisher"]
        auth_stats = self._stats_cache["author"]
        print(f"  Publishers: {len(pub_stats)}")
        print(f"  Authors: {len(auth_stats)}")
        print(f"  Binding types: {len(self._stats_cache['binding'])}")
        print(f"  Global mean rating: {self._stats_cache['global_mean']:.2f}")
        return self

    def build_features(self):
        """构建全量 5 折 OOF 特征矩阵。"""
        print("[ColdStart] Building 5-fold OOF feature matrix...")
        features = build_oof_coldstart_feature_frame(self.df)
        self.feature_names = list(features.columns)
        self.feature_matrix = features.values
        self.book_ids = self.df["ID"].values
        self.titles = self.df["Title"].values
        print(f"  Feature matrix: {self.feature_matrix.shape}")
        return self

    def train(self):
        """评估后在全量 OOF 特征上训练生产模型。"""
        print("[ColdStart] Evaluating and training v4 models...")
        result = train_coldstart_models(self.df)
        self.model = result["model"]
        self.model_lower = result["model_lower"]
        self.model_upper = result["model_upper"]
        self.feature_names = result["feature_names"]
        self.feature_matrix = result["feature_matrix"]
        self.similarity_matrix = result["similarity_matrix"]
        self.similarity_scaler = result["similarity_scaler"]
        self._stats_cache = result["stats_cache"]
        self.metrics = result["metrics"]
        self.book_ids = self.df["ID"].values
        self.titles = self.df["Title"].values
        self._book_meta = {
            int(row.ID): {
                "title": str(row.Title),
                "rating": float(row.Rating),
                "author": str(row.author),
                "publisher": str(row.publisher),
            }
            for _, row in self.df.iterrows()
        }
        print(f"  RMSE (test): {self.metrics['RMSE']:.3f}")
        print(f"  MAE  (test): {self.metrics['MAE']:.3f}")
        print(f"  R2   (test): {self.metrics['R2']:.3f}")
        print(
            f"  CV5  (nested): {self.metrics['CV_R2']:.3f} "
            f"± {self.metrics['CV_R2_std']:.3f}"
        )

        # Feature importance
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
            print(f"    {name:<25s}: {imp:.4f}")

        return self

    def save(self):
        """Save sklearn models and their runtime metadata."""
        import joblib
        # sklearn models via joblib (better cross-version compatibility)
        joblib.dump(self.model, MODEL_DIR / "coldstart_model.joblib")
        joblib.dump(self.model_lower, MODEL_DIR / "coldstart_model_lower.joblib")
        joblib.dump(self.model_upper, MODEL_DIR / "coldstart_model_upper.joblib")
        # metadata via pickle
        meta_path = MODEL_DIR / "coldstart_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "artifact_version": 4,
                "feature_names": self.feature_names,
                "feature_matrix": self.feature_matrix,
                "similarity_matrix": self.similarity_matrix,
                "similarity_scaler": self.similarity_scaler,
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
        with open(
            MODEL_DIR / "coldstart_model_meta.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "artifact_version": 4,
                    "model": "GradientBoostingRegressor",
                    "encoding": "5-fold OOF aggregate features",
                    "similarity": "standardized Euclidean distance",
                    "feature_names": self.feature_names,
                    "metrics": self.metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  [Saved] {MODEL_DIR}")
        return self

    @staticmethod
    def load(path=None):
        """Load sklearn models and runtime metadata."""
        import joblib
        if path is None:
            path = MODEL_DIR / "coldstart_meta.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = ColdStartPredictor()
        predictor.artifact_version = data.get("artifact_version", 3)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            predictor.model = joblib.load(MODEL_DIR / "coldstart_model.joblib")
            predictor.model_lower = joblib.load(
                MODEL_DIR / "coldstart_model_lower.joblib"
            )
            predictor.model_upper = joblib.load(
                MODEL_DIR / "coldstart_model_upper.joblib"
            )
        predictor.feature_names = data["feature_names"]
        predictor.feature_matrix = data["feature_matrix"]
        predictor.similarity_matrix = data.get(
            "similarity_matrix", predictor.feature_matrix
        )
        predictor.similarity_scaler = data.get("similarity_scaler")
        if predictor.similarity_scaler is None:
            from sklearn.preprocessing import StandardScaler

            predictor.similarity_scaler = StandardScaler().fit(
                predictor.feature_matrix
            )
            predictor.similarity_matrix = predictor.similarity_scaler.transform(
                predictor.feature_matrix
            )
        predictor.book_ids = data["book_ids"]
        predictor.titles = data["titles"]
        predictor.metrics = data["metrics"]
        predictor._stats_cache = data["stats_cache"]
        predictor._book_meta = data.get("book_meta", {})
        return predictor

    def predict(self, author, publisher, pub_year, pages, binding, is_translation, is_series, votes_estimate=None):
        """v4: 预测单本书评分（10 维特征，无 votes_log）
        返回 (prediction, lower_bound, upper_bound, feature_vector, similar_books)
        votes_estimate 参数保留兼容但不再使用"""
        from sklearn.metrics.pairwise import euclidean_distances

        input_frame = pd.DataFrame(
            [{
                "author": author,
                "publisher": publisher,
                "pub_year": pub_year,
                "pages": pages,
                "binding": binding,
                "is_translation": int(is_translation),
                "is_series": int(is_series),
            }]
        )
        X = build_coldstart_feature_frame(input_frame, self._stats_cache).values

        # Predict
        pred = float(self.model.predict(X)[0])
        lower_raw = float(self.model_lower.predict(X)[0])
        upper_raw = float(self.model_upper.predict(X)[0])
        lower = min(lower_raw, upper_raw, pred)
        upper = max(lower_raw, upper_raw, pred)

        # Find similar books in a standardized feature space.
        scaled_x = self.similarity_scaler.transform(X)
        distances = euclidean_distances(scaled_x, self.similarity_matrix)[0]
        top_indices = np.argsort(distances)[:5]

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
                "similarity": float(1.0 / (1.0 + distances[idx])),
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
    csp.train()
    csp.save()

    # Quick test
    pred, low, up, _, similar = csp.predict(
        author="余华", publisher="人民文学出版社",
        pub_year=2025, pages=350, binding="平装",
        is_translation=False, is_series=False,
    )
    print("\n[Test] 余华 / 人民文学出版社 / 2025 / 350p")
    print(f"  Predicted: {pred:.2f}  [{low:.2f} - {up:.2f}]")
    print("  Similar books:")
    for sb in similar:
        r = sb.get('rating')
        rs = f"{r:.1f}" if r is not None else "?"
        print(f"    {sb['title'][:30]:<32s} {rs}分 sim={sb['similarity']:.3f}")

    print("\n[Done]")
