"""
图书推荐引擎
基于 jieba 分词 + TF-IDF + 余弦相似度的内容推荐
结合贝叶斯评分的混合推荐策略
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import jieba
import jieba.analyse


class BookRecommender:
    """基于内容的图书推荐引擎"""

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.df = None          # 主数据
        self.tfidf_matrix = None
        self.nn_model = None
        self.vectorizer = None
        self.id_to_idx = {}     # book_id -> matrix index
        self.idx_to_id = {}     # matrix index -> book_id

    # ========== 1. 加载数据与预处理 ==========

    def load_data(self):
        """加载清洗后的图书数据"""
        path = self.data_dir / "processed" / "books_scored.csv"
        self.df = pd.read_csv(path, encoding="utf-8-sig")
        self.df["title"] = self.df["title"].fillna("").astype(str)
        # 去除书名中非中文字符用于分词，但保留原始书名
        print(f"[加载] {len(self.df):,} 本图书")
        return self

    def _tokenize(self, text):
        """纯字符级n-gram分词（解决短书名向量碰撞问题）"""
        chars = "".join(ch for ch in text if self._has_chinese(ch))
        if not chars:
            return ""
        result = []
        for i in range(len(chars)):
            result.append(chars[i])              # 单字
        for i in range(len(chars) - 1):
            result.append(chars[i:i+2])          # bigram
        for i in range(len(chars) - 2):
            result.append(chars[i:i+3])          # trigram
        return " ".join(result)

    @staticmethod
    def _has_chinese(text):
        """判断是否包含中文字符"""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                return True
        return False

    # ========== 2. 构建 TF-IDF 矩阵 ==========

    def build_tfidf(self, max_features=None):
        """对书名进行分词并构建 TF-IDF 矩阵"""
        print("[分词] 处理书名...")
        self.df["tokens"] = self.df["title"].apply(self._tokenize)
        valid = self.df["tokens"].str.len() > 0
        print(f"  有效分词: {valid.sum():,} / {len(self.df):,} 本")

        self.df = self.df[valid].reset_index(drop=True)

        print(f"[TF-IDF] 构建向量矩阵 (max_features={max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,      # 1 + log(tf)
            min_df=1,               # 至少在3本书中出现
            max_df=0.98,             # 最多在80%的书中出现
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tokens"])
        print(f"  TF-IDF 矩阵: {self.tfidf_matrix.shape}")

        # 构建 ID 映射
        for idx, book_id in enumerate(self.df["id"]):
            self.id_to_idx[int(book_id)] = idx
            self.idx_to_id[idx] = int(book_id)

        # 保存
        self._save_artifacts()
        return self

    def _save_artifacts(self):
        """保存模型组件"""
        print("[保存] 模型文件...")
        save_npz(self.model_dir / "tfidf_matrix.npz", self.tfidf_matrix)
        with open(self.model_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        self.df[["id", "title", "rating", "votes", "bayesian_score",
                  "tokens"]].to_csv(
            self.model_dir / "books_for_rec.csv", index=False, encoding="utf-8-sig")
        print("  模型保存完成")

    # ========== 3. 最近邻搜索 ==========

    def build_nn_index(self, n_neighbors=30):
        """构建最近邻索引"""
        if self.tfidf_matrix is None:
            self._load_artifacts()

        print(f"[NN] 构建最近邻索引 (k={n_neighbors})...")
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.nn_model.fit(self.tfidf_matrix)
        print("  完成")

        # 预计算所有图书的邻居
        print("[预计算] 为所有图书计算最近邻...")
        distances, indices = self.nn_model.kneighbors(self.tfidf_matrix)
        self.nn_distances = distances
        self.nn_indices = indices

        # 保存预计算结果
        np.savez_compressed(
            self.model_dir / "nn_neighbors.npz",
            distances=distances.astype(np.float32),
            indices=indices.astype(np.int32),
        )
        print(f"  预计算完成: {indices.shape}")
        return self

    def _load_artifacts(self):
        """加载已保存的模型"""
        self.tfidf_matrix = load_npz(self.model_dir / "tfidf_matrix.npz")
        with open(self.model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        books_path = self.model_dir / "books_for_rec.csv"
        if books_path.exists():
            self.df = pd.read_csv(books_path, encoding="utf-8-sig")
            for idx, book_id in enumerate(self.df["id"]):
                self.id_to_idx[int(book_id)] = idx
                self.idx_to_id[idx] = int(book_id)
        print(f"[加载] 已保存模型 ({len(self.df):,} 本)")

    # ========== 4. 推荐方法 ==========

    def recommend_by_id(self, book_id, top_n=10):
        """根据图书ID推荐相似图书（去重同书名，优先高评分版本）"""
        if book_id not in self.id_to_idx:
            candidates = self.df[self.df["title"].str.contains(
                str(book_id), na=False)]
            if len(candidates) > 0:
                book_id = candidates.iloc[0]["id"]
            else:
                return pd.DataFrame()

        idx = self.id_to_idx[book_id]
        source_title = self.df.iloc[idx]["title"]
        distances = self.nn_distances[idx]
        indices = self.nn_indices[idx]

        seen_titles = {source_title}
        results = []
        # 搜索范围扩大到3倍，确保去重后仍有足够结果
        search_range = min(top_n * 8, len(indices) - 1)

        for dist, nidx in zip(distances[1:search_range+1], indices[1:search_range+1]):
            rec_id = self.idx_to_id[nidx]
            row = self.df[self.df["id"] == rec_id].iloc[0]
            title = row["title"]

            if title in seen_titles:
                continue
            seen_titles.add(title)

            similarity = 1.0 - float(dist)
            results.append({
                "id": rec_id,
                "title": title,
                "rating": row["rating"],
                "votes": row["votes"],
                "bayesian_score": row.get("bayesian_score", 0),
                "similarity": round(similarity, 4),
            })
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def recommend_by_title(self, query, top_n=10):
        """根据书名搜索并推荐（去重同书名）"""
        tokens = self._tokenize(query)
        if not tokens:
            return pd.DataFrame()

        query_vec = self.vectorizer.transform([tokens])

        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 扩大搜索范围，去重同书名
        search_range = min(top_n * 8, len(sims))
        top_indices = np.argsort(sims)[::-1][:search_range]

        seen_titles = set()
        results = []
        for nidx in top_indices:
            rec_id = self.idx_to_id[nidx]
            row = self.df[self.df["id"] == rec_id].iloc[0]
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            results.append({
                "id": rec_id,
                "title": title,
                "rating": row["rating"],
                "votes": row["votes"],
                "bayesian_score": row.get("bayesian_score", 0),
                "similarity": round(float(sims[nidx]), 4),
            })
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)

    def hybrid_recommend(self, book_id, top_n=10, alpha=0.5):
        """混合推荐: alpha*内容相似度 + (1-alpha)*贝叶斯评分，去重推荐"""
        # 扩大搜索范围
        content_recs = self.recommend_by_id(book_id, top_n=max(top_n * 5, 30))
        if content_recs.empty:
            return pd.DataFrame()

        # 归一化
        sim_max = content_recs["similarity"].max()
        bs_max = content_recs["bayesian_score"].max()
        bs_min = content_recs["bayesian_score"].min()

        content_recs["bs_norm"] = 0.5
        if bs_max > bs_min:
            content_recs["bs_norm"] = (content_recs["bayesian_score"] - bs_min) / (bs_max - bs_min)

        content_recs["sim_norm"] = content_recs["similarity"] / sim_max if sim_max > 0 else 1

        content_recs["hybrid_score"] = (
            alpha * content_recs["sim_norm"] +
            (1 - alpha) * content_recs["bs_norm"]
        )

        result = content_recs.nlargest(top_n, "hybrid_score")
        result = result.drop(columns=["sim_norm", "bs_norm"], errors="ignore")
        return result

    def get_popular_recommendations(self, top_n=20, min_votes=100):
        """基于贝叶斯评分的通用推荐"""
        candidates = self.df[self.df["votes"] >= min_votes]
        top = candidates.nlargest(top_n, "bayesian_score")
        return top[["id", "title", "rating", "votes", "bayesian_score"]]


if __name__ == "__main__":
    print("=" * 60)
    print("  图书推荐引擎 - 构建与测试")
    print("=" * 60)

    rec = BookRecommender()
    rec.load_data()
    rec.build_tfidf(max_features=None)
    rec.build_nn_index(n_neighbors=30)

    # 测试: 基于ID推荐
    print("\n--- 测试: recommend_by_id ---")
    r1 = rec.recommend_by_id(4913064, top_n=5)
    print("查询: 《活着》(ID=4913064)")
    print(r1.to_string(index=False))

    # 测试: 基于书名搜索推荐
    print("\n--- 测试: recommend_by_title ---")
    r2 = rec.recommend_by_title("三体", top_n=5)
    print("查询: '三体'")
    print(r2.to_string(index=False))

    # 测试: 混合推荐
    print("\n--- 测试: hybrid_recommend ---")
    r3 = rec.hybrid_recommend(4913064, top_n=5)
    print("查询: 《活着》混合推荐")
    print(r3.to_string(index=False))

    # 测试: 热门推荐
    print("\n--- 测试: get_popular_recommendations ---")
    r4 = rec.get_popular_recommendations(top_n=10)
    print("热门推荐 Top 10:")
    for _, row in r4.iterrows():
        print(f"  {row["title"][:30]:<32s} R={row["rating"]:.1f} V={row["votes"]:>7,} BS={row["bayesian_score"]:.4f}")

    print("\n[Done] 推荐引擎构建完成")
