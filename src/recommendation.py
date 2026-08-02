"""
图书推荐引擎
基于 jieba 分词 + TF-IDF + 余弦相似度的内容推荐
结合贝叶斯评分的混合推荐策略

模式:
  mode='semantic' (默认): jieba 分词, 文档 = 书名+作者+标签+简介
  mode='char': 字符级 n-gram (旧版, 向后兼容)
"""
import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import jieba


class BookRecommender:
    """基于内容的图书推荐引擎"""

    def __init__(self, data_dir=None, mode="semantic"):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        self.df = None
        self.tfidf_matrix = None
        self.nn_model = None
        self.vectorizer = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self._normalized_titles = None

        self._tags = {}
        self._descriptions = {}
        self._author_map = {}
        self._valid_tags = set()
        self._dict_loaded = False

    def _load_aux_data(self):
        if self.mode != "semantic" or self._dict_loaded:
            return

        tags_path = self.data_dir / "processed" / "book_tags.json"
        if tags_path.exists():
            with open(tags_path, "r", encoding="utf-8") as f:
                self._tags = json.load(f)
            print(f"[aux] tags: {len(self._tags):,}")

        tc_path = self.data_dir / "processed" / "tag_counts.csv"
        if tc_path.exists():
            tc = pd.read_csv(tc_path)
            self._valid_tags = set(tc[tc["count"] >= 5]["tag"].tolist())
            print(f"[aux] valid tags: {len(self._valid_tags):,}")

        desc_path = self.data_dir / "processed" / "book_descriptions.json"
        if desc_path.exists():
            with open(desc_path, "r", encoding="utf-8") as f:
                self._descriptions = json.load(f)
            print(f"[aux] descs: {len(self._descriptions):,}")

        detail_path = self.data_dir / "raw" / "Books_detail.csv"
        if detail_path.exists():
            detail = pd.read_csv(detail_path)
            for _, row in detail.iterrows():
                book_id = str(int(row["ID"]))
                author = str(row.get("author", "")).strip()
                if author and author != "nan":
                    author_clean = re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", author).strip()
                    self._author_map[book_id] = author_clean[:60] if author_clean else author[:30]
            print(f"[aux] authors: {len(self._author_map):,}")

        dict_path = self.data_dir / "processed" / "custom_dict.txt"
        if dict_path.exists():
            jieba.load_userdict(str(dict_path))
            print("[aux] jieba dict loaded")

        self._dict_loaded = True

    def load_data(self):
        path = self.data_dir / "processed" / "books_scored.csv"
        self.df = pd.read_csv(path, encoding="utf-8-sig")
        self.df["title"] = self.df["title"].fillna("").astype(str)
        print(f"[load] {len(self.df):,} books")
        return self

    def _build_semantic_document(self, row):
        parts = []
        title = str(row.get("title", "")).strip()
        if title:
            parts.append(title)

        book_id = str(int(row["id"]))
        author = self._author_map.get(book_id, "")
        if author:
            parts.append(author)

        book_tags = self._tags.get(book_id, [])
        filtered = [t for t in book_tags if t in self._valid_tags]
        if filtered:
            parts.extend(filtered * 2)

        desc = self._descriptions.get(book_id, "")
        if desc:
            desc = str(desc).strip()[:300]
            if desc:
                parts.append(desc)

        return " ".join(parts) if parts else title

    def _tokenize_semantic(self, text):
        if not text or not isinstance(text, str):
            return ""
        words = jieba.cut(text)
        # Keep single Chinese chars, filter single non-Chinese
        import re as _re
        tokens = [w.strip() for w in words if len(w.strip()) >= 2 or _re.match(r'[一-鿿]', w.strip())]
        return " ".join(tokens)

    def _tokenize_char(self, text):
        chars = "".join(ch for ch in text if self._has_chinese(ch))
        if not chars:
            return ""
        result = []
        for i in range(len(chars)):
            result.append(chars[i])
        for i in range(len(chars) - 1):
            result.append(chars[i:i+2])
        for i in range(len(chars) - 2):
            result.append(chars[i:i+3])
        return " ".join(result)

    @staticmethod
    def _has_chinese(text):
        for ch in text:
            if "一" <= ch <= "鿿":
                return True
        return False

    def build_tfidf(self, max_features=None):
        if self.mode == "semantic":
            self._load_aux_data()
            self.df["tokens"] = self.df.apply(
                lambda row: self._tokenize_semantic(
                    self._build_semantic_document(row)
                ), axis=1
            )
        else:
            self.df["tokens"] = self.df["title"].apply(self._tokenize_char)

        valid = self.df["tokens"].str.len() > 0
        self.df = self.df[valid].reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features or 50000,
            sublinear_tf=True,
            min_df=2,
            max_df=0.98,
            norm="l2",
            token_pattern=r"(?u)\S+",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tokens"])

        for idx, book_id in enumerate(self.df["id"]):
            self.id_to_idx[int(book_id)] = idx
            self.idx_to_id[idx] = int(book_id)

        self._save_artifacts()
        return self

    def _save_artifacts(self):
        save_npz(self.model_dir / "tfidf_matrix.npz", self.tfidf_matrix)
        with open(self.model_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        meta = {"mode": self.mode, "n_books": len(self.df)}
        with open(self.model_dir / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        self.df[["id", "title", "rating", "votes", "bayesian_score", "tokens"]].to_csv(
            self.model_dir / "books_for_rec.csv", index=False, encoding="utf-8-sig")

    def build_nn_index(self, n_neighbors=30):
        if self.tfidf_matrix is None:
            self._load_artifacts()
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
        self.nn_model.fit(self.tfidf_matrix)
        with open(self.model_dir / "nn_neighbors.pkl", "wb") as f:
            pickle.dump(self.nn_model, f)
        distances, indices = self.nn_model.kneighbors(self.tfidf_matrix)
        np.savez(self.model_dir / "nn_neighbors.npz", distances=distances, indices=indices)

        return self
    def _load_artifacts(self):
        self.tfidf_matrix = load_npz(self.model_dir / "tfidf_matrix.npz")
        with open(self.model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        books_path = self.model_dir / "books_for_rec.csv"
        if books_path.exists():
            self.df = pd.read_csv(books_path, encoding="utf-8-sig")
            self._rebuild_lookup_indexes()
        # Load NN model (evaluate.py recomputes distances/indices on demand)
        nn_pkl = self.model_dir / "nn_neighbors.pkl"
        if nn_pkl.exists():
            with open(nn_pkl, "rb") as f:
                self.nn_model = pickle.load(f)
        meta_path = self.model_dir / "model_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.mode = meta.get("mode", "char")

    @staticmethod
    def _normalize_title(title):
        return re.sub(r"[\W_]+", "", str(title).casefold())

    def _rebuild_lookup_indexes(self):
        """让模型矩阵行号成为唯一查找入口，避免推荐循环反复扫描 DataFrame。"""
        self.id_to_idx = {
            int(book_id): idx for idx, book_id in enumerate(self.df["id"])
        }
        self.idx_to_id = {idx: book_id for book_id, idx in self.id_to_idx.items()}
        self._normalized_titles = np.asarray(
            [self._normalize_title(title) for title in self.df["title"]],
            dtype=object,
        )

    @staticmethod
    def _top_indices(scores, candidate_indices, count):
        """从候选集合中取 Top-K，仅排序最终候选而非完整图书库。"""
        if count <= 0 or len(candidate_indices) == 0:
            return np.asarray([], dtype=np.intp)
        count = min(count, len(candidate_indices))
        candidate_scores = scores[candidate_indices]
        if count == len(candidate_indices):
            local_indices = np.argsort(candidate_scores)[::-1]
        else:
            local_indices = np.argpartition(candidate_scores, -count)[-count:]
            local_indices = local_indices[
                np.argsort(candidate_scores[local_indices])[::-1]
            ]
        return candidate_indices[local_indices]

    def recommend_by_id(self, book_id, top_n=10):
        if self.nn_model is None:
            self._load_artifacts()
        book_id = int(book_id)
        if book_id not in self.id_to_idx:
            return pd.DataFrame()
        idx = self.id_to_idx[book_id]
        vec = self.tfidf_matrix[idx]
        search_range = min(top_n * 8, len(self.df) - 1)
        distances, indices = self.nn_model.kneighbors(vec, n_neighbors=search_range + 1)
        query_title = self.df.iloc[idx]["title"]
        seen_titles = {query_title}
        results = []
        for dist, nidx in zip(distances[0][1:], indices[0][1:]):
            rec_id = self.idx_to_id[nidx]
            row = self.df.iloc[nidx]
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            results.append({
                "id": rec_id, "title": title, "rating": row["rating"],
                "votes": row["votes"],
                "bayesian_score": row.get("bayesian_score", 0),
                "similarity": round(1.0 - float(dist), 4),
            })
            if len(results) >= top_n:
                break
        return pd.DataFrame(results)

    def recommend_by_title(self, query, top_n=10, allowed_ids=None):
        """按语义搜索图书，可选用真实标签对应的图书 ID 集合作为候选范围。"""
        if self.vectorizer is None:
            self._load_artifacts()
        if self.mode == "semantic":
            tokens = self._tokenize_semantic(query)
        else:
            tokens = self._tokenize_char(query)
        if not tokens:
            return pd.DataFrame()
        query_vec = self.vectorizer.transform([tokens])
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        if allowed_ids is None:
            candidate_indices = np.arange(len(sims), dtype=np.intp)
        else:
            candidate_indices = np.fromiter(
                (
                    self.id_to_idx[int(book_id)]
                    for book_id in allowed_ids
                    if int(book_id) in self.id_to_idx
                ),
                dtype=np.intp,
            )
        if len(candidate_indices) == 0:
            return pd.DataFrame()

        search_range = min(max(top_n * 8, top_n), len(candidate_indices))
        top_indices = self._top_indices(sims, candidate_indices, search_range)

        # 书名完全匹配属于用户最明确的意图，应优先于模糊语义结果。
        if self._normalized_titles is None:
            self._rebuild_lookup_indexes()
        normalized_query = self._normalize_title(query)
        exact_indices = candidate_indices[
            self._normalized_titles[candidate_indices] == normalized_query
        ]
        if len(exact_indices):
            exact_indices = exact_indices[np.argsort(sims[exact_indices])[::-1]]
            top_indices = np.concatenate(
                [exact_indices, top_indices[~np.isin(top_indices, exact_indices)]]
            )

        seen_titles = set()
        results = []
        for nidx in top_indices:
            rec_id = self.idx_to_id[nidx]
            row = self.df.iloc[nidx]
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            results.append({
                "id": rec_id, "title": title, "rating": row["rating"],
                "votes": row["votes"],
                "bayesian_score": row.get("bayesian_score", 0),
                "similarity": round(float(sims[nidx]), 4),
            })
            if len(results) >= top_n:
                break
        return pd.DataFrame(results)

    def hybrid_recommend(self, book_id, top_n=10, alpha=0.5):
        content_recs = self.recommend_by_id(book_id, top_n=max(top_n * 5, 30))
        if content_recs.empty:
            return pd.DataFrame()
        sim_max = content_recs["similarity"].max()
        bs_max = content_recs["bayesian_score"].max()
        bs_min = content_recs["bayesian_score"].min()
        content_recs["bs_norm"] = 0.5
        if bs_max > bs_min:
            content_recs["bs_norm"] = (content_recs["bayesian_score"] - bs_min) / (bs_max - bs_min)
        content_recs["sim_norm"] = content_recs["similarity"] / sim_max if sim_max > 0 else 1
        content_recs["hybrid_score"] = alpha * content_recs["sim_norm"] + (1 - alpha) * content_recs["bs_norm"]
        result = content_recs.nlargest(top_n, "hybrid_score")
        result = result.drop(columns=["sim_norm", "bs_norm"], errors="ignore")
        return result

    def get_popular_recommendations(self, top_n=20, min_votes=100):
        candidates = self.df[self.df["votes"] >= min_votes]
        top = candidates.nlargest(top_n, "bayesian_score")
        return top[["id", "title", "rating", "votes", "bayesian_score"]]


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "semantic"
    rec = BookRecommender(mode=mode)
    rec.load_data()
    rec.build_tfidf(max_features=None)
    rec.build_nn_index(n_neighbors=30)

    print("\n--- recommend_by_title('科幻') ---")
    for _, row in rec.recommend_by_title("科幻", top_n=10).iterrows():
        print(f"  {row['title'][:30]:<32s} sim={row['similarity']:.4f}")

    print("\n--- recommend_by_title('推理小说') ---")
    for _, row in rec.recommend_by_title("推理小说", top_n=10).iterrows():
        print(f"  {row['title'][:30]:<32s} sim={row['similarity']:.4f}")

    print("\n--- recommend_by_id(三体=3259440) ---")
    for _, row in rec.recommend_by_id(3259440, top_n=10).iterrows():
        print(f"  {row['title'][:30]:<32s} sim={row['similarity']:.4f}")

    print("\n[Done]")
