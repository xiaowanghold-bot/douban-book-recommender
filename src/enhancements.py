"""
项目增强功能模块
- 图书词云
- 价格分析
- 评分预测模型
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  功能1: 图书词云
# ============================================================

def generate_wordcloud():
    """生成图书书名词云"""
    print("\n[词云] 生成图书书名词云...")
    try:
        from wordcloud import WordCloud
        import jieba
    except ImportError:
        print("  wordcloud/jieba not installed, skipping")
        return

    # 加载清洗后的数据
    df = pd.read_csv(DATA_DIR / "processed" / "books_scored.csv", encoding="utf-8-sig")

    # 取高评分图书的书名
    high_rated = df[df["rating"] >= 8.5]
    texts = " ".join(high_rated["title"].dropna().astype(str).tolist())

    # jieba 分词
    words = jieba.lcut(texts)
    # 过滤：长度>=2，包含中文
    filtered = [w for w in words if len(w) >= 2 and any("\u4e00" <= c <= "\u9fff" for c in w)]

    # 统计词频
    from collections import Counter
    word_freq = Counter(filtered)

    # 停用词
    stopwords = {"本书", "全集", "系列", "新版", "修订", "插图", "注释", "全本",
                 "珍藏", "经典", "英文", "中文", "原版", "彩图", "套装", "第一",
                 "第二", "第三", "第四", "第五", "上下", "上册", "下册", "全册",
                 "完整", "纪念", "特别", "全新", "精选"}
    for sw in stopwords:
        word_freq.pop(sw, None)

    # 生成词云
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        font_path="C:/Windows/Fonts/msyh.ttc",
        max_words=150,
        max_font_size=120,
        min_font_size=14,
        colormap="viridis",
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("高分图书书名词云 (评分 >= 8.5)", fontsize=16, pad=20)
    fig.savefig(FIG_DIR / "13_wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表13] 书名词云")


# ============================================================
#  功能2: 价格分析
# ============================================================

class PriceAnalyzer:

    def __init__(self):
        self.df = None

    def load(self):
        """加载详细数据并解析价格"""
        df = pd.read_csv(DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
        df = df[df["crawl_status"] == "success"].copy()
        df["price_num"] = df["price"].apply(self._parse)
        self.df = df[df["price_num"].notna() & (df["price_num"] > 0) & (df["price_num"] < 1000)]
        print(f"[价格] 有效价格记录: {len(self.df):,}")
        return self

    @staticmethod
    def _parse(text):
        if pd.isna(text): return np.nan
        m = re.search(r"[\d.]+", str(text))
        return float(m.group()) if m else np.nan

    def analyze(self):
        """价格分布与评分关系分析"""
        print("\n[价格分析] 生成图表...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 价格分布
        ax = axes[0, 0]
        prices = self.df["price_num"]
        ax.hist(prices[prices < 200], bins=50, color="#3498DB", alpha=0.8, edgecolor="white")
        ax.axvline(prices.median(), color="red", linestyle="--", label=f"中位数 {prices.median():.1f}元")
        ax.axvline(prices.mean(), color="orange", linestyle="--", label=f"均值 {prices.mean():.1f}元")
        ax.set_xlabel("价格 (元)")
        ax.set_ylabel("图书数量")
        ax.set_title("图书价格分布")
        ax.legend()

        # 图2: 价格 vs 评分
        ax = axes[0, 1]
        sample = self.df.sample(min(3000, len(self.df)))
        ax.scatter(sample["price_num"], sample["Rating"], alpha=0.3, s=10, c="#2ECC71")
        ax.set_xlabel("价格 (元)")
        ax.set_ylabel("豆瓣评分")
        ax.set_title("价格与评分关系")
        ax.grid(True, alpha=0.3)

        # 图3: 价格区间统计
        ax = axes[1, 0]
        bins = [0, 20, 30, 40, 50, 70, 100, 200, 1000]
        labels = ["<20", "20-30", "30-40", "40-50", "50-70", "70-100", "100-200", ">200"]
        self.df["price_tier"] = pd.cut(self.df["price_num"], bins=bins, labels=labels)
        tier_counts = self.df["price_tier"].value_counts().reindex(labels)
        tier_ratings = self.df.groupby("price_tier")["Rating"].mean().reindex(labels)
        ax2 = ax.twinx()
        ax.bar(range(len(labels)), tier_counts.values, color="#9B59B6", alpha=0.7)
        ax2.plot(range(len(labels)), tier_ratings.values, "o-", color="#E74C3C", linewidth=2, markersize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("价格区间 (元)")
        ax.set_ylabel("图书数量", color="#9B59B6")
        ax2.set_ylabel("平均评分", color="#E74C3C")
        ax.set_title("各价格区间图书数量与平均评分")

        # 图4: 高性价比图书 (评分>=9, 价格<=50)
        ax = axes[1, 1]
        value_books = self.df[(self.df["Rating"] >= 9) & (self.df["price_num"] <= 50)]
        ax.scatter(value_books["price_num"], value_books["Rating"],
                   alpha=0.5, s=value_books["Votes"] / 100, c="#F39C12", edgecolors="black", linewidth=0.3)
        ax.set_xlabel("价格 (元)")
        ax.set_ylabel("评分")
        ax.set_title(f"高性价比图书 (评分>=9, <=50元): {len(value_books)} 本")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "14_price_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [图表14] 价格分析")

        # 保存
        self.df.to_csv(DATA_DIR / "processed" / "books_with_price.csv",
                       index=False, encoding="utf-8-sig")
        return self

    def get_best_value(self, max_price=50, min_rating=9.0, top_n=20):
        """获取高性价比图书"""
        books = self.df[(self.df["Rating"] >= min_rating) &
                        (self.df["price_num"] <= max_price)]
        return books.nlargest(top_n, "Votes")[
            ["Title", "Rating", "Votes", "price_num", "author", "publisher"]
        ]


# ============================================================
#  功能3: 评分预测模型
# ============================================================

class RatingPredictor:

    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_names = None
        self.df = None
        self.metrics = {}

    def load_and_prepare(self):
        """加载数据并做特征工程"""
        print("\n[评分预测] 加载数据...")
        df = pd.read_csv(DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
        df = df[df["crawl_status"] == "success"].copy()

        # 解析数字字段
        df["price_num"] = df["price"].apply(PriceAnalyzer._parse)
        df["year_num"] = df["pub_year"].apply(
            lambda x: int(re.search(r"(19|20)\d{2}", str(x)).group())
            if pd.notna(x) and re.search(r"(19|20)\d{2}", str(x)) else np.nan)
        df["pages_num"] = df["pages"].apply(
            lambda x: int(re.search(r"\d+", str(x)).group())
            if pd.notna(x) and re.search(r"\d+", str(x)) else np.nan)

        # 清洗分类字段
        df["author_clean"] = df["author"].apply(
            lambda x: re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(x)).strip()[:30]
            if pd.notna(x) else "未知")
        df["publisher_clean"] = df["publisher"].fillna("未知").astype(str).str[:20]
        df["binding_type"] = df["binding"].fillna("未知").apply(
            lambda x: "平装" if "平装" in str(x) else ("精装" if "精装" in str(x) else "其他"))

        # 选择特征和目标
        df = df.dropna(subset=["Rating", "price_num", "year_num", "pages_num"]).copy()
        df = df[df["year_num"].between(1950, 2025)]

        self.df = df
        print(f"  训练数据: {len(df):,} 条")
        print(f"  年份范围: {df['year_num'].min():.0f}-{df['year_num'].max():.0f}")
        print(f"  价格范围: {df['price_num'].min():.1f}-{df['price_num'].max():.1f}元")
        return self

    def train(self):
        """训练评分预测模型"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error, r2_score

        print("\n[评分预测] 训练模型...")

        df = self.df
        # 特征
        features = pd.DataFrame({
            "price": df["price_num"],
            "year": df["year_num"],
            "pages": df["pages_num"].fillna(df["pages_num"].median()),
            "votes_log": np.log1p(df["Votes"]),
        })

        # 编码分类特征
        for col in ["author_clean", "publisher_clean", "binding_type"]:
            le = LabelEncoder()
            # 低频类别归为"其他"
            counts = df[col].value_counts()
            df[f"{col}_enc"] = df[col].apply(
                lambda x: x if counts.get(x, 0) >= 3 else "其他")
            features[col] = le.fit_transform(df[f"{col}_enc"])
            self.encoders[col] = le

        self.feature_names = list(features.columns)
        X = features.values
        y = df["Rating"].values

        # 训练
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        self.model.fit(X, y)

        # 评估
        y_pred = self.model.predict(X)
        self.metrics["MAE"] = mean_absolute_error(y, y_pred)
        self.metrics["R2"] = r2_score(y, y_pred)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")
        self.metrics["CV_R2"] = cv_scores.mean()

        print(f"  MAE: {self.metrics['MAE']:.3f} (平均误差)")
        print(f"  R2:  {self.metrics['R2']:.3f}")
        print(f"  CV5: {self.metrics['CV_R2']:.3f} (5折交叉验证)")

        # 特征重要性
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("  特征重要性:")
        for i in indices:
            print(f"    {self.feature_names[i]:<20s}: {importances[i]:.4f}")

        self._plot_feature_importance(importances, indices)
        self._plot_predictions(y, y_pred)

        # 保存模型
        model_dir = DATA_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "rating_predictor.pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "encoders": self.encoders,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
            }, f)
        print(f"  [保存] rating_predictor.pkl")

        return self

    def _plot_feature_importance(self, importances, indices):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(indices)), importances[indices], color="#3498DB")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel("重要性")
        ax.set_title("评分预测特征重要性")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "15_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [图表15] 特征重要性")

    def _plot_predictions(self, y_true, y_pred):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_true, y_pred, alpha=0.2, s=5, c="#2ECC71")
        axes[0].plot([2, 10], [2, 10], "r--", linewidth=1)
        axes[0].set_xlabel("真实评分")
        axes[0].set_ylabel("预测评分")
        axes[0].set_title(f"预测 vs 真实 (MAE={self.metrics['MAE']:.2f})")
        axes[0].grid(True, alpha=0.3)

        errors = y_true - y_pred
        axes[1].hist(errors, bins=40, color="#9B59B6", alpha=0.7, edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("预测误差 (真实-预测)")
        axes[1].set_ylabel("频数")
        axes[1].set_title("预测误差分布")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "16_prediction_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [图表16] 预测散点图")

    def predict(self, price, year, pages, votes, author="未知", publisher="未知", binding="平装"):
        """单条预测"""
        if self.model is None:
            return None

        features = {
            "price": price,
            "year": year,
            "pages": pages,
            "votes_log": np.log1p(votes),
        }

        # 清理输入
        author_clean = re.sub(r"\[.*?\]|\(.*?\)|（.*?）", "", str(author)).strip()[:30]
        publisher_clean = str(publisher).strip()[:20]
        binding_type = "平装" if "平装" in str(binding) else ("精装" if "精装" in str(binding) else "其他")

        for col, val in [("author_clean", author_clean),
                         ("publisher_clean", publisher_clean),
                         ("binding_type", binding_type)]:
            counts = self.df[col].value_counts()
            val_enc = val if counts.get(val, 0) >= 3 else "其他"
            try:
                features[col] = self.encoders[col].transform([val_enc])[0]
            except ValueError:
                features[col] = 0

        X = np.array([[features[n] for n in self.feature_names]])
        return float(self.model.predict(X)[0])


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  项目增强功能模块")
    print("=" * 60)

    # 功能1: 词云
    generate_wordcloud()

    # 功能2: 价格分析
    pa = PriceAnalyzer()
    pa.load().analyze()
    best = pa.get_best_value(max_price=50, min_rating=9.0)
    print(f"\n[性价比] Top 5 高性价比图书:")
    for _, row in best.head(5).iterrows():
        print(f"  {row['Title'][:25]:<28s} {row['Rating']:.1f}分 {row['price_num']:.1f}元 {row['author'][:15]}")

    # 功能3: 评分预测
    rp = RatingPredictor()
    rp.load_and_prepare()
    rp.train()

    # 测试预测
    test_book = {
        "price": 39.5, "year": 2014, "pages": 300,
        "votes": 50000, "author": "余华", "publisher": "人民文学出版社", "binding": "平装"
    }
    pred = rp.predict(**test_book)
    print(f"\n[预测示例] 余华/人民文学出版社/39.5元/2014年 -> 预测评分: {pred:.2f}" if pred else "")

    print("\n[Done] 三大增强功能完成！")
