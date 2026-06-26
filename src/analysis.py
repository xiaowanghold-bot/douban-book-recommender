"""
出版社/作者二维评价矩阵分析
基于爬虫详细数据的深度分析模块
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from pathlib import Path

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

class PublisherAuthorAnalyzer:

    def __init__(self):
        self.df_detail = None
        self.df_scored = None
        self.df_merged = None
        self.pub_stats = None
        self.author_stats = None

    def load_and_merge(self):
        """加载并合并详细数据与评分数据"""
        print("[加载] 爬虫详细数据...")
        self.df_detail = pd.read_csv(
            DATA_DIR / "raw" / "Books_detail.csv", encoding="utf-8-sig")
        self.df_detail = self.df_detail[self.df_detail["crawl_status"] == "success"].copy()

        print("[加载] 贝叶斯评分数据...")
        self.df_scored = pd.read_csv(
            DATA_DIR / "processed" / "books_scored.csv", encoding="utf-8-sig")

        # 合并
        self.df_merged = self.df_detail.merge(
            self.df_scored[["id", "bayesian_score", "votes_tier", "rating_tier"]],
            left_on="ID", right_on="id", how="inner"
        )
        print(f"  合并后: {len(self.df_merged):,} 条")

        # 基础清洗
        self._clean_fields()
        return self

    def _clean_fields(self):
        """清洗各字段"""
        # 价格: "29.00元" -> 29.0
        self.df_merged["price_num"] = self.df_merged["price"].apply(self._parse_price)
        # 出版年: "2006-5" -> 2006
        self.df_merged["year_num"] = self.df_merged["pub_year"].apply(self._parse_year)
        # 作者去国家标记 [日] 等
        self.df_merged["author_clean"] = self.df_merged["author"].apply(self._clean_author)
        # 出版社标准化
        self.df_merged["publisher_clean"] = self.df_merged["publisher"].apply(self._clean_publisher)
        # 提取国籍
        self.df_merged["nationality"] = self.df_merged["author"].apply(self._extract_nationality)

    @staticmethod
    def _parse_price(text):
        if pd.isna(text) or not str(text).strip():
            return np.nan
        m = re.search(r"[\d.]+", str(text))
        return float(m.group()) if m else np.nan

    @staticmethod
    def _parse_year(text):
        if pd.isna(text) or not str(text).strip():
            return np.nan
        m = re.search(r"(19|20)\d{2}", str(text))
        return int(m.group()) if m else np.nan

    @staticmethod
    def _clean_author(text):
        if pd.isna(text) or not str(text).strip():
            return "未知"
        text = str(text).strip()
        text = re.sub(r"\[.*?\]", "", text)  # 去国家标记
        text = re.sub(r"\(.*?\)", "", text)  # 去括号
        text = re.sub(r"（.*?）", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else "未知"

    @staticmethod
    def _clean_publisher(text):
        if pd.isna(text) or not str(text).strip():
            return "未知"
        return str(text).strip()

    @staticmethod
    def _extract_nationality(text):
        if pd.isna(text) or not str(text).strip():
            return "其他"
        m = re.search(r"\[(.*?)\]", str(text))
        return m.group(1) if m else "中国"

    # ========== 出版社分析 ==========

    def analyze_publishers(self, min_books=3):
        """出版社综合评价"""
        print(f"\n[出版社分析] 最少图书数: {min_books}")

        pub_group = self.df_merged.groupby("publisher_clean")
        pub_stats = pd.DataFrame({
            "book_count": pub_group.size(),
            "avg_rating": pub_group["Rating"].mean(),
            "avg_bayesian": pub_group["bayesian_score"].mean(),
            "total_votes": pub_group["Votes"].sum(),
            "median_price": pub_group["price_num"].median(),
            "year_range": pub_group["year_num"].agg(lambda x: f"{int(x.min())}-{int(x.max())}" if x.notna().any() else "N/A"),
            "books": pub_group["Title"].apply(list),
        })

        # 过滤
        pub_stats = pub_stats[pub_stats["book_count"] >= min_books].copy()

        # 综合评分
        C = pub_stats["avg_bayesian"].mean()
        m = pub_stats["book_count"].median()
        pub_stats["pub_score"] = (
            (pub_stats["book_count"] / (pub_stats["book_count"] + m)) * pub_stats["avg_bayesian"] +
            (m / (pub_stats["book_count"] + m)) * C
        )

        # 质量等级
        pub_stats["quality_tier"] = pd.cut(
            pub_stats["avg_bayesian"],
            bins=[0, 8.0, 8.5, 9.0, 10],
            labels=["一般", "良好", "优秀", "卓越"],
        )

        self.pub_stats = pub_stats.sort_values("pub_score", ascending=False)
        print(f"  出版社数量: {len(self.pub_stats)}")
        print(f"  Top 5:")
        for i, (name, row) in enumerate(self.pub_stats.head(5).iterrows()):
            print(f"    {i+1}. {name[:20]:<22s} 书{int(row['book_count']):>4d}本 均分{row['avg_rating']:.2f} 综合{row['pub_score']:.4f}")

        return self

    # ========== 作者分析 ==========

    def analyze_authors(self, min_books=2):
        """作者影响力评价"""
        print(f"\n[作者分析] 最少图书数: {min_books}")

        author_group = self.df_merged.groupby("author_clean")
        author_stats = pd.DataFrame({
            "book_count": author_group.size(),
            "avg_rating": author_group["Rating"].mean(),
            "avg_bayesian": author_group["bayesian_score"].mean(),
            "total_votes": author_group["Votes"].sum(),
            "median_price": author_group["price_num"].median(),
            "nationality": author_group["nationality"].first(),
            "books": author_group["Title"].apply(list),
        })

        author_stats = author_stats[author_stats["book_count"] >= min_books].copy()

        # 作者综合评分
        C = author_stats["avg_bayesian"].mean()
        m = author_stats["book_count"].median()
        author_stats["author_score"] = (
            (author_stats["book_count"] / (author_stats["book_count"] + m)) * author_stats["avg_bayesian"] +
            (m / (author_stats["book_count"] + m)) * C
        )

        # 影响力 = 综合评分 * log(总评价人数+1)
        author_stats["influence"] = (
            author_stats["author_score"] * np.log1p(author_stats["total_votes"])
        )

        self.author_stats = author_stats.sort_values("influence", ascending=False)
        print(f"  作者数量: {len(self.author_stats)}")
        print(f"  Top 5:")
        for i, (name, row) in enumerate(self.author_stats.head(5).iterrows()):
            print(f"    {i+1}. {name[:18]:<20s} [{row['nationality']}] 书{int(row['book_count']):>3d}本 均分{row['avg_rating']:.2f} 影响力{row['influence']:.1f}")

        return self

    # ========== 可视化 ==========

    def plot_publisher_matrix(self):
        """出版社二维评价矩阵（数量 × 质量）"""
        if self.pub_stats is None:
            self.analyze_publishers()

        top30 = self.pub_stats.head(30)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 图1: 散点图
        ax = axes[0]
        scatter = ax.scatter(
            top30["book_count"], top30["avg_bayesian"],
            s=top30["total_votes"] / 1000,
            c=top30["pub_score"], cmap="YlOrRd",
            alpha=0.7, edgecolors="black", linewidth=0.5,
        )
        for name, row in top30.iterrows():
            ax.annotate(name[:6], (row["book_count"], row["avg_bayesian"]),
                        fontsize=8, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5))

        ax.set_xlabel("出版图书数量")
        ax.set_ylabel("贝叶斯均分")
        ax.set_title("出版社二维评价矩阵 (气泡=总评价人数)")
        plt.colorbar(scatter, ax=ax, label="综合评分")
        ax.grid(True, alpha=0.3)

        # 图2: Top 15 条形图
        ax = axes[1]
        top15 = self.pub_stats.head(15).iloc[::-1]
        colors = plt.cm.YlOrRd(top15["pub_score"] / top15["pub_score"].max())
        bars = ax.barh(range(len(top15)), top15["pub_score"], color=colors)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels([f"{n[:16]} ({int(c)}本)" for n, c in
                            zip(top15.index, top15["book_count"])], fontsize=9)
        ax.set_xlabel("综合评分")
        ax.set_title("出版社综合评分 Top 15")
        for i, (_, row) in enumerate(top15.iterrows()):
            ax.text(row["pub_score"] + 0.002, i,
                    f"{row['avg_rating']:.1f}分", va="center", fontsize=8)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "10_publisher_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[图表10] 出版社评价矩阵")

    def plot_author_influence(self):
        """作者影响力排行"""
        if self.author_stats is None:
            self.analyze_authors()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 图1: Top 20 作者影响力
        top20 = self.author_stats.head(20).iloc[::-1]
        nation_colors = {"中国": "#E74C3C", "日本": "#3498DB", "美国": "#2ECC71",
                         "英国": "#9B59B6", "法国": "#F39C12"}
        colors = [nation_colors.get(n, "#95A5A6") for n in top20["nationality"]]
        ax = axes[0]
        ax.barh(range(len(top20)), top20["influence"], color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"{n[:14]}" for n in top20.index], fontsize=9)
        ax.set_xlabel("影响力 (综合分 × log评价人数)")
        ax.set_title("作者影响力 Top 20")

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=n) for n, c in nation_colors.items()
                           if n in top20["nationality"].values]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        # 图2: 国籍分布
        ax = axes[1]
        nation_counts = self.author_stats["nationality"].value_counts().head(8)
        ax.pie(nation_counts.values, labels=nation_counts.index, autopct="%1.1f%%",
               colors=[nation_colors.get(n, "#95A5A6") for n in nation_counts.index],
               startangle=90)
        ax.set_title("作者国籍分布")

        plt.tight_layout()
        fig.savefig(FIG_DIR / "11_author_influence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[图表11] 作者影响力分析")

    def plot_year_trend(self):
        """出版年份趋势"""
        df_year = self.df_merged[self.df_merged["year_num"].between(1980, 2025)].copy()
        year_stats = df_year.groupby("year_num").agg(
            book_count=("Title", "count"),
            avg_rating=("Rating", "mean"),
            avg_price=("price_num", "mean"),
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.bar(year_stats.index, year_stats["book_count"], color="#3498DB", alpha=0.8)
        ax.set_xlabel("出版年份")
        ax.set_ylabel("图书数量")
        ax.set_title("高分图书出版年份分布")

        ax = axes[1]
        ax.plot(year_stats.index, year_stats["avg_rating"], "o-",
                color="#E74C3C", linewidth=1.5, markersize=3)
        ax.set_xlabel("出版年份")
        ax.set_ylabel("平均评分")
        ax.set_title("各年份平均评分趋势")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "12_year_trend.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[图表12] 出版年份趋势")

    def run_all(self):
        """一键运行全部分析"""
        print("=" * 60)
        print("  出版社/作者二维评价矩阵分析")
        print("=" * 60)

        self.load_and_merge()
        self.analyze_publishers(min_books=3)
        self.analyze_authors(min_books=2)

        print("\n[可视化] 生成图表...")
        self.plot_publisher_matrix()
        self.plot_author_influence()
        self.plot_year_trend()

        # 保存结果
        self.pub_stats.to_csv(
            DATA_DIR / "processed" / "publisher_stats.csv",
            encoding="utf-8-sig")
        self.author_stats.to_csv(
            DATA_DIR / "processed" / "author_stats.csv",
            encoding="utf-8-sig")
        print(f"\n[保存] publisher_stats.csv + author_stats.csv")

        print("\n[Done] 出版社/作者评价分析完成！")
        return self


if __name__ == "__main__":
    analyzer = PublisherAuthorAnalyzer()
    analyzer.run_all()
