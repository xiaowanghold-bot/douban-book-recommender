"""
数据清洗与预处理模块
豆瓣读书数据集的数据清洗、转换与特征工程
"""
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path


class BookDataCleaner:
    """图书数据清洗器"""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.raw_dir = Path(data_dir) / "raw"
        self.processed_dir = Path(data_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.df_detail = None
    
    # ========== 1. 加载数据 ==========
    
    def load_basic_data(self, filename="Books_1.csv"):
        """加载基础评分数据"""
        path = self.raw_dir / filename
        self.df = pd.read_csv(path, encoding="utf-8-sig")
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        print(f"[加载] {filename}: {len(self.df):,} 条记录")
        return self
    
    def load_detail_data(self, filename="Books_detail.csv"):
        """加载爬虫详细数据（如果存在）"""
        path = self.raw_dir / filename
        if path.exists():
            self.df_detail = pd.read_csv(path, encoding="utf-8-sig")
            self.df_detail.columns = [c.strip().lower() for c in self.df_detail.columns]
            print(f"[加载] {filename}: {len(self.df_detail):,} 条记录")
        else:
            print(f"[提示] {filename} 尚未生成（爬虫运行中）")
        return self
    
    # ========== 2. 基础清洗 ==========
    
    def clean_basic(self):
        """清洗基础评分数据"""
        print("\n--- 基础数据清洗 ---")
        n0 = len(self.df)
        
        # 去除评分/评价数为0的记录
        before = len(self.df)
        self.df = self.df[(self.df["rating"] > 0) & (self.df["votes"] > 0)].copy()
        print(f"  去除无评分记录: {before - len(self.df):,} 条")
        
        # 去除缺失书名的记录
        before = len(self.df)
        self.df = self.df[self.df["title"].notna() & (self.df["title"] != "")].copy()
        print(f"  去除无书名记录: {before - len(self.df):,} 条")
        
        # 评分范围检查 (豆瓣评分 2-10)
        self.df = self.df[(self.df["rating"] >= 2) & (self.df["rating"] <= 10)].copy()
        
        # 去重 (同一ID)
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=["id"], keep="first").copy()
        print(f"  去除重复ID: {before - len(self.df):,} 条")
        
        # 数据类型转换
        self.df["id"] = self.df["id"].astype(int)
        self.df["rating"] = self.df["rating"].astype(float)
        self.df["votes"] = self.df["votes"].astype(int)
        
        print(f"  清洗后: {len(self.df):,} 条 (原始: {n0:,})")
        return self
    
    # ========== 3. 详细数据清洗 ==========
    
    def clean_detail(self):
        """清洗爬虫详细数据"""
        if self.df_detail is None:
            print("[跳过] 无详细数据")
            return self
        
        print("\n--- 详细数据清洗 ---")
        detail = self.df_detail.copy()
        n0 = len(detail)
        
        # 只保留爬取成功的
        detail = detail[detail["crawl_status"] == "success"].copy()
        print(f"  爬取成功: {len(detail)}/{n0}")
        
        # 清洗价格: "28.00元" → 28.0
        detail["price_value"] = detail["price"].apply(self._parse_price)
        print(f"  价格解析成功: {detail['price_value'].notna().sum()}/{len(detail)}")
        
        # 清洗页数
        detail["pages_value"] = detail["pages"].apply(self._parse_pages)
        print(f"  页数解析成功: {detail['pages_value'].notna().sum()}/{len(detail)}")
        
        # 清洗出版年
        detail["pub_year_value"] = detail["pub_year"].apply(self._parse_year)
        print(f"  出版年解析成功: {detail['pub_year_value'].notna().sum()}/{len(detail)}")
        
        # 提取作者国籍
        detail["author_nationality"] = detail["author"].apply(self._extract_nationality)
        
        # ISBN清理
        detail["isbn_clean"] = detail["isbn"].apply(self._clean_isbn)
        
        # 装帧分类
        detail["binding_type"] = detail["binding"].apply(self._classify_binding)
        
        # 保留有效记录
        self.df_detail = detail[detail["price_value"].notna() | 
                                 detail["pages_value"].notna() | 
                                 detail["pub_year_value"].notna()].copy()
        
        print(f"  清洗后有效记录: {len(self.df_detail):,}")
        return self
    
    # ========== 4. 特征工程 ==========
    
    def create_features(self):
        """创建衍生特征"""
        print("\n--- 特征工程 ---")
        
        # 对数评价人数 (处理长尾分布)
        self.df["log_votes"] = np.log1p(self.df["votes"])
        
        # 评分-评价人数综合值
        self.df["score_popularity"] = self.df["rating"] * np.log1p(self.df["votes"])
        
        # 评价人数分桶
        self.df["votes_tier"] = pd.cut(
            self.df["votes"],
            bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, float("inf")],
            labels=["<10", "10-50", "50-100", "100-500", "500-1K", "1K-5K", "5K-10K", ">10K"]
        )
        
        # 评分等级
        self.df["rating_tier"] = pd.cut(
            self.df["rating"],
            bins=[0, 6, 7, 8, 9, 10],
            labels=["<6", "6-7", "7-8", "8-9", "9-10"]
        )
        
        print(f"  衍生特征: log_votes, score_popularity, votes_tier, rating_tier")
        
        # 如果有详细数据，做更多特征
        if self.df_detail is not None and len(self.df_detail) > 0:
            # 出版年代
            self.df_detail["decade"] = (self.df_detail["pub_year_value"] // 10 * 10).astype("Int64")
            # 价格区间
            self.df_detail["price_tier"] = pd.cut(
                self.df_detail["price_value"],
                bins=[0, 20, 30, 40, 50, 80, 200, float("inf")],
                labels=["<20", "20-30", "30-40", "40-50", "50-80", "80-200", ">200"]
            )
            print(f"  详细特征: decade, price_tier")
        
        return self
    
    # ========== 5. 合并数据集 ==========
    
    def merge_datasets(self):
        """合并基础数据和详细数据"""
        if self.df_detail is None or len(self.df_detail) == 0:
            print("[跳过] 无详细数据可合并")
            return self
        
        print("\n--- 数据合并 ---")
        before = len(self.df)
        
        # 左连接：保留所有基础数据，补充详细信息
        detail_cols = ["id", "author", "publisher", "pub_year", "pub_year_value",
                       "pages_value", "price_value", "binding", "binding_type",
                       "isbn_clean", "original_title", "translator", "subtitle",
                       "series", "author_nationality", "decade", "price_tier"]
        detail_cols = [c for c in detail_cols if c in self.df_detail.columns]
        
        self.df = self.df.merge(
            self.df_detail[detail_cols],
            on="id",
            how="left"
        )
        
        merged = self.df["author"].notna().sum()
        print(f"  合并后: {len(self.df):,} 条 (含详细信息: {merged:,})")
        return self
    
    # ========== 6. 保存 ==========
    
    def save(self, filename="books_cleaned.csv"):
        """保存清洗后的数据"""
        path = self.processed_dir / filename
        self.df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[保存] {path} ({len(self.df):,} 条)")
        
        # 同时保存数据摘要
        summary = self._generate_summary()
        summary_path = self.processed_dir / "data_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[保存] {summary_path}")
        return self
    
    def _generate_summary(self):
        """生成数据摘要报告"""
        df = self.df
        lines = []
        lines.append("=" * 50)
        lines.append("  豆瓣图书数据清洗报告")
        lines.append("=" * 50)
        lines.append(f"  总记录数: {len(df):,}")
        lines.append(f"  评分范围: {df['rating'].min():.1f} ~ {df['rating'].max():.1f}")
        lines.append(f"  评分均值: {df['rating'].mean():.2f}")
        lines.append(f"  评分中位数: {df['rating'].median():.1f}")
        lines.append(f"  评价人数均值: {df['votes'].mean():.0f}")
        lines.append(f"  评价人数中位数: {df['votes'].median():.0f}")
        lines.append(f"  评价人数最大值: {df['votes'].max():,}")
        lines.append("")
        
        if "author" in df.columns:
            lines.append(f"  有作者信息: {df['author'].notna().sum():,}")
            lines.append(f"  有出版社信息: {df['publisher'].notna().sum():,}")
            lines.append(f"  有价格信息: {df['price_value'].notna().sum():,}")
            lines.append(f"  有页数信息: {df['pages_value'].notna().sum():,}")
            lines.append(f"  有ISBN信息: {df['isbn_clean'].notna().sum():,}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    # ========== 辅助解析函数 ==========
    
    @staticmethod
    def _parse_price(val):
        """解析价格: '28.00元' → 28.0"""
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r'(\d+\.?\d*)', str(val))
        return float(match.group(1)) if match else np.nan
    
    @staticmethod
    def _parse_pages(val):
        """解析页数: '191' → 191"""
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r'(\d+)', str(val))
        return int(match.group(1)) if match else np.nan
    
    @staticmethod
    def _parse_year(val):
        """解析出版年: '2012-8' → 2012"""
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r'(\d{4})', str(val))
        return int(match.group(1)) if match else np.nan
    
    @staticmethod
    def _extract_nationality(author):
        """提取作者国籍: '[日] 东野圭吾' → '日本'"""
        if pd.isna(author) or str(author).strip() == "":
            return "未知"
        text = str(author).strip()
        nation_map = {
            "日": "日本", "美": "美国", "英": "英国", "法": "法国",
            "德": "德国", "意": "意大利", "俄": "俄罗斯", "中": "中国",
            "韩": "韩国", "西": "西班牙", "荷": "荷兰", "巴": "巴西",
            "奥": "奥地利", "加": "加拿大", "澳": "澳大利亚",
            "哥伦比亚": "哥伦比亚", "阿根廷": "阿根廷",
            "挪威": "挪威", "瑞典": "瑞典", "丹麦": "丹麦",
            "爱尔兰": "爱尔兰", "比利时": "比利时",
        }
        match = re.search(r'\[(.+?)\]', text)
        if match:
            abbrev = match.group(1).strip()
            return nation_map.get(abbrev, abbrev)
        return "中国"  # 没有标注国籍默认为中国
    
    @staticmethod
    def _clean_isbn(val):
        """清理ISBN"""
        if pd.isna(val) or str(val).strip() == "":
            return ""
        return re.sub(r'[^\dXx]', '', str(val))
    
    @staticmethod
    def _classify_binding(val):
        """装帧分类"""
        if pd.isna(val) or str(val).strip() == "":
            return "未知"
        text = str(val).strip()
        if "精装" in text:
            return "精装"
        elif "平装" in text:
            return "平装"
        else:
            return text
    
    # ========== 完整流水线 ==========
    
    def run_pipeline(self, use_detail=True):
        """运行完整清洗流水线"""
        print("=" * 50)
        print("  开始数据清洗流水线")
        print("=" * 50)
        
        self.load_basic_data()
        self.clean_basic()
        self.create_features()
        
        if use_detail:
            self.load_detail_data()
            self.clean_detail()
            self.merge_datasets()
        
        self.save()
        print("\n[OK] 数据清洗完成!")
        return self


if __name__ == "__main__":
    cleaner = BookDataCleaner()
    cleaner.run_pipeline()

