# -*- coding: utf-8 -*-
"""图书探索性数据分析 (EDA)"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# ===== 中文字体配置 =====
fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
df = pd.read_csv(DATA_DIR / "processed" / "books_cleaned.csv", encoding="utf-8-sig")
print(f"[数据] {len(df):,} 本书")

def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")

# ===== 1. 评分分布 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["rating"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(df["rating"].median(), color="red", linestyle="--", linewidth=1.5,
                label="Median={:.1f}".format(df["rating"].median()))
axes[0].axvline(df["rating"].mean(), color="orange", linestyle="--", linewidth=1.5,
                label="Mean={:.2f}".format(df["rating"].mean()))
axes[0].set_xlabel("Rating"); axes[0].set_ylabel("Count")
axes[0].set_title("Rating Distribution"); axes[0].legend()
axes[1].boxplot(df["rating"].dropna(), patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7))
axes[1].set_ylabel("Rating"); axes[1].set_title("Rating Box Plot")
save(fig, "01_rating_distribution.png")

# ===== 2. 评价人数分布 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["votes"], bins=100, color="#55A868", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Votes"); axes[0].set_ylabel("Count")
axes[0].set_title("Votes Distribution (Linear)")
axes[1].hist(df["log_votes"], bins=50, color="#CCB974", edgecolor="white", alpha=0.85)
axes[1].axvline(df["log_votes"].median(), color="red", linestyle="--",
                label="Median={:.1f}".format(df["log_votes"].median()))
axes[1].set_xlabel("Log(Votes+1)"); axes[1].set_ylabel("Count")
axes[1].set_title("Log Votes Distribution"); axes[1].legend()
save(fig, "02_votes_distribution.png")

# ===== 3. 评分 vs 评价人数 =====
fig, ax = plt.subplots(figsize=(10, 7))
sample = df.sample(min(15000, len(df)), random_state=42)
sc = ax.scatter(sample["votes"], sample["rating"],
                c=sample["log_votes"], cmap="viridis", alpha=0.5, s=8, edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel("Votes (Log Scale)"); ax.set_ylabel("Rating")
ax.set_title("Rating vs Votes")
xbins = np.logspace(np.log10(10), np.log10(df["votes"].max()), 20)
ymeans = [df[(df["votes"] >= xbins[i]) & (df["votes"] < xbins[i+1])]["rating"].mean()
          for i in range(len(xbins)-1)]
xmids = np.sqrt(xbins[:-1] * xbins[1:])
ax.plot(xmids, ymeans, "r-", linewidth=2, label="Trend")
ax.legend(); plt.colorbar(sc, ax=ax, label="Log Votes")
save(fig, "03_rating_vs_votes.png")

# ===== 4. Top 20 图书 =====
top20 = df.nlargest(20, "score_popularity")
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, 20))
ax.barh(range(20), top20["score_popularity"].values, color=colors[::-1])
ax.set_yticks(range(20))
ax.set_yticklabels(top20["title"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Score * Log(Votes)"); ax.set_title("Top 20 Books")
for i, (_, r) in enumerate(top20.iterrows()):
    ax.text(r["score_popularity"] + 1, i,
            "{:.1f} ({:,})".format(r["rating"], int(r["votes"])),
            va="center", fontsize=8)
save(fig, "04_top20_books.png")

# ===== 5. 评分等级饼图 =====
fig, ax = plt.subplots(figsize=(8, 8))
tc = df["rating_tier"].value_counts().sort_index()
ax.pie(tc.values, labels=tc.index, autopct="%1.1f%%",
       colors=["#D62728","#FF7F0E","#2CA02C","#1F77B4","#9467BD"],
       startangle=90, pctdistance=0.85)
ax.set_title("Rating Tier Distribution")
save(fig, "05_rating_tiers.png")

# ===== 6. 评价人数等级 =====
fig, ax = plt.subplots(figsize=(10, 6))
tc2 = df["votes_tier"].value_counts().sort_index()
bars = ax.bar(tc2.index, tc2.values, color="#55A868", edgecolor="white")
ax.set_xlabel("Votes Tier"); ax.set_ylabel("Books"); ax.set_title("Votes Tier Distribution")
for bar, c in zip(bars, tc2.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500, f"{c:,}", ha="center", fontsize=9)
plt.xticks(rotation=45)
save(fig, "06_votes_tiers.png")

# ===== 7. 相关性热力图 =====
fig, ax = plt.subplots(figsize=(7, 6))
corr = df[["rating","votes","log_votes","score_popularity"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1,
            mask=mask, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")
save(fig, "07_correlation_heatmap.png")

print("\n[Done] All figures saved to:", FIG_DIR)

