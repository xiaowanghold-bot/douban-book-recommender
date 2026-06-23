# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bayesian_score(votes, rating, C, m):
    return (votes / (votes + m)) * rating + (m / (votes + m)) * C


if __name__ == "__main__":
    print("=" * 55)
    print("  贝叶斯加权评分模型")
    print("=" * 55)
    
    df = pd.read_csv(DATA_DIR / "processed" / "books_cleaned.csv", encoding="utf-8-sig")
    C = df["rating"].mean()
    print(f"[数据] {len(df):,} 本书, 全局均值 C = {C:.4f}")
    
    # === m参数实证扫描 ===
    m_values = [1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
    original_top10 = set(df.nlargest(10, "score_popularity")["id"])
    
    print(f"\n=== m参数实证扫描 ===")
    print(f"  {'m':>5}  {'Top1':<22} {'Top10重叠':>8} {'低评价入Top100':>10}")
    print(f"  {'-'*50}")
    
    scan = {}
    for m in m_values:
        df[f"bs_{m}"] = bayesian_score(df["votes"], df["rating"], C, m)
        bs_top10 = set(df.nlargest(10, f"bs_{m}")["id"])
        overlap = len(bs_top10 & original_top10)
        bs_top100 = df.nlargest(100, f"bs_{m}")
        low_vote = (bs_top100["votes"] < 100).sum()
        top1 = df.nlargest(1, f"bs_{m}")["title"].values[0][:20]
        print(f"  {m:>5}  {top1:<22} {overlap:>8} {low_vote:>10}")
        scan[m] = {"overlap": overlap, "low_vote": low_vote}
    
    # === m参数优化 ===
    print(f"\n=== m参数优化（评价人数中位数法）===")
    high_rated = df[df["rating"] >= 8]
    m_median = int(high_rated["votes"].median())
    p25 = int(high_rated["votes"].quantile(0.25))
    p75 = int(high_rated["votes"].quantile(0.75))
    print(f"  高分图书(>=8分)评价人数分布: P25={p25}, P50={m_median}, P75={p75}")
    print(f"  采用 m = P50 = {m_median}")
    print(f"  含义: 需要 {m_median} 个评价后，原始评分才被充分信任")
    best_m = m_median
    
    df["bayesian_score"] = bayesian_score(df["votes"], df["rating"], C, best_m)
    
    # === 图表8: m参数分析 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m_sorted = sorted(scan.keys())
    axes[0].plot(m_sorted, [scan[m]["overlap"] for m in m_sorted],
                 "o-", color="#4C72B0", linewidth=2, markersize=6)
    axes[0].axvline(best_m, color="red", linestyle="--", label=f"最优 m={best_m}")
    axes[0].set_xlabel("m 参数值")
    axes[0].set_ylabel("Top10 重叠数")
    axes[0].set_title("m 参数 vs Top10 排名稳定性")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(m_sorted, [scan[m]["low_vote"] for m in m_sorted],
                 "s-", color="#55A868", linewidth=2, markersize=6)
    axes[1].axvline(best_m, color="red", linestyle="--", label=f"最优 m={best_m}")
    axes[1].set_xlabel("m 参数值")
    axes[1].set_ylabel("Top100 中低评价图书数\n(Votes < 100)")
    axes[1].set_title("m 参数 vs 小众图书过滤效果")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_m_parameter_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[图表8] m参数分析")
    
    # === 图表9: 贝叶斯Top15对比 ===
    top15 = df.nlargest(15, "bayesian_score")
    top15_raw = df.nlargest(15, "score_popularity")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 15))
    axes[0].barh(range(15), top15_raw["score_popularity"].values, color=colors1[::-1])
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(top15_raw["title"].values, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("原始综合排名")
    axes[0].set_xlabel("评分 * Log(评价人数)")
    
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, 15))
    axes[1].barh(range(15), top15["bayesian_score"].values, color=colors2[::-1])
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(top15["title"].values, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title(f"贝叶斯加权排名 (m={best_m})")
    axes[1].set_xlabel("贝叶斯加权评分")
    for i, (_, r) in enumerate(top15.iterrows()):
        axes[1].text(r["bayesian_score"] + 0.01, i,
                     f"{r['rating']:.1f}分 {r['votes']:,}人",
                     va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_bayesian_top15.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[图表9] 贝叶斯Top15对比")
    
    # === 保存 ===
    cols = ["id","rating","votes","title","bayesian_score",
            "log_votes","score_popularity","votes_tier","rating_tier"]
    df[cols].to_csv(DATA_DIR / "processed" / "books_scored.csv",
                    index=False, encoding="utf-8-sig")
    print(f"[保存] books_scored.csv")
    
    # === Top 20 ===
    top20 = df.nlargest(20, "bayesian_score")
    print(f"\n=== 贝叶斯评分 Top 20 (m={best_m}) ===")
    for i, (_, r) in enumerate(top20.iterrows()):
        t = str(r["title"])[:32]
        print(f"  {i+1:2d}. {t:<34} R={r['rating']:.1f} V={r['votes']:>7,} BS={r['bayesian_score']:.4f}")
    
    # 统计信息
    top100 = df.nlargest(100, "bayesian_score")
    print(f"\n=== 排名统计 ===")
    print(f"  Top100 平均评分: {top100['rating'].mean():.2f}")
    print(f"  Top100 平均评价人数: {top100['votes'].mean():.0f}")
    print(f"  Top100 中评价<100: {(top100['votes']<100).sum()} 本")
    print(f"  Top100 中评价>=1000: {(top100['votes']>=1000).sum()} 本")
    
    print("\n[Done] 贝叶斯加权评分模型完成")
