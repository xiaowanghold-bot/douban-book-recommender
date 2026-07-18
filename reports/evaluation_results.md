# 豆瓣图书推荐系统 — 离线评估报告

**生成时间**: 2026-07-18 17:58:00
**数据规模**: books_for_rec.csv ≈ 164K 行, Books_detail.csv = 6,584 行

---

## 实验 A: 同系列 Recall@K

> ⚠ **方法局限性说明**: 本项目推荐引擎基于字符级 n-gram TF-IDF + 余弦相似度，
> 同系列书名（如《三体》/《三体II》）天然字符重叠度高，Recall 指标会高估实际语义推荐能力。

- 查询书数: 3355
- 系列数: 744
- 系列平均规模: 4.5 本

| 方法 | Recall@10 | Recall@20 | 有效查询数 |
|------|-----------|-----------|-----------|
| recommend_by_id | 0.1609 | 0.1764 | 3355 |
| Random | 0.0001 | 0.0001 | 3355 |
| Popular | 0.0000 | 0.0009 | 3355 |

## 实验 B: 评分预测正规评估

- 训练集: 4,992 条, 测试集: 1,248 条
- 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)
- 特征: price, year, pages, votes_log, author_clean, publisher_clean, binding_type

| 方法 | RMSE | MAE |
|------|------|-----|
| 全局均值 | 0.7115 | 0.5723 |
| 出版社均值 | 0.6588 | 0.5262 |
| 作者均值 | 0.5987 | 0.4553 |
| RandomForest | 0.6154 | 0.4890 |

## 实验 C: 冷启动模型泄露检查

### 现状分析

`coldstart_predictor.py` 的 `build_stats()` 在**全量 `self.df`** 上计算 stats，
然后 `build_features()` 逐行映射。这意味着 `pub_avg_rating` 和 `author_avg_rating`
都**包含目标书本身的评分** → 标签泄露。

`votes_log` 特征对真实新书不可用（votes≈0），属于特征泄露。

### 对比实验

- 训练集: 5,260 条, 测试集: 1,315 条
- 模型: RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)

| 版本 | RMSE | R² |
|------|------|----|
| Baseline (作者均值) | 0.5580 | 0.3788 |
| v1: 11特征(含泄露) | 0.3091 | 0.8094 |
| v2: 10特征(去votes_log) | 0.3147 | 0.8024 |
| v3: 10特征(去votes+LOO统计) | 0.5104 | 0.4803 |

### 结论

- v3（严谨版）R²=0.4803，对比作者均值 baseline R²=0.3788
- 差距 = +0.1015
- ✅ 模型优于作者均值 baseline。

---

_总耗时: 108.2s_
_random_state=42 用于所有随机过程_