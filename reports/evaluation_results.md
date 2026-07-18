# 豆瓣图书推荐系统 — 离线评估报告

**生成时间**: 2026-07-18 20:20:14
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

### 修复前 (LabelEncoder 编码, 任意整数)

- 训练集: 4,992 条, 测试集: 1,248 条
- 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)
- 特征: price, year, pages, votes_log, author_clean, publisher_clean, binding_type (LabelEncoder)

| 方法 | RMSE | MAE |
|------|------|-----|
| 全局均值 | 0.7115 | 0.5723 |
| 出版社均值 | 0.6588 | 0.5262 |
| 作者均值 | 0.5987 | 0.4553 |
| RandomForest (修复前) | 0.6154 | 0.4890 |

> ⚠ RF 输给作者均值 baseline

### 修复后 (train-only 统计均值)

- 训练集: 4,992 条, 测试集: 1,248 条
- 模型: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42)
- 特征: price, year, pages, votes_log, author_mean, publisher_mean, binding_mean (train-only 统计均值)

| 方法 | RMSE | MAE |
|------|------|-----|
| 全局均值 | 0.7115 | 0.5723 |
| 出版社均值 | 0.6588 | 0.5262 |
| 作者均值 | 0.5987 | 0.4553 |
| RandomForest (修复后) | 0.5457 | 0.4120 |

> ✅ RF 反超！RMSE 0.615 → 0.546, MAE 0.489 → 0.412

## 实验 C: 冷启动模型泄露检查

### 现状分析

`coldstart_predictor.py` 的 `build_stats()` 在**全量 `self.df`** 上计算 stats，
然后 `build_features()` 逐行映射。这意味着 `pub_avg_rating` 和 `author_avg_rating`
都**包含目标书本身的评分** → 标签泄露。

`votes_log` 特征对真实新书不可用（votes≈0），属于特征泄露。

### 对比实验

- 训练集: 5,260 条, 测试集: 1,315 条
- 模型: GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42) — 与生产冷启动模型同配置

| 版本 | RMSE | R² |
|------|------|----|
| Baseline (作者均值) | 0.5580 | 0.3788 |
| v1: 11特征(含泄露) | 0.3117 | 0.8062 |
| v2: 10特征(去votes_log) | 0.3132 | 0.8044 |
| v3: 10特征(去votes+LOO统计) | 0.5023 | 0.4967 |

### 结论

- v3（严谨版）R²=0.4967，对比作者均值 baseline R²=0.3788
- 差距 = +0.1179
- ✅ 模型优于作者均值 baseline。

## 实验 E: 真实用户 Leave-One-Out 评估 (IJCAI 数据集)

> **数据来源**: DTCDR (CIKM 2019) / GA-DTCDR (IJCAI 2020) 跨域推荐公开数据集
> 引用: Zhu et al., CIKM 2019; Zhu et al., IJCAI 2020

### 方法学

- **用户筛选**: 总评分 >=10 条且 rating>=4 的高分书中至少 5 本在推荐索引内
  (最终 1595 人, 平均 102 条评分, 平均 72 本高分在库)
- **目标书选取**: 取该用户 rating>=4 的书, 按时间排序, 留出最后一本作为 ground-truth
- **候选集构造**: 对每本历史高分种子书, 各取该书的 top-(K*3) NN 邻居,
  对每个候选书取跨种子的最大 rank-weighted 分数 (1/rank) 做 max-pooling,
  然后取全局 top-K 作为最终推荐列表. 指标为标准的 **Recall@K** (K 本推荐中是否命中目标).
- **Random 基线**: 从全库随机抽 K 本 (排除该用户所有已评分书), 5 次平均
- **Popular 基线**: 按 votes 降序取前 K 本 (排除该用户所有已评分书)
  (注: Popular 基线为 0 是正常的——Top20 永远是《活着》《红楼梦》等国民级畅销书, 1,595 名用户的个性化目标书几乎不可能命中; 改用 bayesian_score 则 top 列表被小众高分书占据, 更无意义)

| 方法 | Recall@10 | Recall@20 | 用户数 |
|------|-----------|-----------|--------|
| recommend_by_id | 0.0038 | 0.0094 | 1595 |
| Random | 0.0000 | 0.0000 | 1595 |
| Popular | 0.0000 | 0.0000 | 1595 |

---

_总耗时: 151.0s_
_random_state=42 用于所有随机过程_