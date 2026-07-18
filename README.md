# 基于豆瓣读书数据的图书评价与推荐系统

江南大学大学生创新训练计划项目

## 项目结构

```
├── data/
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后的数据
├── notebooks/        # Jupyter Notebook 分析
├── src/              # 核心代码
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── scoring.py
│   └── recommendation.py
├── app/              # Streamlit 应用
│   └── main.py
├── crawler/          # 爬虫代码
├── reports/          # 报告与图表
├── tests/            # 测试
└── requirements.txt  # 依赖
```

## 数据来源

- 豆瓣读书公开数据集 (yuzhounh/Douban-books-2020): 288,824 本图书基础评分
- IJCAI 跨域推荐数据集 (DTCDR/GA-DTCDR): 227,251 条书评, 95,872 本书, 19,021 个用户标签
- 481 个豆列 + 897 个标签

### 引用
若使用 IJCAI 数据集部分，请引用:
- Zhu et al., "DTCDR: A Framework for Dual-Target Cross-Domain Recommendation", CIKM 2019
- Zhu et al., "GA-DTCDR: Graph Embeddings for Cross-Domain Recommendation", IJCAI 2020

## 环境

Python 3.12 + Streamlit
