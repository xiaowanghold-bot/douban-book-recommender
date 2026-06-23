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

- 豆瓣读书公开数据集 (yuzhounh/Douban-books-2020)
- 288,824 本图书的基础评分数据
- 481 个豆列 + 897 个标签

## 环境

Python 3.12 + Streamlit
