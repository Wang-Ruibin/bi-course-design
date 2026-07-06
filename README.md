# BI-Course-Design

商务智能方法与应用课程设计 — 基于泰迪杯赛题的商务智能分析与应用

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## 📋 项目简介

本项目为商务智能方法与应用课程设计，基于泰迪杯竞赛真实数据集，完成以下两项核心任务：

- **任务三 — 铁路旅客流量预测**：对 440 个旅客列车梯形密度表进行数据清洗与合并，从车次、站点、区间、时间和气象五个维度进行统计分析，构建线性回归、随机森林和 XGBoost 预测模型，预测未来两周客流并优化列车配置。
- **任务四 — 生产线故障自动识别**：对 M101 生产线 63 万余条传感器记录进行数据清洗与特征工程，构建逻辑回归、随机森林和 XGBoost 二分类模型（AUC 达 0.9935），并将最优模型应用于 M201、M202 生产线完成故障预测，同时对 M301 生产线的产量与合格率进行方差分析。

## 📁 项目结构

```
.
├── code/
│   ├── task3/                    # 任务三：铁路客流预测
│   │   ├── task3_load_data.py    # 数据加载与清洗
│   │   ├── task3_load_weather.py # 气象数据加载
│   │   ├── task3_eda.py          # 探索性数据分析与可视化
│   │   ├── task3_model.py        # 预测模型构建（回归/随机森林/XGBoost）
│   │   ├── task3_forecast.py     # 客流预测与列车配置优化
│   │   ├── task3_report.py       # 报告生成
│   │   ├── task3_eda/            # EDA 可视化输出
│   │   └── task3_forecast_viz/   # 预测可视化输出
│   └── task4/                    # 任务四：生产线故障识别
│       ├── task4_load_data.py    # 数据加载与清洗
│       ├── task4_eda.py          # 探索性数据分析与可视化
│       ├── task4_model.py        # 故障分类模型构建
│       ├── task4_predict.py      # 故障预测
│       ├── task4_yield_analysis.py # 产量与合格率方差分析
│       ├── task4_report.py       # 报告生成
│       ├── task4_eda/            # EDA 可视化输出
│       └── task4_yield_viz/      # 产量分析可视化输出
├── 商务智能方法与应用课程设计报告_含附录.docx
├── 商务智能方法与应用课程设计报告_无附录.docx
├── LICENSE
└── README.md
```

## 🚀 快速开始

### 环境依赖

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn xlrd python-docx joblib
```

### 数据准备

原始数据集需从泰迪杯竞赛官网获取，请将数据放置于 `课程设计资料/` 目录下：

```
课程设计资料/
├── 泰迪杯2016年B题_说明及数据/
│   └── 附件1：旅客列车梯形密度表/   # 440个xls文件
│   └── 附件4/                       # 气象数据
└── 泰迪杯2024年A题_说明及数据/
    └── A题-示例数据/
        ├── 附件1/                   # M101.csv 等传感器数据
        ├── 附件2/                   # M201.csv, M202.csv
        └── 附件3/                   # M301.csv, 操作人员信息表.xlsx
```

### 运行

```bash
# 任务三：铁路客流预测
python code/task3/task3_load_data.py      # 数据加载
python code/task3/task3_load_weather.py   # 气象数据加载
python code/task3/task3_eda.py            # 探索性分析
python code/task3/task3_model.py          # 模型训练
python code/task3/task3_forecast.py       # 客流预测

# 任务四：生产线故障识别
python code/task4/task4_load_data.py      # 数据加载
python code/task4/task4_eda.py            # 探索性分析
python code/task4/task4_model.py          # 模型训练
python code/task4/task4_predict.py        # 故障预测
python code/task4/task4_yield_analysis.py # 产量分析
```

## 📊 核心方法

| 任务 | 方法 | 说明 |
|------|------|------|
| 任务三 | 描述性统计 | 客流规律多维度分析 |
| 任务三 | 线性回归 / 随机森林 / XGBoost | 客流预测模型 |
| 任务四 | 逻辑回归 / 随机森林 / XGBoost | 故障二分类模型 |
| 任务四 | 单因素方差分析 (ANOVA) | 产量与合格率影响因素分析 |

## 📝 课程设计报告

- [商务智能方法与应用课程设计报告（含附录）](商务智能方法与应用课程设计报告_含附录.docx)
- [商务智能方法与应用课程设计报告（无附录）](商务智能方法与应用课程设计报告_无附录.docx)

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。

---

> ⭐ 如果这个项目对你有帮助，欢迎点个 Star 支持一下！
