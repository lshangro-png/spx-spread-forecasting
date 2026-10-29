# SPX 买卖价差的 Regime 感知预测

[English](README.md) | 中文

市场 regime 能否帮助预测标普 500 指数的日度买卖价差？本项目把用于识别 regime 的滚动隐马尔可夫模型（HMM）与线性模型、梯度提升模型结合起来，在 27 年的 Bloomberg 数据上做严格的样本外评估。

**简短回答：** 价差具有中等的可预测性（样本外 R² ≈ 0.36，均方误差比持久性预测低 28%）；以 log 价差为目标、带长记忆（HAR）特征的线性模型效果最好；去除前视偏差后，HMM regime 是有用的风险标志，但没有预测价值。

MATH 628 课程期末项目，作者 Shangrong Li、Siyu Liang、Wenyi Ye。完整报告：[英文](reports/report.md)（[中文翻译](reports/report_zh.md)）。

## 主要结果

| 模型（滚动样本外，5,695 天） | R² | 相对持久性的 skill |
| --- | --- | --- |
| **线性，log 目标** | **0.361** | **0.282** |
| 线性，水平目标 | 0.336 | 0.254 |
| XGBoost | 0.279 | 0.190 |
| LightGBM | 0.275 | 0.186 |
| 持久性（明天 = 今天） | 0.110 | 0 |

- **HAR 长记忆特征**（过去 5/22/63 日的平均 log 价差）是唯一显著提升精度的新增特征：普通交易日的绝对 log 误差从 0.210 降到 0.193（Diebold–Mariano p < 0.001）。
- **HMM regime** 持续性很强（预期持续约 20 天），高价差状态下的预测误差约为低价差状态的 8 倍。但 regime 概率不能提升点预测（消融实验和三种重新设计的 HMM 中 p 均 ≥ 0.33），也不能提升尖峰分类。在预测区间上，它优于常数宽度（p = 0.005），但不如过去残差的简单 EWMA（p < 0.001）。
- **尖峰预测**（价差高于过去一年的 90% 分位数）的 AUC 达到 0.84，但只用一个变量的朴素规则就有 0.83。
- **梯度提升**比线性模型过拟合更严重，也没有胜过线性模型。
- **跨数据口径稳健。** 在三个报价时期内分别检验（见下文），线性模型优于梯度提升、HAR 项有效、HMM 没有增益，这些结论在每个时期都成立。在绝对 log 误差下，全样本的 90% 模型置信集只有线性 + HAR 一个模型。

### 需要注意的数据问题

- Bloomberg 把交易所休市日用前一天的数据原样填充（256 行），这些行已删除。
- 指数 bid/ask 的水平在 2011 年 6 月 1 日跳升约 6 倍、2016 年 12 月 5 日跳升约 4 倍，几乎全部发生在第一天。2016 年 12 月时波动率、收益率和成交量完全没有变化；2011 年 6 月时报价从单边变成了对称。两者都更像数据口径的变化，而不是流动性变化。指数报价是供应商的合成报价，不是可交易价格，因此它的水平不是交易成本。

### 避免前视偏差

以下三种常见做法会泄露未来信息，叠加起来会把样本外 R² 虚增到约 0.89：

1. 对训练窗口内的日子使用前向–后向平滑器（`predict_proba`）得到 HMM 概率。
2. 在构建滞后特征之前，按全样本分位数删除异常值。
3. 用双向插值填补缺口。

本项目的 notebook 都避开了这三点：概率由只用更早数据拟合的模型做前向滤波得到，异常值阈值只用过去的数据，缺口只用前向填充。

## 仓库结构

```
notebooks/
  01_data_and_regimes.ipynb              清洗数据、构建特征、拟合滚动 HMM      -> features 表
  02_forecasting.ipynb                   所有滚动预测和尖峰分类器               -> predictions、cv_windows、
                                                                                  spike_predictions、shap_importance 表
  03_evaluation.ipynb                    报告中的所有表格和图，只从数据库读取
  04_robustness.ipynb                    断点检验、分时期结果、bootstrap 置信区间、模型置信集
reports/
  report.md                              项目报告（英文）
  report_zh.md                           报告的中文翻译
  figures/                               报告中的图
data/                                    data.csv 和项目数据库 spx.db（都不包含，见下文）
```

## 数据

数据是 `SPX Index` 的 Bloomberg 日度字段，不能再分发，所以仓库不包含 `data/data.csv`。如需复现，请从 Bloomberg 导出以下字段（例如在 Excel 中用 BDH），日频，1998-03-02 至 2025-11-13，保存为 `data/data.csv`：

```
Dates, PX_LAST, CUR_MKT_CAP, PX_BID, PX_ASK, PX_MID, CHG_PCT_1D, BEST_EPS, PX_VOLUME, BEST_SALES, BEST_PE_RATIO, PX_CLOSE_1D
```

`Dates` 可以是 Excel 序列号，也可以是日期字符串。分析实际用到的字段是 PX_BID、PX_ASK、PX_MID、PX_LAST、PX_CLOSE_1D、PX_VOLUME 和 BEST_PE_RATIO。

## 运行

建议使用 Python 3.11 及以上版本。

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

按顺序运行四个 notebook。它们之间通过本地 SQLite 数据库 `data/spx.db` 传递数据：notebook 1 创建数据库（约 2 分钟），notebook 2 写入预测结果（约 20 分钟），notebook 3 和 4 只读取数据库（各约 1 分钟），因此重跑评估时不需要重新训练模型。随机种子已固定，结果可复现。

## 作者

Shangrong Li、Siyu Liang、Wenyi Ye 对本项目贡献均等。

## 许可证

代码采用 [MIT 许可证](LICENSE)。Bloomberg 数据不在许可范围内，也不随仓库分发。
