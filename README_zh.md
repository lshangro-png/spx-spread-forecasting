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

### 需要注意的数据问题

- Bloomberg 把交易所休市日用前一天的数据原样填充（256 行），这些行已删除。
- 指数 bid/ask 的水平在 2011 年 6 月跳升约 6 倍、2016 年 12 月跳升约 4 倍，每次都在一个月内完成。这更像是计算口径的变化，而不是流动性变化，因此结果按时段分别报告。

### 避免前视偏差

以下三种常见做法会泄露未来信息，叠加起来会把样本外 R² 虚增到约 0.89：

1. 对训练窗口内的日子使用前向–后向平滑器（`predict_proba`）得到 HMM 概率。
2. 在构建滞后特征之前，按全样本分位数删除异常值。
3. 用双向插值填补缺口。

本项目的 notebook 避开了这三点：概率由只用更早数据拟合的模型做前向滤波得到，异常值阈值只用过去的数据，缺口只用前向填充。

## 仓库结构

```
notebooks/spx_spread_forecasting.ipynb   完整分析，已执行，包含所有图表
scripts/                                 稳健性实验的独立复现脚本
  _nbload.py                             加载 notebook 的流程（数据、特征、HMM、CV 函数）
  hmm_ablation.py                        第 12a 节：有无 HMM 特征的对比
  hmm_variants.py                        第 12b 节：三种重新设计的 HMM
  interval_forecasts.py                  第 13 节：90% 预测区间，HMM vs EWMA
  extensions.py                          第 14 节：HAR / 日历特征、尖峰分类
reports/
  report.md                              项目报告（英文）
  report_zh.md                           报告的中文翻译
  figures/                               报告中的图
data/                                    把 data.csv 放在这里（未包含，见下文）
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
jupyter lab notebooks/spx_spread_forecasting.ipynb
```

在笔记本电脑上完整运行 notebook 约需 20 分钟；第 0–11 节（主要结果）约需 3 分钟。各脚本在 `scripts/` 目录下运行，例如 `cd scripts && python hmm_ablation.py`。随机种子已固定，结果可复现。

## 许可证

代码采用 [MIT 许可证](LICENSE)。Bloomberg 数据不在许可范围内，也不随仓库分发。
