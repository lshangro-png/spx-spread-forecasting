# Regime-Aware Forecasting of S&P 500 Bid-Ask Spreads

Can market regimes help forecast the daily bid-ask spread of the S&P 500 index? This project combines a rolling
Hidden Markov Model (HMM) for regime detection with linear and gradient-boosting forecasts, evaluated strictly out of
sample on 27 years of Bloomberg data.

**Short answer:** spreads are moderately predictable (out-of-sample R² ≈ 0.36, MSE 28% below a persistence forecast),
a linear model on the log spread with long-memory (HAR) features works best, and HMM regimes are a useful risk
flag but add no forecasting value once look-ahead bias is removed.

Final project for MATH 628, by Shangrong Li, Siyu Liang and Wenyi Ye.

## Key results

| Model (rolling out-of-sample, 5,695 days) | R² | Skill vs persistence |
| --- | --- | --- |
| **Linear, log target** | **0.361** | **0.282** |
| Linear, level target | 0.336 | 0.254 |
| XGBoost | 0.279 | 0.190 |
| LightGBM | 0.275 | 0.186 |
| Persistence (tomorrow = today) | 0.110 | 0 |

- **HAR long-memory features** (5/22/63-day average log spread) are the only addition that significantly improves
  accuracy: the typical-day absolute log error drops from 0.210 to 0.193 (Diebold–Mariano p < 0.001).
- **HMM regimes** are persistent (expected duration about 20 days), and forecast errors are about 8× larger in the
  high-spread regime. But the regime probability does not improve point forecasts (p ≥ 0.33 across an ablation and
  three redesigned HMMs) or spike classification. For forecast intervals it beats a constant width (p = 0.005) but
  loses to a simple EWMA of past residuals (p < 0.001).
- **Spike forecasting** (spread above its past-year 90th percentile) reaches AUC 0.84, but a one-variable naive
  rule already gets 0.83.
- **Gradient boosting** overfits more than the linear model and does not beat it.

### Data issues worth knowing about

- Bloomberg fills exchange holidays with a verbatim copy of the previous day (256 rows); these are removed.
- The index bid/ask level jumps about 6× in June 2011 and about 4× in December 2016, each within one month. This looks
  like a methodology change rather than a change in liquidity, and results are reported by sub-period.

### Why an earlier version reported R² ≈ 0.89

The first version of this analysis leaked future information:

1. HMM probabilities came from the forward–backward smoother (`predict_proba`) on days inside the training window.
2. Outliers were removed with full-sample percentiles before the lag features were built.
3. Gaps were filled with two-sided interpolation.

The notebook fixes all three: probabilities are forward-filtered from models fitted on earlier data only, outlier
thresholds use past data only, and gaps are forward-filled.

## Repository layout

```
notebooks/spx_spread_forecasting.ipynb   full analysis, executed, with all figures and tables
scripts/                                 stand-alone reproductions of the robustness experiments
  _nbload.py                             loads the notebook's pipeline (data, features, HMM, CV helpers)
  hmm_ablation.py                        Section 12a: with vs without the HMM feature
  hmm_variants.py                        Section 12b: three redesigned HMMs
  interval_forecasts.py                  Section 13: HMM vs EWMA for 90% forecast intervals
  extensions.py                          Section 14: HAR / calendar features, spike classification
reports/
  report_zh.md                           project report (Chinese)
  Math628_Final_Report.pdf               original course report
data/                                    put data.csv here (not included, see below)
```

The project report (in Chinese) is [`reports/report_zh.md`](reports/report_zh.md). The PDF in
`reports/` is the original submission; its headline numbers (R² ≈ 0.89) are superseded by the
notebook, for the reasons above.

## Data

The data are daily Bloomberg fields for the `SPX Index`, which cannot be redistributed, so `data/data.csv` is not
included. To reproduce, export these fields from Bloomberg (for example with BDH in Excel), daily, 1998-03-02 to
2025-11-13, and save them as `data/data.csv`:

```
Dates, PX_LAST, CUR_MKT_CAP, PX_BID, PX_ASK, PX_MID, CHG_PCT_1D, BEST_EPS, PX_VOLUME, BEST_SALES, BEST_PE_RATIO, PX_CLOSE_1D
```

`Dates` may be an Excel serial number or a date string. The analysis uses PX_BID, PX_ASK, PX_MID, PX_LAST,
PX_CLOSE_1D, PX_VOLUME and BEST_PE_RATIO.

## Running it

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
jupyter lab notebooks/spx_spread_forecasting.ipynb
```

A full run of the notebook takes about 20 minutes on a laptop; Sections 0–11 (the main results) take about
3 minutes. Each script runs from the `scripts/` folder, for example `cd scripts && python hmm_ablation.py`.
Random seeds are fixed, so results are reproducible.

## License

The code is released under the [MIT License](LICENSE). The Bloomberg data is not covered and is not distributed.
