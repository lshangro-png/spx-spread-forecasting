"""Section 13: can the HMM regime improve 90% forecast intervals?

Intervals log ŷ ± 1.645·σ_t around the linear (log-target) forecast, with σ_t from past OOS residuals:
constant, EWMA (λ = 0.94), regime-specific variances (baseline HMM, local-z HMM, an rv_5 flag), and EWMA scaled
by the HMM regime ratio. Scored by coverage and the Gneiting-Raftery interval score. Runtime: about 2 minutes.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from _nbload import load

ns = load()
df, df_model, model_features = ns['df'], ns['df_model'], ns['model_features']
rolling_cv, diebold_mariano, run_hmm = ns['rolling_cv'], ns['diebold_mariano'], ns['run_hmm']

LB, ALPHA = 250, 0.10
z = norm.ppf(1 - ALPHA / 2)


def local_z(s, w=63):
    return (s - s.shift(1).rolling(w).mean()) / s.shift(1).rolling(w).std()


F = pd.DataFrame({c: local_z(df[c]) for c in ['log_rel_spread', 'log_rv_5', 'log_volume', 'log_ami']}) \
      .assign(price_trend=df['price_trend'])
probs, _, _ = run_hmm(F)
df_model['hmm_local_z_lag1'] = probs['prob_state_1'].shift(1).reindex(df_model.index)

r, _, _ = rolling_cv(df_model, model_features, model_type='lin')
r['e'] = np.log(r['y_true']) - np.log(r['y_pred'])           # OOS residual in log space
past_e = r['e'].shift(1)                                      # residual of day t is known only after day t


def sigma_constant():
    return past_e.rolling(LB, min_periods=60).std()


def sigma_ewma(lam=0.94):
    return np.sqrt((past_e ** 2).ewm(alpha=1 - lam, min_periods=60).mean())


def sigma_regime(p):
    # p_t = P(high regime at t | info up to t-1); regime variances estimated on the past LB days
    p_past, e2 = p.shift(1), past_e ** 2
    v_hi = (e2 * p_past).rolling(LB, min_periods=60).sum() / p_past.rolling(LB, min_periods=60).sum()
    v_lo = (e2 * (1 - p_past)).rolling(LB, min_periods=60).sum() / (1 - p_past).rolling(LB, min_periods=60).sum()
    return np.sqrt(p * v_hi + (1 - p) * v_lo)


rv_flag = (df_model['rv_5'] > df_model['rv_5'].shift(1).rolling(252, min_periods=60).median()).astype(float)
sigmas = {
    'constant': sigma_constant(),
    'EWMA': sigma_ewma(),
    'rv_5 flag (non-HMM)': sigma_regime(rv_flag.reindex(r.index)),
    'baseline HMM': sigma_regime(df_model['prob_state_1_lag1'].reindex(r.index)),
    'local-z HMM': sigma_regime(df_model['hmm_local_z_lag1'].reindex(r.index)),
}
sigmas['EWMA x local-z HMM'] = sigmas['EWMA'] * sigmas['local-z HMM'] / sigmas['constant']

valid = pd.concat(sigmas, axis=1).notna().all(axis=1)
y_log, f_log = np.log(r['y_true']), np.log(r['y_pred'])
state = r.loc[valid, 'state']

rows, scores = [], {}
for name, sig in sigmas.items():
    lo, hi = f_log - z * sig, f_log + z * sig
    score = (hi - lo) + 2 / ALPHA * (lo - y_log).clip(lower=0) + 2 / ALPHA * (y_log - hi).clip(lower=0)
    covered = ((y_log >= lo) & (y_log <= hi))[valid]
    scores[name] = score[valid]
    rows.append({'method': name, 'coverage': covered.mean(), 'coverage state 0': covered[state == 0].mean(),
                 'coverage state 1': covered[state == 1].mean(), 'interval score': scores[name].mean()})
print(pd.DataFrame(rows).set_index('method').round(4).to_string())

# DM on the interval score: with a zero "truth" and power=1, d_t = IS_a - IS_b
zero = np.zeros(valid.sum())
dm_rows = [{'method': name, 'vs': ref, **dict(zip(['DM', 'p-value'],
            diebold_mariano(zero, scores[name], scores[ref], power=1)))}
           for name in ['rv_5 flag (non-HMM)', 'baseline HMM', 'local-z HMM', 'EWMA x local-z HMM']
           for ref in ['constant', 'EWMA']]
print()
print(pd.DataFrame(dm_rows).round(4).to_string(index=False))
