"""Section 12b: three redesigned HMMs, compared with the no-HMM specification.

* local-z HMM: inputs standardized against their own past 63 days (removes data-vendor era effects)
* dynamics HMM: inputs are Δ log spread, |return|, Δ log volume (no overlap with the regressors)
* interactions: baseline HMM probability × spread lags (regime-dependent persistence)

Runtime: about 5 minutes (two extra rolling HMMs plus 10 rolling CVs).
"""
import numpy as np
import pandas as pd

from _nbload import load

ns = load()
df, df_model, model_features = ns['df'], ns['df_model'], ns['model_features']
rolling_cv, metrics, diebold_mariano, run_hmm = ns['rolling_cv'], ns['metrics'], ns['diebold_mariano'], ns['run_hmm']
no_hmm = [f for f in model_features if f != 'prob_state_1_lag1']


def local_z(s, w=63):
    return (s - s.shift(1).rolling(w).mean()) / s.shift(1).rolling(w).std()


hmm_inputs = {
    'local_z': pd.DataFrame({c: local_z(df[c]) for c in ['log_rel_spread', 'log_rv_5', 'log_volume', 'log_ami']})
                 .assign(price_trend=df['price_trend']),
    'dynamics': pd.DataFrame({'dlog_spread': df['log_rel_spread'].diff(), 'abs_return': df['abs_return'],
                              'dlog_volume': df['log_volume'].diff()}),
}
for key, F in hmm_inputs.items():
    probs, _, _ = run_hmm(F)
    df_model[f'hmm_{key}_lag1'] = probs['prob_state_1'].shift(1).reindex(df_model.index)

df_model['hmm_x_lag1'] = df_model['prob_state_1_lag1'] * df_model['log_spread_lag1']
df_model['hmm_x_dlag'] = df_model['prob_state_1_lag1'] * (df_model['log_spread_lag1'] - df_model['log_spread_lag2'])

specs = {
    'no HMM': no_hmm,
    'baseline HMM': model_features,
    'local-z HMM': no_hmm + ['hmm_local_z_lag1'],
    'dynamics HMM': no_hmm + ['hmm_dynamics_lag1'],
    'HMM + interactions': model_features + ['hmm_x_lag1', 'hmm_x_dlag'],
}
data = df_model.dropna(subset=['hmm_local_z_lag1', 'hmm_dynamics_lag1']).copy()

rows = []
for model_type, model_name in [('lin', 'Linear'), ('lgb', 'LightGBM')]:
    res = {label: rolling_cv(data, feats, model_type=model_type)[0] for label, feats in specs.items()}
    ref = res['no HMM']
    for label, r in res.items():
        stat, p = (np.nan, np.nan) if label == 'no HMM' else diebold_mariano(r['y_true'], r['y_pred'], ref['y_pred'])
        rows.append({'model': model_name, 'spec': label, 'R2': metrics(r['y_true'], r['y_pred'])['R2'],
                     'DM vs no HMM': stat, 'p-value': p})
print(pd.DataFrame(rows).round(4).to_string(index=False))
