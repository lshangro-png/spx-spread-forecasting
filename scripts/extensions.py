"""Section 14: HAR long-memory and calendar features, and spike classification.

Part A compares the linear model and LightGBM with/without HAR and calendar features (level R² and mean absolute
log error, DM tests). Part B forecasts whether tomorrow's log spread exceeds its past-year 90th percentile.
Runtime: about 10 minutes.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from _nbload import load

ns = load()
df, df_model, model_features, SEED = ns['df'], ns['df_model'], ns['model_features'], ns['SEED']
rolling_cv, metrics, diebold_mariano = ns['rolling_cv'], ns['metrics'], ns['diebold_mariano']

har = ['har_5', 'har_22', 'har_63']
cal = ['pre_holiday', 'post_holiday', 'month_end', 'month_start', 'opex', 'quad_witching',
       'dow_mon', 'dow_tue', 'dow_wed', 'dow_thu']

# ---------- Part A: HAR and calendar features ----------
data = df_model.dropna(subset=har).copy()
specs = {'base': model_features, '+HAR': model_features + har,
         '+calendar': model_features + cal, '+HAR+calendar': model_features + har + cal}
rows = []
for model_type, model_name in [('lin', 'Linear'), ('lgb', 'LightGBM')]:
    res = {label: rolling_cv(data, feats, model_type=model_type)[0] for label, feats in specs.items()}
    ref = res['base']
    for label, r in res.items():
        row = {'model': model_name, 'spec': label, 'R2': metrics(r['y_true'], r['y_pred'])['R2'],
               'MAE (log)': np.mean(np.abs(np.log(r['y_true']) - np.log(r['y_pred'])))}
        if label != 'base':
            row['DM log p'] = diebold_mariano(np.log(r['y_true']), np.log(r['y_pred']), np.log(ref['y_pred']))[1]
        rows.append(row)
print('Part A: HAR and calendar features')
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------- Part B: spike classification ----------
log_spread = np.log(df['rel_spread'])
threshold = log_spread.shift(1).rolling(252, min_periods=126).quantile(0.9)
df['spike'] = (log_spread > threshold).astype(float).where(threshold.notna())
df['spike_lag1'] = df['spike'].shift(1)
df['dist_to_thr'] = log_spread.shift(1) - threshold
for c in ['spike', 'spike_lag1', 'dist_to_thr']:
    data[c] = df[c].reindex(data.index)
data = data.dropna(subset=['spike', 'spike_lag1', 'dist_to_thr'])


def rolling_clf(d, features, model='logit', train_window=1000, step=21):
    out = []
    for i in range(train_window, len(d), step):
        tr, te = d.iloc[i - train_window:i], d.iloc[i:i + step]
        Xtr, ytr, Xte = tr[features].values, tr['spike'].values, te[features].values
        if ytr.sum() < 10:
            p = np.full(len(te), ytr.mean())
        elif model == 'logit':
            sc = StandardScaler().fit(Xtr)
            p = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
        else:
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=8, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
                                     random_state=SEED, verbose=-1, n_jobs=4)
            p = clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        out.append(pd.DataFrame({'y': te['spike'].values, 'p': p}, index=te.index))
    return pd.concat(out)


no_hmm = [f for f in model_features if f != 'prob_state_1_lag1']
base = no_hmm + ['spike_lag1', 'dist_to_thr']
clf_specs = {
    'logit, no HMM': ('logit', base),
    'logit, +HMM': ('logit', base + ['prob_state_1_lag1']),
    'logit, +HMM+HAR+cal': ('logit', base + ['prob_state_1_lag1'] + har + cal),
    'LGB, +HMM+HAR+cal': ('lgb', base + ['prob_state_1_lag1'] + har + cal),
}
res = {label: rolling_clf(data, feats, m) for label, (m, feats) in clf_specs.items()}
idx = res['logit, no HMM'].index
res['naive: dist_to_thr'] = res['logit, no HMM'][['y']].assign(p=data.loc[idx, 'dist_to_thr'])

rows = []
for label, r in res.items():
    top = r.nlargest(int(len(r) * 0.1), 'p')
    rows.append({'classifier': label, 'AUC': roc_auc_score(r['y'], r['p']),
                 'Avg precision': average_precision_score(r['y'], r['p']),
                 'Brier': brier_score_loss(r['y'], r['p']) if not label.startswith('naive') else np.nan,
                 'Precision@top10%': top['y'].mean()})
print(f"\nPart B: spike classification (base rate {res['logit, no HMM']['y'].mean():.3f})")
print(pd.DataFrame(rows).round(4).to_string(index=False))
