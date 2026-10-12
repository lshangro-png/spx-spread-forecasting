"""Section 12a: does the HMM regime probability improve out-of-sample point forecasts?

Runs the same rolling CV with and without `prob_state_1_lag1` for the linear model, XGBoost and LightGBM,
and reports OOS R² plus a Diebold-Mariano test of the two forecasts. Runtime: a few minutes.
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from _nbload import load

ns = load()
rolling_cv, metrics, diebold_mariano = ns['rolling_cv'], ns['metrics'], ns['diebold_mariano']
df_model, model_features = ns['df_model'], ns['model_features']
no_hmm = [f for f in model_features if f != 'prob_state_1_lag1']

rows = []
for name, model_type in [('Linear', 'lin'), ('XGBoost', 'xgb'), ('LightGBM', 'lgb')]:
    r_with, _, _ = rolling_cv(df_model, model_features, model_type=model_type)
    r_without, _, _ = rolling_cv(df_model, no_hmm, model_type=model_type)
    stat, p = diebold_mariano(r_with['y_true'], r_with['y_pred'], r_without['y_pred'])
    rows.append({'model': name,
                 'R2 with HMM': metrics(r_with['y_true'], r_with['y_pred'])['R2'],
                 'R2 without HMM': metrics(r_without['y_true'], r_without['y_pred'])['R2'],
                 'DM (with vs without)': stat, 'p-value': p})
print(pd.DataFrame(rows).set_index('model').round(4).to_string())

# in-sample check: OLS on standardized features with HAC standard errors
X = StandardScaler().fit_transform(df_model[model_features])
ols = sm.OLS(df_model['log_y'], sm.add_constant(X)).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
print('\nFull-sample OLS of log spread on standardized features (HAC):')
print(pd.DataFrame({'coef': ols.params[1:].values, 't': ols.tvalues[1:].values}, index=model_features)
      .round(4).to_string())
