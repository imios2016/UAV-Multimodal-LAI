# -*- coding: utf-8 -*-
"""
Multi-dimensional model comparison and selection with persistence.
Compares RF, XGB, LGBM, and TabPFN; selects optimal model via comprehensive scoring.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel
from datetime import datetime
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("Warning: TabPFN not installed, skipping")

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
IJoRS_COLORS = {
    'RF': '#E41A1C', 'XGB': '#377EB8', 'LGBM': '#4DAF4A',
    'TabPFN': '#984EA3', 'Ensemble': '#FF7F00', 'Baseline': '#000000'
}

OUTPUT_DIR = r"YOUR_OUTPUT_PATH"
DATA_PATH = r"YOUR_DATA_PATH"
DEPLOY_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_deployment_model.pkl")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_OUTER_REPEATS = 20
ALPHA = 0.05
N_TRIALS = 20
N_INNER_FOLDS = 3
EARLY_STOPPING_ROUNDS = 50
VAL_SIZE = 0.1

VI_FEATURES = [
    'VI_NGRDI_max', 'VI_NGRDI_mean', 'VI_NGRDI_std',
    'VI_ARI1_max', 'VI_ARI1_mean', 'VI_ARI1_std',
    'VI_ARI2_max', 'VI_ARI2_mean', 'VI_ARI2_std',
    'VI_ATSAVI_max', 'VI_ATSAVI_mean', 'VI_ATSAVI_std',
    'VI_CCCI_max', 'VI_CCCI_mean', 'VI_CCCI_std',
    'VI_CIgre_max', 'VI_CIgre_mean', 'VI_CIgre_std',
    'VI_CIredge_max', 'VI_CIredge_mean', 'VI_CIredge_std',
    'VI_GRVI_max', 'VI_GRVI_mean', 'VI_GRVI_std',
    'VI_GNDVI_max', 'VI_GNDVI_mean', 'VI_GNDVI_std',
    'VI_MCARI_max', 'VI_MCARI_mean', 'VI_MCARI_std',
    'VI_MSAVI_max', 'VI_MSAVI_mean', 'VI_MSAVI_std',
    'VI_NDRE_max', 'VI_NDRE_mean', 'VI_NDRE_std',
    'VI_NDVI_max', 'VI_NDVI_mean', 'VI_NDVI_std',
    'VI_OSAVI_max', 'VI_OSAVI_mean', 'VI_OSAVI_std',
    'VI_PSRI_max', 'VI_PSRI_mean', 'VI_PSRI_std',
    'VI_PVI_max', 'VI_PVI_mean', 'VI_PVI_std',
    'VI_RDVI_max', 'VI_RDVI_mean', 'VI_RDVI_std',
    'VI_RVI_max', 'VI_RVI_mean', 'VI_RVI_std',
    'VI_TSAVI_max', 'VI_TSAVI_mean', 'VI_TSAVI_std',
    'VI_WDRVI_max', 'VI_WDRVI_mean', 'VI_WDRVI_std',
]
STRUCT_FEATURES = [
    'PCD_Mean', 'PCD_Max', 'PCD_Std',
    'CVI', 'CV_Height', 'CRR', 'H_top10_avg',
    'Tex_ASM', 'Tex_Contrast', 'Tex_Corr', 'Tex_Entropy', 'Tex_Homo'
]
GBK_COLS_RAW = ['GBK_DH', 'GBK_Mixed', 'GBK_TEM', 'GBK_TST']
GBK_COLS_USE = ['GBK_DH', 'GBK_Mixed', 'GBK_TEM', 'GBK_TST']
TARGET_COL = 'LAI'

LOG_FILE = os.path.join(OUTPUT_DIR, "model_comparison_log.txt")
def log(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}
")

log("="*80)
log("Starting model comparison and standard file generation")
log("="*80)

summary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "final_summary_with_CI.csv"))

model_list = ['RF', 'XGB', 'LGBM']
if HAS_TABPFN:
    model_list.append('TabPFN')
model_df = summary_df[summary_df['Model'].isin(model_list)].reset_index(drop=True)

log("
" + "-"*60)
log(f"1. Generating raw_metrics_repeats.csv ({N_OUTER_REPEATS} repeats)")
all_metrics = []
for repeat in range(N_OUTER_REPEATS):
    with open(os.path.join(OUTPUT_DIR, f"results_repeat{repeat}.pkl"), 'rb') as f:
        results = pickle.load(f)
    for model in model_list:
        metrics = results[model]['test_metrics']
        all_metrics.append({
            'Model': model,
            'Repeat': repeat,
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
raw_metrics_df = pd.DataFrame(all_metrics)
raw_metrics_df = raw_metrics_df[['Model', 'Repeat', 'R2', 'RMSE', 'MAE']]
raw_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "raw_metrics_repeats.csv"), index=False)
log(f"  Generated: raw_metrics_repeats.csv, {raw_metrics_df.shape[0]} rows")
print(raw_metrics_df.head().round(4).to_string(index=False))

log("
" + "-"*60)
log("2. Generating model_grouped_statistics.csv")
grouped_stats = raw_metrics_df.groupby('Model').agg(
    R2_mean=('R2', 'mean'),
    R2_std=('R2', 'std'),
    R2_min=('R2', 'min'),
    R2_max=('R2', 'max'),
    RMSE_mean=('RMSE', 'mean'),
    RMSE_std=('RMSE', 'std'),
    RMSE_min=('RMSE', 'min'),
    RMSE_max=('RMSE', 'max'),
    MAE_mean=('MAE', 'mean'),
    MAE_std=('MAE', 'std'),
    MAE_min=('MAE', 'min'),
    MAE_max=('MAE', 'max')
).reset_index()
grouped_stats.to_csv(os.path.join(OUTPUT_DIR, "model_grouped_statistics.csv"), index=False)
log(f"  Generated: model_grouped_statistics.csv")
print(grouped_stats.round(4).to_string(index=False))

log("
" + "-"*60)
log("3. Generating model_paired_ttest.csv")
from itertools import combinations
test_results = []

def paired_ttest_standard(model1, model2, metric):
    m1_vals = raw_metrics_df[raw_metrics_df['Model']==model1][metric].values
    m2_vals = raw_metrics_df[raw_metrics_df['Model']==model2][metric].values
    t_stat, p_val = ttest_rel(m1_vals, m2_vals)
    if metric == 'R2':
        better = model1 if np.mean(m1_vals) > np.mean(m2_vals) else model2
    else:
        better = model1 if np.mean(m1_vals) < np.mean(m2_vals) else model2
    if p_val < ALPHA:
        conclusion = f"{better} significantly better on {metric} (p={p_val:.4f}<{ALPHA})"
    else:
        conclusion = f"{better} better on {metric}, but not significant (p={p_val:.4f}>={ALPHA})"
    return {
        'Model1': model1,
        'Model2': model2,
        'Metric': metric,
        't-statistic': t_stat,
        'p-value': p_val,
        'Better_Model': better,
        'Conclusion': conclusion
    }

for model1, model2 in combinations(model_list, 2):
    for metric in ['R2', 'RMSE', 'MAE']:
        test_results.append(paired_ttest_standard(model1, model2, metric))

ttest_df = pd.DataFrame(test_results)
ttest_df = ttest_df[['Model1', 'Model2', 'Metric', 't-statistic', 'p-value', 'Better_Model', 'Conclusion']]
ttest_df.to_csv(os.path.join(OUTPUT_DIR, "model_paired_ttest.csv"), index=False)
log(f"  Generated: model_paired_ttest.csv, {ttest_df.shape[0]} tests")
print(ttest_df.head().round(4).to_string(index=False))

log("
" + "-"*60)
log("4. Confidence interval details:")
ci_detail_df = model_df[['Model', 
                         'R2_mean', 'R2_CI_lower', 'R2_CI_upper',
                         'RMSE_mean', 'RMSE_CI_lower', 'RMSE_CI_upper',
                         'MAE_mean', 'MAE_CI_lower', 'MAE_CI_upper']].copy()
ci_detail_df.columns = ['Model', 
                        'R2_mean', 'R2_CI_lower', 'R2_CI_upper',
                        'RMSE_mean', 'RMSE_CI_lower', 'RMSE_CI_upper',
                        'MAE_mean', 'MAE_CI_lower', 'MAE_CI_upper']
print(ci_detail_df.round(4).to_string(index=False))
ci_detail_df.to_csv(os.path.join(OUTPUT_DIR, "model_ci_detail.csv"), index=False, encoding='utf-8-sig')
log(f"  CI details saved: model_ci_detail.csv")

log("
" + "-"*60)
log("5. Generating bar plots with CI...")
def plot_metric_with_ci(df, metric, y_label, save_name):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    models = df['Model'].values
    means = df[f'{metric}_mean'].values
    ci_lower = df[f'{metric}_CI_lower'].values
    ci_upper = df[f'{metric}_CI_upper'].values
    errors = [means - ci_lower, ci_upper - means]

    bars = ax.bar(
        models, means, 
        yerr=errors, capsize=5, 
        color=[IJoRS_COLORS[m] for m in models],
        edgecolor='black', linewidth=0.8, alpha=0.8
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if 'R2' in metric else 0.05),
            f'{mean:.3f}', ha='center', va='bottom', fontsize=8
        )

    ax.set_ylabel(y_label, fontsize=8)
    ax.set_xlabel('Model', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for fmt in ['png', 'eps', 'pdf']:
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{save_name}.{fmt}'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
    plt.close()

plot_metric_with_ci(model_df, 'R2', 'R2', 'model_r2_ci')
plot_metric_with_ci(model_df, 'RMSE', 'RMSE', 'model_rmse_ci')
plot_metric_with_ci(model_df, 'MAE', 'MAE', 'model_mae_ci')
log("  Bar plots saved")

log("
" + "-"*60)
log("6. Boxplot statistics:")
def calculate_boxplot_stats(df, metric):
    stats_list = []
    for model in model_list:
        data = df[df['Model']==model][metric].values
        q1 = np.percentile(data, 25)
        median = np.median(data)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        whisker_min = np.min(data[data >= q1 - 1.5*iqr])
        whisker_max = np.max(data[data <= q3 + 1.5*iqr])
        outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
        stats_list.append({
            'Model': model,
            f'{metric}_Min': whisker_min,
            f'{metric}_Q1': q1,
            f'{metric}_Median': median,
            f'{metric}_Q3': q3,
            f'{metric}_Max': whisker_max,
            f'{metric}_Outlier_Count': len(outliers)
        })
    return pd.DataFrame(stats_list)

r2_box_stats = calculate_boxplot_stats(raw_metrics_df, 'R2')
rmse_box_stats = calculate_boxplot_stats(raw_metrics_df, 'RMSE')
mae_box_stats = calculate_boxplot_stats(raw_metrics_df, 'MAE')
box_stats_df = r2_box_stats.merge(rmse_box_stats, on='Model').merge(mae_box_stats, on='Model')
box_stats_df.to_csv(os.path.join(OUTPUT_DIR, "model_boxplot_stats.csv"), index=False)
log(f"  Boxplot stats saved: model_boxplot_stats.csv")

def plot_boxplot(df, metric, y_label, save_name):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    boxplot = ax.boxplot(
        [df[df['Model']==m][metric] for m in model_list],
        labels=model_list, patch_artist=True,
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        medianprops=dict(linewidth=0.8, color='black'),
        flierprops=dict(marker='o', markersize=3, alpha=0.6)
    )
    for patch, model in zip(boxplot['boxes'], model_list):
        patch.set_facecolor(IJoRS_COLORS[model])
        patch.set_alpha(0.8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_xlabel('Model', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for fmt in ['png', 'eps', 'pdf']:
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{save_name}.{fmt}'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
    plt.close()

plot_boxplot(raw_metrics_df, 'R2', 'R2', 'model_r2_boxplot')
plot_boxplot(raw_metrics_df, 'RMSE', 'RMSE', 'model_rmse_boxplot')
plot_boxplot(raw_metrics_df, 'MAE', 'MAE', 'model_mae_boxplot')
log("  Boxplots saved")

log("
" + "-"*60)
log("7. Comprehensive scoring:")
def calculate_comprehensive_score(df):
    df['R2_norm'] = (df['R2_mean'] - df['R2_mean'].min()) / (df['R2_mean'].max() - df['R2_mean'].min())
    df['RMSE_norm'] = 1 - (df['RMSE_mean'] - df['RMSE_mean'].min()) / (df['RMSE_mean'].max() - df['RMSE_mean'].min())
    df['MAE_norm'] = 1 - (df['MAE_mean'] - df['MAE_mean'].min()) / (df['MAE_mean'].max() - df['MAE_mean'].min())
    df['Stability_norm'] = 1 - (df['R2_std'] - df['R2_std'].min()) / (df['R2_std'].max() - df['R2_std'].min())
    df['Comprehensive_Score'] = (
        0.4 * df['R2_norm'] + 
        0.25 * df['RMSE_norm'] + 
        0.25 * df['MAE_norm'] + 
        0.1 * df['Stability_norm']
    )
    df = df.sort_values('Comprehensive_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df

model_df_scored = calculate_comprehensive_score(model_df.copy())
score_columns = ['Rank', 'Model', 'R2_mean', 'RMSE_mean', 'MAE_mean', 'R2_std', 'Comprehensive_Score']
print(model_df_scored[score_columns].round(4).to_string(index=False))
model_df_scored.to_csv(os.path.join(OUTPUT_DIR, "model_comprehensive_score_ranking.csv"), index=False)
log(f"  Ranking saved: model_comprehensive_score_ranking.csv")

log("
" + "-"*60)
log("8. Final model training and persistence")

def engineer_features(df):
    vi_pool = [c for c in VI_FEATURES if c in df.columns]
    struct_pool = [c for c in STRUCT_FEATURES if c in df.columns]
    gbk_pool = [c for c in GBK_COLS_USE if c in df.columns]
    all_features = list(dict.fromkeys(vi_pool + struct_pool + gbk_pool))
    return df, all_features

def check_data_quality(df, feature_cols, target_col):
    df_check = df[feature_cols + [target_col]].copy()
    n_nan = df_check.isnull().sum().sum()
    n_inf = np.isinf(df_check.select_dtypes(include=[np.number]).values).sum()
    if n_nan > 0:
        df = df.dropna(subset=feature_cols + [target_col]).copy()
    if n_inf > 0:
        df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.loc[df_check.dropna().index].copy()
    return df

def select_features_global(df_train, feature_pool, target_col=TARGET_COL,
                           mi_ratio=0.5, corr_threshold=0.95):
    available = list(dict.fromkeys([c for c in feature_pool if c in df_train.columns]))
    if len(available) <= 1:
        return available
    mandatory = [c for c in GBK_COLS_USE if c in available]
    X = df_train[available].values
    y = df_train[target_col].values
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=available).sort_values(ascending=False)
    n_keep_mi = max(2, int(np.ceil(len(available) * mi_ratio)))
    mi_selected = mi_series.head(n_keep_mi).index.tolist()
    for feat in mandatory:
        if feat not in mi_selected:
            mi_selected.append(feat)
    if len(mi_selected) <= 1:
        return mi_selected
    keep = [f for f in mandatory if f in mi_selected]
    if not keep:
        keep = [mi_selected[0]]
    corr_matrix = df_train[mi_selected].corr().abs().fillna(0.0)
    for feat in mi_selected:
        if feat in keep:
            continue
        corr_values = [corr_matrix.at[feat, kept] for kept in keep]
        if all(v < corr_threshold for v in corr_values):
            keep.append(feat)
    return keep

def get_param_space(model_name, trial):
    if model_name == 'RF':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
            'random_state': RANDOM_STATE, 'n_jobs': -1
        }
    elif model_name == 'XGB':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1
        }
    elif model_name == 'LGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 15, 60),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 3),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1
        }
    raise ValueError(f"Unknown model: {model_name}")

def create_model(model_name, params):
    if model_name == 'RF':
        return RandomForestRegressor(**params)
    elif model_name == 'XGB':
        return xgb.XGBRegressor(**params)
    elif model_name == 'LGBM':
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Unknown model: {model_name}")

def objective(trial, model_name, X_train, y_train):
    params = get_param_space(model_name, trial)
    model = create_model(model_name, params)
    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    return np.mean(rmse_scores)

def train_with_early_stopping(model, X_train, y_train, model_name, random_state=RANDOM_STATE):
    if model_name in ['XGB', 'LGBM']:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE, random_state=random_state)
        eval_set = [(X_tr, y_tr), (X_val, y_val)]
        if model_name == 'XGB':
            model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_tr, y_tr, eval_set=eval_set,
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)])
        if hasattr(model, 'best_iteration'):
            best_iter = model.best_iteration
            if model_name == 'XGB':
                model.set_params(n_estimators=best_iter, early_stopping_rounds=None)
                model.fit(X_train, y_train, verbose=False)
            else:
                model.set_params(n_estimators=best_iter)
                model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
    else:
        model.fit(X_train, y_train)
    return model

def train_tabpfn(X_train, y_train, random_state=RANDOM_STATE):
    try:
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_6, random_state=random_state, device='cpu')
    except Exception as e:
        print(f"TabPFN fallback: {e}")
        model = TabPFNRegressor(random_state=random_state, device='cpu')
    model.fit(X_train, y_train)
    return model

def train_final_model_on_full_data(df, algorithm):
    df, feat_pool = engineer_features(df)
    df = check_data_quality(df, feat_pool, TARGET_COL)
    selected = select_features_global(df, feat_pool, target_col=TARGET_COL)
    X = df[selected].values
    y = df[TARGET_COL].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == 'TabPFN':
        model = train_tabpfn(X_scaled, y, random_state=RANDOM_STATE)
        return model, scaler, selected

    print(f"Optimizing {algorithm} hyperparameters...")
    study = optuna.create_study(direction='minimize',
                                sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, algorithm, X_scaled, y),
                   n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params

    if algorithm == 'RF':
        bp.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    elif algorithm == 'XGB':
        bp.update({'random_state': RANDOM_STATE, 'verbosity': 0, 'n_jobs': -1})
    elif algorithm == 'LGBM':
        bp.update({'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1})

    model = create_model(algorithm, bp)
    model = train_with_early_stopping(model, X_scaled, y, algorithm, RANDOM_STATE)
    return model, scaler, selected

log("
" + "="*80)
log("9. Final summary and model persistence")
log("="*80)

best_model = model_df_scored.loc[0, 'Model']
best_r2 = model_df_scored.loc[0, 'R2_mean']
best_rmse = model_df_scored.loc[0, 'RMSE_mean']
best_mae = model_df_scored.loc[0, 'MAE_mean']
best_score = model_df_scored.loc[0, 'Comprehensive_Score']

log(f"Best model (comprehensive decision): {best_model}")
log(f"Best performance: R2={best_r2:.4f}, RMSE={best_rmse:.4f}, MAE={best_mae:.4f}")
log(f"Best score: {best_score:.4f}")

significant_wins = ttest_df[(ttest_df['Better_Model'] == best_model) & (ttest_df['p-value'] < ALPHA)]
log(f"Significantly better than: {len(significant_wins['Model2'].unique())} models")

log(f"
Training {best_model} on full data...")
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
for col in GBK_COLS_RAW:
    if col in df.columns:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).astype(int)

final_model, scaler, features = train_final_model_on_full_data(df, best_model)

deployment = {
    'model': final_model,
    'scaler': scaler,
    'features': features,
    'algorithm': best_model,
    'n_features': len(features),
    'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'performance': {
        'R2_mean': best_r2,
        'RMSE_mean': best_rmse,
        'MAE_mean': best_mae,
        'comprehensive_score': best_score
    }
}

os.makedirs(os.path.dirname(DEPLOY_MODEL_PATH), exist_ok=True)
with open(DEPLOY_MODEL_PATH, 'wb') as f:
    pickle.dump(deployment, f)

log(f"
Deployment model saved: {DEPLOY_MODEL_PATH}")
log(f"Selected features ({len(features)}): {features}")

with open(os.path.join(OUTPUT_DIR, "final_model_comparison_report.txt"), 'w', encoding='utf-8') as f:
    f.write("="*80 + "
")
    f.write("Model Comparison Final Report
")
    f.write("="*80 + "

")
    f.write(f"Repeats: {N_OUTER_REPEATS}
")
    f.write(f"Alpha: {ALPHA}
")
    f.write(f"Selection method: Multi-dimensional (R2:40%, RMSE:25%, MAE:25%, Stability:10%)

")
    f.write(f"Best model: {best_model}
")
    f.write(f"  R2: {best_r2:.4f} (95% CI: [{model_df_scored.loc[0, 'R2_CI_lower']:.4f}, {model_df_scored.loc[0, 'R2_CI_upper']:.4f}])
")
    f.write(f"  RMSE: {best_rmse:.4f} (95% CI: [{model_df_scored.loc[0, 'RMSE_CI_lower']:.4f}, {model_df_scored.loc[0, 'RMSE_CI_upper']:.4f}])
")
    f.write(f"  MAE: {best_mae:.4f} (95% CI: [{model_df_scored.loc[0, 'MAE_CI_lower']:.4f}, {model_df_scored.loc[0, 'MAE_CI_upper']:.4f}])
")
    f.write(f"  Score: {best_score:.4f}

")
    f.write(f"Ranking:
")
    for idx, row in model_df_scored.iterrows():
        f.write(f"  {idx+1}. {row['Model']} - Score: {row['Comprehensive_Score']:.4f}
")
    f.write(f"
Model path: {DEPLOY_MODEL_PATH}
")
    f.write(f"Features: {len(features)}
")
    f.write(f"Feature list: {features}

")
    f.write("="*80 + "
")
    f.write("All results saved to: " + OUTPUT_DIR + "
")
    f.write("="*80 + "
")

log(f"  Final report saved: final_model_comparison_report.txt")
log("
" + "="*80)
log("Model comparison and persistence complete!")
log("="*80)
