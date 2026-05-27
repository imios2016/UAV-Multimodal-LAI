# -*- coding: utf-8 -*-
"""
Repeated experiment (20 rounds) with mutual information feature selection.
Trains RF, XGB, LGBM, TabPFN, and Ensemble models for maize LAI estimation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
import warnings
import time
import pickle
from datetime import datetime
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression

os.environ["TABPFN_TOKEN"] = "YOUR_TABPFN_TOKEN_HERE"
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import optuna
from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

try:
    import torch
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
except ImportError:
    pass

DATA_PATH = r"YOUR_DATA_PATH"
OUTPUT_DIR = r"YOUR_OUTPUT_PATH"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_log.txt")

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}
")

if not os.path.exists(DATA_PATH):
    log(f"Warning: Data file not found at {DATA_PATH}")

TEST_SIZE = 0.2
N_TRIALS = 20
N_INNER_FOLDS = 3
N_OUTER_REPEATS = 20
EARLY_STOPPING_ROUNDS = 50
VAL_SIZE = 0.1

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
IJoRS_CONFIG = {
    'figure_width_single': 8.5,
    'figure_width_double': 17.5,
    'figure_height_max': 23.0,
    'font_size_label': 8, 'font_size_tick': 7, 'font_size_title': 9,
    'font_size_legend': 7, 'font_size_annotation': 7,
    'line_width': 0.8, 'marker_size': 20, 'alpha_scatter': 0.6,
    'dpi_png': 300, 'dpi_eps': 300,
}
IJoRS_COLORS = {
    'RF': '#E41A1C', 'XGB': '#377EB8', 'LGBM': '#4DAF4A',
    'TabPFN': '#984EA3', 'Ensemble': '#FF7F00',
    'reference': '#000000', 'rug': '#666666', 'residual': '#FF7F00',
}

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
DATE_COL = 'date'
GROUP_COL = 'label'

def save_figure_ijors(fig, filename_base, output_dir):
    for fmt in ['png', 'eps', 'pdf']:
        path = os.path.join(output_dir, f"{filename_base}.{fmt}")
        fig.savefig(path, dpi=IJoRS_CONFIG['dpi_png'], bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    log(f"  Figure saved: {filename_base}.png/.eps/.pdf")

def calculate_metrics(y_true, y_pred, n_features=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100
    y_range = y_true.max() - y_true.min()
    nrmse_range = rmse / y_range if y_range != 0 else np.inf
    y_mean = y_true.mean()
    nrmse_mean = rmse / y_mean if y_mean != 0 else np.inf
    adj_r2 = None
    if n_features is not None:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if (n - n_features - 1) > 0 else None
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    rrse = np.sqrt(np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)) if y_mean != 0 else np.inf
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape,
        'NRMSE(range)': nrmse_range, 'NRMSE(mean)': nrmse_mean,
        'R2': r2, 'Adjusted_R2': adj_r2,
        'Pearson_r': pearson_r, 'Pearson_p': pearson_p,
        'Kendall_tau': kendall_tau, 'Kendall_p': kendall_p,
        'MBE': mbe, 'RRSE': rrse, 'n_samples': n
    }

def remove_outliers_zscore(df, cols, target_col, z_thresh=3.5):
    log("No outliers detected.")
    return df

def check_tabpfn_input(n_samples, n_features):
    if n_samples > 2000:
        log(f"Warning: TabPFN recommended for n<=2000, current: {n_samples}")
    if n_features > 100:
        log(f"Warning: TabPFN recommended for p<=100, current: {n_features}")

def train_with_early_stopping(model, X_train, y_train, model_name, val_size=VAL_SIZE,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=RANDOM_STATE):
    if model_name in ['XGB', 'LGBM']:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state)
        eval_set = [(X_tr, y_tr), (X_val, y_val)]
        if model_name == 'XGB':
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_tr, y_tr, eval_set=eval_set,
                      callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)])
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

def engineer_features(df):
    vi_pool = [c for c in VI_FEATURES if c in df.columns]
    struct_pool = [c for c in STRUCT_FEATURES if c in df.columns]
    gbk_pool = [c for c in GBK_COLS_USE if c in df.columns]
    all_features = list(dict.fromkeys(vi_pool + struct_pool + gbk_pool))
    return df, all_features

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
    log(f"  MI selection: {len(available)} -> {len(mi_selected)} (top {mi_ratio*100:.0f}% + mandatory)")
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
    log(f"  Correlation dedup: {len(mi_selected)} -> {len(keep)} (corr < {corr_threshold})")
    return keep

def check_data_quality(df, feature_cols, target_col):
    df_check = df[feature_cols + [target_col]].copy()
    n_nan = df_check.isnull().sum().sum()
    n_inf = np.isinf(df_check.select_dtypes(include=[np.number]).values).sum()
    if n_nan > 0:
        log(f"Warning: {n_nan} NaN detected, dropping rows")
        df = df.dropna(subset=feature_cols + [target_col]).copy()
    if n_inf > 0:
        log(f"Warning: {n_inf} Inf detected, replacing with NaN")
        df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.loc[df_check.dropna().index].copy()
    return df

def group_split_by_field(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[GROUP_COL].values
    for train_idx, test_idx in gss.split(df, groups=groups):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
    log(f"Group split (seed={random_state}): Train fields {df_train[GROUP_COL].nunique()}, "
        f"Test fields {df_test[GROUP_COL].nunique()}")
    return df_train, df_test

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

def run_single_experiment(df_train, df_test, repeat_seed):
    df_train_feat, feat_pool = engineer_features(df_train)
    df_test_feat, _ = engineer_features(df_test)
    df_train_feat = check_data_quality(df_train_feat, feat_pool, TARGET_COL)
    df_test_feat = check_data_quality(df_test_feat, feat_pool, TARGET_COL)
    selected_feats = select_features_global(df_train_feat, feat_pool)
    X_train = df_train_feat[selected_feats].values
    y_train = df_train_feat[TARGET_COL].values
    X_test = df_test_feat[selected_feats].values
    y_test = df_test_feat[TARGET_COL].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n_features = X_train_scaled.shape[1]

    results = {}
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_metrics = calculate_metrics(y_test, baseline_pred, n_features=0)
    results['Baseline'] = {
        'model_name': 'Baseline',
        'test_metrics': baseline_metrics,
        'y_test': y_test, 'y_pred': baseline_pred
    }

    check_tabpfn_input(X_train_scaled.shape[0], n_features)
    log("  Training TabPFN...")
    try:
        tabpfn_model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_6, random_state=repeat_seed, device='cpu')
    except Exception as e:
        log(f"TabPFN init fallback: {e}")
        tabpfn_model = TabPFNRegressor(random_state=repeat_seed, device='cpu')
    tabpfn_model.fit(X_train_scaled, y_train)
    y_pred_tabpfn = tabpfn_model.predict(X_test_scaled)
    results['TabPFN'] = {
        'model_name': 'TabPFN',
        'model': tabpfn_model,
        'test_metrics': calculate_metrics(y_test, y_pred_tabpfn, n_features),
        'y_test': y_test, 'y_pred': y_pred_tabpfn
    }

    tree_models = ['RF', 'XGB', 'LGBM']
    for model_name in tree_models:
        log(f"  Optimizing {model_name}...")
        study = optuna.create_study(direction='minimize',
                                    sampler=TPESampler(seed=repeat_seed))
        study.optimize(lambda t: objective(t, model_name, X_train_scaled, y_train),
                       n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        if model_name == 'RF':
            best_params.update({'random_state': repeat_seed, 'n_jobs': -1})
        elif model_name == 'XGB':
            best_params.update({'random_state': repeat_seed, 'verbosity': 0, 'n_jobs': -1})
        elif model_name == 'LGBM':
            best_params.update({'random_state': repeat_seed, 'verbosity': -1, 'n_jobs': -1})

        final_model = create_model(model_name, best_params)
        final_model = train_with_early_stopping(final_model, X_train_scaled, y_train,
                                                model_name, random_state=repeat_seed)
        y_pred = final_model.predict(X_test_scaled)
        results[model_name] = {
            'model_name': model_name,
            'model': final_model,
            'test_metrics': calculate_metrics(y_test, y_pred, n_features),
            'best_params': best_params,
            'y_test': y_test, 'y_pred': y_pred
        }

    ensemble_pred = np.mean([results[m]['y_pred'] for m in ['RF', 'XGB', 'LGBM', 'TabPFN']], axis=0)
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred, n_features)
    results['Ensemble'] = {
        'model_name': 'Ensemble',
        'test_metrics': ensemble_metrics,
        'y_test': y_test, 'y_pred': ensemble_pred
    }

    return results, selected_feats

def compute_group_errors(y_true, y_pred, groups, group_name):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
    metrics = []
    for g, sub in df.groupby('group'):
        if len(sub) > 1:
            r2 = r2_score(sub['y_true'], sub['y_pred'])
            rmse = np.sqrt(mean_squared_error(sub['y_true'], sub['y_pred']))
        else:
            r2, rmse = np.nan, np.nan
        metrics.append({'group': g, 'R2': r2, 'RMSE': rmse})
    return pd.DataFrame(metrics)

def plot_model_scatter(results_dict, model_names, output_dir, suffix=''):
    n_models = len(model_names)
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
    axes = axes.flatten()
    for idx, m in enumerate(model_names):
        ax = axes[idx]
        res = results_dict[m]
        ax.scatter(res['y_test'], res['y_pred'], alpha=0.6, s=20, color=IJoRS_COLORS.get(m, 'gray'))
        lims = [min(res['y_test'].min(), res['y_pred'].min()),
                max(res['y_test'].max(), res['y_pred'].max())]
        ax.plot(lims, lims, 'k--', lw=0.8)
        ax.set_title(f"{m}\nR2={res['test_metrics']['R2']:.3f} RMSE={res['test_metrics']['RMSE']:.3f}")
        ax.set_xlabel('Observed'); ax.set_ylabel('Predicted')
        ax.axis('equal')
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    save_figure_ijors(fig, f'scatter_{suffix}', output_dir)
    plt.close()

def main():
    log("Experiment started with outer repeated group splits.")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    log(f"Data loaded: {df.shape}")

    for col in GBK_COLS_RAW:
        if col in df.columns:
            df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).astype(int)

    feat_pool, _ = engineer_features(df)
    df_clean = remove_outliers_zscore(df, feat_pool, TARGET_COL, z_thresh=3.5)
    log(f"Data after outlier removal: {df_clean.shape}")

    all_test_metrics = []
    all_group_errors = []

    for repeat in range(N_OUTER_REPEATS):
        seed = RANDOM_STATE + repeat
        log(f"\n{'='*60}")
        log(f"Outer Repeat {repeat+1}/{N_OUTER_REPEATS} (seed={seed})")
        log(f"{'='*60}")
        df_train, df_test = group_split_by_field(df_clean, random_state=seed)
        results, features = run_single_experiment(df_train, df_test, repeat_seed=seed)
        for model_name, res in results.items():
            metrics = res['test_metrics']
            metrics['Model'] = model_name
            metrics['Repeat'] = repeat
            all_test_metrics.append(metrics)
        ens_res = results['Ensemble']
        date_errors = compute_group_errors(ens_res['y_test'], ens_res['y_pred'],
                                           df_test[DATE_COL], 'Date')
        field_errors = compute_group_errors(ens_res['y_test'], ens_res['y_pred'],
                                            df_test[GROUP_COL], 'Field')
        all_group_errors.append({
            'Repeat': repeat,
            'Avg_R2_Date': date_errors['R2'].mean(),
            'Std_R2_Date': date_errors['R2'].std(),
            'Avg_RMSE_Date': date_errors['RMSE'].mean(),
            'Avg_R2_Field': field_errors['R2'].mean(),
            'Std_R2_Field': field_errors['R2'].std(),
            'Avg_RMSE_Field': field_errors['RMSE'].mean()
        })
        with open(os.path.join(OUTPUT_DIR, f"results_repeat{repeat}.pkl"), 'wb') as f:
            pickle.dump(results, f)

    metrics_df = pd.DataFrame(all_test_metrics)
    summary = metrics_df.groupby('Model').agg(
        R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
        MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
    ).reset_index()
    from scipy.stats import t
    n_repeats = N_OUTER_REPEATS
    confidence = 0.95
    t_val = t.ppf((1 + confidence) / 2, n_repeats - 1)
    for col in ['R2', 'RMSE', 'MAE']:
        summary[f'{col}_CI_lower'] = summary[f'{col}_mean'] - t_val * summary[f'{col}_std'] / np.sqrt(n_repeats)
        summary[f'{col}_CI_upper'] = summary[f'{col}_mean'] + t_val * summary[f'{col}_std'] / np.sqrt(n_repeats)

    log("\n======= Final Summary (Mean +/- Std) [95% CI] =======")
    log(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "final_summary_with_CI.csv"), index=False)

    group_df = pd.DataFrame(all_group_errors)
    group_summary = group_df.describe()
    log("\n======= Group Error Summary (Ensemble) =======")
    log(group_summary.to_string())
    group_df.to_csv(os.path.join(OUTPUT_DIR, "group_error_analysis.csv"), index=False)

    last_seed = RANDOM_STATE + N_OUTER_REPEATS - 1
    df_train_last, df_test_last = group_split_by_field(df_clean, random_state=last_seed)
    last_results, _ = run_single_experiment(df_train_last, df_test_last, repeat_seed=last_seed)
    plot_model_scatter(last_results, ['RF','XGB','LGBM','TabPFN','Baseline','Ensemble'],
                       OUTPUT_DIR, suffix='final')

    log("\nExperiment completed. All results saved.")

if __name__ == "__main__":
    main()
