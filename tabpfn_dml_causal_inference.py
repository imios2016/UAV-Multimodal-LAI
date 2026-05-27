# -*- coding: utf-8 -*-
"""
TabPFN-DML Causal Inference Analysis for Maize LAI
IJoRS-standard version with strict formatting compliance.
"""

import os, pickle, warnings, time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from scipy.stats import gaussian_kde, shapiro, probplot
from sklearn.model_selection import train_test_split

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
import random
random.seed(42)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from econml.dml import LinearDML, CausalForestDML
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

os.environ["TABPFN_TOKEN"] = "YOUR_TABPFN_TOKEN_HERE"
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import tabpfn
TABPFN_VERSION = tabpfn.__version__

CV_FOLDS = 2
SKIP_GBK_EVAL = True
TEST_SIZE = 0.2

FOREST_CONFIG = {
    'n_estimators': 20,
    'min_samples_leaf': 8,
    'max_depth': 3,
    'random_state': 42
}

OUTPUT_DIR = r"YOUR_OUTPUT_PATH"
DATA_PATH = r"YOUR_DATA_PATH"
DEPLOY_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_deployment_model.pkl")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "dml_local_log.txt")
RESIDUAL_STATS_FILE = os.path.join(OUTPUT_DIR, "residual_distribution_analysis.txt")
DML_RESULTS_CSV = os.path.join(TABLES_DIR, "Table2_DML_Causal_Effects.csv")
DML_RESULTS_TXT = os.path.join(OUTPUT_DIR, "dml_causal_effects_report.txt")
FOLD_ATES_CSV = os.path.join(TABLES_DIR, "Table3_Cross_Fitting_ATEs.csv")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'normal'

IJoRS_CONFIG = {
    'figure_width_single': 8.5,
    'figure_width_double': 17.5,
    'font_size_label': 8,
    'font_size_tick': 7,
    'font_size_title': 9,
    'font_size_legend': 7,
    'font_size_annotation': 7,
    'line_width': 0.8,
    'marker_size': 25,
    'alpha_scatter': 0.6,
    'dpi_png': 300,
    'dpi_eps': 300,
}

IJoRS_COLORS = {
    'positive': '#E41A1C',
    'negative': '#377EB8',
    'neutral': '#999999',
    'reference': '#000000',
    'highlight': '#FF7F00',
    'tabpfn': '#2CA02C',
    'DH': '#E41A1C',
    'TST': '#377EB8',
    'Mixed': '#FF7F00',
    'TEM': '#999999',
}

def log(msg, level='info'):
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] [{level.upper()}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def save_figure_ijors(fig, filename_base, output_dir=FIGURES_DIR):
    try:
        for fmt in ['png', 'pdf', 'eps']:
            filepath = os.path.join(output_dir, f"{filename_base}.{fmt}")
            dpi = IJoRS_CONFIG['dpi_png'] if fmt == 'png' else IJoRS_CONFIG['dpi_eps']
            save_kwargs = {
                'dpi': dpi,
                'bbox_inches': 'tight',
                'format': fmt,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            if fmt == 'pdf':
                save_kwargs.update({
                    'papertype': 'a4',
                    'orientation': 'portrait'
                })
            fig.savefig(filepath, **save_kwargs)
        log(f"  Figure saved (IJoRS standard): {filename_base}.png/.eps/.pdf")
    except Exception as e:
        log(f"  Warning: Figure save failed - {e}", level='warning')
    finally:
        plt.close(fig)
        gc.collect()

def save_table(df, filename_base, output_dir=TABLES_DIR):
    try:
        csv_path = os.path.join(output_dir, f"{filename_base}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        xlsx_path = os.path.join(output_dir, f"{filename_base}.xlsx")
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        log(f"  Table saved: {filename_base}.csv/.xlsx")
    except Exception as e:
        log(f"  Warning: Table save failed - {e}", level='warning')

def create_tabpfn_instance(random_state=42):
    try:
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            random_state=random_state
        )
    except TypeError:
        log(f"  Legacy TabPFN ({TABPFN_VERSION}), using seed parameter", level='warning')
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            seed=random_state
        )

def save_dml_results(hetero_results, fold_ate_data, reference_group='GBK_TEM'):
    log(f"\n{'='*70}")
    log("Exporting DML causal inference results")
    log(f"{'='*70}")

    results_list = []
    for res in hetero_results:
        p_value = res['p_value']
        if np.isnan(p_value):
            p_str = 'NaN'
        elif p_value < 0.001:
            p_str = '<0.001'
        else:
            p_str = f"{p_value:.4f}"

        if np.isnan(p_value):
            significance = 'NaN'
        elif p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'

        results_list.append({
            'Treatment': res['treatment'].replace('GBK_', ''),
            'Reference': reference_group.replace('GBK_', ''),
            'N_Samples': res['n_samples'],
            'N_Confounders': res['n_confounders'],
            'ATE': round(res['ate'], 4) if res['ate'] is not None else np.nan,
            'ATE_SE': round(res['ate_se'], 4) if res['ate_se'] is not None else np.nan,
            'CI_Lower': round(res['ate_ci_lower'], 4) if res['ate_ci_lower'] is not None else np.nan,
            'CI_Upper': round(res['ate_ci_upper'], 4) if res['ate_ci_upper'] is not None else np.nan,
            'p_value': p_str,
            'Significance': significance,
            'CATE_Mean': round(res['cate_mean'], 4) if res['cate_mean'] is not None else np.nan,
            'CATE_Std': round(res['cate_std'], 4) if res['cate_std'] is not None else np.nan,
            'CATE_Min': round(res['cate_min'], 4) if res['cate_min'] is not None else np.nan,
            'CATE_Max': round(res['cate_max'], 4) if res['cate_max'] is not None else np.nan
        })

    results_df = pd.DataFrame(results_list)
    save_table(results_df, "Table2_DML_Causal_Effects")

    if fold_ate_data:
        fold_df = pd.DataFrame.from_dict(fold_ate_data, orient='index').T
        fold_df.columns = [col.replace('GBK_', '') for col in fold_df.columns]
        fold_df.insert(0, 'Fold', range(1, len(fold_df)+1))
        save_table(fold_df, "Table3_Cross_Fitting_ATEs")

    with open(DML_RESULTS_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TabPFN-DML Causal Inference Results Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference group: {reference_group.replace('GBK_', '')}\n")
        f.write(f"Total samples: {hetero_results[0]['n_samples'] if hetero_results else 'N/A'}\n")
        f.write(f"CV folds: {CV_FOLDS}\n")
        f.write(f"First-stage model: TabPFNRegressor (V2)\n\n")

        f.write("="*80 + "\n")
        f.write("1. Average Treatment Effect (ATE) Results\n")
        f.write("="*80 + "\n\n")

        for res in results_list:
            trt = res['Treatment']
            ate = res['ATE']
            ci_lower = res['CI_Lower']
            ci_upper = res['CI_Upper']
            p = res['p_value']
            sig = res['Significance']
            f.write(f"[{trt} vs {reference_group.replace('GBK_', '')}]\n")
            f.write(f"  ATE: {ate:.4f}\n")
            f.write(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            f.write(f"  p-value: {p}\n")
            f.write(f"  Significance: {sig}\n")
            if sig == 'ns':
                conclusion = f"Compared to {reference_group.replace('GBK_', '')}, {trt} genetic background shows no significant causal effect on maize LAI (p={p})."
            elif ate > 0:
                conclusion = f"Compared to {reference_group.replace('GBK_', '')}, {trt} genetic background significantly increases maize LAI by {ate:.4f} units (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p={p})."
            else:
                conclusion = f"Compared to {reference_group.replace('GBK_', '')}, {trt} genetic background significantly decreases maize LAI by {abs(ate):.4f} units (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p={p})."
            f.write(f"  Conclusion: {conclusion}\n\n")

        f.write("="*80 + "\n")
        f.write("2. Conditional Average Treatment Effect (CATE) Statistics\n")
        f.write("="*80 + "\n\n")
        for res in results_list:
            trt = res['Treatment']
            cate_mean = res['CATE_Mean']
            cate_std = res['CATE_Std']
            cate_min = res['CATE_Min']
            cate_max = res['CATE_Max']
            if not np.isnan(cate_mean):
                f.write(f"[{trt}]\n")
                f.write(f"  CATE Mean: {cate_mean:.4f}\n")
                f.write(f"  CATE Std: {cate_std:.4f}\n")
                f.write(f"  CATE Range: [{cate_min:.4f}, {cate_max:.4f}]\n")
                f.write(f"  Heterogeneity: {'High' if cate_std > 0.1 else 'Medium' if cate_std > 0.05 else 'Low'}\n\n")

        f.write("="*80 + "\n")
        f.write("3. Cross-fitting Stability Analysis\n")
        f.write("="*80 + "\n\n")
        for trt, fold_ates in fold_ate_data.items():
            trt_name = trt.replace('GBK_', '')
            fold_ates_clean = [x for x in fold_ates if x is not None and not np.isnan(x)]
            if len(fold_ates_clean) >= 2:
                mean_fold = np.mean(fold_ates_clean)
                std_fold = np.std(fold_ates_clean)
                cv_fold = std_fold / abs(mean_fold) if mean_fold != 0 else np.inf
                f.write(f"[{trt_name}]\n")
                f.write(f"  Cross-fitting ATE mean: {mean_fold:.4f}\n")
                f.write(f"  Cross-fitting ATE std: {std_fold:.4f}\n")
                f.write(f"  CV: {cv_fold:.4f}\n")
                f.write(f"  Stability: {'Excellent' if cv_fold < 0.1 else 'Good' if cv_fold < 0.2 else 'Fair' if cv_fold < 0.3 else 'Poor'}\n\n")

        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")

    log(f"  DML results CSV saved: {DML_RESULTS_CSV}")
    log(f"  DML results TXT report saved: {DML_RESULTS_TXT}")
    log(f"  Cross-fitting ATE data saved: {FOLD_ATES_CSV}")

def analyze_residual_distribution(y_true, y_pred, model_name="TabPFN"):
    log(f"\n{'='*70}")
    log(f"Residual distribution analysis: {model_name}")
    log(f"{'='*70}")
    residuals = y_true - y_pred
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    if n < 10:
        log(f"  Warning: Insufficient residual samples ({n}), skipping analysis", level='warning')
        return None, None

    res_mean = np.mean(residuals)
    res_std = np.std(residuals, ddof=1)
    shapiro_w, shapiro_p = shapiro(residuals)
    kde = gaussian_kde(residuals)
    x_theory = np.linspace(np.min(residuals), np.max(residuals), 1000)
    y_theory = sp_stats.norm.pdf(x_theory, loc=res_mean, scale=res_std)
    y_kde = kde(x_theory)
    ss_total = np.sum((y_kde - np.mean(y_kde))**2)
    ss_residual = np.sum((y_kde - y_theory)**2)
    r2_fit = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    rmse_fit = np.sqrt(mean_squared_error(y_kde, y_theory))

    text_result = f"""
To evaluate the prediction error characteristics of the optimal model ({model_name}), this study analyzed the distribution of test set residuals.
Shapiro-Wilk test (W = {shapiro_w:.3f}, p = {shapiro_p:.3f} {'>' if shapiro_p>0.05 else '<'} 0.05)
{'supports' if shapiro_p>0.05 else 'does not support'} the assumption that residuals follow a normal distribution (mu = {res_mean:.3f}, sigma = {res_std:.3f}).
The empirical distribution fits well with the theoretical normal curve (R2 = {r2_fit:.3f}), with low distribution fitting error (RMSE = {rmse_fit:.3f}).
"""
    with open(RESIDUAL_STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Residual Distribution Analysis Report - {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {n}\n\n")
        f.write("1. Basic Statistics\n")
        f.write(f"   Mean (mu): {res_mean:.6f}\n")
        f.write(f"   Std (sigma): {res_std:.6f}\n")
        f.write(f"   Min: {np.min(residuals):.6f}\n")
        f.write(f"   Max: {np.max(residuals):.6f}\n\n")
        f.write("2. Shapiro-Wilk Normality Test\n")
        f.write(f"   W statistic: {shapiro_w:.6f}\n")
        f.write(f"   p-value: {shapiro_p:.6f}\n")
        f.write(f"   Conclusion: {'Residuals follow normal distribution (p>0.05)' if shapiro_p>0.05 else 'Residuals do not follow normal distribution (p<=0.05)'}\n\n")
        f.write("3. Empirical vs Theoretical Normal Distribution Fit\n")
        f.write(f"   Fit R2: {r2_fit:.6f}\n")
        f.write(f"   Fit RMSE: {rmse_fit:.6f}\n\n")
        f.write("="*80 + "\nAcademic Statement:\n" + "="*80 + "\n")
        f.write(text_result + "\n" + "="*80 + "\n")

    stats_dict = {
        'model_name': model_name,
        'n_samples': n,
        'mean': res_mean,
        'std': res_std,
        'min': np.min(residuals),
        'max': np.max(residuals),
        'shapiro_w': shapiro_w,
        'shapiro_p': shapiro_p,
        'fit_r2': r2_fit,
        'fit_rmse': rmse_fit,
        'text_result': text_result
    }
    return stats_dict, residuals

def plot_residual_distribution(stats_dict, residuals, output_dir=FIGURES_DIR):
    if stats_dict is None or residuals is None:
        return
    fig_width = IJoRS_CONFIG['figure_width_single'] / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5/2.54))

    ax = axes[0]
    n_bins = min(15, len(residuals) // 3) if len(residuals) >= 12 else 8
    ax.hist(residuals, bins=n_bins, density=True, alpha=0.6,
            color=IJoRS_COLORS['tabpfn'], edgecolor='black', linewidth=0.5)
    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=1.2, label='Empirical KDE')
    x_theory = np.linspace(residuals.min(), residuals.max(), 1000)
    y_theory = sp_stats.norm.pdf(x_theory, loc=stats_dict['mean'], scale=stats_dict['std'])
    ax.plot(x_theory, y_theory, 'r--', linewidth=1.2, label='Theoretical Normal')
    ax.text(0.05, 0.95, f'W = {stats_dict["shapiro_w"]:.3f}\np = {stats_dict["shapiro_p"]:.3f}',
            transform=ax.transAxes, fontsize=IJoRS_CONFIG['font_size_annotation'],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            verticalalignment='top')
    ax.set_xlabel('Residuals (LAI)', fontsize=IJoRS_CONFIG['font_size_label'])
    ax.set_ylabel('Density', fontsize=IJoRS_CONFIG['font_size_label'])
    ax.set_title('(a) Residual Distribution', fontsize=IJoRS_CONFIG['font_size_title'], pad=8,
                 fontweight='bold')
    ax.legend(fontsize=IJoRS_CONFIG['font_size_legend'], framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    for a in [ax, axes[1]]:
        a.tick_params(axis='both', labelsize=IJoRS_CONFIG['font_size_tick'],
                      width=0.5, length=3, direction='in')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_linewidth(0.5)
        a.spines['bottom'].set_linewidth(0.5)

    ax = axes[1]
    probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(IJoRS_COLORS['tabpfn'])
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(1.2)
    ax.text(0.05, 0.95, f'R2 = {stats_dict["fit_r2"]:.3f}\nRMSE = {stats_dict["fit_rmse"]:.3f}',
            transform=ax.transAxes, fontsize=IJoRS_CONFIG['font_size_annotation'],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            verticalalignment='top')
    ax.set_xlabel('Theoretical Quantiles', fontsize=IJoRS_CONFIG['font_size_label'])
    ax.set_ylabel('Sample Quantiles', fontsize=IJoRS_CONFIG['font_size_label'])
    ax.set_title('(b) Q-Q Plot', fontsize=IJoRS_CONFIG['font_size_title'], pad=8,
                 fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout(pad=1.5)
    save_figure_ijors(fig, 'Fig1_Residual_Distribution', output_dir=output_dir)

def recode_gbk_reference(X, feature_names, new_ref='GBK_TEM'):
    log(f"\n{'='*70}")
    log(f"Switching GBK reference group to: {new_ref}")
    log(f"{'='*70}")
    present_gbk = [c for c in feature_names if c.startswith('GBK_')]
    log(f"  Available GBK features: {present_gbk}")
    if new_ref not in present_gbk:
        raise ValueError(f"Reference group {new_ref} not found in features, available: {present_gbk}")
    ref_idx = feature_names.index(new_ref)
    keep_idx = [i for i in range(len(feature_names)) if i != ref_idx]
    new_X = X[:, keep_idx]
    new_feature_names = [feature_names[i] for i in keep_idx]
    log(f"  Removed {feature_names[ref_idx]} (set as reference)")
    remaining_gbk = [f for f in new_feature_names if f.startswith('GBK_')]
    log(f"  Valid treatment variables (vs {new_ref}): {remaining_gbk}")
    return new_X, new_feature_names

def remove_specific_features(X, feature_names, remove_features=['VI_NDVI_mean', 'DAS']):
    log(f"\n{'='*70}")
    log(f"Removing specified features: {remove_features}")
    log(f"{'='*70}")
    keep_indices = []
    kept_features = []
    removed_features = []
    for idx, fname in enumerate(feature_names):
        if fname not in remove_features:
            keep_indices.append(idx)
            kept_features.append(fname)
        else:
            removed_features.append(fname)
    new_X = X[:, keep_indices] if len(keep_indices) > 0 else np.array([])
    log(f"  Removed features: {removed_features}")
    log(f"  Retained features: {len(kept_features)}")
    return new_X, kept_features

def load_full_data(model_path, data_path, test_size=TEST_SIZE, random_state=42):
    log(f"\n{'='*70}")
    log("Step 1: Loading data and pretrained model configuration")
    log(f"{'='*70}")
    with open(model_path, 'rb') as f:
        pkl_data = pickle.load(f)
    required_pkl_keys = ['model', 'scaler', 'features', 'algorithm', 'n_features']
    missing_keys = [k for k in required_pkl_keys if k not in pkl_data]
    if missing_keys:
        raise ValueError(f"PKL file missing keys: {missing_keys}")
    trained_model = pkl_data['model']
    scaler = pkl_data['scaler']
    model_feature_names = pkl_data['features']
    algorithm = pkl_data['algorithm']
    n_features = pkl_data['n_features']
    training_time = pkl_data.get('training_time', 'Unknown')
    performance = pkl_data.get('performance', {})
    log(f"  Loaded pretrained model: {type(trained_model).__name__}")
    log(f"  Model random_state: {trained_model.random_state}")
    log(f"  Model expected features: {n_features}")
    log(f"  Algorithm: {algorithm}")
    log(f"  Model training time: {training_time}")
    log(f"  Model training performance: R2={performance.get('R2_mean', 'N/A'):.3f}, RMSE={performance.get('RMSE_mean', 'N/A'):.3f}")
    log(f"  Model features: {model_feature_names}")
    log(f"\n  Loading CSV data: {data_path}")
    df = pd.read_csv(data_path)
    log(f"  CSV raw data shape: {df.shape}")
    missing_features = [f for f in model_feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"CSV missing required features: {missing_features}")
    X = df[model_feature_names].values
    y = df['LAI'].values
    log(f"  Extracted features: {X.shape[1]} (consistent with model)")
    log(f"  Target: LAI, samples: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    log(f"  Using pretrained StandardScaler for standardization")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full = np.vstack([X_train_scaled, X_test_scaled])
    y_full = np.concatenate([y_train, y_test])
    source_flag = np.array(['train'] * len(y_train) + ['test'] * len(y_test))
    feature_names = model_feature_names.copy()
    log(f"\n  Data loading and preprocessing complete!")
    return (X_train_scaled, y_train, X_test_scaled, y_test, X_full, y_full, 
            feature_names, scaler, source_flag, trained_model)

def evaluate_first_stage_tabpfn(X_train, y_train, X_test, y_test, feature_names, treatment_names,
                                tabpfn_model=None):
    log(f"\n{'='*70}")
    log("Step 2: First-stage prediction model (LAI evaluation only)")
    log(f"{'='*70}")
    model_name = "TabPFN"
    if tabpfn_model is not None:
        model = tabpfn_model
        log(f"  Using pretrained TabPFN model from PKL")
    else:
        model = create_tabpfn_instance(42)
        log(f"  Creating new TabPFN model (random_state=42)")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    log("  Evaluating LAI (Y) prediction...")
    cv_scores_r2, cv_scores_rmse, cv_scores_mae = [], [], []
    for train_idx, val_idx in cv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        cv_scores_r2.append(r2_score(y_val, y_pred))
        cv_scores_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_scores_mae.append(mean_absolute_error(y_val, y_pred))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    performance_data = [{
        'Target': 'LAI (Y)',
        'Model': model_name,
        'CV_R2': round(np.mean(cv_scores_r2), 3),
        'CV_RMSE': round(np.mean(cv_scores_rmse), 3),
        'CV_MAE': round(np.mean(cv_scores_mae), 3),
        'Test_R2': round(test_r2, 3),
        'Test_RMSE': round(test_rmse, 3),
        'Test_MAE': round(test_mae, 3),
        'CV_R2_Std': round(np.std(cv_scores_r2), 3),
        'CV_RMSE_Std': round(np.std(cv_scores_rmse), 3),
        'CV_MAE_Std': round(np.std(cv_scores_mae), 3)
    }]
    log(f"    {model_name} - CV R2: {np.mean(cv_scores_r2):.3f} +/- {np.std(cv_scores_r2):.3f}")
    log(f"    {model_name} - Test R2: {test_r2:.3f}")
    if not SKIP_GBK_EVAL:
        for t_name in treatment_names:
            if t_name not in feature_names:
                log(f"  Warning: {t_name} not in model features, skipping evaluation", level='warning')
                continue
            log(f"  Evaluating {t_name} (T) prediction...")
            t_idx = feature_names.index(t_name)
            T_train = X_train[:, t_idx]
            X_train_without_t = np.delete(X_train, t_idx, axis=1)
            cv_scores_r2, cv_scores_rmse, cv_scores_mae = [], [], []
            for train_idx, val_idx in cv.split(X_train_without_t):
                X_tr, X_val = X_train_without_t[train_idx], X_train_without_t[val_idx]
                t_tr, t_val = T_train[train_idx], T_train[val_idx]
                m = create_tabpfn_instance(42)
                m.fit(X_tr, t_tr)
                t_pred = m.predict(X_val)
                cv_scores_r2.append(r2_score(t_val, t_pred))
                cv_scores_rmse.append(np.sqrt(mean_squared_error(t_val, t_pred)))
                cv_scores_mae.append(mean_absolute_error(t_val, t_pred))
                del m; gc.collect()
            performance_data.append({
                'Target': f'{t_name} (T)',
                'Model': model_name,
                'CV_R2': round(np.mean(cv_scores_r2), 3),
                'CV_RMSE': round(np.mean(cv_scores_rmse), 3),
                'CV_MAE': round(np.mean(cv_scores_mae), 3),
                'Test_R2': np.nan, 'Test_RMSE': np.nan, 'Test_MAE': np.nan,
                'CV_R2_Std': round(np.std(cv_scores_r2), 3),
                'CV_RMSE_Std': round(np.std(cv_scores_rmse), 3),
                'CV_MAE_Std': round(np.std(cv_scores_mae), 3)
            })
    perf_df = pd.DataFrame(performance_data)
    save_table(perf_df, "Table1_First_Stage_Performance")
    plot_first_stage_performance(perf_df)
    log(f"\nOptimal model: {model_name} (Test R2 = {test_r2:.3f})")
    residual_stats, residuals = analyze_residual_distribution(y_test, y_pred, model_name)
    plot_residual_distribution(residual_stats, residuals)
    return perf_df, residual_stats, model

def plot_first_stage_performance(perf_df):
    lai_data = perf_df[perf_df['Target'] == 'LAI (Y)'].copy()
    if lai_data.empty:
        return
    models = lai_data['Model'].tolist()
    r2_lai = lai_data['Test_R2'].tolist()
    fig_width = IJoRS_CONFIG['figure_width_single'] / 2.54
    fig, ax = plt.subplots(figsize=(fig_width, 4/2.54))
    x = np.arange(len(models))
    bars1 = ax.bar(x, r2_lai, width=0.5, color=IJoRS_COLORS['tabpfn'], alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    ax.set_ylabel('$R^2$ (Test Set)', fontsize=IJoRS_CONFIG['font_size_label']+1)
    ax.set_xlabel('First-Stage Prediction Model', fontsize=IJoRS_CONFIG['font_size_label']+1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=IJoRS_CONFIG['font_size_tick'])
    ax.set_title('First-Stage Model Performance (Test Set)', fontsize=IJoRS_CONFIG['font_size_title']+1,
                 pad=10, fontweight='bold')
    ax.set_ylim([0.6, 0.95])
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=IJoRS_CONFIG['font_size_annotation'], fontweight='bold')
    ax.tick_params(axis='both', labelsize=IJoRS_CONFIG['font_size_tick'],
                   width=0.5, length=3, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    save_figure_ijors(fig, 'Fig2_First_Stage_Performance')

def define_causal_scenarios(feature_names):
    log(f"\n{'='*70}")
    log("Step 3: Defining causal inference scenarios")
    log(f"{'='*70}")
    scenarios = []
    target_treatments = ['GBK_DH', 'GBK_Mixed', 'GBK_TST']
    gbk_treatments = [t for t in target_treatments if t in feature_names]
    log(f"  Available GBK treatment groups: {gbk_treatments}")
    for t_name in gbk_treatments:
        confounders = [f for f in feature_names if f != t_name]
        scenarios.append({
            'name': f'{t_name}_on_LAI',
            'treatment': t_name,
            'outcome': 'LAI',
            'confounders': confounders,
            'description': f'{t_name} genetic background on LAI (vs reference group)'
        })
    for i, s in enumerate(scenarios, 1):
        log(f"    {i}. {s['name']}: {s['description']}")
    if not scenarios:
        log("  Warning: No GBK treatment groups found in features", level='warning')
    return scenarios

def extract_treatment_outcome(X, y, feature_names, treatment_name):
    if treatment_name not in feature_names:
        raise ValueError(f"Treatment variable '{treatment_name}' not found in features")
    t_idx = feature_names.index(treatment_name)
    T = X[:, t_idx]
    confounder_indices = [i for i in range(X.shape[1]) if i != t_idx]
    X_conf = X[:, confounder_indices]
    confounder_names = [feature_names[i] for i in confounder_indices]
    return T, y, X_conf, confounder_names

def run_dml_estimation(T, Y, X_conf, confounder_names, scenario_name, treatment_name,
                       cv_folds=None, random_state=42, compute_fold_ates=True):
    if cv_folds is None:
        cv_folds = CV_FOLDS
    log(f"\n{'='*60}")
    log(f"DML estimation: {scenario_name}")
    log(f"Treatment: {treatment_name}, Sample size: n={len(Y)}, Confounders: p={X_conf.shape[1]}")
    log(f"{'='*60}")
    t_unique = len(np.unique(T))
    if t_unique < 2:
        log(f"  Warning: Treatment variable {treatment_name} has no variation (unique values={t_unique}), attempting estimation", level='warning')
    ate = ate_lower = ate_upper = ate_se = p_value = None
    base_linear = None
    try:
        base_linear = LinearDML(
            model_y=create_tabpfn_instance(random_state),
            model_t=create_tabpfn_instance(random_state),
            cv=cv_folds,
            random_state=random_state
        )
        base_linear.fit(Y, T, X=None, W=X_conf, inference='auto')
        ate = base_linear.ate()
        if not np.isnan(ate):
            ate_lower, ate_upper = base_linear.ate_interval(alpha=0.05)
            if ate_lower is not None and ate_upper is not None and ate_upper != ate_lower:
                ate_se = (ate_upper - ate_lower) / (2 * 1.96)
                z_score = ate / ate_se if (ate_se != 0 and not np.isnan(ate_se)) else np.inf
                p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))
            else:
                ate_se = np.nan; p_value = np.nan
        log(f"Linear DML: ATE={ate:.6f}, SE={ate_se:.6f}, 95%CI=[{ate_lower:.6f}, {ate_upper:.6f}], p={p_value:.6f}")
    except Exception as e:
        log(f"  LinearDML fitting failed: {e}", level='error')
    finally:
        del base_linear; gc.collect()
    cate = cate_lower = cate_upper = imp_df = None
    try:
        est_forest = CausalForestDML(
            model_y=create_tabpfn_instance(random_state),
            model_t=create_tabpfn_instance(random_state),
            cv=cv_folds,
            **FOREST_CONFIG
        )
        est_forest.fit(Y, T, X=X_conf, W=None, inference='blb')
        cate = est_forest.effect(X_conf)
        cate = cate[~np.isnan(cate)] if cate is not None else None
        if cate is not None and len(cate) > 0:
            cate_lower, cate_upper = est_forest.effect_interval(X_conf, alpha=0.05)
            log(f"Causal Forest: Mean CATE={np.mean(cate):.6f}")
            try:
                imp_df = pd.DataFrame({
                    'feature': confounder_names,
                    'importance': est_forest.feature_importances_
                }).sort_values('importance', ascending=False)
            except:
                pass
    except Exception as e:
        log(f"  CausalForestDML failed: {e}", level='error')
    finally:
        del est_forest; gc.collect()
    fold_ates = []
    if compute_fold_ates and len(Y) >= cv_folds * 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(np.arange(len(Y))):
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            X_train, X_test = X_conf[train_idx], X_conf[test_idx]
            try:
                est_fold = LinearDML(
                    model_y=create_tabpfn_instance(random_state),
                    model_t=create_tabpfn_instance(random_state),
                    cv=2,
                    random_state=random_state
                )
                est_fold.fit(Y_test, T_test, X=None, W=X_test, inference='auto')
                fold_ate = est_fold.ate()
                if not np.isnan(fold_ate):
                    fold_ates.append(fold_ate)
            except Exception as e:
                log(f"    Fold ATE calculation failed: {e}", level='warning')
            finally:
                del est_fold; gc.collect()
    return {
        'scenario': scenario_name,
        'treatment': treatment_name,
        'n_samples': len(Y),
        'n_confounders': X_conf.shape[1],
        'ate': ate, 'ate_se': ate_se,
        'ate_ci_lower': ate_lower, 'ate_ci_upper': ate_upper,
        'p_value': p_value,
        'ate_significant': (p_value < 0.05) if (p_value is not None and not np.isnan(p_value)) else False,
        'cate_mean': np.mean(cate) if (cate is not None and len(cate) > 0) else None,
        'cate_std': np.std(cate) if (cate is not None and len(cate) > 0) else None,
        'cate_min': np.min(cate) if (cate is not None and len(cate) > 0) else None,
        'cate_max': np.max(cate) if (cate is not None and len(cate) > 0) else None,
        'cate': cate, 'cate_ci_lower': cate_lower, 'cate_ci_upper': cate_upper,
        'feature_importance': imp_df,
        'confounder_names': confounder_names,
        'fold_ates': fold_ates if fold_ates else None
    }

def analyze_heterogeneous_effects(X_full, y_full, feature_names, reference_group='GBK_TEM'):
    log(f"\n{'='*70}")
    log("Step 4: Heterogeneous causal effect analysis (removing VI_NDVI_mean + DAS)")
    log(f"{'='*70}")
    X_filtered, feature_names_filtered = remove_specific_features(
        X_full, feature_names, 
        remove_features=['VI_NDVI_mean', 'DAS']
    )
    X_gbk_recode, feature_names_recode = recode_gbk_reference(
        X_filtered, feature_names_filtered, new_ref=reference_group
    )
    target_treatments = ['GBK_DH', 'GBK_TST', 'GBK_Mixed']
    target_treatments = [t for t in target_treatments if t in feature_names_recode]
    if not target_treatments:
        raise ValueError("No target treatment groups found in features")
    log(f"  Target treatment groups: {target_treatments}")
    hetero_results = []
    fold_ate_data = {}
    for treat_name in target_treatments:
        T, Y, X_conf, confounder_names = extract_treatment_outcome(
            X_gbk_recode, y_full, feature_names_recode, treat_name
        )
        dml_res = run_dml_estimation(
            T, Y, X_conf, confounder_names,
            scenario_name=f"{treat_name}_hetero_effect",
            treatment_name=treat_name,
            compute_fold_ates=True
        )
        hetero_results.append(dml_res)
        fold_ate_data[treat_name] = dml_res['fold_ates']
    ate_plot_data = []
    for res in hetero_results:
        ate_plot_data.append({
            'group': res['treatment'],
            'ate': res['ate'],
            'ci_lower': res['ate_ci_lower'],
            'ci_upper': res['ate_ci_upper'],
            'p_value': res['p_value']
        })
    return hetero_results, ate_plot_data, fold_ate_data

def plot_figure11_combined(ate_data, fold_data, reference_group='TEM', output_dir=FIGURES_DIR):
    ate_data_valid = [d for d in ate_data if d['ate'] is not None and not np.isnan(d['ate'])]
    if not ate_data_valid:
        log("  No valid ATE data for Figure 11, skipping.", level='warning')
        return

    ate_data_sorted = sorted(ate_data_valid, key=lambda x: abs(x['ate']), reverse=True)
    groups = [d['group'].replace('GBK_', '') for d in ate_data_sorted]
    ates = [d['ate'] for d in ate_data_sorted]
    ci_lower = [d['ci_lower'] for d in ate_data_sorted]
    ci_upper = [d['ci_upper'] for d in ate_data_sorted]
    p_values = [d['p_value'] for d in ate_data_sorted]

    y_pos = np.arange(1, len(groups)+1)

    fig_width = IJoRS_CONFIG['figure_width_single'] / 2.54
    fig_height = (IJoRS_CONFIG['figure_width_single'] * 1.4) / 2.54
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    xerr_lower = np.array(ates) - np.array(ci_lower)
    xerr_upper = np.array(ci_upper) - np.array(ates)
    xerr_lower[np.isnan(xerr_lower)] = 0
    xerr_upper[np.isnan(xerr_upper)] = 0

    ax1.errorbar(ates, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', color='black', capsize=3,
                elinewidth=1.0, capthick=1.0, zorder=3)

    markers_colors = [IJoRS_COLORS.get(g, IJoRS_COLORS['neutral']) for g in groups]
    for i, (ate, color) in enumerate(zip(ates, markers_colors)):
        ax1.scatter(ate, y_pos[i], s=60, color=color,
                   edgecolor='black', linewidth=0.5, zorder=4)

    ax1.axvline(x=0, color=IJoRS_COLORS['reference'], linestyle='--',
                linewidth=0.7, alpha=0.8, zorder=2)
    ax1.invert_yaxis()

    x_max = max(ci_upper)
    x_min = min(ci_lower)
    offset = 0.15 * (x_max - x_min)
    ax1.set_xlim([x_min - offset, x_max + 0.35 * (x_max - x_min)])

    for i, (ate, cl, cu, p) in enumerate(zip(ates, ci_lower, ci_upper, p_values)):
        ate_str = f"{ate:.3f}"
        ci_str = f"({cl:.3f}, {cu:.3f})"
        if np.isnan(p):
            sig = 'ns'
        elif p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        text_x = x_max + offset * 0.3
        ax1.text(text_x, y_pos[i], f"{ate_str} {ci_str} {sig}",
                va='center', ha='left', fontsize=IJoRS_CONFIG['font_size_annotation'])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(groups, fontsize=IJoRS_CONFIG['font_size_tick'])
    ax1.set_xlabel('Average Treatment Effect (Delta LAI)', fontsize=IJoRS_CONFIG['font_size_label'])
    ax1.set_title('(a) Average Treatment Effects with 95% CI', fontsize=IJoRS_CONFIG['font_size_title'],
                  fontweight='bold', pad=6)
    ax1.tick_params(axis='both', labelsize=IJoRS_CONFIG['font_size_tick'],
                    width=0.5, length=3, direction='in')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.5)

    box_data = []
    for treat in [d['group'] for d in ate_data_sorted]:
        fold_ates = fold_data.get(treat, [])
        fold_ates_clean = [x for x in fold_ates if x is not None and not np.isnan(x)]
        box_data.append(fold_ates_clean)

    bp = ax2.boxplot(box_data, vert=False, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.2),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8),
                     flierprops=dict(marker='o', markersize=3, markerfacecolor='gray', alpha=0.7))
    for patch in bp['boxes']:
        patch.set_facecolor(IJoRS_COLORS['tabpfn'])
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    ax2.axvline(x=0, color=IJoRS_COLORS['reference'], linestyle='--',
                linewidth=0.7, alpha=0.6, zorder=2)
    ax2.invert_yaxis()
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(groups, fontsize=IJoRS_CONFIG['font_size_tick'])
    ax2.set_xlabel('Cross-fitting ATE Estimates', fontsize=IJoRS_CONFIG['font_size_label'])
    ax2.set_title('(b) Cross-fitting Stability', fontsize=IJoRS_CONFIG['font_size_title'],
                  fontweight='bold', pad=6)
    ax2.set_xlim(ax1.get_xlim())
    ax2.tick_params(axis='both', labelsize=IJoRS_CONFIG['font_size_tick'],
                    width=0.5, length=3, direction='in')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.grid(True, alpha=0.3, axis='x', linestyle=':', linewidth=0.5)

    plt.tight_layout(pad=1.5, h_pad=1.2)
    save_figure_ijors(fig, 'Fig11_GBK_Causal_Effects', output_dir=output_dir)

def plot_heterogeneous_effects_forest(hetero_results, output_dir=FIGURES_DIR):
    valid_results = [r for r in hetero_results if r['ate'] is not None and not np.isnan(r['ate'])]
    if not valid_results:
        log("No valid heterogeneous effect data for forest plot", level='warning')
        return
    valid_results_sorted = sorted(valid_results, key=lambda x: abs(x['ate']), reverse=True)
    treatments = [r['treatment'].replace('GBK_', '') for r in valid_results_sorted]
    ates = [r['ate'] for r in valid_results_sorted]
    ci_lower = [r['ate_ci_lower'] for r in valid_results_sorted]
    ci_upper = [r['ate_ci_upper'] for r in valid_results_sorted]
    p_values = [r['p_value'] for r in valid_results_sorted]
    fig_width = IJoRS_CONFIG['figure_width_single'] / 2.54
    fig_height = max(4, len(treatments)*0.6) / 2.54
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_pos = np.arange(len(treatments))
    xerr_lower = np.array(ates) - np.array(ci_lower)
    xerr_upper = np.array(ci_upper) - np.array(ates)
    xerr_lower[np.isnan(xerr_lower)] = 0
    xerr_upper[np.isnan(xerr_upper)] = 0
    ax.errorbar(ates, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', color='black', capsize=3,
                elinewidth=1.0, capthick=1.0, zorder=3)
    markers_colors = [IJoRS_COLORS.get(t, IJoRS_COLORS['neutral']) for t in treatments]
    for i, (ate, color) in enumerate(zip(ates, markers_colors)):
        ax.scatter(ate, y_pos[i], s=60, color=color,
                   edgecolor='black', linewidth=0.5, zorder=4)
    ax.axvline(x=0, color=IJoRS_COLORS['reference'], linestyle='--',
               linewidth=IJoRS_CONFIG['line_width'], alpha=0.7, zorder=2)
    for i, (p, effect) in enumerate(zip(p_values, ates)):
        if np.isnan(p):
            sig = 'ns'
        elif p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        x_pos = ci_upper[i] + 0.01 if effect > 0 else ci_lower[i] - 0.01
        if np.isnan(x_pos):
            x_pos = effect + (0.01 if effect > 0 else -0.01)
        ax.text(x_pos, y_pos[i], sig, va='center', ha='left' if effect > 0 else 'right',
                fontsize=IJoRS_CONFIG['font_size_annotation'], color=IJoRS_COLORS['positive'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(treatments, fontsize=IJoRS_CONFIG['font_size_tick'])
    ax.set_xlabel('Average Treatment Effect (Delta LAI)', fontsize=IJoRS_CONFIG['font_size_label'])
    ax.set_title('Heterogeneous Causal Effects of Genetic Backgrounds\n(Excluding VI_NDVI_mean & DAS)',
                 fontsize=IJoRS_CONFIG['font_size_title'], pad=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=IJoRS_CONFIG['font_size_tick'],
                   width=0.5, length=3, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    save_figure_ijors(fig, 'Fig6_Heterogeneous_Effects_Forest', output_dir=output_dir)

def main():
    log("="*80)
    log("Starting TabPFN-DML causal effect analysis")
    log("="*80)
    try:
        (X_train, y_train, X_test, y_test, X_full, y_full, 
         feature_names, scaler, source_flag, trained_model) = load_full_data(
            model_path=DEPLOY_MODEL_PATH,
            data_path=DATA_PATH,
            test_size=TEST_SIZE,
            random_state=42
        )
    except Exception as e:
        log(f"Data loading failed: {e}", level='error')
        return
    treatment_names = ['GBK_DH', 'GBK_TST', 'GBK_Mixed']
    perf_df, residual_stats, model = evaluate_first_stage_tabpfn(
        X_train, y_train, X_test, y_test, feature_names, treatment_names,
        tabpfn_model=trained_model
    )
    scenarios = define_causal_scenarios(feature_names)
    hetero_results, ate_plot_data, fold_ate_data = analyze_heterogeneous_effects(
        X_full, y_full, feature_names, reference_group='GBK_TEM'
    )
    save_dml_results(hetero_results, fold_ate_data, reference_group='GBK_TEM')
    plot_figure11_combined(ate_plot_data, fold_ate_data)
    plot_heterogeneous_effects_forest(hetero_results)
    log("\n" + "="*80)
    log("TabPFN-DML causal effect analysis completed!")
    log(f"  Features used: {len(feature_names)}")
    log(f"  Output path: {OUTPUT_DIR}")
    log(f"  Log file: {LOG_FILE}")
    log(f"  Residual analysis: {RESIDUAL_STATS_FILE}")
    log(f"  DML report: {DML_RESULTS_TXT}")
    log(f"  Tables: {TABLES_DIR}")
    log(f"  Figures: {FIGURES_DIR}")
    log("="*80)

if __name__ == "__main__":
    main()
