# -*- coding: utf-8 -*-
"""
TabPFN multi-temporal LAI prediction and evaluation system.
Date-based full-sample prediction with SCI-compliant combined figures.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUTPUT_DIR = r"YOUR_OUTPUT_PATH"
DATA_PATH = r"YOUR_DATA_PATH"
DEPLOY_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_deployment_model.pkl")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "date_based_predictions_FULL")
PLOT_DIR = os.path.join(OUTPUT_DIR, "date_based_plots")
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

IJoRS_COLORS = {'Scatter': '#984EA3', 'Perfect': '#4DAF4A'}
def cm2inch(cm):
    return cm * 0.3937

def explore_full_data(df):
    print("="*50)
    print("[Full Data Report]")
    print("="*50)

    df['date_str'] = df['date'].astype(str).str.strip()
    all_dates = sorted(df['date_str'].unique())
    print(f"1. Dates: {all_dates}")
    print(f"2. Total samples: {len(df)}")
    print(f"3. Samples per date: {len(df) // len(all_dates)}")

    lai_observed_df = df[df['LAI'].notna()].copy()
    print(f"
4. Observed LAI samples: {len(lai_observed_df)}")

    print("
5. Distribution by date:")
    for date, count in df['date_str'].value_counts().sort_index().items():
        obs_count = len(lai_observed_df[lai_observed_df['date_str'] == date])
        print(f"   - {date}: {count} total, {obs_count} observed")

    return df, lai_observed_df, all_dates

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
full_df, lai_observed_df, all_date_list = explore_full_data(df)

final_model = None
scaler = None
selected_features = None
best_algorithm = None

print("
" + "="*50)
print("Loading deployment model...")
try:
    with open(DEPLOY_MODEL_PATH, 'rb') as f:
        deployment = pickle.load(f)
    final_model = deployment['model']
    scaler = deployment['scaler']
    selected_features = deployment['features']
    best_algorithm = deployment['algorithm']
    print(f"Model loaded: {best_algorithm}, {len(selected_features)} features")
except Exception as e:
    print(f"Model load failed: {str(e)}")
    exit()

print("
" + "="*50)
print("Starting date-based prediction...")
accuracy_metrics = []

for date in all_date_list:
    date_full_df = full_df[full_df['date_str'] == date].reset_index(drop=True)
    date_observed_df = lai_observed_df[lai_observed_df['date_str'] == date].copy()

    print(f"
Date: {date}")
    print(f"  Total: {len(date_full_df)}")
    print(f"  Observed: {len(date_observed_df)}")

    X_full = date_full_df[selected_features].values
    y_full_pred = final_model.predict(scaler.transform(X_full))

    result_df = date_full_df[['label', 'date_str']].copy()
    result_df.columns = ['label', 'date']
    result_df['LAI_Observed'] = date_full_df['LAI'].round(4)
    result_df['LAI_Predicted'] = y_full_pred.round(4)

    output_path = os.path.join(PREDICTION_DIR, f"LAI_Prediction_FULL_{date}.csv")
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  Full-sample results saved")

    if len(date_observed_df) > 0:
        X_obs = date_observed_df[selected_features].values
        y_true = date_observed_df['LAI'].values
        y_pred = final_model.predict(scaler.transform(X_obs))

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)

        accuracy_metrics.append({
            'Date': date,
            'Total_Samples': len(date_full_df),
            'Observed_Samples': len(date_observed_df),
            'R2': round(r2, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'Bias': round(bias, 4),
            'Mean_Observed': round(np.mean(y_true), 4),
            'Mean_Predicted': round(np.mean(y_pred), 4)
        })
        print(f"  Accuracy: R2={r2:.4f}, RMSE={rmse:.4f}")

if accuracy_metrics:
    metrics_df = pd.DataFrame(accuracy_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "date_based_accuracy_metrics_FULL.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print("
" + "="*50)
    print("Accuracy summary:")
    print(metrics_df.to_string(index=False))

def plot_scatter_ijors(date, y_true, y_pred, save_dir):
    fig, ax = plt.subplots(figsize=(cm2inch(8.5), cm2inch(8.5)))

    ax.scatter(y_true, y_pred, color=IJoRS_COLORS['Scatter'], 
               alpha=0.6, s=15, edgecolors='black', linewidths=0.3)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 
            color=IJoRS_COLORS['Perfect'], linestyle='--', linewidth=1)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f'{date} (R2={r2:.3f}, RMSE={rmse:.3f})', fontsize=9, fontweight='bold')

    ax.set_xlabel('Observed LAI', fontsize=8)
    ax.set_ylabel('Predicted LAI', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    for fmt in ['png', 'eps', 'pdf']:
        save_path = os.path.join(save_dir, f"LAI_Scatter_{date}.{fmt}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if accuracy_metrics:
    print("
" + "="*50)
    print("Plotting individual scatter plots...")
    for metrics in accuracy_metrics:
        date = metrics['Date']
        date_obs_df = lai_observed_df[lai_observed_df['date_str'] == date]
        y_true = date_obs_df['LAI'].values
        y_pred = final_model.predict(scaler.transform(date_obs_df[selected_features].values))
        plot_scatter_ijors(date, y_true, y_pred, PLOT_DIR)
        print(f"  {date} scatter plot done")

print("
" + "="*50)
print("Generating SCI combined figure...")

fig, axes = plt.subplots(1, 3, figsize=(cm2inch(17.5), cm2inch(6)))
plt.subplots_adjust(wspace=0.35)

for i, metrics in enumerate(accuracy_metrics):
    ax = axes[i]
    date = metrics['Date']
    date_obs_df = lai_observed_df[lai_observed_df['date_str'] == date]
    y_true = date_obs_df['LAI'].values
    y_pred = final_model.predict(scaler.transform(date_obs_df[selected_features].values))

    ax.scatter(y_true, y_pred, color=IJoRS_COLORS['Scatter'], 
               alpha=0.6, s=12, edgecolors='black', linewidths=0.3)
    min_val = min(np.min(y_true), np.min(y_pred)) - 0.1
    max_val = max(np.max(y_true), np.max(y_pred)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 
            color=IJoRS_COLORS['Perfect'], linestyle='--', linewidth=1)

    ax.set_title(
        f'({chr(97+i)}) {date}\n(R2={metrics["R2"]:.3f}, RMSE={metrics["RMSE"]:.3f})',
        fontsize=8, fontweight='bold', pad=8
    )

    ax.set_xlabel('Observed LAI', fontsize=7)
    ax.set_ylabel('Predicted LAI', fontsize=7)
    ax.tick_params(axis='both', labelsize=6, width=1.0)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

for fmt in ['png', 'eps', 'pdf']:
    save_path = os.path.join(PLOT_DIR, f"FigureX_TabPFN_LAI_Prediction_Accuracy.{fmt}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Combined figure {fmt.upper()} saved")

plt.close()
print("Combined figure done")

print("
" + "="*50)
print("Generating report...")

report_content = f"""
TabPFN Maize LAI Date-based Full-sample Prediction Report
====================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Data
   - Total samples: {len(full_df)}
   - Dates: {len(all_date_list)}
   - Samples/date: {len(full_df) // len(all_date_list)}
   - Observed LAI: {len(lai_observed_df)}

2. Model
   - Algorithm: {best_algorithm}
   - Features: {len(selected_features)}
   - Path: {DEPLOY_MODEL_PATH}

3. Accuracy by date
   Date        Total  Observed  R2      RMSE    MAE     Bias
   --------------------------------------------------------
"""
for metrics in accuracy_metrics:
    report_content += (
        f"   {metrics['Date']}    {metrics['Total_Samples']:>6}    {metrics['Observed_Samples']:>6}    "
        f"{metrics['R2']:.4f}    {metrics['RMSE']:.4f}    {metrics['MAE']:.4f}    {metrics['Bias']:.4f}
"
    )

report_content += f"""
4. Outputs
   - Predictions: {PREDICTION_DIR}
   - Metrics: {metrics_path if accuracy_metrics else 'N/A'}
   - Plots: {PLOT_DIR}
   - Combined: {os.path.join(PLOT_DIR, "FigureX_TabPFN_LAI_Prediction_Accuracy.*")}

5. Figure specs
   - Font: Arial (IJoRS)
   - Combined: 17.5cm x 6cm (double column)
   - Single: 8.5cm x 8.5cm (single column)
   - DPI: 300
"""

report_path = os.path.join(OUTPUT_DIR, "date_based_LAI_prediction_report_FULL.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print("
" + "="*80)
print("All tasks completed!")
print(f"SCI combined figure: {PLOT_DIR}")
print("="*80)
