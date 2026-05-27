# -*- coding: utf-8 -*-
"""
Observed vs predicted LAI scatter plots for three observation dates.
Accuracy metrics (R2, RMSE, MAE) with 1:1 reference line.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['axes.unicode_minus'] = False

LABEL_SIZE = 8
TICK_SIZE = 7
TITLE_SIZE = 9

COLORS = ['#009E73', '#D55E00', '#0072B2']

file_path = r"YOUR_DATA_PATH"
df = pd.read_csv(file_path, encoding='gbk')

data = df[['date', 'LAI_Observed', 'LAI_Predicted']].dropna()
dates = sorted(data['date'].unique())

accuracy_results = {}

width_cm = 17.5
width_inch = width_cm * 0.3937
height_inch = width_inch / 3.0 * 1.15

fig, axes = plt.subplots(1, 3, figsize=(width_inch, height_inch), dpi=300)
fig.subplots_adjust(wspace=0.35, bottom=0.18, top=0.88)

print("="*50)
print("LAI Prediction Accuracy by Date")
print("="*50)

for idx, (ax, date) in enumerate(zip(axes, dates)):
    sub = data[data['date'] == date]
    obs = sub['LAI_Observed'].values
    pred = sub['LAI_Predicted'].values

    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)

    accuracy_results[date] = {"R2": r2, "RMSE": rmse, "MAE": mae}

    print(f"
Date: {date}")
    print(f"  R2    = {r2:.3f}")
    print(f"  RMSE  = {rmse:.3f}")
    print(f"  MAE   = {mae:.3f}")

    ax.scatter(obs, pred, c=COLORS[idx % len(COLORS)], alpha=0.7, edgecolors='w', linewidth=0.3, s=25)

    lims = [min(obs.min(), pred.min()) * 0.95, max(obs.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.6)

    text_str = f"$R^2$ = {r2:.3f}
RMSE = {rmse:.3f}
MAE = {mae:.3f}"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            fontsize=LABEL_SIZE, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Observed LAI', fontsize=LABEL_SIZE)
    ax.set_ylabel('Predicted LAI', fontsize=LABEL_SIZE)
    ax.set_title(f'{date}', fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

print("
" + "="*50)
print("Summary Table")
print("="*50)
summary_df = pd.DataFrame(accuracy_results).T
summary_df = summary_df.round(3)
print(summary_df.rename(columns={"R2": "R2"}))

save_path = r"YOUR_OUTPUT_PATH"
plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}.eps", format='eps', dpi=300, bbox_inches='tight')
plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()

summary_df.to_csv(f"{save_path}_accuracy_results.csv", encoding='utf-8-sig', index_label="Date")

print(f"
Results saved: {save_path}_accuracy_results.csv")
