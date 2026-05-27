# -*- coding: utf-8 -*-
"""
Maize LAI temporal dynamics and genetic background difference visualization.
Histograms, violin plots, and grouped bar charts compliant with IJoRS standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

colors = {
    '2017-06-29': '#009E73',
    '2017-07-11': '#D55E00',
    '2017-07-28': '#0072B2'
}
gbk_order = ['DH', 'Mixed', 'TEM', 'TST']
date_order = ['2017-06-29', '2017-07-11', '2017-07-28']

double_column_cm = 17.5
fig_width = double_column_cm * 0.3937
fig_height = 13.0 * 0.3937

df = pd.read_csv(r'YOUR_DATA_PATH')

print("="*50)
print("Data Overview")
print("="*50)
print(f"Total samples: {len(df)}")
print(f"Dates: {df['date'].unique().tolist()}")
print(f"Genetic backgrounds: {df['GBK'].unique().tolist()}")
print(f"Missing values:
{df[['label', 'GBK', 'date', 'LAI_Predicted']].isnull().sum()}
")

df['date'] = df['date'].astype(str).apply(
    lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}" if len(x) == 8 else x
)

df = df[df['date'].isin(date_order)].reset_index(drop=True)
df = df.dropna(subset=['LAI_Predicted']).reset_index(drop=True)
print(f"After cleaning: {len(df)}
")

date_stats = df.groupby('date')['LAI_Predicted'].agg(['mean', 'median', 'std']).reindex(date_order)
gbk_date_stats = df.groupby(['GBK', 'date'])['LAI_Predicted'].agg(['mean', 'std']).reset_index()

fig = plt.figure(figsize=(fig_width, fig_height))
main_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)

gs_top = main_gs[0].subgridspec(1, 3, wspace=0.25)
ax_a1 = fig.add_subplot(gs_top[0, 0])
ax_a2 = fig.add_subplot(gs_top[0, 1])
ax_a3 = fig.add_subplot(gs_top[0, 2])
ax_a = [ax_a1, ax_a2, ax_a3]

gs_bottom = main_gs[1].subgridspec(1, 2, wspace=0.25)
ax_b = fig.add_subplot(gs_bottom[0, 0])
ax_c = fig.add_subplot(gs_bottom[0, 1])

for i, date in enumerate(date_order):
    ax = ax_a[i]
    data = df[df['date'] == date]['LAI_Predicted']
    mean_val = date_stats.loc[date, 'mean']
    median_val = date_stats.loc[date, 'median']

    sns.histplot(
        data=data, 
        bins=30, 
        kde=False, 
        color=colors[date], 
        ax=ax,
        edgecolor='white',
        linewidth=0.5
    )

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.0, label=f'mean: {mean_val:.3f}')
    ax.axvline(median_val, color='black', linestyle='-', linewidth=1.0, label=f'median: {median_val:.3f}')

    ax.set_xlim(0.5, 2.7)
    ax.set_xlabel('Predicted LAI', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(date, fontsize=9, fontweight='bold')

fig.text(0.5, 0.95, 'a  LAI Distribution Across Observation Dates', 
         ha='center', fontsize=9, fontweight='bold')

sns.violinplot(
    data=df, 
    x='date', 
    y='LAI_Predicted', 
    order=date_order,
    palette=colors,
    inner=None,
    linewidth=1.0,
    scale='width',
    ax=ax_b
)

sns.boxplot(
    data=df,
    x='date',
    y='LAI_Predicted',
    order=date_order,
    width=0.3,
    color='white',
    linewidth=1.0,
    boxprops={'color': 'black', 'zorder': 3},
    whiskerprops={'color': 'black', 'zorder': 3},
    capprops={'color': 'black', 'zorder': 3},
    medianprops={'color': 'white', 'linewidth': 1.0, 'zorder': 4},
    showfliers=False,
    ax=ax_b
)

ax_b.set_ylim(0.5, 2.7)
ax_b.set_xlabel('Observation Date', fontsize=8)
ax_b.set_ylabel('Predicted LAI', fontsize=8)
ax_b.set_title('b  LAI Distribution Across Observation Dates', 
               fontsize=9, fontweight='bold', pad=15)

bar_width = 0.25
x = np.arange(len(gbk_order))

for i, date in enumerate(date_order):
    means = []
    stds = []
    for gbk in gbk_order:
        stats = gbk_date_stats[(gbk_date_stats['GBK'] == gbk) & (gbk_date_stats['date'] == date)]
        means.append(stats['mean'].values[0])
        stds.append(stats['std'].values[0])

    ax_c.bar(
        x + i*bar_width, 
        means, 
        width=bar_width, 
        color=colors[date],
        yerr=stds,
        capsize=2,
        edgecolor='black',
        linewidth=0.8,
        label=date
    )

    for j, mean_val in enumerate(means):
        ax_c.text(
            x[j] + i*bar_width, 
            mean_val + 0.05, 
            f'{mean_val:.2f}', 
            ha='center', 
            fontsize=7,
            fontweight='bold'
        )

def add_significance(ax, x1, x2, y, text, bar_offset=0):
    x1_pos = x1 + bar_offset
    x2_pos = x2 + bar_offset
    ax.plot([x1_pos, x1_pos, x2_pos, x2_pos], [y, y+0.05, y+0.05, y], 
            color='black', linewidth=1.0)
    ax.text((x1_pos + x2_pos)/2, y+0.07, text, ha='center', 
            fontsize=7, fontweight='bold')

ax_c.set_xticks(x + bar_width)
ax_c.set_xticklabels(gbk_order)
ax_c.set_xlabel('Genetic Background', fontsize=8)
ax_c.set_ylabel('Predicted LAI', fontsize=8)
ax_c.set_ylim(0.5, 2.7)
ax_c.legend(fontsize=7, loc='upper right')
ax_c.set_title('c  LAI Comparison Across Genetic Background', 
               fontsize=9, fontweight='bold', pad=15)

plt.tight_layout()

plt.savefig('Figure7_Maize_LAI_Dynamics_IJoRS.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('Figure7_Maize_LAI_Dynamics_IJoRS.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('Figure7_Maize_LAI_Dynamics_IJoRS.eps', 
            dpi=300, bbox_inches='tight', format='eps', backend='ps')

plt.show()

print("="*50)
print("Done! Files generated (IJoRS compliant):")
print("1. Figure7_Maize_LAI_Dynamics_IJoRS.png (300 DPI)")
print("2. Figure7_Maize_LAI_Dynamics_IJoRS.pdf (vector)")
print("3. Figure7_Maize_LAI_Dynamics_IJoRS.eps (vector)")
print("="*50)
