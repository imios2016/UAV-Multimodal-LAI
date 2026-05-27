# -*- coding: utf-8 -*-
"""
Maize LAI temporal dynamics with automatic pairwise significance testing.
Includes histograms, violin plots with boxplots, and grouped bar charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import ttest_ind, f_oneway, shapiro, levene, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

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
fig_height = 14.0 * 0.3937

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

print("="*50)
print("Panel b: Date-wise LAI difference significance test")
print("="*50)

group_data = {date: df[df['date'] == date]['LAI_Predicted'].values for date in date_order}
groups = list(group_data.values())
group_names = list(group_data.keys())

print("
1. Normality test (Shapiro-Wilk):")
normality_results = {}
for date, data in group_data.items():
    stat, p = shapiro(data)
    normality_results[date] = p > 0.05
    print(f"   {date}: W={stat:.4f}, p={p:.4f} {'Normal' if p>0.05 else 'Non-normal'}")

stat, p_levene = levene(*groups)
print(f"
2. Homogeneity (Levene): F={stat:.4f}, p={p_levene:.4f} {'Homogeneous' if p_levene>0.05 else 'Heterogeneous'}")

all_normal = all(normality_results.values())
if all_normal and p_levene > 0.05:
    print("
3. Overall test: One-way ANOVA")
    f_stat, p_overall = f_oneway(*groups)
    print(f"   F={f_stat:.4f}, p={p_overall:.4e}")
    test_method = "ANOVA + Tukey HSD"
else:
    print("
3. Overall test: Kruskal-Wallis")
    h_stat, p_overall = kruskal(*groups)
    print(f"   H={h_stat:.4f}, p={p_overall:.4e}")
    test_method = "Kruskal-Wallis + Dunn"

if p_overall < 0.001:
    print("   Highly significant overall difference (***)")
elif p_overall < 0.01:
    print("   Significant overall difference (**)")
elif p_overall < 0.05:
    print("   Significant overall difference (*)")
else:
    print("   No significant overall difference")

print(f"
4. Post-hoc: {test_method}")
sig_results = {}

if test_method == "ANOVA + Tukey HSD":
    tukey = pairwise_tukeyhsd(endog=df['LAI_Predicted'], groups=df['date'], alpha=0.05)
    print(tukey)
    for row in tukey.summary().data[1:]:
        g1, g2, _, pval, _, _, reject = row
        key = tuple(sorted([g1, g2]))
        if pval < 0.001:
            sig_results[key] = '***'
        elif pval < 0.01:
            sig_results[key] = '**'
        elif pval < 0.05:
            sig_results[key] = '*'
        else:
            sig_results[key] = 'ns'
else:
    dunn_results = posthoc_dunn(df, val_col='LAI_Predicted', group_col='date', p_adjust='bonferroni')
    print(dunn_results)
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j:
                pval = dunn_results.loc[g1, g2]
                key = tuple(sorted([g1, g2]))
                if pval < 0.001:
                    sig_results[key] = '***'
                elif pval < 0.01:
                    sig_results[key] = '**'
                elif pval < 0.05:
                    sig_results[key] = '*'
                else:
                    sig_results[key] = 'ns'

print("
5. Pairwise significance summary:")
for (g1, g2), sig in sig_results.items():
    print(f"   {g1} vs {g2}: {sig}")
print("="*50)

def add_significance(ax, x1, x2, y, text, bar_offset=0):
    x1_pos = x1 + bar_offset
    x2_pos = x2 + bar_offset
    ax.plot([x1_pos, x1_pos, x2_pos, x2_pos], [y, y+0.05, y+0.05, y], 
            color='black', linewidth=1.0)
    ax.text((x1_pos + x2_pos)/2, y+0.07, text, ha='center', 
            fontsize=7, fontweight='bold')

fig = plt.figure(figsize=(fig_width, fig_height))
main_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.3)

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

group_max = [data.max() for data in groups]
base_y = max(group_max) + 0.15
y_step = 0.12

comparison_pairs = [(0, 1), (0, 2), (1, 2)]

for i, (x1, x2) in enumerate(comparison_pairs):
    g1 = date_order[x1]
    g2 = date_order[x2]
    key = tuple(sorted([g1, g2]))
    sig = sig_results[key]

    if sig != 'ns':
        y = base_y + i * y_step
        add_significance(ax_b, x1, x2, y, sig)

ax_b.set_ylim(0.5, base_y + len(comparison_pairs)*y_step + 0.1)
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
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Figure7_Maize_LAI_Dynamics_IJoRS.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('Figure7_Maize_LAI_Dynamics_IJoRS.eps', 
            dpi=300, bbox_inches='tight', format='eps', backend='ps')

plt.show()

print("
" + "="*50)
print("Done! IJoRS-compliant files generated")
print("Panel b: automatic pairwise significance testing completed")
print("="*50)
