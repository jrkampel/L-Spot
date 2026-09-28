import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('./runs/charts', exist_ok=True)

GRAY   = '#888780'
TEAL   = '#1D9E75'
PURPLE = '#7F77DD'
AMBER  = '#BA7517'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'figure.dpi': 150,
})

# ── Chart 1: F1 Progression ───────────────────────────────────────────────────
labels = [
    'Baseline\n(YOLOv8m)',
    'Rule B\npost-proc',
    'TTA+conf\n+Rule B',
    'Standard\n∩ TTA',
    '3-way\nensemble'
]
f1_values = [0.615, 0.636, 0.653, 0.675, 0.733]
bar_colors = [GRAY, TEAL, TEAL, TEAL, PURPLE]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, f1_values, color=bar_colors, width=0.5, zorder=3)
ax.set_ylim(0.55, 0.80)
ax.set_ylabel('F1 score', fontsize=11, color=GRAY)
ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

for bar, val in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='500')

legend_patches = [
    mpatches.Patch(color=GRAY,   label='Baseline'),
    mpatches.Patch(color=TEAL,   label='Improvement'),
    mpatches.Patch(color=PURPLE, label='Best result'),
]
ax.legend(handles=legend_patches, loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('./runs/charts/chart1_f1_progression.png',
            bbox_inches='tight')
plt.close()
print("Chart 1 saved.")

# ── Chart 2: Precision vs Recall scatter ──────────────────────────────────────
approaches = [
    ('Baseline\n(YOLOv8m)',        0.566, 0.674, GRAY,   'o'),
    ('Rule B\npost-proc',           0.632, 0.639, TEAL,   'o'),
    ('TTA+conf\n+Rule B',           0.655, 0.651, TEAL,   '^'),
    ('Standard\n∩ TTA',             0.729, 0.628, TEAL,   'D'),
    ('3-way ensemble\n(≥2 of 3)',  0.820, 0.664, PURPLE, 'D'),
    ('3-way ensemble\n(all 3)',    0.897, 0.525, GRAY,   'D'),
]

fig, ax = plt.subplots(figsize=(9, 7))

for name, prec, rec, col, marker in approaches:
    ax.scatter(rec, prec, color=col, marker=marker, s=90, zorder=3, linewidths=0)
    ax.annotate(name, (rec, prec),
                xytext=(rec + 0.005, prec + 0.008),
                fontsize=8, color='#444441',
                ha='left', va='bottom')

f1_levels = [0.55, 0.60, 0.65, 0.70, 0.73]
r_range = np.linspace(0.35, 0.95, 200)
for f1 in f1_levels:
    p_range = f1 * r_range / (2 * r_range - f1)
    mask = (p_range > 0) & (p_range <= 1.0)
    ax.plot(r_range[mask], p_range[mask], color='#d0d0d0',
            linewidth=0.8, linestyle='--', zorder=0)
    idx = np.where(mask)[0]
    if len(idx) > 10:
        xi = r_range[idx[len(idx)//2]]
        yi = p_range[idx[len(idx)//2]]
        ax.text(xi, yi, f'F1={f1}', fontsize=7.5, color='#aaaaaa',
                ha='center', va='bottom')

ax.set_xlabel('Recall', fontsize=11, color=GRAY)
ax.set_ylabel('Precision', fontsize=11, color=GRAY)
ax.set_xlim(0.35, 1.0)
ax.set_ylim(0.35, 1.0)
ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
ax.xaxis.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color=GRAY,   label='Baseline / no gain'),
    mpatches.Patch(color=TEAL,   label='Improvement'),
    mpatches.Patch(color=PURPLE, label='Best result'),
]
ax.legend(handles=legend_patches, loc='lower left', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('./runs/charts/chart2_precision_recall.png',
            bbox_inches='tight')
plt.close()
print("Chart 2 saved.")
print("\nAll charts saved to runs/charts/")
