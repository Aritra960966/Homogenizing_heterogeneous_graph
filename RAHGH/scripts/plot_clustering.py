"""
Plot clustering results for RAHGH.

Generates:
  1. bar_chart_metrics       -- NMI / ARI / ACC  per dataset with error bars
  2. box_plot_metrics        -- per-dataset distribution of NMI, ARI, ACC
  3. training_curves         -- reconstruction loss over epochs (all seeds)
  4. relation_importance     -- learned alpha weights per seed (heatmap)
  5. nmi_vs_ari_scatter      -- NMI vs ARI scatter across datasets
  6. cv_heatmap              -- hyper-param sensitivity (K x d) on CV NMI

Usage:
    python scripts/plot_clustering.py
    python scripts/plot_clustering.py --datasets dblp acm imdb
    python scripts/plot_clustering.py --plots bar box curves
"""

import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results' / 'clustering'
PLOT_DIR     = RESULTS_ROOT / 'plots'
COLORS       = {'dblp': '#4c72b0', 'acm': '#dd8452', 'imdb': '#55a868'}
FMT_MAP      = {'nmi': 'NMI', 'ari': 'ARI', 'acc': 'ACC'}


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {PLOT_DIR}")


# ─────────────────────────────────────────────
#  1.  Bar chart -- NMI / ARI / ACC per dataset
# ─────────────────────────────────────────────

def plot_bar_chart_metrics(datasets):
    rows = []
    for ds in datasets:
        sp = RESULTS_ROOT / ds / 'summary.csv'
        if not sp.exists():
            print(f"  [skip] {ds}: no summary.csv")
            continue
        df = pd.read_csv(sp)
        for m in ['nmi', 'ari', 'acc']:
            rows.append({
                'dataset': ds.upper(),
                'metric': FMT_MAP[m],
                'mean': df[f'{m}_mean'].values[0],
                'sd': df[f'{m}_sd'].values[0],
            })
    if not rows:
        print("  No summary data found. Run clustering experiments first.")
        return

    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=pdf, x='dataset', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i, bar in enumerate(ax.patches):
        row = pdf.iloc[i % len(pdf)]
        sd = row['sd']
        if sd > 0:
            bar_center = bar.get_x() + bar.get_width() / 2
            ax.errorbar(bar_center, bar.get_height(), yerr=sd,
                        fmt='none', ecolor='0.15', capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}' if x <= 1 else f'{x:.2f}'))
    fig.tight_layout()
    path = PLOT_DIR / 'bar_chart_metrics.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────
#  2.  Box plot -- per-dataset metric distribution
# ─────────────────────────────────────────────

def plot_box_plot_metrics(datasets):
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp)
        for m in ['nmi', 'ari', 'acc']:
            vals = df[m].dropna().values
            for v in vals:
                rows.append({'dataset': ds.upper(), 'metric': FMT_MAP[m], 'value': v})
    if not rows:
        print("  No per_run_results data found.")
        return

    pdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (mname, grp) in zip(axes, pdf.groupby('metric')):
        sns.boxplot(data=grp, x='dataset', y='value', ax=ax,
                    palette='muted', width=0.5, linewidth=1.2,
                    flierprops=dict(marker='o', markersize=5))
        sns.stripplot(data=grp, x='dataset', y='value', ax=ax,
                      color='0.2', size=4, alpha=0.5, jitter=0.08)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}' if x <= 1 else f'{x:.2f}'))
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = PLOT_DIR / 'box_plot_metrics.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────
#  3.  Training curves (recon loss over epochs)
# ─────────────────────────────────────────────

def plot_training_curves(datasets):
    for ds in datasets:
        ep_dir = RESULTS_ROOT / ds / 'epoch_logs'
        if not ep_dir.exists():
            print(f"  [skip] {ds}: no epoch_logs/")
            continue

        csvs = sorted(glob.glob(str(ep_dir / 'seed*_epochs.csv')))
        if not csvs:
            print(f"  [skip] {ds}: no epoch CSVs")
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        all_losses = {}
        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            col = [c for c in df.columns if 'loss' in c.lower()]
            if not col:
                continue
            col = col[0]
            for _, r in df.iterrows():
                ep = int(r['epoch'])
                all_losses.setdefault(ep, []).append(r[col])
            lbl = Path(csv_path).stem.replace('_epochs', '')
            ax.plot(df['epoch'], df[col], alpha=0.25, linewidth=0.6,
                    color=COLORS.get(ds, '#4c72b0'))

        if all_losses:
            eps_sorted = sorted(all_losses.keys())
            mean_vals = [np.mean(all_losses[e]) for e in eps_sorted]
            ax.plot(eps_sorted, mean_vals, color='black', linewidth=1.8,
                    label='Mean')
            lower = [np.percentile(all_losses[e], 25) for e in eps_sorted]
            upper = [np.percentile(all_losses[e], 75) for e in eps_sorted]
            ax.fill_between(eps_sorted, lower, upper,
                            alpha=0.15, color='black')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title(f'{ds.upper()} -- Training Curves', fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=False)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
        fig.tight_layout()
        path = PLOT_DIR / f'training_curves_{ds}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"  -> {path}")


# ─────────────────────────────────────────────
#  4.  Relation importance (alpha weights)
# ─────────────────────────────────────────────

def plot_relation_importance(datasets):
    for ds in datasets:
        ap = RESULTS_ROOT / ds / 'alpha_weights.csv'
        if not ap.exists():
            print(f"  [skip] {ds}: no alpha_weights.csv")
            continue
        df = pd.read_csv(ap)
        alpha_cols = [c for c in df.columns if c.startswith('alpha_')]
        if not alpha_cols:
            continue

        rel_names = [c.replace('alpha_', '') for c in alpha_cols]
        n_seeds = len(df)
        data_mat = df[alpha_cols].values.T

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10 + 0.4 * len(rel_names), 3.5 + 0.25 * n_seeds),
            gridspec_kw={'width_ratios': [3, 1]})

        sns.heatmap(data_mat, ax=ax1, cmap='YlOrRd', annot=True,
                    fmt='.3f', linewidths=0.5,
                    xticklabels=[f'seed{s}' for s in df['seed']],
                    yticklabels=rel_names,
                    vmin=0, vmax=1, cbar_kws={'label': 'a weight'})
        ax1.set_title(f'{ds.upper()} -- a per Seed', fontweight='bold')
        ax1.set_xlabel('Seed')
        ax1.set_ylabel('Relation')
        ax1.xaxis.set_ticks_position('top')
        ax1.xaxis.set_label_position('top')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='left')

        means = data_mat.mean(axis=1)
        stds = data_mat.std(axis=1)
        ax2.barh(range(len(rel_names)), means, xerr=stds,
                 color=[COLORS.get(ds, '#4c72b0')], capsize=4,
                 edgecolor='0.2', linewidth=0.8)
        ax2.set_yticks(range(len(rel_names)))
        ax2.set_yticklabels(rel_names)
        ax2.set_xlabel('Mean a')
        ax2.set_title('Mean +/- SD', fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1.05)

        fig.tight_layout()
        path = PLOT_DIR / f'relation_importance_{ds}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"  -> {path}")

    all_dfs = []
    for ds in datasets:
        ap = RESULTS_ROOT / ds / 'alpha_weights.csv'
        if ap.exists():
            df = pd.read_csv(ap)
            for m in [c for c in df.columns if c.startswith('alpha_')]:
                all_dfs.append({
                    'dataset': ds.upper(),
                    'relation': m.replace('alpha_', ''),
                    'alpha': df[m].mean(),
                })
    if all_dfs:
        pdf = pd.DataFrame(all_dfs)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=pdf, x='relation', y='alpha', hue='dataset',
                    ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
        ax.set_ylabel('Mean a weight')
        ax.set_xlabel('Relation')
        ax.legend(frameon=True, fancybox=False)
        ax.set_title('Relation Importance Across Datasets', fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}' if x <= 1 else f'{x:.2f}'))
        fig.tight_layout()
        path = PLOT_DIR / 'relation_importance_all.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"  -> {path}")


# ─────────────────────────────────────────────
#  5.  NMI vs ARI scatter
# ─────────────────────────────────────────────

def plot_nmi_vs_ari_scatter(datasets):
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp)
        for _, r in df.iterrows():
            rows.append({
                'dataset': ds.upper(),
                'nmi': r['nmi'],
                'ari': r['ari'],
                'acc': r['acc'],
            })
    if not rows:
        print("  No per_run_results data for scatter.")
        return

    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for ds_upper, grp in pdf.groupby('dataset'):
        ax.scatter(grp['nmi'], grp['ari'], s=60, label=ds_upper,
                   c=COLORS.get(ds_upper.lower(), '#4c72b0'),
                   edgecolors='0.2', linewidth=0.8, alpha=0.85, zorder=3)
        if len(grp) >= 3:
            from matplotlib.patches import Ellipse
            cx, cy = grp['nmi'].mean(), grp['ari'].mean()
            w = 2 * grp['nmi'].std()
            h = 2 * grp['ari'].std()
            ellipse = Ellipse((cx, cy), w, h, alpha=0.12,
                              color=COLORS.get(ds_upper.lower(), '#4c72b0'),
                              linewidth=0)
            ax.add_patch(ellipse)

    ax.plot([0, 1], [0, 1], '--', color='0.6', linewidth=0.8, zorder=0)
    ax.set_xlabel('NMI')
    ax.set_ylabel('ARI')
    ax.set_title('NMI vs ARI (per seed)', fontweight='bold')
    ax.legend(frameon=True, fancybox=False)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}' if x <= 1 else f'{x:.2f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}' if x <= 1 else f'{x:.2f}'))
    fig.tight_layout()
    path = PLOT_DIR / 'nmi_vs_ari_scatter.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────
#  6.  CV hyper-parameter sensitivity heatmap
# ─────────────────────────────────────────────

def plot_cv_heatmap(datasets):
    for ds in datasets:
        cvp = RESULTS_ROOT / ds / 'cv_fold_scores.csv'
        if not cvp.exists():
            print(f"  [skip] {ds}: no cv_fold_scores.csv")
            continue
        df = pd.read_csv(cvp)
        if 'hp_K' not in df.columns or 'hp_d' not in df.columns:
            print(f"  [skip] {ds}: cv_fold_scores missing hp_K / hp_d")
            continue
        pivot = df.groupby(['hp_K', 'hp_d'])['val_nmi'].mean().reset_index()
        pv = pivot.pivot_table(index='hp_K', columns='hp_d', values='val_nmi')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(pv, ax=ax, annot=True, fmt='.3f', cmap='YlGnBu',
                    linewidths=0.5, cbar_kws={'label': 'Mean val NMI'},
                    vmin=0, vmax=1)
        ax.set_title(f'{ds.upper()} -- CV NMI (K x d)', fontweight='bold')
        ax.set_xlabel('Hidden dim (d)')
        ax.set_ylabel('Diffusion depth (K)')
        fig.tight_layout()
        path = PLOT_DIR / f'cv_heatmap_{ds}.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"  -> {path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot clustering results')
    parser.add_argument('--datasets', nargs='+', default=['dblp', 'acm', 'imdb'],
                        choices=['dblp', 'acm', 'imdb'],
                        help='Datasets to plot')
    parser.add_argument('--plots', nargs='+',
                        default=['bar', 'box', 'curves', 'alpha', 'scatter', 'cv'],
                        choices=['bar', 'box', 'curves', 'alpha', 'scatter', 'cv'],
                        help='Which plots to generate')
    args = parser.parse_args()

    _ensure_plot_dir()

    plot_fn = {
        'bar':     ('Bar chart -- metrics per dataset', plot_bar_chart_metrics),
        'box':     ('Box plot -- metric distributions', plot_box_plot_metrics),
        'curves':  ('Training curves', plot_training_curves),
        'alpha':   ('Relation importance (a)', plot_relation_importance),
        'scatter': ('NMI vs ARI scatter', plot_nmi_vs_ari_scatter),
        'cv':      ('CV hyper-param heatmap', plot_cv_heatmap),
    }

    for key in args.plots:
        name, fn = plot_fn[key]
        print(f"\n--- {name} ---")
        fn(args.datasets)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == '__main__':
    main()
