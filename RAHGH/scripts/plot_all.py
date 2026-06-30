"""
Comprehensive plotting for all RAHGH results.

Generates for each task/dataset with available results:

  NC:  bar_chart_nc_metrics  — macro_f1 / micro_f1 / accuracy with error bars
       box_plot_nc           — per-seed metric distribution
       training_curves_nc    — loss + train_acc over epochs (all seeds)
       cv_heatmap_nc         — hyper-param sensitivity (K x d) on val_macro

  LP:  bar_chart_lp_metrics  — AUC / AP / Macro-F1 / Hits@K with error bars
       box_plot_lp           — per-seed metric distribution

  CL:  bar_chart_cl_metrics  — NMI / ARI / ACC with error bars
       box_plot_cl           — per-seed metric distribution
       training_curves_cl    — recon loss over epochs
       relation_importance   — learned alpha weights per seed

  REC: bar_chart_rec_metrics — Recall@K / NDCG@K / Hit@K with error bars
       box_plot_rec          — per-seed metric distribution
       training_curves_rec   — loss over epochs

Usage:
    python scripts/plot_all.py                          # all available plots
    python scripts/plot_all.py --tasks nc cl            # specific tasks
    python scripts/plot_all.py --datasets imdb          # specific datasets
    python scripts/plot_all.py --plots bar curves       # specific plot types
    python scripts/plot_all.py --output plots           # custom output dir
"""

import argparse, os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'
OUT_DIR      = RESULTS_ROOT / 'plots'

COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b2', '#937860', '#da8bc3', '#8c8c8c']

FMT_NC  = {'macro_f1': 'Macro-F1', 'micro_f1': 'Micro-F1', 'accuracy': 'Accuracy', 'auc': 'AUC'}
FMT_LP  = {'auc': 'AUC', 'ap': 'AP', 'macro_f1': 'Macro-F1'}
FMT_CL  = {'nmi': 'NMI', 'ari': 'ARI', 'acc': 'ACC'}
FMT_REC = {}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _ls_results(task_dir, task=None):
    """Return list of dataset names that have at least some result."""
    if not task_dir.exists():
        return []
    results = []
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        has_any = (d / 'summary.csv').exists() or (d / 'alpha_weights.csv').exists()
        if has_any:
            results.append(d.name)
    return sorted(results)


def _detect_task(dataset_path):
    """Auto-detect task type from summary.csv columns."""
    sp = dataset_path / 'summary.csv'
    if not sp.exists():
        return None
    df = pd.read_csv(sp, on_bad_lines='skip')
    cols = df.columns.tolist()
    if 'task' in cols:
        return df['task'].values[0]
    if any('macro_f1' in c for c in cols):
        if any('hits@' in c for c in cols):
            return 'lp'
        return 'nc'
    if any(m in cols for m in ['nmi_mean', 'ari_mean', 'acc_mean']):
        return 'cl'
    if any('recall@' in c for c in cols):
        return 'rec'
    return None


def _human_fmt(val, _):
    return f'{val:.0%}' if val <= 1 else f'{val:.2f}'


def _read_epoch_metrics(dataset_path, task):
    """Read all epoch_metrics_seed*.csv files."""
    if task == 'nc':
        pattern = str(dataset_path / 'epoch_metrics_seed*.csv')
    elif task == 'cl':
        pattern = str(dataset_path / 'epoch_logs' / 'seed*_epochs.csv')
    else:
        return {}
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        return {}
    all_data = defaultdict(list)
    for cp in csvs:
        df = pd.read_csv(cp)
        for _, row in df.iterrows():
            ep = int(row['epoch'])
            for col in df.columns:
                if col != 'epoch':
                    all_data[col].append((ep, row[col]))
    return all_data


# ════════════════════════════════════════════
#  NC Plots
# ════════════════════════════════════════════

def plot_nc_bar(datasets):
    rows = []
    for ds in datasets:
        sp = RESULTS_ROOT / 'nc' / ds / 'summary.csv'
        if not sp.exists():
            continue
        df = pd.read_csv(sp, on_bad_lines='skip')
        for m, label in FMT_NC.items():
            mean_col = f'{m}_mean'
            sd_col   = f'{m}_sd'
            if mean_col in df.columns:
                rows.append({
                    'dataset': ds.upper(), 'metric': label,
                    'mean': df[mean_col].values[0], 'sd': df[sd_col].values[0],
                })
    if not rows:
        print("  No NC summary data.")
        return
    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=pdf, x='dataset', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i in range(len(ax.patches)):
        row_i = pdf.iloc[i % len(pdf)]
        if row_i['sd'] > 0:
            bar = ax.patches[i]
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=row_i['sd'], fmt='none', ecolor='0.15',
                        capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
    fig.tight_layout()
    path = OUT_DIR / 'nc_bar_chart_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_nc_box(datasets):
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / 'nc' / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp, on_bad_lines='skip')
        for m, label in FMT_NC.items():
            col = f'test_{m}'
            if col not in df.columns:
                col = m
            if col in df.columns:
                for v in df[col].dropna().values:
                    rows.append({'dataset': ds.upper(), 'metric': label, 'value': v})
    if not rows:
        print("  No NC per-run data.")
        return
    pdf = pd.DataFrame(rows)
    metrics = pdf['metric'].unique()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4), sharey=True)
    if n_metrics == 1:
        axes = [axes]
    for ax, (mname, grp) in zip(axes, pdf.groupby('metric')):
        sns.boxplot(data=grp, x='dataset', y='value', ax=ax,
                    palette='muted', width=0.5, linewidth=1.2,
                    flierprops=dict(marker='o', markersize=5))
        sns.stripplot(data=grp, x='dataset', y='value', ax=ax,
                      color='0.2', size=4, alpha=0.5, jitter=0.08)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = OUT_DIR / 'nc_box_plot_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_nc_curves(datasets):
    for ds in datasets:
        dsp = RESULTS_ROOT / 'nc' / ds
        csvs = sorted(glob.glob(str(dsp / 'epoch_metrics_seed*.csv')))
        if not csvs:
            print(f"  [skip] {ds}: no epoch CSVs")
            continue

        all_data = defaultdict(lambda: defaultdict(list))
        for cp in csvs:
            df = pd.read_csv(cp)
            for _, row in df.iterrows():
                ep = int(row['epoch'])
                for col in df.columns:
                    if col != 'epoch':
                        all_data[col][ep].append(row[col])

        n_cols = len(all_data)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
        if n_cols == 1:
            axes = [axes]

        color = COLORS[0]
        for ax, (col, ep_data) in zip(axes, sorted(all_data.items())):
            eps_sorted = sorted(ep_data.keys())
            vals = [ep_data[e] for e in eps_sorted]
            for v in vals:
                pass

            for cp in csvs:
                df = pd.read_csv(cp)
                if col not in df.columns:
                    continue
                ax.plot(df['epoch'], df[col], alpha=0.15, linewidth=0.5, color=color)

            means = [np.mean(ep_data[e]) for e in eps_sorted]
            ax.plot(eps_sorted, means, color='black', linewidth=1.8, label='Mean')
            lower = [np.percentile(ep_data[e], 25) for e in eps_sorted]
            upper = [np.percentile(ep_data[e], 75) for e in eps_sorted]
            ax.fill_between(eps_sorted, lower, upper, alpha=0.12, color='black')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.set_title(f'{ds.upper()} — {col.replace("_", " ").title()}', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))

        fig.tight_layout()
        path = OUT_DIR / f'nc_training_curves_{ds}.png'
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")


def plot_nc_cv_heatmap(datasets):
    for ds in datasets:
        cvp = RESULTS_ROOT / 'nc' / ds / 'cv_fold_scores.csv'
        if not cvp.exists():
            print(f"  [skip] {ds}: no cv_fold_scores.csv")
            continue
        df = pd.read_csv(cvp, on_bad_lines='skip')
        val_col = [c for c in df.columns if 'val' in c.lower() and c != 'val_nmi']
        if not val_col:
            val_col = [c for c in df.columns if c.startswith('val_')]
        if not val_col:
            val_col = ['val_macro'] if 'val_macro' in df.columns else None
        if not val_col:
            print(f"  [skip] {ds}: no val metric column found")
            continue
        val_col = val_col[0]

        if 'hp_K' not in df.columns or 'hp_d' not in df.columns:
            print(f"  [skip] {ds}: missing hp_K / hp_d")
            continue

        pivot = df.groupby(['hp_K', 'hp_d'])[val_col].mean().reset_index()
        pv = pivot.pivot_table(index='hp_K', columns='hp_d', values=val_col)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(pv, ax=ax, annot=True, fmt='.3f', cmap='YlGnBu',
                    linewidths=0.5, cbar_kws={'label': f'Mean {val_col}'},
                    vmin=0, vmax=1)
        ax.set_title(f'{ds.upper()} — CV {val_col} (K x d)', fontweight='bold')
        ax.set_xlabel('Hidden dim (d)')
        ax.set_ylabel('Diffusion depth (K)')
        fig.tight_layout()
        path = OUT_DIR / f'nc_cv_heatmap_{ds}.png'
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")


# ════════════════════════════════════════════
#  LP Plots
# ════════════════════════════════════════════

def plot_lp_bar(datasets):
    rows = []
    for ds in datasets:
        sp = RESULTS_ROOT / 'lp' / ds / 'summary.csv'
        if not sp.exists():
            continue
        df = pd.read_csv(sp, on_bad_lines='skip')
        for m, label in FMT_LP.items():
            mean_col = f'{m}_mean'
            sd_col   = f'{m}_sd'
            if mean_col in df.columns:
                rows.append({
                    'dataset': ds.upper(), 'metric': label,
                    'mean': df[mean_col].values[0], 'sd': df[sd_col].values[0],
                })
        for col in df.columns:
            m = re.match(r'^hits@(\d+)_mean$', col)
            if m:
                k = m.group(1)
                sd_col = f'hits@{k}_sd'
                rows.append({
                    'dataset': ds.upper(), 'metric': f'Hits@{k}',
                    'mean': df[col].values[0],
                    'sd': df[sd_col].values[0] if sd_col in df.columns else 0,
                })
    if not rows:
        print("  No LP summary data.")
        return
    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=pdf, x='dataset', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i in range(len(ax.patches)):
        row_i = pdf.iloc[i % len(pdf)]
        if row_i['sd'] > 0:
            bar = ax.patches[i]
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=row_i['sd'], fmt='none', ecolor='0.15',
                        capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
    fig.tight_layout()
    path = OUT_DIR / 'lp_bar_chart_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_lp_box(datasets):
    metric_cols = ['test_auc', 'test_ap', 'test_macro_f1']
    metric_labels = ['AUC', 'AP', 'Macro-F1']
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / 'lp' / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp, on_bad_lines='skip')
        for col, label in zip(metric_cols, metric_labels):
            if col in df.columns:
                for v in df[col].dropna().values:
                    rows.append({'dataset': ds.upper(), 'metric': label, 'value': v})
        for col in df.columns:
            m = re.match(r'^hits@(\d+)$', col)
            if m:
                for v in df[col].dropna().values:
                    rows.append({'dataset': ds.upper(), 'metric': f'Hits@{m.group(1)}', 'value': v})
    if not rows:
        print("  No LP per-run data.")
        return
    pdf = pd.DataFrame(rows)
    metrics = pdf['metric'].unique()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (mname, grp) in zip(axes, pdf.groupby('metric')):
        sns.boxplot(data=grp, x='dataset', y='value', ax=ax,
                    palette='muted', width=0.5, linewidth=1.2)
        sns.stripplot(data=grp, x='dataset', y='value', ax=ax,
                      color='0.2', size=4, alpha=0.5, jitter=0.08)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = OUT_DIR / 'lp_box_plot_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ════════════════════════════════════════════
#  Clustering Plots
# ════════════════════════════════════════════

def plot_cl_bar(datasets):
    rows = []
    for ds in datasets:
        sp = RESULTS_ROOT / 'clustering' / ds / 'summary.csv'
        if not sp.exists():
            continue
        df = pd.read_csv(sp, on_bad_lines='skip')
        for m, label in FMT_CL.items():
            mean_col = f'{m}_mean'
            sd_col   = f'{m}_sd'
            if mean_col in df.columns:
                rows.append({
                    'dataset': ds.upper(), 'metric': label,
                    'mean': df[mean_col].values[0], 'sd': df[sd_col].values[0],
                })
    if not rows:
        print("  No CL summary data.")
        return
    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=pdf, x='dataset', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i in range(len(ax.patches)):
        row_i = pdf.iloc[i % len(pdf)]
        if row_i['sd'] > 0:
            bar = ax.patches[i]
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=row_i['sd'], fmt='none', ecolor='0.15',
                        capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
    fig.tight_layout()
    path = OUT_DIR / 'cl_bar_chart_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_cl_box(datasets):
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / 'clustering' / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp, on_bad_lines='skip')
        for m, label in FMT_CL.items():
            if m in df.columns:
                for v in df[m].dropna().values:
                    rows.append({'dataset': ds.upper(), 'metric': label, 'value': v})
    if not rows:
        print("  No CL per-run data.")
        return
    pdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (mname, grp) in zip(axes, pdf.groupby('metric')):
        sns.boxplot(data=grp, x='dataset', y='value', ax=ax,
                    palette='muted', width=0.5, linewidth=1.2)
        sns.stripplot(data=grp, x='dataset', y='value', ax=ax,
                      color='0.2', size=4, alpha=0.5, jitter=0.08)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = OUT_DIR / 'cl_box_plot_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_cl_curves(datasets):
    for ds in datasets:
        ep_dir = RESULTS_ROOT / 'clustering' / ds / 'epoch_logs'
        if not ep_dir.exists():
            print(f"  [skip] {ds}: no epoch_logs/")
            continue
        csvs = sorted(glob.glob(str(ep_dir / 'seed*_epochs.csv')))
        if not csvs:
            print(f"  [skip] {ds}: no epoch CSVs")
            continue

        all_losses = defaultdict(list)
        for cp in csvs:
            df = pd.read_csv(cp)
            col = [c for c in df.columns if 'loss' in c.lower()]
            if not col:
                continue
            col = col[0]
            for _, r in df.iterrows():
                all_losses[int(r['epoch'])].append(r[col])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        color = COLORS[0]
        for cp in csvs:
            df = pd.read_csv(cp)
            col = [c for c in df.columns if 'loss' in c.lower()][0]
            ax.plot(df['epoch'], df[col], alpha=0.2, linewidth=0.6, color=color)

        eps_sorted = sorted(all_losses.keys())
        means = [np.mean(all_losses[e]) for e in eps_sorted]
        ax.plot(eps_sorted, means, color='black', linewidth=1.8, label='Mean')
        lower = [np.percentile(all_losses[e], 25) for e in eps_sorted]
        upper = [np.percentile(all_losses[e], 75) for e in eps_sorted]
        ax.fill_between(eps_sorted, lower, upper, alpha=0.15, color='black')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title(f'{ds.upper()} — Training Curves', fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=False)
        fig.tight_layout()
        path = OUT_DIR / f'cl_training_curves_{ds}.png'
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")


def plot_cl_alpha(datasets):
    for ds in datasets:
        ap = RESULTS_ROOT / 'clustering' / ds / 'alpha_weights.csv'
        if not ap.exists():
            print(f"  [skip] {ds}: no alpha_weights.csv")
            continue
        df = pd.read_csv(ap)
        alpha_cols = [c for c in df.columns if c.startswith('alpha_')]
        if not alpha_cols:
            print(f"  [skip] {ds}: no alpha columns")
            continue

        rel_names = [c.replace('alpha_', '') for c in alpha_cols]
        n_seeds = len(df)
        n_rels = len(alpha_cols)
        data_mat = df[alpha_cols].values.T

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10 + 0.4 * n_rels, 3.5 + 0.25 * n_seeds),
            gridspec_kw={'width_ratios': [3, 1]})

        seed_labels = [f'seed{s}' for s in df.index]
        sns.heatmap(data_mat, ax=ax1, cmap='YlOrRd', annot=True,
                    fmt='.3f', linewidths=0.5,
                    xticklabels=seed_labels,
                    yticklabels=rel_names,
                    vmin=0, vmax=1, cbar_kws={'label': 'a weight'})
        ax1.set_title(f'{ds.upper()} — a per Seed', fontweight='bold')
        ax1.set_xlabel('Seed')
        ax1.set_ylabel('Relation')
        ax1.xaxis.set_ticks_position('top')
        ax1.xaxis.set_label_position('top')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='left')

        means = data_mat.mean(axis=1)
        stds = data_mat.std(axis=1)
        ax2.barh(range(n_rels), means, xerr=stds,
                 color=COLORS[0], capsize=4,
                 edgecolor='0.2', linewidth=0.8)
        ax2.set_yticks(range(n_rels))
        ax2.set_yticklabels(rel_names)
        ax2.set_xlabel('Mean a')
        ax2.set_title('Mean +/- SD', fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1.05)

        fig.tight_layout()
        path = OUT_DIR / f'cl_relation_importance_{ds}.png'
        fig.savefig(path); plt.close(fig)
        print(f"  -> {path}")


# ════════════════════════════════════════════
#  Recommendation Plots
# ════════════════════════════════════════════

def plot_rec_bar(datasets):
    rows = []
    for ds in datasets:
        sp = RESULTS_ROOT / 'recommendation' / ds / 'summary.csv'
        if not sp.exists():
            continue
        df = pd.read_csv(sp, on_bad_lines='skip')
        for col in df.columns:
            m = re.match(r'^(recall|ndcg|hit|precision|mrr)@(\d+)_mean$', col)
            if m:
                metric_name = f"{m.group(1).title()}@{m.group(2)}"
                sd_col = f"{m.group(1)}@{m.group(2)}_sd"
                rows.append({
                    'dataset': ds.upper(), 'metric': metric_name,
                    'mean': df[col].values[0],
                    'sd': df[sd_col].values[0] if sd_col in df.columns else 0,
                })
    if not rows:
        print("  No REC summary data.")
        return
    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=pdf, x='dataset', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i in range(len(ax.patches)):
        row_i = pdf.iloc[i % len(pdf)]
        if row_i['sd'] > 0:
            bar = ax.patches[i]
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=row_i['sd'], fmt='none', ecolor='0.15',
                        capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False, ncol=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
    fig.tight_layout()
    path = OUT_DIR / 'rec_bar_chart_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


def plot_rec_box(datasets):
    rows = []
    for ds in datasets:
        pp = RESULTS_ROOT / 'recommendation' / ds / 'per_run_results.csv'
        if not pp.exists():
            continue
        df = pd.read_csv(pp, on_bad_lines='skip')
        for col in df.columns:
            m = re.match(r'^(recall|ndcg|hit|precision|mrr)@(\d+)$', col)
            if m:
                label = f"{m.group(1).title()}@{m.group(2)}"
                for v in df[col].dropna().values:
                    rows.append({'dataset': ds.upper(), 'metric': label, 'value': v})
    if not rows:
        print("  No REC per-run data.")
        return
    pdf = pd.DataFrame(rows)
    metrics = pdf['metric'].unique()
    n = len(metrics)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (mname, grp) in zip(axes, pdf.groupby('metric')):
        sns.boxplot(data=grp, x='dataset', y='value', ax=ax,
                    palette='muted', width=0.5, linewidth=1.2)
        sns.stripplot(data=grp, x='dataset', y='value', ax=ax,
                      color='0.2', size=4, alpha=0.5, jitter=0.08)
        ax.set_title(mname, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
        ax.set_ylim(0, 1.05)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout()
    path = OUT_DIR / 'rec_box_plot_metrics.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ════════════════════════════════════════════
#  Combined cross-task comparison
# ════════════════════════════════════════════

def plot_all_summary_table():
    """Build a combined figure comparing best metrics across all tasks."""
    all_rows = []
    for task in ['nc', 'lp', 'cl', 'rec']:
        task_dir = RESULTS_ROOT / TASK_DIR_MAP[task]
        for ds in _ls_results(task_dir, task=task):
            sp = task_dir / ds / 'summary.csv'
            if not sp.exists():
                continue
            df = pd.read_csv(sp, on_bad_lines='skip')
            task_label = {'nc': 'NC', 'lp': 'LP', 'cl': 'CL', 'rec': 'REC'}[task]
            if task == 'nc':
                for m in ['macro_f1', 'micro_f1', 'accuracy', 'auc']:
                    mc, sc = f'{m}_mean', f'{m}_sd'
                    if mc in df.columns:
                        all_rows.append({'task': f'{task_label}/{ds.upper()}', 'metric': m,
                                         'mean': df[mc].values[0], 'sd': df[sc].values[0]})
            elif task == 'lp':
                for m in ['auc', 'ap']:
                    mc, sc = f'{m}_mean', f'{m}_sd'
                    if mc in df.columns:
                        all_rows.append({'task': f'{task_label}/{ds.upper()}', 'metric': m,
                                         'mean': df[mc].values[0], 'sd': df[sc].values[0]})
                for col in df.columns:
                    mm = re.match(r'^hits@(\d+)_mean$', col)
                    if mm:
                        sc = f'hits@{mm.group(1)}_sd'
                        all_rows.append({'task': f'{task_label}/{ds.upper()}', 'metric': f'hits@{mm.group(1)}',
                                         'mean': df[col].values[0], 'sd': df[sc].values[0]})
            elif task == 'cl':
                for m in ['nmi', 'ari', 'acc']:
                    mc, sc = f'{m}_mean', f'{m}_sd'
                    if mc in df.columns:
                        all_rows.append({'task': f'{task_label}/{ds.upper()}', 'metric': m,
                                         'mean': df[mc].values[0], 'sd': df[sc].values[0]})
            elif task == 'rec':
                for col in df.columns:
                    mm = re.match(r'^(recall|ndcg|hit)@(\d+)_mean$', col)
                    if mm:
                        sc = f"{mm.group(1)}@{mm.group(2)}_sd"
                        all_rows.append({'task': f'{task_label}/{ds.upper()}',
                                         'metric': f"{mm.group(1)}@{mm.group(2)}",
                                         'mean': df[col].values[0], 'sd': df[sc].values[0]})
    if not all_rows:
        print("  No summary data for combined table.")
        return

    pdf = pd.DataFrame(all_rows)
    fig, ax = plt.subplots(figsize=(max(8, len(pdf['task'].unique()) * 1.5), 5))
    sns.barplot(data=pdf, x='task', y='mean', hue='metric',
                ax=ax, palette='muted', edgecolor='0.2', linewidth=0.8)
    for i in range(len(ax.patches)):
        row_i = pdf.iloc[i % len(pdf)]
        if row_i['sd'] > 0:
            bar = ax.patches[i]
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        yerr=row_i['sd'], fmt='none', ecolor='0.15',
                        capsize=3, capthick=1.2)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Metric', frameon=True, fancybox=False, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_human_fmt))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    path = OUT_DIR / 'all_tasks_summary.png'
    fig.savefig(path); plt.close(fig)
    print(f"  -> {path}")


# ════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════

TASK_DIR_MAP = {
    'nc':  'nc',
    'lp':  'lp',
    'cl':  'clustering',
    'rec': 'recommendation',
}

AVAILABLE_PLOTS = {
    'nc': {
        'bar':     ('NC bar chart', plot_nc_bar),
        'box':     ('NC box plot', plot_nc_box),
        'curves':  ('NC training curves', plot_nc_curves),
        'cv':      ('NC CV heatmap', plot_nc_cv_heatmap),
    },
    'lp': {
        'bar':     ('LP bar chart', plot_lp_bar),
        'box':     ('LP box plot', plot_lp_box),
    },
    'cl': {
        'bar':     ('CL bar chart', plot_cl_bar),
        'box':     ('CL box plot', plot_cl_box),
        'curves':  ('CL training curves', plot_cl_curves),
        'alpha':   ('CL relation importance', plot_cl_alpha),
    },
    'rec': {
        'bar':     ('REC bar chart', plot_rec_bar),
        'box':     ('REC box plot', plot_rec_box),
    },
}


def main():
    parser = argparse.ArgumentParser(description='Plot all RAHGH results')
    parser.add_argument('--tasks', nargs='+', default=['nc', 'lp', 'cl', 'rec'],
                        choices=['nc', 'lp', 'cl', 'rec'],
                        help='Tasks to plot (default: all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets (default: auto-detect from results)')
    parser.add_argument('--plots', nargs='+', default=None,
                        help='Plot types: bar, box, curves, cv, alpha (default: all)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/plots)')
    args = parser.parse_args()

    global OUT_DIR
    if args.output:
        OUT_DIR = Path(args.output)
    _ensure_dir(OUT_DIR)
    print(f"Output directory: {OUT_DIR}")

    # Auto-detect datasets if not specified
    user_datasets = args.datasets

    for task in args.tasks:
        task_plots = AVAILABLE_PLOTS[task]
        task_dir = RESULTS_ROOT / TASK_DIR_MAP[task]
        available_ds = _ls_results(task_dir, task=task)
        if not available_ds:
            print(f"\n=== {task.upper()} — no results found ===")
            continue

        datasets = user_datasets if user_datasets else available_ds
        datasets = [d for d in datasets if d in available_ds]
        if not datasets:
            continue

        print(f"\n{'='*60}")
        print(f"  {task.upper()} — datasets: {datasets}")
        print(f"{'='*60}")

        plot_keys = args.plots if args.plots else list(task_plots.keys())
        for key in plot_keys:
            if key not in task_plots:
                continue
            name, fn = task_plots[key]
            print(f"\n--- {name} ---")
            fn(datasets)

    # Combined summary across all tasks
    print(f"\n{'='*60}")
    print("  Combined summary across all tasks")
    print(f"{'='*60}")
    plot_all_summary_table()

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
