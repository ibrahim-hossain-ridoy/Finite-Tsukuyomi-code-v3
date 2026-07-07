"""
Figure generation for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
This script produces a 6-panel figure from the live data pipeline. All
values are read from the CSV outputs of analyze_results_v3.py and from
the raw .npz simulation files — nothing is hardcoded.

  Top row (training dynamics, from raw .npz via load_all_runs):
    Panel 1: F_proxy vs epoch — generative quality during mixed training.
    Panel 2: w_proxy vs epoch — domain separability in latent space.
    Panel 3: dmn_proxy vs epoch — default-mode imagination shift.

  Bottom row (summary statistics, from CSVs):
    Panel 4: Final-epoch w_proxy and dmn_proxy vs r — do the two
             proxies move together or diverge across the sweep?
    Panel 5: Dose-response — mean epoch to dmn >= 0.9 vs r.
    Panel 6: Censoring rate — fraction of seeds that never crossed
             the 0.9 threshold within the training window.

If the CSVs change (new seeds, different parameters), the figure
updates automatically. No random calls of any kind.

Usage:
    python make_figures_v3.py          # Full sweep figure
    python make_figures_v3.py --debug  # Debug output figure
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_v3 as cfg
from analyze_results_v3 import load_all_runs


# ═══════════════════════════════════════════════════════════════════════
# Color palette for r values
# ═══════════════════════════════════════════════════════════════════════

# Perceptually distinct colors keyed to r values, chosen for readability
# on both light and dark backgrounds and for colorblind accessibility.
R_COLORS = {
    0.0:  '#e63946',
    0.20: '#f4a261',
    0.30: '#e9c46a',
    0.40: '#2a9d8f',
    0.50: '#264653',
    1.0:  '#6a4c93',
}

DEFAULT_CMAP = plt.cm.viridis


def get_color(r_val):
    """Get a color for a given r value, with fallback to viridis."""
    if r_val in R_COLORS:
        return R_COLORS[r_val]
    return DEFAULT_CMAP(r_val)


# ═══════════════════════════════════════════════════════════════════════
# Top row: training curves (from raw .npz data)
# ═══════════════════════════════════════════════════════════════════════

def plot_metric_vs_epoch(ax, df, metric, ylabel, title):
    """
    Plot a metric vs epoch, one line per r value.
    Shows mean across seeds as a solid line with a +/-1 std shaded band.
    """
    for r_val in sorted(df['r'].unique()):
        subset = df[df['r'] == r_val]
        grouped = subset.groupby('epoch')[metric].agg(['mean', 'std'])

        epochs = grouped.index.values
        mean = grouped['mean'].values
        std = np.nan_to_num(grouped['std'].values, nan=0.0)

        color = get_color(r_val)
        ax.plot(epochs, mean, color=color, label=f'r={r_val:.2f}',
                linewidth=1.8, marker='o', markersize=3)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════
# Bottom-left: final-epoch metrics vs r (from summary_by_r_v3.csv)
# ═══════════════════════════════════════════════════════════════════════

def plot_final_vs_r(ax, summary):
    """
    Plot final-epoch w_proxy and dmn_proxy against r, with error bars.
    If the two lines track each other, representational separability
    and default-mode behavior are coupled.
    """
    r_vals = summary['r'].values

    ax.errorbar(r_vals, summary['w_proxy_mean'],
                yerr=summary['w_proxy_std'],
                color='#2a9d8f', marker='s', markersize=6, linewidth=2,
                capsize=4, label='w_proxy (probe accuracy)')
    ax.errorbar(r_vals, summary['dmn_proxy_mean'],
                yerr=summary['dmn_proxy_std'],
                color='#e63946', marker='^', markersize=6, linewidth=2,
                capsize=4, label='dmn_proxy (P(env=B))')

    ax.set_xlabel('Rehearsal ratio (r)')
    ax.set_ylabel('Metric value')
    ax.set_title('Final-epoch metrics vs r')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


# ═══════════════════════════════════════════════════════════════════════
# Bottom-center: dose-response (from epoch_to_threshold_v3.csv)
# ═══════════════════════════════════════════════════════════════════════

def plot_dose_response(ax, threshold_df):
    """
    Plot mean epoch to dmn >= 0.9 vs rehearsal ratio, with error bars.
    Annotates each point with n_crossed / n_total. Conditions where no
    seed ever crossed are marked separately.
    """
    grouped = threshold_df.groupby('r').agg(
        mean_epoch=('epoch_to_dmn_0.9', 'mean'),
        std_epoch=('epoch_to_dmn_0.9', 'std'),
        n_crossed=('epoch_to_dmn_0.9', 'count'),
        n_total=('seed', 'count'),
    ).reset_index()

    # Separate conditions that crossed from those that never did
    has_crossings = grouped[grouped['n_crossed'] > 0]
    no_crossings = grouped[grouped['n_crossed'] == 0]

    # Plot conditions where at least one seed crossed
    if not has_crossings.empty:
        std_vals = np.nan_to_num(has_crossings['std_epoch'].values, nan=0.0)
        ax.errorbar(
            has_crossings['r'], has_crossings['mean_epoch'],
            yerr=std_vals,
            fmt='-o', linewidth=2.5, elinewidth=2, capsize=5,
            color='#1f77b4', zorder=4,
        )

        for _, row in has_crossings.iterrows():
            n_c = int(row['n_crossed'])
            n_t = int(row['n_total'])
            ax.annotate(
                f"n={n_c}/{n_t}",
                (row['r'], row['mean_epoch']),
                textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=8, color='#555555',
            )

    # Mark conditions with zero crossings
    for _, row in no_crossings.iterrows():
        n_t = int(row['n_total'])
        ax.scatter(row['r'], 0, color='red', marker='x',
                   s=100, linewidths=2.5, zorder=5)
        ax.annotate(
            f"0/{n_t} crossed",
            (row['r'], 0),
            textcoords="offset points",
            xytext=(0, 12), ha='center', fontsize=8,
            color='red', fontweight='bold',
        )

    ax.set_xlabel('Rehearsal ratio (r)')
    ax.set_ylabel('Mean epoch to dmn >= 0.9')
    ax.set_title('Dose-response: retention vs shift speed')
    ax.grid(True, linestyle='--', alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════
# Bottom-right: censoring rate (from epoch_to_threshold_v3.csv)
# ═══════════════════════════════════════════════════════════════════════

def plot_censoring_rate(ax, threshold_df):
    """
    Bar chart of the fraction of seeds that never crossed the 0.9
    threshold within the training window, by rehearsal ratio.
    """
    grouped = threshold_df.groupby('r').agg(
        n_never=('never_crossed', 'sum'),
        n_total=('seed', 'count'),
    ).reset_index()
    grouped['fraction_never_crossed'] = grouped['n_never'] / grouped['n_total']

    bars = ax.bar(
        grouped['r'], grouped['fraction_never_crossed'],
        color='#d62728', width=0.06, edgecolor='black', alpha=0.85,
    )

    for bar, (_, row) in zip(bars, grouped.iterrows()):
        height = bar.get_height()
        n_nc = int(row['n_never'])
        n_t = int(row['n_total'])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.02,
            f"{n_nc}/{n_t}",
            ha='center', va='bottom', fontsize=9,
        )

    ax.set_xlabel('Rehearsal ratio (r)')
    ax.set_ylabel('Fraction never crossed')
    ax.set_title('Censoring rate (dmn < 0.9 for all epochs)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.15)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the v3 experiment."
    )
    parser.add_argument('--debug', action='store_true',
                        help="Use debug-mode output.")
    args = parser.parse_args()

    # ── Load data from the pipeline (no hardcoded values) ──
    summary_path = os.path.join(cfg.RESULTS_DIR, "summary_by_r_v3.csv")
    threshold_path = os.path.join(cfg.RESULTS_DIR, "epoch_to_threshold_v3.csv")

    for path in [summary_path, threshold_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run analyze_results_v3.py first."
            )

    summary_df = pd.read_csv(summary_path)
    threshold_df = pd.read_csv(threshold_path)

    # Per-epoch data for training curves (from raw .npz files)
    epoch_df = load_all_runs(cfg.RAW_RESULTS_DIR)

    # ── Build figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'v3 Continual Learning Dominance-Shift Experiment',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Top row: training curves
    plot_metric_vs_epoch(
        axes[0, 0], epoch_df, 'F_proxy',
        ylabel='F_proxy (val ELBO)',
        title='F_proxy: Generative quality over training',
    )
    plot_metric_vs_epoch(
        axes[0, 1], epoch_df, 'w_proxy',
        ylabel='w_proxy (probe accuracy)',
        title='w_proxy: Domain separability over training',
    )
    plot_metric_vs_epoch(
        axes[0, 2], epoch_df, 'dmn_proxy',
        ylabel='dmn_proxy (P(env=B))',
        title='dmn_proxy: Default-mode behavior over training',
    )

    # Bottom row: summary panels (all from CSVs)
    plot_final_vs_r(axes[1, 0], summary_df)
    plot_dose_response(axes[1, 1], threshold_df)
    plot_censoring_rate(axes[1, 2], threshold_df)

    n_seeds = summary_df['n_seeds'].iloc[0] if 'n_seeds' in summary_df else '?'
    fig.text(
        0.5, 0.01,
        f"n={n_seeds} seeds per condition. "
        f"Error bars/bands = std across seeds. "
        f"All data read from pipeline CSVs.",
        ha='center', fontsize=10, style='italic', color='#555555',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(cfg.FIGURES_DIR, "v3_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()