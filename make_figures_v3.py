"""
Figure generation for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
The 4-panel figure is designed to show the full picture of how rehearsal
ratio r affects the three proxy metrics during Stage 2 training:

  Panel 1 (F_proxy vs epoch): Does the world model's generative quality
  improve, plateau, or degrade during mixed training? Lower is better
  (negative ELBO).

  Panel 2 (w_proxy vs epoch): Does the linear separability of A vs B
  in latent space increase with training? Higher = more separable = the
  model has developed distinct representations for each domain.

  Panel 3 (dmn_proxy vs epoch): Does the GRU's free-running imagination
  shift toward Environment B over training? Values near 0 = imagines A,
  near 1 = imagines B, near 0.5 = balanced.

  Panel 4 (final-epoch w_proxy and dmn_proxy vs r): The key summary plot.
  If w and dmn move together across r, the model's representational
  structure and its default-mode behavior are coupled. If they diverge,
  these are independent phenomena — which would be the more interesting
  scientific finding.

This script reads ONLY from saved CSVs and makes NO random calls of any
kind, so the figure is fully deterministic given the same input data.

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


# ═══════════════════════════════════════════════════════════════════════
# Color palette for r values
# ═══════════════════════════════════════════════════════════════════════

# Perceptually distinct colors keyed to r values, chosen for readability
# on both light and dark backgrounds and for colorblind accessibility.
R_COLORS = {
    0.0:  '#e63946',   # red — no rehearsal
    0.20: '#f4a261',   # amber
    0.30: '#e9c46a',   # gold
    0.40: '#2a9d8f',   # teal
    0.50: '#264653',   # dark blue-gray
    1.0:  '#6a4c93',   # purple — full rehearsal (no new domain)
}

# Fallback for r values not in the dictionary
DEFAULT_CMAP = plt.cm.viridis


def get_color(r_val):
    """Get a color for a given r value."""
    if r_val in R_COLORS:
        return R_COLORS[r_val]
    # Normalize r to [0, 1] for the colormap
    return DEFAULT_CMAP(r_val)


def plot_metric_vs_epoch(ax, df, metric, ylabel, title):
    """
    Plot a metric vs epoch, one line per r value.

    Shows mean across seeds as a solid line with a ±1 std shaded band.
    """
    for r_val in sorted(df['r'].unique()):
        subset = df[df['r'] == r_val]
        grouped = subset.groupby('epoch')[metric].agg(['mean', 'std'])

        epochs = grouped.index.values
        mean = grouped['mean'].values
        std = grouped['std'].values
        # Handle NaN std (e.g., single seed in debug mode)
        std = np.nan_to_num(std, nan=0.0)

        color = get_color(r_val)
        ax.plot(epochs, mean, color=color, label=f'r={r_val:.2f}',
                linewidth=1.8, marker='o', markersize=3)
        ax.fill_between(epochs, mean - std, mean + std,
                         color=color, alpha=0.15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def plot_final_vs_r(ax, summary):
    """
    Plot final-epoch w_proxy and dmn_proxy against r, with error bars.

    This is the key summary panel: if the two lines track each other,
    representational separability and default-mode behavior are coupled.
    """
    r_vals = summary['r'].values

    ax.errorbar(r_vals, summary['w_proxy_mean'], yerr=summary['w_proxy_std'],
                color='#2a9d8f', marker='s', markersize=6, linewidth=2,
                capsize=4, label='w_proxy (probe accuracy)')
    ax.errorbar(r_vals, summary['dmn_proxy_mean'], yerr=summary['dmn_proxy_std'],
                color='#e63946', marker='^', markersize=6, linewidth=2,
                capsize=4, label='dmn_proxy (P(env=B))')

    ax.set_xlabel('Rehearsal ratio (r)')
    ax.set_ylabel('Metric value')
    ax.set_title('Final-epoch metrics vs rehearsal ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the v3 experiment."
    )
    parser.add_argument('--debug', action='store_true',
                        help="Use debug-mode output.")
    args = parser.parse_args()

    # Load data
    all_runs_path = os.path.join(cfg.RESULTS_DIR, "all_runs_v3.csv")
    summary_path = os.path.join(cfg.RESULTS_DIR, "summary_by_r_v3.csv")

    if not os.path.exists(all_runs_path):
        raise FileNotFoundError(
            f"{all_runs_path} not found. Run analyze_results_v3.py first."
        )

    df = pd.read_csv(all_runs_path)
    summary = pd.read_csv(summary_path)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'v3 Continual Learning Dominance-Shift Experiment',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Panel 1: F_proxy vs epoch
    plot_metric_vs_epoch(
        axes[0, 0], df, 'F_proxy',
        ylabel='F_proxy (val ELBO)',
        title='F_proxy: Generative quality over training',
    )

    # Panel 2: w_proxy vs epoch
    plot_metric_vs_epoch(
        axes[0, 1], df, 'w_proxy',
        ylabel='w_proxy (probe accuracy)',
        title='w_proxy: Domain separability over training',
    )

    # Panel 3: dmn_proxy vs epoch
    plot_metric_vs_epoch(
        axes[1, 0], df, 'dmn_proxy',
        ylabel='dmn_proxy (P(env=B))',
        title='dmn_proxy: Default-mode behavior over training',
    )

    # Panel 4: Final-epoch w_proxy and dmn_proxy vs r
    plot_final_vs_r(axes[1, 1], summary)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(cfg.FIGURES_DIR, "v3_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
