"""
Analysis script for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
This script aggregates raw per-epoch metrics from all experimental runs
into exactly two output files:

  summary_by_r_v3.csv  — final-epoch values aggregated by rehearsal ratio
  r, reporting mean +/- std across seeds. This directly answers the
  research question: how does rehearsal ratio affect the three proxies?

  epoch_to_threshold_v3.csv — per-run tracking of when dmn_proxy first
  crosses the 0.9 threshold, including censored (never-crossed) runs.
  This enables dose-response and survival-style analyses of how
  rehearsal ratio affects the speed of dominance shift.

Additionally, Pearson correlations between w_proxy and dmn_proxy are
printed to stdout as a direct numerical test of proxy coupling.

Usage:
    python analyze_results_v3.py          # Analyze full sweep
    python analyze_results_v3.py --debug  # Analyze debug output
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

import config_v3 as cfg

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DMN_THRESHOLD = 0.9


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_all_runs(raw_dir):
    """
    Load all .npz result files from the raw results directory.

    Filenames follow the pattern seed{N}_r{R}.npz. Each file contains
    arrays: epoch, F_proxy, w_proxy, dmn_proxy.

    Returns a long-form DataFrame with columns:
    [seed, r, epoch, F_proxy, w_proxy, dmn_proxy]
    """
    pattern = os.path.join(raw_dir, "seed*_r*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {raw_dir}. Run the experiment first."
        )

    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = re.match(r"seed(\d+)_r([\d.]+)\.npz", fname)
        if not match:
            print(f"Warning: skipping unexpected filename {fname}")
            continue

        seed = int(match.group(1))
        r = float(match.group(2))

        data = np.load(fpath)
        n_epochs = len(data['epoch'])

        for i in range(n_epochs):
            rows.append({
                'seed': seed,
                'r': r,
                'epoch': int(data['epoch'][i]),
                'F_proxy': float(data['F_proxy'][i]),
                'w_proxy': float(data['w_proxy'][i]),
                'dmn_proxy': float(data['dmn_proxy'][i]),
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(files)} result files, {len(df)} total epoch records.")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Output 1: summary_by_r_v3.csv
# ═══════════════════════════════════════════════════════════════════════

def compute_summary(df):
    """
    Compute final-epoch summary statistics grouped by rehearsal ratio r.

    For each r, reports mean and std of the three metrics across seeds
    at the final epoch. No run is excluded for any reason — all seeds
    contribute to the statistics.
    """
    final = df.loc[df.groupby(['seed', 'r'])['epoch'].idxmax()]

    summary = final.groupby('r').agg(
        F_proxy_mean=('F_proxy', 'mean'),
        F_proxy_std=('F_proxy', 'std'),
        w_proxy_mean=('w_proxy', 'mean'),
        w_proxy_std=('w_proxy', 'std'),
        dmn_proxy_mean=('dmn_proxy', 'mean'),
        dmn_proxy_std=('dmn_proxy', 'std'),
        n_seeds=('seed', 'nunique'),
    ).reset_index()

    return summary


# ═══════════════════════════════════════════════════════════════════════
# Output 2: epoch_to_threshold_v3.csv
# ═══════════════════════════════════════════════════════════════════════

def compute_epoch_to_threshold(df, threshold=DMN_THRESHOLD):
    """
    For each (seed, r) run, find the first epoch where dmn_proxy >= threshold.

    Runs that never cross are recorded with epoch_to_dmn_0.9=NaN and
    never_crossed=True (right-censored observations in survival analysis
    terminology). This preserves the censored data rather than silently
    dropping it, which would bias the mean epoch-to-crossing downward.

    Returns a per-run DataFrame with columns:
    [seed, r, epoch_to_dmn_0.9, final_dmn, never_crossed]
    """
    rows = []
    for (seed, r_val), group in df.groupby(['seed', 'r']):
        group_sorted = group.sort_values('epoch')
        dmn_values = group_sorted['dmn_proxy'].values
        epoch_values = group_sorted['epoch'].values

        crossed_mask = dmn_values >= threshold
        crossed_indices = np.where(crossed_mask)[0]

        if len(crossed_indices) > 0:
            epoch_to_cross = int(epoch_values[crossed_indices[0]])
            never_crossed = False
        else:
            epoch_to_cross = None
            never_crossed = True

        rows.append({
            'seed': seed,
            'r': r_val,
            'epoch_to_dmn_0.9': epoch_to_cross,
            'final_dmn': float(dmn_values[-1]),
            'never_crossed': never_crossed,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Pearson Correlations (stdout only, not a file output)
# ═══════════════════════════════════════════════════════════════════════

def print_correlations(df):
    """
    Compute and print Pearson correlation between w_proxy and dmn_proxy
    trajectories, pooled across seeds, for each rehearsal ratio r.

    This directly tests whether the model's representational separability
    (w_proxy) and its default-mode imagination (dmn_proxy) are coupled or
    independent phenomena across training.
    """
    print("\n" + "=" * 60)
    print("Pearson correlation: w_proxy vs dmn_proxy (across epochs)")
    print("=" * 60)
    print(f"{'r':>6s}  {'corr':>8s}  {'p-value':>10s}  {'n':>5s}")
    print("-" * 35)

    for r_val in sorted(df['r'].unique()):
        subset = df[df['r'] == r_val]
        w = subset['w_proxy'].values
        d = subset['dmn_proxy'].values

        if len(w) < 3:
            print(f"{r_val:6.2f}  {'N/A':>8s}  {'N/A':>10s}  {len(w):5d}")
            continue

        corr, pval = stats.pearsonr(w, d)
        print(f"{r_val:6.2f}  {corr:8.4f}  {pval:10.4e}  {len(w):5d}")


# ═══════════════════════════════════════════════════════════════════════
# Main — single clean execution flow
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze v3 experiment results."
    )
    parser.add_argument('--debug', action='store_true',
                        help="Analyze debug-mode output.")
    args = parser.parse_args()

    raw_dir = cfg.RAW_RESULTS_DIR
    out_dir = cfg.RESULTS_DIR

    # ── Step 1: Load raw data ──
    df = load_all_runs(raw_dir)
    all_runs_path = os.path.join(out_dir, "all_runs_v3.csv")
    df.to_csv(all_runs_path, index=False)
    print(f"Saved: {all_runs_path}")
    
    # ── Step 2: Output 1 — summary_by_r_v3.csv ──
    summary = compute_summary(df)
    summary_path = os.path.join(out_dir, "summary_by_r_v3.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))

    # ── Step 3: Output 2 — epoch_to_threshold_v3.csv ──
    threshold_df = compute_epoch_to_threshold(df)
    threshold_path = os.path.join(out_dir, "epoch_to_threshold_v3.csv")
    threshold_df.to_csv(threshold_path, index=False)
    print(f"\nSaved: {threshold_path}")

    # Print threshold summary to stdout (informational, not a file output)
    thresh_summary = threshold_df.groupby('r').agg(
        mean_epoch_to_cross=('epoch_to_dmn_0.9', 'mean'),
        std_epoch_to_cross=('epoch_to_dmn_0.9', 'std'),
        n_crossed=('epoch_to_dmn_0.9', 'count'),
        n_total=('seed', 'count'),
        fraction_never_crossed=('never_crossed', 'mean'),
    ).reset_index()
    print("\nThreshold crossing summary (dmn >= 0.9):")
    print(thresh_summary.round(3).to_string(index=False))

    # ── Step 4: Pearson correlations (stdout only) ──
    print_correlations(df)

    print("\n" + "=" * 60)
    print("Analysis complete. Output files:")
    print(f"  1. {summary_path}")
    print(f"  2. {threshold_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
