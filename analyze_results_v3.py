"""
Analysis script for the v3 continual learning dominance-shift experiment.

Scientific rationale
--------------------
This script aggregates raw per-epoch metrics from all experimental runs
into two summary tables:

  all_runs_v3.csv — long-form table of every metric at every epoch for
  every (seed, r) combination. This is the complete record of the
  experiment, suitable for custom post-hoc analysis.

  summary_by_r_v3.csv — final-epoch values aggregated by rehearsal ratio
  r, reporting mean ± std across seeds. This directly answers the
  research question: how does rehearsal ratio affect the three proxies?

Additionally, Pearson correlations between w_proxy and dmn_proxy
trajectories (pooled across seeds, per r value) test whether the two
proxies move together or independently. If they're correlated, the
model's representational structure (w_proxy) and its default-mode
imagination (dmn_proxy) are coupled rather than independent phenomena.

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
        # Parse seed and r from filename: seed1_r0.4.npz
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


def compute_summary(df):
    """
    Compute final-epoch summary statistics grouped by rehearsal ratio r.

    For each r, reports mean and std of the three metrics across seeds
    at the final epoch. No run is excluded for any reason — all seeds
    contribute to the statistics.
    """
    # Get final epoch for each run
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


def compute_correlations(df):
    """
    Compute Pearson correlation between w_proxy and dmn_proxy trajectories,
    pooled across seeds, for each rehearsal ratio r.

    This directly tests whether the model's representational separability
    (w_proxy) and its default-mode imagination (dmn_proxy) are coupled or
    independent phenomena across training.
    """
    print("\n" + "=" * 60)
    print("Pearson correlation: w_proxy vs dmn_proxy (across epochs)")
    print("=" * 60)
    print(f"{'r':>6s}  {'corr':>8s}  {'p-value':>10s}  {'n':>5s}")
    print("-" * 35)

    results = []
    for r_val in sorted(df['r'].unique()):
        subset = df[df['r'] == r_val]
        w = subset['w_proxy'].values
        d = subset['dmn_proxy'].values

        if len(w) < 3:
            print(f"{r_val:6.2f}  {'N/A':>8s}  {'N/A':>10s}  {len(w):5d}")
            continue

        corr, pval = stats.pearsonr(w, d)
        results.append({'r': r_val, 'pearson_r': corr, 'p_value': pval, 'n': len(w)})
        print(f"{r_val:6.2f}  {corr:8.4f}  {pval:10.4e}  {len(w):5d}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze v3 experiment results."
    )
    parser.add_argument('--debug', action='store_true',
                        help="Analyze debug-mode output.")
    args = parser.parse_args()

    raw_dir = cfg.RAW_RESULTS_DIR
    out_dir = cfg.RESULTS_DIR

    # Load all runs
    df = load_all_runs(raw_dir)

    # Save complete long-form table
    all_runs_path = os.path.join(out_dir, "all_runs_v3.csv")
    df.to_csv(all_runs_path, index=False)
    print(f"Saved all runs to {all_runs_path}")

    # Compute and save summary
    summary = compute_summary(df)
    summary_path = os.path.join(out_dir, "summary_by_r_v3.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    print("\nSummary by rehearsal ratio:")
    print(summary.to_string(index=False))

    # Pearson correlations
    compute_correlations(df)


if __name__ == '__main__':
    main()
