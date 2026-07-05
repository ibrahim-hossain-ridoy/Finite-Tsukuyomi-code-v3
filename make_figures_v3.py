import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ডিরেক্টরি তৈরি (যদি না থাকে)
os.makedirs("results_v3", exist_ok=True)

# ম্যানুয়াল ডেটা ইনপুট (নিখুঁত প্লটের জন্য)
data = {
    'r': [0.0, 0.2, 0.3, 0.4, 0.5, 1.0],
    'mean_epoch': [0.000, 0.200, 0.733, 2.133, 3.571, np.nan],
    'std_epoch': [0.000, 0.414, 0.594, 1.060, 2.070, np.nan],
    'n_crossed': [15, 15, 15, 15, 7, 0],
    'n_total': [15, 15, 15, 15, 15, 15],
    'fraction_never_crossed': [0.000, 0.000, 0.000, 0.000, 0.533, 1.000]
}
df = pd.DataFrame(data)

# প্লট সেটআপ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

# --- বাম পাশের গ্রাফ: Dose-response ---
valid_df = df[df['mean_epoch'].notna()]
ax1.errorbar(valid_df['r'], valid_df['mean_epoch'], yerr=valid_df['std_epoch'],
             fmt='-o', linewidth=2.5, elinewidth=2, capsize=5, color='#1f77b4')

for idx, row in valid_df.iterrows():
    ax1.annotate(f"n={int(row['n_crossed'])}/{int(row['n_total'])}", 
                 (row['r'], row['mean_epoch']),
                 textcoords="offset points", 
                 xytext=(15, -5) if row['r'] == 0.5 else (-25, 10), 
                 ha='center', fontsize=10, color='#555555')

ax1.scatter([1.0], [0], color='red', marker='x', s=100, linewidths=2.5, zorder=5)
ax1.text(0.62, 4.2, "r=1.0:\n0/15 crossed\n(no shift observed)", 
         color='red', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))

ax1.set_title("Dose-response: retention vs shift speed", pad=15)
ax1.set_xlabel("Retention ratio (r)")
ax1.set_ylabel("Mean epoch to dmn >= 0.9\n(among seeds that crossed within 14 epochs)")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(-0.5, 6)

# --- ডান পাশের গ্রাফ: Censoring rate ---
bars = ax2.bar(df['r'], df['fraction_never_crossed'], color='#d62728', width=0.06, edgecolor='black', alpha=0.85)

for bar, idx, row in zip(bars, df.index, df.iterrows()):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f"{int(row[1]['n_total'] - row[1]['n_crossed'])}/{int(row[1]['n_total'])}",
             ha='center', va='bottom', fontsize=10, color='black')

ax2.set_title("Censoring rate by condition", pad=15)
ax2.set_xlabel("Retention ratio (r)")
ax2.set_ylabel("Fraction never crossed within 14 epochs")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(0, 1.1)

fig.suptitle("Final Analysis: n=15 seeds per condition. Error bars = std across seeds that crossed.", 
             fontsize=12, y=0.98, style='italic')

plt.tight_layout()
plt.savefig("results_v3/final_figure_v3.png", dpi=300)
print("SUCCESS: final_figure_v3.png has been generated successfully!")