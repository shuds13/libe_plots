#!/usr/bin/env python3

"""
Plot f by optimization run and identify mininum f for each run.

Random points are shown in grey - circled if start optimization runs.

To be run from the calling script directory.
"""

import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import pickle
import sys

# Take argument as problem name or use current directory name
if len(sys.argv) > 1:
    run_name = sys.argv[1]
else:
    run_name = os.path.basename(os.getcwd())

title_font = 14
label_font = 14

# N = 12  # number of opt runs to show. - removed since showing all runs
plt.rcParams.update({'font.size': 12})

# Find the most recent .npy and pickle files
try:
    H_file = max(glob.glob("*.npy"), key=os.path.getmtime)
    persis_info_file = max(glob.iglob('*.pickle'), key=os.path.getctime)
except Exception:
    sys.exit("Need a *.npy and a *.pickle files in run dir. Exiting...")

H = np.load(H_file)

with open(persis_info_file, "rb") as f:
    index_sets_raw = pickle.load(f)["run_order"]

# Remove last element (incomplete) from each run
index_sets = {
    key: [i for i in indices if H['sim_ended'][i]]
    for key, indices in index_sets_raw.items()
}

# import pdb; pdb.set_trace()

# Start the main figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot all optimization runs
for key, indices in index_sets.items():
    f_values = H['f'][indices]
    line = ax.plot(indices, f_values, marker='o', label=f'Opt run {key}', zorder=2)
    run_color = line[0].get_color()

    min_index = indices[np.argmin(f_values)]
    min_f_value = np.min(f_values)
    ax.scatter(min_index, min_f_value, color='red', edgecolor='black', s=50, zorder=3)

    if not H['local_pt'][indices[0]]:
        ax.scatter(indices[0], f_values[0], facecolors='lightgrey', edgecolors=run_color, s=80, linewidths=2, zorder=4)

ax.scatter([], [], color='red', edgecolor='black', s=50, label='Best value of opt run')

# Mark APOSMM-identified local minima
if 'local_min' in H.dtype.names:
    lm_idx = np.where(H['local_min'])[0]
    if len(lm_idx):
        ax.scatter(lm_idx, H['f'][lm_idx], marker='*', color='gold', edgecolor='black',
                   s=150, linewidths=1, zorder=5, label=f'Local min ({len(lm_idx)})')

# Mark sims with charge loss above threshold.
# Set INITIAL_CHARGE_C and (optionally) CHARGE_LOSS_THRESHOLD env vars to enable.
INITIAL_CHARGE = float(os.environ.get('INITIAL_CHARGE_C', '0') or 0)
LOSS_THRESHOLD = float(os.environ.get('CHARGE_LOSS_THRESHOLD', '0.01'))
if INITIAL_CHARGE:
    charge_arr = None
    if 'charge_C' in H.dtype.names:
        charge_arr = H['charge_C']
    elif os.path.exists('H_explore.npy'):
        He = np.load('H_explore.npy')
        if 'charge_C' in He.dtype.names:
            charge_by_id = dict(zip(He['sim_id'], He['charge_C']))
            charge_arr = np.array([charge_by_id.get(int(i), np.nan) for i in H['_id']])
    if charge_arr is not None:
        loss = 1.0 - np.abs(charge_arr) / INITIAL_CHARGE
        loss_mask = H['sim_ended'] & (loss > LOSS_THRESHOLD)
        if loss_mask.any():
            loss_idx = np.where(loss_mask)[0]
            ax.scatter(loss_idx, H['f'][loss_idx], marker='x', color='red',
                       s=40, linewidths=1.5, zorder=6,
                       label=f'Charge loss > {LOSS_THRESHOLD:.0%} ({loss_mask.sum()})')

ax.scatter(np.where(H['sim_ended'])[0], H['f'][H['sim_ended']], color='lightgrey', label='Random points', zorder=1)

# Add labels, title, and legend to the main plot
ax.set_xlabel('Simulation ID', fontsize=label_font)
ax.set_ylabel('f value', fontsize=label_font)
ax.set_title(f'{run_name}: f values by optimization runs', fontsize=title_font)
ax.legend(ncol=2)
ax.grid(True)

# Save and show the plot
plt.savefig(f"{run_name}_opt_runs.png")
plt.show()
