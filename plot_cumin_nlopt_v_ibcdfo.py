#!/usr/bin/env python3
"""
Plot comparison of NLopt and IBCDFO cumulative minimums.

Uses data from H_final.npy in nlopt and ibcdfo directories.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load both structured arrays
H1 = np.load('nlopt/H_final.npy')
H2 = np.load('ibcdfo/H_final.npy')

# Filter data to only include completed simulations
H1_filtered = H1[H1['sim_ended'] == True]
H2_filtered = H2[H2['sim_ended'] == True]
print(f"NLopt:  Completed simulations: {len(H1_filtered)}")
print(f"IBCDFO: Completed simulations: {len(H2_filtered)}")

# Calculate cumulative minimums
cummin1 = np.minimum.accumulate(H1_filtered['f'])
cummin2 = np.minimum.accumulate(H2_filtered['f'])

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(H1_filtered['sim_id'], cummin1, 'r-', linewidth=2, label='NLopt')
plt.plot(H2_filtered['sim_id'], cummin2, 'b-', linewidth=2, label='IBCDFO')
plt.xlabel('Simulation ID')
plt.ylabel('Objective Value (f)')
plt.title('Objective Function Values vs Simulation ID')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Force integer ticks on x-axis
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Add some statistics
if len(H1_filtered) > 0:
    best_f1 = np.min(H1_filtered['f'])
    best_sim1 = H1_filtered['sim_id'][np.argmin(H1_filtered['f'])]
    plt.axhline(y=best_f1, color='r', linestyle='--', alpha=0.7, 
                label=f'NLOPT Best = {best_f1:.6f} (sim {best_sim1})')

if len(H2_filtered) > 0:
    best_f2 = np.min(H2_filtered['f'])
    best_sim2 = H2_filtered['sim_id'][np.argmin(H2_filtered['f'])]
    plt.axhline(y=best_f2, color='b', linestyle='--', alpha=0.7, 
                label=f'IBCDFO Best = {best_f2:.6f} (sim {best_sim2})')

plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('nlopt_v_ibcdfo.png', dpi=300, bbox_inches='tight')
plt.show()
