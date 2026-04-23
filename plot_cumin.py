#!/usr/bin/env python3
"""Cumulative-min comparison of NLopt and IBCDFO from H_explore.npy (optimas exploration history).

Reads H_explore.npy in nlopt/ and ibcdfo/ subdirs (pull from
exploration_history_after_sim_*.npy on perlmutter and rename to H_explore.npy).
"""

import numpy as np
import matplotlib.pyplot as plt

import os
INITIAL_CHARGE = float(os.environ.get("INITIAL_CHARGE_C", "0") or 0)
LOSS_THRESHOLD = float(os.environ.get("CHARGE_LOSS_THRESHOLD", "0.01"))

H1 = np.load("nlopt/H_explore.npy")
H2 = np.load("ibcdfo/H_explore.npy")

H1 = H1[H1["sim_ended"] & ~np.isnan(H1["f"])]
H2 = H2[H2["sim_ended"] & ~np.isnan(H2["f"])]
H1 = H1[np.argsort(H1["sim_id"])]
H2 = H2[np.argsort(H2["sim_id"])]

cummin1 = np.minimum.accumulate(H1["f"])
cummin2 = np.minimum.accumulate(H2["f"])
best1 = np.min(H1["f"])
best2 = np.min(H2["f"])

print(f"NLopt:  {len(H1)} sims, best f = {best1:.4e}")
print(f"IBCDFO: {len(H2)} sims, best f = {best2:.4e}")

plt.figure(figsize=(12, 8))
plt.plot(H1["sim_id"], H1["f"], "r-o", alpha=0.5, markersize=5, linewidth=1, label="NLopt f")
plt.plot(H2["sim_id"], H2["f"], "b-o", alpha=0.5, markersize=5, linewidth=1, label="IBCDFO f")
plt.plot(H1["sim_id"], cummin1, "r-", linewidth=2.5, label=f"NLopt cummin (best {best1:.3e})")
plt.plot(H2["sim_id"], cummin2, "b-", linewidth=2.5, label=f"IBCDFO cummin (best {best2:.3e})")

# Shade initial-sample region (read boundary from aposmm_hist.npy if available)
import os
for path in ("nlopt/aposmm_hist.npy", "ibcdfo/aposmm_hist.npy"):
    if os.path.exists(path):
        Ha = np.load(path)
        if "local_pt" in Ha.dtype.names and Ha["local_pt"].any():
            init_end = int(Ha["_id"][np.argmax(Ha["local_pt"])])
            plt.axvspan(0, init_end, alpha=0.15, color="grey", label=f"Initial sample (sims 0–{init_end - 1})")
            break

# Mark sims with charge loss above threshold (set INITIAL_CHARGE_C env var to enable).
if INITIAL_CHARGE:
    for H, color, name, marker, size in [(H1, "red", "NLopt", "+", 80), (H2, "blue", "IBCDFO", "x", 50)]:
        if "charge_C" in H.dtype.names:
            loss = 1.0 - np.abs(H["charge_C"]) / INITIAL_CHARGE
            mask = loss > LOSS_THRESHOLD
            if mask.any():
                plt.scatter(
                    H["sim_id"][mask],
                    H["f"][mask],
                    marker=marker,
                    color=color,
                    s=size,
                    linewidths=1,
                    zorder=5,
                    label=f"{name} charge loss > {LOSS_THRESHOLD:.0%} ({mask.sum()})",
                )

plt.xlabel("Simulation ID")
plt.ylabel("Objective f")
plt.title("NLopt vs IBCDFO")
plt.grid(True, alpha=0.3)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("nlopt_v_ibcdfo.png", dpi=150, bbox_inches="tight")
