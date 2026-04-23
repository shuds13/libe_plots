#!/usr/bin/env python3
"""Stitch the cumin comparison and the two opt_runs plots into one PNG.

Run from a run_set_<N>/ directory after the three component plots exist:
    nlopt_v_ibcdfo.png
    nlopt/nlopt_opt_runs.png
    ibcdfo/ibcdfo_opt_runs.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

paths = [
    "nlopt_v_ibcdfo.png",
    "nlopt/nlopt_opt_runs.png",
    "ibcdfo/ibcdfo_opt_runs.png",
]
titles = ["Cumulative minimum (NLopt vs IBCDFO)", "NLopt: opt runs", "IBCDFO: opt runs"]

missing = [p for p in paths if not os.path.exists(p)]
if missing:
    raise SystemExit(f"Missing input PNGs: {missing}")

fig, axes = plt.subplots(3, 1, figsize=(14, 24))
for ax, p, t in zip(axes, paths, titles):
    ax.imshow(mpimg.imread(p))
    ax.set_title(t, fontsize=14)
    ax.axis("off")

plt.tight_layout()
out = os.path.basename(os.getcwd()) + "_summary.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
