"""Plot inter-card pj/bit vs transfer size from a TransferCurve JSON (default
benchmarks/pcie_transfer_curve.json) with a smooth PCHIP fit and the achieved bandwidth
on a twin axis. Regenerate the data with characterize_transfer_curve.py.

    python plot_pjbit_curve.py [curve.json]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

src = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/pcie_transfer_curve.json"
with open(src) as f:
    c = json.load(f)
S = np.array(c["sizes_bytes"], float)
BW = np.array(c["bw_GBps"], float)
PJ = np.array(c["pj_per_bit"], float)
order = np.argsort(S); S, BW, PJ = S[order], BW[order], PJ[order]
logS = np.log2(S)

# smooth fits in log2(size) space (PCHIP = monotone segments, no overshoot)
xs = np.linspace(logS.min(), logS.max(), 400)
pj_fit = PchipInterpolator(logS, PJ)(xs)
bw_fit = PchipInterpolator(logS, BW)(xs)

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xscale("linear")
ax.plot(xs, pj_fit, "-", color="C3", lw=2, label="pj/bit (PCHIP fit)")
ax.plot(logS, PJ, "o", color="C3", ms=7, zorder=5)
ax.set_ylabel("energy per bit  (pJ/bit)", color="C3", fontsize=12)
ax.tick_params(axis="y", labelcolor="C3")
ax.set_ylim(0, 950)

# bandwidth on twin axis
ax2 = ax.twinx()
ax2.plot(xs, bw_fit, "-", color="C0", lw=1.5, alpha=0.7, label="bandwidth")
ax2.plot(logS, BW, "s", color="C0", ms=5, alpha=0.7)
ax2.set_ylabel("achieved bandwidth  (GB/s)", color="C0", fontsize=12)
ax2.tick_params(axis="y", labelcolor="C0")
ax2.set_ylim(0, 60)

# x ticks as human sizes
def human(b):
    if b >= 1024**3: return f"{b//1024**3} GB"
    if b >= 1024**2: return f"{b//1024**2} MB"
    return f"{b//1024} KB"
tick_bytes = [65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456,
              1073741824, 4294967296, 17179869184]
ax.set_xticks(np.log2(tick_bytes))
ax.set_xticklabels([human(b) for b in tick_bytes], rotation=30, ha="right")
ax.set_xlabel("transfer size", fontsize=12)

# annotate regimes
imin = int(np.argmin(PJ))
ax.annotate(f"min ~{PJ[imin]:.0f} pJ/bit\n@ {human(int(S[imin]))}",
            (logS[imin], PJ[imin]), xytext=(logS[imin], 300),
            ha="center", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("KB regime: BW collapses\n-> pj/bit explodes (latency-bound)",
            (logS[0], PJ[0]), xytext=(logS[2]+0.5, 760), fontsize=9, color="dimgray")
ax.annotate(f"plateau ~{PJ[-1]:.0f} pJ/bit\n(>=256 MB)",
            (logS[-1], PJ[-1]), xytext=(logS[-4]-0.5, 250),
            ha="center", fontsize=9, color="dimgray",
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.axhline(200, ls="--", color="gray", lw=0.8, alpha=0.6)
ax.text(logS[0], 207, "DEFAULT_PJ_PER_BIT = 200", fontsize=8, color="gray")

ax.set_title("NCCL inter-card transfer: pj/bit vs size  (RTX PRO 6000, PCIe, gl1808)",
             fontsize=12)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out = "pcie_pjbit_curve.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
