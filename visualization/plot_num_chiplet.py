import seaborn as sns
palette = sns.color_palette("Set2", 4)
from matplotlib.patches import Patch

from visualization.parse_bench import incremental_sweep_energies, incremental_sweep_energies_x_cost, incremental_sweep_edps, incremental_sweep_edps_x_cost

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['font.family'] = 'serif'

# Optionally, specify a list of preferred serif fonts in order of preference
# Matplotlib will try to find the first available font from this list
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern Roman']
def parse_dict_list(dict_list):
    parsed = []
    first = None
    for i, d in enumerate(dict_list):
        if first is None:
            first = d['geometric_mean']['min_energy']
        parsed.append((i+1, d['geometric_mean']['min_energy']/first))
    return parsed

fig, axes = plt.subplots(figsize=(6, 4))

energies = parse_dict_list(incremental_sweep_energies)
energies_x_cost = parse_dict_list(incremental_sweep_energies_x_cost)
edps = parse_dict_list(incremental_sweep_edps)
edps_x_cost = parse_dict_list(incremental_sweep_edps_x_cost)

series = [energies, energies_x_cost, edps, edps_x_cost]
labels = ["Energy", "Energy x Cost", "EDP", "EDP x Cost"]

for data, label, color in zip(series, labels, palette):
    xs, ys = zip(*data)
    axes.plot(xs, ys, label=label, color=color, linewidth=1.8)

axes.axvline(x=8, color="gray", linestyle="--", linewidth=0.8)

# add a label slightly above the line
axes.text(7.8, axes.get_ylim()[1]*0.95, "Our Chiplet\nPool",
        rotation=0, va="top", ha="right", fontsize=12, fontweight='bold')

axes.set_xticks(range(1, len(energies) + 1))
axes.set_xlabel("Number of Chiplets", fontweight='bold', fontsize=12)
axes.set_ylabel("Normalized Value", fontweight='bold', fontsize=12)
legend_handles = [Patch(facecolor=color, label=label) 
                  for color, label in zip(palette, labels)]
axes.legend(handles=legend_handles,fontsize=11)
axes.grid(True, alpha=0.3, axis='y', which='major')
plt.tight_layout()
plt.savefig("graphs/num_chiplet_sweep.pdf", bbox_inches='tight')

plt.show()