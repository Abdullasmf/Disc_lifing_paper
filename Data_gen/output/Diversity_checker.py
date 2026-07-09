"""
critical_region_analysis.py  —  run from repo root (or wherever the h5 files live)

For each sample in the given HDF5 file:
  - Finds the node with MINIMUM life_raw (i.e. the life-limiting / most critical node)
  - Records that node's (x, r) coordinates, zone_id, and stress value
  - Builds:
      1. A histogram of which zone_id is life-limiting across the dataset
      2. A scatter plot of critical-node (x, r) locations, colored by zone_id
      3. A CSV export of all per-sample critical-node data for further analysis

Usage:
    python critical_region_analysis.py
Edit H5_FILENAME below to point at the dataset you want to inspect.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---- EDIT THIS to point at the HDF5 file you want to analyze ----
H5_FILENAME = "disc_dataset_edge_deriv_zonal.h5"
path = Path(current_dir, H5_FILENAME)

records = []

with h5py.File(path, "r") as f:
    sample_names = sorted(f["samples"].keys())
    print(f"N samples: {len(sample_names)}")

    for name in sample_names:
        grp = f["samples"][name]

        coords = grp["node_coords_mm"][:]           # [N, 2] -> (x, r)
        zone_id = grp["zone_id"][:].astype(np.float32)
        stress = grp["stress_max_vm"][:]
        life = grp["life_raw"][:]

        # Guard against zero/negative life values before log10
        life_safe = np.clip(life, a_min=1e-12, a_max=None)
        log_life = np.log10(life_safe).astype(np.float32)

        # Life-limiting node = minimum life (most critical)
        crit_idx = int(np.argmin(log_life))

        records.append({
            "sample": name,
            "crit_x_mm": float(coords[crit_idx, 0]),
            "crit_r_mm": float(coords[crit_idx, 1]),
            "crit_zone_id": float(zone_id[crit_idx]),
            "crit_stress_MPa": float(stress[crit_idx]),
            "crit_log_life": float(log_life[crit_idx]),
            "min_log_life_overall": float(log_life.min()),
            "max_stress_overall": float(stress.max()),
        })

df = pd.DataFrame(records)

output_dir = Path(current_dir, "output")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "critical_region_analysis.csv"
df.to_csv(csv_path, index=False)
print(f"Saved per-sample critical-node data to: {csv_path}")

# ---- Zone distribution summary ----
zone_counts = df["crit_zone_id"].value_counts().sort_index()
print("\nLife-limiting zone distribution:")
print(zone_counts)

zone_pct = (zone_counts / len(df) * 100).round(1)
print("\nAs percentage of samples:")
print(zone_pct)

# ---- Plot 1: Histogram of which zone is life-limiting ----
fig1, ax1 = plt.subplots(figsize=(8, 6))
zone_counts.plot(kind="bar", ax=ax1, edgecolor="black")
ax1.set_xlabel("Zone ID")
ax1.set_ylabel("Number of samples where this zone is life-limiting")
ax1.set_title("Which Zone Is Life-Limiting Across Dataset")
fig1.tight_layout()
fig1.savefig(output_dir / "critical_zone_histogram.png", dpi=150)
print(f"\nSaved: {output_dir / 'critical_zone_histogram.png'}")

# ---- Plot 2: Scatter of critical-node (x, r) locations, colored by zone ----
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(
    df["crit_x_mm"], df["crit_r_mm"],
    c=df["crit_zone_id"], cmap="tab10", s=15, alpha=0.7
)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("r (mm)")
ax2.set_title("Location of Life-Limiting Node Across All Samples")
legend1 = ax2.legend(*scatter.legend_elements(), title="Zone ID", loc="best")
ax2.add_artist(legend1)
fig2.tight_layout()
fig2.savefig(output_dir / "critical_node_scatter.png", dpi=150)
print(f"Saved: {output_dir / 'critical_node_scatter.png'}")

plt.show()