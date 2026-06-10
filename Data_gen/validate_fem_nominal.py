"""Standalone validation of the axisymmetric FEM stress solver on nominal geometry.

Runs ``generate_sample`` at nominal geometry (zero offsets), prints mesh and
result diagnostics, saves the von Mises stress and life fields as PNGs, and
asserts that the peak von Mises stress lies in a physically plausible band.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

try:
    from .config import CYCLE_PHASES, ZONE_ID_TO_NAME
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import CYCLE_PHASES, ZONE_ID_TO_NAME
    from Data_gen.sample_generator import generate_sample


PEAK_STRESS_MIN_MPA = 100.0
PEAK_STRESS_MAX_MPA = 600.0


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = generate_sample(
        param_offsets={},
        representation="full",
        seed=0,
        include_derivatives=False,
        include_debug_fields=True,
        lifing_mode="zonal",
    )

    nodes = sample["node_coords_mm"]
    triangles = sample["triangles"]
    zone_ids = sample["zone_id"]
    stress_max_vm = sample["stress_max_vm"]
    life_raw = sample["life_raw"]

    n_nodes = nodes.shape[0]
    n_elems = triangles.shape[0]

    peak_idx = int(np.argmax(stress_max_vm))
    peak_vm = float(stress_max_vm[peak_idx])
    peak_xy = nodes[peak_idx]
    peak_zone = ZONE_ID_TO_NAME[int(zone_ids[peak_idx])]

    min_life_idx = int(np.argmin(life_raw))
    min_life = float(life_raw[min_life_idx])
    max_life = float(np.max(life_raw))
    min_life_zone = ZONE_ID_TO_NAME[int(zone_ids[min_life_idx])]

    print("=== FEM nominal validation ===")
    print(f"Mesh nodes:     {n_nodes}")
    print(f"Mesh elements:  {n_elems}")
    print(f"Peak von Mises stress: {peak_vm:.2f} MPa")
    print(f"  at [x, r] = [{peak_xy[0]:.3f}, {peak_xy[1]:.3f}] mm")
    print(f"  zone: {peak_zone}")
    print(f"Min life: {min_life:.4e} cycles  (zone: {min_life_zone})")
    print(f"Max life: {max_life:.4e} cycles")
    print(f"Phases: {list(CYCLE_PHASES)}")

    # --- Stress figure (von Mises at takeoff, == stress_max_vm) ---
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    fig1, ax1 = plt.subplots(figsize=(6, 8))
    tcf = ax1.tripcolor(triang, stress_max_vm, cmap="inferno", shading="gouraud")
    ax1.plot(peak_xy[0], peak_xy[1], "c*", markersize=14, label="peak")
    ax1.set_title(f"von Mises stress (takeoff)\npeak {peak_vm:.1f} MPa @ {peak_zone}")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("r [mm]")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right")
    fig1.colorbar(tcf, ax=ax1, fraction=0.046, label="von Mises [MPa]")
    fig1.tight_layout()
    stress_png = out_dir / "fem_stress_nominal.png"
    fig1.savefig(stress_png, dpi=180)
    plt.close(fig1)
    print(f"Saved: {stress_png}")

    # --- Life figure (log10 life, zonal mode) ---
    life_log10 = np.log10(np.maximum(life_raw, 1e-10))
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    tcf2 = ax2.tripcolor(triang, life_log10, cmap="viridis", shading="gouraud")
    ax2.set_title("log10(life_raw) [zonal S-N]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("r [mm]")
    ax2.set_aspect("equal", adjustable="box")
    fig2.colorbar(tcf2, ax=ax2, fraction=0.046, label="log10(cycles)")
    fig2.tight_layout()
    life_png = out_dir / "fem_life_nominal.png"
    fig2.savefig(life_png, dpi=180)
    plt.close(fig2)
    print(f"Saved: {life_png}")

    if not (PEAK_STRESS_MIN_MPA <= peak_vm <= PEAK_STRESS_MAX_MPA):
        print("WARNING: peak stress outside expected range — check omega_ref or material constants.")
    assert PEAK_STRESS_MIN_MPA <= peak_vm <= PEAK_STRESS_MAX_MPA, (
        f"Peak von Mises {peak_vm:.2f} MPa outside [{PEAK_STRESS_MIN_MPA}, {PEAK_STRESS_MAX_MPA}] MPa"
    )
    print("PASS: peak von Mises within expected range.")


if __name__ == "__main__":
    main()
