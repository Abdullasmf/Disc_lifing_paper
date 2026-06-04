"""Generate one debugging plot for one deterministic sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

try:
    from .config import CYCLE_PHASES
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import CYCLE_PHASES
    from Data_gen.sample_generator import generate_sample


def _load_offsets(json_path: Path | None) -> dict[str, float]:
    if json_path is None:
        return {}
    data = json.loads(json_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Offset JSON must be a dict")
    return {k: float(v) for k, v in data.items()}


def create_example_plot(
    output_png: Path,
    representation: str,
    seed: int,
    param_offsets: dict[str, float] | None = None,
    include_derivatives: bool = True,
) -> None:
    sample = generate_sample(
        param_offsets=param_offsets or {},
        representation=representation,
        seed=seed,
        include_derivatives=include_derivatives,
        include_debug_fields=True,
    )

    full = generate_sample(
        param_offsets=param_offsets or {},
        representation="full",
        seed=seed,
        include_derivatives=False,
        include_debug_fields=True,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.ravel()

    contour = sample["contour_points_mm"]
    contour_zone = sample["contour_zone_id"]

    sc0 = ax[0].scatter(contour[:, 0], contour[:, 1], c=contour_zone, s=9, cmap="tab10")
    ax[0].plot(contour[:, 0], contour[:, 1], "k-", lw=0.8, alpha=0.7)
    ax[0].set_title("Contour colored by zone_id")
    ax[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc0, ax=ax[0], fraction=0.046)

    triang = mtri.Triangulation(full["node_coords_mm"][:, 0], full["node_coords_mm"][:, 1], full["triangles"])

    tc1 = ax[1].tripcolor(triang, full["region_id"], cmap="tab10", shading="flat")
    ax[1].set_title("Region map")
    ax[1].set_aspect("equal", adjustable="box")
    fig.colorbar(tc1, ax=ax[1], fraction=0.046)

    tc2 = ax[2].tripcolor(triang, full["stress_max_vm"], cmap="inferno", shading="gouraud")
    ax[2].set_title("stress_max_vm")
    ax[2].set_aspect("equal", adjustable="box")
    fig.colorbar(tc2, ax=ax[2], fraction=0.046)

    tc3 = ax[3].tripcolor(triang, full["life_raw"], cmap="viridis", shading="gouraud")
    ax[3].set_title("life_raw")
    ax[3].set_aspect("equal", adjustable="box")
    fig.colorbar(tc3, ax=ax[3], fraction=0.046)

    phase_idx = list(CYCLE_PHASES).index("takeoff")
    tc4 = ax[4].tripcolor(triang, full["phase_stress_eq"][:, phase_idx], cmap="magma", shading="gouraud")
    ax[4].set_title("Phase stress: takeoff")
    ax[4].set_aspect("equal", adjustable="box")
    fig.colorbar(tc4, ax=ax[4], fraction=0.046)

    if representation == "edge" and sample["node_features"].shape[1] >= 3:
        curv = sample["node_features"][:, 2]
        sc5 = ax[5].scatter(sample["node_coords_mm"][:, 0], sample["node_coords_mm"][:, 1], c=curv, s=10, cmap="cividis")
        ax[5].set_title("Edge curvature")
        ax[5].set_aspect("equal", adjustable="box")
        fig.colorbar(sc5, ax=ax[5], fraction=0.046)
    else:
        ax[5].text(0.05, 0.5, "Edge curvature available only\nfor edge representation", fontsize=11)
        ax[5].set_title("Edge curvature")
        ax[5].set_axis_off()

    for a in ax[:5]:
        a.set_xlabel("x [mm]")
        a.set_ylabel("r [mm]")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one sample debugging plot.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--representation", type=str, default="edge", choices=("edge", "edge_proximity", "full"))
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("Data_gen/output/example_sample.png"))
    parser.add_argument("--no-derivatives", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_example_plot(
        output_png=args.output,
        representation=args.representation,
        seed=args.seed,
        param_offsets=_load_offsets(args.offsets_json),
        include_derivatives=not args.no_derivatives,
    )


if __name__ == "__main__":
    main()
