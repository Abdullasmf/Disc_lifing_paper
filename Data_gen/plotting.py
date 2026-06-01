"""Optional lightweight validation plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_validation_plot(
    output_dir: Path,
    sample_id: int,
    contour_points: np.ndarray,
    nodes: np.ndarray,
    region_id: np.ndarray,
    stress_max_vm: np.ndarray,
    life_raw: np.ndarray,
):
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(contour_points[:, 0], contour_points[:, 1], "k-", lw=1.2)
    s0 = axes[0].scatter(nodes[:, 0], nodes[:, 1], c=region_id, s=3, cmap="tab10")
    axes[0].set_title("Region IDs")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(s0, ax=axes[0], fraction=0.046)

    s1 = axes[1].scatter(nodes[:, 0], nodes[:, 1], c=stress_max_vm, s=3, cmap="inferno")
    axes[1].set_title("stress_max_vm")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(s1, ax=axes[1], fraction=0.046)

    s2 = axes[2].scatter(nodes[:, 0], nodes[:, 1], c=life_raw, s=3, cmap="viridis")
    axes[2].set_title("life_raw")
    axes[2].set_aspect("equal", adjustable="box")
    fig.colorbar(s2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(plot_dir / f"sample_{sample_id:06d}.png", dpi=160)
    plt.close(fig)

