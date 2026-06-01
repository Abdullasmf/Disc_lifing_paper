"""HDF5 export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import h5py
import numpy as np

from .config import CYCLE_PHASES, CYCLE_SPEED_FACTORS, REGION_NAME_TO_ID


CONFIG_NAMES = ("edge", "edge_derivatives", "edge_proximity", "full")


def create_output_files(output_dir: Path, seed: int, edge_proximity_distance_mm: float):
    """Create one HDF5 file per node configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, h5py.File] = {}
    for cfg in CONFIG_NAMES:
        path = output_dir / f"disc_dataset_{cfg}.h5"
        h5f = h5py.File(path, "w")
        h5f.attrs["generator"] = "synthetic_axisymmetric_disc_v1"
        h5f.attrs["seed"] = int(seed)
        h5f.attrs["units_length"] = "mm"
        h5f.attrs["stress_target"] = "stress_max_vm"
        h5f.attrs["life_target"] = "life_raw"
        h5f.attrs["cycle_type"] = "fixed_rotation_only"
        h5f.attrs["edge_proximity_distance_mm"] = float(edge_proximity_distance_mm)
        h5f.create_dataset("cycle_phase_names", data=np.array(CYCLE_PHASES, dtype="S32"))
        h5f.create_dataset("cycle_speed_factors", data=CYCLE_SPEED_FACTORS.astype(np.float64))
        h5f.create_dataset(
            "region_name_to_id",
            data=np.array([f"{k}:{v}" for k, v in REGION_NAME_TO_ID.items()], dtype="S32"),
        )
        h5f.create_group("samples")
        files[cfg] = h5f
    return files


def write_sample(
    h5f: h5py.File,
    sample_id: int,
    payload: Dict[str, np.ndarray],
    geometry_params: Dict[str, float],
    seed: int,
):
    """Write one sample to one HDF5 file."""
    sg = h5f["samples"].create_group(f"sample_{sample_id:06d}")
    sg.attrs["sample_id"] = int(sample_id)
    sg.attrs["seed"] = int(seed)

    for key, value in payload.items():
        sg.create_dataset(key, data=value, compression="gzip")

    gp = sg.create_group("geometry_params_mm")
    for key, value in geometry_params.items():
        gp.attrs[key] = float(value)


def close_output_files(files: Dict[str, h5py.File]):
    for h5f in files.values():
        h5f.close()

