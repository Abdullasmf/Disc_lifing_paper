"""HDF5 writer utilities for single-file dataset output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from .config import (
    BLADE_EQUIV_CG_RADIUS_MM,
    BLADE_EQUIV_MASS_KG,
    BLADE_EQUIV_NUM_BLADES,
    CYCLE_PHASES,
    CYCLE_PHASE_WEIGHTS,
    CYCLE_SPEED_FACTORS,
    MAX_OFFSET_MM,
    MAX_RIM_FEATURE_OFFSET_MM,
    MIN_OFFSET_MM,
    MIN_RIM_FEATURE_OFFSET_MM,
    NOMINAL_GEOMETRY_MM,
    NOMINAL_RIM_FEATURE_MM,
    REGION_NAME_TO_ID,
    SUBZONE_NAME_TO_ID,
    ZONE_TO_REGION,
    ZONE_TO_SUBZONE,
    ZONE_NAME_TO_ID,
)


def _as_key_value_table(table: Dict[str, float], dtype: str = "S128") -> np.ndarray:
    return np.array([f"{k}:{float(v)}" for k, v in table.items()], dtype=dtype)


def create_dataset_file(
    output_h5_path: Path,
    representation: str,
    include_derivatives: bool,
    seed: int,
) -> h5py.File:
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(output_h5_path, "w")

    h5f.attrs["generator_name"] = "synthetic_axisymmetric_disc_two_layer"
    h5f.attrs["generator_version"] = "4.1"
    h5f.attrs["generator/name"] = "synthetic_axisymmetric_disc_two_layer"
    h5f.attrs["generator/version"] = "4.1"
    h5f.attrs["representation"] = representation
    h5f.attrs["include_derivatives"] = bool(include_derivatives)
    h5f.attrs["units"] = "mm"
    h5f.attrs["seed"] = int(seed)

    h5f.create_dataset("cycle_phase_names", data=np.array(CYCLE_PHASES, dtype="S32"))
    h5f.create_dataset("cycle_speed_factors", data=CYCLE_SPEED_FACTORS.astype(np.float64))
    h5f.create_dataset("cycle_weights", data=CYCLE_PHASE_WEIGHTS.astype(np.float64))

    h5f.create_dataset("nominal_parameter_table", data=_as_key_value_table(NOMINAL_GEOMETRY_MM))
    h5f.create_dataset("min_offset_table", data=_as_key_value_table(MIN_OFFSET_MM))
    h5f.create_dataset("max_offset_table", data=_as_key_value_table(MAX_OFFSET_MM))

    # Rim-feature parameter tables (v5.0: C-groove + rear annular drive arm).
    h5f.create_dataset("nominal_rim_feature_table", data=_as_key_value_table(NOMINAL_RIM_FEATURE_MM))
    h5f.create_dataset("min_rim_feature_offset_table", data=_as_key_value_table(MIN_RIM_FEATURE_OFFSET_MM))
    h5f.create_dataset("max_rim_feature_offset_table", data=_as_key_value_table(MAX_RIM_FEATURE_OFFSET_MM))

    h5f.create_dataset(
        "zone_name_to_id_mapping",
        data=np.array([f"{k}:{v}" for k, v in ZONE_NAME_TO_ID.items()], dtype="S64"),
    )
    h5f.create_dataset(
        "region_name_to_id_mapping",
        data=np.array([f"{k}:{v}" for k, v in REGION_NAME_TO_ID.items()], dtype="S64"),
    )
    h5f.create_dataset(
        "zone_to_region_mapping",
        data=np.array([f"{k}:{ZONE_TO_REGION[k]}" for k in ZONE_NAME_TO_ID.keys()], dtype="S64"),
    )
    # Subzone mapping (v4.0 addition).
    h5f.create_dataset(
        "subzone_name_to_id_mapping",
        data=np.array([f"{k}:{v}" for k, v in SUBZONE_NAME_TO_ID.items()], dtype="S64"),
    )
    h5f.create_dataset(
        "zone_to_subzone_mapping",
        data=np.array([f"{k}:{ZONE_TO_SUBZONE.get(k, 'rim_main')}" for k in ZONE_NAME_TO_ID.keys()], dtype="S64"),
    )

    # Blade-equivalent centrifugal load metadata (fixed for every sample).
    h5f.attrs["blade_equiv_num_blades"] = int(BLADE_EQUIV_NUM_BLADES)
    h5f.attrs["blade_equiv_mass_per_blade_kg"] = float(BLADE_EQUIV_MASS_KG)
    h5f.attrs["blade_equiv_cg_radius_mm"] = float(BLADE_EQUIV_CG_RADIUS_MM)

    h5f.create_group("samples")
    return h5f


def write_sample_group(h5f: h5py.File, sample_id: int, sample_seed: int, sample: Dict) -> None:
    sg = h5f["samples"].create_group(f"sample_{sample_id:06d}")
    sg.attrs["sample_id"] = int(sample_id)
    sg.attrs["seed"] = int(sample_seed)

    offs = sg.create_group("param_offsets")
    for key, value in sample["param_offsets"].items():
        offs.attrs[key] = float(value)

    actual = sg.create_group("geometry_parameters_actual")
    for key, value in sample["geometry_parameters_actual"].items():
        actual.attrs[key] = float(value)

    # Rim-feature parameters (v5.0: C-groove + rear annular drive arm).
    if "rim_feature_offsets" in sample:
        rf_offs = sg.create_group("rim_feature_offsets")
        for key, value in sample["rim_feature_offsets"].items():
            rf_offs.attrs[key] = float(value)

    if "rim_feature_parameters_actual" in sample:
        rf_actual = sg.create_group("rim_feature_parameters_actual")
        for key, value in sample["rim_feature_parameters_actual"].items():
            rf_actual.attrs[key] = float(value)

    # Blade-equivalent load metadata (v5.0 addition).
    if "blade_equiv_force_N" in sample:
        sg.attrs["blade_equiv_force_N"] = float(sample["blade_equiv_force_N"])
    if "blade_equiv_load_description" in sample:
        sg.attrs["blade_equiv_load_description"] = str(sample["blade_equiv_load_description"])

    write_keys = [
        "node_coords_mm",
        "zone_id",
        "region_id",
        "stress_max_vm",
        "life_raw",
        "phase_stress_eq",
        "node_features",
        "node_feature_names",
        "triangles",
        "contour_points_mm",
        "contour_zone_id",
        "contour_region_id",
        "contour_arc_length_mm",
        "zone_names",
        "radial_breaks_mm",
    ]

    # Optional new fields (v4.0) — written only if present.
    optional_new_keys = [
        "subzone_id",
        "contour_subzone_id",
        "subzone_names",
    ]
    for k in optional_new_keys:
        if k in sample:
            write_keys.append(k)

    if "arc_length_mm" in sample:
        write_keys.append("arc_length_mm")
    if "distance_to_contour_mm" in sample:
        write_keys.append("distance_to_contour_mm")
    if "nearest_contour_index" in sample:
        write_keys.append("nearest_contour_index")

    for key in write_keys:
        sg.create_dataset(key, data=sample[key], compression="gzip")

    if "feature_landmarks_mm" in sample:
        fg = sg.create_group("feature_landmarks_mm")
        for key, value in sample["feature_landmarks_mm"].items():
            fg.create_dataset(key, data=value)


def close_file(h5f: h5py.File) -> None:
    h5f.close()
