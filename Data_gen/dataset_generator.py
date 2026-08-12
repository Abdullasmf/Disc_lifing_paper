"""Dataset driver layer for explicit offsets or Latin hypercube sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats.qmc import LatinHypercube

DEFAULT_NUM_SAMPLES = 200

try:
    from .config import (
        MAX_OFFSET_MM,
        MAX_RIM_FEATURE_OFFSET_MM,
        MIN_OFFSET_MM,
        MIN_RIM_FEATURE_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        RIM_FEATURE_PARAMETERS,
        REPRESENTATIONS,
        clip_offsets_to_bounds,
        clip_rim_feature_offsets_to_bounds,
    )
    from .io_hdf5 import close_file, create_dataset_file, write_sample_group
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        MAX_OFFSET_MM,
        MAX_RIM_FEATURE_OFFSET_MM,
        MIN_OFFSET_MM,
        MIN_RIM_FEATURE_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        RIM_FEATURE_PARAMETERS,
        REPRESENTATIONS,
        clip_offsets_to_bounds,
        clip_rim_feature_offsets_to_bounds,
    )
    from Data_gen.io_hdf5 import close_file, create_dataset_file, write_sample_group
    from Data_gen.sample_generator import generate_sample


def _load_offsets_list(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("offset list JSON must be a list of dicts")
    out: List[Dict[str, float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each offset item must be a dict")
        out.append(clip_offsets_to_bounds({k: float(v) for k, v in item.items()}))
    return out


def _load_offset_bounds(path: Path | None, default_table: Dict[str, float]) -> Dict[str, float]:
    if path is None:
        return {k: float(v) for k, v in default_table.items()}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("offset bounds JSON must be a dict")
    out = {k: float(default_table[k]) for k in PUBLIC_GEOMETRY_PARAMETERS}
    for k, v in data.items():
        if k not in out:
            raise ValueError(f"Unknown offset key in bounds: {k}")
        out[k] = float(v)
    return out


def _load_rim_feature_offsets_list(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("rim-feature offset list JSON must be a list of dicts")
    out: List[Dict[str, float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each rim-feature offset item must be a dict")
        out.append(clip_rim_feature_offsets_to_bounds({k: float(v) for k, v in item.items()}))
    return out


def _load_rim_feature_offset_bounds(path: Path | None, default_table: Dict[str, float]) -> Dict[str, float]:
    if path is None:
        return {k: float(v) for k, v in default_table.items()}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("rim-feature offset bounds JSON must be a dict")
    out = {k: float(default_table[k]) for k in RIM_FEATURE_PARAMETERS}
    for k, v in data.items():
        if k not in out:
            raise ValueError(f"Unknown rim-feature offset key in bounds: {k}")
        out[k] = float(v)
    return out


def sample_offsets_lhs(
    num_samples: int,
    min_offsets: Dict[str, float],
    max_offsets: Dict[str, float],
    seed: int,
) -> List[Dict[str, float]]:
    """LHS sample of core geometry parameter offsets."""
    d = len(PUBLIC_GEOMETRY_PARAMETERS)
    lhs = LatinHypercube(d=d, seed=seed % (2**31 - 1))
    u = lhs.random(n=num_samples)

    lo = np.array([min_offsets[k] for k in PUBLIC_GEOMETRY_PARAMETERS], dtype=np.float64)
    hi = np.array([max_offsets[k] for k in PUBLIC_GEOMETRY_PARAMETERS], dtype=np.float64)
    vec = lo[None, :] + u * (hi - lo)[None, :]

    out: List[Dict[str, float]] = []
    for row in vec:
        row_dict = {k: float(v) for k, v in zip(PUBLIC_GEOMETRY_PARAMETERS, row)}
        out.append(clip_offsets_to_bounds(row_dict))
    return out


def sample_rim_feature_offsets_lhs(
    num_samples: int,
    min_offsets: Dict[str, float],
    max_offsets: Dict[str, float],
    seed: int,
) -> List[Dict[str, float]]:
    """LHS sample of rim-feature parameter offsets, independent of the main geometry LHS."""
    d = len(RIM_FEATURE_PARAMETERS)
    # Use a different seed offset for the rim-feature LHS to ensure independence.
    lhs = LatinHypercube(d=d, seed=(seed + 999983) % (2**31 - 1))
    u = lhs.random(n=num_samples)

    lo = np.array([min_offsets[k] for k in RIM_FEATURE_PARAMETERS], dtype=np.float64)
    hi = np.array([max_offsets[k] for k in RIM_FEATURE_PARAMETERS], dtype=np.float64)
    vec = lo[None, :] + u * (hi - lo)[None, :]

    out: List[Dict[str, float]] = []
    for row in vec:
        row_dict = {k: float(v) for k, v in zip(RIM_FEATURE_PARAMETERS, row)}
        out.append(clip_rim_feature_offsets_to_bounds(row_dict))
    return out


def validate_lhs_spread(num_samples: int = 30, seed: int = 7) -> bool:
    """Lightweight diagnostic: confirm LHS produces nonzero spread for every
    active core and rim-feature parameter, with independent front/rear variation.

    Returns True if all checks pass, False otherwise.  Prints a brief report.
    """
    print("\n=== LHS spread diagnostic ===")

    core_list = sample_offsets_lhs(
        num_samples=num_samples,
        min_offsets=MIN_OFFSET_MM,
        max_offsets=MAX_OFFSET_MM,
        seed=seed,
    )
    rim_feature_list = sample_rim_feature_offsets_lhs(
        num_samples=num_samples,
        min_offsets=MIN_RIM_FEATURE_OFFSET_MM,
        max_offsets=MAX_RIM_FEATURE_OFFSET_MM,
        seed=seed,
    )

    all_pass = True

    # Core params
    for k in PUBLIC_GEOMETRY_PARAMETERS:
        vals = np.array([d[k] for d in core_list])
        spread = float(vals.max() - vals.min())
        lo = float(MIN_OFFSET_MM[k])
        hi = float(MAX_OFFSET_MM[k])
        expected_range = hi - lo
        ok = spread > 0.5 * expected_range
        print(f"  [{'PASS' if ok else 'FAIL'}] core/{k}: spread={spread:.4f} (range={expected_range:.4f})")
        if not ok:
            all_pass = False

    # Rim-feature params
    for k in RIM_FEATURE_PARAMETERS:
        vals = np.array([d[k] for d in rim_feature_list])
        spread = float(vals.max() - vals.min())
        lo = float(MIN_RIM_FEATURE_OFFSET_MM[k])
        hi = float(MAX_RIM_FEATURE_OFFSET_MM[k])
        expected_range = hi - lo
        ok = spread > 0.5 * expected_range
        print(f"  [{'PASS' if ok else 'FAIL'}] rim_feature/{k}: spread={spread:.4f} (range={expected_range:.4f})")
        if not ok:
            all_pass = False

    # Independent front/rear variation: C-groove and arm parameters must not be identical
    rf_vals = {k: np.array([d[k] for d in rim_feature_list]) for k in RIM_FEATURE_PARAMETERS}
    cg_depth = rf_vals["front_cgroove_axial_depth"]
    cg_span  = rf_vals["front_cgroove_radial_span"]
    arm_proj = rf_vals["rear_arm_axial_projection"]
    arm_h    = rf_vals["rear_arm_radial_height"]
    ind_cg_vs_arm_depth = float(np.std(cg_depth - arm_proj)) > 1e-6
    ind_cg_span_vs_arm_h = float(np.std(cg_span - arm_h)) > 1e-6
    print(f"  [{'PASS' if ind_cg_vs_arm_depth else 'FAIL'}] C-groove depth != arm projection (std of diff = {np.std(cg_depth - arm_proj):.4f})")
    print(f"  [{'PASS' if ind_cg_span_vs_arm_h else 'FAIL'}] C-groove span != arm height (std of diff = {np.std(cg_span - arm_h):.4f})")
    if not (ind_cg_vs_arm_depth and ind_cg_span_vs_arm_h):
        all_pass = False

    # Verify rim-feature offsets are actually passed into generate_sample and produce different geometry
    sample0 = core_list[0]
    rf0 = rim_feature_list[0]
    rf1 = rim_feature_list[1]
    from .sample_generator import generate_sample
    s0 = generate_sample(param_offsets=sample0, representation="edge", seed=0, include_derivatives=False, rim_feature_offsets=rf0)
    s1 = generate_sample(param_offsets=sample0, representation="edge", seed=0, include_derivatives=False, rim_feature_offsets=rf1)
    fp0 = s0["rim_feature_parameters_actual"]
    fp1 = s1["rim_feature_parameters_actual"]
    params_differ = any(abs(fp0[k] - fp1[k]) > 1e-9 for k in RIM_FEATURE_PARAMETERS)
    print(f"  [{'PASS' if params_differ else 'FAIL'}] Rim-feature params reach geometry.py: sample 0 vs 1 actual values differ")
    # Different params must produce different contour geometry
    c0 = s0["contour_points_mm"]
    c1 = s1["contour_points_mm"]
    contours_differ = float(np.max(np.abs(c0 - c1))) > 1e-9
    print(f"  [{'PASS' if contours_differ else 'FAIL'}] Different rim-feature params produce different contour geometry")
    if not (params_differ and contours_differ):
        all_pass = False

    print(f"=== LHS spread: {'ALL PASS' if all_pass else 'SOME FAIL'} ===\n")
    return all_pass


def generate_dataset(
    output_h5_path: Path,
    representation: str,
    include_derivatives: bool,
    seed: int,
    explicit_param_offsets: List[Dict[str, float]] | None = None,
    lhs_num_samples: int | None = None,
    lhs_min_offsets: Dict[str, float] | None = None,
    lhs_max_offsets: Dict[str, float] | None = None,
    include_debug_fields: bool = False,
    lifing_mode: str = "zonal",
    explicit_rim_feature_offsets: List[Dict[str, float]] | None = None,
    lhs_min_rim_feature_offsets: Dict[str, float] | None = None,
    lhs_max_rim_feature_offsets: Dict[str, float] | None = None,
) -> None:
    """Generate a dataset of synthetic disc samples with C-groove and rear arm geometry.

    Rim-feature offsets are sampled independently from main geometry offsets via a
    second LHS draw.  Both share the same ``seed`` but with a different scramble
    to guarantee independence (see ``sample_rim_feature_offsets_lhs``).

    When ``explicit_rim_feature_offsets`` is provided it must have the same length
    as the main offset list.  If *None*, rim-feature offsets are generated by LHS.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    explicit_mode = explicit_param_offsets is not None
    lhs_mode = lhs_num_samples is not None
    if explicit_mode == lhs_mode:
        raise ValueError("Choose exactly one mode: explicit parameter list or LHS")

    if explicit_mode:
        offsets_list = [clip_offsets_to_bounds(d) for d in explicit_param_offsets or []]
    else:
        min_offsets = lhs_min_offsets or MIN_OFFSET_MM
        max_offsets = lhs_max_offsets or MAX_OFFSET_MM
        offsets_list = sample_offsets_lhs(
            num_samples=int(lhs_num_samples),
            min_offsets=min_offsets,
            max_offsets=max_offsets,
            seed=int(seed),
        )

    n_samples = len(offsets_list)

    if explicit_rim_feature_offsets is not None:
        if len(explicit_rim_feature_offsets) != n_samples:
            raise ValueError(
                f"explicit_rim_feature_offsets length {len(explicit_rim_feature_offsets)} "
                f"!= offsets_list length {n_samples}"
            )
        rim_feature_offsets_list = [clip_rim_feature_offsets_to_bounds(d) for d in explicit_rim_feature_offsets]
    else:
        min_rf = lhs_min_rim_feature_offsets or MIN_RIM_FEATURE_OFFSET_MM
        max_rf = lhs_max_rim_feature_offsets or MAX_RIM_FEATURE_OFFSET_MM
        rim_feature_offsets_list = sample_rim_feature_offsets_lhs(
            num_samples=n_samples,
            min_offsets=min_rf,
            max_offsets=max_rf,
            seed=int(seed),
        )

    h5f = create_dataset_file(
        output_h5_path=output_h5_path,
        representation=representation,
        include_derivatives=include_derivatives,
        seed=seed,
    )
    import tqdm
    try:
        for sample_id, (offsets, rim_feature_offs) in tqdm.tqdm(
            enumerate(zip(offsets_list, rim_feature_offsets_list)),
            total=n_samples, desc="Generating samples"
        ):
            sample_seed = int((int(seed) * 1_000_003 + sample_id * 7_919 + 97) % (2**31 - 1))
            sample = generate_sample(
                param_offsets=offsets,
                representation=representation,
                seed=sample_seed,
                include_derivatives=include_derivatives,
                include_debug_fields=include_debug_fields,
                lifing_mode=lifing_mode,
                rim_feature_offsets=rim_feature_offs,
            )
            write_sample_group(h5f, sample_id=sample_id, sample_seed=sample_seed, sample=sample)
    finally:
        close_file(h5f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset using explicit offsets or LHS.")
    parser.add_argument("--output-h5", type=Path, default=Path("Data_gen/output/disc_dataset_edge.h5"))
    parser.add_argument("--representation", type=str, default="edge", choices=REPRESENTATIONS)
    parser.add_argument("--include-derivatives", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-debug-fields", action="store_true")
    parser.add_argument("--lifing-mode", type=str, default="zonal", choices=["zonal", "uniform"])

    parser.add_argument("--param-list-json", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--min-offsets-json", type=Path, default=None)
    parser.add_argument("--max-offsets-json", type=Path, default=None)

    parser.add_argument("--rim-feature-list-json", type=Path, default=None,
                        help="JSON list of per-sample rim-feature offset dicts (same length as main param list)")
    parser.add_argument("--min-rim-feature-offsets-json", type=Path, default=None,
                        help="JSON dict of min rim-feature offset bounds for LHS sampling")
    parser.add_argument("--max-rim-feature-offsets-json", type=Path, default=None,
                        help="JSON dict of max rim-feature offset bounds for LHS sampling")
    parser.add_argument("--validate-lhs", action="store_true",
                        help="Run LHS spread diagnostic and exit (no dataset generated)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_lhs:
        ok = validate_lhs_spread(num_samples=30, seed=args.seed)
        sys.exit(0 if ok else 1)

    rim_feature_list = None
    if args.rim_feature_list_json is not None:
        rim_feature_list = _load_rim_feature_offsets_list(args.rim_feature_list_json)

    min_rim_feature = _load_rim_feature_offset_bounds(args.min_rim_feature_offsets_json, MIN_RIM_FEATURE_OFFSET_MM)
    max_rim_feature = _load_rim_feature_offset_bounds(args.max_rim_feature_offsets_json, MAX_RIM_FEATURE_OFFSET_MM)

    if args.param_list_json is not None:
        offsets_list = _load_offsets_list(args.param_list_json)
        generate_dataset(
            output_h5_path=args.output_h5,
            representation=args.representation,
            include_derivatives=args.include_derivatives,
            seed=args.seed,
            explicit_param_offsets=offsets_list,
            include_debug_fields=args.include_debug_fields,
            lifing_mode=args.lifing_mode,
            explicit_rim_feature_offsets=rim_feature_list,
            lhs_min_rim_feature_offsets=min_rim_feature,
            lhs_max_rim_feature_offsets=max_rim_feature,
        )
    else:
        min_offsets = _load_offset_bounds(args.min_offsets_json, MIN_OFFSET_MM)
        max_offsets = _load_offset_bounds(args.max_offsets_json, MAX_OFFSET_MM)
        generate_dataset(
            output_h5_path=args.output_h5,
            representation=args.representation,
            include_derivatives=args.include_derivatives,
            seed=args.seed,
            lhs_num_samples=args.num_samples,
            lhs_min_offsets=min_offsets,
            lhs_max_offsets=max_offsets,
            include_debug_fields=args.include_debug_fields,
            lifing_mode=args.lifing_mode,
            explicit_rim_feature_offsets=rim_feature_list,
            lhs_min_rim_feature_offsets=min_rim_feature,
            lhs_max_rim_feature_offsets=max_rim_feature,
        )

if __name__ == "__main__":
    main()
