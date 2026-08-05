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
        FLANGE_GEOMETRY_PARAMETERS,
        MAX_FLANGE_OFFSET_MM,
        MAX_OFFSET_MM,
        MIN_FLANGE_OFFSET_MM,
        MIN_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        REPRESENTATIONS,
        clip_flange_offsets_to_bounds,
        clip_offsets_to_bounds,
    )
    from .io_hdf5 import close_file, create_dataset_file, write_sample_group
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        FLANGE_GEOMETRY_PARAMETERS,
        MAX_FLANGE_OFFSET_MM,
        MAX_OFFSET_MM,
        MIN_FLANGE_OFFSET_MM,
        MIN_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        REPRESENTATIONS,
        clip_flange_offsets_to_bounds,
        clip_offsets_to_bounds,
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


def _load_flange_offsets_list(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("flange offset list JSON must be a list of dicts")
    out: List[Dict[str, float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each flange offset item must be a dict")
        out.append(clip_flange_offsets_to_bounds({k: float(v) for k, v in item.items()}))
    return out


def _load_flange_offset_bounds(path: Path | None, default_table: Dict[str, float]) -> Dict[str, float]:
    if path is None:
        return {k: float(v) for k, v in default_table.items()}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("flange offset bounds JSON must be a dict")
    out = {k: float(default_table[k]) for k in FLANGE_GEOMETRY_PARAMETERS}
    for k, v in data.items():
        if k not in out:
            raise ValueError(f"Unknown flange offset key in bounds: {k}")
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


def sample_flange_offsets_lhs(
    num_samples: int,
    min_offsets: Dict[str, float],
    max_offsets: Dict[str, float],
    seed: int,
) -> List[Dict[str, float]]:
    """LHS sample of flange parameter offsets, independent of the main geometry LHS."""
    d = len(FLANGE_GEOMETRY_PARAMETERS)
    # Use a different seed offset for the flange LHS to ensure independence.
    lhs = LatinHypercube(d=d, seed=(seed + 999983) % (2**31 - 1))
    u = lhs.random(n=num_samples)

    lo = np.array([min_offsets[k] for k in FLANGE_GEOMETRY_PARAMETERS], dtype=np.float64)
    hi = np.array([max_offsets[k] for k in FLANGE_GEOMETRY_PARAMETERS], dtype=np.float64)
    vec = lo[None, :] + u * (hi - lo)[None, :]

    out: List[Dict[str, float]] = []
    for row in vec:
        row_dict = {k: float(v) for k, v in zip(FLANGE_GEOMETRY_PARAMETERS, row)}
        out.append(clip_flange_offsets_to_bounds(row_dict))
    return out


def validate_lhs_spread(num_samples: int = 30, seed: int = 7) -> bool:
    """Lightweight diagnostic: confirm LHS produces nonzero spread for every
    active core and flange parameter, with independent front/rear variation.

    Returns True if all checks pass, False otherwise.  Prints a brief report.
    """
    print("\n=== LHS spread diagnostic ===")

    core_list = sample_offsets_lhs(
        num_samples=num_samples,
        min_offsets=MIN_OFFSET_MM,
        max_offsets=MAX_OFFSET_MM,
        seed=seed,
    )
    flange_list = sample_flange_offsets_lhs(
        num_samples=num_samples,
        min_offsets=MIN_FLANGE_OFFSET_MM,
        max_offsets=MAX_FLANGE_OFFSET_MM,
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

    # Flange params
    for k in FLANGE_GEOMETRY_PARAMETERS:
        vals = np.array([d[k] for d in flange_list])
        spread = float(vals.max() - vals.min())
        lo = float(MIN_FLANGE_OFFSET_MM[k])
        hi = float(MAX_FLANGE_OFFSET_MM[k])
        expected_range = hi - lo
        ok = spread > 0.5 * expected_range
        print(f"  [{'PASS' if ok else 'FAIL'}] flange/{k}: spread={spread:.4f} (range={expected_range:.4f})")
        if not ok:
            all_pass = False

    # Independent front/rear variation: front and rear should not be identical
    fl_vals = {k: np.array([d[k] for d in flange_list]) for k in FLANGE_GEOMETRY_PARAMETERS}
    front_ax = fl_vals["front_flange_axial_length"]
    rear_ax = fl_vals["rear_flange_axial_length"]
    front_h = fl_vals["front_flange_radial_height"]
    rear_h = fl_vals["rear_flange_radial_height"]
    ind_ax = float(np.std(front_ax - rear_ax)) > 1e-6
    ind_h  = float(np.std(front_h  - rear_h))  > 1e-6
    print(f"  [{'PASS' if ind_ax else 'FAIL'}] Flange axial_length front != rear (std of diff = {np.std(front_ax - rear_ax):.4f})")
    print(f"  [{'PASS' if ind_h  else 'FAIL'}] Flange radial_height front != rear (std of diff = {np.std(front_h  - rear_h):.4f})")
    if not (ind_ax and ind_h):
        all_pass = False

    # Verify flange offsets are actually passed into generate_sample
    sample0 = core_list[0]
    fl0 = flange_list[0]
    fl1 = flange_list[1]
    from .sample_generator import generate_sample
    s0 = generate_sample(param_offsets=sample0, representation="edge", seed=0, include_derivatives=False, flange_param_offsets=fl0)
    s1 = generate_sample(param_offsets=sample0, representation="edge", seed=0, include_derivatives=False, flange_param_offsets=fl1)
    fp0 = s0["flange_parameters_actual"]
    fp1 = s1["flange_parameters_actual"]
    params_differ = any(abs(fp0[k] - fp1[k]) > 1e-9 for k in FLANGE_GEOMETRY_PARAMETERS)
    print(f"  [{'PASS' if params_differ else 'FAIL'}] Flange params reach geometry.py: sample 0 vs 1 actual values differ")
    if not params_differ:
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
    explicit_flange_offsets: List[Dict[str, float]] | None = None,
    lhs_min_flange_offsets: Dict[str, float] | None = None,
    lhs_max_flange_offsets: Dict[str, float] | None = None,
) -> None:
    """Generate a dataset of synthetic disc samples with flange geometry.

    Flange offsets are sampled independently from main geometry offsets via a
    second LHS draw.  Both share the same ``seed`` but with a different scramble
    to guarantee independence (see ``sample_flange_offsets_lhs``).

    When ``explicit_flange_offsets`` is provided it must have the same length as
    the main offset list.  If *None*, flange offsets are generated by LHS.
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

    if explicit_flange_offsets is not None:
        if len(explicit_flange_offsets) != n_samples:
            raise ValueError(
                f"explicit_flange_offsets length {len(explicit_flange_offsets)} "
                f"!= offsets_list length {n_samples}"
            )
        flange_offsets_list = [clip_flange_offsets_to_bounds(d) for d in explicit_flange_offsets]
    else:
        min_fl = lhs_min_flange_offsets or MIN_FLANGE_OFFSET_MM
        max_fl = lhs_max_flange_offsets or MAX_FLANGE_OFFSET_MM
        flange_offsets_list = sample_flange_offsets_lhs(
            num_samples=n_samples,
            min_offsets=min_fl,
            max_offsets=max_fl,
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
        for sample_id, (offsets, flange_offs) in tqdm.tqdm(
            enumerate(zip(offsets_list, flange_offsets_list)),
            total=n_samples, desc="Generating samples"
        ):
            # Deterministic per-sample seed without hidden random-process modifiers.
            # Uses large coprimes and an offset (1_000_003, 7_919, 97) to spread
            # seeds across the 31-bit range while remaining reproducible.
            sample_seed = int((int(seed) * 1_000_003 + sample_id * 7_919 + 97) % (2**31 - 1))
            sample = generate_sample(
                param_offsets=offsets,
                representation=representation,
                seed=sample_seed,
                include_derivatives=include_derivatives,
                include_debug_fields=include_debug_fields,
                lifing_mode=lifing_mode,
                flange_param_offsets=flange_offs,
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

    parser.add_argument("--flange-list-json", type=Path, default=None,
                        help="JSON list of per-sample flange offset dicts (same length as main param list)")
    parser.add_argument("--min-flange-offsets-json", type=Path, default=None,
                        help="JSON dict of min flange offset bounds for LHS sampling")
    parser.add_argument("--max-flange-offsets-json", type=Path, default=None,
                        help="JSON dict of max flange offset bounds for LHS sampling")
    parser.add_argument("--validate-lhs", action="store_true",
                        help="Run LHS spread diagnostic and exit (no dataset generated)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_lhs:
        ok = validate_lhs_spread(num_samples=30, seed=args.seed)
        sys.exit(0 if ok else 1)

    flange_list = None
    if args.flange_list_json is not None:
        flange_list = _load_flange_offsets_list(args.flange_list_json)

    min_flange = _load_flange_offset_bounds(args.min_flange_offsets_json, MIN_FLANGE_OFFSET_MM)
    max_flange = _load_flange_offset_bounds(args.max_flange_offsets_json, MAX_FLANGE_OFFSET_MM)

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
            explicit_flange_offsets=flange_list,
            lhs_min_flange_offsets=min_flange,
            lhs_max_flange_offsets=max_flange,
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
            explicit_flange_offsets=flange_list,
            lhs_min_flange_offsets=min_flange,
            lhs_max_flange_offsets=max_flange,
        )

if __name__ == "__main__":
    main()
