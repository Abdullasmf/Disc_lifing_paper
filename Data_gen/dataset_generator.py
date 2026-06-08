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
        MIN_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        REPRESENTATIONS,
        clip_offsets_to_bounds,
    )
    from .io_hdf5 import close_file, create_dataset_file, write_sample_group
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        MAX_OFFSET_MM,
        MIN_OFFSET_MM,
        PUBLIC_GEOMETRY_PARAMETERS,
        REPRESENTATIONS,
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


def sample_offsets_lhs(
    num_samples: int,
    min_offsets: Dict[str, float],
    max_offsets: Dict[str, float],
    seed: int,
) -> List[Dict[str, float]]:
    d = len(PUBLIC_GEOMETRY_PARAMETERS)
    lhs = LatinHypercube(d=d, seed=seed)
    u = lhs.random(n=num_samples)

    lo = np.array([min_offsets[k] for k in PUBLIC_GEOMETRY_PARAMETERS], dtype=np.float64)
    hi = np.array([max_offsets[k] for k in PUBLIC_GEOMETRY_PARAMETERS], dtype=np.float64)
    vec = lo[None, :] + u * (hi - lo)[None, :]

    out: List[Dict[str, float]] = []
    for row in vec:
        row_dict = {k: float(v) for k, v in zip(PUBLIC_GEOMETRY_PARAMETERS, row)}
        out.append(clip_offsets_to_bounds(row_dict))
    return out


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
) -> None:
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

    h5f = create_dataset_file(
        output_h5_path=output_h5_path,
        representation=representation,
        include_derivatives=include_derivatives,
        seed=seed,
    )
    try:
        for sample_id, offsets in enumerate(offsets_list):
            # Deterministic per-sample seed without hidden random-process modifiers.
            sample_seed = int((int(seed) * 1_000_003 + sample_id * 7_919 + 97) % (2**31 - 1))
            sample = generate_sample(
                param_offsets=offsets,
                representation=representation,
                seed=sample_seed,
                include_derivatives=include_derivatives,
                include_debug_fields=include_debug_fields,
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

    parser.add_argument("--param-list-json", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--min-offsets-json", type=Path, default=None)
    parser.add_argument("--max-offsets-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.param_list_json is not None:
        offsets_list = _load_offsets_list(args.param_list_json)
        generate_dataset(
            output_h5_path=args.output_h5,
            representation=args.representation,
            include_derivatives=args.include_derivatives,
            seed=args.seed,
            explicit_param_offsets=offsets_list,
            include_debug_fields=args.include_debug_fields,
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
        )


if __name__ == "__main__":
    main()
