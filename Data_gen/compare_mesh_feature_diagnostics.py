"""Compare medium vs fine mesh feature-neighbourhood diagnostics.

Reads JSON files produced by ``mesh_feature_diagnostics.py`` and computes
medium-to-fine relative change for p90 stress and median life at each feature.

Convergence criterion: ≤ 15 % change for both p90 stress and median life.

Usage
-----
    python Data_gen/compare_mesh_feature_diagnostics.py
    python Data_gen/compare_mesh_feature_diagnostics.py --input-dir Data_gen/output/diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

CONVERGENCE_THRESHOLD = 0.15  # 15 %


def _relative_change(a: float, b: float) -> float:
    """Absolute relative change |b - a| / max(|a|, 1e-12)."""
    return abs(b - a) / max(abs(a), 1e-12)


def _log_relative_change(a: float, b: float) -> float:
    """Relative change in log10 space, normalised to one decade.

    Life spans many orders of magnitude; a 2× change (0.30 decades) is
    physically significant, while a 50 % linear change at 10^10 cycles
    (0.18 decades) is not.  Using |Δlog10| / max(|log10(a)|, 1) keeps the
    convergence criterion proportional to engineering significance.
    """
    if a <= 0 or b <= 0:
        return _relative_change(a, b)
    la, lb = np.log10(max(a, 1e-300)), np.log10(max(b, 1e-300))
    return abs(lb - la) / max(abs(la), 1.0)


def compare_geometry(
    medium_results: Dict,
    fine_results: Dict,
    geometry_name: str,
) -> bool:
    """Compare medium vs fine results for a single geometry preset.

    Returns True if all convergence checks pass.
    """
    print(f"\n--- Geometry: {geometry_name} ---")
    print(f"  Medium mesh: {medium_results['n_nodes']} nodes, "
          f"{medium_results['n_elements']} elements")
    print(f"  Fine   mesh: {fine_results['n_nodes']} nodes, "
          f"{fine_results['n_elements']} elements")
    print(f"  Node-count ratio (fine/medium): "
          f"{fine_results['n_nodes'] / max(medium_results['n_nodes'], 1):.2f}")

    # Global metrics
    dp_stress = _relative_change(
        medium_results["global_peak_stress_mpa"],
        fine_results["global_peak_stress_mpa"],
    )
    print(f"\n  Global peak stress:  medium={medium_results['global_peak_stress_mpa']:.1f} MPa  "
          f"fine={fine_results['global_peak_stress_mpa']:.1f} MPa  "
          f"rel_change={dp_stress:.1%}")

    # Feature-level convergence
    med_feats = medium_results.get("features", {})
    fine_feats = fine_results.get("features", {})

    all_features = sorted(set(med_feats.keys()) | set(fine_feats.keys()))

    header = (f"  {'Feature':<30} {'med_p90':>10} {'fin_p90':>10} {'dp90%':>8} "
              f"{'med_life':>12} {'fin_life':>12} {'dlife%':>8} {'status':>8}")
    print("\n" + header)
    print("  " + "-" * 110)

    n_pass = 0
    n_fail = 0
    n_skip = 0
    all_pass = True

    for feat in all_features:
        mf = med_feats.get(feat, {})
        ff = fine_feats.get(feat, {})

        if "error" in mf or "error" in ff or "p90_stress_mpa" not in mf or "p90_stress_mpa" not in ff:
            print(f"  {feat:<30} {'(skipped)':>48}")
            n_skip += 1
            continue

        med_p90  = mf["p90_stress_mpa"]
        fin_p90  = ff["p90_stress_mpa"]
        med_life = mf["median_life_cycles"]
        fin_life = ff["median_life_cycles"]

        d_p90  = _relative_change(med_p90, fin_p90)
        d_life = _log_relative_change(med_life, fin_life)

        stress_pass = d_p90  <= CONVERGENCE_THRESHOLD
        life_pass   = d_life <= CONVERGENCE_THRESHOLD
        feat_pass   = stress_pass and life_pass

        if feat_pass:
            n_pass += 1
            status = "PASS"
        else:
            n_fail += 1
            status = "FAIL"
            all_pass = False

        print(f"  {feat:<30} {med_p90:>10.1f} {fin_p90:>10.1f} {d_p90:>7.1%} "
              f"{med_life:>12.3e} {fin_life:>12.3e} {d_life:>7.1%} {status:>8}")

        if not stress_pass:
            print(f"    *** p90 stress: {d_p90:.1%} > {CONVERGENCE_THRESHOLD:.0%} threshold")
        if not life_pass:
            print(f"    *** median life: {d_life:.1%} > {CONVERGENCE_THRESHOLD:.0%} threshold")

    print(f"\n  Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} skipped")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare medium vs fine mesh feature diagnostics.")
    parser.add_argument("--input-dir", type=Path,
                        default=Path("Data_gen/output/diagnostics"),
                        help="Directory containing JSON diagnostic files.")
    args = parser.parse_args()

    in_dir = args.input_dir

    print("=== Medium vs Fine Mesh Convergence Comparison ===")
    print(f"  Input directory: {in_dir}")
    print(f"  Convergence threshold: ≤ {CONVERGENCE_THRESHOLD:.0%} for p90 stress and median life")

    # Discover geometry presets from available files
    medium_files = sorted(in_dir.glob("medium_*.json"))
    if not medium_files:
        print(f"\n  ERROR: No medium_*.json files found in {in_dir}")
        print("  Run: python Data_gen/mesh_feature_diagnostics.py --mesh medium")
        sys.exit(1)

    all_pass = True
    for med_path in medium_files:
        geom_name = med_path.stem.replace("medium_", "")
        fine_path = in_dir / f"fine_{geom_name}.json"

        if not fine_path.exists():
            print(f"\n  WARNING: No fine results for '{geom_name}'. "
                  f"Run: python Data_gen/mesh_feature_diagnostics.py --mesh fine")
            all_pass = False
            continue

        with open(med_path) as f:
            med_results = json.load(f)
        with open(fine_path) as f:
            fine_results = json.load(f)

        geom_pass = compare_geometry(med_results, fine_results, geom_name)
        if not geom_pass:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("=== OVERALL: ALL CONVERGENCE CHECKS PASS ===")
    else:
        print("=== OVERALL: SOME CONVERGENCE CHECKS FAILED ===")
        print("  (This may indicate a sensitivity to raw maximum stress;")
        print("   check p90 and median life — see problem spec section I.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
