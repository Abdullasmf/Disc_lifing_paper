r"""Compare coarse, medium, and fine outputs from mesh_feature_diagnostics.py.

Example:
  & c:\Users\abfat\miniconda3\envs\MLEnv\python.exe Data_gen\compare_mesh_feature_diagnostics.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ORDER = ["coarse", "medium", "fine"]
METRICS = ["stress_p90_mpa", "stress_max_mpa", "life_median", "life_min"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / a if np.isfinite(a) and abs(a) > 1e-12 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare fixed-neighborhood mesh diagnostics.")
    p.add_argument("--input-dir", type=Path, default=Path("Data_gen/output/mesh_feature_diagnostics"))
    p.add_argument("--output-dir", type=Path, default=Path("Data_gen/output/mesh_feature_diagnostics/comparison"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for mesh in ORDER:
        path = args.input_dir / mesh / "feature_statistics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run mesh_feature_diagnostics.py --mesh {mesh} first.")
        data[mesh] = {r["feature"]: r for r in read_csv(path)}

    features = sorted(set().union(*(d.keys() for d in data.values())))
    rows = []
    for feature in features:
        row = {"feature": feature}
        for mesh in ORDER:
            r = data[mesh].get(feature, {})
            for metric in METRICS:
                row[f"{mesh}_{metric}"] = f(r.get(metric, "nan"))
        for metric in METRICS:
            row[f"medium_to_fine_pct_{metric}"] = pct(row[f"medium_{metric}"], row[f"fine_{metric}"])
            value = abs(row[f"medium_to_fine_pct_{metric}"])
            row[f"medium_to_fine_status_{metric}"] = "PASS" if value <= 15.0 else "CAUTION"
        rows.append(row)

    fields = sorted({k for r in rows for k in r})
    with (args.output_dir / "mesh_comparison.csv").open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for metric, ylabel in [("stress_p90_mpa", "Stress p90 [MPa]"), ("life_median", "Median life")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(features))
        for mesh, marker in zip(ORDER, ["o", "s", "^"]):
            y = [f(data[mesh].get(feature, {}).get(metric, "nan")) for feature in features]
            ax.plot(x, y, marker=marker, label=mesh)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Feature-neighborhood {metric}: mesh convergence")
        if metric == "life_median":
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{metric}_convergence.png", dpi=180)
        plt.close(fig)

    print("Medium → fine changes (absolute values >15% require CAUTION):")
    for row in rows:
        print(f"  {row['feature']}")
        print(f"    stress p90: {row['medium_to_fine_pct_stress_p90_mpa']:+.1f}% [{row['medium_to_fine_status_stress_p90_mpa']}]")
        print(f"    median life: {row['medium_to_fine_pct_life_median']:+.1f}% [{row['medium_to_fine_status_life_median']}]")
    print(f"Wrote: {args.output_dir / 'mesh_comparison.csv'}")


if __name__ == "__main__":
    main()
