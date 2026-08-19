#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

Q = (0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.00)
REQ = {"node_coords_mm", "node_features", "node_feature_names", "stress_max_vm", "life_raw", "zone_id", "region_id"}


def py(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return [py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def stats(a):
    a = np.asarray(a, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    out = {"n": int(a.size), "finite": int(finite.size), "nonfinite": int(a.size - finite.size)}
    if finite.size:
        out.update({f"p{int(q * 100):02d}": float(np.quantile(finite, q)) for q in Q})
    return out


def inventory(obj, name="/"):
    result = {"path": name, "type": "group" if isinstance(obj, h5py.Group) else "dataset", "attrs": {k: py(v) for k, v in obj.attrs.items()}}
    if isinstance(obj, h5py.Dataset):
        result.update({"shape": list(obj.shape), "dtype": str(obj.dtype), "compression": obj.compression})
    else:
        result["children"] = [inventory(obj[k], f"{name.rstrip('/')}/{k}") for k in obj.keys()]
    return result


def direct_attrs(group, name):
    if name in group and isinstance(group[name], h5py.Group):
        return {k: py(v) for k, v in group[name].attrs.items()}
    return {}


def fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def md_table(rows, cols):
    if not rows:
        return "_None_\n"
    s = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    return s + "\n".join("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Read-only audit for Disc_lifing_paper pilot HDF5 files.")
    ap.add_argument("h5", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.h5.with_suffix("").with_name(args.h5.stem + "_audit")
    out.mkdir(parents=True, exist_ok=True)

    report = []
    summary = {"file": str(args.h5.resolve()), "size_bytes": args.h5.stat().st_size}
    with h5py.File(args.h5, "r") as f:
        summary["root_attrs"] = {k: py(v) for k, v in f.attrs.items()}
        summary["inventory"] = inventory(f)
        if "samples" not in f or not isinstance(f["samples"], h5py.Group):
            raise RuntimeError("Expected top-level /samples group was not found.")
        names = sorted(f["samples"].keys())
        rows, stress_all, life_all, loglife_all = [], [], [], []
        zone_counts, region_counts, missing = Counter(), Counter(), Counter()
        feature_names = Counter()

        for sid in names:
            g = f["samples"][sid]
            keys = set(g.keys())
            for k in REQ - keys:
                missing[k] += 1
            row = {"sample": sid, "sample_attrs": json.dumps({k: py(v) for k, v in g.attrs.items()}, sort_keys=True)}
            coords = np.asarray(g["node_coords_mm"]) if "node_coords_mm" in g else np.empty((0, 2))
            row["n_points"] = int(coords.shape[0])
            for target, collector in (("stress_max_vm", stress_all), ("life_raw", life_all)):
                if target in g:
                    a = np.asarray(g[target], dtype=float).ravel()
                    collector.extend(a.tolist())
                    row[f"{target}_n"] = int(a.size)
                    row[f"{target}_nonfinite"] = int(np.count_nonzero(~np.isfinite(a)))
                    if np.isfinite(a).any():
                        row[f"{target}_min"] = float(np.nanmin(a))
                        row[f"{target}_max"] = float(np.nanmax(a))
                else:
                    row[f"{target}_n"] = 0
            if "life_raw" in g:
                life = np.asarray(g["life_raw"], dtype=float).ravel()
                good = np.isfinite(life) & (life > 0)
                loglife_all.extend(np.log10(life[good]).tolist())
                row["life_nonpositive"] = int(np.count_nonzero(np.isfinite(life) & (life <= 0)))
                row["lowlife_log10_lt_2"] = int(np.count_nonzero(good & (np.log10(life) < 2)))
                row["lowlife_log10_lt_3"] = int(np.count_nonzero(good & (np.log10(life) < 3)))
                row["lowlife_log10_lt_4"] = int(np.count_nonzero(good & (np.log10(life) < 4)))
            for label, counts in (("zone_id", zone_counts), ("region_id", region_counts)):
                if label in g:
                    vals = np.asarray(g[label]).ravel()
                    counts.update(map(int, vals.tolist()))
            if "node_feature_names" in g:
                feature_names.update(map(str, py(np.asarray(g["node_feature_names"])) ))
            for group_name in ("param_offsets", "rim_feature_offsets", "geometry_parameters_actual", "rim_feature_parameters_actual", "geometry_parameters_requested", "geometry_parameters_resolved"):
                for k, v in direct_attrs(g, group_name).items():
                    row[f"{group_name}.{k}"] = v
            rows.append(row)

    summary.update({
        "n_samples": len(names),
        "missing_required_field_sample_counts": dict(missing),
        "point_count": stats([r["n_points"] for r in rows]),
        "stress_max_vm": stats(stress_all),
        "life_raw": stats(life_all),
        "log10_life_raw": stats(loglife_all),
        "stress_above_1500_mpa": int(np.count_nonzero(np.asarray(stress_all) > 1500)),
        "life_nonpositive": int(np.count_nonzero(np.asarray(life_all) <= 0)),
        "fraction_loglife_lt_2": float(np.mean(np.asarray(loglife_all) < 2)) if loglife_all else None,
        "fraction_loglife_lt_3": float(np.mean(np.asarray(loglife_all) < 3)) if loglife_all else None,
        "fraction_loglife_lt_4": float(np.mean(np.asarray(loglife_all) < 4)) if loglife_all else None,
        "zone_counts": dict(sorted(zone_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "node_feature_names": dict(feature_names),
    })

    all_cols = sorted({k for r in rows for k in r})
    with (out / "sample_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    parameter_cols = [c for c in all_cols if "." in c and c.split(".", 1)[0] in {"param_offsets", "rim_feature_offsets", "geometry_parameters_actual", "rim_feature_parameters_actual", "geometry_parameters_requested", "geometry_parameters_resolved"}]
    param_rows = []
    for c in parameter_cols:
        a = np.array([r.get(c, np.nan) for r in rows], dtype=float)
        d = {"parameter": c, **stats(a), "unique_finite": int(np.unique(a[np.isfinite(a)]).size)}
        param_rows.append(d)
    with (out / "parameter_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        cols = ["parameter", "n", "finite", "nonfinite", "p00", "p01", "p05", "p50", "p95", "p99", "p100", "unique_finite"]
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(param_rows)
    summary["parameter_groups_found"] = sorted(set(c.split(".", 1)[0] for c in parameter_cols))

    with (out / "audit_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, allow_nan=False)

    worst_stress = sorted(rows, key=lambda r: r.get("stress_max_vm_max", -np.inf), reverse=True)[:5]
    worst_life = sorted(rows, key=lambda r: r.get("life_raw_min", np.inf))[:5]
    report += ["# Pilot HDF5 audit", "", "## Verdict inputs", f"- File: `{summary['file']}`", f"- Size: {summary['size_bytes']:,} bytes", f"- Stored samples: {summary['n_samples']}", f"- Root attributes: `{json.dumps(summary['root_attrs'], sort_keys=True)}`", "", "## Aggregate targets", "```json", json.dumps({k: summary[k] for k in ("point_count", "stress_max_vm", "life_raw", "log10_life_raw", "stress_above_1500_mpa", "life_nonpositive", "fraction_loglife_lt_2", "fraction_loglife_lt_3", "fraction_loglife_lt_4")}, indent=2), "```", "", "## Schema checks", f"- Missing required fields (number of samples): `{summary['missing_required_field_sample_counts']}`", f"- Feature names: `{summary['node_feature_names']}`", f"- Parameter groups found: `{summary['parameter_groups_found']}`", f"- Zone counts: `{summary['zone_counts']}`", f"- Region counts: `{summary['region_counts']}`", "", "## Five highest per-geometry stress maxima", md_table(worst_stress, ["sample", "n_points", "stress_max_vm_max", "life_raw_min", "life_nonpositive"]), "## Five lowest per-geometry life minima", md_table(worst_life, ["sample", "n_points", "life_raw_min", "stress_max_vm_max", "life_nonpositive"]), "## Parameter summary", "See `parameter_summary.csv`.", "", "## Files", "- `audit_summary.json`: machine-readable inventory and aggregate checks", "- `sample_summary.csv`: one row per geometry, including parameters stored as attributes", "- `parameter_summary.csv`: actual/requested/offset parameter distributions", ""]
    (out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote audit artifacts to: {out}")
    print(f"Stored samples: {summary['n_samples']}; stress >1500 MPa: {summary['stress_above_1500_mpa']}; nonpositive life: {summary['life_nonpositive']}")


if __name__ == "__main__":
    main()
