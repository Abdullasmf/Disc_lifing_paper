#!/usr/bin/env python3
"""Read-only plots and metadata export for stored Disc_lifing pilot HDF5 samples."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SAMPLES = "sample_000047,sample_000035,sample_000003,sample_000064,sample_000067"


def native(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    if isinstance(x, np.ndarray):
        return [native(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    return x


def attrs(group):
    return {k: native(v) for k, v in group.attrs.items()}


def group_attrs(group, key):
    return attrs(group[key]) if key in group and isinstance(group[key], h5py.Group) else {}


def landmark_data(group):
    if "feature_landmarks_mm" not in group:
        return {}
    out = {}
    for name, obj in group["feature_landmarks_mm"].items():
        if isinstance(obj, h5py.Dataset):
            out[name] = native(np.asarray(obj))
    return out


def label_map(group, id_key, names_key):
    if id_key not in group or names_key not in group:
        return {}
    names = [str(native(x)) for x in np.asarray(group[names_key])]
    ids = sorted(set(map(int, np.asarray(group[id_key]).ravel().tolist())))
    return {str(i): names[i] if 0 <= i < len(names) else f"id_{i}" for i in ids}


def add_landmarks(ax, landmarks):
    for name, value in landmarks.items():
        v = np.asarray(value, dtype=float).ravel()
        if v.size != 2 or not np.isfinite(v).all():
            continue
        ax.plot(v[0], v[1], marker="x", color="cyan", ms=5, mew=1.2, zorder=5)
        ax.annotate(name.replace("_", " "), (v[0], v[1]), xytext=(3, 3), textcoords="offset points", fontsize=5, color="cyan")


def equal(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")


def plot_one(group, sample_name, output_dir):
    xy = np.asarray(group["node_coords_mm"], dtype=float)
    stress = np.asarray(group["stress_max_vm"], dtype=float).ravel()
    life = np.asarray(group["life_raw"], dtype=float).ravel()
    zone = np.asarray(group["zone_id"], dtype=int).ravel() if "zone_id" in group else np.zeros(len(stress), dtype=int)
    subzone = np.asarray(group["subzone_id"], dtype=int).ravel() if "subzone_id" in group else None
    loglife = np.log10(np.maximum(life, np.finfo(float).tiny))
    i_stress, i_life = int(np.nanargmax(stress)), int(np.nanargmin(life))
    landmarks = landmark_data(group)
    zone_labels = label_map(group, "zone_id", "zone_names")
    subzone_labels = label_map(group, "subzone_id", "subzone_names")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    ax = axs[0, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=zone, s=7, cmap="tab10", rasterized=True)
    ax.scatter(*xy[i_stress], marker="*", s=110, c="white", edgecolors="black", label="max stress")
    ax.scatter(*xy[i_life], marker="X", s=70, c="lime", edgecolors="black", label="min life")
    add_landmarks(ax, landmarks); equal(ax); ax.set_title("Zone ID and landmarks"); ax.legend(loc="best", fontsize=8)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("zone_id")

    ax = axs[0, 1]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=stress, s=7, cmap="inferno", rasterized=True)
    ax.scatter(*xy[i_stress], marker="*", s=120, c="cyan", edgecolors="black")
    add_landmarks(ax, landmarks); equal(ax); ax.set_title(f"von Mises stress [MPa]; max={stress[i_stress]:.2f}")
    fig.colorbar(sc, ax=ax, label="stress_max_vm [MPa]")

    ax = axs[1, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=loglife, s=7, cmap="viridis_r", rasterized=True)
    ax.scatter(*xy[i_life], marker="X", s=80, c="magenta", edgecolors="black")
    add_landmarks(ax, landmarks); equal(ax); ax.set_title(f"LogLife; min={life[i_life]:.4g} cycles")
    fig.colorbar(sc, ax=ax, label="log10(life_raw)")

    ax = axs[1, 1]; ax.axis("off")
    subzone_text = "not stored" if subzone is None else f"{int(subzone[i_stress])} / {int(subzone[i_life])}"
    text = [
        f"{sample_name}",
        f"points: {len(stress)}",
        f"max stress: {stress[i_stress]:.4f} MPa @ node {i_stress}",
        f"min life: {life[i_life]:.6g} cycles @ node {i_life}",
        f"max-stress zone / min-life zone: {int(zone[i_stress])} / {int(zone[i_life])}",
        f"max-stress subzone / min-life subzone: {subzone_text}",
        "",
        "Zone labels:",
        *[f"  {k}: {v}" for k, v in zone_labels.items()],
        "",
        "Subzone labels:",
        *[f"  {k}: {v}" for k, v in subzone_labels.items()],
    ]
    ax.text(0.01, 0.99, "\n".join(text), va="top", family="monospace", fontsize=8)
    fig.suptitle("Stored pilot FEM result — read-only diagnostic", fontsize=14)
    fig.savefig(output_dir / f"{sample_name}_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "sample": sample_name,
        "sample_attributes": attrs(group),
        "extrema": {
            "max_stress_mpa": float(stress[i_stress]), "max_stress_node": i_stress,
            "max_stress_xy_mm": native(xy[i_stress]), "max_stress_zone": int(zone[i_stress]),
            "min_life_cycles": float(life[i_life]), "min_life_node": i_life,
            "min_life_xy_mm": native(xy[i_life]), "min_life_zone": int(zone[i_life]),
        },
        "zone_labels": zone_labels,
        "subzone_labels": subzone_labels,
        "cgroove_sampling_controls_requested": group_attrs(group, "cgroove_sampling_controls_requested"),
        "cgroove_control_mapping_metadata": group_attrs(group, "cgroove_control_mapping_metadata"),
        "rim_feature_parameters_resolved_presanitization": group_attrs(group, "rim_feature_parameters_resolved_presanitization"),
        "rim_feature_parameters_actual": group_attrs(group, "rim_feature_parameters_actual"),
        "geometry_parameters_actual": group_attrs(group, "geometry_parameters_actual"),
        "feature_landmarks_mm": landmarks,
    }
    with (output_dir / f"{sample_name}_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def main():
    p = argparse.ArgumentParser(description="Plot stored pilot HDF5 extreme samples without modifying the HDF5 file.")
    p.add_argument("h5", type=Path, help="Input HDF5 path")
    p.add_argument("--samples", default=DEFAULT_SAMPLES, help="Comma-separated /samples names")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out = args.out_dir or args.h5.with_suffix("").with_name(args.h5.stem + "_extreme_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.samples.split(",") if x.strip()]
    with h5py.File(args.h5, "r") as f:
        if "samples" not in f:
            raise RuntimeError("Expected top-level /samples group.")
        for name in names:
            if name not in f["samples"]:
                raise KeyError(f"Sample not found: {name}")
            plot_one(f["samples"][name], name, out)
            print(f"Wrote {name}_diagnostic.png and {name}_metadata.json")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
