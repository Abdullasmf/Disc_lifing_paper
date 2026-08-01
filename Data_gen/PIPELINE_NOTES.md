# Data_gen Pipeline Notes

## Pipeline Overview (v4.0)

### Conceptual flow

```
config.py          → geometry parameters, S-N curves, zone/subzone maps
        ↓
geometry.py        → build_disc_contour() → ContourData (points, zone_ids, subzone_ids)
        ↓
mesh_ops.py        → generate_mesh()  → MeshData (MeshTri, nodes, triangles)
        ↓
physics.py         → axisymmetric FEM solve (scikit-fem, Ti-6Al-4V, 4000 rad/s)
                   → phase stress scaling (7 flight phases)
                   → Palmgren-Miner fatigue life (zonal or uniform S-N)
        ↓
sample_generator.py → generate_sample() → dict with all arrays
        ↓
io_hdf5.py         → write_sample_group() → HDF5 file per dataset
        ↓
dataset_generator.py → generate_dataset() (LHS or explicit offsets, batch)
```

### File roles

| File | Role |
|------|------|
| `config.py` | All constants: zone/subzone IDs, nominal geometry, S-N params, flange params |
| `geometry.py` | Contour construction: `build_disc_contour()`, flange outer-cap builder |
| `mesh_ops.py` | Gmsh-based unstructured triangular mesh, zone/region assignment |
| `physics.py` | FEM stress solve + fatigue life (Palmgren-Miner, piecewise log-log S-N) |
| `sample_generator.py` | Single-sample orchestrator |
| `dataset_generator.py` | Batch driver: LHS sampling + HDF5 output |
| `io_hdf5.py` | HDF5 writer utilities |
| `features.py` | Arc-length resampling and tangent/curvature edge features |
| `validate_fem_nominal.py` | FEM sanity check on nominal geometry |
| `validate_contour.py` | **NEW** Contour comparison and diagnostic plots |
| `plot_example_sample.py` | Debug plot for one sample |

---

## Changes in v4.0 (Flange geometry addition)

### What was changed

#### 1. `config.py`
- Added `FLANGE_GEOMETRY_PARAMETERS` (10 new parameters, see below).
- Added `NOMINAL_FLANGE_MM`, `MIN_FLANGE_OFFSET_MM`, `MAX_FLANGE_OFFSET_MM`.
- Added `SUBZONE_NAME_TO_ID` (9 subzones: bore, lower_transition, web,
  upper_transition, rim_main, front_flange, rear_flange, front_shoulder, rear_shoulder).
- Added `SUBZONE_ID_TO_NAME`, `ZONE_TO_SUBZONE` mappings.
- Added helper functions: `resolve_flange_parameters()`, `clip_flange_offsets_to_bounds()`,
  `flange_offset_vector_to_dict()`, `flange_offsets_dict_to_vector()`.

#### 2. `geometry.py`
- `ContourData` dataclass gains two new fields: `subzone_ids`, `subzone_names`.
- Added `sanitize_flange_parameters()` — enforces fillet-radius ≤ flange height,
  total axial extent < 90% of rim thickness, etc.
- Added `_quarter_arc_points()` — circular arc for 90° fillet corners.
- Added `_cosine_descent()` — smooth shoulder blend (cosine interpolation).
- Added `_build_outer_cap_with_flanges()` — constructs the full outer-cap path
  with front/rear flanges, returning both points and subzone_id arrays.
- Added `_subzone_by_zone()` — maps zone_id to subzone_id for non-flange points.
- `build_disc_contour()` now accepts optional `flange_params` dict.
  If `None`, the legacy flat outer cap is used (backward compatible).
  If a dict, the new flange outer cap is built and subzone_ids are assigned.

#### 3. `sample_generator.py`
- `generate_sample()` accepts new `flange_param_offsets` kwarg (default `None` →
  uses nominal flange values).
- All three representations (`edge`, `edge_proximity`, `full`) now include:
  - `subzone_id` (per-node subzone label)
  - `flange_param_offsets`, `flange_parameters_actual`
- Contour output now includes `contour_subzone_id` and `subzone_names`.

#### 4. `dataset_generator.py`
- `generate_dataset()` accepts new `explicit_flange_offsets`, `lhs_min_flange_offsets`,
  `lhs_max_flange_offsets` kwargs.
- Flange offsets are sampled by a second independent LHS draw with seed offset +999983.
- Added `sample_flange_offsets_lhs()`.
- CLI gains `--flange-list-json`, `--min-flange-offsets-json`, `--max-flange-offsets-json`.

#### 5. `io_hdf5.py`
- Generator version bumped to `4.0`.
- File-level datasets added: `nominal_flange_table`, `min_flange_offset_table`,
  `max_flange_offset_table`, `subzone_name_to_id_mapping`, `zone_to_subzone_mapping`.
- `write_sample_group()` writes per-sample groups `flange_param_offsets` and
  `flange_parameters_actual`, plus optional datasets `subzone_id`,
  `contour_subzone_id`, `subzone_names`.

#### 6. `mesh_ops.py`
- `generate_mesh()` now computes `r_max_contour` from the actual contour points.
- When flanges are present (r_max_contour > r5 + 0.5 mm), two additional mesh
  refinement fields are added:
  - LC_FILLET refinement at the flange outer radius.
  - LC_FILLET refinement at the shoulder region (r5 < r < r_max_contour).

#### 7. `validate_fem_nominal.py`
- Peak stress bounds widened to [300, 800] MPa to accommodate local stress
  concentrations at the flange shoulder fillets.

#### 8. New: `validate_contour.py`
- Generates 3–5 diagnostic PNG files:
  - `contour_comparison.png` — old vs new contour overlay.
  - `flange_variants.png` — 5 deviated flange parameter variants.
  - `subzone_labels.png` — subzone colour map on the new contour.
  - `stress_life_no_flange.png`, `stress_life_with_flanges.png` — FEM stress/life
    (generated when `--skip-stress` is NOT passed).

### What was NOT changed

- **S-N curves**: `ZONAL_SN_PARAMS` and `UNIFORM_SN_PARAMS` are unchanged.
  Flanges are in the "rim" zone (zone_id=4) and therefore use the existing rim
  S-N parameters. No new fatigue law was introduced.
- **Loading cycle**: `CYCLE_SPEED_FACTORS`, `CYCLE_PHASE_WEIGHTS`, `OMEGA_REF_RAD_S`
  are unchanged.
- **FEM physics**: The axisymmetric elasticity solve, boundary conditions, and
  Palmgren-Miner life computation are unchanged.
- **Core zone IDs (0-4)**: `ZONE_NAME_TO_ID` is unchanged. All downstream training
  code that reads `zone_id` continues to work without modification.
- **Output targets**: `stress_max_vm` and `life_raw` have the same semantics.
- **HDF5 backward compatibility**: All existing datasets remain; only new datasets
  and groups are added.

---

## New geometry parameters

### FLANGE_GEOMETRY_PARAMETERS

All lengths in millimetres (mm). Nominal values defined in `NOMINAL_FLANGE_MM`.

| Parameter | Nominal | Offset range | Description |
|-----------|---------|-------------|-------------|
| `front_flange_axial_length` | 3.5 | ±0.30 | Axial depth of front flange (measured inward from front face) |
| `rear_flange_axial_length` | 3.5 | ±0.30 | Axial depth of rear flange |
| `front_flange_radial_height` | 2.5 | ±0.20 | Radial height of front flange above r5 |
| `rear_flange_radial_height` | 2.5 | ±0.20 | Radial height of rear flange above r5 |
| `front_shoulder_offset` | 1.5 | ±0.20 | Width of shoulder cosine-blend region (front) |
| `rear_shoulder_offset` | 1.5 | ±0.20 | Width of shoulder cosine-blend region (rear) |
| `front_fillet_radius` | 1.0 | ±0.10 | Circular-arc fillet at top-front corner of front flange |
| `rear_fillet_radius` | 1.0 | ±0.10 | Circular-arc fillet at top-rear corner of rear flange |
| `rim_to_flange_fillet_radius_front` | 0.8 | ±0.10 | Shoulder fillet at inner edge of front flange top |
| `rim_to_flange_fillet_radius_rear` | 0.8 | ±0.10 | Shoulder fillet at inner edge of rear flange top |

**Physical constraints** enforced by `sanitize_flange_parameters()`:
- `front/rear_fillet_radius ≤ 0.45 × front/rear_flange_radial_height`
- `rim_to_flange_fillet_radius ≤ 0.45 × front/rear_shoulder_offset`
- `front_flange_axial_length + front_shoulder_offset + rear_flange_axial_length + rear_shoulder_offset ≤ 0.90 × rim_thickness`

**Recommended ranges for realistic discs** (from design intent, not yet validated by FEM):
- `front/rear_flange_axial_length`: [2.0, 5.0] mm
- `front/rear_flange_radial_height`: [1.5, 4.0] mm
- `front/rear_shoulder_offset`: [1.0, 2.5] mm
- `front/rear_fillet_radius`: [0.5, 1.5] mm
- `rim_to_flange_fillet_radius_*`: [0.4, 1.2] mm

---

## Subzone labeling

`subzone_id` (dtype int32) is an additional array attached to every sample. It
refines the existing `zone_id` without replacing it.

| subzone_id | subzone_name | Parent zone | Description |
|-----------|-------------|-------------|-------------|
| 0 | bore | bore | Inner bore face and inner cap |
| 1 | lower_transition | lower_transition | Lower fillet zone |
| 2 | web | web | Web body |
| 3 | upper_transition | upper_transition | Upper fillet zone |
| 4 | rim_main | rim | Main outer cap (flat at r = r5) |
| 5 | front_flange | rim | Front flange face + top + top-corner fillet |
| 6 | rear_flange | rim | Rear flange face + top + top-corner fillet |
| 7 | front_shoulder | rim | Shoulder cosine-blend from flange top → r5 (front) |
| 8 | rear_shoulder | rim | Shoulder cosine-blend from r5 → flange top (rear) |

**For contour points**: assigned during construction in `_build_outer_cap_with_flanges()`.

**For edge / edge_proximity / full-mesh points**: assigned by nearest-contour lookup
in `generate_sample()`. Interior mesh nodes inherit the subzone of their nearest
contour point.

---

## HDF5 schema changes (v4.0, backward compatible)

### New file-level datasets

| Dataset | dtype | Content |
|---------|-------|---------|
| `nominal_flange_table` | S128 | `"key:value"` strings for `NOMINAL_FLANGE_MM` |
| `min_flange_offset_table` | S128 | Minimum flange offset bounds |
| `max_flange_offset_table` | S128 | Maximum flange offset bounds |
| `subzone_name_to_id_mapping` | S64 | `"name:id"` strings |
| `zone_to_subzone_mapping` | S64 | `"zone_name:subzone_name"` strings |

### New per-sample groups

| Group | Content |
|-------|---------|
| `flange_param_offsets/` | Attributes: per-key flange offset values |
| `flange_parameters_actual/` | Attributes: per-key actual flange values |

### New per-sample datasets

| Dataset | dtype | Shape | Description |
|---------|-------|-------|-------------|
| `subzone_id` | int32 | (N,) | Subzone label per sample node |
| `contour_subzone_id` | int32 | (M,) | Subzone label per contour point |
| `subzone_names` | S32 | (9,) | Ordered subzone name list |

All existing datasets (`zone_id`, `region_id`, `stress_max_vm`, `life_raw`, etc.)
are **unchanged**. Existing readers will find all expected datasets in their original
positions.

---

## How to regenerate the dataset

```bash
# Edge representation, 200 samples, LHS sampling, flanges at nominal+LHS offsets
python -m Data_gen.dataset_generator \
    --output-h5 Data_gen/output/disc_dataset_edge_flanged.h5 \
    --representation edge \
    --include-derivatives \
    --seed 7 \
    --num-samples 200 \
    --lifing-mode zonal

# Edge representation with wider flange offset bounds
python -m Data_gen.dataset_generator \
    --output-h5 Data_gen/output/disc_dataset_edge_wide_flanges.h5 \
    --representation edge \
    --include-derivatives \
    --seed 7 \
    --num-samples 200 \
    --min-flange-offsets-json /path/to/min_flange.json \
    --max-flange-offsets-json /path/to/max_flange.json

# Generate contour validation plots
python -m Data_gen.validate_contour --output-dir Data_gen/output/validation_contour

# Include FEM stress comparison (slow, ~2-5 min per sample)
python -m Data_gen.validate_contour --output-dir Data_gen/output/validation_contour
```

Example `min_flange.json` for wider variation:
```json
{
  "front_flange_axial_length": -1.5,
  "rear_flange_axial_length": -1.5,
  "front_flange_radial_height": -1.0,
  "rear_flange_radial_height": -1.0,
  "front_shoulder_offset": -0.5,
  "rear_shoulder_offset": -0.5,
  "front_fillet_radius": -0.3,
  "rear_fillet_radius": -0.3,
  "rim_to_flange_fillet_radius_front": -0.2,
  "rim_to_flange_fillet_radius_rear": -0.2
}
```
