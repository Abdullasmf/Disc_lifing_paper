# Data_gen Pipeline Notes

## Pipeline Overview (v5.0 — C-groove + rear drive arm)

### Conceptual flow

```
config.py          → geometry parameters, rim-feature parameters, S-N curves, zone/subzone maps
        ↓
geometry.py        → build_disc_contour() → ContourData (points, zone_ids, subzone_ids, landmarks)
        ↓
mesh_ops.py        → generate_mesh()  → MeshData (MeshTri, nodes, triangles)
        ↓
physics.py         → axisymmetric FEM solve (scikit-fem, Ti-6Al-4V, 4000 rad/s)
                   → blade-equivalent traction on rear arm end face
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
| `config.py` | Constants: zone/subzone IDs, nominal geometry, rim-feature params, S-N curves |
| `geometry.py` | Contour construction: `build_disc_contour()`, C-groove + arm outer-cap builder |
| `mesh_ops.py` | Gmsh-based unstructured triangular mesh, zone/region assignment |
| `physics.py` | FEM stress solve + blade traction + fatigue life (Palmgren-Miner) |
| `sample_generator.py` | Single-sample orchestrator |
| `dataset_generator.py` | Batch driver: LHS sampling + HDF5 output |
| `io_hdf5.py` | HDF5 writer utilities |
| `features.py` | Arc-length resampling and tangent/curvature edge features |
| `validate_fem_nominal.py` | FEM sanity check on nominal geometry |
| `validate_contour.py` | Contour comparison and diagnostic plots |
| `plot_example_sample.py` | Debug plot for one sample |
| `mesh_feature_diagnostics.py` | Feature-neighbourhood mesh + stress diagnostics |
| `compare_mesh_feature_diagnostics.py` | Medium vs fine mesh convergence comparison |
| `analyze_locality_probe.py` | Local feature-neighbourhood stress/life report |

---

## Changes in v5.0 (C-groove + rear drive arm)

The previous smooth flange/collar geometry has been replaced with:

1. A **front-side externally open C-groove** — cut into the front axial face of the rim.
2. A **rear annular drive arm** — with a visible narrow neck/root, arm body, outer corner fillet,
   and a finite vertical end/load face.
3. A **finite visible ligament** — between the C-groove floor and the arm neck/root.

### What was changed

#### `config.py`
- Replaced `FLANGE_GEOMETRY_PARAMETERS` / `NOMINAL_FLANGE_MM` / etc. with:
  - `RIM_FEATURE_PARAMETERS` (11 parameters)
  - `NOMINAL_RIM_FEATURE_MM` — nominal C-groove and arm values
  - `MIN_RIM_FEATURE_OFFSET_MM`, `MAX_RIM_FEATURE_OFFSET_MM` — LHS bounds
  - `resolve_rim_feature_parameters()` — applies offsets to nominal
  - `clip_rim_feature_offsets_to_bounds()` — clips offsets to configured bounds
- Updated `SUBZONE_NAME_TO_ID` (11 subzones) for the new geometry.
- Added blade-equivalent load constants: `BLADE_EQUIV_NUM_BLADES`, `BLADE_EQUIV_MASS_KG`,
  `BLADE_EQUIV_CG_RADIUS_MM`.

#### `geometry.py`
- `ContourData` gains `subzone_ids` and `subzone_names` fields.
- `sanitize_rim_feature_parameters()` — enforces meshable, non-overlapping limits.
- `_build_outer_cap_cgroove_arm()` — builds the full C-groove + arm outer contour.
- `build_disc_contour(..., rim_feature_params=...)` — accepts new rim-feature dict.
- Returns landmarks for all C-groove, ligament, arm, and blade-face features.
- `rim_core_reference` landmark placed at interior rim (x=0, r=r4+40%×rim_height),
  away from stress concentrations for stable convergence metrics.

#### `physics.py`
- Blade-equivalent traction applied on the tagged rear arm end face via `FacetBasis`.
- Arm face identified from `blade_arm_face_x_end_mm`, `blade_arm_face_r_min_mm`,
  `blade_arm_face_r_max_mm` geometry metadata — not broad radial thresholds.
- If no arm facets found, the load is skipped with a warning (not silently omitted).

#### `sample_generator.py`
- Accepts `rim_feature_offsets` kwarg (default `None` → nominal rim features).
- Passes arm-face metadata to FEM for precise blade traction application.

#### `dataset_generator.py`
- LHS samples all 11 rim-feature parameters independently.
- `--validate-lhs` proves nonzero spread for every parameter.

#### `io_hdf5.py`
- Stores `rim_feature_offsets` and `rim_feature_parameters_actual` per sample.

#### `mesh_ops.py`
- Named refinement targets: C-groove entry/floor/exit, ligament, arm root/neck/corner/end face.

#### `mesh_feature_diagnostics.py` *(new)*
- Feature-neighbourhood diagnostics at medium and fine mesh.
- Reports p90 stress, max stress, median life, min life per feature.
- Saves JSON for convergence comparison.

#### `compare_mesh_feature_diagnostics.py` *(new)*
- Loads medium and fine JSON files.
- Computes medium-to-fine relative change: linear for p90 stress, log-scale for median life.
- Convergence criterion: ≤ 15 % for both metrics.

#### `analyze_locality_probe.py` *(new)*
- Local feature-neighbourhood stress/life report for nominal and high-feature geometries.
- Reports feature-vs-baseline comparisons for all 11 feature landmarks.

---

## Rim-feature parameters

### RIM_FEATURE_PARAMETERS

All lengths in millimetres (mm). Nominal values defined in `NOMINAL_RIM_FEATURE_MM`.

#### Front C-groove parameters

| Parameter | Nominal | Offset range | Description |
|-----------|---------|-------------|-------------|
| `front_cgroove_axial_depth` | 4.0 | ±1.0 | Axial penetration from front face inward |
| `front_cgroove_radial_span` | 3.0 | ±0.5 | Radial height of groove opening |
| `front_cgroove_radial_pos` | 0.8 | ±0.2 | r offset of groove bottom above r5 |
| `front_cgroove_entry_radius` | 0.6 | ±0.15 | Entry fillet radius |
| `front_cgroove_floor_radius` | 0.6 | ±0.15 | Floor corner fillet radius |
| `front_cgroove_exit_radius` | 0.6 | ±0.15 | Exit fillet radius |

#### Rear drive-arm parameters

| Parameter | Nominal | Offset range | Description |
|-----------|---------|-------------|-------------|
| `rear_arm_axial_projection` | 4.0 | ±0.5 | Axial extent of arm beyond rear face |
| `rear_arm_radial_height` | 5.0 | ±0.4 | Radial height of arm body above r5 |
| `rear_arm_neck_thickness` | 2.0 | ±0.3 | Radial height of arm neck/root (< radial_height) |
| `rear_arm_root_radius` | 0.6 | ±0.15 | Root/transition fillet radius |
| `rear_arm_outer_corner_radius` | 0.6 | ±0.15 | Outer arm corner fillet radius |

**Physical constraints** enforced by `sanitize_rim_feature_parameters()`:
- neck_thickness < 0.80 × radial_height
- all fillet radii ≥ 0.30 mm (mesh resolution limit)
- arm projection < 0.45 × bore_thickness (clearance constraint)
- C-groove depth ≤ rim_thickness − 2 mm (minimum 2 mm ligament)
- C-groove radial position + span ≤ arm radial height (groove within arm extent)

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
| 4 | rim_main | rim | Main rim cap (flat at r = r5) |
| 5 | front_face | rim | Front axial face above/below C-groove |
| 6 | front_cgroove | rim | C-groove: entry fillet, walls, floor, exit fillet |
| 7 | rear_arm_neck | rim | Arm root neck face, shelf, neck top corner |
| 8 | rear_arm_land | rim | Arm body left face + arm land (horizontal) |
| 9 | rear_arm_corner | rim | Arm outer corner fillet |
| 10 | rear_arm_end_face | rim | Arm rear end/load-transfer face |

All rim-feature subzones (5–10) inherit zone_id=4 (rim) and the existing rim S-N curve.

---

## Geometry landmarks

Landmarks are stored in `ContourData.landmarks_mm` and in the generated sample dict.

| Landmark | Description |
|----------|-------------|
| `front_cgroove_entry` | Entry fillet location [x, r] |
| `front_cgroove_floor` | Floor mid-point [x, r] |
| `front_cgroove_exit` | Exit fillet location [x, r] |
| `ligament_reference` | Midpoint of ligament axial path [x, r] |
| `rear_arm_root` | Arm root / neck corner location [x, r] |
| `rear_arm_neck` | Mid-neck location [x, r] |
| `rear_arm_outer_corner` | Outer corner fillet location [x, r] |
| `rear_arm_load_face_centroid` | End-face centroid [x, r] |
| `rim_core_reference` | Interior rim reference (x=0, r=r4+40%×rim_height) [x, r] |
| `lower_transition_start` | Lower fillet start [0, r1] |
| `upper_transition_start` | Upper fillet start [0, r3] |
| `blade_arm_face_x_end_mm` | Arm end-face x coordinate [mm] |
| `blade_arm_face_r_min_mm` | Arm end-face r_min [mm] |
| `blade_arm_face_r_max_mm` | Arm end-face r_max [mm] |

---

## Blade-equivalent load

The blade-equivalent centrifugal resultant is:
```
F = N_blades × m_blade × ω² × r_cg
  = 60 × 0.003 kg × (4000 rad/s)² × 0.115 m ≈ 331 kN
```
Applied as radial distributed traction over the tagged rear arm end face.

**Fixed across all samples** — not LHS-sampled.

---

## HDF5 schema (v5.0, backward compatible)

### New per-sample groups

| Group | Content |
|-------|---------|
| `rim_feature_offsets/` | Per-key rim-feature offset values |
| `rim_feature_parameters_actual/` | Per-key resolved rim-feature values |

### New per-sample datasets

| Dataset | dtype | Shape | Description |
|---------|-------|-------|-------------|
| `subzone_id` | int32 | (N,) | Subzone label per sample node |
| `contour_subzone_id` | int32 | (M,) | Subzone label per contour point |
| `subzone_names` | S32 | (11,) | Ordered subzone name list |

---

## Mesh configuration

| Setting | Medium | Fine |
|---------|--------|------|
| LC_EDGE | 0.50 mm | 0.30 mm |
| LC_FILLET | 0.30 mm | 0.18 mm |
| Use case | Production | Validation |

Named local refinement targets:
- C-groove entry, floor, exit
- Ligament
- Rear arm root, neck, outer corner, end face
- Lower and upper transition boundaries

---

## How to run

```bash
# Validate LHS spread for all parameters
python -m Data_gen.dataset_generator --validate-lhs

# Contour validation plots (fast, no FEM)
python Data_gen/validate_contour.py --skip-stress

# FEM nominal validation
python Data_gen/validate_fem_nominal.py

# Example sample plot
python Data_gen/plot_example_sample.py

# Medium mesh feature diagnostics
python Data_gen/mesh_feature_diagnostics.py --mesh medium

# Fine mesh feature diagnostics
python Data_gen/mesh_feature_diagnostics.py --mesh fine

# Medium vs fine convergence comparison
python Data_gen/compare_mesh_feature_diagnostics.py

# Generate a dataset (200 samples, edge representation)
python -m Data_gen.dataset_generator \
    --output-h5 Data_gen/output/disc_dataset_edge.h5 \
    --representation edge \
    --include-derivatives \
    --seed 7 \
    --num-samples 200 \
    --lifing-mode zonal
```

---

## Invariants preserved from earlier versions

- 2-D axisymmetric FEM (solid bore, bore/web/rim family)
- Core geometry parameters (PUBLIC_GEOMETRY_PARAMETERS, 11 keys)
- Disc centrifugal body force active in every sample
- Ti-6Al-4V material (E=114 GPa, ν=0.33, ρ=4430 kg/m³)
- 7 flight phases, fixed speed factors and phase weights
- Bore/web/rim zonal S-N behavior (ZONAL_SN_PARAMS)
- Bore zone shot-peen benefit (high knee stress)
- Fillet zones steep Basquin slope (slope_high=13)
- All new C-groove/arm/ligament subzones inherit rim S-N curve (zone_id=4)
- No temperature field, no new S-N curves, no life multipliers, no target noise
