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
                   → blade-equivalent traction on rim-top blade-attachment face
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
| `limited_adequacy_review.py` | Targeted mesh/S-N/deviation adequacy review for nominal + severe valid LHS |
| `validate_rim_load_and_physics.py` | Rim-load placement, closure, phase-scaling, decomposition, validity, and LHS-sanitization audit |

---

## Changes in v6.0 (correct blade load path + enlarged rim geometry)

### Fixed modelling decisions (v6.0)

1. **Blade centrifugal load moved to the rim-top face** — previously applied to the
   small rear arm end face, now applied to the physically correct boundary: the
   horizontal rim-top face at r = r5 + h_arm (ligament + arm land).
2. **Rear drive arm receives no direct blade load** — the arm experiences stress only
   through structural continuity with the loaded rim, disc body force, and redistribution.
3. **Larger rear arm geometry** — substantially enlarged to be a credible structural feature:
   axial projection 8 mm (was 4 mm), radial height 8 mm (was 5 mm), neck thickness 4 mm
   (was 2 mm), root fillets 1.0 mm (was 0.6 mm).
4. **Deeper C-groove** — axial depth 6 mm (was 4 mm), radial span 4 mm (was 3 mm),
   fillets 0.8 mm (was 0.6 mm).
5. **LHS offset bounds widened** to reflect larger nominal dimensions.
6. **Nominal peak stress target: 300–1500 MPa** — replacing the previously
   unphysical 3.37–3.70 GPa from the overloaded arm end face.

### Changes in v5.0 (C-groove + rear drive arm)

The previous smooth flange/collar geometry has been replaced with:

1. A **front-side externally open C-groove** — cut into the front axial face of the rim.
2. A **rear annular drive arm** — with a visible neck/root, arm body, outer corner fillet,
   and a finite vertical end face (not a load face by default).
3. A **finite visible ligament** — between the C-groove floor and the arm neck/root.

### What was changed in v6.0

#### `config.py`
- `NOMINAL_RIM_FEATURE_MM` updated with larger/credible arm and groove dimensions.
- `MIN/MAX_RIM_FEATURE_OFFSET_MM` widened to match new nominals.
- Blade-load comment updated: load is at rim-top face, NOT rear arm end face.

#### `geometry.py`
- `sanitize_rim_feature_parameters()`: h_arm cap raised (0.45→0.55 × t_rim);
  arm projection `proj_max` changed from bore_thickness-limited to fixed generous limit (20 mm).
- `rf_root_max` formula simplified (removed self-referential term).
- `_build_outer_cap_cgroove_arm()`: `blade_arm_face_*` metadata replaced with
  `blade_rim_top_*` (r = r_arm_top, x from x_front to x_arm_end).
- `build_disc_contour()`: metadata stores `blade_rim_top_r_mm`, `blade_rim_top_x_min_mm`,
  `blade_rim_top_x_max_mm`; arm face keys removed.

#### `physics.py`
- `_assemble_and_solve()` now selects **horizontal** rim-top facets (near-zero Δr) at
  r ≈ rim_top_r_m instead of vertical arm end-face facets.
- `compute_phase_equivalent_stresses()` parameter renamed `arm_face_metadata` →
  `rim_face_metadata`; metadata keys updated to `blade_rim_top_*`.
- Shared helpers now expose production-consistent facet selection, load-face
  geometry, traction, resultant recovery, and optional body/rim load toggles for
  validation-only decomposition runs.

#### `sample_generator.py`
- Metadata key prefix updated from `blade_arm_face_` to `blade_rim_top_`.
- `arm_face_metadata` → `rim_face_metadata` in FEM call.
- `blade_equiv_load_description` updated to name the correct boundary.

#### `validate_fem_nominal.py`
- `PEAK_STRESS_MAX_MPA` reduced from 8000 → 1500 MPa to enforce physical validity.

#### What was changed in v5.0

- `config.py`: Replaced `FLANGE_GEOMETRY_PARAMETERS` with `RIM_FEATURE_PARAMETERS` (11 parameters),
  `NOMINAL_RIM_FEATURE_MM`, bounds, helpers, `SUBZONE_NAME_TO_ID`, blade-load constants.
- `geometry.py`: `ContourData` gains `subzone_ids`/`subzone_names`; `sanitize_rim_feature_parameters()`;
  `_build_outer_cap_cgroove_arm()`; `build_disc_contour(..., rim_feature_params=...)`.
- `physics.py`: Blade-equivalent traction applied via `FacetBasis`. (v6.0 changes location.)
- `sample_generator.py`: supports coupled C-groove normalized controls; stores resolved pre-sanitization + final sanitized rim-feature parameters.
- `dataset_generator.py`: LHS samples non-coupled rim params in mm and C-groove controls in normalized space, then maps controls to mm before sanitization.
- `io_hdf5.py`: stores requested C-groove controls, mapping metadata, resolved pre-sanitization rim-feature values, and final sanitized values per sample.
- `mesh_ops.py`: Named refinement targets: C-groove, ligament, arm features.
- `mesh_feature_diagnostics.py` *(new)*: Feature-neighbourhood diagnostics.
- `compare_mesh_feature_diagnostics.py` *(new)*: Medium vs fine convergence.
- `analyze_locality_probe.py` *(new)*: Local feature stress/life probe.
- `limited_adequacy_review.py` *(new)*: targeted nominal/severe medium-vs-fine mesh, S-N adequacy, and deviation-realism checks.

---

## Rim-feature parameters

### RIM_FEATURE_PARAMETERS

All lengths in millimetres (mm). Nominal values defined in `NOMINAL_RIM_FEATURE_MM`.

#### Front C-groove parameters

| Parameter | Nominal | Offset range | Description |
|-----------|---------|-------------|-------------|
| `front_cgroove_axial_depth` | 6.0 | ±1.5 | Axial penetration from front face inward |
| `front_cgroove_radial_span` | 4.0 | ±1.0 | Radial height of groove opening |
| `front_cgroove_radial_pos` | 1.0 | ±0.3 | r offset of groove bottom above r5 |
| `front_cgroove_entry_radius` | 0.8 | ±0.2 | Entry fillet radius |
| `front_cgroove_floor_radius` | 0.8 | ±0.2 | Floor corner fillet radius |
| `front_cgroove_exit_radius` | 0.8 | ±0.2 | Exit fillet radius |

#### Coupled C-groove normalized controls (LHS space)

| Control | Bounds | Mapped output |
|---------|--------|---------------|
| `cgroove_radial_pos_control` | [0.10, 0.90] | `front_cgroove_radial_pos` |
| `cgroove_span_fraction` | [0.10, 0.90] | `front_cgroove_radial_span` |
| `cgroove_entry_radius_fraction` | [0.10, 0.90] | `front_cgroove_entry_radius` |
| `cgroove_floor_radius_fraction` | [0.10, 0.90] | `front_cgroove_floor_radius` |
| `cgroove_exit_radius_fraction` | [0.10, 0.90] | `front_cgroove_exit_radius` |

Mapping uses the same inequalities enforced by `sanitize_rim_feature_parameters()` and applies conservative clearance margins so controls do not fill geometric limits by default.

Coupled-mapping implementation notes:
- Control-to-mm bounds are derived from the configured physical parameter ranges (`NOMINAL_RIM_FEATURE_MM` + min/max offsets), then intersected with geometry-fit limits.
- The floor-radius mapping now enforces a span lower target based on the configured floor-radius minimum (`span >= floor_min / 0.225`) before sampling, preventing collapse of `front_cgroove_floor_radius` to a clipped constant.
- Per-sample mapping metadata records the effective sampled mm bounds (`*_sampling_min_mm`, `*_sampling_max_mm`) and floor-fit terms used for audit.

#### Rear drive-arm parameters

| Parameter | Nominal | Offset range | Description |
|-----------|---------|-------------|-------------|
| `rear_arm_axial_projection` | 8.0 | ±1.5/+2.0 | Axial extent of arm beyond rear face |
| `rear_arm_radial_height` | 8.0 | ±1.0/+2.0 | Radial height of arm body above r5 |
| `rear_arm_neck_thickness` | 4.0 | ±0.8/+1.0 | Radial height of arm neck/root (< radial_height) |
| `rear_arm_root_radius` | 1.0 | ±0.2/+0.3 | Root/transition fillet radius |
| `rear_arm_outer_corner_radius` | 1.0 | ±0.2/+0.3 | Outer arm corner fillet radius |

**Physical constraints** enforced by `sanitize_rim_feature_parameters()`:
- neck_thickness < 0.80 × radial_height
- all fillet radii ≥ 0.30 mm (mesh resolution limit)
- arm projection < 0.45 × bore_thickness (clearance constraint)
- C-groove depth ≤ rim_thickness − 2 mm (minimum 2 mm ligament)
- C-groove radial position + span ≤ arm radial height (groove within arm extent)
- C-groove control mapping computes available space from the same inequalities before sanitization and records requested controls, resolved mm values, and final sanitized mm values.

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
| `blade_rim_top_r_mm` | Rim-top blade-attachment face radial position [mm] |
| `blade_rim_top_x_min_mm` | Rim-top face axial start (x_front) [mm] |
| `blade_rim_top_x_max_mm` | Rim-top face axial end (x_arm_end) [mm] |

---

## Blade-equivalent load

The blade-equivalent centrifugal resultant is:
```
F = N_blades × m_blade × ω² × r_cg
  = 60 × 0.003 kg × (4000 rad/s)² × 0.115 m ≈ 331 kN
```
Applied as **radial distributed traction** over the **rim-top blade-attachment face**:
the horizontal boundary at r = r5 + h_arm (ligament + arm land).

This represents blades pulling the disc rim radially outward through their attachment region.

**The rear drive arm receives NO direct blade-equivalent traction by default.**
The arm experiences stress only through:
- structural continuity with the loaded rim;
- disc centrifugal body force;
- internal stress redistribution.

Metadata keys in `ContourData.metadata` and sample dict:
- `blade_rim_top_r_mm` — radial position of blade-attachment face [mm]
- `blade_rim_top_x_min_mm` — axial start of blade-attachment face [mm]
- `blade_rim_top_x_max_mm` — axial end of blade-attachment face [mm]

**Fixed across all samples** — not LHS-sampled.

Nominal traction estimate (takeoff):
```
l_face ≈ 22 mm (meridional arc: ligament + arm land)
r_mid ≈ 115 mm
t_r = F / (2π × r_mid × l_face) ≈ 21 MPa
```

Production and validation share the same helpers in `physics.py`:

- `select_blade_rim_top_facets(...)`
- `compute_blade_rim_face_geometry(...)`
- `compute_blade_rim_traction_pa(...)`
- `recover_blade_rim_resultant_n(...)`
- `solve_axisymmetric_response(...)`

Optional validation toggles are available without changing production defaults:

```python
include_body_force=True
include_blade_rim_load=True
```

---

## Rim-load and physics validation

`validate_rim_load_and_physics.py` complements the existing contour, FEM, mesh,
and locality checks. It rebuilds the production contour, remeshes with the
production mesh pipeline, reuses the production rim-top facet selection, and
stores machine-readable evidence for:

- actual loaded rim-top facets and their geometry;
- force-resultant recovery and phase-force scaling;
- stress scaling with `ω²`;
- body-only / rim-load-only / combined-load decomposition;
- nominal, extrema, coupled, and LHS physical validity;
- requested vs actual LHS coverage after sanitization/clipping.

### CLI

```bash
python Data_gen/validate_rim_load_and_physics.py --case nominal
python Data_gen/validate_rim_load_and_physics.py --case extrema
python Data_gen/validate_rim_load_and_physics.py --case lhs --num-samples 20 --seed 7
python Data_gen/validate_rim_load_and_physics.py --case all --num-samples 20 --seed 7
```

Optional arguments:

```bash
--output-dir Data_gen/output/rim_load_validation
--mesh medium
--mesh fine
--save-plots
--fail-on-invalid
--num-samples 20
--seed 7
```

### Saved outputs

- one JSON result file per analysed geometry/case;
- `all_cases_summary.csv`;
- `summary.json`;
- `lhs_coverage_requested.csv`;
- `lhs_coverage_actual.csv`;
- `lhs_sanitization_summary.json`;
- optional PNGs:
  - selected blade-load face overlays;
  - decomposition stress and log-life fields;
  - requested-vs-actual LHS scatter plots;
  - parameter coverage panel figures.

### Force-resultant closure and phase scaling

The validator checks:

```text
t_r = F_blade / (2π r̄ l_face)
F_recovered = Σ t_r (2π r_i Δs_i)
```

Expected acceptance:

- selected facet set is non-empty;
- no clearly unintended loaded facets;
- closure error ≤ 1 %;
- phase stresses and blade-resultant scaling remain consistent with `ω²`.

### Load decomposition

Nominal geometry is re-solved for:

- `body_only`
- `rim_load_only`
- `combined`

The saved outputs include peak stress, peak location, local p90 stresses,
minimum life, median life, median `log10(life)`, life-threshold fractions, and
force metadata for each load split.

### Physical-validity thresholds

- preferred nominal peak-stress band: `300–1300 MPa`
- warning peak-stress band: `1300–1500 MPa`
- invalid peak stress: `>1500 MPa`
- invalid life: `<1 cycle`

### PASS / WARNING / FAIL

- `FAIL`
  - contour/mesh/FEM generation failure;
  - empty blade-load face;
  - force-closure error > 1 %;
  - selected facet outside the intended rim-top interval;
  - global peak stress > 1500 MPa;
  - any node with life < 1 cycle.
- `WARNING`
  - peak stress in `1300–1500 MPa`;
  - any node with `1 ≤ N < 10`;
  - excessive sanitization/clipping;
  - actual LHS coverage substantially smaller than the configured range.
- `PASS`
  - geometry, mesh, load-face selection, force closure, stress, and life all
    remain within the acceptance limits.

### Interpreting clipping and coverage

- High `fraction_changed_by_sanitizer` means the requested LHS space is pushing
  against constructibility limits rather than producing distinct final FEM
  geometries.
- Low `actual_range_over_intended_range` means the final geometry range seen by
  FEM is narrower than the configured offset range.
- Repeated hits at active lower/upper limits identify which constraints dominate
  the sampled design space.
- Parameter status definitions:
  - `PASS`: actual range ≥ 70 % of intended range and < 10 % clipped;
  - `WARNING`: actual range 40–70 % or 10–30 % clipped;
  - `FAIL`: actual range < 40 %, > 30 % clipped, or near-zero final spread.

---

## HDF5 schema (v5.0, backward compatible)

### New per-sample groups

| Group | Content |
|-------|---------|
| `rim_feature_offsets/` | Per-key rim-feature offset values |
| `cgroove_sampling_controls_requested/` | Requested normalized C-groove controls |
| `cgroove_control_mapping_metadata/` | Per-sample control→mm mapping limits/clearance values |
| `rim_feature_parameters_resolved_pre_sanitization/` | Rim-feature mm parameters before sanitizer |
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

# Rim-load nominal audit
python Data_gen/validate_rim_load_and_physics.py --case nominal --mesh medium --save-plots

# Rim-load extrema audit
python Data_gen/validate_rim_load_and_physics.py --case extrema --mesh medium

# Rim-load LHS audit
python Data_gen/validate_rim_load_and_physics.py \
    --case lhs \
    --num-samples 20 \
    --seed 7 \
    --mesh medium \
    --save-plots

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
