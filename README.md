This is Abdulla's repo for the first publication for geometry aware ML lifing models.

## Project purpose

This repository contains the code for a research publication on **geometry-aware machine-learning models for turbine-disc fatigue lifing**. Given the meridional (2D axisymmetric) cross-section of a turbine disc, the models predict per-point **stress** and **fatigue life** (`log10(life)`). A synthetic dataset generator (`Data_gen`) produces the training data, and two experiment trees (`Uniform/`, `Zonal/`) hold a matrix of ablation studies, each trained with two model families.

## Repository layout

```
Data_gen/                 synthetic dataset generator (see section below)
Uniform/                  experiments on uniformly-sampled point sets
Zonal/                    experiments on zone-aware point sets
  <Ablation>/
    PointNetMLPJoint/     PointNet++ encoder + joint MLP head
      Training_script.py      main (L2 / MSE) trainer
      Training_script_L1.py   L1-loss trainer variant
      pn_models.py            model definitions (shared, identical across folders)
      GPUL1.py / GPUL2.py     batch launchers (sweep presets, OOM fallback)
      model_presets.json      named architecture/hyper-parameter presets
    ArGEnT_self_att_noSDF/  ArGEnT self-attention operator network (no SDF)
      Training_script.py
      benchmarks.py           ArGEnTDeepONet definition (shared, identical)
      pn_models.py
      GPU0.py
      model_presets.json
```

`pn_models.py`, `benchmarks.py`, and the `load_h5_pointsets()` loader are byte-identical across every ablation folder; only each `Training_script*.py` is customised per ablation via a small config block.

## Uniform vs Zonal trees

Both trees run the same model code and the same ablation set; they differ only in the dataset they consume:

- **`Uniform/`** — trains on the `*_uniform.h5` datasets (uniformly-sampled geometry point sets).
- **`Zonal/`** — trains on the `*_zonal.h5` datasets (zone-aware sampling) and additionally includes the `Edge_zoneID` ablation, which feeds the per-point zone id as an extra input channel.

## Ablation matrix

Each `Training_script*.py` defines a per-ablation config block:

```python
TARGET_NAMES: List[str]   # ["Stress", "LogLife"] or ["LogLife"]
INPUT_COLS:   List[int]   # which tensor columns feed the encoder
H5_FILENAME:  str         # dataset consumed
EXPECTED_REPR: str        # asserted against the H5 "representation" attribute
```

Per-point tensor column convention: `0 = x`, `1 = r` (coordinates, min-max normalised), `2 = zone_id` (scaled by `/4.0`), `3 = arc-length`, `4–7 = extra geometric/derivative features` (z-scored). The loader appends the targets at the last columns (`stress = width-2`, `log10(life) = width-1`), so `TARGET_COLS` are derived dynamically from tensor width.

| Tree | Ablation | INPUT_COLS | Encoder inputs | Targets | EXPECTED_REPR | H5 file |
|------|----------|------------|----------------|---------|---------------|---------|
| Uniform | Edge | `[0,1]` | x, r | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_uniform.h5` |
| Uniform | Edge_no_stress | `[0,1]` | x, r | LogLife only | `edge` | `disc_dataset_edge_deriv_uniform.h5` |
| Uniform | Edge_arc | `[0,1,3]` | x, r, arc-length | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_uniform.h5` |
| Uniform | Edge_arc_feat | `[0,1,3,4,5,6,7]` | x, r, arc + extra feats | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_uniform.h5` |
| Uniform | Edge_Prox | `[0,1]` | x, r | Stress, LogLife | `edge_proximity` | `disc_dataset_edge_proximity_uniform.h5` |
| Uniform | Full | `[0,1]` | x, r | Stress, LogLife | `mesh` | `disc_dataset_full_uniform.h5` |
| Zonal | Edge | `[0,1]` | x, r | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_zonal.h5` |
| Zonal | Edge_no_stress | `[0,1]` | x, r | LogLife only | `edge` | `disc_dataset_edge_deriv_zonal.h5` |
| Zonal | Edge_arc | `[0,1,3]` | x, r, arc-length | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_zonal.h5` |
| Zonal | Edge_arc_feat | `[0,1,3,4,5,6,7]` | x, r, arc + extra feats | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_zonal.h5` |
| Zonal | Edge_zoneID | `[0,1,2]` | x, r, zone_id | Stress, LogLife | `edge` | `disc_dataset_edge_deriv_zonal.h5` |
| Zonal | Edge_Prox | `[0,1]` | x, r | Stress, LogLife | `edge_proximity` | `disc_dataset_edge_proximity_zonal.h5` |
| Zonal | Full | `[0,1]` | x, r | Stress, LogLife | `mesh` | `disc_dataset_full_zonal.h5` |

The training script asserts the H5 file's stored `representation` attribute equals `EXPECTED_REPR` and that the tensor is wide enough for the requested `INPUT_COLS`, failing fast on a mismatched dataset.

## Model families

- **`PointNetMLPJoint`** — a 2D PointNet++ encoder (`PointNet2Encoder2D`, parametrised `in_channels = len(INPUT_COLS)`) feeding a joint MLP head that regresses all targets together. Architecture is selected from named presets in `model_presets.json`. `Training_script.py` uses an MSE/L2 objective; `Training_script_L1.py` uses L1.
- **`ArGEnT_self_att_noSDF`** — `ArGEnTDeepONet` configured with `attention_type="self", use_sdf=False`: a Galerkin self-attention operator network over the query coordinates `(x, r)` with no branch network and no SDF channel. `out_channels = NUM_TARGETS`.

## Datasets and the `Data_gen` pipeline

Training scripts resolve the dataset directory relative to their own location: `repo_dir = <script>/../../..`, then load `repo_dir/Data_gen/output/<H5_FILENAME>`. **Place the generated `.h5` files under `Data_gen/output/`** using the naming scheme in the matrix above (`disc_dataset_<repr>_<tree>.h5`). Generate them with the `Data_gen` CLI (see below); pick `--representation` to match the ablation's `EXPECTED_REPR` (`edge`, `edge_proximity`, or `full`/`mesh`).

## Running a training script

Each `Training_script.py` is runnable directly (it has a `main(preset_name, batch)` entry point and a `__main__` guard), or via the `GPU*.py` launchers, which sweep a list of presets and retry with smaller batch sizes on CUDA OOM:

```bash
# Direct, single preset (from inside the ablation/model folder)
cd Uniform/Edge/PointNetMLPJoint
python Training_script.py            # runs main(<default preset>, batch)

# L1 variant
python Training_script_L1.py

# Launcher sweeping presets with OOM fallback
python GPUL2.py
```

Available preset names are the keys of the folder's `model_presets.json`.

## Checkpoints and normalization stats

The best checkpoint is written to `<ablation>/<model>/Trained_models/<base_name>_<arch_hash>.pt` (the 8-char `arch_hash` is an MD5 of the architecture config, so different architectures don't collide). If a matching checkpoint already exists it is **resumed**. Each checkpoint stores:

- `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`
- `arch` (architecture config) and `model_name`
- normalization stats: `coord_center`, `coord_half_range`, `target_mean`, `target_std`, and **`extra_feat_stats`** (per-column mean/std used to normalise the extra `INPUT_COLS`)
- `history` and `config` (`epochs_trained`, `best_val`); `best_val_loss` is mirrored for validator tooling

On resume the normalization stats — including `extra_feat_stats` — are reloaded from the checkpoint rather than recomputed (older checkpoints without `extra_feat_stats` fall back to recomputation), and the model state is loaded with `strict=True`.

## Known issues / limitations

- **ArGEnT self-attention ignores the geometry channels.** The `ArGEnTDeepONet` self-attention/no-SDF variant builds its only learned input projection from the query coordinates `(x, r)` (`in_ch_query = 2`, hardcoded in `benchmarks.py`); `geom_points` is passed in but never consumed in the forward pass. Consequently the wider-input ablations do **not** crash for ArGEnT, but the extra encoder features (arc-length, zone_id, extra geometric features) have **no effect** on the ArGEnT model. For this family, `Edge`, `Edge_arc`, `Edge_arc_feat`, and `Edge_zoneID` are effectively the same model and should not be interpreted as a true input-feature ablation. The PointNet family does consume `INPUT_COLS` via its `in_channels`-parametrised encoder. `benchmarks.py` is intentionally left unmodified.
- **Cannot be validated in this environment.** There is no PyTorch runtime and no `.h5` data available in the sandbox, so the scripts have only been verified to compile (`py_compile`) and audited statically. End-to-end training, checkpoint resume, and the H5 representation/width assertions have not been executed against real data.

## Synthetic dataset generator (`Data_gen`)

`Data_gen` now uses a strict two-layer pipeline for a 2D axisymmetric turbine-disc meridional section:

1. **Single-sample deterministic layer** (`Data_gen.sample_generator.generate_sample`)
2. **Dataset driver layer** (`Data_gen.dataset_generator.generate_dataset`)

Geometry family (fixed):

- bore
- lower_transition
- web
- upper_transition
- rim

The generator keeps raw mm units, deterministic seeds, without added noise, and outputs stress/life targets for ML training.

### Single-sample API

```python
generate_sample(
    param_offsets: dict[str, float],
    representation: str,
    seed: int = 0,
    include_derivatives: bool = True,
    include_debug_fields: bool = False,
) -> dict
```

Supported `representation` values:

- `edge`
- `edge_proximity`
- `full`

### Dataset driver CLI

Package mode:

```bash
python -m Data_gen.dataset_generator --output-h5 Data_gen/output/disc_dataset_edge.h5 --representation edge --include-derivatives --num-samples 200 --seed 7
```

Direct script mode:

```bash
python Data_gen/dataset_generator.py --output-h5 Data_gen/output/disc_dataset_edge.h5 --representation edge --include-derivatives --num-samples 200 --seed 7
```

Explicit offset list mode:

```bash
python -m Data_gen.dataset_generator --output-h5 Data_gen/output/disc_dataset_full.h5 --representation full --param-list-json /path/to/offsets.json --seed 7
```

### Plot one sample

Package mode:

```bash
python -m Data_gen.plot_example_sample --representation edge --seed 7 --output Data_gen/output/example_sample.png
```

Direct script mode:

```bash
python Data_gen/plot_example_sample.py --representation edge --seed 7 --output Data_gen/output/example_sample.png
```

`--offsets-json` can be passed to plot a custom offset dictionary.

### Dependencies

```bash
pip install numpy scipy h5py scikit-fem matplotlib
```
