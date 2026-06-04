This is Abdulla's repo for the first publication for geometry aware ML lifing models.

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

The generator keeps raw mm units, deterministic seeds, no added noise, and outputs stress/life targets for ML training.

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
