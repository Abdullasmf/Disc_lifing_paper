This is Abdulla's repo for the first publication for geometry aware ML lifing models.

## Synthetic dataset generator (`Data_gen`)

This repository includes a Python-based synthetic data generator for a 2D axisymmetric rotor-disc meridional cross-section, focused on **data generation only**.

### What it generates

For each sample, the generator builds a parameterized disc geometry with three semantic regions:

- `bore`
- `web`
- `rim`

It computes per-node phase-wise equivalent stresses using a lightweight rotating-disc-inspired surrogate:

- centrifugal-type phase scaling with `speed_factor^2`
- radial and hoop-like stress-shape terms
- local thickness amplification
- stress concentration factors near bore-web and web-rim transitions
- mild region-specific scaling

Then it computes targets:

- `stress_max_vm` (max equivalent stress over cycle)
- `life_raw` (raw life via phase-wise Miner damage accumulation with region-specific nonlinear Basquin S-N laws)
- `phase_stress_eq` (equivalent stress for each phase)

No normalization or noise is applied.

### Outputs

Running the generator writes four HDF5 files in the output directory:

- `disc_dataset_edge.h5`
- `disc_dataset_edge_derivatives.h5`
- `disc_dataset_edge_proximity.h5`
- `disc_dataset_full.h5`

All files contain the same sample IDs and include node coordinates, `region_id`, `segment_id`, targets, phase-wise stresses, geometry parameters, cycle metadata, segment metadata, and sample seed metadata.

### Node configurations

- `edge`: ordered contour samples (canonical edge representation)
- `edge_derivatives`: same ordered contour samples + derivative features (`tangent_x`, `tangent_r`, `curvature`, `curvature_gradient`)
- `edge_proximity`: all contour samples + interior nodes within configurable edge distance (no contour duplicates)
- `full`: all mesh nodes

### Run

Install dependencies:

```bash
pip install numpy scipy h5py scikit-fem matplotlib
```

Run generation (example):

```bash
python -m Data_gen.generate_dataset --num-samples 200 --seed 7 --output-dir Data_gen/output
```

Optional validation plots for first few samples:

```bash
python -m Data_gen.generate_dataset --num-samples 20 --save-validation-plots --validation-plot-count 3
```

Generate one deterministic high-quality example figure:

```bash
python -m Data_gen.plot_example_sample --seed 7 --output Data_gen/output/example_sample.png
```
