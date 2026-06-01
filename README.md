This is Abdulla's repo for the first publication for geometry aware ML lifing models.

## Synthetic dataset generator (`Data_gen`)

This repository now includes a Python-based synthetic data generator for a 2D axisymmetric rotor-disc meridional cross-section, focused on **data generation only**.

### What it generates

For each sample, the generator builds a parameterized disc geometry with three semantic regions:

- `bore`
- `web`
- `rim`

It then computes per-node targets across a fixed 7-phase mission cycle (rotation-only):

- `stress_max_vm` (max equivalent stress over cycle)
- `life_raw` (raw life via phase-wise damage accumulation, region-specific nonlinear S-N law)
- `phase_stress_eq` (equivalent stress for each phase)

No normalization or noise is applied.

### Outputs

Running the generator writes four HDF5 files in the output directory:

- `disc_dataset_edge.h5`
- `disc_dataset_edge_derivatives.h5`
- `disc_dataset_edge_proximity.h5`
- `disc_dataset_full.h5`

All files contain the same sample IDs and include node coordinates, region IDs, targets, phase-wise stresses, geometry parameters, cycle metadata, and seed metadata.

### Node configurations

- `edge`: boundary nodes only
- `edge_derivatives`: boundary nodes + ordered-contour derivative features (`tangent_x`, `tangent_r`, `curvature`, `second_derivative_like`)
- `edge_proximity`: boundary nodes + interior nodes within configurable edge distance
- `full`: all geometry nodes

### Run

Install dependencies:

```bash
pip install numpy scipy h5py scikit-fem matplotlib
```

Run generation (example):

```bash
python -m Data_gen.generate_dataset --num-samples 200 --seed 7 --output-dir Data_gen/output
```

Optional lightweight plots for first few samples:

```bash
python -m Data_gen.generate_dataset --num-samples 20 --save-validation-plots --validation-plot-count 3
```