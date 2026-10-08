# TopoDiff-DGA

Official implementation of **TopoDiff-DGA: Topology-Aware Diffusion for Drug–Gene Association Prediction**.

TopoDiff-DGA reconstructs a complete gene-indexed association profile for each target drug by combining topology-aware profile diffusion with drug-specific context.

## Method-to-code correspondence

- `degree_corrected_gene_graph` constructs the degree-normalized G–D–G gene graph from the training association matrix.
- `GraphFilter` combines the normalized local graph with its leading spectral subspace and implements the scheduled forward filters.
- `DrugContextEncoder` aggregates the top D–G–D neighbors of each target drug.
- `ConditionalDenoiser` uses the diffused profile, masked input profile, drug context, and diffusion-step embedding to predict the clean association profile.
- `TopoDiffDGA.reconstruct` applies the reverse update and returns one score for every gene.

Only training associations are used to construct the two graph views. A zero in the association matrix is treated as an unknown association, not as a confirmed negative.

## Repository layout

```text
src/topodiff_dga/       model, graph construction, data loading, and metrics
data/dgidb/folds/       DGIdb 4.0 train/test matrices
data/dgidb/split/       DGIdb 4.0 train/validation/test matrices
tests/                  small end-to-end consistency test
train.py                training and evaluation entry point
predict.py              full-gene ranking from a saved checkpoint
```

The bundled DGIdb matrices are stored as SciPy sparse matrices in genes-by-drugs layout. BindingDB and ChEMBL are not redistributed; processed matrices can be supplied as SciPy `.npz`, NumPy `.npy`, dense `.csv`/`.tsv`, or pickled SciPy matrices.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## DGIdb 4.0 example

```bash
python train.py \
  --train-matrix data/dgidb/folds/train_mat_fold0 \
  --test-matrix data/dgidb/folds/test_mat_fold0 \
  --layout genes-by-drugs \
  --output outputs/dgidb_fold0
```

To use the provided validation split:

```bash
python train.py \
  --train-matrix data/dgidb/split/train_mat \
  --validation-matrix data/dgidb/split/val_mat \
  --test-matrix data/dgidb/split/test_mat \
  --layout genes-by-drugs \
  --output outputs/dgidb_split
```

The defaults match the settings stated in the paper: hidden dimension 128, batch size 400, 30 graph neighbors, spectral rank 128, masking probability 0.2, 100 diffusion steps, Adam learning rate 0.001, weight decay $10^{-5}$, and early stopping patience 20. The remaining paper search variables are exposed as command-line arguments, including `--omega`, `--positive-weight`, and `--association-weight`.

## Candidate-gene ranking

```bash
python predict.py \
  --checkpoint outputs/dgidb_fold0/best_model.pt \
  --train-matrix data/dgidb/folds/train_mat_fold0 \
  --layout genes-by-drugs \
  --drug-ids 0 1 2 \
  --top-k 100 \
  --output outputs/dgidb_fold0/rankings.json
```

The ranking contains genes that are not linked to the requested drug in the training matrix, ordered by the reconstructed association score.

## Data preparation

Provide three binary matrices with the same shape. Rows must identify drugs and columns must identify genes when `--layout drugs-by-genes` is used. With `--layout genes-by-drugs`, the loader transposes the matrices. Training, validation, and test matrices should contain disjoint positive associations.

## Test

```bash
pytest
```

## Citation

Please cite the TopoDiff-DGA paper if this repository is useful in your work. The full bibliographic entry will be added after publication.
