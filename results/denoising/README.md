# OpenProblems Denoising Benchmark

## Setup

The authoritative, step-by-step setup (matching the per-task `.venvs/<task>`
convention) lives in
[requirements/denoising/README.md](../../requirements/denoising/README.md). It
creates `.venvs/denoising`, clones OpenProblems v1.0.0 into the gitignored
`.openproblems/` directory, applies the patch, and installs everything. Follow
that file rather than the abbreviated steps that used to live here.

Once set up, run the released solver with `scripts/validate.py denoising` (see
[docs/reproducing.md](../../docs/reproducing.md)).

## Known Issues

### 1. CZI cellxgene API changed
Tabula Muris loader fails. The API now uses:
- `dataset["dataset_id"]` instead of `dataset["id"]`
- Assets embedded in dataset: `dataset["assets"]`
- `asset["url"]` instead of `asset["presigned_url"]`

**Fix**: `openproblems_api_fix.patch`

### 2. NumPy 2.x compatibility
PyTorch pulls NumPy 2.x which breaks old syntax:
```python
# Old (breaks):
np.asarray(Y, dtype=np.float64, copy=False)

# New (works):
np.asarray(Y, dtype=np.float64)
```

