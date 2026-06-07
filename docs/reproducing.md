# Reproducing Results

## Setup

To run a job, provision an environment with the correct Python version, then
install the required dependencies for the task you want to run.

- **gpu_mode** environments must use **Python 3.13.11**
- All other tasks are recommended to use **Python 3.11**

### Reproducible toolchain (uv + Nix)

The repository ships a pinned toolchain so the interpreter and CLI tools are
identical across machines:

- `flake.nix` / `flake.lock` provide `uv`, Python 3.11 and 3.13, the system
  libraries that PyPI binary wheels link against, and - on NixOS hosts - the
  system NVIDIA driver path so torch's CUDA wheels can find it. Enter the dev
  shell with `nix develop` (optional; skip if `uv` is already on your `PATH`).
- Each task gets its own isolated `uv` virtualenv. The frozen `requirements/`
  files remain the source of truth for exact dependency pins, so `uv pip
  install -r` reproduces the authors' tested environment pin-for-pin (no
  re-resolution).

```bash
# Default (Python 3.11) tasks: math, AHC, AtCoder, denoising, ...
uv venv .venvs/math --python 3.11
uv pip install --python .venvs/math/bin/python -r requirements/requirements-math.txt
.venvs/math/bin/python -m examples.<task_dir>.env

# gpu_mode tasks (Python 3.13)
uv venv .venvs/gpumode --python 3.13
uv pip install --python .venvs/gpumode/bin/python -r requirements/requirements-gpumode.txt
.venvs/gpumode/bin/python -m examples.gpu_mode.env
```

Run tasks with the venv's interpreter directly (as above), not `uv run` - in
this repo `uv run` would try to sync the project's own `pyproject.toml`
dependencies and re-resolve, defeating the frozen per-task pins.

A plain Conda or `venv` environment with the same Python version works equally
well - the `requirements/` files are the source of truth either way.

Each task has its own `requirements.txt` located under `requirements/`

Install dependencies with the application-specific requirements:
- Math: `requirements/requirements-math.txt`
- GPU kernels: `requirements/requirements-gpumode.txt`
- AtCoder: `requirements/requirements-ahc.txt`  
- Denoising: `requirements/denoising/requirements-denoising.txt` (see [README](requirements/denoising/README.md))

### Which venv for which task

The example `env.py` modules import their task-specific dependencies at import
time, so a task must be run with the venv built from its own requirements file.
Using the wrong venv fails with a `ModuleNotFoundError` (for example, the
denoising env needs `scprep`, which only the denoising venv has).

| Example module | Venv | Python |
|----------------|------|--------|
| `examples.erdos_min_overlap.env`, `examples.ac_inequalities.env`, `examples.circle_packing.env` | math | 3.11 |
| `examples.denoising.env` | denoising | 3.11 |
| `examples.ahc.env` | ahc | 3.11 |
| `examples.gpu_mode.env` | gpumode | 3.13 |

## Validate your setup without Tinker

The reward evaluators do not need Tinker, and the discovered solutions ship
under `results/`. Before launching a real (paid) run, confirm your environment
works by scoring those solutions with `scripts/validate.py`:

```bash
.venvs/math/bin/python      scripts/validate.py math        # fast, no GPU/network
.venvs/denoising/bin/python scripts/validate.py denoising   # downloads Pancreas (~90s)
.venvs/gpumode/bin/python   scripts/validate.py gpumode     # needs a local GPU
```

Each check prints the measured metric next to the value reported in the paper.

## Running Tasks

After installing dependencies, locate the corresponding application under `examples/`

Each application provides a script in `env.py` that launches the job. Run it
with the matching venv's interpreter (see the table above):

```
.venvs/<task>/bin/python -m examples.<task_dir>.env
```

Note that `python -m examples.<task_dir>.env` invokes `discover()`, which needs
the Tinker / Hugging Face / Weights & Biases credentials from the README.


## Getting Final Performance

We use the `raw_score` metric for logging our task performance. The final performance is typically the max (or min) of the `raw_score` metric, logged as `env/all/raw_score/max` (or `env/all/raw_score/min` for applications that minimize a value), **across all steps**. For example, in a training run for 50 steps of circle packing, if the max raw score at step 12 is 2.63 and no earlier or later step exceeds it, the final performance is still 2.63.

Some applications may require extra processing for our final results, such as denoising.

The following examples are maximization tasks: `second ac inequalities`, `circle packing`, and `AHC`. For performance, you should track the `env/all/raw_score/max`.

The following examples are minimization tasks: `first ac inequalities`, `erdos minimum overlap`, `denoising`, and `gpu mode`. For performance, you should track the `env/all/raw_score/min`.

## Multi-node Execution

Multi-node execution is supported via Slurm.

## Hardware Requirements and Performance Notes

All reported results were run using HPC-grade CPUs.

Mathematics and AHC tasks will perform significantly worse if they are not run on HPC-grade CPUs or if they are limited to a small number of cores. For these tasks, it is strongly recommended to use a large number of CPU cores and multiple hosts.

## GPU Mode (running locally without Modal)

By default `examples.gpu_mode.env` submits kernels to Modal, which needs Modal
credentials. The underlying GPU MODE evaluation harness, however, runs fine on a
local GPU - `scripts/validate.py gpumode` drives it directly. To run it by hand,
stage the harness files and a submission into one directory and invoke `eval.py`
with the popcorn contract:

```bash
cd examples/gpu_mode/lib/bioml/trimul
cp ../../../../../results/kernel-engineering/trimul.py submission.py
# tests.txt holds one "key: val; key: val" line per case (see task.yml `tests:`)
POPCORN_FD=1 POPCORN_SEED=42 python eval.py test tests.txt        # correctness
POPCORN_FD=1 POPCORN_SEED=42 python eval.py leaderboard tests.txt # timing (ns)
```

The leaderboard score is the geometric mean of the per-benchmark runtimes; the
ranking metric is meaningful across machines, but absolute microseconds vary by
H100 SKU (NVL vs SXM).

## AHC Container Requirements

For AHC tasks, jobs must be launched inside the ALE-Bench provided C++ container:

`yimjk/ale-bench:cpp20-202301`

Docker Hub:
https://hub.docker.com/layers/yimjk/ale-bench/cpp20-202301/images/sha256-946af1b209a84160594e43262b5157aec933938c99e2585b53042cac9bc3f43c

The container is required because the judge compiles submissions with
`g++-12 -DATCODER -I/opt/ac-library -I/opt/boost/gcc/include -lgmpxx -lgmp -I/usr/include/eigen3`,
and the case runner executes that command in the **same process environment**
(it does not shell out to a separate container per case). On a bare host without
that exact toolchain you get `COMPILATION_ERROR: g++-12 not found` for every
case. First download the cached test inputs and tester binaries
(`bash examples/ahc/get_cache.sh`), then launch the job from inside the
container.

We support the Pyxis Slurm plugin to launch this container across multiple nodes for AHC, but it is not strictly required.