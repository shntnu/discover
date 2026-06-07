#!/usr/bin/env python3
"""Validate a TTT-Discover install without Tinker.

The full ``discover()`` loop trains an LLM through Tinker and needs
``TINKER_API_KEY`` / ``HF_TOKEN`` / ``WANDB_*``. The *evaluation* half of the
pipeline - the per-task reward evaluators - is independent of all of that, and
this repo already ships the discovered state-of-the-art solutions under
``results/``. This script scores those released solutions with the same
evaluators the training loop uses, and prints the measured metric next to the
value reported in the paper.

Use it to confirm a freshly provisioned environment works (and reproduces the
headline numbers) before spending Tinker compute on a real run.

Each check needs the matching per-task venv (see docs/reproducing.md):

    .venvs/math/bin/python      scripts/validate.py math        # fast, no GPU/network
    .venvs/denoising/bin/python scripts/validate.py denoising   # downloads Pancreas (~90s)
    .venvs/gpumode/bin/python   scripts/validate.py gpumode     # needs a local GPU

Run from the repository root. Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
# Ensure `import examples.*` resolves when run as `python scripts/validate.py`.
sys.path.insert(0, str(REPO_ROOT))


def _line(name: str, measured: str, paper: str, ok: bool) -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:24s} measured={measured:<12s} paper={paper}")


def validate_math() -> bool:
    """Score the released Erdos and autocorrelation-inequality constructions.

    Pure CPU, no network or GPU. This is the recommended smoke test.
    """
    try:
        import numpy as np
        from examples.erdos_min_overlap.env import evaluate_erdos_solution
        from examples.ac_inequalities.env import (
            evaluate_sequence_ac1,
            evaluate_sequence_ac2,
        )
    except ModuleNotFoundError as e:
        print(f"  cannot run math checks ({e}). Use the math venv:")
        print("    .venvs/math/bin/python scripts/validate.py math")
        return False

    print("Mathematics (scoring results/mathematics/ with the env evaluators)")
    ok = True

    # Erdos minimum overlap: minimize C5. Paper SOTA 0.380876; beats AlphaEvolve 0.380924.
    seq = json.loads((RESULTS / "mathematics/ttt_erdos_sequence.json").read_text())["sequence"]
    h = np.asarray(seq, dtype=np.float64)
    c5 = evaluate_erdos_solution(h, _c5(h), len(h))
    passed = c5 < 0.380924  # strictly below previous-best AI
    _line("Erdos min overlap (C5)", f"{c5:.6f}", "0.380876 (min)", passed)
    ok &= passed

    # First autocorrelation inequality: minimize C1. Paper 1.50287.
    ac1 = _seq(RESULTS / "mathematics/ttt_ac1_sequence.json")
    c1 = evaluate_sequence_ac1(ac1)
    passed = c1 < 1.50314  # below previous-best AI
    _line("Autocorr AC1 (C1)", f"{c1:.5f}", "1.50287 (min)", passed)
    ok &= passed

    # Second autocorrelation inequality: maximize C2. Paper 0.9591.
    ac2 = _seq(RESULTS / "mathematics/ttt_ac2_sequence.json")
    c2 = evaluate_sequence_ac2(ac2)
    passed = c2 > 0.95  # clearly in range
    _line("Autocorr AC2 (C2)", f"{c2:.5f}", "0.9591 (max)", passed)
    ok &= passed

    return ok


def _c5(h):
    import numpy as np

    dx = 2.0 / len(h)
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))


def _seq(path: Path):
    d = json.loads(path.read_text())
    return d["sequence"] if isinstance(d, dict) and "sequence" in d else d


def validate_denoising() -> bool:
    """Run the released single-cell denoiser through the OpenProblems evaluator.

    Downloads the Pancreas dataset and runs MAGIC; takes ~90s. This exercises
    examples.denoising.utils.run_denoising_eval (the training-set eval wired
    into the environment) and checks the released solver beats the MAGIC
    baseline. It does not reproduce the held-out PBMC/Tabula leaderboard scores.
    """
    try:
        import importlib.util

        from examples.denoising.utils import run_denoising_eval
    except ModuleNotFoundError as e:
        print(f"  cannot run denoising check ({e}). Use the denoising venv:")
        print("    .venvs/denoising/bin/python scripts/validate.py denoising")
        return False

    print("Biology (released denoiser vs MAGIC baseline on Pancreas)")
    spec = importlib.util.spec_from_file_location(
        "denoise_ttt", RESULTS / "denoising/denoise_ttt.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mse, poisson = run_denoising_eval(mod.magic_denoise, seed=42)
    baseline_mse = 0.2316  # MAGIC baseline (examples/denoising/env.py initial state)
    passed = mse < baseline_mse
    _line("Denoising MSE", f"{mse:.4f}", f"< {baseline_mse} (MAGIC)", passed)
    print(f"         (poisson={poisson:.4f}; lower MSE is better)")
    return passed


def validate_gpumode() -> bool:
    """Compile and correctness-test the released TriMul kernel on a local GPU.

    Runs the GPU MODE eval harness in ``test`` mode directly (no Modal). Pass
    ``--leaderboard`` to also time it; absolute microseconds vary across H100
    SKUs (NVL vs SXM), so only the geometric mean is meaningful.
    """
    import os
    import re
    import shutil
    import subprocess
    import tempfile

    try:
        import torch

        sys.path.insert(0, str(REPO_ROOT / "examples/gpu_mode/lib"))
        from libkernelbot.run_eval import build_test_string
    except ModuleNotFoundError as e:
        print(f"  cannot run gpumode check ({e}). Use the gpumode venv (Python 3.13):")
        print("    .venvs/gpumode/bin/python scripts/validate.py gpumode")
        return False

    if not torch.cuda.is_available():
        print("  no CUDA device visible; the kernel check needs a local GPU.")
        return False

    print(f"Kernel engineering (TriMul correctness on {torch.cuda.get_device_name(0)})")
    src = REPO_ROOT / "examples/gpu_mode/lib/bioml/trimul"
    work = Path(tempfile.mkdtemp(prefix="ttt_validate_trimul_"))
    try:
        for f in ("eval.py", "task.py", "utils.py", "reference.py"):
            shutil.copy(src / f, work / f)
        shutil.copy(RESULTS / "kernel-engineering/trimul.py", work / "submission.py")

        import yaml

        task = yaml.safe_load((src / "task.yml").read_text())
        (work / "tests.txt").write_text(build_test_string(task["tests"]))

        out_path = work / "result_test.txt"
        with open(out_path, "w") as outf:
            env = dict(os.environ, POPCORN_FD=str(outf.fileno()), POPCORN_SEED="42")
            subprocess.run(
                [sys.executable, "eval.py", "test", "tests.txt"],
                env=env, pass_fds=[outf.fileno()], cwd=work, check=False,
            )
        report = out_path.read_text()
        n_pass = len(re.findall(r"test\.\d+\.status: pass", report))
        n_total = int(re.search(r"test-count: (\d+)", report).group(1)) if "test-count" in report else n_pass
        passed = "check: pass" in report and n_pass == n_total and n_total > 0
        _line("TriMul correctness", f"{n_pass}/{n_total}", "all pass", passed)
        return passed
    finally:
        shutil.rmtree(work, ignore_errors=True)


CHECKS = {
    "math": validate_math,
    "denoising": validate_denoising,
    "gpumode": validate_gpumode,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "task", nargs="?", default="math", choices=[*CHECKS, "all"],
        help="which validation to run (default: math)",
    )
    args = parser.parse_args()

    tasks = list(CHECKS) if args.task == "all" else [args.task]
    results = {t: CHECKS[t]() for t in tasks}

    print("\nSummary:")
    for t, ok in results.items():
        print(f"  {t:12s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
