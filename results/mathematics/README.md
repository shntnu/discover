# Mathematics Results

The discovered constructions and the data behind the paper's mathematics figures.

## Discovered solutions

| File | Problem | Metric (paper) | Direction |
|------|---------|----------------|-----------|
| `ttt_erdos_sequence.json` | Erdős minimum overlap | C5 = 0.380876 | minimize |
| `ttt_ac1_sequence.json` | First autocorrelation inequality | C1 = 1.50287 | minimize |
| `ttt_ac2_sequence.json` | Second autocorrelation inequality | C2 = 0.9591 | maximize |

Each JSON holds the construction under the `sequence` key.

## Scoring them

The constructions are scored by the same evaluators the training loop uses.
From the repository root, with the math venv:

```bash
.venvs/math/bin/python scripts/validate.py math
```

This loads each `*_sequence.json`, runs it through the corresponding environment
evaluator (`examples.erdos_min_overlap.env`, `examples.ac_inequalities.env`),
and prints the measured constant next to the paper's value.

## Baselines and figures

- `human_best_erdos.py`, `erdos_data.py` - the Haugland and AlphaEvolve
  constructions used for comparison.
- `ac1_data.py`, `ac2_data.py` - prior-art sequences for the inequalities.
- `Results.ipynb` - regenerates the comparison plots
  (`erdos_comparison.pdf`, `ac1_comparison.pdf`, `ac1_overlay.pdf`).
