# PR: issue #216, automatic bounds discovery for error-budgeting

## Summary

This work implements and checks automatic bounds discovery for issue #216 (`request: automatically find bounds in error-budgeting`). `get_error_budget` gains an optional `bounds` argument via `find_bounds_auto`. The hypotheses in the issue are checked on an independent testbed.

12/12 items validated (T1-T8, E1, E2, E4, E5).

## Changes

### New features

1. `find_bounds_auto(eps, shots, ...)` finds bounds for each noise parameter:
   - exponential growth of eps in log space
   - dual stop: Lambda sensitivity (ratio > 1.2) and p_L in a feasible Monte Carlo range
   - stop at eps_cap for insensitive parameters instead of widening forever

2. Optional `bounds` on `get_error_budget`. If omitted, `find_bounds_auto` supplies them.

3. Insensitive-parameter flag when the gradient is negligible and the bracket is wider than the support.

### Hypotheses validated

| Test | Hypothesis | Verdict | Key result |
|------|-----------|---------|------------|
| T1 | H2: CRN reduces variance | VALIDATED | 6.8x reduction (ratio 0.147) |
| T2 | H4: score function gradient | VALIDATED | 10x fewer shots, z < 2.0 |
| T3 | H3: reweighting + ESS | VALIDATED | 7 points from 1 campaign |
| T4 | H7: log parametrization | VALIDATED | R2(log)=0.992 vs R2(lin)=0.872 |
| T5 | H6: heteroscedasticity / WLS | VALIDATED | 19.0x sigma spread, WLS 69.5% better |
| T6 | H5: c-optimal vs Chebyshev | VALIDATED | 4.34x variance reduction |
| T7 | H8: Lambda validity guard | VALIDATED | 2 invalid points detected |
| T8 | PO2: Lambda second-level | VALIDATED | bootstrap/delta = 1.37 |
| E1 | find_bounds_auto | VALIDATED | all params bounded |
| E2 | optional bounds on the real API | VALIDATED | 1.12 sigma vs documented |
| E4 | reconciliation | VALIDATED | c-optimal wins |
| E5 | boundary regimes | VALIDATED | reproducible = True |

### Section 12 (Deltakit API)

Section 12 calls `deltakit.explorer.analysis.error_budget` on deltakit 0.9.0 (Python 3.12):

- Documented baseline: contribution = 0.4327 +- 0.0063 (bounds [0.001, 0.02])
- Auto-explored bounds: [0.00125, 0.01], containing p/2 = 0.0025 (gradient at noise_parameters / 2)
- Auto contribution: 0.4186 +- 0.0109
- Agreement: 1.12 sigma (AGREE)
- PO6: return exposes contributions and contribution_stddevs only, no bounds or diagnostics. Gap confirmed.

The optional-bounds signature works on the real pipeline. The return still needs uncertainty and diagnostics.

## Files

| File | Description |
|------|-------------|
| `deltakit_216_error_budgeting_suite.ipynb` | Adapted notebook (en-US, matplotlib/seaborn) |
| `outputs/REPORT.md` | Validation report |
| `outputs/results.json` | Numerical results (deltakit: ok=True, 1.12 sigma) |
| `outputs/DK_api_integration.csv` | Documented vs auto-explored table |
| `outputs/figures/DK_api_integration.png` | Brackets plus p/2 |
| `outputs/figures/*.png` | T1-T8, E1, E4, E5 |
| `outputs_quick_true/` | QUICK=True reference |

## Test environment

- OS: WSL Ubuntu-26.04 (Windows 11)
- Python: 3.14.4 (suite) / 3.12 (Section 12, deltakit 0.9.0)
- Packages: numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.1, seaborn 0.13.2, stim 1.16.0, pymatching 2.4.0
- Deltakit (Section 12): deltakit 0.9.0, deltakit-explorer 0.9.0 (requires Python <3.14)

## How to reproduce

```bash
# Build notebook
python build_notebook.py

# Run QUICK=True (~15 min)
source .venv-ubuntu/bin/activate
set_quick.py True
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    deltakit_216_error_budgeting_suite.ipynb

# Run QUICK=False (~60 min)
set_quick.py False
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    deltakit_216_error_budgeting_suite.ipynb

# Section 12 (real deltakit API) requires Python <3.14
uv venv .venv312 --python 3.12
uv pip install --python .venv312/Scripts/python.exe "deltakit==0.9.0"
.venv312/Scripts/python.exe section12_deltakit.py
```

## Community Fund proposal reading

1. T1 + T2 + T3 change the scope of the issue. If CRN, score function and reweighting work, the "narrow interval drowns signal in noise" dilemma is largely an artifact of independent sampling. For Pauli models the gradient can be obtained without a bracket. find_bounds_auto is the generic path, not the only path.
2. T6 + E4 treat the point the issue calls non-trivial by substitution, not reconciliation: Chebyshev answers a different question.
3. T7 adds a third constraint the issue does not list. Without it the stop criterion is pulled toward the region where Lambda stops existing.
4. T8 + PO6: the API return must carry uncertainty and diagnostics, not just a number.
