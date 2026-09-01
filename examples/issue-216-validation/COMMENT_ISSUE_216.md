# Comment for Deltakit issue #216

`request: automatically find bounds in error-budgeting`

## Summary

This notebook checks the hypotheses behind automatic bounds discovery for error-budgeting (issue #216). The suite ran on WSL Ubuntu-26.04 with Python 3.14.4 on a repetition-code testbed at the DEM level. Section 12 ran against the public Deltakit 0.9.0 API on Python 3.12.

Result: 12/12 items validated (T1-T8, E1, E2, E4, E5), on both QUICK=True and QUICK=False for the testbed, plus a live API call for E2.

## Hypotheses and evidence

### H2 - Common random numbers (T1)

Claim: the "narrow interval drowns signal in noise" dilemma is largely an artifact of independent sampling. Coupled streams cut Var(DeltaLambda) by about an order of magnitude.

Evidence (QUICK=False):

- Var_CRN / Var_IND = 0.147 (6.8x variance reduction)
- Pass criterion: < 0.2. VALIDATED

### H4 - Score function estimator (T2)

Claim: for Pauli noise the gradient follows from the score-function estimator without discretization or polynomial fitting. If that holds, bounds search is a fallback, not the core.

Evidence (QUICK=False):

- Score function agrees with coupled finite differences within 1 sigma
- Uses 10x fewer shots than finite differences
- Pass criterion: z-score < 2.0. VALIDATED

### H3 - Likelihood reweighting + ESS (T3)

Claim: sampling once at p* yields Lambda(p) over a neighborhood via likelihood ratio. The stop criterion becomes ESS (computable offline), so exploration and fitting points no longer fight each other.

Evidence (QUICK=False):

- 7 sweep points covered by one sampling campaign
- ESS_fail fraction range: 0.09 - 1.0
- Pass criterion: >= 3 points with < 15% error. VALIDATED

### H7 - Log vs linear parametrization (T4)

Claim: linear-scale loops violate positivity within a few steps for small p. In log space, exponential growth becomes uniform steps, positivity is automatic, and polynomial fits are better conditioned.

Evidence (QUICK=False):

- R2(log) = 0.992 vs R2(linear) = 0.872
- Pass criterion: R2_log >= R2_lin. VALIDATED

### H6 - Heteroscedasticity: WLS vs OLS (T5)

Claim: sigma_Lambda is not constant inside a bracket (p_L spans orders of magnitude). WLS cuts derivative variance by at least 20% versus OLS.

Evidence (QUICK=False):

- sigma_Lambda varies 19.0x inside the bracket
- WLS cuts variance by 69.5% (2903 -> 885)
- Pass criterion: spread >= 3x and WLS variance < OLS. VALIDATED

### H5, PO4 - Chebyshev vs c-optimal design (T6)

Claim: Chebyshev nodes minimize interpolation sup-norm under uniform weight. The quantity of interest is the derivative, so the right criterion is c-optimal design (Elfving 1952; Pukelsheim 2006).

Evidence (QUICK=False):

- c-optimal beats Chebyshev by 4.34x in derivative variance
- Pass criterion: ratio < 0.8. VALIDATED

### H8 - Lambda validity guard (T7)

Claim: p_L proportional to Lambda^(-(d+1)/2) is asymptotic and only holds well below threshold. The algorithm is pulled toward the region where Lambda has no meaning.

Evidence (QUICK=False):

- 2 points with high SNR and R2 < 0.95 (invalid model)
- Pass criterion: SNR grows while R2 degrades. VALIDATED

### PO2 - Lambda as a second-level estimator (T8)

Claim: Lambda is a fitted quantity, not an observable. Budget uncertainty goes through two nested fits. The delta method underestimates sigma_Lambda.

Evidence (QUICK=False):

- sigma_bootstrap / sigma_delta = 1.37 (delta method underestimates by 37%)
- Pass criterion: ratio > 1.2. VALIDATED

### E1 - find_bounds_auto (boundary regimes)

Claim: insensitive parameters must be flagged. The algorithm should stop, not widen forever.

Evidence:

- dummy0 (insensitive): stopped at eps_cap, insensitive=True, gradient = 0.0
- boundary (weakly sensitive): stopped at snr_satisfied, gradient = 152
- bulk (highly sensitive): stopped at snr_satisfied, gradient = -326
- Reproducible: same seed -> same bounds

### E4 - Reconciliation with fitting

Claim: the tension ("exploration points sit far from Chebyshev nodes") is resolved by substitution, not reconciliation. Chebyshev answers a different question.

Evidence:

- c-optimal design has the lowest derivative variance (356 vs 1163 for Chebyshev vs 929 for exponential reuse)

## Section 12 - Deltakit API integration

Status: VALIDATED (deltakit 0.9.0 + deltakit-explorer 0.9.0, Python 3.12)

The public API `deltakit.explorer.analysis.error_budget.get_error_budget` ran end to end:

- Documented baseline: contribution = 0.4327 +- 0.0063 (bounds [0.001, 0.02])
- Auto-explored bounds: [0.00125, 0.01], containing p = 0.005 and p/2 = 0.0025. The API evaluates the gradient at noise_parameters / 2.
- Auto contribution: 0.4186 +- 0.0109
- Agreement: 1.12 sigma (AGREE)
- PO6 check: the return exposes contributions and contribution_stddevs only. No bounds, diagnostics, shots_used, or snr. Diagnostics gap confirmed.

The optional-bounds signature works against the real pipeline. The return still lacks the diagnostics PO6 asks for.

## Comparison: QUICK=True vs QUICK=False

| Metric | QUICK=True (N_REP=12) | QUICK=False (N_REP=30) |
|--------|----------------------|------------------------|
| T1 ratio (CRN/IND) | 0.152 (6.6x) | 0.147 (6.8x) |
| T8 ratio (boot/delta) | 1.33 | 1.37 |
| T6 gain (Cheb/c-opt) | 4.28x | 4.34x |
| T5 sigma spread | 17.4x | 19.0x |
| E4 winner | c-optimal | c-optimal |

Both runs validate the testbed hypotheses. QUICK=False uses higher N_REP, N_BOOT, and SHOTS and tightens the same numbers.

## Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| E1 | find_bounds_auto with CRN + validity guard | Implemented |
| E2 | Optional bounds signature for get_error_budget | Validated (1.12 sigma vs documented) |
| E3 | Automatic bounds in log space with dual stop criterion | Implemented |
| E4 | c-optimal design replaces Chebyshev | Validated |
| E5 | Insensitive parameter detection + reproducibility | Validated |

## Artifacts

- `deltakit_216_error_budgeting_suite.ipynb` - adapted notebook (en-US)
- `outputs/REPORT.md` - consolidated report
- `outputs/results.json` - numerical results (deltakit: ok=True, 1.12 sigma)
- `outputs/DK_api_integration.csv` - documented vs auto-explored table
- `outputs/figures/DK_api_integration.png` - brackets plus p/2
- `outputs/figures/*.png` - T1-T8, E1, E4, E5
- `outputs_quick_true/` - QUICK=True reference
