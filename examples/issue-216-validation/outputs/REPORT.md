# Deltakit issue #216 - Validation suite report

QUICK=False | SHOTS=80000 | SHOTS_BIG=240000 | N_REP=30

## Verdicts

| Test | Hypothesis / PO | #216 Deliverable | Verdict |
|------|-----------------|------------------|---------|
| T1 | H2  - Common Random Numbers | E1, E3, E5 | VALIDATED |
| T2 | H4  - Score function | E1, E2, E4, E5 | VALIDATED |
| T3 | H3  - Reweighting + ESS | E1, E3, E4 | VALIDATED |
| T4 | H7  - Log parametrization | E3, E5 | VALIDATED |
| T5 | H6  - Heteroscedasticity / WLS | E1, E3, E4 | VALIDATED |
| T6 | H5, PO4 - Chebyshev vs c-optimal | E4 | VALIDATED |
| T7 | H8  - Lambda validity guard | E1, E3 | VALIDATED |
| T8 | PO2 - Lambda as 2nd-level estimator | E1 | VALIDATED |
| E4 | Reconciliation with the fitting | E4 | VALIDATED |
| E5 | Boundary regimes + reproducibility | E5 | VALIDATED |

## Key numbers

- **T1 (CRN):** variance of DeltaLambda drops 6.8x under coupling (ratio 0.147).
- **T2 (score function):** agrees with finite differences using 10x fewer shots.
- **T3 (reweighting):** 7 sweep points covered by a single sampling campaign.
- **T5 (heteroscedasticity):** sigma_Lambda varies 19.0x within the bracket.
- **T6 (design):** c-optimal beats Chebyshev by 4.34x in derivative variance.
- **T7 (validity):** 2 points with high SNR and Lambda-model R2 below 0.95.
- **T8 (2nd level):** sigma bootstrap / sigma delta-method = 1.37.
- **E1 (autobound):** bounds and stop criteria per parameter: {'boundary': 'snr_satisfied', 'bulk': 'snr_satisfied', 'dummy0': 'eps_cap'}.
- **E4 (reconciliation):** winning strategy = C_c_optimal.
- **E5 (boundary):** reproducible = True.

## Integration with the Deltakit API

- section skipped: import failed / unavailable

## Reading for the Community Fund proposal

1. **T1 + T2 + T3 change the scope of the issue.** If CRN, score function and
   reweighting work, the central dilemma ('narrow interval drowns in noise')
   is in large part an artifact of independent sampling, and the gradient can be
   obtained without a bracket for Pauli models. An honest proposal states this and
   positions `find_bounds_auto` as the generic path, not the only path.
2. **T6 + E4 resolve the point the issue marks as non-trivial** by substitution,
   not reconciliation: Chebyshev answers a different question.
3. **T7 adds a third constraint** the issue does not list and without which the
   stop criterion is attracted to the region where Lambda stops existing.
4. **T8 + PO6** indicate the API return must carry uncertainty and diagnostics,
   not just a number.
