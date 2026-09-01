# Standalone Section 12: integration with the real Deltakit API (deliverable E2)
# Runs against deltakit 0.9.0 (Python 3.12). Reproduces the documented baseline,
# demonstrates the optional-bounds signature, and checks PO6 diagnostics exposure.
import os, json, math, time, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

QUICK = False
OUTPUT_DIR = r"C:\Users\armorking\Python6\DeltaKitPOC\outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def save_table(df, name):
    path = f"{OUTPUT_DIR}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path

def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
    return path

from deltakit_circuit.noise_channels import Depolarise1, Depolarise2
from deltakit_explorer.qpu import NoiseParameters, QPU
from deltakit.explorer.analysis.error_budget import get_error_budget, SamplingParameters

def simple_noise_model(circuit, noise_parameters):
    gate_noise = [
        lambda nc: Depolarise1.generator_from_prob(noise_parameters[0])(
            nc.gate_layer_qubits(None, gate_qubit_count=1)),
        lambda nc: Depolarise2.generator_from_prob(noise_parameters[0])(
            nc.gate_layer_qubits(None, gate_qubit_count=2)),
    ]
    qpu = QPU(circuit.qubits, noise_model=NoiseParameters(gate_noise=gate_noise))
    return qpu.compile_and_add_noise_to_circuit(circuit)

P_DK = [5e-3]
NUM_ROUNDS_DK = {d: [16] for d in [3, 5, 7]}
BOUNDS_DOC = [(1e-3, 2e-2)]
MAX_SHOTS_DK = 500_000  # QUICK=False equivalent

def get_error_budget_optional_bounds(noise_model, P, num_rounds, bounds=None,
                                     sampling_parameters=None, **kw):
    """Wrapper demonstrating the requested signature (optional bounds)."""
    auto = None
    if bounds is None:
        import stim as _stim, pymatching as _pm

        def probe_pL(p, d, rounds, shots=40_000):
            c = _stim.Circuit.generated("surface_code:rotated_memory_z",
                                        rounds=rounds, distance=d,
                                        after_clifford_depolarization=p)
            dem = c.detector_error_model(decompose_errors=True)
            m = _pm.Matching(dem)
            det, obs = c.compile_detector_sampler().sample(
                shots, separate_observables=True, bit_packed=False)
            pred = m.decode_batch(det)
            return float((pred != obs).any(axis=1).mean())

        # The real API computes the gradient at the half-point p/2 and requires the
        # bounds to strictly contain it (a < p/2 < p < b). Explore outward in log
        # space centred on p/2 (the evaluation point), stopping on the same pL
        # feasibility criteria used elsewhere in the suite.
        half_p = P[0] / 2.0
        eps = 0.2
        lo, hi = half_p, half_p
        for _ in range(12):
            lo, hi = half_p * math.exp(-eps), half_p * math.exp(eps)
            pl_lo = probe_pL(lo, max(num_rounds), num_rounds[max(num_rounds)][0])
            pl_hi = probe_pL(hi, min(num_rounds), num_rounds[min(num_rounds)][0])
            if pl_lo < 2e-4 or pl_hi > 0.40:
                break
            if pl_hi / max(pl_lo, 1e-9) > 20:
                break
            eps *= 1.5
        # Guarantee the bracket strictly contains p/2 and p (the API requirement).
        lo = min(lo, half_p * 0.5)
        hi = max(hi, P[0] * 2.0)
        bounds = [(lo, hi)]
        auto = dict(explored=True, bounds=bounds[0])
    r = get_error_budget(noise_model, P, num_rounds, bounds,
                         sampling_parameters=sampling_parameters, **kw)
    return r, bounds, auto

deltakit_result = dict(ok=False)
try:
    t0 = time.time()
    res_doc = get_error_budget(
        simple_noise_model, P_DK, NUM_ROUNDS_DK, BOUNDS_DOC,
        sampling_parameters=SamplingParameters(max_shots=MAX_SHOTS_DK))
    t_doc = time.time() - t0
    c_doc = float(res_doc.contributions[0])
    s_doc = float(res_doc.contribution_stddevs[0])
    print(f"documented baseline: contribution = {c_doc:.5f} +- {s_doc:.5f}  ({t_doc:.1f}s)")

    t0 = time.time()
    res_auto, bnds_auto, auto_diag = get_error_budget_optional_bounds(
        simple_noise_model, P_DK, NUM_ROUNDS_DK, bounds=None,
        sampling_parameters=SamplingParameters(max_shots=MAX_SHOTS_DK))
    t_auto = time.time() - t0
    c_auto = float(res_auto.contributions[0])
    s_auto = float(res_auto.contribution_stddevs[0])
    diff_sigma = abs(c_auto - c_doc) / math.sqrt(s_auto ** 2 + s_doc ** 2)

    print(f"automatic bounds  : {tuple(round(v, 6) for v in bnds_auto[0])}")
    print(f"documented bounds : (0.001, 0.02)")
    print(f"auto contribution : {c_auto:.5f} +- {s_auto:.5f}  ({t_auto:.1f}s)")
    print(f"\nDifference: {diff_sigma:.2f} sigma {'-> AGREE' if diff_sigma < 3 else '-> DIVERGE'}")

    # Bracket must contain the half-point p/2 (the gradient is computed at noise/2).
    half_p = P_DK[0] / 2.0
    contains_half = (bnds_auto[0][0] < half_p < bnds_auto[0][1])

    attrs = [a for a in dir(res_doc) if not a.startswith("_")]
    has_diag = any(k in attrs for k in ("bounds", "diagnostics", "shots_used", "snr"))
    print(f"\nAttributes of get_error_budget return: {attrs}")
    print(f"Exposes effective bounds/diagnostics? {has_diag}  "
          f"(PO6: {'ok' if has_diag else 'gap confirmed'})")

        contains_p_doc = (1e-3 < P_DK[0] < 2e-2)
        contains_p_auto = (bnds_auto[0][0] < P_DK[0] < bnds_auto[0][1])
        df_dk = pd.DataFrame([
            dict(method="manual_documented", bound_lo=1e-3, bound_hi=2e-2,
                 p=P_DK[0], p_over_2=half_p, contains_p=contains_p_doc,
                 contains_half_point=(1e-3 < half_p < 2e-2),
                 contribution=c_doc, stddev=s_doc, seconds=t_doc,
                 diff_sigma=0.0, agree=True),
            dict(method="auto_explored", bound_lo=bnds_auto[0][0], bound_hi=bnds_auto[0][1],
                 p=P_DK[0], p_over_2=half_p, contains_p=contains_p_auto,
                 contains_half_point=contains_half,
                 contribution=c_auto, stddev=s_auto, seconds=t_auto,
                 diff_sigma=float(diff_sigma), agree=bool(diff_sigma < 3)),
        ])
        save_table(df_dk, "DK_api_integration")

    # ---- Figure: documented vs auto bracket + half-point p/2 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (m, row) in zip([ax1, ax2], df_dk.iterrows()):
        color = "tab:blue" if m == 0 else "tab:orange"
        lo, hi = row.bound_lo, row.bound_hi
        ax.axvspan(lo, hi, color=color, alpha=0.22, label=f"bracket [{lo:.4g}, {hi:.4g}]")
        ax.axvline(P_DK[0], color="tab:green", lw=2, ls="--", label=f"noise p={P_DK[0]:.4g}")
        ax.axvline(half_p, color="crimson", lw=2, ls=":", label=f"grad point p/2={half_p:.4g}")
        ax.set_xscale("log")
        ax.set_title(row.method.replace("_", " ") + f"\ncontribution {row.contribution:.5f} ± {row.stddev:.5f}")
        ax.set_xlabel("noise parameter")
        ax.legend(fontsize=8)
    fig.suptitle("Error-budget integration with real deltakit API — documented vs auto-explored bounds", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "DK_api_integration")

    deltakit_result = dict(ok=True, diff_sigma=float(diff_sigma),
                           auto_bounds=[float(bnds_auto[0][0]), float(bnds_auto[0][1])],
                           documented_bounds=[1e-3, 2e-2],
                           contains_half_point=bool(contains_half),
                           half_point=float(half_p),
                           contribution_doc=float(c_doc), stddev_doc=float(s_doc),
                           contribution_auto=float(c_auto), stddev_auto=float(s_auto),
                           returns_diagnostics=bool(has_diag),
                           runtime_doc_sec=float(t_doc), runtime_auto_sec=float(t_auto))
except Exception as exc:
    print(f"deltakit section failed at runtime: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    deltakit_result = dict(ok=False, error=f"{type(exc).__name__}: {exc}")

# --- update results.json deltakit block ---
rj_path = os.path.join(OUTPUT_DIR, "results.json")
if os.path.exists(rj_path):
    with open(rj_path, "r") as f:
        rj = json.load(f)
    rj["deltakit"] = deltakit_result
    with open(rj_path, "w") as f:
        json.dump(rj, f, indent=2, default=str)
    print("[updated] results.json -> deltakit")

print("\nDELTAKIT_RESULT=" + json.dumps(deltakit_result))