"""beta=100 ground-state-projector validation (referee Major item 6).

No model required.  For each GS family (random / const) on the d=20 era:
sample C.BETA100_NSAMPLES held-out Hamiltonians (seed C.VAL_SEED+1, distinct
from train 42/43 and val 1007), exact dense ED per sample in float64, and
report
  (a) spectral-gap E1-E0 distribution (min, p1, p5, median),
  (b) thermal weight 1-p0 at beta=100 (max, median),
  (c) ||rho2(beta=100) - rho2(T=0)||_F over the pair-scattering (kkbar)
      block -- the same RDM contraction the dataset generator uses
      (compute_rho_m / compute_rdm_trace einsum 'bji,kij->bk'), evaluated in
      float64 so the projector residual is resolvable below the float32 floor,
  (d) beta * Delta_gap minimum.
This discharges the projector-condition validation for the gs ensembles
(solve_batch_kernel applies effective_beta = 100.0 whenever is_thermal=False).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import engine

PROVENANCE = "referee-response campaign 2026-06; satelite2/campaign"


def _stats(a, percentiles=()):
    a = np.asarray(a, dtype=np.float64)
    out = {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def _figure(family, gaps, w_th, dfro, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].hist(gaps, bins=40, color="navy", alpha=0.85)
    axes[0].axvline(np.min(gaps), color="crimson", ls="--", lw=2,
                    label=f"min = {np.min(gaps):.3e}")
    axes[0].set_xlabel(r"$E_1 - E_0$")
    axes[0].set_ylabel("samples")
    axes[0].set_title("spectral gap")
    axes[0].legend()

    pos_w = w_th[w_th > 0]
    if pos_w.size:
        axes[1].hist(np.log10(pos_w), bins=40, color="forestgreen", alpha=0.85)
        axes[1].axvline(np.log10(np.max(pos_w)), color="crimson", ls="--",
                        lw=2, label=f"max = {np.max(w_th):.3e}")
        axes[1].legend()
    axes[1].set_xlabel(r"$\log_{10}(1 - p_0)$ at $\beta=100$")
    axes[1].set_title("thermal weight outside the GS")

    pos_d = dfro[dfro > 0]
    if pos_d.size:
        axes[2].hist(np.log10(pos_d), bins=40, color="darkorange", alpha=0.85)
        axes[2].axvline(np.log10(np.max(pos_d)), color="crimson", ls="--",
                        lw=2, label=f"max = {np.max(dfro):.3e}")
        axes[2].legend()
    axes[2].set_xlabel(
        r"$\log_{10}\|\rho^{(2)}_{\beta=100}-\rho^{(2)}_{T=0}\|_F$")
    axes[2].set_title("RDM projector residual")

    fig.suptitle(rf"$\beta=100$ ground-state projector validation "
                 rf"({family} family, d=20)")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)
    return out_prefix + ".png"


def _run(family, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine.init_d20(h_type=family, state_type="gs", beta=C.BETA_GS)
    beta = float(C.BETA_GS)

    n = 16 if C.SMOKE else int(C.BETA100_NSAMPLES)
    seed = int(C.VAL_SEED) + 1

    import jax
    import jax.numpy as jnp

    # held-out interaction stream (same GGenerator draw protocol as the era,
    # seed disjoint from the train/val dataset seeds)
    gen = engine.GGenerator(engine.basis, family, n,
                            g_init=C.G_INIT, g_stop=C.G_STOP)
    _, labels = gen.generate(jax.random.PRNGKey(seed))
    G_batch = np.asarray(gen.reconstruct(jnp.asarray(labels))
                         ).astype(np.float64)

    m = int(engine.M_PAIRS)
    ops = engine._ensure_dense(engine.rho_2_kkbar_arrays).astype(np.float64)
    ops_flat = ops.reshape(-1, ops.shape[-2], ops.shape[-1])  # (m*m, DN, DN)

    gaps = np.empty(n, dtype=np.float64)
    w_th = np.empty(n, dtype=np.float64)
    dfro = np.empty(n, dtype=np.float64)

    for i in range(n):
        # float64 dense many-body H from the era sparse builder (the float32
        # two_body_hamiltonian_dense path caps residual resolution at ~1e-5,
        # cf. RESULTS_prev.md sec 3)
        H = engine._shots_h_dense64(G_batch[i])
        evals, evecs = np.linalg.eigh(H)
        gaps[i] = float(evals[1] - evals[0])

        p = np.exp(-beta * (evals - evals[0]))
        Z = float(p.sum())
        w_th[i] = float(p[1:].sum() / Z)        # 1 - p0, cancellation-free

        rho_b = (evecs * (p / Z)) @ evecs.T      # thermal state at beta=100
        v0 = evecs[:, 0]
        rho_0 = np.outer(v0, v0)                 # T=0 ground-state projector

        # dataset-generation RDM contraction (cell_12 'bji,kij->bk'), float64
        r2_b = np.einsum("ji,kij->k", rho_b, ops_flat).reshape(m, m)
        r2_0 = np.einsum("ji,kij->k", rho_0, ops_flat).reshape(m, m)
        dfro[i] = float(np.linalg.norm(r2_b - r2_0))

    gap_stats = _stats(gaps, percentiles=(1, 5))
    w_stats = _stats(w_th)
    d_stats = _stats(dfro)

    fig_path = None
    try:
        fig_path = _figure(family, gaps, w_th, dfro,
                           os.path.join(out_dir, f"beta100_{family}_hist"))
    except Exception as exc:
        print(f"[t_beta100] figure failed for {family}: {exc}", flush=True)

    result = {
        "task": f"beta100_{family}",
        "provenance": PROVENANCE,
        "referee_item": "Major 6: beta=100 ground-state projector validation",
        "family": family,
        "ensemble": f"d20 / {family} / gs",
        "beta": beta,
        "n_samples": n,
        "seed": seed,
        "seed_note": "held-out stream PRNGKey(VAL_SEED+1); train 42/43, val 1007",
        "basis_dim": int(engine.basis.size),
        "gap": gap_stats,                                    # (a)
        "thermal_weight_1_minus_p0": w_stats,                # (b)
        "rho2_fro_beta100_vs_T0": d_stats,                   # (c)
        "beta_gap_min": beta * gap_stats["min"],             # (d)
        "rho2_block": "pair-scattering kkbar block (m x m), m = "
                      f"{m}; contraction 'bji,kij->bk' in float64",
        "h_builder": "two_body_hamiltonian_sp float64 dense "
                     "(_shots_h_dense64) + numpy.linalg.eigh",
        "figure": fig_path,
        "smoke": bool(C.SMOKE),
        "wall_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, f"beta100_{family}.json"), "w") as f:
        json.dump(result, f, indent=1)

    return {
        "family": family,
        "n_samples": n,
        "gap_min": gap_stats["min"],
        "gap_p1": gap_stats["p1"],
        "gap_p5": gap_stats["p5"],
        "gap_median": gap_stats["median"],
        "w_thermal_max": w_stats["max"],
        "w_thermal_median": w_stats["median"],
        "rho2_diff_max": d_stats["max"],
        "rho2_diff_median": d_stats["median"],
        "beta_gap_min": result["beta_gap_min"],
    }


def run_random(out_dir):
    return _run("random", out_dir)


def run_const(out_dir):
    return _run("const", out_dir)
