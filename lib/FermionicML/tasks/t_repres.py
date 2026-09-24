"""Representability stress-test (June-2026 follow-up): run(out_dir).

Probes whether the rdm objective (which only ever sees N-representable 2-RDMs,
produced by exact diagonalization, and enforces representability of its own
forward-simulated RDM) yields a model that degrades more gracefully when the
INPUT 2-RDM is pushed OFF the N-representable manifold -- the regime the paper
flags as unprotected at inference (Sec. VI, the N-representability inference gap).

Contrast with the Sec. V.C finite-shot protocol: that noises the eigenvalue
POPULATIONS over the exact eigenbasis, so every perturbed input stays exactly
N-representable. Here we add Gaussian noise DIRECTLY to the pair-scattering block
(symmetrized), which generically violates N-representability (a measure-zero
boundary in operator space). We sweep the perturbation amplitude delta and, for
each trained model (gram_thermal_random_50 vs rdm_thermal_random_50), report the
median gauge-aligned parameter error vs the clean target Hamiltonian, plus a
PSD-violation proxy for how far outside representability each delta pushes.

Self-contained: reuses the checkpoint loader of t_shots and the val-loader of
t_gevp_diag; both live in the same tasks package.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import engine
from tasks import t_shots, t_gevp_diag

PROVENANCE = "referee-response campaign 2026-06 (follow-up); satelite2/campaign"
MODELS = ["gram_thermal_random_50", "rdm_thermal_random_50"]
# relative perturbation amplitudes (delta = fraction of the per-batch RMS of the
# clean pair block); 0.0 is the clean control.
DELTAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
N_EVAL = 512          # held-out samples per (model, delta)
EVAL_BS = 256


def _jsonable(o, _d=0):
    if _d > 10:
        return str(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v, _d + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v, _d + 1) for v in o]
    if isinstance(o, (str, bool)) or o is None:
        return o
    if isinstance(o, (int, float)):
        return o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 4096 else f"<ndarray {o.shape}>"
    return str(o)


def _gauge_aligned_rel_err(g_pred, g_true):
    """Per-sample diagonal-mean-gauge-aligned ||dG||_F / ||G_true||_F.

    Mirrors Sec. V.C: align the uniform diagonal-shift gauge before scoring.
    g_pred, g_true: (B, M, M) real arrays.
    """
    gp = np.asarray(g_pred, dtype=np.float64)
    gt = np.asarray(g_true, dtype=np.float64)
    m = gp.shape[-1]
    diag = np.arange(m)
    c = (gp[:, diag, diag].mean(axis=1) - gt[:, diag, diag].mean(axis=1))
    gp_aligned = gp.copy()
    gp_aligned[:, diag, diag] -= c[:, None]
    num = np.linalg.norm((gp_aligned - gt).reshape(len(gp), -1), axis=1)
    den = np.linalg.norm(gt.reshape(len(gt), -1), axis=1)
    den = np.where(den > 0, den, 1.0)
    return num / den


def _symmetrize_block(x):
    """Symmetrize over the two pair indices (axes 1,2), shape (B, M, M[, 1])."""
    if x.ndim == 4 and x.shape[-1] == 1:
        core = x[..., 0]
        core = 0.5 * (core + np.swapaxes(core, 1, 2))
        return core[..., None]
    if x.ndim == 3:
        return 0.5 * (x + np.swapaxes(x, 1, 2))
    return x


def _psd_violation_frac(x):
    """Fraction of samples whose symmetrized pair block is not PSD (a necessary
    condition for N-representability of the pair-scattering block). Proxy only."""
    core = x[..., 0] if (x.ndim == 4 and x.shape[-1] == 1) else x
    if core.ndim != 3:
        return None
    try:
        w = np.linalg.eigvalsh(0.5 * (core + np.swapaxes(core, 1, 2)))
        return float(np.mean(w.min(axis=1) < -1e-9))
    except Exception:
        return None


def _eval_model(model_name, out_dir):
    import jax.numpy as jnp
    spec = C.MODELS[model_name]
    state, _model, _cfg = t_shots._load_checkpoint_state(model_name)
    loader = t_gevp_diag._val_loader(spec)
    recon = engine.g_gen.reconstruct

    # collect up to N_EVAL clean (bx, be, by) once, in numpy
    bxs, bes, bys = [], [], []
    n = 0
    for bx, be, by in loader:
        bxs.append(np.asarray(bx))
        bes.append(np.asarray(be) if be is not None else None)
        bys.append(np.asarray(by))
        n += len(by)
        if n >= N_EVAL:
            break
    BX = np.concatenate(bxs, axis=0)[:N_EVAL]
    BY = np.concatenate(bys, axis=0)[:N_EVAL]
    have_e = all(b is not None for b in bes)
    BE = np.concatenate([b for b in bes], axis=0)[:N_EVAL] if have_e else None

    rms = float(np.sqrt(np.mean(BX.astype(np.float64) ** 2))) or 1.0
    curve, psd_frac = [], []
    for di, delta in enumerate(DELTAS):
        # deterministic perturbation, independent of model so both see the SAME
        # corrupted inputs (stream keyed on delta index only)
        rng = np.random.default_rng(np.random.SeedSequence([0, 7, di]))
        if delta == 0.0:
            BXp = BX
        else:
            noise = rng.standard_normal(BX.shape).astype(BX.dtype) * (delta * rms)
            BXp = _symmetrize_block(BX + noise)
        psd_frac.append(_psd_violation_frac(BXp))

        gp_l, gt_l = [], []
        for s in range(0, len(BXp), EVAL_BS):
            e = min(s + EVAL_BS, len(BXp))
            bxj = jnp.array(BXp[s:e])
            byj = jnp.array(BY[s:e])
            bej = jnp.array(BE[s:e]) if BE is not None else None
            logits = np.asarray(engine.eval_step(state, bxj, bej, byj))
            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            gp_l.append(np.asarray(recon(jnp.array(logits))))
            gt_l.append(np.asarray(recon(byj)))
        g_pred = np.concatenate(gp_l, axis=0)
        g_true = np.concatenate(gt_l, axis=0)
        rel = _gauge_aligned_rel_err(g_pred, g_true)
        curve.append({
            "delta": float(delta),
            "rel_err_median": float(np.median(rel)),
            "rel_err_p90": float(np.percentile(rel, 90)),
            "rel_err_mean": float(np.mean(rel)),
        })
    base = curve[0]["rel_err_median"] or 1e-12
    for c in curve:
        c["degradation_x"] = c["rel_err_median"] / base
    return {"model": model_name, "loss": spec["loss"], "n_eval": int(len(BX)),
            "input_rms": rms, "curve": curve, "psd_violation_frac": psd_frac}


def _plot(results, deltas, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for r in results:
            xs = [c["delta"] for c in r["curve"]]
            ys = [c["rel_err_median"] for c in r["curve"]]
            ax.plot(xs[1:], ys[1:], marker="o", lw=2, label=f"{r['loss']} ({r['model']})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"input perturbation $\delta$ (rel. to pair-block RMS)")
        ax.set_ylabel(r"median gauge-aligned $\|\Delta G\|/\|G\|$")
        ax.set_title("Representability stress-test: gram vs rdm")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "repres_stress.pdf")
        fig.savefig(path); fig.savefig(path[:-4] + ".png", dpi=200)
        plt.close(fig)
        return path
    except Exception as exc:
        return f"<plot failed: {exc}>"


def run(out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine.init_d20(h_type="random", state_type="thermal", beta=C.BETA_THERMAL)
    engine.init_gram()

    if C.SMOKE:
        global N_EVAL, DELTAS
        N_EVAL = 16
        DELTAS = (0.0, 1e-3, 1e-1)

    results = [_eval_model(m, out_dir) for m in MODELS]

    # head-to-head: rdm/gram median-error ratio at each delta (>1 => rdm worse)
    by_model = {r["model"]: r for r in results}
    ratio = []
    g = by_model.get("gram_thermal_random_50")
    d = by_model.get("rdm_thermal_random_50")
    if g and d:
        for cg, cd in zip(g["curve"], d["curve"]):
            ratio.append({"delta": cg["delta"],
                          "rdm_over_gram": (cd["rel_err_median"] /
                                            (cg["rel_err_median"] or 1e-12))})

    fig = _plot(results, DELTAS, out_dir)
    payload = {
        "task": "repres_stress",
        "provenance": PROVENANCE,
        "deltas": list(DELTAS),
        "models": MODELS,
        "results": results,
        "rdm_over_gram_ratio": ratio,
        "figure": fig,
        "smoke": bool(C.SMOKE),
        "wall_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "repres_stress.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=1)

    return _jsonable({
        "models": MODELS,
        "clean_rel_err": {r["model"]: r["curve"][0]["rel_err_median"]
                          for r in results},
        "degradation_x_at_max_delta": {r["model"]: r["curve"][-1]["degradation_x"]
                                       for r in results},
        "rdm_over_gram_ratio": ratio,
        "wall_s": time.time() - t0,
    })
