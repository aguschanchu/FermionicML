# =============================================================================
# tasks/t_rg_overlay.py — d=12 BCS panels + Richardson-Gaudin overlay
#
# Thin driver per the campaign task contract. Heavy lifting lives in
# engine_parts/p6_bcs_rg.py. Flow:
#   1. init_d12(const, gs) + init_gram; load d12_const checkpoint.
#   2. Uniform-panel target grid: C.RG_N_TARGETS couplings in [0.1, 1.0];
#      engine.bcs_uniform_inversions + engine.rg_canonical_inversion (same
#      grid) + OGN predictions on the SAME exact-ED targets.
#      -> bcs_const_order_parameter.* and bcs_const_interaction_rg.* figures
#         (OGN scatter + regularized/unregularized BCS + RG overlay on the
#          identity line).
#   3. init_d12(vect, gs) + init_gram; load d12_vect checkpoint;
#      engine.bcs_vect_inversions + OGN overlays for the vect panels.
#   4. rg_overlay.json: RG stats, OGN-vs-RG gap numbers, BCS_GAMMA
#      (value-from-code for manuscript l.120), figure paths.
# =============================================================================
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import engine


# ----------------------------------------------------------------- utilities
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _jsonable(o.item())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _pull_ckpt_from_gcs(model_name, ckpt_dir):
    if C.SMOKE:  # no GCS traffic in smoke runs
        return None
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        C.gcs_sync_dir(f"{C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name}",
                       ckpt_dir)
    except Exception:
        pass


def _load_checkpoint_state(model_name):
    """Rebuild the model from its config.MODELS spec and load final_state.msgpack
    (engine.save_model_and_history layout) via msgpack_restore + from_state_dict
    (no dependence on the training optimizer pytree). The matching era init_d12
    MUST already have been called (label_size/M_PAIRS come from era globals)."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization as flax_ser

    spec = C.MODELS[model_name]
    ckpt_dir = os.path.join(C.CKPT_DIR, model_name)
    state_path = os.path.join(ckpt_dir, "final_state.msgpack")
    if not os.path.exists(state_path):
        _pull_ckpt_from_gcs(model_name, ckpt_dir)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"checkpoint for {model_name!r} not found at "
                                f"{state_path} (local and GCS)")

    label_size = engine.g_gen.label_size()
    dim = engine.M_PAIRS                       # input_type 'rho2kkbar'
    input_shape = (1, dim, dim, 1)
    model = engine.build_model(
        spec.get("arch", "ogn"), label_size, spec.get("res", 3), True,
        use_scatter=spec.get("use_scatter", True),
        use_reinject=spec.get("use_reinject", True),
        use_orb_emb=spec.get("use_orb_emb", True),
        readout_bias=spec.get("readout_bias", 0.55))
    variables = model.init(jax.random.PRNGKey(0),
                           jnp.zeros(input_shape, jnp.float32),
                           jnp.zeros((1, 1), jnp.float32), training=False)
    with open(state_path, "rb") as f:
        raw = flax_ser.msgpack_restore(f.read())
    raw_params = raw.get("params", raw) if isinstance(raw, dict) else raw
    params = flax_ser.from_state_dict(variables["params"], raw_params)
    bs_tmpl = variables.get("batch_stats", {})
    raw_bs = raw.get("batch_stats", {}) if isinstance(raw, dict) else {}
    batch_stats = flax_ser.from_state_dict(bs_tmpl, raw_bs) if raw_bs else bs_tmpl
    state = engine.TrainState.create(apply_fn=model.apply, params=params,
                                     tx=optax.adamw(1e-4),
                                     batch_stats=batch_stats)
    return state


def _ogn_predict(state, rho_targets, energy_targets, chunk=256):
    """OGN labels for exact-ED targets: bx = rho2 pair block (B, m, m, 1),
    be = total energy (B, 1) — the same feature convention as the d12 loaders
    and the p5 protocol."""
    import jax.numpy as jnp
    label_size = engine.g_gen.label_size()
    rho = np.asarray(rho_targets, dtype=np.float32)
    if rho.ndim == 3:
        rho = rho[..., None]
    en = np.asarray(energy_targets, dtype=np.float32).reshape(-1, 1)
    preds = []
    for s in range(0, len(rho), chunk):
        bx = jnp.array(rho[s:s + chunk])
        be = jnp.array(en[s:s + chunk])
        by_dummy = jnp.zeros((bx.shape[0], label_size), jnp.float32)
        preds.append(np.array(engine.eval_step(state, bx, be, by_dummy)))
    return np.concatenate(preds, axis=0).astype(np.float64)


def _gs_rho2_from_G(G_mat, rho2_dense):
    """Exact ground-state pair RDM of H(G) via the era float64 sparse builder
    (engine._shots_h_dense64 uses g_gen.h_type) + dense eigh + compute_rho_m —
    the cell_37 rho_ml reconstruction, on documented p1/p5 APIs."""
    H = engine._shots_h_dense64(np.asarray(G_mat, dtype=np.float64))
    _, V = np.linalg.eigh(H)
    v0 = V[:, 0]
    rho_gs = np.outer(v0, v0)
    rdm = engine.compute_rho_m(np.array([rho_gs]), rho2_dense, 1)[0]
    return np.real(np.asarray(rdm)).squeeze()


# ----------------------------------------------------------------------- run
def run(out_dir):
    t0 = time.time()
    n_uniform = 6 if C.SMOKE else C.RG_N_TARGETS
    n_vect = 4 if C.SMOKE else 512
    g_grid = np.linspace(0.1, 1.0, n_uniform)
    figures = []

    # ------------------------------------------------ uniform (const) panel
    engine.init_d12(h_type="const", state_type="gs")
    engine.init_gram()
    state_const = _load_checkpoint_state("d12_const")

    uni = engine.bcs_uniform_inversions(g_grid, is_gs=True)
    # Same explicit grid for the RG canonical inversion -> OGN, BCS and RG
    # are compared on identical targets.
    rg = engine.rg_canonical_inversion(g_grid, seed=0)

    labels_const = _ogn_predict(state_const, uni["rho_targets"],
                                uni["energy_targets"])
    g_pred_const = labels_const[:, 0]           # 'const' label IS the coupling
    ogn_abs_err = np.abs(g_pred_const - uni["g_true"])

    prefix_c = os.path.join(out_dir, "bcs_const")
    written = engine.plot_bcs_uniform_panels(uni, prefix_c,
                                             g_pred=g_pred_const, rg=rg)
    # contract figure names: bcs_const_order_parameter.* stays as written;
    # bcs_const_interaction.* -> bcs_const_interaction_rg.* (RG overlay panel)
    for p in list(written):
        if os.path.basename(p).startswith("bcs_const_interaction."):
            q = p.replace("bcs_const_interaction.", "bcs_const_interaction_rg.")
            os.replace(p, q)
            written[written.index(p)] = q
    figures += written

    ogn_vs_rg = {
        "ogn_median_abs_err": float(np.median(ogn_abs_err)),
        "ogn_max_abs_err": float(ogn_abs_err.max()),
        "rg_median_abs_err": float(rg["median_abs_err"]),
        "rg_max_abs_err": float(rg["max_abs_err"]),
        "ogn_over_rg_median_ratio": float(
            np.median(ogn_abs_err) / max(rg["median_abs_err"], 1e-300)),
        "statement": (
            f"On the same {n_uniform}-point uniform grid G in [0.1, 1.0], the "
            f"exact Richardson-Gaudin canonical inversion reaches median "
            f"|G_rg - G_true| = {rg['median_abs_err']:.3e} "
            f"(max {rg['max_abs_err']:.3e}, "
            f"{int(rg['n_not_converged'])} non-converged), while the trained "
            f"d12_const OGN reaches median |G_ogn - G_true| = "
            f"{np.median(ogn_abs_err):.3e} (max {ogn_abs_err.max():.3e}); "
            f"the RG curve sits on the identity line and bounds the "
            f"information-theoretic floor of the uniform panel."),
    }

    # --------------------------------------------------- vectorial panel(s)
    engine.init_d12(h_type="vect", state_type="gs")
    engine.init_gram()
    state_vect = _load_checkpoint_state("d12_vect")

    vect = engine.bcs_vect_inversions(n_vect, seed=0)
    labels_vect = _ogn_predict(state_vect, vect["rho_targets"],
                               vect["energy_targets"])
    valid = np.asarray(vect["valid_mask"], dtype=bool)
    err_ml_all = np.linalg.norm(labels_vect - vect["g_true_labels"], axis=1)
    err_ml = err_ml_all[valid]

    # psi_ml per cell_37: order parameter of the exact RDM reconstructed from
    # the PREDICTED labels, normalized with the TRUE label vector.
    import jax.numpy as jnp
    rho2_dense = engine._safe_dense(engine.rho_2_kkbar_arrays)
    G_pred_mats = np.array(engine.g_gen.reconstruct(
        jnp.array(labels_vect, dtype=jnp.float32))).astype(np.float64)
    psi_ml = np.array([
        engine.get_order_parameter(
            _gs_rho2_from_G(G_pred_mats[i], rho2_dense),
            np.asarray(vect["g_true_labels"][i], dtype=np.float64))
        for i in np.where(valid)[0]])

    prefix_v = os.path.join(out_dir, "bcs_vect")
    figures += engine.plot_bcs_vect_panels(vect, prefix_v,
                                           err_ml=err_ml, psi_ml=psi_ml)

    # ------------------------------------------------------------- numbers
    result = {
        # value-from-code for manuscript l.120 (energy-term weight of the
        # composite BCS objective)
        "bcs_gamma": float(engine.BCS_GAMMA),
        "rg": {
            "n_targets": int(rg["n_targets"]),
            "median_abs_err": float(rg["median_abs_err"]),
            "max_abs_err": float(rg["max_abs_err"]),
            "n_not_converged": int(rg["n_not_converged"]),
            "gate_pass": bool(rg["gate_pass"]),
            "g_range": _jsonable(rg.get("g_range")),
            "wall_s": float(rg.get("wall_s", float("nan"))),
        },
        "bcs_uniform_summary": uni["summary"],
        "ogn_const": {
            "model": "d12_const",
            "n_targets": int(n_uniform),
            "median_abs_err": float(np.median(ogn_abs_err)),
            "max_abs_err": float(ogn_abs_err.max()),
        },
        "ogn_vs_rg": ogn_vs_rg,
        "bcs_vect_summary": vect["summary"],
        "ogn_vect": {
            "model": "d12_vect",
            "n_targets": int(n_vect),
            "n_valid": int(valid.sum()),
            "median_err_ml": float(np.median(err_ml)) if err_ml.size else None,
            "max_err_ml": float(err_ml.max()) if err_ml.size else None,
            "median_err_phys_bcs": vect["summary"]["median_err_phys"],
        },
        "figures": figures,
        "smoke": bool(C.SMOKE),
        "task_wall_s": time.time() - t0,
        "provenance": "referee-response campaign 2026-06; satelite2/campaign",
    }

    with open(os.path.join(out_dir, "rg_overlay.json"), "w") as f:
        json.dump(_jsonable(result), f, indent=1)

    # Raw overlay arrays for offline re-plotting.
    np.savez_compressed(
        os.path.join(out_dir, "rg_overlay_raw.npz"),
        g_true=uni["g_true"], g_bcs=uni["g_bcs"], g_bcs_alt=uni["g_bcs_alt"],
        g_pred_ogn=g_pred_const, rg_g_true=rg["g_true"], rg_g_rg=rg["g_rg"],
        psi_exact=uni["psi_exact"], psi_bcs=uni["psi_bcs"],
        psi_bcs_alt=uni["psi_bcs_alt"],
        vect_g_true_labels=vect["g_true_labels"],
        vect_labels_pred=labels_vect, vect_valid_mask=valid,
        vect_err_ml=err_ml, vect_psi_ml=psi_ml)

    payload = {
        "bcs_gamma": float(engine.BCS_GAMMA),
        "rg_median_abs_err": float(rg["median_abs_err"]),
        "rg_max_abs_err": float(rg["max_abs_err"]),
        "rg_n_not_converged": int(rg["n_not_converged"]),
        "rg_gate_pass": bool(rg["gate_pass"]),
        "ogn_const_median_abs_err": float(np.median(ogn_abs_err)),
        "vect_median_err_ml": (float(np.median(err_ml)) if err_ml.size
                               else None),
        "n_figures": len(figures),
        "task_wall_s": time.time() - t0,
    }
    return _jsonable(payload)
