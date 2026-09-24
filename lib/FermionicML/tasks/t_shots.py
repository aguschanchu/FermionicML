# =============================================================================
# tasks/t_shots.py — Sec V.C finite-shot protocol (manuscript log_inversion_wls)
#
# Thin driver per the campaign task contract: era init (d20 random/thermal,
# beta=1) + init_gram, load the prod_thermal_random checkpoint, run the p5
# protocol twice (gaussian + multinomial) on IDENTICAL held-out systems and
# seeded draws (seed 0), p0 statistics, curve analysis, figure regeneration,
# and shots.json in the values_prev.json 'shots' schema.
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

MODEL_NAME = "prod_thermal_random"


# ----------------------------------------------------------------- utilities
def _jsonable(o):
    """Recursively convert numpy containers/scalars; NaN/inf -> None."""
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
    """Best-effort GCS pull when the local checkpoint dir is empty (resume)."""
    if C.SMOKE:  # no GCS traffic in smoke runs
        return None
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        C.gcs_sync_dir(f"{C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name}",
                       ckpt_dir)
    except Exception:
        pass


def _load_checkpoint_state(model_name):
    """Rebuild the model from its config.MODELS spec and load the final
    TrainState saved by engine.save_model_and_history (final_state.msgpack).

    Deserializes via msgpack_restore + from_state_dict so the load does NOT
    depend on reproducing the training optimizer pytree (eval-only TrainState
    with a placeholder tx). Returns (state, model, ckpt_config_dict).
    """
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
        raise FileNotFoundError(
            f"checkpoint for {model_name!r} not found at {state_path} "
            f"(local and GCS {C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name})")

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

    state = engine.TrainState.create(
        apply_fn=model.apply, params=params,
        tx=optax.adamw(1e-4),                  # placeholder; eval-only state
        batch_stats=batch_stats)

    cfg = {}
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return state, model, cfg


# ----------------------------------------------------------------------- run
def run(out_dir, model_name=MODEL_NAME):
    t0 = time.time()
    spec = C.MODELS[model_name]
    engine.init_d20(h_type=spec["h_type"], state_type=spec["state_type"],
                    beta=C.BETA_THERMAL)
    engine.init_gram()
    state, _model, ckpt_cfg = _load_checkpoint_state(model_name)

    if C.SMOKE:
        n_states, n_noise, n_pts = 4, 4, 5
    else:
        n_states, n_noise, n_pts = (C.SHOTS_N_STATES, C.SHOTS_N_NOISE,
                                    C.SHOTS_SWEEP_NPTS)
    sweep = np.geomspace(C.SHOTS_SWEEP_HI, C.SHOTS_SWEEP_LO, n_pts)

    # Same seed=0 for both noise models: the held-out systems are identical
    # and the per-(shot, state, trial) streams differ only by the documented
    # stream id (2 = gaussian, 3 = multinomial).
    gauss = engine.run_shots_protocol(
        state, n_states=n_states, n_noise=n_noise, n_shots_sweep=sweep,
        noise_model="gaussian", seed=0, include_energy=True)
    multi = engine.run_shots_protocol(
        state, n_states=n_states, n_noise=n_noise, n_shots_sweep=sweep,
        noise_model="multinomial", seed=0, include_energy=True)

    p0 = engine.p0_statistics(n_states, seed=0)
    analysis = engine.analyze_curves(gauss, multi)

    fig_prefix = os.path.join(out_dir, "log_inversion_wls")
    fig_path = engine.plot_shots_figure(gauss, multi, analysis, fig_prefix)

    # Raw per-(shots, state, draw) arrays for offline re-analysis.
    np.savez_compressed(
        os.path.join(out_dir, "shots_raw.npz"),
        n_shots_sweep=gauss["n_shots_sweep"], epsilons=gauss["epsilons"],
        gauss_err_wls=gauss["err_wls"], gauss_err_ogn=gauss["err_ogn"],
        gauss_n_censored=gauss["n_censored"],
        multi_err_wls=multi["err_wls"], multi_err_ogn=multi["err_ogn"],
        multi_n_censored=multi["n_censored"],
        noiseless_wls_err=gauss["noiseless_wls_err"],
        p0_per_state=gauss["p0_per_state"],
        norm_g_true=gauss["norm_g_true"])

    hist = ckpt_cfg.get("hist")
    if isinstance(hist, dict):
        ckpt_epochs = len(hist.get("loss") or [])
    elif isinstance(hist, (list, tuple)):
        ckpt_epochs = len(hist)
    else:
        ckpt_epochs = None

    shots = dict(analysis)                      # values_prev.json 'shots' schema
    shots["p0_stats"] = p0                      # referee request: p0 at beta=1
    shots["protocol"] = {
        "model": model_name,
        "checkpoint_epochs": ckpt_epochs,
        "n_states": int(n_states), "n_noise": int(n_noise),
        "n_shots_sweep_pts": int(n_pts),
        "sweep_hi": float(C.SHOTS_SWEEP_HI), "sweep_lo": float(C.SHOTS_SWEEP_LO),
        "seed": 0, "beta": float(engine.BETA),
        "heldout_seed_base": int(engine.SHOTS_HELDOUT_SEED),
        "wls_drop_p": float(C.WLS_DROP_P),
        "wls_spectral_cutoff": float(C.WLS_SPECTRAL_CUTOFF),
        "smoke": bool(C.SMOKE),
        "provenance": "referee-response campaign 2026-06; satelite2/campaign",
    }
    shots["figure"] = fig_path

    with open(os.path.join(out_dir, "shots.json"), "w") as f:
        json.dump(_jsonable(shots), f, indent=1)

    gates = analysis.get("gates", {})
    payload = {
        "crossover_Nshots": analysis.get("crossover_Nshots"),
        "crossover_eps": analysis.get("crossover_eps"),
        "crossover_extrapolated": analysis.get("crossover_extrapolated"),
        "advantage_window_shots": analysis.get("advantage_window_shots"),
        "wls_slope": analysis.get("wls_slope"),
        "ogn_plateau": analysis.get("ogn_plateau"),
        "p0_median": p0.get("p0_median"),
        "gates_pass": {k: (v.get("pass") if isinstance(v, dict) else v)
                       for k, v in gates.items()},
        "n_states": int(n_states), "n_noise": int(n_noise),
        "task_wall_s": time.time() - t0,
    }
    return _jsonable(payload)


# --- June-2026 follow-up: finite-shot protocol on the converged rdm/gram-50
#     checkpoints (same WLS baseline, same seeded draws). ---
def run_rdm(out_dir):
    return run(out_dir, model_name="rdm_thermal_random_50")


def run_gram50(out_dir):
    return run(out_dir, model_name="gram_thermal_random_50")
