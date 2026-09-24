# =============================================================================
# tasks/t_ablation_eval.py — [ABL-REF] ablation evaluation + gamma_base probe
#
# run(out_dir):
#   For prod_thermal_random + every abl_* model in C.MODELS (+ the
#   kernel_ridge baseline via its saved predictions): clean-data metrics on
#   the SAME held-out set (thermal val ds, seed C.VAL_SEED, shuffle=False,
#   first 4096 samples), f_err/eps_I via engine.drive_panel on a 512-sample
#   subset, and finite-shot rows on IDENTICAL seeded noisy draws (one shared
#   precomputed input set; the model is the only thing that varies). Also the
#   cell_26-style gram-vs-rdm paired-by-seed verdict. Writes ablation.json,
#   tab_ablation.tex, claim_map.tex.
#
# run_gbase_probe(out_dir):
#   cell_26-style gamma_base gradient-norm probe: 4 short rdm-loss runs
#   (gamma_base in {0, 1e-6, 1e-4, 1e-2}, 2 epochs on 200k samples) with the
#   per-component gradient-norm probe_cb (engine.get_probe_batch +
#   engine.make_component_probe). Exhibits the 'confirmed' sweep of
#   manuscript l.542. Writes gbase_probe.json + figure.
# =============================================================================
import gc
import inspect
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

ENSEMBLE_MODEL = "prod_thermal_random"
DS_ROOT_KEY = "ds_d20_random_thermal"
GAMMA_RDM, GAMMA_TRACE = 100.0, 1.0            # production rdm-loss weights
GBASE_SWEEP = (0.0, 1e-6, 1e-4, 1e-2)          # manuscript l.542 sweep
GRAM_BY_SEED = {42: "prod_thermal_random", 43: "abl_gram_s43",
                44: "abl_gram_s44"}
RDM_BY_SEED = {42: "abl_rdm_s42", 43: "abl_rdm_s43", 44: "abl_rdm_s44"}

# manuscript architecture-attribution sentences mapped by claim_map.tex
CLAIM_SENTENCES = [
    ("main.tex l.632", "``genuine geometric regularization''",
     "$\\varepsilon_I$/$f_{\\rm err}$ of prod\\_thermal\\_random vs "
     "abl\\_mlp (params-matched) and the structural toggles "
     "(abl\\_noscatter / abl\\_noreinject / abl\\_noembed / "
     "abl\\_neutralbias)", "arch_clean"),
    ("main.tex caption l.655", "``implicit geometric regularizer''",
     "finite-shot rows: prod\\_thermal\\_random vs abl\\_mlp and toggles at "
     "budgets $10^4$--$10^8$ (identical seeded draws)", "arch_shots"),
    ("main.tex l.665", "``effective implicit geometric regularization''",
     "finite-shot rows (same as caption l.655)", "arch_shots"),
    ("main.tex l.677", "``Acting as an implicit geometric regularizer''",
     "finite-shot rows + clean-data parameter error (prod vs abl\\_mlp, "
     "kernel\\_ridge)", "arch_both"),
    ("main.tex l.681", "``useful implicit regularization'' "
     "(pairing-structured graph models)",
     "structural toggles vs prod (does removing scatter / reinjection / "
     "orbital embedding / readout bias measurably hurt?) + abl\\_mlp",
     "structure_matters"),
]


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


def _bind_call(fn, pool, label):
    """Call fn binding parameters BY NAME from a candidate pool (the p3/p4
    engine parts are authored separately; this keeps the driver thin and the
    binding explicit). Raises with a clear message when a required parameter
    cannot be supplied."""
    sig = inspect.signature(fn)
    kwargs, missing = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if name in pool:
            kwargs[name] = pool[name]
        elif p.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(f"{label}: cannot bind required parameter(s) "
                        f"{missing}; pool offers {sorted(pool)}")
    return fn(**kwargs)


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
    """Rebuild the model from its config.MODELS spec; load final_state.msgpack
    (engine.save_model_and_history layout) via msgpack_restore +
    from_state_dict (independent of the training optimizer pytree).
    Returns (state, model, ckpt_config_dict)."""
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
    dim = engine.M_PAIRS
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
    cfg = {}
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return state, model, cfg


def _ensure_val_dataset():
    spec = C.MODELS[ENSEMBLE_MODEL]
    gen_bs = 64 if C.SMOKE else 4096
    val_dir = os.path.join(C.DATA_DIR, DS_ROOT_KEY, "val")
    n_val = max(gen_bs, int(0.05 * spec["num_samples"]))
    engine.gen_dataset(spec["h_type"], C.G_INIT, C.G_STOP,
                       spec["state_type"], "rho2kkbar", True, C.BETA_THERMAL,
                       num_samples=n_val, cache_path=val_dir,
                       batch_size=gen_bs, seed=C.VAL_SEED)
    return val_dir


def _collect_heldout(val_dir, n):
    """First n samples of the val stream, deterministic (shuffle=False)."""
    load_bs = 64 if C.SMOKE else 2048
    loader = engine.NumpyLoader(val_dir, load_bs, shuffle=False)
    rd, en, yt = [], [], []
    got = 0
    for bx, be, by in loader:
        x = np.asarray(bx, dtype=np.float32)
        if x.ndim == 3:
            x = x[..., None]
        rd.append(x)
        en.append(np.asarray(be, dtype=np.float32).reshape(len(x), -1)[:, :1])
        yt.append(np.asarray(by, dtype=np.float32))
        got += len(x)
        if got >= n:
            break
    if got < n:
        raise RuntimeError(f"val loader yielded only {got} < {n} samples")
    return (np.concatenate(rd)[:n], np.concatenate(en)[:n],
            np.concatenate(yt)[:n])


def _predict(state, rdms, energies, chunk=256):
    import jax.numpy as jnp
    label_size = engine.g_gen.label_size()
    preds = []
    for s in range(0, len(rdms), chunk):
        bx = jnp.array(rdms[s:s + chunk])
        be = jnp.array(energies[s:s + chunk])
        by_dummy = jnp.zeros((bx.shape[0], label_size), jnp.float32)
        preds.append(np.array(engine.eval_step(state, bx, be, by_dummy)))
    return np.concatenate(preds, axis=0).astype(np.float64)


def _gauge_aligned_errors(g_pred, g_true, chunk=512):
    import jax.numpy as jnp
    n = len(g_pred)
    errs = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        P = np.array(engine.g_gen.reconstruct(
            jnp.array(np.asarray(g_pred[s:e], dtype=np.float32)))
        ).astype(np.float64)
        T = np.array(engine.g_gen.reconstruct(
            jnp.array(np.asarray(g_true[s:e], dtype=np.float32)))
        ).astype(np.float64)
        for i in range(e - s):
            errs[s + i] = engine._shots_shifted_error(
                P[i], T[i], np.linalg.norm(T[i]))
    return errs


def _rel_S_errors(g_pred, g_true, chunk=512):
    """Relative S-metric (Gram) error sqrt(dG S dG / G S G) per sample
    (cell_26 convention; NOT f_err — that comes from drive_panel)."""
    import jax.numpy as jnp
    S = None
    for nm in ("S_matrix_np", "S_matrix", "S_matrix_global"):
        S = getattr(engine, nm, None)
        if S is not None:
            break
    if S is None:
        return None
    S64 = np.asarray(S, dtype=np.float64)
    n = len(g_pred)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        P = np.array(engine.g_gen.reconstruct(
            jnp.array(np.asarray(g_pred[s:e], dtype=np.float32)))
        ).astype(np.float64).reshape(e - s, -1)
        T = np.array(engine.g_gen.reconstruct(
            jnp.array(np.asarray(g_true[s:e], dtype=np.float32)))
        ).astype(np.float64).reshape(e - s, -1)
        d = P - T
        dSd = np.einsum("bi,ij,bj->b", d, S64, d)
        tSt = np.einsum("bi,ij,bj->b", T, S64, T)
        out[s:e] = np.sqrt(np.maximum(dSd, 0.0) / np.maximum(tSt, 1e-12))
    return out


def _sec_per_epoch(cfg):
    hist = (cfg or {}).get("hist")
    if isinstance(hist, dict):
        es = hist.get("epoch_seconds") or []
        if len(es) > 1:
            return float(np.median(es[1:]))
        if es:
            return float(es[0])
    return None


def _scalarize(v):
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return float(np.median(arr))


def _find_key(obj, names):
    """Recursive search for the first of `names` in a nested dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names:
                return v
        for v in obj.values():
            r = _find_key(v, names)
            if r is not None:
                return r
    return None


def _drive_panel_metrics(state, g_pred, g_true, rdms, energies, n_sub,
                         out_dir, tag):
    """f_err + eps_I via engine.drive_panel (p4 GEVP suite) on a subset.
    The p4 part is authored separately; bind by name and never let a panel
    failure kill the whole evaluation."""
    fn = getattr(engine, "drive_panel", None)
    if fn is None:
        return {"f_err": None, "eps_I": None,
                "error": "engine.drive_panel not available"}
    n_sub = min(n_sub, len(g_pred))
    pool = {
        "state": state, "state_model": state, "model_state": state,
        "model": state, "trained_state": state,
        "g_pred": g_pred[:n_sub], "g_true": g_true[:n_sub],
        "preds": g_pred[:n_sub], "predictions": g_pred[:n_sub],
        "labels": g_true[:n_sub],
        "rdms": rdms[:n_sub], "energies": energies[:n_sub],
        "n_samples": n_sub, "num_samples": n_sub, "n": n_sub,
        "nsamples": n_sub, "n_eval": n_sub,
        "include_energy": True,
        "h_type": "random", "state_type": "thermal",
        "panel": "random", "tag": tag, "name": tag, "label": tag,
        "beta": float(engine.BETA), "is_thermal": True,
        "seed": C.VAL_SEED, "out_dir": out_dir, "save_dir": out_dir,
        "lambda_ridge": C.LAMBDA_RIDGE, "ridge": C.LAMBDA_RIDGE,
        "cutoff": C.SNORM_CUTOFF, "snorm_cutoff": C.SNORM_CUTOFF,
        "zero_tol": C.LAMBDA_POS_THRESHOLD,
        "make_plots": False, "plot": False, "progress": False,
    }
    try:
        res = _bind_call(fn, pool, "engine.drive_panel")
    except Exception as exc:
        return {"f_err": None, "eps_I": None,
                "error": f"drive_panel failed: {exc!r}"}
    f_err = _scalarize(_find_key(res, {"f_err", "ferr", "f_error"}))
    eps_i = _scalarize(_find_key(res, {"eps_I", "eps_i", "epsI",
                                       "epsilon_I"}))
    out = {"f_err": f_err, "eps_I": eps_i, "drive_panel_n": int(n_sub)}
    if f_err is None and eps_i is None:
        out["error"] = (f"drive_panel returned no f_err/eps_I keys; "
                        f"keys={list(res)[:20] if isinstance(res, dict) else type(res).__name__}")
    return out


# ---------------------------------------------------- shared finite-shot rows
def _build_shot_inputs(budgets, n_states, n_noise, seed=0):
    """Precompute the noisy OGN inputs ONCE so every model sees byte-identical
    draws (gaussian stream id 2, SeedSequence([seed, 2, budget_idx, state,
    trial]) — the p5 protocol stream)."""
    true_systems, _h0 = engine.prepare_heldout_systems(
        n_states, seed=seed, with_wls_rows=False, progress=True)
    rho2_dense = engine._safe_dense(engine.rho_2_kkbar_arrays)
    per_budget = []
    for idx, N_shots in enumerate(budgets):
        rdm_b, e_b, sidx = [], [], []
        for state_idx, sysd in enumerate(true_systems):
            for trial in range(int(n_noise)):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [int(seed), 2, idx, state_idx, trial]))
                p_meas = engine._shots_draw_gaussian(
                    sysd["p_true"], float(N_shots), rng)
                rho_meas = (sysd["V_true"] * p_meas) @ sysd["V_true"].T
                rdm = engine.compute_rho_m(np.array([rho_meas]),
                                           rho2_dense, 1)[0]
                rdm_b.append(np.real(np.asarray(rdm)))
                e_b.append([float(np.sum(p_meas * sysd["E_true"]))])
                sidx.append(state_idx)
        bx = np.asarray(rdm_b, dtype=np.float32)
        if bx.ndim == 3:
            bx = bx[..., None]
        per_budget.append({"N_shots": float(N_shots), "bx": bx,
                           "be": np.asarray(e_b, dtype=np.float32),
                           "state_idx": np.asarray(sidx)})
    return true_systems, per_budget


def _model_shot_rows(state, true_systems, per_budget):
    import jax.numpy as jnp
    label_size = engine.g_gen.label_size()
    rows = []
    for binp in per_budget:
        bx = jnp.array(binp["bx"])
        be = jnp.array(binp["be"])
        by_dummy = jnp.zeros((bx.shape[0], label_size), jnp.float32)
        logits = engine.eval_step(state, bx, be, by_dummy)
        G_ml = np.array(engine.g_gen.reconstruct(logits)).astype(np.float64)
        errs = np.empty(len(G_ml), dtype=np.float64)
        for i, si in enumerate(binp["state_idx"]):
            sysd = true_systems[int(si)]
            errs[i] = engine._shots_shifted_error(
                G_ml[i], sysd["G_true"], sysd["norm_g_true"])
        rows.append({"N_shots": binp["N_shots"],
                     "median_err": float(np.median(errs)),
                     "mean_err": float(np.mean(errs)),
                     "q25_err": float(np.percentile(errs, 25)),
                     "q75_err": float(np.percentile(errs, 75))})
    return rows


# --------------------------------------------------------------- tex writers
def _fmt(x, spec="{:.2e}"):
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "--"
        return spec.format(x)
    except (TypeError, ValueError):
        return "--"


def _tex(s):
    """Escape dynamic text for LaTeX (verdict/basis strings only; static
    template strings are pre-escaped by hand)."""
    if s is None:
        return ""
    out = str(s).replace("\\", "")
    for ch, rep in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"),
                    ("_", "\\_"), ("$", "\\$")):
        out = out.replace(ch, rep)
    return out


def _write_tab_ablation(path, order, rows, budgets):
    n_b = len(budgets)
    bud_heads = " & ".join("$10^{%d}$" % int(round(math.log10(b)))
                           for b in budgets)
    lines = [
        "% Auto-generated by tasks/t_ablation_eval.py — referee-response",
        "% campaign 2026-06. One row per model; clean-data metrics on the",
        "% shared held-out set (seed {}, first N samples); finite-shot".format(C.VAL_SEED),
        "% medians on identical seeded draws (gaussian, seed 0).",
        r"\begin{tabular}{l r c c c c " + "c " * n_b + r"r}",
        r"\toprule",
        (r" & & \multicolumn{4}{c}{clean data} & "
         r"\multicolumn{" + str(n_b) + r"}{c}{median $\|\Delta G\|/\|G\|$ "
         r"at $N_{\rm shots}$} & \\"),
        (r"model & params & med.\ $\|\Delta G\|/\|G\|$ & rel.\ $S$-err & "
         r"$f_{\rm err}$ & $\varepsilon_I$ & " + bud_heads + r" & s/epoch \\"),
        r"\midrule",
    ]
    for name in order:
        r = rows.get(name)
        if r is None:
            continue
        esc_name = name.replace("_", "\\_")
        if r.get("error") and r.get("rel_param_err_median") is None:
            lines.append(esc_name + " & "
                         + " & ".join(["--"] * (5 + n_b + 1))
                         + r" \\  % " + str(r["error"]))
            continue
        shot_by_budget = {s["N_shots"]: s for s in r.get("shots", [])}
        shot_cells = []
        for b in budgets:
            s = shot_by_budget.get(float(b))
            shot_cells.append(_fmt(s["median_err"]) if s else "--")
        cells = [
            esc_name,
            _fmt(r.get("n_params"), "{:,d}"),
            _fmt(r.get("rel_param_err_median")),
            _fmt(r.get("rel_S_err_median")),
            _fmt(r.get("f_err")),
            _fmt(r.get("eps_I")),
        ] + shot_cells + [_fmt(r.get("sec_per_epoch"), "{:.0f}")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_claim_map(path, verdicts, gram_vs_rdm):
    lines = [
        "% Auto-generated by tasks/t_ablation_eval.py — referee-response",
        "% campaign 2026-06. Maps each manuscript architecture-attribution",
        "% sentence to the ablation rows that license or refute it",
        "% (numbers in ablation.json / tab_ablation.tex).",
        r"\begin{description}",
    ]
    for ref, quote, rows_desc, vkey in CLAIM_SENTENCES:
        v = verdicts.get(vkey, {})
        lines.append(r"\item[" + ref + "] " + quote + ".")
        lines.append(r"  \emph{Evidence rows:} " + rows_desc + ".")
        lines.append(r"  \emph{Verdict:} "
                     + _tex(v.get("verdict", "pending")) + " --- "
                     + _tex(v.get("basis", "metrics unavailable")))
    lines.append(
        r"\item[main.tex l.41] ``marginally more accurate'' (RDM objective "
        r"vs Gram objective).")
    lines.append(
        r"  \emph{Evidence rows:} paired-by-seed gram vs rdm "
        r"(seeds 42/43/44) at matched budget.")
    lines.append(r"  \emph{Verdict:} "
                 + _tex(gram_vs_rdm.get("verdict", "pending")))
    lines += [r"\end{description}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------- run
def run(out_dir):
    t0 = time.time()
    spec = C.MODELS[ENSEMBLE_MODEL]
    engine.init_d20(h_type=spec["h_type"], state_type=spec["state_type"],
                    beta=C.BETA_THERMAL)
    engine.init_gram()

    if C.SMOKE:
        n_eval, n_panel = 16, 8
        n_states, n_noise = 3, 3
    else:
        n_eval, n_panel = 4096, 512
        n_states, n_noise = C.ABL_SHOTS_N_STATES, C.ABL_SHOTS_N_NOISE
    budgets = [float(b) for b in C.ABL_SHOT_BUDGETS]

    val_dir = _ensure_val_dataset()
    rdms, energies, g_true = _collect_heldout(val_dir, n_eval)

    # identical noisy draws for every model (computed once)
    true_systems, per_budget = _build_shot_inputs(budgets, n_states,
                                                  n_noise, seed=0)

    model_names = [ENSEMBLE_MODEL] + [m for m in C.MODELS
                                      if m.startswith("abl_")]
    rows = {}
    pred_store = {"g_true": g_true}
    dim = engine.M_PAIRS
    input_shape = (1, dim, dim, 1)

    for name in model_names:
        mspec = C.MODELS[name]
        try:
            state, model, cfg = _load_checkpoint_state(name)
        except Exception as exc:
            rows[name] = {"error": f"checkpoint load failed: {exc!r}",
                          "rel_param_err_median": None}
            continue
        g_pred = _predict(state, rdms, energies)
        pred_store[f"g_pred_{name}"] = g_pred

        errs = _gauge_aligned_errors(g_pred, g_true)
        relS = _rel_S_errors(g_pred, g_true)
        try:
            n_params = int(cfg.get("n_params") or engine.count_params(
                model, input_shape, True, seed=mspec.get("init_seed", 42)))
        except Exception:
            n_params = None

        row = {
            "loss": mspec.get("loss"), "arch": mspec.get("arch"),
            "init_seed": mspec.get("init_seed"),
            "n_params": n_params,
            "rel_param_err_median": float(np.median(errs)),
            "rel_param_err_mean": float(np.mean(errs)),
            "rel_S_err_median": (float(np.median(relS))
                                 if relS is not None else None),
            "sec_per_epoch": _sec_per_epoch(cfg),
            "epochs_trained": (len(cfg["hist"]["loss"])
                               if isinstance(cfg.get("hist"), dict)
                               and cfg["hist"].get("loss") else None),
        }
        row.update(_drive_panel_metrics(state, g_pred, g_true, rdms,
                                        energies, n_panel, out_dir, name))
        row["shots"] = _model_shot_rows(state, true_systems, per_budget)
        rows[name] = row
        del state, model
        gc.collect()

    # ------------------------------------------------- kernel ridge baseline
    krr_npz = os.path.join(C.RESULTS_DIR, "kernel_ridge",
                           "krr_predictions.npz")
    krr_json = os.path.join(C.RESULTS_DIR, "kernel_ridge", "kernel_ridge.json")
    if os.path.exists(krr_npz):
        z = np.load(krr_npz)
        kp, kt = z["g_pred"].astype(np.float64), z["g_true"].astype(np.float64)
        match = (kt.shape == g_true.shape
                 and bool(np.allclose(kt, g_true, atol=1e-5)))
        errs = _gauge_aligned_errors(kp, kt)
        relS = _rel_S_errors(kp, kt)
        kcfg = {}
        if os.path.exists(krr_json):
            try:
                with open(krr_json) as f:
                    kcfg = json.load(f)
            except Exception:
                kcfg = {}
        rows["kernel_ridge"] = {
            "loss": "krr-rbf", "arch": "kernel_ridge", "n_params": None,
            "dual_size": int(kp.shape[0]) * int(kp.shape[1]),
            "rel_param_err_median": float(np.median(errs)),
            "rel_param_err_mean": float(np.mean(errs)),
            "rel_S_err_median": (float(np.median(relS))
                                 if relS is not None else None),
            "sec_per_epoch": (kcfg.get("refit_s")),
            "f_err": None, "eps_I": None,
            "shots": [],
            "heldout_matches_ablation_set": match,
            "best_hyperparams": kcfg.get("best"),
            "note": ("predictions-only baseline (no refittable model object); "
                     "finite-shot rows and GEVP panel not applicable"),
        }
    else:
        rows["kernel_ridge"] = {"error": f"missing {krr_npz}",
                                "rel_param_err_median": None}

    np.savez_compressed(os.path.join(out_dir, "ablation_predictions.npz"),
                        **pred_store)

    # --------------------------------- gram vs rdm paired-by-seed (cell_26)
    common = [s for s in sorted(GRAM_BY_SEED)
              if rows.get(GRAM_BY_SEED[s], {}).get("rel_param_err_median")
              is not None
              and rows.get(RDM_BY_SEED.get(s, ""), {}).get(
                  "rel_param_err_median") is not None]
    gram_vs_rdm = {"seeds": common}
    if len(common) >= 2:
        gm = np.array([rows[GRAM_BY_SEED[s]]["rel_param_err_median"]
                       for s in common])
        rm = np.array([rows[RDM_BY_SEED[s]]["rel_param_err_median"]
                       for s in common])
        deltas = rm - gm
        spread = float(gm.max() - gm.min())
        g_se = [rows[GRAM_BY_SEED[s]].get("sec_per_epoch") for s in common]
        r_se = [rows[RDM_BY_SEED[s]].get("sec_per_epoch") for s in common]
        K = (float(np.nanmedian([x for x in r_se if x is not None] or [np.nan]))
             / float(np.nanmedian([x for x in g_se if x is not None]
                                  or [np.nan])))
        indist = bool(np.all(np.abs(deltas) <= spread))
        rdm_better = bool(np.median(deltas) < 0)
        supports = (not indist) and rdm_better
        if indist:
            verdict = (f"statistically indistinguishable at matched budget "
                       f"(all |paired deltas| <= gram seed-spread "
                       f"{spread:.3e}); 'marginally more accurate' is NOT "
                       f"licensed — report 'indistinguishable accuracy at "
                       f"{K:.0f}x per-epoch cost'.")
        else:
            sgn = "rdm better" if rdm_better else "gram better"
            verdict = (f"{sgn} by {100 * abs(float(np.median(deltas))) / float(np.median(gm)):.1f}% "
                       f"(median paired delta {float(np.median(deltas)):.3e} "
                       f"vs gram seed-spread {spread:.3e}), at K={K:.1f}x "
                       f"per-epoch cost; "
                       + ("this LICENSES 'marginally more accurate'."
                          if rdm_better else
                          "this REFUTES 'marginally more accurate' "
                          "(direction is reversed)."))
        gram_vs_rdm.update({
            "gram_err_by_seed": {s: float(g) for s, g in zip(common, gm)},
            "rdm_err_by_seed": {s: float(r) for s, r in zip(common, rm)},
            "paired_deltas_rdm_minus_gram": [float(d) for d in deltas],
            "mean_paired_delta": float(np.mean(deltas)),
            "median_paired_delta": float(np.median(deltas)),
            "gram_seed_spread": spread,
            "cost_factor_K_sec_per_epoch": (None if math.isnan(K) else
                                            float(K)),
            "indistinguishable": indist,
            "supports_marginally_more_accurate": bool(supports),
            "verdict": verdict,
        })
    else:
        gram_vs_rdm["verdict"] = ("insufficient paired seeds with finished "
                                  "checkpoints; pending")

    # ----------------------------------------------------- claim verdicts
    def _pair(a, b, key):
        ra, rb = rows.get(a, {}), rows.get(b, {})
        va, vb = ra.get(key), rb.get(key)
        if va is None or vb is None:
            return None
        return float(va), float(vb)

    verdicts = {}
    p = _pair(ENSEMBLE_MODEL, "abl_mlp", "eps_I")
    basis_key = "eps_I"
    if p is None:
        p = _pair(ENSEMBLE_MODEL, "abl_mlp", "rel_param_err_median")
        basis_key = "rel_param_err_median"
    if p is not None:
        ok = p[0] < p[1]
        verdicts["arch_clean"] = {
            "verdict": "licensed" if ok else "refuted",
            "basis": (f"{basis_key}: prod={p[0]:.3e} vs abl_mlp={p[1]:.3e} "
                      f"(params-matched)"),
        }
    else:
        verdicts["arch_clean"] = {"verdict": "pending",
                                  "basis": "prod/abl_mlp metrics unavailable"}

    prod_shots = {s["N_shots"]: s["median_err"]
                  for s in rows.get(ENSEMBLE_MODEL, {}).get("shots", [])}
    mlp_shots = {s["N_shots"]: s["median_err"]
                 for s in rows.get("abl_mlp", {}).get("shots", [])}
    shared_b = sorted(set(prod_shots) & set(mlp_shots))
    if shared_b:
        wins = sum(1 for b in shared_b if prod_shots[b] < mlp_shots[b])
        ok = wins > len(shared_b) / 2
        verdicts["arch_shots"] = {
            "verdict": "licensed" if ok else "refuted",
            "basis": (f"prod beats abl_mlp at {wins}/{len(shared_b)} shot "
                      f"budgets on identical noisy draws"),
        }
    else:
        verdicts["arch_shots"] = {"verdict": "pending",
                                  "basis": "finite-shot rows unavailable"}

    if (verdicts["arch_clean"]["verdict"] != "pending"
            and verdicts["arch_shots"]["verdict"] != "pending"):
        both = (verdicts["arch_clean"]["verdict"] == "licensed"
                and verdicts["arch_shots"]["verdict"] == "licensed")
        either = (verdicts["arch_clean"]["verdict"] == "licensed"
                  or verdicts["arch_shots"]["verdict"] == "licensed")
        verdicts["arch_both"] = {
            "verdict": "licensed" if both else ("mixed" if either
                                                else "refuted"),
            "basis": (verdicts["arch_clean"]["basis"] + "; "
                      + verdicts["arch_shots"]["basis"]),
        }
    else:
        verdicts["arch_both"] = {"verdict": "pending",
                                 "basis": "see arch_clean / arch_shots"}

    prod_err = rows.get(ENSEMBLE_MODEL, {}).get("rel_param_err_median")
    toggle_hurt = []
    for t in ("abl_noscatter", "abl_noreinject", "abl_noembed",
              "abl_neutralbias", "abl_mlp"):
        te = rows.get(t, {}).get("rel_param_err_median")
        if prod_err is not None and te is not None and te > 1.1 * prod_err:
            toggle_hurt.append(t)
    if prod_err is None:
        verdicts["structure_matters"] = {"verdict": "pending",
                                         "basis": "prod metrics unavailable"}
    elif toggle_hurt:
        verdicts["structure_matters"] = {
            "verdict": "licensed",
            "basis": (f"removing structure measurably degrades accuracy "
                      f"(>10% relative) for: {', '.join(toggle_hurt)}"),
        }
    else:
        verdicts["structure_matters"] = {
            "verdict": "refuted",
            "basis": ("no structural toggle (nor the params-matched MLP) "
                      "degrades median parameter error by >10% — the "
                      "regularization is not attributable to the graph "
                      "structure on this evidence"),
        }

    # ------------------------------------------------------------ outputs
    result = {
        "held_out": {"seed": C.VAL_SEED, "n_samples": int(n_eval),
                     "dataset": val_dir},
        "shots_protocol": {"budgets": budgets, "n_states": int(n_states),
                           "n_noise": int(n_noise), "seed": 0,
                           "noise_model": "gaussian",
                           "identical_draws": True},
        "drive_panel_n_samples": int(n_panel),
        "models": rows,
        "gram_vs_rdm": gram_vs_rdm,
        "claim_verdicts": verdicts,
        "smoke": bool(C.SMOKE),
        "task_wall_s": time.time() - t0,
        "provenance": "referee-response campaign 2026-06; satelite2/campaign",
    }
    with open(os.path.join(out_dir, "ablation.json"), "w") as f:
        json.dump(_jsonable(result), f, indent=1)

    order = model_names + ["kernel_ridge"]
    _write_tab_ablation(os.path.join(out_dir, "tab_ablation.tex"),
                        order, rows, budgets)
    _write_claim_map(os.path.join(out_dir, "claim_map.tex"),
                     verdicts, gram_vs_rdm)

    payload = {
        "n_models_evaluated": sum(1 for r in rows.values()
                                  if r.get("rel_param_err_median") is not None),
        "n_models_failed": sum(1 for r in rows.values() if r.get("error")
                               and r.get("rel_param_err_median") is None),
        "prod_rel_param_err_median": prod_err,
        "gram_vs_rdm_verdict": gram_vs_rdm.get("verdict"),
        "claim_verdicts": {k: v.get("verdict") for k, v in verdicts.items()},
        "task_wall_s": time.time() - t0,
    }
    return _jsonable(payload)


# ============================================================ gbase probe
def _manual_probe_batch(probe_path, val_dir, n=64):
    """Fallback with engine.get_probe_batch semantics: one fixed batch,
    saved once at probe_path, reused by every run."""
    if os.path.exists(probe_path):
        z = np.load(probe_path)
        return z["bx"], (z["be"] if "be" in z.files else None), z["by"]
    load_bs = max(64, n)
    loader = engine.NumpyLoader(val_dir, load_bs, shuffle=False)
    for bx, be, by in loader:
        x = np.asarray(bx, dtype=np.float32)
        if x.ndim == 3:
            x = x[..., None]
        bx_n, by_n = x[:n], np.asarray(by, dtype=np.float32)[:n]
        be_n = np.asarray(be, dtype=np.float32).reshape(len(x), -1)[:n, :1]
        break
    else:
        raise RuntimeError(f"no data in {val_dir}")
    np.savez_compressed(probe_path, bx=bx_n, be=be_n, by=by_n)
    return bx_n, be_n, by_n


def run_gbase_probe(out_dir):
    """cell_26-style gamma_base gradient-norm probe (manuscript l.542):
    4 short rdm-loss runs, gamma_base in {0, 1e-6, 1e-4, 1e-2}, 2 epochs on
    200k samples, per-component gradient norms on ONE fixed probe batch."""
    t0 = time.time()
    spec = C.MODELS[ENSEMBLE_MODEL]
    engine.init_d20(h_type=spec["h_type"], state_type=spec["state_type"],
                    beta=C.BETA_THERMAL)
    engine.init_gram()

    if C.SMOKE:
        n_samples, epochs, res, batch_size = 512, 2, 1, 64
    else:
        n_samples, epochs, res, batch_size = 200_000, 2, 3, 256

    # probe-specific dataset cache (diagnostic-only; seed = train half0 stream
    # so the probe sees production-distribution data)
    gen_bs = 64 if C.SMOKE else 2048
    probe_ds = os.path.join(C.DATA_DIR, f"{DS_ROOT_KEY}_probe{n_samples}")
    dataset = engine.gen_dataset(spec["h_type"], C.G_INIT, C.G_STOP,
                                 spec["state_type"], "rho2kkbar", True,
                                 C.BETA_THERMAL, num_samples=n_samples,
                                 cache_path=probe_ds, batch_size=gen_bs,
                                 seed=C.TRAIN_SEEDS[0])
    val_dir = _ensure_val_dataset()

    probe_path = os.path.join(out_dir, "probe_batch.npz")
    try:
        pb = _bind_call(getattr(engine, "get_probe_batch"), {
            "probe_path": probe_path, "path": probe_path, "n": 64,
            "cache_path": val_dir, "val_cache": val_dir, "val_dir": val_dir,
            "seed": C.VAL_SEED, "h_type": spec["h_type"],
            "state_type": spec["state_type"], "input_type": "rho2kkbar",
            "include_energy": True, "beta": float(engine.BETA),
            "num_samples": 4096 if not C.SMOKE else 64,
            "batch_size": gen_bs, "g_init": C.G_INIT, "g_stop": C.G_STOP,
        }, "engine.get_probe_batch")
    except Exception as exc:
        print(f"[gbase_probe] engine.get_probe_batch unusable ({exc!r}); "
              f"using the manual fixed-batch fallback (same semantics)")
        pb = _manual_probe_batch(probe_path, val_dir, n=64)

    label_size = engine.g_gen.label_size()
    runs = []
    for gb in GBASE_SWEEP:
        rid = f"rdm_gb{gb:g}_s42"
        print(f"\n[gbase_probe] === run {rid}: gamma_base={gb:g} ===")
        probe_cb = _bind_call(getattr(engine, "make_component_probe"), {
            "loss_type": "rdm", "loss": "rdm", "is_thermal": True,
            "beta": float(engine.BETA), "probe_batch": pb, "probe": pb,
            "batch": pb,
        }, "engine.make_component_probe")
        ckpt_dir = os.path.join(out_dir, f"ckpt_{rid}")
        os.makedirs(ckpt_dir, exist_ok=True)
        t_run = time.time()
        out = _bind_call(getattr(engine, "train_model"), {
            "train_loader": dataset, "dataset": dataset, "loader": dataset,
            "label_size": label_size, "input_type": "rho2kkbar",
            "M_PAIRS": engine.M_PAIRS, "m_pairs": engine.M_PAIRS,
            "include_energy": True,
            "total_samples": n_samples, "num_samples": n_samples,
            "batch_size": batch_size, "epochs": epochs,
            "num_epochs": epochs, "res": res,
            "loss_type": "rdm", "loss": "rdm",
            "is_thermal": True, "beta": float(engine.BETA),
            "init_seed": 42, "seed": 42,
            "gammas": {"gamma_rdm": GAMMA_RDM, "gamma_trace": GAMMA_TRACE,
                       "gamma_base": gb},
            "probe_cb": probe_cb,
            "ckpt_dir": ckpt_dir, "sync_cb": None,
            "arch": "ogn", "model_name": rid, "name": rid,
            "peak_lr": spec.get("peak_lr", 3e-4),
            "weight_decay": spec.get("weight_decay", 1e-4),
            "clip": spec.get("clip", 1.0),
            "h_type": spec["h_type"], "state_type": spec["state_type"],
            "use_scatter": True, "use_reinject": True, "use_orb_emb": True,
            "readout_bias": 0.55,
        }, "engine.train_model")
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            state, hist = out[0], out[1]
        else:
            state, hist = out, {}
        if not isinstance(hist, dict):
            hist = {"loss": list(hist)}
        runs.append({"run_id": rid, "gamma_base": float(gb),
                     "gamma_rdm": GAMMA_RDM, "gamma_trace": GAMMA_TRACE,
                     "loss_history": hist.get("loss"),
                     "epoch_seconds": hist.get("epoch_seconds"),
                     "grad_norms": hist.get("grad_norms", []),
                     "wall_s": time.time() - t_run})
        del state, out
        gc.collect()

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        gl = r["grad_norms"] or []
        if not gl:
            continue
        eps = [g.get("epoch", i) + 1 for i, g in enumerate(gl)]
        for key in sorted(k for k in gl[0] if k.startswith("gnorm_")):
            ax.semilogy(eps, [g.get(key, np.nan) for g in gl], marker="o",
                        label=f"gb={r['gamma_base']:g} {key[6:]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|\nabla \mathcal{L}_i\|$ (unweighted component)")
    ax.set_title(r"rdm-loss component gradient norms vs $\gamma_{base}$")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig_png = os.path.join(out_dir, "gbase_probe_gradnorms.png")
    fig.savefig(fig_png, dpi=300)
    fig.savefig(os.path.join(out_dir, "gbase_probe_gradnorms.pdf"))
    plt.close(fig)

    # ------------------------------------------- hierarchy check (l.542)
    ratios = {}
    for r in runs:
        gl = r["grad_norms"] or []
        gb = r["gamma_base"]
        if not gl:
            ratios[f"{gb:g}"] = None
            continue
        last = gl[-1]
        g_rdm = last.get("gnorm_rdm_mse")
        g_base = last.get("gnorm_base")
        if g_rdm is None or g_base is None or gb == 0.0:
            ratios[f"{gb:g}"] = None       # gb=0: base term absent by weight
            continue
        ratios[f"{gb:g}"] = float((GAMMA_RDM * g_rdm)
                                  / max(gb * g_base, 1e-300))
    finite = [v for v in ratios.values() if v is not None]
    confirmed = bool(finite) and all(v > 10.0 for v in finite)
    hierarchy = {
        "criterion": ("gamma_rdm*||grad L_rdm|| > 10 x "
                      "gamma_base*||grad L_base|| at the final probed epoch, "
                      "for every gamma_base > 0 in the sweep"),
        "weighted_ratio_by_gamma_base": ratios,
        "min_weighted_ratio": (min(finite) if finite else None),
        "confirmed": confirmed,
        "manuscript_ref": "main.tex l.542 ('a hierarchy confirmed by ...')",
    }

    result = {
        "sweep_gamma_base": list(GBASE_SWEEP),
        "loss_type": "rdm",
        "n_samples": int(n_samples), "epochs": int(epochs),
        "res": int(res), "batch_size": int(batch_size),
        "init_seed": 42,
        "probe_batch": {"path": probe_path, "n": 64,
                        "source": f"val stream seed {C.VAL_SEED}"},
        "runs": runs,
        "hierarchy": hierarchy,
        "figure": fig_png,
        "smoke": bool(C.SMOKE),
        "task_wall_s": time.time() - t0,
        "provenance": "referee-response campaign 2026-06; satelite2/campaign",
    }
    with open(os.path.join(out_dir, "gbase_probe.json"), "w") as f:
        json.dump(_jsonable(result), f, indent=1)

    payload = {
        "hierarchy_confirmed": confirmed,
        "min_weighted_ratio": hierarchy["min_weighted_ratio"],
        "n_runs": len(runs),
        "task_wall_s": time.time() - t0,
    }
    return _jsonable(payload)
