"""GEVP / covariance diagnostics panels (d=20 ground-state ensembles).

run_random: trained panel (prod_gs_random) + faithful near-init "untrained"
panel (nearinit_random) on the d20/random/gs ensemble.
run_const:  trained const panel (prod_gs_const) on d20/const/gs.

Each panel = engine.drive_panel (ridge 1e-9 / cutoff 1e-7 / positive-lambda
threshold 1e-12) + engine.sweep_stability over C.RIDGE_SWEEP x C.CUTOFF_SWEEP
+ engine.plot_scree_qk figure.  The trained panels additionally report the
per-sample ||dw||_S distribution (referee Major item 9: reconciling f_err
with the ~1e-3 parameter-error plateau).  eps_I / f_par are surfaced at the
top level of gevp_random.json / gevp_const.json; panels are keyed like the
values_prev.json `covariance.*` blocks.
"""
import inspect
import json
import os
import subprocess
import sys
import time

# defensive: runner already puts the campaign dir (this file's parent's parent)
# on sys.path, but tasks must also import standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import engine

PROVENANCE = "referee-response campaign 2026-06; satelite2/campaign"


# --------------------------------------------------------------------------
# generic glue (p3/p4 signatures are owned by other engine parts; calls below
# pass exactly the contract-documented arguments and silently drop keyword
# arguments a given implementation does not take)
# --------------------------------------------------------------------------
def _flexcall(fn, *args, **kwargs):
    try:
        sig = inspect.signature(fn)
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kwargs)


def _jsonable(obj, max_list=4096):
    """Recursively convert engine return values into JSON-serializable data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v, max_list) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__array__"):
        a = np.asarray(obj)
        if a.size <= max_list:
            return _jsonable(a.tolist(), max_list)
        try:
            return {"__array_summary__": True, "shape": list(a.shape),
                    "dtype": str(a.dtype), "min": float(np.nanmin(a)),
                    "max": float(np.nanmax(a)), "mean": float(np.nanmean(a))}
        except Exception:
            return {"__array_summary__": True, "shape": list(a.shape),
                    "dtype": str(a.dtype)}
    return repr(obj)


def _pick(d, *names):
    """Case-insensitive key lookup in a (possibly differently-cased) dict."""
    if isinstance(d, dict):
        lowered = {str(k).lower(): v for k, v in d.items()}
        for n in names:
            if n.lower() in lowered:
                return lowered[n.lower()]
    return None


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def _ckpt_present(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return False
    entries = [e for e in os.listdir(ckpt_dir) if e != "config.json"]
    return len(entries) > 0


def _ensure_ckpt(model_name):
    """Local checkpoint dir for <model_name>; pull from the contract GCS path
    if missing locally, else raise."""
    ckpt_dir = os.path.join(C.CKPT_DIR, model_name)
    if not _ckpt_present(ckpt_dir) and not C.SMOKE:  # no GCS traffic in smoke runs
        os.makedirs(ckpt_dir, exist_ok=True)
        src = f"{C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name}"
        try:  # best-effort, per contract
            C.gcs_sync_dir(src, ckpt_dir)
        except Exception as exc:
            print(f"[t_gevp_diag] GCS pull failed for {model_name}: {exc}",
                  flush=True)
    if not _ckpt_present(ckpt_dir):
        raise FileNotFoundError(
            f"checkpoint for '{model_name}' not found locally ({ckpt_dir}) nor "
            f"at {C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name}; "
            f"run train_{model_name} first")
    return ckpt_dir


def _as_state(obj):
    # engine.load_model may return state or (state, hist)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return obj[0]
    return obj


def _load_state(model_name):
    """engine.load_model with the spec from C.MODELS, adapted to whatever
    signature p3_training exposes."""
    ckpt_dir = _ensure_ckpt(model_name)
    spec = dict(C.MODELS[model_name])
    fn = engine.load_model
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = None

    if params is not None:
        kwargs = {}
        if "model_name" in params:
            kwargs["model_name"] = model_name
        elif "name" in params:
            kwargs["name"] = model_name
        for k in ("ckpt_dir", "load_dir", "save_dir", "model_dir", "path"):
            if k in params:
                kwargs[k] = ckpt_dir
                break
        for k in ("spec", "model_spec", "cfg", "config"):
            if k in params:
                kwargs[k] = spec
                break
        m = int(engine.M_PAIRS)
        extras = {
            "label_size": int(engine.g_gen.label_size()),
            "res": spec.get("res"),
            "include_energy": spec.get("include_energy", True),
            "input_shape": (1, m, m, 1),
            "arch": spec.get("arch"),
            "use_scatter": spec.get("use_scatter", True),
            "use_reinject": spec.get("use_reinject", True),
            "use_orb_emb": spec.get("use_orb_emb", True),
            "readout_bias": spec.get("readout_bias", 0.55),
            "init_seed": spec.get("init_seed", C.INIT_SEED),
        }
        for k, v in extras.items():
            if k in params and k not in kwargs:
                kwargs[k] = v
        required = [p for p in params.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY)]
        if kwargs and all(p.name in kwargs for p in required):
            return _as_state(fn(**kwargs))

    for args in ((model_name, ckpt_dir), (model_name,), (ckpt_dir,)):
        try:
            return _as_state(fn(*args))
        except TypeError:
            continue
    raise RuntimeError(
        f"could not adapt engine.load_model signature for '{model_name}'")


# --------------------------------------------------------------------------
# validation predictions
# --------------------------------------------------------------------------
def _ensemble_val_count(era, h_type, state_type):
    """Shared val split = 5% of the LARGEST num_samples among models on the
    same ensemble (cache key ds_{era}_{htype}_{state} is shared)."""
    n_max = max(spec["num_samples"] for spec in C.MODELS.values()
                if spec["era"] == era and spec["h_type"] == h_type
                and spec["state_type"] == state_type)
    return max(int(0.05 * n_max), 1)


def _val_loader(spec):
    era, h_type, state_type = spec["era"], spec["h_type"], spec["state_type"]
    val_dir = os.path.join(C.DATA_DIR, f"ds_{era}_{h_type}_{state_type}", "val")
    n_val = _ensemble_val_count(era, h_type, state_type)
    gen_bs = 64 if C.SMOKE else 4096
    beta = C.BETA_GS if state_type == "gs" else C.BETA_THERMAL
    include_energy = bool(spec.get("include_energy", True))
    engine.gen_dataset(h_type, C.G_INIT, C.G_STOP, state_type, "rho2kkbar",
                       include_energy, beta, num_samples=n_val,
                       cache_path=val_dir, batch_size=gen_bs, seed=C.VAL_SEED)
    # deterministic iteration order for the panels (the gen_dataset return
    # value uses the NumpyLoader default shuffle=True)
    return engine.NumpyLoader(val_dir, gen_bs, shuffle=False)


def _npz_key(z, *names):
    for n in names:
        if n in z.files:
            return np.asarray(z[n])
    return None


def _predict_val(model_name, spec, state, n_needed, out_dir):
    """(g_true, g_pred) on the shared val split: prefer the checkpoint's
    predictions_val.npz, else regenerate the val loader and run eval_step."""
    ckpt_dir = os.path.join(C.CKPT_DIR, model_name)
    npz_path = os.path.join(ckpt_dir, "predictions_val.npz")
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path) as z:
                g_pred = _npz_key(z, "g_pred", "pred", "preds", "predictions",
                                  "logits")
                g_true = _npz_key(z, "g_true", "true", "labels", "actual", "y")
            if g_pred is not None and g_true is not None \
                    and len(g_pred) >= n_needed and len(g_true) >= n_needed:
                print(f"[t_gevp_diag] using {npz_path}", flush=True)
                return g_true[:n_needed], g_pred[:n_needed]
            print(f"[t_gevp_diag] {npz_path} short/unusable; regenerating",
                  flush=True)
        except Exception as exc:
            print(f"[t_gevp_diag] failed reading {npz_path}: {exc}", flush=True)

    import jax.numpy as jnp
    include_energy = bool(spec.get("include_energy", True))
    loader = _val_loader(spec)
    bs_eval = int(engine.GPU_BATCH_SIZE)
    g_true_l, g_pred_l, n = [], [], 0
    for big_x, big_e, big_y in loader:
        chunk = big_x.shape[0]
        for s in range(0, chunk, bs_eval):
            e = min(s + bs_eval, chunk)
            bx = jnp.array(big_x[s:e])
            by = jnp.array(big_y[s:e])
            be = jnp.array(big_e[s:e]) if (include_energy and big_e is not None) \
                else None
            logits = engine.eval_step(state, bx, be, by)
            logits = np.asarray(logits)
            if logits.ndim == 3:  # pmap-style leading device axis
                logits = logits.reshape(-1, logits.shape[-1])
            g_pred_l.append(np.nan_to_num(logits))
            g_true_l.append(np.asarray(big_y[s:e]))
            n += e - s
            if n >= n_needed:
                break
        if n >= n_needed:
            break
    if not g_pred_l:
        raise RuntimeError(f"validation loader yielded no data for {model_name}")
    g_true = np.concatenate(g_true_l)[:n_needed]
    g_pred = np.concatenate(g_pred_l)[:n_needed]
    np.savez_compressed(
        os.path.join(out_dir, f"predictions_val_{model_name}.npz"),
        g_pred=g_pred, g_true=g_true)
    return g_true, g_pred


# --------------------------------------------------------------------------
# ||dw||_S distribution (referee Major item 9)
# --------------------------------------------------------------------------
def _gram_S():
    """Locate the gram metric S (M^2 x M^2) set up by engine.init_gram()."""
    m = int(engine.M_PAIRS)
    want = m * m
    preferred = ["S_tensor_global", "S_matrix_np_global", "S_matrix_global",
                 "S_gram_global", "S_global", "S_tensor", "S_matrix"]
    keys = [k for k in preferred if k in engine.ns]
    keys += [k for k in engine.ns
             if isinstance(k, str) and k not in keys
             and ("s_tensor" in k.lower() or "s_matrix" in k.lower())]
    for k in keys:
        try:
            a = np.asarray(engine.ns[k], dtype=np.float64)
        except Exception:
            continue
        if a.ndim == 2 and a.shape == (want, want):
            return a
    raise RuntimeError("gram metric S not found in engine namespace "
                       "(did engine.init_gram() run?)")


def _dw_S_stats(g_pred, g_true, out_prefix):
    """Per-sample ||dw||_S / ||w_true||_S distribution + histogram figure."""
    import jax.numpy as jnp
    S = _gram_S()
    recon = engine.g_gen.reconstruct

    def _flat(labels):
        out = []
        for s in range(0, len(labels), 512):
            mat = np.asarray(recon(jnp.array(labels[s:s + 512],
                                             dtype=jnp.float32)))
            out.append(mat.reshape(mat.shape[0], -1).astype(np.float64))
        return np.concatenate(out)

    wp, wt = _flat(g_pred), _flat(g_true)
    dw = wp - wt
    s_dw = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", dw, S, dw), 0.0))
    s_wt = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", wt, S, wt), 0.0))
    rel = s_dw / np.maximum(s_wt, 1e-300)

    stats = {
        "n": int(rel.size),
        "rel_median": float(np.median(rel)),
        "rel_mean": float(np.mean(rel)),
        "rel_p10": float(np.percentile(rel, 10)),
        "rel_p90": float(np.percentile(rel, 90)),
        "rel_min": float(np.min(rel)),
        "rel_max": float(np.max(rel)),
        "abs_median": float(np.median(s_dw)),
        "abs_mean": float(np.mean(s_dw)),
        "norm_wtrue_S_median": float(np.median(s_wt)),
        "definition": ("per-sample sqrt(dw^T S dw)/sqrt(w_true^T S w_true), "
                       "dw = flattened reconstructed G_pred - G_true, S = the "
                       "gram metric of init_gram()"),
        "note": ("referee Major item 9: distribution reconciling f_err with "
                 "the ~1e-3 parameter-error plateau"),
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pos = rel[rel > 0]
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        if pos.size:
            ax.hist(np.log10(pos), bins=40, color="navy", alpha=0.85)
        ax.axvline(np.log10(max(stats["rel_median"], 1e-300)), color="crimson",
                   ls="--", lw=2,
                   label=f"median = {stats['rel_median']:.3e}")
        ax.set_xlabel(r"$\log_{10}\,\|\delta w\|_S/\|w_{\rm true}\|_S$")
        ax.set_ylabel("samples")
        ax.set_title(r"Trained-panel $\|\delta w\|_S$ distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_prefix + ".png", dpi=300)
        fig.savefig(out_prefix + ".pdf")
        plt.close(fig)
        stats["figure"] = out_prefix + ".png"
    except Exception as exc:
        stats["figure"] = None
        stats["figure_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[t_gevp_diag] dw_S histogram failed: {exc}", flush=True)
    return stats


# --------------------------------------------------------------------------
# panel driver
# --------------------------------------------------------------------------
def _run_panel(tag, g_pred, g_true, n_samples, out_dir):
    """drive_panel + sweep_stability + scree figure for one panel."""
    ns_eff = int(min(n_samples, len(g_pred)))
    panel = _flexcall(engine.drive_panel, g_pred, g_true,
                      n_samples=ns_eff, lam_ridge=C.LAMBDA_RIDGE,
                      cutoff=C.SNORM_CUTOFF,
                      lam_pos_threshold=C.LAMBDA_POS_THRESHOLD)
    stab = _flexcall(engine.sweep_stability, g_pred, g_true,
                     C.RIDGE_SWEEP, C.CUTOFF_SWEEP,
                     n_samples=ns_eff,
                     lam_pos_threshold=C.LAMBDA_POS_THRESHOLD)

    block = _jsonable(panel) if isinstance(panel, dict) \
        else {"panel": _jsonable(panel)}
    block["stability"] = _jsonable(stab)
    block.setdefault("N_s", ns_eff)
    block.setdefault("ridge", C.LAMBDA_RIDGE)
    block.setdefault("cutoff", C.SNORM_CUTOFF)
    block.setdefault("lambda_pos_threshold", C.LAMBDA_POS_THRESHOLD)
    block.setdefault("ridge_sweep", list(C.RIDGE_SWEEP))
    block.setdefault("cutoff_sweep", list(C.CUTOFF_SWEEP))

    prefix = os.path.join(out_dir, f"scree_{tag}")
    try:
        _flexcall(engine.plot_scree_qk, panel, prefix)
        block["figure"] = prefix + ".png"
    except Exception as exc:
        block["figure"] = None
        block["figure_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[t_gevp_diag] plot_scree_qk failed for {tag}: {exc}",
              flush=True)
    return block


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------
def run_random(out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine.init_d20(h_type="random", state_type="gs", beta=C.BETA_GS)
    engine.init_gram()

    n_tr = 16 if C.SMOKE else int(C.GEVP_NSAMPLES)
    n_un = 8 if C.SMOKE else int(C.GEVP_NSAMPLES_UNTRAINED)

    cov = {}

    # trained panel: prod_gs_random
    spec_tr = C.MODELS["prod_gs_random"]
    state_tr = _load_state("prod_gs_random")
    gt_tr, gp_tr = _predict_val("prod_gs_random", spec_tr, state_tr, n_tr,
                                out_dir)
    cov["trained_random"] = _run_panel("random_trained", gp_tr, gt_tr, n_tr,
                                       out_dir)
    dw = _dw_S_stats(gp_tr, gt_tr,
                     os.path.join(out_dir, "dwS_hist_random_trained"))

    # untrained panel: faithful near-init reconstruction (1 epoch / 10k)
    spec_un = C.MODELS["nearinit_random"]
    state_un = _load_state("nearinit_random")
    gt_un, gp_un = _predict_val("nearinit_random", spec_un, state_un, n_un,
                                out_dir)
    cov["untrained_random"] = _run_panel("random_untrained", gp_un, gt_un,
                                         n_un, out_dir)

    result = {
        "task": "gevp_random",
        "provenance": PROVENANCE,
        "models": {"trained": "prod_gs_random", "untrained": "nearinit_random"},
        # referee headline numbers, surfaced at top level
        "eps_I": _jsonable(_pick(cov["trained_random"], "eps_I", "epsI")),
        "f_par": _jsonable(_pick(cov["trained_random"], "f_par", "fpar")),
        "f_err_trained": _jsonable(_pick(cov["trained_random"], "f_err",
                                         "ferr")),
        "eps_I_untrained": _jsonable(_pick(cov["untrained_random"], "eps_I",
                                           "epsI")),
        "f_err_untrained": _jsonable(_pick(cov["untrained_random"], "f_err",
                                           "ferr")),
        "dw_S_trained": dw,
        "covariance": cov,
        "smoke": bool(C.SMOKE),
        "wall_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "gevp_random.json"), "w") as f:
        json.dump(result, f, indent=1)

    return {
        "f_err_trained": result["f_err_trained"],
        "f_err_untrained": result["f_err_untrained"],
        "eps_I": result["eps_I"],
        "f_par": result["f_par"],
        "dw_S_rel_median": dw.get("rel_median"),
        "n_trained": int(min(n_tr, len(gp_tr))),
        "n_untrained": int(min(n_un, len(gp_un))),
    }


def run_rdm_gs(out_dir):
    """GEVP covariance diagnostics on the rdm-trained GS model
    (rdm_gs_random_50) -- the 'covariance analysis under the rdm loss'. Compared
    offline against the existing gram GS panel (gevp_random.json / prod_gs_random)
    to test whether the representability-respecting rdm objective yields a more
    null-confined (lower-R) or differently-leaking prediction."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine.init_d20(h_type="random", state_type="gs", beta=C.BETA_GS)
    engine.init_gram()

    n_tr = 16 if C.SMOKE else int(C.GEVP_NSAMPLES)
    model = "rdm_gs_random_50"
    spec = C.MODELS[model]
    state = _load_state(model)
    g_true, g_pred = _predict_val(model, spec, state, n_tr, out_dir)
    block = _run_panel("rdm_gs_trained", g_pred, g_true, n_tr, out_dir)
    dw = _dw_S_stats(g_pred, g_true,
                     os.path.join(out_dir, "dwS_hist_rdm_gs_trained"))

    result = {
        "task": "gevp_rdm_gs",
        "provenance": PROVENANCE,
        "models": {"trained": model},
        "loss": "rdm",
        "eps_I": _jsonable(_pick(block, "eps_I", "epsI")),
        "f_par": _jsonable(_pick(block, "f_par", "fpar")),
        "f_err_trained": _jsonable(_pick(block, "f_err", "ferr")),
        "R": _jsonable(_pick(block, "R", "orientation_ratio", "orientation_R",
                             "R_orientation")),
        "dw_S_trained": dw,
        "covariance": {"rdm_gs_trained": block},
        "smoke": bool(C.SMOKE),
        "wall_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "gevp_rdm_gs.json"), "w") as f:
        json.dump(result, f, indent=1)

    return {
        "f_err_trained": result["f_err_trained"],
        "eps_I": result["eps_I"],
        "f_par": result["f_par"],
        "R": result["R"],
        "dw_S_rel_median": dw.get("rel_median"),
        "n_trained": int(min(n_tr, len(g_pred))),
    }


def run_const(out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    engine.init_d20(h_type="const", state_type="gs", beta=C.BETA_GS)
    engine.init_gram()

    n_tr = 16 if C.SMOKE else int(C.GEVP_NSAMPLES)

    spec = C.MODELS["prod_gs_const"]
    state = _load_state("prod_gs_const")
    g_true, g_pred = _predict_val("prod_gs_const", spec, state, n_tr, out_dir)
    block = _run_panel("const_trained", g_pred, g_true, n_tr, out_dir)
    dw = _dw_S_stats(g_pred, g_true,
                     os.path.join(out_dir, "dwS_hist_const_trained"))

    # eps_I is undefined on the const panel: w_true is const => P_perp w_true=0
    eps_note = "N/A (undefined: P_perp w_true = 0)"

    result = {
        "task": "gevp_const",
        "provenance": PROVENANCE,
        "models": {"trained": "prod_gs_const"},
        "eps_I": eps_note,
        "eps_I_panel_raw": _jsonable(_pick(block, "eps_I", "epsI")),
        "f_par": _jsonable(_pick(block, "f_par", "fpar")),
        "f_err_trained": _jsonable(_pick(block, "f_err", "ferr")),
        "dw_S_trained": dw,
        "covariance": {"trained_const": block},
        "smoke": bool(C.SMOKE),
        "wall_s": time.time() - t0,
    }
    # keep the N/A note visible inside the panel block too
    result["covariance"]["trained_const"]["eps_I_note"] = eps_note

    with open(os.path.join(out_dir, "gevp_const.json"), "w") as f:
        json.dump(result, f, indent=1)

    return {
        "f_err_trained": result["f_err_trained"],
        "eps_I": eps_note,
        "f_par": result["f_par"],
        "dw_S_rel_median": dw.get("rel_median"),
        "n_trained": int(min(n_tr, len(g_pred))),
    }
