# =============================================================================
# tasks/t_kernel_ridge.py — RBF kernel-ridge baseline on the thermal ensemble
#
# Declared protocol (full-dataset KRR is O(N^3)-infeasible; see config):
#   - features: upper triangle (incl. diagonal) of the SYMMETRIZED rho2 pair
#     block + the total energy; standardized with train-split moments.
#   - kernel: RBF  K(x, x') = exp(-gamma * ||x - x'||^2); ridge alpha on the
#     kernel diagonal; cholesky solve. numpy/scipy ONLY (no sklearn).
#   - grid: C.KRR_ALPHAS x C.KRR_GAMMAS ('scale' = 1 / (n_feat * var(X_train)),
#     the sklearn convention), selected on a selection-val split DISJOINT from
#     both the train split and the held-out set.
#   - final eval: the SAME held-out set as ablation_eval (val dataset, seed
#     C.VAL_SEED, shuffle=False, first 4096 samples), gauge-aligned relative
#     parameter error.
# Outputs: krr_predictions.npz (g_pred, g_true) for ablation_eval +
# kernel_ridge.json.
# =============================================================================
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg

import config as C
import engine

ENSEMBLE_MODEL = "prod_thermal_random"          # defines the shared d20 ensemble
DS_ROOT_KEY = "ds_d20_random_thermal"


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


def _ensure_datasets():
    """Shared-cache datasets per the campaign contract: two train halves
    (seeds C.TRAIN_SEEDS, num_samples/2 each) + val (seed C.VAL_SEED, 5%).
    engine.gen_dataset skips regeneration when the cache is complete."""
    spec = C.MODELS[ENSEMBLE_MODEL]
    root = os.path.join(C.DATA_DIR, DS_ROOT_KEY)
    half_n = spec["num_samples"] // 2
    gen_bs = 64 if C.SMOKE else 4096
    half_dirs = []
    for i, seed in enumerate(C.TRAIN_SEEDS):
        d = os.path.join(root, f"half{i}")
        engine.gen_dataset(spec["h_type"], C.G_INIT, C.G_STOP,
                           spec["state_type"], "rho2kkbar", True,
                           C.BETA_THERMAL, num_samples=half_n,
                           cache_path=d, batch_size=gen_bs, seed=seed)
        half_dirs.append(d)
    val_dir = os.path.join(root, "val")
    n_val = max(gen_bs, int(0.05 * spec["num_samples"]))
    engine.gen_dataset(spec["h_type"], C.G_INIT, C.G_STOP,
                       spec["state_type"], "rho2kkbar", True,
                       C.BETA_THERMAL, num_samples=n_val,
                       cache_path=val_dir, batch_size=gen_bs, seed=C.VAL_SEED)
    return half_dirs, val_dir


def _collect_features(loader, n_max):
    """Deterministic feature/label extraction from a shuffle=False loader:
    X = [triu(symmetrized rho2 block), energy], Y = labels. First n_max rows."""
    Xs, Ys = [], []
    got = 0
    iu = None
    for bx, be, by in loader:
        R = np.asarray(bx, dtype=np.float64)
        if R.ndim == 4 and R.shape[-1] == 1:
            R = R[..., 0]
        Rs = 0.5 * (R + np.transpose(R, (0, 2, 1)))
        if iu is None:
            iu = np.triu_indices(Rs.shape[1], k=0)
        feats = Rs[:, iu[0], iu[1]]
        en = np.asarray(be, dtype=np.float64).reshape(len(R), -1)[:, :1]
        Xs.append(np.concatenate([feats, en], axis=1))
        Ys.append(np.asarray(by, dtype=np.float64))
        got += len(R)
        if got >= n_max:
            break
    if got < n_max:
        raise RuntimeError(f"loader yielded only {got} < {n_max} samples")
    X = np.concatenate(Xs, axis=0)[:n_max]
    Y = np.concatenate(Ys, axis=0)[:n_max]
    return X, Y


def _sqdist(A, B):
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    D = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    return D


def _krr_fit(K_train, Y_train, alpha):
    """Dual coefficients of kernel ridge: (K + alpha I) dual = Y, via
    cholesky. Returns (n_train, n_outputs)."""
    Kreg = K_train.copy()
    Kreg[np.diag_indices_from(Kreg)] += alpha
    c, low = scipy.linalg.cho_factor(Kreg, lower=True, overwrite_a=True,
                                     check_finite=False)
    return scipy.linalg.cho_solve((c, low), Y_train, check_finite=False)


def _gauge_aligned_errors(g_pred, g_true, chunk=512):
    """Gauge-aligned relative parameter error per sample:
    ||G_pred(+diag shift) - G_true||_F / ||G_true||_F via engine.g_gen
    reconstruction + engine._shots_shifted_error."""
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


# ----------------------------------------------------------------------- run
def run(out_dir):
    t0 = time.time()
    spec = C.MODELS[ENSEMBLE_MODEL]
    engine.init_d20(h_type=spec["h_type"], state_type=spec["state_type"],
                    beta=C.BETA_THERMAL)
    engine.init_gram()
    half_dirs, val_dir = _ensure_datasets()

    if C.SMOKE:
        n_train, n_sel, n_eval = 256, 64, 16
    else:
        n_train, n_sel, n_eval = C.KRR_N_TRAIN, C.KRR_N_VAL, 4096

    # train + selection-val from the train halves (deterministic order);
    # held-out eval from the val stream (the SAME first-4096 protocol as
    # ablation_eval).
    load_bs = 64 if C.SMOKE else 2048
    train_loader = engine.NumpyLoader(half_dirs, load_bs, shuffle=False)
    X_all, Y_all = _collect_features(train_loader, n_train + n_sel)
    X_tr, Y_tr = X_all[:n_train], Y_all[:n_train]
    X_sel, Y_sel = X_all[n_train:n_train + n_sel], Y_all[n_train:n_train + n_sel]

    val_loader = engine.NumpyLoader(val_dir, load_bs, shuffle=False)
    X_te, Y_te = _collect_features(val_loader, n_eval)

    # standardize with train moments (declared; recorded in the JSON)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z_tr = (X_tr - mu) / sd
    Z_sel = (X_sel - mu) / sd
    Z_te = (X_te - mu) / sd
    n_feat = Z_tr.shape[1]
    gamma_scale = 1.0 / (n_feat * max(Z_tr.var(), 1e-12))

    gammas = [(g, gamma_scale if g == "scale" else float(g))
              for g in C.KRR_GAMMAS]
    alphas = [float(a) for a in C.KRR_ALPHAS]

    # ----------------------------------------------------- grid selection
    D_tr = _sqdist(Z_tr, Z_tr)
    D_sel = _sqdist(Z_sel, Z_tr)
    grid = []
    best = None
    for g_name, g_val in gammas:
        K_tr = np.exp(-g_val * D_tr)
        K_sel = np.exp(-g_val * D_sel)
        for alpha in alphas:
            t_fit = time.time()
            dual = _krr_fit(K_tr, Y_tr, alpha)
            pred_sel = K_sel @ dual
            sel_err = _gauge_aligned_errors(pred_sel, Y_sel)
            rec = {"gamma": g_name, "gamma_value": float(g_val),
                   "alpha": alpha,
                   "val_rel_param_err_median": float(np.median(sel_err)),
                   "val_rel_param_err_mean": float(np.mean(sel_err)),
                   "val_label_mse": float(np.mean((pred_sel - Y_sel) ** 2)),
                   "fit_s": time.time() - t_fit}
            grid.append(rec)
            if best is None or (rec["val_rel_param_err_median"]
                                < best["val_rel_param_err_median"]):
                best = rec
            del dual
        del K_tr, K_sel
    del D_tr, D_sel

    # ------------------------------------------------ refit best + heldout
    g_val = best["gamma_value"]
    alpha = best["alpha"]
    t_fit = time.time()
    K_tr = np.exp(-g_val * _sqdist(Z_tr, Z_tr))
    dual = _krr_fit(K_tr, Y_tr, alpha)
    refit_s = time.time() - t_fit
    del K_tr
    g_pred = np.exp(-g_val * _sqdist(Z_te, Z_tr)) @ dual

    err = _gauge_aligned_errors(g_pred, Y_te)
    heldout = {
        "n_samples": int(n_eval),
        "rel_param_err_median": float(np.median(err)),
        "rel_param_err_mean": float(np.mean(err)),
        "label_mse": float(np.mean((g_pred - Y_te) ** 2)),
    }

    npz_path = os.path.join(out_dir, "krr_predictions.npz")
    np.savez_compressed(npz_path, g_pred=g_pred, g_true=Y_te,
                        rel_param_err=err)

    result = {
        "protocol": {
            "kernel": "rbf",
            "features": ("triu(symmetrized rho2 pair block, k=0) + energy, "
                         "standardized with train-split moments"),
            "solver": "cholesky on (K + alpha I), numpy/scipy only",
            "n_train": int(n_train), "n_selection_val": int(n_sel),
            "n_heldout": int(n_eval), "n_features": int(n_feat),
            "alphas": alphas,
            "gammas": [g for g, _ in gammas],
            "gamma_scale_value": float(gamma_scale),
            "selection_metric": "median gauge-aligned relative parameter error",
            "train_seeds": list(C.TRAIN_SEEDS), "heldout_seed": C.VAL_SEED,
            "note": ("declared baseline protocol; full-dataset KRR is "
                     "O(N^3)-infeasible at 5e6 samples"),
            "smoke": bool(C.SMOKE),
            "provenance": "referee-response campaign 2026-06; satelite2/campaign",
        },
        "grid": grid,
        "best": best,
        "refit_s": refit_s,
        "heldout": heldout,
        "predictions_npz": npz_path,
        "task_wall_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "kernel_ridge.json"), "w") as f:
        json.dump(_jsonable(result), f, indent=1)

    payload = {
        "best_alpha": alpha, "best_gamma": best["gamma"],
        "heldout_rel_param_err_median": heldout["rel_param_err_median"],
        "heldout_rel_param_err_mean": heldout["rel_param_err_mean"],
        "n_train": int(n_train), "n_heldout": int(n_eval),
        "task_wall_s": time.time() - t0,
    }
    return _jsonable(payload)
