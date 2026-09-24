"""Training task driver: run(model_name, out_dir).

Thin driver per the campaign task contract:
  * era init via engine.init_d20 / engine.init_d12 + engine.init_gram()
  * dataset cache ds_{era}_{htype}_{state} under C.DATA_DIR
    (two train halves, seeds C.TRAIN_SEEDS; val 5%, seed C.VAL_SEED)
  * train via engine.train_model with the model spec, resumable checkpoints
    under C.CKPT_DIR/<model_name> and best-effort GCS sync
  * final engine.save_model_and_history + config.json + val metrics +
    predictions_val.npz (consumed by the GEVP/ablation tasks)

Engine calls go through a signature filter so the driver stays compatible
with the exact keyword set the training part (p3) exposes.
"""
import inspect
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import engine

INPUT_TYPE = "rho2kkbar"
PROVENANCE = "referee-response campaign 2026-06; satelite2/campaign"
TOGGLE_KEYS = ("use_scatter", "use_reinject", "use_orb_emb", "readout_bias")
EVAL_BS = 64 if C.SMOKE else 256


# --------------------------------------------------------------- utilities
def _call_filtered(fn, **kwargs):
    """Call fn with only the kwargs its signature accepts (aliases allowed)."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    accepted = {name: kwargs[name]
                for name, p in sig.parameters.items()
                if name in kwargs
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)}
    return fn(**accepted)


def _jsonable(o, _depth=0):
    if _depth > 12:
        return str(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in o]
    if isinstance(o, (str, bool)) or o is None:
        return o
    if isinstance(o, (int, float)):
        return o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 4096 else f"<ndarray shape={o.shape}>"
    if callable(o):
        return f"<callable {getattr(o, '__name__', '?')}>"
    return str(o)


def _hist_to_list(hist):
    """Normalize whatever history object train_model returned to [float]."""
    if hist is None:
        return []
    if isinstance(hist, dict):
        for k in ("loss", "train_loss", "history", "losses"):
            if k in hist:
                return _hist_to_list(hist[k])
        return []
    out = []
    try:
        for item in list(hist):
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                if isinstance(item, dict):
                    for k in ("loss", "train_loss"):
                        if k in item:
                            out.append(float(item[k]))
                            break
    except TypeError:
        return []
    return out


def _hardware():
    try:
        import jax
        devs = jax.devices()
        return f"{len(devs)}x{devs[0].device_kind} ({jax.default_backend()})"
    except Exception:
        return None


# --------------------------------------------------------------- GCS sync
def _gcs_ckpt_path(model_name):
    if C.SMOKE:  # no GCS traffic in smoke runs
        return None
    return f"{C.GCS_BUCKET}/checkpoints/{C.VM_NAME}/{model_name}"


def _gcs_push(ckpt_dir, model_name):
    """Best-effort checkpoint push; never raises."""
    if C.SMOKE:  # no GCS traffic in smoke runs
        return None
    try:
        C.gcs_sync_dir(ckpt_dir, _gcs_ckpt_path(model_name))
    except Exception:
        pass


def _gcs_pull_if_empty(ckpt_dir, model_name):
    """On resume with a fresh disk, try restoring the checkpoint from GCS."""
    if C.SMOKE:  # no GCS traffic in smoke runs
        return None
    try:
        if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
            return
        os.makedirs(ckpt_dir, exist_ok=True)
        C.gcs_sync_dir(_gcs_ckpt_path(model_name), ckpt_dir)
    except Exception:
        pass


# --------------------------------------------------------------- datasets
def _build_datasets(spec, beta):
    """Generate (or reuse) the shared per-ensemble dataset cache; return loaders."""
    key = f"ds_{spec['era']}_{spec['h_type']}_{spec['state_type']}"
    root = os.path.join(C.DATA_DIR, key)
    half_dirs = [os.path.join(root, "half0"), os.path.join(root, "half1")]
    val_dir = os.path.join(root, "val")

    n_half = max(1, int(spec["num_samples"]) // 2)
    n_val = max(1, int(round(spec["num_samples"] * 0.05)))
    # d12 era diagonalizes 924-dim H per sample: a 4096 gen batch would put
    # ~14 GB of eigh workspace per v5e chip -> OOM; 512 keeps it under ~2 GB.
    gen_bs = 64 if C.SMOKE else (512 if spec["era"] == "d12" else 4096)

    jobs = ((half_dirs[0], C.TRAIN_SEEDS[0], n_half),
            (half_dirs[1], C.TRAIN_SEEDS[1], n_half),
            (val_dir, C.VAL_SEED, n_val))

    # Cross-worker generation lock. Multiple workers on the same VM can share an
    # ensemble (e.g. workers 0/2/3 all use ds_d20_random_thermal). gen_dataset
    # rmtree's an "incomplete" cache before regenerating, so concurrent
    # generation of the SAME cache corrupts it. Serialize with an atomic mkdir
    # lock on the ensemble root: the winner generates all shards; losers poll
    # until the caches are complete, then fall through and just load them.
    lock = root + ".lock"
    os.makedirs(C.DATA_DIR, exist_ok=True)

    def _ready():
        return all(engine.check_existing_dataset(cp, n, gen_bs) for cp, _s, n in jobs)

    STALE_LOCK_S, WAIT_MAX_S, waited = 3 * 3600, 5 * 3600, 0
    while not _ready():
        try:
            os.mkdir(lock)                                   # atomic acquire
        except FileExistsError:
            try:                                             # break a dead worker's stale lock
                if time.time() - os.path.getmtime(lock) > STALE_LOCK_S:
                    os.rmdir(lock); continue
            except OSError:
                pass
            if waited >= WAIT_MAX_S:
                raise RuntimeError(f"timed out waiting for dataset {key} (lock held by another worker)")
            time.sleep(20); waited += 20
            continue
        try:                                                 # we hold the lock: generate
            for cache_path, seed, n in jobs:
                engine.gen_dataset(spec["h_type"], C.G_INIT, C.G_STOP,
                                   spec["state_type"], INPUT_TYPE, True, beta,
                                   num_samples=n, cache_path=cache_path,
                                   batch_size=gen_bs, seed=seed)
        finally:
            try: os.rmdir(lock)
            except OSError: pass
        break

    train_loader = engine.NumpyLoader(half_dirs, batch_size=gen_bs, shuffle=True)
    val_loader = engine.NumpyLoader(val_dir, batch_size=gen_bs, shuffle=False)
    info = {"key": key, "root": root, "train_seeds": list(C.TRAIN_SEEDS),
            "val_seed": C.VAL_SEED, "n_train": 2 * n_half, "n_val": n_val,
            "gen_batch_size": gen_bs}
    return train_loader, val_loader, info


# --------------------------------------------------------------- prediction
def _predict_val(state, val_loader, n_val, include_energy=True):
    """(g_pred, g_true, energies) on the val set, via predict_and_load if present."""
    if "predict_and_load" in engine.ns:
        try:
            out = _call_filtered(
                engine.ns["predict_and_load"], loader=val_loader,
                dataset=val_loader, state=state, num_samples=n_val,
                gpu_batch_size=EVAL_BS, include_energy=include_energy)
            if isinstance(out, tuple) and len(out) == 4:
                _, energy, g_true, g_pred = out
                return (np.asarray(g_pred), np.asarray(g_true),
                        np.asarray(energy))
        except Exception:
            pass  # fall through to the manual loop

    import jax.numpy as jnp
    preds, trues, ens = [], [], []
    seen = 0
    for bx, be, by in val_loader:
        n = len(by)
        for s in range(0, n, EVAL_BS):
            e = min(s + EVAL_BS, n)
            bxj = jnp.array(bx[s:e])
            byj = jnp.array(by[s:e])
            bej = jnp.array(be[s:e]) if (include_energy and be is not None) else None
            logits = engine.eval_step(state, bxj, bej, byj)
            preds.append(np.asarray(logits))
            trues.append(np.asarray(by[s:e]))
            ens.append(np.asarray(be[s:e]) if be is not None
                       else np.zeros((e - s, 1), np.float32))
        seen += n
        if seen >= n_val:
            break
    if not preds:
        raise ValueError("validation loader yielded no data")
    g_pred = np.concatenate(preds, axis=0)[:n_val]
    g_true = np.concatenate(trues, axis=0)[:n_val]
    energies = np.concatenate(ens, axis=0)[:n_val]
    return g_pred, g_true, energies


def _val_metrics(state, val_loader, is_thermal, beta, g_pred, g_true):
    dp = g_pred.astype(np.float64) - g_true.astype(np.float64)
    metrics = {
        "label_mse": float(np.mean(dp ** 2)),
        "label_rmse": float(np.sqrt(np.mean(dp ** 2))),
        "label_mae": float(np.mean(np.abs(dp))),
        "n_val": int(len(g_true)),
    }
    for fname in ("evaluate_run_metrics", "evaluate_universal_metrics"):
        if fname not in engine.ns:
            continue
        try:
            m = _call_filtered(
                engine.ns[fname], state=state, dataset=val_loader,
                loader=val_loader, val_loader=val_loader, include_energy=True,
                S_matrix_np=engine.ns.get("S_tensor_global"),
                gpu_batch_size=EVAL_BS, is_thermal=is_thermal, beta=beta)
            metrics[fname] = _jsonable(m)
            break
        except Exception as exc:  # quick metrics must never sink the run
            metrics[f"{fname}_error"] = str(exc)
    return metrics


# --------------------------------------------------------------- figures
def _plot_history(hist_list, out_dir, model_name):
    if not hist_list:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(hist_list) + 1), hist_list, marker="o", lw=2)
        if min(hist_list) > 0:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.set_title(model_name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(os.path.join(out_dir, f"loss_history_{model_name}.{ext}"), **kw)
        plt.close(fig)
    except Exception:
        pass


# --------------------------------------------------------------- main entry
def run(model_name, out_dir):
    spec = dict(C.MODELS[model_name])
    if spec.get("v2"):                      # [V2] production-run v2 lane -> t_train_v2
        from tasks import t_train_v2
        return t_train_v2.run(spec["lane"], out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---- era init (h_type/state_type from spec; thermal beta=1 on d20,
    #      gs handled by state_type; d12 keeps its era defaults)
    era = spec["era"]
    if era == "d20":
        beta = C.BETA_THERMAL
        engine.init_d20(spec["h_type"], spec["state_type"], beta=beta,
                        g_init=C.G_INIT, g_stop=C.G_STOP)
    elif era == "d12":
        engine.init_d12(spec["h_type"], spec["state_type"],
                        g_init=C.G_INIT, g_stop=C.G_STOP)
        beta = float(engine.BETA)
    else:
        raise ValueError(f"unknown era {era!r} for model {model_name}")
    engine.init_gram()
    is_thermal = (spec["state_type"] == "thermal")

    # ---- datasets (shared cache across models of the same ensemble)
    train_loader, val_loader, ds_info = _build_datasets(spec, beta)
    label_size = int(engine.g_gen.label_size())

    # ---- checkpoints: local dir, GCS restore on fresh disks
    ckpt_dir = os.path.join(C.CKPT_DIR, model_name)
    _gcs_pull_if_empty(ckpt_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    def sync_cb(*_a, **_k):
        _gcs_push(ckpt_dir, model_name)

    toggles = {k: spec[k] for k in TOGGLE_KEYS if k in spec}

    # ---- train (kwargs superset; signature filter keeps what p3 accepts)
    train_kwargs = dict(
        dataset=train_loader, loader=train_loader, train_loader=train_loader,
        label_size=label_size, input_type=INPUT_TYPE,
        M_PAIRS=int(engine.M_PAIRS), m_pairs=int(engine.M_PAIRS),
        include_energy=True,
        total_samples=ds_info["n_train"], num_samples=ds_info["n_train"],
        batch_size=spec["batch_size"], epochs=spec["epochs"],
        res=spec["res"], arch=spec["arch"],
        loss=spec["loss"], loss_type=spec["loss"],
        init_seed=spec["init_seed"], seed=spec["init_seed"],
        peak_lr=spec.get("peak_lr"), weight_decay=spec.get("weight_decay"),
        clip=spec.get("clip"),
        is_thermal=is_thermal, beta=beta,
        ckpt_dir=ckpt_dir, resume=True, sync_cb=sync_cb,
        **toggles)
    out = _call_filtered(engine.train_model, **train_kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        state, hist = out
    elif isinstance(out, dict):
        state = out.get("state")
        hist = out.get("hist", out.get("history"))
    else:
        state, hist = out, []
    hist_list = _hist_to_list(hist)
    epochs_run = len(hist_list) if hist_list else spec["epochs"]
    final_loss = hist_list[-1] if hist_list else None

    # ---- final save + provenance config.json + GCS push
    _call_filtered(engine.save_model_and_history, state=state, hist=hist,
                   history=hist, save_dir=ckpt_dir, ckpt_dir=ckpt_dir,
                   out_dir=ckpt_dir)

    n_params = None
    try:
        dim = int(engine.M_PAIRS) ** 2 if INPUT_TYPE == "rho2block" else int(engine.M_PAIRS)
        model = engine.build_model(spec["arch"], label_size, spec["res"],
                                   True, **toggles)
        n_params = int(engine.count_params(model, (1, dim, dim, 1), True))
    except Exception:
        pass

    model_config = {
        "model": model_name,
        "spec": _jsonable(spec),
        "n_params": n_params,
        "label_size": label_size,
        "input_type": INPUT_TYPE,
        "include_energy": True,
        "era": era,
        "beta": beta,
        "is_thermal": is_thermal,
        "dataset": ds_info,
        "hist": hist_list,
        "epochs_run": epochs_run,
        "hardware": _hardware(),
        "smoke": C.SMOKE,
        "vm": C.VM_NAME,
        "worker": C.WORKER_ID,
        "provenance": PROVENANCE,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=1)
    _gcs_push(ckpt_dir, model_name)

    # ---- val predictions (consumed by GEVP / ablation tasks) + quick metrics
    g_pred, g_true, energies = _predict_val(state, val_loader, ds_info["n_val"])
    np.savez_compressed(os.path.join(ckpt_dir, "predictions_val.npz"),
                        g_pred=g_pred, g_true=g_true, energies=energies)
    val_metrics = _val_metrics(state, val_loader, is_thermal, beta, g_pred, g_true)
    _gcs_push(ckpt_dir, model_name)

    _plot_history(hist_list, out_dir, model_name)

    payload = {"n_params": n_params, "final_loss": final_loss,
               "val_metrics": val_metrics, "epochs_run": epochs_run}

    task_name = os.path.basename(os.path.normpath(out_dir))
    results = {"model": model_name, "hist": hist_list,
               "dataset": ds_info, "wall_s": time.time() - t0, **payload}
    with open(os.path.join(out_dir, f"{task_name}.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=1)

    return _jsonable(payload)
