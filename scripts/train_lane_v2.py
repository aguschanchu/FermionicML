#!/usr/bin/env python
"""train_lane_v2 -- the v2 production trainer (ognrepro.v2 behind V2Config).

    python train_lane.py --v2 <lane> [--smoke] [--cache-dir DIR] [--out DIR]
                         [--resume] [--gcs gs://prefix] [--force]
    python train_lane.py --v2 --list

Per lane (ognrepro.v2.registry.V2_LANES): bind the engine era, build the
operator tables, ensure the FAMILY cache (half0/half1 seeds 42/43 + val seed
1007; HIGHEST precision; sidecar = family aux) under
<cache-dir>/datasets_v2/<key>, recompute the standardization statistics on the
lane's own training halves, resolve the V2Config, install the v2 model factory
and training step into the engine namespace and call the engine's train_model
with v2=cfg.  Outputs in <out>/<lane>/: final_state.msgpack, hist.json,
std_stats.npz, quick_val.json, predictions_val.npz, config.json, DONE.

Gates enforced here (plan Part 3): G1 schema+key (gen.cache_complete), G3
std stats train-only and recomputed per cache, G6 val split present, G9 param
pin (full runs), G10 config echo, G12 be width = 1 + aux.  Sealed-tree writes
are refused (guard on --out / --cache-dir).
"""
import argparse
import dataclasses
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_ROOT))
if os.path.isdir(os.path.join(_ROOT, "lib")):
    sys.path.insert(0, os.path.join(_ROOT, "lib"))
    os.environ.setdefault("OGNREPRO_ROOT", _ROOT)
if not os.environ.get("OGNREPRO_ENGINE_DIR") and os.path.isfile(os.path.join(_REPO, "campaign", "engine.py")):
    os.environ["OGNREPRO_ENGINE_DIR"] = os.path.join(_REPO, "campaign")

import numpy as np  # noqa: E402

SEALED_PARTS = ("manuscript", "release_bundle", "results/", "results_", "notebook_release/dist/fermionicml-notebooks-v1")
N_VAL_PRED = 4096


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def assert_writable_out(path):
    p = os.path.realpath(path).replace("\\", "/") + "/"
    for part in SEALED_PARTS:
        if ("/" + part) in p and "/results_v2/" not in p:
            raise SystemExit("[train_lane_v2] refusing to write under a sealed tree: %s" % path)


def _env_receipt():
    rec = dict(python=sys.version.split()[0], executable=sys.executable, hostname=socket.gethostname(),
               numpy=np.__version__, engine_dir=os.environ.get("OGNREPRO_ENGINE_DIR"))
    try:
        import jax  # noqa: PLC0415
        rec.update(jax=jax.__version__, backend=jax.default_backend(), n_local_devices=jax.local_device_count(),
                   device_kind=jax.local_devices()[0].device_kind, x64=bool(jax.config.read("jax_enable_x64")))
    except Exception as exc:  # noqa: BLE001
        rec["jax_error"] = str(exc)
    try:
        import flax  # noqa: PLC0415
        rec["flax"] = flax.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        rec["git_head"] = subprocess.check_output(["git", "-C", _REPO, "rev-parse", "HEAD"],
                                                  stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        rec["git_head"] = None
    return rec


def _hist_list(hist):
    if hist is None:
        return []
    if isinstance(hist, dict):
        for k in ("loss", "train_loss", "history", "losses"):
            if k in hist:
                return _hist_list(hist[k])
        return []
    out = []
    for item in list(hist):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            pass
    return out


def make_sync_cb(gcs_prefix, lane):
    """Blocking per-epoch mirror of the lane directory (G8); returns (cb, final)."""
    def _push(out_dir):
        dst = gcs_prefix.rstrip("/") + "/" + lane + "/"
        subprocess.run(["gsutil", "-m", "-q", "rsync", "-r", out_dir, dst], check=False)

    def cb(*_a, **kw):
        d = kw.get("ckpt_dir") or kw.get("save_dir") or (_a[0] if _a and isinstance(_a[0], str) else None)
        if d:
            _push(d)
    return cb, _push


def quick_val(engine, model, state, val_dir, cfg, ops, n_max=8192, batch=256):
    """Device-f32 quick metrics on the val split (gauge-aligned relative error,
    the p3_training.py:684-694 convention; rel-S error; M/psi0 quadratic
    residual when the family sidecar is present).  Diagnostic only: the
    scored numbers come from the exact64_cpu_highest stages."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from ognrepro.v2.loader import V2Loader  # noqa: PLC0415
    from ognrepro.v2 import ops_index as OI  # noqa: PLC0415

    params, bstats = state.params, state.batch_stats

    @jax.jit
    def apply(bx, be):
        return model.apply({"params": params, "batch_stats": bstats}, bx, be, training=False)

    m, A, D = int(ops["m"]), int(ops["A"]), int(ops["D"])
    aux = cfg.aux
    loader = V2Loader(val_dir, batch_size=batch, shuffle=False, aux=aux)
    S = np.asarray(ops["S"], np.float64)
    iu = np.triu_indices(A)
    rel, relS, resM, logits_all, labels_all, E_all = [], [], [], [], [], []
    n = 0
    for bx, be, by in loader:
        energy = be[:, :1]
        logits = np.asarray(apply(jnp.asarray(bx), jnp.asarray(energy)), np.float32)
        if n < N_VAL_PRED:
            logits_all.append(logits); labels_all.append(np.asarray(by, np.float32)); E_all.append(np.asarray(energy, np.float32))
        Gp = np.asarray(engine.g_gen.reconstruct(jnp.asarray(logits)), np.float64)
        Gt = np.asarray(engine.g_gen.reconstruct(jnp.asarray(by, jnp.float32)), np.float64)
        shift = (np.trace(Gt, axis1=1, axis2=2) - np.trace(Gp, axis1=1, axis2=2)) / m
        Ga = Gp + shift[:, None, None] * np.eye(m)
        num = np.linalg.norm((Ga - Gt).reshape(len(Gt), -1), axis=1)
        den = np.linalg.norm(Gt.reshape(len(Gt), -1), axis=1)
        rel.extend((num / den).tolist())
        dw = (Gp - Gt).reshape(len(Gt), -1)
        wt = Gt.reshape(len(Gt), -1)
        relS.extend(np.sqrt(np.einsum("bi,ij,bj->b", dw, S, dw) / np.einsum("bi,ij,bj->b", wt, S, wt)).tolist())
        if aux == "M":
            Mp = np.asarray(be[:, 1:], np.float64)
            Mf = np.zeros((len(dw), A, A)); Mf[:, iu[0], iu[1]] = Mp; Mf[:, iu[1], iu[0]] = Mp
            resM.extend(np.sqrt(np.maximum(np.einsum("bi,bij,bj->b", dw, Mf, dw), 0)).tolist())
        elif aux == "psi0":
            psi = np.asarray(be[:, 1:1 + D], np.float64)
            resM.extend([np.sqrt(max(OI.cov_gs_numpy(ops, psi[i], dw[i]), 0.0)) for i in range(len(dw))])
        n += len(by)
        if n >= n_max:
            break
    out = dict(n=int(n), rel_param_err_median=float(np.median(rel)), rel_param_err_mean=float(np.mean(rel)),
               rel_S_err_median=float(np.median(relS)), convention="device f32, gauge-aligned; diagnostic only")
    if resM:
        out["state_residual_median"] = float(np.median(resM))
    # keys as the campaign's predictions_val.npz (t_train.py): g_pred / g_true / energies
    preds = dict(g_pred=np.concatenate(logits_all)[:N_VAL_PRED], g_true=np.concatenate(labels_all)[:N_VAL_PRED],
                 energies=np.concatenate(E_all)[:N_VAL_PRED]) if logits_all else None
    return out, preds


def _smoke_cache_d16_v2(TL, CERT, spec_d16, cache_root):
    """--smoke d16: the release trainer's 8-shard half0/half1 smoke cache +
    a 2-shard val split + the psi0 sidecars (single build, no certificate)."""
    from ognrepro import blocked as B  # noqa: PLC0415
    from ognrepro.v2 import d16_aux  # noqa: PLC0415
    root, key = TL._smoke_cache_d16(spec_d16, cache_root)
    val_dir = os.path.join(root, "val")
    if not all(os.path.isfile(os.path.join(val_dir, B.shard_name(i))) for i in range(2)):
        eps = B.ladder(CERT.D16_M)
        blocks = B.build_blocks(CERT.D16_M, CERT.D16_N_ELEC)
        labels = CERT.mint_labels("val", spec_d16["h_type"], hi=2)
        CERT.build_shard_range_d16(labels, val_dir, 0, 2, spec_d16["h_type"], blocks, eps, beta=CERT.BETA_GS)
    d16_aux.build_psi0_cache(root, spec_d16["h_type"], {"half0": 4, "half1": 4, "val": 2}, nproc=2,
                             double=False, verbose=False, ds_key=key)
    return root, key


def _bind_d20(engine, s):
    beta = float(s.get("beta", 1.0))
    engine.init_d20(s["h_type"], s["state_type"], beta=beta, g_init=0.1, g_stop=1.0)
    engine.init_gram()
    return beta


def run_lane_v2(lane, args):
    from ognrepro import v2 as V2  # noqa: PLC0415
    from ognrepro.v2 import registry as R, gen as G, stats as ST, ops_index as OI  # noqa: PLC0415
    from ognrepro.v2.config import V2Config, resolve  # noqa: PLC0415
    from ognrepro.v2.loader import V2Loader  # noqa: PLC0415
    from ognrepro.v2.model import model_build_record  # noqa: PLC0415
    from ognrepro import repro_common  # noqa: PLC0415

    smoke = bool(args.smoke)
    s = R.spec(lane, smoke=smoke)
    out_root = os.path.abspath(args.out)
    cache_root = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(out_root, "ds_cache")
    assert_writable_out(out_root); assert_writable_out(cache_root)
    out_dir = os.path.join(out_root, lane)
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "DONE")
    if os.path.isfile(done_path) and not args.force:
        print("[train_lane_v2] %s already DONE (%s); skip" % (lane, done_path))
        return 0
    t0 = time.time()
    engine = repro_common.get_engine()
    receipt = _env_receipt()
    is_d16 = s["era"] == "d16n8"
    is_gs = s["state_type"] == "gs"
    era_block = None
    rebatch = None
    if is_d16:
        # certified caches (read-only) + psi0 sidecar; era install of the release trainer
        import train_lane as TL  # noqa: PLC0415
        import d16prog_certs as CERT  # noqa: PLC0415
        gen_bs = int(CERT.GEN_BS)
        ds_key = CERT.DS_KEYS[s["sector"]]
        spec_d16 = dict(h_type=s["h_type"], label_size=int(s["label_size"]), ds_key=ds_key)
        if smoke:
            ds_root, key = _smoke_cache_d16_v2(TL, CERT, spec_d16, cache_root)
            cache_receipt = dict(smoke=True, key=key, certificate_claim=False)
            splits = {"half0": 4, "half1": 4, "val": 2}
        else:
            if not args.cache_dir:
                raise SystemExit("[train_lane_v2] d16 lane needs --cache-dir with the certified caches")
            cache_receipt = CERT.verify_cache_v2(cache_root, ds_key)
            ds_root, key = os.path.join(cache_root, ds_key), ds_key
            splits = dict(CERT.SPLIT_NB)
        era_block = TL._install_era_d16(engine, spec_d16, gen_bs)
        beta = float(CERT.BETA_GS)
        ops = OI.build_d16_s0(TL.load_gram_s_checked())
        m, A, D = int(ops["m"]), int(ops["A"]), int(ops["D"])
        label_size = int(s["label_size"])
        assert int(engine.g_gen.label_size()) == label_size
        cfg = resolve(V2Config(**s["cfg"]), state_type="gs", label_size=label_size, m=m,
                      arch=s["arch"], use_energy_input=bool(s["use_energy_input"]))
        ccfg = R.cache_cfg(cfg, "gs")
        half_dirs = [os.path.join(ds_root, "half0"), os.path.join(ds_root, "half1")]
        val_dir = os.path.join(ds_root, "val")
        # the certified caches hold 7813 x 64 x 2 = 1,000,064 training rows; the published lanes trained on
        # exactly those rows (train_lane.py: n_train = SPLIT_ROWS half0 + half1, 3906 steps/epoch), the registry's
        # 1e6 is the nominal budget
        n_train = (512 if smoke else int(CERT.SPLIT_ROWS["half0"] + CERT.SPLIT_ROWS["half1"]))
        s["num_samples_nominal"] = int(s["num_samples"]); s["num_samples"] = n_train
        ds_info = dict(root=os.path.realpath(ds_root), key=key, n_train=n_train, gen_batch_size=gen_bs,
                       splits=splits, cache_receipt=cache_receipt, family_aux="psi0", loader_aux=cfg.aux,
                       rebatch_chunk=TL.TRAIN_CHUNK)
        rebatch = TL.TRAIN_CHUNK
        assert rebatch >= s["batch_size"] and rebatch % s["batch_size"] == 0, (rebatch, s["batch_size"])
    else:
        beta = _bind_d20(engine, s)
        ops = OI.build(engine)
        m, A, D = int(ops["m"]), int(ops["A"]), int(ops["D"])
        label_size = int(engine.g_gen.label_size())
        cfg = resolve(V2Config(**s["cfg"]), state_type=s["state_type"], label_size=label_size, m=m,
                      arch=s["arch"], use_energy_input=bool(s["use_energy_input"]))

        # ---- family cache (HIGHEST or the control's default precision; sidecar = family aux)
        ccfg = R.cache_cfg(cfg, s["state_type"]) if cfg.data_precision == "highest" or cfg.loss == "metric" \
            else dataclasses.replace(cfg, aux=None)
        gen_bs = int(s["gen_bs"])
        assert gen_bs >= s["batch_size"] and gen_bs % s["batch_size"] == 0, (gen_bs, s["batch_size"])
        key = G.dataset_key(s["era"], s["h_type"], s["state_type"], ccfg, tag=("smoke512" if smoke else None))
        ds_root = os.path.join(cache_root, "datasets_v2", key)
        half_dirs, val_dir, ds_info = G.ensure_caches(engine, ops, ds_root, era=s["era"], h_type=s["h_type"],
                                                      state_type=s["state_type"], beta=beta,
                                                      num_samples=int(s["cache_samples"]), gen_bs=gen_bs, cfg=ccfg)
        ds_info.update(key=key, family_aux=ccfg.aux, loader_aux=cfg.aux, rows_per_half=s.get("rows_per_half"))
        n_train = int(s["num_samples"])
        if s.get("rows_per_half"):
            assert 2 * int(s["rows_per_half"]) == n_train, (s["rows_per_half"], n_train)
        if getattr(args, "gen_only", False):
            certs = {os.path.basename(d): G.certify_cache(d) for d in half_dirs + [val_dir]}
            with open(os.path.join(ds_root, "certificate.json"), "w") as fh:
                json.dump(dict(key=key, cache_v2config=ccfg.to_json(), dataset=ds_info, certificate=certs,
                               host=socket.gethostname(), env=receipt, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                          fh, indent=1, default=str)
            print("[train_lane_v2] GEN-ONLY %s: cache %s complete; certificate written (%s)"
                  % (lane, key, {k: v.get("labels_sha256", "")[:12] for k, v in certs.items()}), flush=True)
            return 0

    # ---- standardization statistics: train halves only, recomputed for this cache (G3)
    std_stats = std_info = None
    if cfg.std_convention:
        n_max = 256 if smoke else int(cfg.std_n_max)
        if cfg.std_convention == "std1_entry":
            mu, sg, n_st = ST.std1_entry_stats(half_dirs, gen_bs, m, n_max=n_max, sigma_floor=cfg.sigma_floor,
                                               max_rows_per_dir=s.get("rows_per_half"))
        else:
            mu, sg, n_st = ST.triu55_stats(half_dirs, gen_bs, m, n_max=n_max, max_rows_per_dir=s.get("rows_per_half"))
        std_stats = (mu, sg)
        sp = ST.save_std_stats(os.path.join(out_dir, "std_stats.npz"), cfg.std_convention, mu, sg, n_st,
                               source="%s first %d rows of half0(+half1), shuffle=False" % (key, n_st))
        std_info = dict(kind=cfg.std_convention, n=int(n_st), shape=list(mu.shape), path=sp,
                        mean_sha256=_sha_arr(mu), std_sha256=_sha_arr(sg), cache_key=key)

    # ---- schedule + install
    steps_per_epoch = max(1, n_train // int(s["batch_size"]))
    total_steps = steps_per_epoch * int(s["epochs"])
    notes = V2.install(engine, cfg, ops, total_steps=total_steps, aux_kind=cfg.aux, is_gs=is_gs,
                       std_stats=std_stats, use_energy_input=bool(s["use_energy_input"]),
                       n_elec=int(engine.N_ELEC), mlp_width=s.get("mlp_width"), mlp_blocks=s.get("mlp_blocks", 8))
    loader = V2Loader(half_dirs, batch_size=gen_bs, shuffle=True, aux=cfg.aux,
                      max_rows_per_dir=s.get("rows_per_half"))
    if rebatch:
        import train_lane as TL  # noqa: PLC0415
        loader = TL.RebatchLoader(loader, rebatch)      # 64-row shards -> 512-row chunks (F-REBATCH)
    # G12: the packed channel is exactly [E | aux]
    bx0, be0, by0 = next(iter(V2Loader(half_dirs[:1], batch_size=gen_bs, shuffle=False, aux=cfg.aux)))
    want_w = 1 + G.aux_width(cfg.aux, A, D)
    assert be0.shape[1] == want_w, "[G12] be width %d != 1 + aux %d" % (be0.shape[1], want_w - 1)
    assert bx0.shape[1:] == (m, m, 1) and by0.shape[1] == label_size, (bx0.shape, by0.shape)

    toggles = dict(s.get("toggles") or {})
    mdl = engine.ns["build_model"](s["arch"], label_size, int(s["res"]), True, v2=cfg, **toggles)
    n_par = int(engine.count_params(mdl, (1, m, m, 1), True))
    pin = R.param_pin(s, m)
    if not smoke and pin is not None and not toggles and s["use_energy_input"]:
        assert n_par == pin, "[G9] n_params %d != pin %d" % (n_par, pin)
    if era_block is not None:
        ds_info["era_install"] = era_block
    build_rec = model_build_record(mdl, s["arch"], cfg, std_stats_sha=(std_info or {}).get("mean_sha256"))

    sync_cb = final_sync = None
    if args.gcs:
        sync_cb, final_sync = make_sync_cb(args.gcs, lane)

    print("[train_lane_v2] %s: arch=%s res=%d label=%d n_train=%d epochs=%d batch=%d seed=%d | "
          "data=%s aux=%s readout=%s/%s loss=%s lam=%g->%g@%.2f w_gauge=%g std=%s | total_steps=%d "
          "n_params=%d devices=%s"
          % (lane, s["arch"], s["res"], label_size, n_train, s["epochs"], s["batch_size"], s["init_seed"],
             cfg.data_precision, cfg.aux, cfg.readout, cfg.shell, cfg.loss, cfg.lam0, cfg.lam1, cfg.anneal_frac,
             cfg.w_gauge, cfg.std_convention, total_steps, n_par, receipt.get("n_local_devices")), flush=True)

    state, hist = engine.train_model(
        train_loader=loader, label_size=label_size, include_energy=True, input_shape=(1, m, m, 1),
        epochs=int(s["epochs"]), res=int(s["res"]), loss_type=cfg.loss, is_thermal=(not is_gs), beta=beta,
        init_seed=int(s["init_seed"]), arch=s["arch"], batch_size=int(s["batch_size"]), ckpt_dir=out_dir,
        resume=bool(args.resume), sync_cb=sync_cb, peak_lr=s["peak_lr"], weight_decay=s["weight_decay"],
        clip=s["clip"], total_samples=n_train, v2=cfg, **toggles)
    hist_list = _hist_list(hist)
    if not hist_list or not all(np.isfinite(hist_list)):
        raise SystemExit("[train_lane_v2] non-finite or empty loss history %r" % hist_list)
    step_kind = engine.ns.get("_last_train_step_kind")
    want_kind = "published" if (cfg.published() or cfg.loss == "rdm") else "v2"
    assert step_kind == want_kind, "[G10] engine served the %r train step, expected %r" % (step_kind, want_kind)
    engine.save_model_and_history(state, hist, out_dir)

    qv, preds = quick_val(engine, mdl, state, val_dir, cfg, ops, n_max=(256 if smoke else 8192),
                          batch=min(256, gen_bs))
    with open(os.path.join(out_dir, "quick_val.json"), "w") as fh:
        json.dump(qv, fh, indent=1)
    if preds is not None:
        np.savez(os.path.join(out_dir, "predictions_val.npz"), **preds)

    wall = time.time() - t0
    config = dict(cfg_version=cfg.cfg_version, lane=lane, spec={k: s[k] for k in sorted(s)}, smoke=smoke,
                  v2config=cfg.to_json(), cache_v2config=ccfg.to_json(), dataset=ds_info, std_stats=std_info,
                  model_build=build_rec, n_params=n_par, param_pin=pin, install=notes, train_step_kind=step_kind,
                  total_steps=total_steps, steps_per_epoch=steps_per_epoch, hist=hist_list,
                  epoch_seconds=(hist.get("epoch_seconds") if isinstance(hist, dict) else None),
                  final_loss=hist_list[-1], epochs_run=len(hist_list), quick_val=qv, env=receipt,
                  argv=sys.argv, wall_s=wall,
                  final_state_sha256=_sha256_file(os.path.join(out_dir, "final_state.msgpack")),
                  produced_by="notebook_release/src/scripts/train_lane_v2.py")
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1, default=str)
    with open(done_path, "w") as fh:
        fh.write("%s\n" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    if final_sync:
        final_sync(out_dir)
    print("[train_lane_v2] DONE %s: epochs=%d final_loss=%.6g wall=%.1fs quick_val=%s"
          % (lane, len(hist_list), hist_list[-1], wall, json.dumps(qv)), flush=True)
    return 0


def add_args(ap):
    ap.add_argument("--gcs", default=None, help="v2: gs:// prefix mirrored per epoch and at the end")
    ap.add_argument("--force", action="store_true", help="v2: rerun a lane that has a DONE marker")
    ap.add_argument("--gen-only", action="store_true",
                    help="v2: build + certify the lane's family cache, then exit (no training)")
    return ap


def list_lanes():
    from ognrepro.v2 import registry as R  # noqa: PLC0415
    for name, s in R.V2_LANES.items():
        print("%-32s tier=%-2s era=%-5s h=%-6s state=%-7s arch=%-7s res=%d ep=%-2d n=%.1e seed=%d cfg=%s"
              % (name, s["tier"], s["era"], s["h_type"], s["state_type"], s["arch"], s["res"], s["epochs"],
                 s["num_samples"], s["init_seed"], s["cfg"] or "v2"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lane", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="./retrain_out_v2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--list", action="store_true")
    add_args(ap)
    args = ap.parse_args(argv)
    if args.list:
        list_lanes(); return 0
    if not args.lane:
        ap.error("lane required (or --list)")
    return run_lane_v2(args.lane, args)


if __name__ == "__main__":
    sys.exit(main())
