#!/usr/bin/env python
"""train_lane -- clean full-scale retraining driver for every bundled lane.

    python train_lane.py <lane> [--smoke] [--cache-dir DIR] [--out DIR]
                         [--seed-override N] [--device-report]
                         [--std-stats PATH] [--gram-s PATH] [--resume]
    python train_lane.py --list
    python train_lane.py --device-report          (receipt only, no training)

Engine-based and campaign-free: no gate machinery, no record minting, no
GCS.  Reproduces the published training protocols exactly (same engine
train_model, same loss/optimizer/seed conventions, same cache geometry);
retrained weights are hardware/stack-dependent, so equivalence to the
published checkpoints is at the forward-pass/score level, never byte level
(see RETRAIN.md).

LANES (transcribed specs; sources cited at the registry below):
  * 14 d20/d12-era lanes  -- campaign4/config.py MODELS:126-171 (the 13
    bundled names) + ogn_noE from campaign5/train/config5.py:157-161;
    live dataset generation via engine.gen_dataset (campaign4
    tasks/t_train.py:136-195 conventions: splits half0/half1 at seeds
    42/43, gen_bs 4096 d20 / 512 d12, beta 1.0 thermal / era default).
  * 12 d16prog lanes      -- experiments/experiment-campaign2-r2/stages/
    t_d16prog_lanes.py (frozen roster :18-30, base spec :1643-1664):
    2 sectors {const_gs, vect_gs} x {orig, std} x seeds {42,43,44},
    epochs 10, 1e6 samples, batch 256, peak_lr 3e-4, wd 1e-4, clip 1.0,
    gen_bs 64, beta=100 GS convention.  Caches are READ from --cache-dir
    and digest-gated against the embedded build certificates BEFORE
    training (d16prog_certs.verify_cache); train_lane never builds a
    full-scale d16 cache (build_cache.py does).

--smoke: any lane at 512 samples / 1 epoch / res 1 / batch 64 / gen_bs 64
(CPU-runnable in minutes).  d16 lanes generate a 512-row blocked cache on
the fly (ognrepro.blocked solver; labels CPU-minted -- for the const
sector that is a disclosed different stream class, fine for smoke).
"""
import argparse
import hashlib
import inspect
import json
import os
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                     # bundle root (or src/)
if os.path.isdir(os.path.join(_ROOT, "lib")):
    sys.path.insert(0, os.path.join(_ROOT, "lib"))
    os.environ.setdefault("OGNREPRO_ROOT", _ROOT)
sys.path.insert(0, _HERE)                          # d16prog_certs

import numpy as np                                  # noqa: E402

import d16prog_certs as CERT                        # noqa: E402

# ---------------------------------------------------------------- registry
# Shared optimizer block [campaign4/config.py:126 _PROD_OPT]
_OPT = dict(batch_size=256, peak_lr=3e-4, weight_decay=1e-4, clip=1.0)

# ---- 14 d20/d12-era bundled checkpoints.
# [campaign4/config.py MODELS:129-171, transcribed field-for-field; ogn_noE
#  from campaign5/train/config5.py:157-161.  Consistent with
#  ognrepro.restore.D20_MODELS (arch/res/toggles/label sizes).  Every
#  campaign call site trains with include_energy=True; loss key 'gram' /
#  'rdm' selects the p3_training branch, anything else falls through to
#  label-MSE.]
LANES = {
    "prod_thermal_random": dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="ogn", res=3,
                                init_seed=42, **_OPT),
    "prod_gs_random":      dict(era="d20", h_type="random", state_type="gs",
                                num_samples=5_000_000, epochs=50,
                                loss="gram", arch="ogn", res=3,
                                init_seed=42, **_OPT),
    "prod_gs_const":       dict(era="d20", h_type="const", state_type="gs",
                                num_samples=2_000_000, epochs=10,
                                loss="gram", arch="ogn", res=3,
                                init_seed=42, **_OPT),
    "nearinit_random":     dict(era="d20", h_type="random", state_type="gs",
                                num_samples=10_000, epochs=1, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_OPT),
    "d12_const":           dict(era="d12", h_type="const", state_type="gs",
                                num_samples=1_000_000, epochs=10,
                                loss="gram", arch="ogn", res=3,
                                init_seed=42, **_OPT),
    "d12_vect":            dict(era="d12", h_type="vect", state_type="gs",
                                num_samples=1_000_000, epochs=10,
                                loss="gram", arch="ogn", res=3,
                                init_seed=42, **_OPT),
    "abl_mlp":             dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="mlp", res=4,
                                init_seed=42, **_OPT),
    "abl_noscatter":       dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="ogn", res=3,
                                init_seed=42, use_scatter=False, **_OPT),
    "abl_noreinject":      dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="ogn", res=3,
                                init_seed=42, use_reinject=False, **_OPT),
    "abl_neutralbias":     dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="ogn", res=3,
                                init_seed=42, readout_bias=0.0, **_OPT),
    "gram_thermal_random_50":    dict(era="d20", h_type="random",
                                      state_type="thermal",
                                      num_samples=5_000_000, epochs=50,
                                      loss="gram", arch="ogn", res=3,
                                      init_seed=42, **_OPT),
    "rdm_thermal_random_50":     dict(era="d20", h_type="random",
                                      state_type="thermal",
                                      num_samples=5_000_000, epochs=50,
                                      loss="rdm", arch="ogn", res=3,
                                      init_seed=42, **_OPT),
    "rdm_thermal_random_50_s43": dict(era="d20", h_type="random",
                                      state_type="thermal",
                                      num_samples=5_000_000, epochs=50,
                                      loss="rdm", arch="ogn", res=3,
                                      init_seed=43, **_OPT),
    # [campaign5/train/config5.py:157-161] V2-F01 energy-channel ablation
    "ogn_noE":             dict(era="d20", h_type="random",
                                state_type="thermal", num_samples=5_000_000,
                                epochs=25, loss="gram", arch="ogn", res=3,
                                init_seed=42, use_energy_input=False,
                                **_OPT),
}

# ---- 12 d16prog lanes [t_d16prog_lanes.py:18-30 roster, :754-792 SECTORS,
#      :1643-1664 spec factory: d12 GS budget lifted to d16; std arm delta
#      is EXACTLY arch->'ogn_std' + standardize].  Era d16n8: D_SP 16,
#      N_ELEC 8, m 8, beta=100 GS convention (p1_core.py:413), gen_bs 64.
#      label_size: const 1, vect ceil(8/2)-1 = 3 (p1_core.py:470-482).
for _sector, _ht, _ls in (("const_gs", "const", 1), ("vect_gs", "vect", 3)):
    for _kind in ("orig", "std"):
        for _seed in (42, 43, 44):
            _name = "d16prog%s_%s%s" % (_kind, _sector,
                                        "" if _seed == 42 else "_s%d" % _seed)
            _s = dict(era="d16n8", h_type=_ht, state_type="gs",
                      num_samples=1_000_000, epochs=10, loss="gram",
                      arch=("ogn_std" if _kind == "std" else "ogn"), res=3,
                      init_seed=_seed, label_size=_ls, sector=_sector,
                      ds_key=CERT.DS_KEYS[_sector], **_OPT)
            if _kind == "std":
                _s["standardize"] = True
            LANES[_name] = _s

D16_SECTOR_N_PARAMS = {"const_gs": 17_771_458,     # [t_d16prog_lanes:295-296]
                       "vect_gs": 17_772_228}
TRAIN_CHUNK = 512          # [t_d16prog_lanes:256-258] F-REBATCH d12 geometry
SIGMA_FLOOR = 1e-8         # [t_d16prog_lanes:261] STD-1 per-entry floor
G_INIT, G_STOP = 0.1, 1.0  # [campaign4/config.py:64]
BETA_THERMAL_D20 = 1.0     # [campaign4/config.py:65]
TRAIN_SEEDS = (42, 43)     # [campaign4/config.py:69]
INPUT_TYPE = "rho2kkbar"   # [campaign4/tasks/t_train.py:29]

GEN_BS_GOTCHA = (
    "gen_bs (the NumpyLoader chunk size) must be >= batch_size and a "
    "multiple of it: p3_training.py:433-438 slices each loader chunk into "
    "exact batch_size pieces and SKIPS any partial piece, so a smaller "
    "gen_bs silently yields ZERO optimizer steps and a NaN epoch.")


# ---------------------------------------------------------------- helpers
def _call_filtered(fn, **kwargs):
    """campaign4 tasks/t_train.py:36-46 -- call with accepted kwargs only."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD
           for p in sig.parameters.values()):
        return fn(**kwargs)
    accepted = {name: kwargs[name] for name, p in sig.parameters.items()
                if name in kwargs
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)}
    return fn(**accepted)


def _hist_to_list(hist):
    if hist is None:
        return []
    if isinstance(hist, dict):
        for k in ("loss", "train_loss", "history", "losses"):
            if k in hist:
                return _hist_to_list(hist[k])
        return []
    out = []
    for item in list(hist):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            pass
    return out


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _env_receipt():
    rec = dict(python=sys.version.split()[0], executable=sys.executable,
               hostname=socket.gethostname(),
               numpy=np.__version__)
    try:
        import jax
        rec.update(jax=jax.__version__, backend=jax.default_backend(),
                   n_local_devices=jax.local_device_count(),
                   device_kind=jax.local_devices()[0].device_kind,
                   x64=bool(jax.config.read("jax_enable_x64")))
    except Exception as exc:                                 # noqa: BLE001
        rec["jax_error"] = str(exc)
    try:
        import flax
        rec["flax"] = flax.__version__
    except Exception:                                        # noqa: BLE001
        pass
    return rec


def _print_receipt(rec, title):
    print("---- %s" % title)
    for k in sorted(rec):
        print("  %-16s %s" % (k, rec[k]))


def _get_engine():
    from ognrepro import repro_common
    return repro_common.get_engine()


# ---------------------------------------------------------------- loaders
class RebatchLoader:
    """[t_d16prog_lanes.py:3009-3064 _RebatchLoader, machinery-free copy]
    Concatenate consecutive NumpyLoader chunks so the engine's train loop
    sees `chunk` rows at a time.  WHY: the gen_bs=64 d16 shard geometry
    would otherwise hit the p3_training.py:433-438 skip (see
    GEN_BS_GOTCHA); chunk=512 reproduces the d12 chunk geometry exactly
    (one chunk -> two 256-row optimizer steps).  Trailing partial chunks
    are dropped -- the engine would skip them anyway."""

    def __init__(self, inner, chunk=TRAIN_CHUNK):
        self.inner = inner
        self.chunk = int(chunk)

    def __iter__(self):
        bx, be, by, n = [], [], [], 0
        for cx, ce, cy in self.inner:
            bx.append(np.asarray(cx))
            by.append(np.asarray(cy))
            if ce is not None:
                be.append(np.asarray(ce))
            n += len(bx[-1])
            while n >= self.chunk:
                X = np.concatenate(bx, 0)
                Y = np.concatenate(by, 0)
                E = np.concatenate(be, 0) if be else None
                item = (X[:self.chunk],
                        None if E is None else E[:self.chunk],
                        Y[:self.chunk])
                rx, ry = X[self.chunk:], Y[self.chunk:]
                rE = None if E is None else E[self.chunk:]
                bx = [rx] if len(rx) else []
                by = [ry] if len(ry) else []
                be = [rE] if (rE is not None and len(rE)) else []
                n = len(rx)
                yield item


# ---------------------------------------------------------------- d20/d12
def _build_datasets_d20(engine, spec, beta, cache_root, smoke):
    """campaign4 tasks/t_train.py:136-195 conventions, lock-free (single
    process): shared cache ds_<era>_<htype>_<state>/{half0,half1}, seeds
    TRAIN_SEEDS=(42,43); gen_bs 4096 d20 / 512 d12 / 64 smoke.  The val
    split is not generated (train_lane trains only; the sealed evaluation
    streams live in the bundle's data/streams)."""
    key = "ds_%s_%s_%s" % (spec["era"], spec["h_type"], spec["state_type"])
    if smoke:
        key += "_smoke512"
    root = os.path.join(cache_root, key)
    half_dirs = [os.path.join(root, "half0"), os.path.join(root, "half1")]
    n_half = max(1, int(spec["num_samples"]) // 2)
    gen_bs = 64 if smoke else (512 if spec["era"] == "d12" else 4096)
    assert gen_bs >= spec["batch_size"] and gen_bs % spec["batch_size"] == 0, (
        "[train_lane] gen_bs=%d vs batch_size=%d: %s"
        % (gen_bs, spec["batch_size"], GEN_BS_GOTCHA))
    os.makedirs(root, exist_ok=True)
    for cache_path, seed in zip(half_dirs, TRAIN_SEEDS):
        engine.gen_dataset(spec["h_type"], G_INIT, G_STOP,
                           spec["state_type"], INPUT_TYPE, True, beta,
                           num_samples=n_half, cache_path=cache_path,
                           batch_size=gen_bs, seed=seed)
    loader = engine.NumpyLoader(half_dirs, batch_size=gen_bs, shuffle=True)
    info = dict(key=key, root=os.path.realpath(root),
                train_seeds=list(TRAIN_SEEDS), n_train=2 * n_half,
                gen_batch_size=gen_bs, generated_live=True)
    return loader, info


def _noe_build_model_patch(engine):
    """ogn_noE: the vendored campaign4 engine's PhysicsOrbitalGraphNet has
    no use_energy_input attribute (p2_models.py:264,:353-377), so the
    energy-ablated architecture is built through ognrepro.arch_variants
    (the campaign5 p2_models5 copy whose defaults reproduce the production
    module bit-for-bit -- same class name, same flax auto-naming).  The
    patch rebinds engine.ns['build_model'] for THIS process only, exactly
    the t_stdd12.py:1806-1832 rebind pattern."""
    from ognrepro import arch_variants
    orig = engine.ns["build_model"]

    def build_model_noe(arch, label_size, res, include_energy,
                        use_scatter=True, use_reinject=True,
                        use_orb_emb=True, readout_bias=0.55):
        if arch == "ogn":
            return arch_variants.build_ogn(
                label_size=label_size, res=res,
                include_energy=include_energy, use_scatter=use_scatter,
                use_reinject=use_reinject, use_orb_emb=use_orb_emb,
                readout_bias=readout_bias, use_energy_input=False)
        return orig(arch, label_size, res, include_energy,
                    use_scatter=use_scatter, use_reinject=use_reinject,
                    use_orb_emb=use_orb_emb, readout_bias=readout_bias)

    engine.ns["build_model"] = build_model_noe
    return "arch_variants.build_ogn(use_energy_input=False)"


# ---------------------------------------------------------------- d16 era
def _install_era_d16(engine, spec, gen_bs):
    """Minimal d16n8 era install for the TRAINER, into engine.ns.  Mirrors
    t_d16prog_lanes._install_era_d16prog (:2727-2820) MINUS everything the
    gram train step never reads:

      * NO fmb FixedBasis / rho tensors: GGenerator accepts the integer
        mode count (p1_core.py:449 `basis.d if hasattr(basis,'d') else
        basis`), and the 'gram' train step (p3_training.py:186-207)
        consumes ONLY S_tensor + g_gen.reconstruct + the batch itself.
      * S_tensor_global comes from the PINNED 64x64 artifact (dual
        sha-gated), not a live sparse rebuild -- identical bytes by pin.
      * d_inter_tensor_global = None exactly as the lanes set it
        (flax replicate(None) is None -- asserted); d_rho_1_diag_global and
        base_e_params_global are dead operands of the gram branch (only the
        'rdm' branch reads them): base_e is installed with the true era
        ladder, rho_1_diag as zeros of the true (16, 12870) shape.  Neither
        value can affect a gram-loss gradient (unused in the traced graph).
    """
    import jax.numpy as jnp
    from flax.jax_utils import replicate as _flax_replicate
    ns = engine.ns
    m = CERT.D16_M
    for k in ("S_matrix_np", "S_tensor_global", "d_rho_1_diag_global",
              "d_inter_tensor_global", "base_e_params_global"):
        ns.pop(k, None)                      # stale-metric leak guard
    levels = np.arange(0, m) - m // 2 + 0.5  # [F-LADDER, from M_PAIRS]
    energies = np.repeat(levels, 2).astype(np.float64) / 1.0
    ns.update({
        "D_SP": CERT.D16_D_SP, "N_ELEC": CERT.D16_N_ELEC, "M_PAIRS": m,
        "USE_PAIRING_RESTRICTION": False, "GPU_BATCH_SIZE": int(gen_bs),
        "GEN_GPU_BATCH_SIZE": 64, "BETA": CERT.BETA_GS, "SCALE_FACTOR": 1.0,
        "levels": levels, "energies": energies,
        "U_ENERGY_SEED": np.array([energies for _ in range(int(gen_bs))]),
        "g_gen": ns["GGenerator"](CERT.D16_D_SP, spec["h_type"],
                                  int(gen_bs), g_init=G_INIT, g_stop=G_STOP),
        "STATE_TYPE": "gs",
        "rho_2_arrays": None, "rho_2_block_arrays": None,
    })
    got_ls = int(ns["g_gen"].label_size())
    assert got_ls == spec["label_size"], (
        "[train_lane] derived label_size %d != sector's %d"
        % (got_ls, spec["label_size"]))
    S = load_gram_s_checked()
    assert _flax_replicate(None) is None, (
        "[train_lane] flax replicate(None) is no longer None -- the "
        "d_inter_tensor bypass contract broke (t_d16probe patch 5.2)")
    ns.update({
        "S_matrix_np": S,
        "S_tensor_global": jnp.array(S, dtype=jnp.float32),
        "d_rho_1_diag_global": jnp.zeros((CERT.D16_D_SP, CERT.D16_BASIS),
                                         jnp.float32),
        "base_e_params_global": jnp.array(ns["U_ENERGY_SEED"][0]),
        "d_inter_tensor_global": None,
    })
    return dict(era="d16n8", d_sp=CERT.D16_D_SP, n_elec=CERT.D16_N_ELEC,
                m_pairs=m, beta=CERT.BETA_GS,
                beta_authority=("p1_core.py:413 effective_beta = beta if "
                                "is_thermal else 100.0 -- full-spectrum "
                                "softmax at beta=100"),
                label_size=got_ls, gram_s_array_sha256_pin_matched=True,
                d_inter_tensor_global=None,
                rho_1_diag_note=("zeros placeholder; dead operand of the "
                                 "gram train step (only the rdm branch "
                                 "reads it -- p3_training.py:209-242)"))


_GRAM_S_PATH = [None]


def load_gram_s_checked():
    path = _GRAM_S_PATH[0] or CERT.find_gram_s(bundle_root=_ROOT)
    return CERT.load_gram_s(path)


def _smoke_cache_d16(spec, cache_root):
    """--smoke: 512-row on-the-fly blocked cache (8 shards of 64: half0
    0..3 at seed 42, half1 0..3 at seed 43 -- strict prefixes of the
    canonical label chains).  CPU-minted labels: for the const sector this
    is the disclosed cpu-blocked stream class, NOT the certified TPU
    extract -- fine for smoke, no certificate claim is made."""
    from ognrepro import blocked as B
    key = spec["ds_key"] + "_smoke512"
    root = os.path.join(cache_root, key)
    eps = B.ladder(CERT.D16_M)
    blocks = B.build_blocks(CERT.D16_M, CERT.D16_N_ELEC)
    t0 = time.time()
    for split, n_shards in (("half0", 4), ("half1", 4)):
        labels = CERT.mint_labels(split, spec["h_type"], hi=n_shards)
        CERT.build_shard_range_d16(labels, os.path.join(root, split),
                                   0, n_shards, spec["h_type"], blocks, eps,
                                   beta=CERT.BETA_GS)
    print("[train_lane] smoke d16 cache at %s: 8 shards x 64 rows in %.1fs "
          "(labels CPU-minted; const sector: disclosed stream class, smoke "
          "only)" % (root, time.time() - t0))
    return root, key


def _std_stats_from_cache(engine, root, gen_bs, m, n_max):
    """STD-1 stats estimator [t_stdd12.py:2209-2254 _std_stats, verbatim
    numerics]: per-entry mean/std of the symmetrized m x m block over the
    first n_max rows of half0+half1 (shuffle=False), sigma floored 1e-8.
    Smoke fallback only -- full-scale std lanes load the bundled
    std_stats.npz sidecar."""
    half_dirs = [os.path.join(root, "half0"), os.path.join(root, "half1")]
    loader = engine.NumpyLoader(half_dirs, batch_size=gen_bs, shuffle=False)
    n = 0
    s = np.zeros((m, m), np.float64)
    ss = np.zeros((m, m), np.float64)
    for bx, _be, _by in loader:
        X = np.asarray(bx, np.float64)
        if X.ndim == 4:
            X = X[..., 0]
        assert X.shape[1] == X.shape[2] == m, X.shape
        Xs = 0.5 * (X + np.swapaxes(X, 1, 2))
        s += Xs.sum(0)
        ss += (Xs * Xs).sum(0)
        n += len(Xs)
        if n >= n_max:
            break
    if n == 0:
        raise RuntimeError("[train_lane] std stats: loader empty")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    sigma = np.maximum(np.sqrt(var), SIGMA_FLOOR)
    return mean.astype(np.float32), sigma.astype(np.float32), int(n)


def _std_build_model_patch(engine, mu, sigma):
    """arch 'ogn_std' -> ognrepro.std_arch.StdPhysicsOrbitalGraphNet with
    per-process stats; every other arch falls through to the engine
    factory.  [t_stdd12.py:1806-1832 _build_model_std/_install_..., with
    the stats passed explicitly instead of ambient module state -- one lane
    per process holds by construction here.]"""
    from ognrepro import std_arch
    mu_t, sg_t = std_arch.std_stats_tuples(mu, sigma)
    orig = engine.ns["build_model"]

    def build_model_std(arch, label_size, res, include_energy,
                        use_scatter=True, use_reinject=True,
                        use_orb_emb=True, readout_bias=0.55):
        if arch == "ogn_std":
            return std_arch.StdPhysicsOrbitalGraphNet(
                label_size=label_size, res=res,
                include_energy=include_energy, use_scatter=use_scatter,
                use_reinject=use_reinject, use_orb_emb=use_orb_emb,
                readout_bias=readout_bias, use_energy_input=True,
                std_mu=mu_t, std_sigma=sg_t)
        return orig(arch, label_size, res, include_energy,
                    use_scatter=use_scatter, use_reinject=use_reinject,
                    use_orb_emb=use_orb_emb, readout_bias=readout_bias)

    engine.ns["build_model"] = build_model_std
    return "std_arch.StdPhysicsOrbitalGraphNet(per-lane stats)"


def _resolve_std_stats(lane, spec, args, engine, cache_root, gen_bs):
    """std arms: mu/sigma from --std-stats, else the bundled sidecar
    checkpoints/d16prog/<lane>/std_stats.npz, else (smoke only) recompute
    from the smoke cache with the t_stdd12 estimator."""
    cands = []
    if args.std_stats:
        if not os.path.isfile(args.std_stats):
            raise SystemExit("[train_lane] --std-stats %r does not exist"
                             % args.std_stats)
        cands.append(args.std_stats)
    cands.append(os.path.join(_ROOT, "checkpoints", "d16prog", lane,
                              "std_stats.npz"))
    for p in cands:
        if p and os.path.isfile(p):
            with np.load(p) as z:
                mu = np.asarray(z["mu"])
                sigma = np.asarray(z["sigma"])
            assert mu.shape == sigma.shape == (CERT.D16_M, CERT.D16_M), (
                "[train_lane] std stats %r are %r; d16 wants 8x8"
                % (p, mu.shape))
            src = p
            return mu, sigma, dict(
                source=os.path.realpath(src),
                mu_sha256=hashlib.sha256(mu.tobytes()).hexdigest(),
                sigma_sha256=hashlib.sha256(sigma.tobytes()).hexdigest())
    if not args.smoke:
        raise FileNotFoundError(
            "[train_lane] std lane %s needs std_stats.npz (searched %r); "
            "pass --std-stats or place the bundle checkpoints alongside "
            "this script's parent directory." % (lane, cands))
    mu, sigma, n = _std_stats_from_cache(engine, cache_root, gen_bs,
                                         CERT.D16_M, 256)
    print("[train_lane] smoke std stats recomputed from the smoke cache "
          "(n=%d, t_stdd12 estimator, sigma floor %g)" % (n, SIGMA_FLOOR))
    return mu, sigma, dict(source="smoke-cache-recompute", n_stats=n)


# ---------------------------------------------------------------- runner
def run_lane(lane, args):
    spec = dict(LANES[lane])
    smoke = bool(args.smoke)
    if args.seed_override is not None:
        spec["init_seed"] = int(args.seed_override)
        print("[train_lane] seed override: init_seed=%d (dataset label "
              "streams stay canonical)" % spec["init_seed"])
    if smoke:  # [campaign4/config.py:204-209 smoke overrides + gen_bs 64]
        spec.update(num_samples=512, epochs=1, res=1, batch_size=64)
    t0 = time.time()
    out_dir = os.path.join(os.path.abspath(args.out), lane)
    os.makedirs(out_dir, exist_ok=True)
    cache_root = os.path.abspath(args.cache_dir) if args.cache_dir else \
        os.path.join(os.path.abspath(args.out), "ds_cache")

    engine = _get_engine()
    receipt = _env_receipt()
    if args.device_report:
        _print_receipt(receipt, "device/env receipt")
    is_d16 = spec["era"] == "d16n8"
    build_note = None
    ds_info = {}

    if is_d16:
        gen_bs = CERT.GEN_BS
        if smoke:
            root, key = _smoke_cache_d16(spec, cache_root)
            cache_receipt = dict(smoke=True, key=key,
                                 root=os.path.realpath(root),
                                 certificate_claim=False)
            n_train = 512
        else:
            if not args.cache_dir:
                raise SystemExit(
                    "[train_lane] full-scale d16 lane %s needs --cache-dir "
                    "pointing at the certified blocked caches (build them "
                    "with build_cache.py)." % lane)
            cache_receipt = CERT.verify_cache(cache_root, spec["ds_key"])
            root = os.path.join(cache_root, spec["ds_key"])
            n_train = CERT.SPLIT_ROWS["half0"] + CERT.SPLIT_ROWS["half1"]
        era_block = _install_era_d16(engine, spec, gen_bs)
        if spec["arch"] == "ogn_std":
            mu, sigma, std_src = _resolve_std_stats(lane, spec, args,
                                                    engine, root, gen_bs)
            build_note = _std_build_model_patch(engine, mu, sigma)
            era_block["std_stats"] = std_src
        # F-REBATCH: 64-row shards -> 512-row chunks for the train loop
        chunk = TRAIN_CHUNK
        assert chunk >= spec["batch_size"] and \
            chunk % spec["batch_size"] == 0, (
            "[train_lane] rebatch chunk=%d vs batch_size=%d: %s"
            % (chunk, spec["batch_size"], GEN_BS_GOTCHA))
        half_dirs = [os.path.join(root, "half0"), os.path.join(root, "half1")]
        loader = RebatchLoader(
            engine.NumpyLoader(half_dirs, batch_size=gen_bs, shuffle=True),
            chunk)
        beta = CERT.BETA_GS
        is_thermal = False
        m = CERT.D16_M
        ds_info = dict(key=(cache_receipt.get("key", spec["ds_key"])
                            if smoke else spec["ds_key"]),
                       root=os.path.realpath(root),
                       n_train=n_train, gen_batch_size=gen_bs,
                       rebatch_chunk=chunk, cache_receipt=cache_receipt)
        # per-sector n_params gate (full runs at res 3 only)
        if not smoke:
            n_par = int(engine.count_params(
                engine.ns["build_model"](spec["arch"], spec["label_size"],
                                         spec["res"], True),
                (1, m, m, 1), True))
            pin = D16_SECTOR_N_PARAMS[spec["sector"]]
            assert n_par == pin, (
                "[train_lane] n_params %d != sector pin %d (label_size %d)"
                % (n_par, pin, spec["label_size"]))
            ds_info["n_params_checked"] = n_par
    else:
        if spec["era"] == "d20":
            beta = BETA_THERMAL_D20
            engine.init_d20(spec["h_type"], spec["state_type"], beta=beta,
                            g_init=G_INIT, g_stop=G_STOP)
        else:
            engine.init_d12(spec["h_type"], spec["state_type"],
                            g_init=G_INIT, g_stop=G_STOP)
            beta = float(engine.BETA)
        engine.init_gram()
        if spec.get("use_energy_input") is False:
            build_note = _noe_build_model_patch(engine)
        is_thermal = (spec["state_type"] == "thermal")
        loader, ds_info = _build_datasets_d20(engine, spec, beta,
                                              cache_root, smoke)
        n_train = ds_info["n_train"]
        m = int(engine.M_PAIRS)

    label_size = (spec["label_size"] if is_d16
                  else int(engine.g_gen.label_size()))
    toggles = {k: spec[k] for k in ("use_scatter", "use_reinject",
                                    "use_orb_emb", "readout_bias")
               if k in spec}

    print("[train_lane] %s: era=%s arch=%s res=%d label_size=%d "
          "n_train=%d epochs=%d batch=%d seed=%d loss=%s beta=%g "
          "is_thermal=%s smoke=%s"
          % (lane, spec["era"], spec["arch"], spec["res"], label_size,
             n_train, spec["epochs"], spec["batch_size"],
             spec["init_seed"], spec["loss"], beta, is_thermal, smoke),
          flush=True)

    train_kwargs = dict(
        dataset=loader, loader=loader, train_loader=loader,
        label_size=label_size, input_type=INPUT_TYPE,
        input_shape=(1, m, m, 1), M_PAIRS=m, m_pairs=m, include_energy=True,
        total_samples=n_train, num_samples=n_train,
        batch_size=spec["batch_size"], epochs=spec["epochs"],
        res=spec["res"], arch=spec["arch"],
        loss=spec["loss"], loss_type=spec["loss"],
        init_seed=spec["init_seed"], seed=spec["init_seed"],
        peak_lr=spec.get("peak_lr"), weight_decay=spec.get("weight_decay"),
        clip=spec.get("clip"), is_thermal=is_thermal, beta=beta,
        ckpt_dir=out_dir, resume=bool(args.resume), **toggles)
    state, hist = _call_filtered(engine.train_model, **train_kwargs)
    hist_list = _hist_to_list(hist)
    final_loss = hist_list[-1] if hist_list else None
    if not all(np.isfinite(hist_list)):
        raise SystemExit("[train_lane] NON-FINITE epoch loss %r -- the "
                         "classic signature is the gen_bs gotcha: %s"
                         % (hist_list, GEN_BS_GOTCHA))

    _call_filtered(engine.save_model_and_history, state=state, hist=hist,
                   history=hist, save_dir=out_dir, ckpt_dir=out_dir,
                   out_dir=out_dir)

    wall = time.time() - t0
    config = dict(lane=lane, spec={k: spec[k] for k in sorted(spec)},
                  smoke=smoke, seed_override=args.seed_override,
                  dataset=ds_info, build_model_patch=build_note,
                  hist=hist_list, final_loss=final_loss,
                  epochs_run=len(hist_list), env=receipt, wall_s=wall,
                  produced_by="notebook_release scripts/train_lane.py",
                  equivalence_note=(
                      "retrained weights are hardware/stack-dependent; "
                      "equivalence to the published checkpoint is at the "
                      "forward-pass/score level (see RETRAIN.md), byte "
                      "SHAs WILL differ."))
    if is_d16:
        config["era_install"] = era_block
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1, default=str)

    print("\n[train_lane] DONE %s: epochs=%d final_loss=%s wall=%.1fs"
          % (lane, len(hist_list), final_loss, wall))
    print("[train_lane] output sha256 receipts:")
    for fn in ("final_state.msgpack", "hist.npy", "hist.json",
               "config.json"):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            print("  %s  %s  (%d bytes)"
                  % (_sha256_file(p), os.path.join(lane, fn),
                     os.path.getsize(p)))
    print("[train_lane] wall-time receipt: %.1f s total (train %s epochs; "
          "host %s, backend %s x%s)"
          % (wall, len(hist_list), receipt.get("hostname"),
             receipt.get("backend"), receipt.get("n_local_devices")))
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lane", nargs="?", help="lane name (see --list)")
    ap.add_argument("--smoke", action="store_true",
                    help="512 samples / 1 epoch / res 1 / batch 64 (CPU)")
    ap.add_argument("--cache-dir", default=None,
                    help="dataset cache root (d16 full-scale: REQUIRED, "
                    "the certified blocked caches; d20/d12: where live "
                    "generation caches land; default <out>/ds_cache)")
    ap.add_argument("--out", default="./retrain_out",
                    help="output root; lane outputs land in <out>/<lane>/")
    ap.add_argument("--seed-override", type=int, default=None,
                    help="override the lane's init_seed")
    ap.add_argument("--device-report", action="store_true",
                    help="print the device/env receipt (alone: exit after)")
    ap.add_argument("--std-stats", default=None,
                    help="std lanes: path to std_stats.npz (default: the "
                    "bundle's checkpoints/d16prog/<lane>/std_stats.npz)")
    ap.add_argument("--gram-s", default=None,
                    help="d16 lanes: path to the pinned d16_gram_S_64x64"
                    ".npy artifact (default: bundle data/, then the "
                    "research-repo path)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from an unfinished epoch_state.msgpack "
                    "in <out>/<lane> (default: fresh training)")
    ap.add_argument("--list", action="store_true", help="list lanes")
    # [V2] production-run v2 lanes (ognrepro.v2.registry) -> train_lane_v2
    ap.add_argument("--v2", action="store_true",
                    help="run a v2 lane (ognrepro.v2 behind V2Config; "
                    "see --v2 --list); v1 lanes are untouched")
    import train_lane_v2 as _V2                            # noqa: PLC0415
    _V2.add_args(ap)
    args = ap.parse_args(argv)
    if args.v2:
        if args.list:
            _V2.list_lanes()
            return 0
        if args.lane is None:
            ap.error("--v2 needs a lane (or --v2 --list)")
        return _V2.run_lane_v2(args.lane, args)
    if args.gram_s:
        _GRAM_S_PATH[0] = args.gram_s

    if args.list:
        for name in LANES:
            s = LANES[name]
            print("%-28s era=%-5s h=%-6s state=%-7s arch=%-7s res=%d "
                  "ep=%-2d n=%.0e seed=%d loss=%s"
                  % (name, s["era"], s["h_type"], s["state_type"],
                     s["arch"], s["res"], s["epochs"], s["num_samples"],
                     s["init_seed"], s["loss"]))
        return 0
    if args.lane is None:
        if args.device_report:
            _print_receipt(_env_receipt(), "device/env receipt")
            return 0
        ap.error("lane required (or --list / --device-report)")
    if args.lane not in LANES:
        ap.error("unknown lane %r; see --list" % args.lane)
    return run_lane(args.lane, args)


if __name__ == "__main__":
    sys.exit(main())
