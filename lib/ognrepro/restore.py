"""ognrepro.restore -- CPU-f64 checkpoint restore + forward (sealed-record
conventions), plus the d20 published-checkpoint registry.

Every function here is meant to run under repro_common.enable_eval_convention()
(JAX_PLATFORMS=cpu + JAX_ENABLE_X64=1 BEFORE the first jax import, matmul
precision 'highest') -- the regenerated-exact/CPU-f64 primary convention of
the sealed D16PROG-P4 record.

PROVENANCE
    Source: experiments/experiment-campaign2-r2/results/D16PROG-P4-SCORES/
            expc2r2_d16prog_p4_eval_as_run.py  (sealed READ-ONLY record;
            copied out, the sealed tree is not touched)
    Source file sha256:
            0a9edddc60696917e3d6bc61d4745410ea7139a856c0a55a750aa080f3453c59
    Line ranges extracted:
        1090-1097  init_template          (VERBATIM)
        1100-1120  restore_params_f64     (VERBATIM)
        1123-1150  forward_f64            (VERBATIM)
        1153-1174  build_arm_model        (ADAPTED, see diffs)
    Additional sources:
        campaign4/engine_parts/p3_training.py lines 541-604 load_model
            (sha256 8e2a1a70013342c2c12719e809ca0fbd09cfb755dadb3323c7877b48f4c904b6)
            -- the template-TrainState restore path restore_d20 mirrors
            (params-only: msgpack_restore + from_state_dict, per the sealed
            P4 convention; opt_state deliberately NOT restored)
        campaign4/config.py MODELS lines 126-171
            (sha256 1e15971b36cc442a780c3461582f3a742279c300f935939ae0b997503cd4d63f)
            + campaign5/train/config5.py "ogn_noE" lines 157-160
            (sha256 23d472073edc9e47c1c02d3d3f0281fe16558a73f113870b1b587ebd3a2cd2e4)
            -- the arch/res/toggle registry for the 14 bundled d20-era
            checkpoints, transcribed into D20_MODELS below (label sizes from
            the era + h_type via p1_core.GGenerator.label_size:470-482:
            random -> m(m+1)/2, const -> 1, vect -> ceil(m/2)-1).
    Extraction date: 2026-08-23.
    Tag: init_template / restore_params_f64 / forward_f64 VERBATIM (only the
    as-run's PLC0415 noqa comments and the `flax_ser` alias are kept as-is).
    build_arm_model ADAPTED with EXACTLY these diffs:
      * the std arm builds ognrepro.std_arch.StdPhysicsOrbitalGraphNet
        directly (the as-run built the same class via its lanes module L.STD;
        construction call and f32-tuple stats conversion identical);
      * the orig arm builds via the vendored engine
        (repro_common.get_engine().build_model('ogn', label_size, 3, True)),
        exactly the as-run's L.engine.build_model call;
      * signature (is_std, label_size, mu, sigma) instead of
        (L, entry, label_size, mu, sigma) -- no lanes module here.
    ADAPTED ADDITIONS: D20_MODELS registry, restore_d20(), free().
"""
import gc
import os

import numpy as np

from . import repro_common
from . import std_arch
from . import arch_variants


# ======================================================================
# restore + f64 forward (the CPU-f64 primary convention)
# ======================================================================
def init_template(model, m):
    """PRNGKey(0) init template (single channel; the OGN signature
    (x, energy, training) -- expc2r2_eval_mirror.init_template semantics)."""
    import jax                                           # noqa: PLC0415
    import jax.numpy as jnp                              # noqa: PLC0415
    return model.init(jax.random.PRNGKey(0),
                      jnp.zeros((1, m, m, 1), jnp.float32),
                      jnp.zeros((1, 1), jnp.float32), training=False)


def restore_params_f64(model, raw_bytes, m):
    """Init template + restore from ALREADY-HASHED bytes, then cast the
    restored tree to f64 (the expc2r2_eval_cpu64 f64-params convention).
    Returns (params64, bstats64, n_params)."""
    import jax                                           # noqa: PLC0415
    import jax.numpy as jnp                              # noqa: PLC0415
    from flax import serialization as flax_ser           # noqa: PLC0415
    variables = init_template(model, m)
    raw = flax_ser.msgpack_restore(raw_bytes)
    raw_params = raw.get("params", raw) if isinstance(raw, dict) else raw
    params = flax_ser.from_state_dict(variables["params"], raw_params)
    bs_t = variables.get("batch_stats", {})
    raw_bs = raw.get("batch_stats", {}) if isinstance(raw, dict) else {}
    bstats = flax_ser.from_state_dict(bs_t, raw_bs) if raw_bs else bs_t
    n_par = int(sum(np.asarray(x).size
                    for x in jax.tree_util.tree_leaves(params)))
    p64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64),
                                 params)
    b64 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64),
                                 bstats)
    return p64, b64, n_par


def forward_f64(model, params64, bstats64, X, E, chunk=256):
    """f64 forward under default_matmul_precision('highest') baked at trace
    time; returns (preds f64 (n,label_size), deterministic bool)."""
    import jax                                           # noqa: PLC0415
    import jax.numpy as jnp                              # noqa: PLC0415
    apply_fn = model.apply

    @jax.jit
    def _ap(bx, be):
        return apply_fn({"params": params64, "batch_stats": bstats64},
                        bx, be, training=False)

    outs = []
    with jax.default_matmul_precision("highest"):
        for s in range(0, len(X), chunk):
            e = min(s + chunk, len(X))
            o = np.asarray(_ap(jnp.asarray(X[s:e], jnp.float64),
                               jnp.asarray(E[s:e], jnp.float64)))
            if o.ndim == 3:
                o = o.reshape(-1, o.shape[-1])
            outs.append(o)
        k = min(chunk, len(X))
        o0 = np.asarray(_ap(jnp.asarray(X[:k], jnp.float64),
                            jnp.asarray(E[:k], jnp.float64)))
        if o0.ndim == 3:
            o0 = o0.reshape(-1, o0.shape[-1])
    det = bool(np.array_equal(o0, outs[0]))
    return np.concatenate(outs, 0).astype(np.float64), det


def build_arm_model(is_std, label_size, mu=None, sigma=None):
    """Training-identical construction of a d16prog lane arm.  std: the
    t_stdd12 class DIRECTLY with per-arm hashable stats (BATCH SAFETY -- the
    ambient one-lane-per-process machinery is never armed); orig: the plain
    campaign6 factory call build_model('ogn', label_size, 3, True) via the
    vendored engine.  [ADAPTED from as-run:1153-1174 -- see header]"""
    if is_std:
        if mu is None or sigma is None:
            raise ValueError("std arm needs mu/sigma")
        mu_t, sg_t = std_arch.std_stats_tuples(mu, sigma)
        mdl = std_arch.StdPhysicsOrbitalGraphNet(
            label_size=int(label_size), res=3, include_energy=True,
            use_scatter=True, use_reinject=True, use_orb_emb=True,
            readout_bias=0.55, use_energy_input=True,
            std_mu=mu_t, std_sigma=sg_t)
        return mdl, "StdPhysicsOrbitalGraphNet(direct, per-arm stats)"
    engine = repro_common.get_engine()
    mdl = engine.build_model("ogn", int(label_size), 3, True)
    return mdl, "PhysicsOrbitalGraphNet(engine.build_model 'ogn')"


# ======================================================================
# d20-era published checkpoint registry (the 14 bundled names)
# ======================================================================
# [ADAPTED transcription -- campaign4/config.py MODELS:126-171 (+ config5.py
#  ogn_noE:157-160).  m: d20 -> M_PAIRS 10, d12 -> M_PAIRS 6 (config.py:58-62).
#  label_size from h_type at era m (p1_core.py:470-482): random -> m(m+1)/2,
#  const -> 1, vect -> ceil(m/2)-1.  include_energy=True at every campaign
#  call site (campaign4 tasks).  Toggles exactly the MODELS spec entries.]
D20_MODELS = {
    "prod_thermal_random":       dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "prod_gs_random":            dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "prod_gs_const":             dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="const", label_size=1),
    "nearinit_random":           dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "d12_const":                 dict(era="d12", m=6, arch="ogn", res=3,
                                      h_type="const", label_size=1),
    "d12_vect":                  dict(era="d12", m=6, arch="ogn", res=3,
                                      h_type="vect", label_size=2),
    "abl_mlp":                   dict(era="d20", m=10, arch="mlp", res=4,
                                      h_type="random", label_size=55),
    "abl_noscatter":             dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55,
                                      use_scatter=False),
    "abl_noreinject":            dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55,
                                      use_reinject=False),
    "abl_neutralbias":           dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55,
                                      readout_bias=0.0),
    "gram_thermal_random_50":    dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "rdm_thermal_random_50":     dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "rdm_thermal_random_50_s43": dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55),
    "ogn_noE":                   dict(era="d20", m=10, arch="ogn", res=3,
                                      h_type="random", label_size=55,
                                      use_energy_input=False),
}


def restore_d20(name, checkpoints_dir):
    """Restore one of the 14 bundled published checkpoints (params-only).

    Mirrors campaign4 engine_parts/p3_training.py:541 load_model's template
    path (build model from the registry spec, init dummy template, map the
    checkpoint bytes) but restores params + batch_stats ONLY via
    msgpack_restore + from_state_dict (restore_params_f64 above) -- no dummy
    optimizer, no opt_state: lighter, and the eval convention casts to f64.

    'ogn' archs build through ognrepro.arch_variants (the campaign5 copy
    whose defaults reproduce the production module bit-for-bit and whose
    use_energy_input=False restores ogn_noE); non-'ogn' archs (abl_mlp)
    build through the vendored engine factory.

    Returns dict(model, params, batch_stats, n_params, m, spec, ckpt_path).
    """
    if name not in D20_MODELS:
        raise KeyError("unknown bundled checkpoint %r (have %s)"
                       % (name, sorted(D20_MODELS)))
    spec = dict(D20_MODELS[name])
    m = int(spec["m"])
    if spec["arch"] == "ogn":
        model = arch_variants.build_ogn(
            label_size=int(spec["label_size"]), res=int(spec["res"]),
            include_energy=True,
            use_scatter=spec.get("use_scatter", True),
            use_reinject=spec.get("use_reinject", True),
            use_orb_emb=spec.get("use_orb_emb", True),
            readout_bias=spec.get("readout_bias", 0.55),
            use_energy_input=spec.get("use_energy_input", True))
    else:
        engine = repro_common.get_engine()
        # The engine model classes read the era global N_ELEC late-bound
        # from the exec'd namespace (p2_models.py:73/:163, "late-bound,
        # p1_core") -- normally installed by init_d20/init_d12, which a
        # restore-only path never runs.  At half filling N_ELEC == M_PAIRS
        # == the registry's m (d20: 10, d12: 6).  setdefault so a live
        # engine init is never clobbered.
        engine.ns.setdefault("N_ELEC", m)
        model = engine.build_model(spec["arch"], int(spec["label_size"]),
                                   int(spec["res"]), True)
    ckpt_path = os.path.join(checkpoints_dir, name, "final_state.msgpack")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("restore_d20: %s not found" % ckpt_path)
    with open(ckpt_path, "rb") as fh:
        raw = fh.read()
    p64, b64, n_par = restore_params_f64(model, raw, m)
    return dict(model=model, params=p64, batch_stats=b64,
                n_params=int(n_par), m=m, spec=spec, ckpt_path=ckpt_path)


def free(*objs):
    """Drop references and collect -- restored d20 trees are >1 GB as f64;
    call between successive restores in one process.  NOTE: the caller must
    also drop its OWN names (del x / x = None) -- this helper releases the
    argument references it holds and runs gc.collect()."""
    objs = None
    gc.collect()


# ======================================================================
# [V2] production-run v2 checkpoints (train_lane_v2 config.json schema)
# ======================================================================
# Final-state sha256 pins of the sealed v2 checkpoints, transcribed from
# results_v2/registry/V2-CHECKPOINTS.json once a lane is sealed (empty until
# then; restore_v2 verifies a pin whenever one is present for the lane).
V2_PINS = {}


def build_from_config(cfg_json, std_stats=None):
    """Rebuild the v2 network from a lane's config.json 'model_build' block
    (written by ognrepro.v2.model.model_build_record) -- no engine, no
    registry lookups: the class name + its flax fields are the whole spec.
    std_stats: (mu, sigma) for StdPhysicsOrbitalGraphNet, (mean, std) for the
    standardized MLP; the 'std_convention' of the block says which."""
    from . import mlp_variants                           # noqa: PLC0415
    mb = cfg_json["model_build"] if "model_build" in cfg_json else cfg_json
    cls, f = mb["cls"], dict(mb["fields"])
    if "pair_energies" in f:
        f["pair_energies"] = tuple(float(v) for v in f["pair_energies"])
    if cls == "StdPhysicsOrbitalGraphNet":
        if std_stats is None:
            raise ValueError("std class needs (mu, sigma)")
        mu_t, sg_t = std_arch.std_stats_tuples(std_stats[0], std_stats[1])
        return std_arch.StdPhysicsOrbitalGraphNet(std_mu=mu_t, std_sigma=sg_t, **f)
    if cls == "PhysicsOrbitalGraphNet":
        return arch_variants.PhysicsOrbitalGraphNet(**f)
    if cls == "DeepResMLPV2":
        if mb.get("std_convention") == "triu55":
            if std_stats is None:
                raise ValueError("standardized MLP needs (mean, std)")
            f["std_mean"] = tuple(float(v) for v in np.asarray(std_stats[0]).reshape(-1))
            f["std_std"] = tuple(float(v) for v in np.asarray(std_stats[1]).reshape(-1))
        return mlp_variants.DeepResMLPV2(**f)
    raise ValueError("build_from_config: unknown class %r" % (cls,))


def restore_v2(ckpt_dir, expect_sha=None, cast_f64=True):
    """Restore a v2 lane directory (config.json + final_state.msgpack +
    std_stats.npz) under the eval convention.  Returns dict(model, params,
    batch_stats, n_params, m, cfg, config, ckpt_path, final_state_sha256).
    The final_state sha is checked against `expect_sha`, else V2_PINS[lane]
    when transcribed, and always against config.json's own record."""
    import hashlib                                       # noqa: PLC0415
    import json                                          # noqa: PLC0415
    from .v2.config import V2Config                      # noqa: PLC0415
    from .v2 import stats as v2stats                     # noqa: PLC0415
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as fh:
        config = json.load(fh)
    if int(config.get("cfg_version", 0)) != 2:
        raise ValueError("restore_v2: %s is not a v2 lane config (cfg_version %r)"
                         % (cfg_path, config.get("cfg_version")))
    cfg = V2Config.from_json(config["v2config"])
    std_stats = None
    if config["model_build"].get("std_convention"):
        kind, a, b, _n = v2stats.load_std_stats(os.path.join(ckpt_dir, "std_stats.npz"))
        if kind != config["model_build"]["std_convention"]:
            raise ValueError("restore_v2: std_stats kind %r != config %r"
                             % (kind, config["model_build"]["std_convention"]))
        want = config.get("std_stats", {}).get("mean_sha256")
        got = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        if want and want != got:
            raise ValueError("restore_v2: std_stats.npz sha %s != config %s" % (got[:16], want[:16]))
        std_stats = (a, b)
    model = build_from_config(config, std_stats)
    ckpt_path = os.path.join(ckpt_dir, "final_state.msgpack")
    with open(ckpt_path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    lane = config.get("lane")
    pin = expect_sha or V2_PINS.get(lane)
    if pin and sha != pin:
        raise ValueError("restore_v2: %s final_state sha %s != pin %s" % (lane, sha[:16], pin[:16]))
    rec_sha = config.get("final_state_sha256")
    if rec_sha and rec_sha != sha:
        raise ValueError("restore_v2: final_state sha %s != config.json record %s" % (sha[:16], rec_sha[:16]))
    m = {"d20": 10, "d16n8": 8, "d12": 6}[config["spec"]["era"]]
    if cast_f64:
        p, b, n_par = restore_params_f64(model, raw, m)
    else:
        import jax                                       # noqa: PLC0415
        from flax import serialization as flax_ser       # noqa: PLC0415
        variables = init_template(model, m)
        rawt = flax_ser.msgpack_restore(raw)
        raw_params = rawt.get("params", rawt) if isinstance(rawt, dict) else rawt
        p = flax_ser.from_state_dict(variables["params"], raw_params)
        bs_t = variables.get("batch_stats", {})
        raw_bs = rawt.get("batch_stats", {}) if isinstance(rawt, dict) else {}
        b = flax_ser.from_state_dict(bs_t, raw_bs) if raw_bs else bs_t
        n_par = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(p)))
    if config.get("n_params") is not None and int(config["n_params"]) != int(n_par):
        raise ValueError("restore_v2: n_params %d != config %d" % (n_par, int(config["n_params"])))
    return dict(model=model, params=p, batch_stats=b, n_params=int(n_par), m=m, cfg=cfg,
                config=config, ckpt_path=ckpt_path, final_state_sha256=sha, lane=lane)
