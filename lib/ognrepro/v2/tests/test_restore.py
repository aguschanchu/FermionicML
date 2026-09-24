"""restore_v2 round trip: a v2 lane directory written the way train_lane_v2
writes it (config.json model_build block + std_stats.npz + final_state.msgpack)
restores to the identical parameter tree with the identical class."""
import hashlib
import json
import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization as flax_ser

from ..config import V2Config, resolve
from .. import model as MDL
from .. import stats as ST
from ... import restore as RS


def _lane_dir(ctx, arch, cfg_over, std_stats, mlp_width=None, mlp_blocks=8, label_size=55):
    ops = ctx["ops"]
    cfg = resolve(V2Config(**cfg_over), state_type="thermal", label_size=label_size, m=10, arch=arch)
    mdl = MDL.build_v2_model(arch, label_size, 1, True, cfg, ops, std_stats=std_stats, n_elec=10,
                             mlp_width=mlp_width, mlp_blocks=mlp_blocks)
    v = mdl.init(jax.random.PRNGKey(3), jnp.zeros((1, 10, 10, 1), jnp.float32), jnp.zeros((1, 1), jnp.float32),
                 training=False)
    d = tempfile.mkdtemp(prefix="v2_restore_")
    raw = flax_ser.msgpack_serialize({"params": v["params"], "batch_stats": v.get("batch_stats", {})})
    with open(os.path.join(d, "final_state.msgpack"), "wb") as fh:
        fh.write(raw)
    std_info = None
    if cfg.std_convention:
        ST.save_std_stats(os.path.join(d, "std_stats.npz"), cfg.std_convention, std_stats[0], std_stats[1], 256, "test")
        std_info = dict(kind=cfg.std_convention, mean_sha256=hashlib.sha256(np.ascontiguousarray(std_stats[0]).tobytes()).hexdigest())
    n_par = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(v["params"])))
    config = dict(cfg_version=2, lane="test_" + arch, spec=dict(era="d20"), v2config=cfg.to_json(),
                  model_build=MDL.model_build_record(mdl, arch, cfg), n_params=n_par, std_stats=std_info,
                  final_state_sha256=hashlib.sha256(raw).hexdigest())
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(config, fh, default=str)
    return d, mdl, v, n_par


def _trees_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_restore_v2_round_trip(ctx):
    rng = np.random.default_rng(0)
    mu = rng.standard_normal((10, 10)).astype(np.float32); mu = 0.5 * (mu + mu.T)
    sg = (0.5 + rng.random((10, 10))).astype(np.float32); sg = 0.5 * (sg + sg.T)
    m55, s55 = rng.standard_normal(55).astype(np.float32), (0.5 + rng.random(55)).astype(np.float32)
    cases = [("ogn_std", {}, (mu, sg), None), ("ogn", {}, None, None),
             ("mlp", {}, None, None), ("mlp_std", {}, (m55, s55), 32)]
    for arch, over, stds, width in cases:
        d, mdl, v, n_par = _lane_dir(ctx, arch, over, stds, mlp_width=width, mlp_blocks=2)
        r = RS.restore_v2(d, cast_f64=False)
        assert type(r["model"]).__name__ == type(mdl).__name__, arch
        assert r["n_params"] == n_par
        assert _trees_equal(r["params"], v["params"]), arch
        # same forward on a random input
        x = jnp.asarray(rng.standard_normal((2, 10, 10, 1)).astype(np.float32)); e = jnp.asarray(rng.standard_normal((2, 1)).astype(np.float32))
        o1 = mdl.apply(v, x, e, training=False)
        o2 = r["model"].apply({"params": r["params"], "batch_stats": r["batch_stats"]}, x, e, training=False)
        assert np.array_equal(np.asarray(o1), np.asarray(o2)), arch
        if arch == "ogn_std":
            assert r["model"].proj_gauge and r["model"].proj_shell == "euclid" and len(r["model"].pair_energies) == 10
        # a wrong pin is refused
        try:
            RS.restore_v2(d, expect_sha="0" * 64, cast_f64=False)
            raise AssertionError("pin mismatch not refused")
        except ValueError:
            pass
        # a tampered std_stats file is refused
        if stds is not None:
            p = os.path.join(d, "std_stats.npz")
            with np.load(p) as z:
                dd = {k: z[k] for k in z.files}
            key = "mu" if "mu" in dd else "mean"
            dd[key] = dd[key] + 1.0
            np.savez(p, **dd)
            try:
                RS.restore_v2(d, cast_f64=False)
                raise AssertionError("tampered std_stats not refused")
            except ValueError:
                pass
