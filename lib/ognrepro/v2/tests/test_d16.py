"""d16 psi0 sidecar + seniority-zero operator tables (CPU, engine-free)."""
import os
import tempfile

import numpy as np
import jax.numpy as jnp

from .. import d16_aux as DA
from .. import ops_index as OI
from .. import losses as L
from ..loader import V2Loader
from ... import blocked as B

M, N = 8, 8
_STATE = {}


def _pinned_S():
    import hashlib  # noqa: PLC0415
    here = os.path.abspath(__file__)
    repo = here
    for _ in range(7):
        repo = os.path.dirname(repo)
    p = os.path.join(repo, "experiments", "experiment-campaign2-r2", "recon", "d16_prototype",
                     "adversarial", "d16_gram_S_64x64.npy")
    S = np.load(p)
    assert hashlib.sha256(np.ascontiguousarray(S).tobytes()).hexdigest().startswith("733219c449e518a8")
    return S


def _mini_cache(htype="vect", n_shards=2):
    """2 shards x 64 rows per split, labels replayed by the pinned chain (vect:
    the certified stream class; const: the disclosed CPU class -- fine here)."""
    key = (htype, n_shards)
    if key in _STATE:
        return _STATE[key]
    root = tempfile.mkdtemp(prefix="v2_d16_%s_" % htype)
    eps = B.ladder(M)
    blocks = B.build_blocks(M, N)
    ls = 1 if htype == "const" else 3
    seeds = {"half0": 42, "half1": 43, "val": 1007}
    for split, seed in seeds.items():
        labels = np.asarray(B.replay_labels(seed, n_shards, gen_bs=64, ndev=4, label_size=ls, htype=htype,
                                            lo=0, hi=n_shards), np.float32)
        d = os.path.join(root, split)
        os.makedirs(d)
        for i in range(n_shards):
            lab = labels[i * 64:(i + 1) * 64]
            V = np.stack([B.v_matrix(htype, M, r) for r in lab])
            f2, en, _z, _e0, nst, _sg = B.blocked_solve(blocks, eps, V, M, beta=100.0)
            assert nst == 12870
            B.save_shard(os.path.join(d, B.shard_name(i)), lab, en.astype(np.float32)[:, None],
                         f2.astype(np.float32)[..., None])
    _STATE[key] = root
    return root


def test_d16_tables_match_block_hops(ctx):
    S = _pinned_S()
    ops = OI.build_d16_s0(S)
    assert (ops["D"], ops["A"], ops["m"]) == (70, 64, 8)
    blk = B.FreeSetBlock(M, N, ())
    # every (y, x, a_rem, b_add) hop of the block appears once: h_{a} for a = i*m + j removes i, adds j
    src = ops["src_pad"].astype(np.int64)
    for y, x, a_rem, b_add in zip(blk.ty, blk.tx, blk.ta, blk.tb):
        assert src[a_rem * M + b_add, y] == x
    n_hops = int((src < 70).sum()) - sum(len(c) for c in blk.cfgs)     # minus the number operators
    assert n_hops == len(blk.tx)
    # number operators: src[kk, x] == x iff k in cfg x
    for x, c in enumerate(blk.cfgs):
        for k in range(M):
            assert (src[k * M + k, x] == x) == (k in c)
    # u and S_inf
    assert np.isclose(ops["u"][0], 3003 / 12870) and ops["u"][1] == 0.0
    w = np.linalg.eigvalsh(ops["S_inf"])
    assert w.min() > -1e-12
    assert np.allclose(ops["pair_energies"], 2.0 * B.ladder(M))


def test_d16_psi0_sidecar_and_variance(ctx):
    root = _mini_cache("vect")
    cert = DA.build_psi0_cache(root, "vect", {"half0": 2, "half1": 2, "val": 2}, nproc=2, double=True,
                               verbose=False, ds_key="mini_vect")
    assert cert["run_b_equal"] and os.path.isfile(os.path.join(root, "psi0_certificate.json"))
    for split in ("half0", "half1", "val"):
        d = cert["per_split"][split]
        assert d["n_rows"] == 128 and d["e0_dev_max"] < DA.E0_TOL, d
        assert not os.path.isdir(os.path.join(root, split, ".psi0_runB"))
    rec = DA.verify_psi0(root, {"half0": 2, "half1": 2, "val": 2}, verbose=False)
    assert rec["matches_certificate"]
    # the shards are untouched (psi0 lives beside them)
    with np.load(os.path.join(root, "half0", B.shard_name(0))) as z:
        assert set(z.files) == {"labels", "energy", "features"}
    psi = np.load(os.path.join(root, "half0", "shard_00000_psi0.npy"))
    assert psi.shape == (64, 70) and psi.dtype == np.float32
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0, atol=1e-5)
    amax = np.argmax(np.abs(psi), axis=1)
    assert np.all(psi[np.arange(64), amax] > 0)
    # ground-state property: H psi = e0 psi in the block, and <psi|N_pairs|psi> = 4
    ops = OI.build_d16_s0(_pinned_S())
    with np.load(os.path.join(root, "half0", B.shard_name(0))) as z:
        lab = np.asarray(z["labels"]); E = np.asarray(z["energy"]).reshape(-1)
    blk = B.FreeSetBlock(M, N, ())
    V = np.stack([B.v_matrix("vect", M, r) for r in lab])
    H = blk.h_batch(B.ladder(M), V, 0.0)
    psi64, e0, _gap = DA.psi0_rows(lab, "vect", blk, B.ladder(M))
    res = np.linalg.norm(np.einsum("bij,bj->bi", H, psi64) - e0[:, None] * psi64, axis=1)
    assert res.max() < 1e-10 and np.abs(E - e0).max() < DA.E0_TOL
    # variance in the ground state: gather path (device f32) vs dense reference, symmetric dw
    rng = np.random.default_rng(0)
    dG = rng.standard_normal((4, M, M)); dG = dG + np.swapaxes(dG, 1, 2)
    dw = dG.reshape(4, -1)
    ref = np.array([OI.cov_gs_numpy(ops, psi64[b], dw[b]) for b in range(4)])
    got = np.asarray(L.var_gs(jnp.asarray(dw, jnp.float32), jnp.asarray(psi64[:4], jnp.float32),
                              jnp.asarray(ops["src_pad"]), L.HI))
    assert np.abs(got - ref).max() < 1e-4 * max(ref.max(), 1e-9), (got, ref)
    # identity shifts are invisible to the GS variance (N_pairs = 4 in the block)
    dI = np.tile(np.eye(M).reshape(1, -1), (4, 1))
    assert np.abs([OI.cov_gs_numpy(ops, psi64[b], dI[b]) for b in range(4)]).max() < 1e-10
    # loader fallback: be = [E | psi0] from the sibling sidecar
    ld = V2Loader([os.path.join(root, "half0")], batch_size=64, shuffle=False, aux="psi0")
    bx, be, by = next(iter(ld))
    assert be.shape == (64, 71) and bx.shape == (64, 8, 8, 1) and by.shape == (64, 3)
    assert np.array_equal(be[:, 1:], psi)


def test_d16_e0_gate_refuses_wrong_state(ctx):
    """A shard whose energy is not the v=0 ground energy is refused."""
    root = _mini_cache("vect")
    bad = tempfile.mkdtemp(prefix="v2_d16_bad_")
    src = os.path.join(root, "half1", B.shard_name(0))
    with np.load(src) as z:
        lab, en, feat = z["labels"], z["energy"], z["features"]
    B.save_shard(os.path.join(bad, B.shard_name(0)), lab, en + 1e-3, feat)
    try:
        DA.build_psi0_range(bad, 0, 1, "vect")
        raise AssertionError("e0 gate did not fire")
    except RuntimeError as exc:
        assert "|E - e0|" in str(exc)
