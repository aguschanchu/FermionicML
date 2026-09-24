"""v2 generator / loader / statistics tests (CPU)."""
import os
import shutil
import tempfile

import numpy as np

from .. import gen as GEN
from .. import ops_index as OI
from ..config import V2Config, V1_PUBLISHED, resolve
from ..loader import V2Loader
from .. import stats as ST


def _scratch(name):
    root = os.path.join(tempfile.gettempdir(), "v2_tests", name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return root


def _cfg(**kw):
    base = dict(data_precision="highest", aux=None, readout="published", shell="off",
                loss="gram", w_gauge=1.0)
    base.update(kw)
    return V2Config(**base)


def _f64_reference(ctx, labels, beta_eff):
    import jax.numpy as jnp
    engine = ctx["engine"]
    G = np.asarray(engine.g_gen.reconstruct(jnp.asarray(labels)), np.float64)
    B = len(G)
    e_params = np.tile(np.asarray(engine.U_ENERGY_SEED[0], np.float64), (B, 1))
    if e_params.shape[-1] == engine.d_rho_1_diag_global.shape[0] * 2:
        e_params = e_params.reshape(B, -1, 2).sum(-1)
    rho_1_diag = np.asarray(engine.d_rho_1_diag_global, np.float64)
    inter = np.asarray(engine.d_inter_tensor_global, np.float64)
    H = -np.einsum("bij,jirc->brc", G, inter)
    h0 = e_params @ rho_1_diag
    idx = np.arange(H.shape[-1])
    H[:, idx, idx] += h0
    H = 0.5 * (H + np.swapaxes(H, 1, 2))
    vals, vecs = np.linalg.eigh(H)
    p = np.exp(-beta_eff * (vals - vals.min(1, keepdims=True)))
    p /= p.sum(1, keepdims=True)
    rho = np.einsum("bnk,bk,bmk->bnm", vecs, p, vecs)
    dense = np.asarray(engine.rho_2_kkbar_arrays.todense(), np.float64)
    f2 = np.einsum("bnm,klmn->bkl", rho, dense, optimize=True)
    E = (p * vals).sum(1)
    return f2, E, rho, vals, vecs


def test_labels_identical_to_engine_generator(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("labels")
    ref = os.path.join(root, "engine")
    engine.gen_dataset("random", 0.1, 1.0, "thermal", "rho2kkbar", True, 1.0,
                       num_samples=128, cache_path=ref, batch_size=64, seed=42)
    for prec in ("default", "highest"):
        ours = os.path.join(root, "v2_" + prec)
        GEN.gen_dataset_v2(engine, ops, ours, h_type="random", state_type="thermal", beta=1.0,
                           num_samples=128, gen_bs=64, seed=42, cfg=_cfg(data_precision=prec),
                           verbose=False)
        for i in range(2):
            with np.load(os.path.join(ref, "shard_%05d.npz" % i)) as a, \
                    np.load(GEN.shard_path(ours, i)) as b:
                assert np.array_equal(a["labels"], b["labels"]), (prec, i)
                assert np.allclose(a["features"], b["features"], atol=2e-6), (prec, i)
                assert np.allclose(a["energy"], b["energy"], atol=2e-5), (prec, i)
        sc = GEN.read_schema(ours)
        assert sc["schema_version"] == 2 and sc["matmul_precision"] == prec and sc["complete"]
        assert sc["labels_digest_first2"] == GEN.labels_digest(ours)
    # the engine's own loader reads a v2 cache unchanged
    n = sum(len(by) for _bx, _be, by in engine.NumpyLoader(ours, 64, shuffle=False))
    assert n == 128


def test_highest_matches_f64_reference(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("f64ref")
    GEN.gen_dataset_v2(engine, ops, root, h_type="random", state_type="thermal", beta=1.0,
                       num_samples=64, gen_bs=64, seed=3, cfg=_cfg(), verbose=False)
    with np.load(GEN.shard_path(root, 0)) as z:
        labels, f2, E = z["labels"], z["features"][..., 0], z["energy"][:, 0]
    f2r, Er, _rho, _v, _w = _f64_reference(ctx, labels, 1.0)
    d = np.abs(f2.astype(np.float64) - f2r)
    assert d.max() < 2e-6 and np.median(d) < 2e-7, (d.max(), np.median(d))
    assert np.abs(E.astype(np.float64) - Er).max() < 2e-5


def test_thermal_M_sidecar_host_and_device(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    cfg = _cfg(aux="M", loss="metric", m_dtype="float32")
    for tag, force in (("host", False), ("device", True)):
        root = _scratch("thermalM_" + tag)
        GEN.gen_dataset_v2(engine, ops, root, h_type="random", state_type="thermal", beta=1.0,
                           num_samples=64, gen_bs=64, seed=7, cfg=cfg, verbose=False,
                           force_device_m=force)
        with np.load(GEN.shard_path(root, 0)) as z:
            labels = z["labels"]; f2 = z["features"][..., 0]
        Mp = np.load(GEN.sidecar_path(root, 0))
        assert Mp.shape == (64, 5050) and Mp.dtype == np.float32
        _f, _E, rho, _v, _w = _f64_reference(ctx, labels, 1.0)
        iu = np.triu_indices(ops["A"])
        for r in (0, 17, 63):
            M, hbar = OI.cov_thermal_numpy(ops, rho[r])
            scale = np.abs(M).max()
            assert np.abs(Mp[r].astype(np.float64) - M[iu]).max() < 2e-4 * scale, (tag, r)
            assert np.allclose(hbar, f2[r].reshape(-1), atol=2e-6)
        cert = GEN.certify_cache(root)
        assert cert["n_rows"] == 64 and cert["aux_sha256"] is not None
    # float16 storage: relative error of the packed covariance <= 1e-3
    root16 = _scratch("thermalM_f16")
    GEN.gen_dataset_v2(engine, ops, root16, h_type="random", state_type="thermal", beta=1.0,
                       num_samples=64, gen_bs=64, seed=7, cfg=_cfg(aux="M", loss="metric"),
                       verbose=False)
    M16 = np.load(GEN.sidecar_path(root16, 0))
    assert M16.dtype == np.float16
    ref = np.load(GEN.sidecar_path(_scratch.__globals__["os"].path.join(
        tempfile.gettempdir(), "v2_tests", "thermalM_host"), 0)).astype(np.float64)
    assert np.abs(M16.astype(np.float64) - ref).max() < 1e-3 * np.abs(ref).max()


def test_gs_psi0_sidecar(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("gs_psi0")
    GEN.gen_dataset_v2(engine, ops, root, h_type="random", state_type="gs", beta=1.0,
                       num_samples=64, gen_bs=64, seed=5, cfg=_cfg(aux="psi0", loss="metric"),
                       verbose=False)
    with np.load(GEN.shard_path(root, 0)) as z:
        labels, psi = z["labels"], z["psi0"]
    assert psi.shape == (64, ops["D"])
    _f, _E, _rho, vals, vecs = _f64_reference(ctx, labels, 100.0)
    for r in (0, 9, 63):
        v = vecs[r, :, 0]
        v = v * np.sign(v[np.argmax(np.abs(v))])
        assert min(np.abs(psi[r] - v).max(), np.abs(psi[r] + v).max()) < 1e-3
        assert np.abs(psi[r] - v).max() < 1e-3, "sign rule"
    sc = GEN.read_schema(root)
    assert sc["aux"] == "psi0" and sc["aux_width"] == ops["D"] and sc["effective_beta"] == 100.0


def test_cache_complete_negatives_and_foreign_dir(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("complete")
    cfg = _cfg()
    GEN.gen_dataset_v2(engine, ops, root, h_type="random", state_type="thermal", beta=1.0,
                       num_samples=64, gen_bs=64, seed=11, cfg=cfg, verbose=False)
    assert GEN.cache_complete(root, 64, 64, cfg, h_type="random", state_type="thermal", beta=1.0, seed=11)
    assert not GEN.cache_complete(root, 64, 64, _cfg(aux="M", loss="metric"))
    assert not GEN.cache_complete(root, 64, 64, _cfg(data_precision="default"))
    assert not GEN.cache_complete(root, 64, 32, cfg)
    assert not GEN.cache_complete(root, 128, 64, cfg)
    assert not GEN.cache_complete(root, 64, 64, cfg, seed=12)
    assert not GEN.cache_complete(root, 64, 64, cfg, state_type="gs")
    # a foreign dir (shards, no schema) is refused, never overwritten
    foreign = _scratch("foreign")
    engine.gen_dataset("random", 0.1, 1.0, "thermal", "rho2kkbar", True, 1.0,
                       num_samples=64, cache_path=foreign, batch_size=64, seed=42)
    assert not GEN.cache_complete(foreign, 64, 64, cfg)
    try:
        GEN.gen_dataset_v2(engine, ops, foreign, h_type="random", state_type="thermal", beta=1.0,
                           num_samples=64, gen_bs=64, seed=42, cfg=cfg, verbose=False)
        raise AssertionError("foreign cache overwritten")
    except RuntimeError:
        pass
    assert os.path.isfile(os.path.join(foreign, "shard_00000.npz"))
    # key carries precision + aux
    assert GEN.dataset_key("d20", "random", "thermal", _cfg(aux="M", loss="metric")) == \
        "dsv2_d20_random_thermal__phi_aM_f16"
    assert GEN.dataset_key("d20", "random", "gs", _cfg(aux="psi0", loss="metric"), tag="prod") == \
        "dsv2_d20_random_gs__phi_apsi0_prod"
    assert GEN.dataset_key("d20", "random", "thermal", V1_PUBLISHED) == "dsv2_d20_random_thermal__pdef_anone"


def test_loader_packing_and_order(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("loader")
    cfg = _cfg(aux="M", loss="metric", m_dtype="float32")
    GEN.gen_dataset_v2(engine, ops, root, h_type="random", state_type="thermal", beta=1.0,
                       num_samples=128, gen_bs=64, seed=21, cfg=cfg, verbose=False)
    # aux packing
    np.random.seed(0)
    bx, be, by = next(iter(V2Loader(root, 32, shuffle=True, aux="M")))
    assert bx.shape == (32, 10, 10, 1) and be.shape == (32, 1 + 5050) and by.shape == (32, 55)
    # determinism: same global seed -> same stream
    np.random.seed(5); s1 = [by.copy() for _bx, _be, by in V2Loader(root, 32, shuffle=True, aux="M")]
    np.random.seed(5); s2 = [by.copy() for _bx, _be, by in V2Loader(root, 32, shuffle=True, aux="M")]
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2)) and len(s1) == 4
    # aux=None: identical order to the engine's NumpyLoader under the same RNG state
    np.random.seed(9); e1 = [by.copy() for _bx, _be, by in engine.NumpyLoader(root, 32, shuffle=True)]
    np.random.seed(9); e2 = [by.copy() for _bx, _be, by in V2Loader(root, 32, shuffle=True, aux=None)]
    assert all(np.array_equal(a, b) for a, b in zip(e1, e2))
    # the aux slice is the sidecar row of the same sample
    np.random.seed(1)
    bx, be, by = next(iter(V2Loader(root, 64, shuffle=False, aux="M")))
    Mp = np.load(GEN.sidecar_path(root, 0))
    assert np.array_equal(be[:, 1:], Mp.astype(np.float32))
    with np.load(GEN.shard_path(root, 0)) as z:
        assert np.array_equal(be[:, :1], z["energy"].astype(np.float32))
    # prefix loader (learning curve): only the first shard of the dir
    n = sum(len(by) for _bx, _be, by in V2Loader(root, 64, shuffle=False, aux=None, max_rows_per_dir=64))
    assert n == 64


def test_std_stats_conventions(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    root = _scratch("stats")
    cfg = _cfg()
    dirs = []
    for sd in (42, 43):
        d = os.path.join(root, "half%d" % (sd - 42))
        GEN.gen_dataset_v2(engine, ops, d, h_type="random", state_type="thermal", beta=1.0,
                           num_samples=128, gen_bs=64, seed=sd, cfg=cfg, verbose=False)
        dirs.append(d)
    mu, sg, n = ST.std1_entry_stats(dirs, 64, 10, n_max=200)
    assert n == 256 and mu.shape == (10, 10) and sg.dtype == np.float32   # overshoot-inclusive break
    # direct recomputation over the same rows
    X = np.concatenate([np.load(GEN.shard_path(d, i))["features"][..., 0]
                        for d in dirs for i in range(2)]).astype(np.float64)
    Xs = 0.5 * (X + np.swapaxes(X, 1, 2))
    assert np.allclose(mu, Xs.mean(0).astype(np.float32), atol=1e-7)
    assert np.allclose(sg, np.maximum(Xs.std(0), 1e-8).astype(np.float32), atol=1e-6)
    assert np.array_equal(mu, mu.T)
    mean, std, n2 = ST.triu55_stats(dirs, 64, 10, n_max=200)
    assert n2 == 256 and mean.shape == (55,) and np.all(std > 0)
    r, c, mask = ST._triu_mask(10)
    F = Xs[:, r, c] * mask
    assert np.allclose(mean, F.mean(0).astype(np.float32), atol=1e-6)
    p = os.path.join(root, "std_stats.npz")
    ST.save_std_stats(p, "std1_entry", mu, sg, n, "test")
    kind, mu2, sg2, n3 = ST.load_std_stats(p)
    assert kind == "std1_entry" and np.array_equal(mu2, mu) and n3 == n


def test_resolve_rules(ctx):
    cfg = V2Config()
    r = resolve(cfg, state_type="thermal", label_size=55, m=10, arch="ogn_std")
    assert r.aux == "M" and r.readout == "affine" and r.w_gauge == 0.0 and r.std_convention == "std1_entry"
    r = resolve(cfg, state_type="gs", label_size=55, m=10, arch="ogn_std")
    assert r.aux == "psi0" and r.readout == "affine"
    r = resolve(cfg, state_type="gs", label_size=1, m=10, arch="ogn")
    assert r.readout == "published" and r.w_gauge == 1.0 and r.std_convention is None
    r = resolve(cfg, state_type="thermal", label_size=55, m=10, arch="ogn_std", use_energy_input=False)
    assert r.readout == "affine" and r.shell == "off"
    r = resolve(cfg, state_type="thermal", label_size=55, m=10, arch="mlp_std")
    assert r.readout == "published" and r.std_convention == "triu55"
    r = resolve(V2Config(loss="mse"), state_type="thermal", label_size=55, m=10, arch="ogn_std")
    assert r.aux is None and r.w_gauge == 0.0
    assert V1_PUBLISHED.published() and not cfg.published()
    assert V2Config.from_json(r.to_json()) == r
    try:
        V2Config(readout="affine", w_gauge=1.0); raise AssertionError("w_gauge accepted with affine")
    except ValueError:
        pass
