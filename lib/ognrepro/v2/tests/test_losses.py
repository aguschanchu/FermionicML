"""v2 loss / training-step tests (CPU)."""
import numpy as np
import jax
import jax.numpy as jnp

from .. import losses as L
from .. import ops_index as OI
from .. import train_step as TS
from .. import model as MDL
from ..config import V2Config, resolve
from ... import arch_variants
from .test_model import _rho_from_labels, _features


def _batch(ctx, B=4, seed=0, beta=1.0):
    engine = ctx["engine"]
    _, labels = engine.g_gen.generate(jax.random.PRNGKey(seed))
    labels = np.asarray(labels[:B], np.float32)
    G, vals, vecs, p, rho = _rho_from_labels(ctx, labels, beta)
    f2 = _features(ctx, rho).astype(np.float32)[..., None]
    E = (p * vals).sum(1).astype(np.float32)[:, None]
    rng = np.random.default_rng(seed)
    logits = (labels + 0.02 * rng.standard_normal(labels.shape)).astype(np.float32)
    return labels, logits, f2, E, G, rho, vecs


def _jac_reconstruct(ctx):
    """Jacobian (A, n_lab) of the linear label -> G map (C-order flattening)."""
    g_gen = ctx["engine"].g_gen
    n = 55
    J = jax.jacfwd(lambda l: g_gen.reconstruct(l[None])[0].reshape(-1))(jnp.zeros((n,), jnp.float32))
    return np.asarray(J, np.float64)


def test_lambda_schedule(ctx):
    cfg = V2Config()
    T = 488_281
    lam = L.make_lambda(cfg, T)
    vals = [float(lam(jnp.asarray(s))) for s in (0, int(0.4 * T), int(0.8 * T), T)]
    assert abs(vals[0] - 10.0) < 1e-4
    assert abs(vals[1] - np.sqrt(10.0 * 1e-2)) < 2e-3
    assert abs(vals[2] - 1e-2) < 2e-5 and abs(vals[3] - 1e-2) < 2e-5
    assert L.make_lambda(V2Config(loss="gram", readout="published", w_gauge=1.0), T) is None
    lam_c = L.make_lambda(V2Config(lam0=0.3, lam1=0.3), T)
    assert abs(float(lam_c(jnp.asarray(0))) - 0.3) < 1e-6 and abs(float(lam_c(jnp.asarray(T))) - 0.3) < 1e-6


def test_metric_loss_matches_numpy_and_is_gauge_invariant(ctx):
    ops = ctx["ops"]; engine = ctx["engine"]
    labels, logits, f2, E, G, rho, vecs = _batch(ctx, B=4, seed=1)
    Gp = np.asarray(engine.g_gen.reconstruct(jnp.asarray(logits)), np.float64)
    dw = (Gp - G).reshape(4, -1)
    iu = np.triu_indices(ops["A"])
    S_engine = jnp.asarray(ops["S"], jnp.float32)
    # thermal: packed covariance of the true state
    Mp = np.stack([OI.cov_thermal_numpy(ops, rho[b])[0][iu] for b in range(4)]).astype(np.float32)
    cfg_t = resolve(V2Config(), state_type="thermal", label_size=55, m=10, arch="ogn_std")
    fn_t = L.make_loss(cfg_t, ops, 1000, "M", False)
    tot, parts = fn_t(jnp.asarray(logits), jnp.asarray(labels), jnp.asarray(Mp), jnp.asarray(0),
                      engine.g_gen, S_engine)
    qM_ref = np.array([dw[b] @ OI.cov_thermal_numpy(ops, rho[b])[0] @ dw[b] for b in range(4)])
    qS_ref = np.einsum("bi,ij,bj->b", dw, ops["S_inf"], dw)
    assert abs(float(parts["qM"]) - qM_ref.mean()) < 1e-4 * max(qM_ref.mean(), 1e-12)
    assert abs(float(parts["qS"]) - qS_ref.mean()) < 1e-4 * max(qS_ref.mean(), 1e-12)
    ridge_ref = (dw ** 2).sum(1).mean()
    exp_tot = (qM_ref + 10.0 * qS_ref).mean() + cfg_t.w_ridge * ridge_ref   # w_gauge = 0 (affine)
    assert cfg_t.w_gauge == 0.0 and abs(float(tot) - exp_tot) < 1e-4 * exp_tot
    # ground state: psi0
    _l, logits_g, _f, _E, G_g, rho_g, vecs_g = _batch(ctx, B=4, seed=2, beta=100.0)
    psi = np.stack([vecs_g[b, :, 0] for b in range(4)]).astype(np.float32)
    Gp_g = np.asarray(engine.g_gen.reconstruct(jnp.asarray(logits_g)), np.float64)
    dw_g = (Gp_g - G_g).reshape(4, -1)
    cfg_g = resolve(V2Config(), state_type="gs", label_size=55, m=10, arch="ogn_std")
    fn_g = L.make_loss(cfg_g, ops, 1000, "psi0", True)
    _tot, parts_g = fn_g(jnp.asarray(logits_g), jnp.asarray(_l), jnp.asarray(psi), jnp.asarray(0),
                         engine.g_gen, S_engine)
    qM_g = np.array([OI.cov_gs_numpy(ops, psi[b], dw_g[b]) for b in range(4)])
    assert abs(float(parts_g["qM"]) - qM_g.mean()) < 1e-4 * max(qM_g.mean(), 1e-12), (float(parts_g["qM"]), qM_g.mean())
    # gauge invariance of the metric parts: shift every diagonal coupling by c
    r, c = np.triu_indices(10)
    shift = np.where(r == c, 0.05, 0.0).astype(np.float32)
    _t2, parts2 = fn_t(jnp.asarray(logits + shift), jnp.asarray(labels), jnp.asarray(Mp), jnp.asarray(0),
                       engine.g_gen, S_engine)
    assert abs(float(parts2["qM"]) - float(parts["qM"])) < 2e-4 * max(float(parts["qM"]), 1e-9)
    assert abs(float(parts2["qS"]) - float(parts["qS"])) < 2e-4 * max(float(parts["qS"]), 1e-9)
    assert float(parts2["gauge"]) > float(parts["gauge"])


def test_metric_gradient_matches_analytic(ctx):
    ops = ctx["ops"]; engine = ctx["engine"]
    labels, logits, f2, E, G, rho, vecs = _batch(ctx, B=3, seed=4)
    iu = np.triu_indices(ops["A"])
    Ms = [OI.cov_thermal_numpy(ops, rho[b])[0] for b in range(3)]
    Mp = np.stack([M[iu] for M in Ms]).astype(np.float32)
    cfg = V2Config(w_ridge=0.0)
    cfg = resolve(cfg, state_type="thermal", label_size=55, m=10, arch="ogn_std")
    fn = L.make_loss(cfg, ops, 1000, "M", False)
    S_engine = jnp.asarray(ops["S"], jnp.float32)
    lam0 = 10.0
    g = np.asarray(jax.grad(lambda lg: fn(lg, jnp.asarray(labels), jnp.asarray(Mp), jnp.asarray(0),
                                         engine.g_gen, S_engine)[0])(jnp.asarray(logits)), np.float64)
    J = _jac_reconstruct(ctx)                                    # (100, 55)
    Gp = np.asarray(engine.g_gen.reconstruct(jnp.asarray(logits)), np.float64)
    dw = (Gp - G).reshape(3, -1)
    g_ref = np.stack([J.T @ (2.0 * (Ms[b] + lam0 * ops["S_inf"]) @ dw[b]) for b in range(3)]) / 3.0
    assert np.abs(g - g_ref).max() < 1e-4 * max(np.abs(g_ref).max(), 1e-12), np.abs(g - g_ref).max()


def test_gram_branch_matches_engine_formula(ctx):
    ops = ctx["ops"]; engine = ctx["engine"]
    labels, logits, f2, E, G, rho, vecs = _batch(ctx, B=4, seed=5)
    cfg = resolve(V2Config(loss="gram", readout="published", w_gauge=1.0), state_type="thermal",
                  label_size=55, m=10, arch="ogn")
    assert cfg.w_gauge == 1.0 and cfg.aux is None
    fn = L.make_loss(cfg, ops, 1000, None, False)
    S_engine = jnp.asarray(ops["S"], jnp.float32)
    tot, parts = fn(jnp.asarray(logits), jnp.asarray(labels), jnp.zeros((4, 0), jnp.float32),
                    jnp.asarray(0), engine.g_gen, S_engine)
    Gp = np.asarray(engine.g_gen.reconstruct(jnp.asarray(logits)), np.float64)
    dw = (Gp - G).reshape(4, -1)
    phys = np.einsum("bi,ij,bj->b", dw, ops["S"], dw).mean()
    trace = np.mean((np.trace(Gp, axis1=1, axis2=2) / 10 - np.trace(G, axis1=1, axis2=2) / 10) ** 2)
    ridge = (dw ** 2).sum(1).mean()
    ref = phys + 1.0 * trace + 1e-2 * ridge
    assert abs(float(tot) - ref) < 1e-5 * ref, (float(tot), ref)
    # mse branch = engine base_loss
    cfg_m = resolve(V2Config(loss="mse"), state_type="thermal", label_size=55, m=10, arch="ogn_std")
    fn_m = L.make_loss(cfg_m, ops, 1000, None, False)
    tot_m, _ = fn_m(jnp.asarray(logits), jnp.asarray(labels), jnp.zeros((4, 0), jnp.float32),
                    jnp.asarray(0), engine.g_gen, S_engine)
    assert abs(float(tot_m) - ((logits - labels) ** 2).sum(1).mean()) < 1e-6


def test_S_inf_is_beta0_covariance_and_psd(ctx):
    ops = ctx["ops"]
    D = ops["D"]
    M0, _ = OI.cov_thermal_numpy(ops, np.eye(D) / D)
    # S = Tr(h_a^dag h_b)/D and Tr(h_a h_b)/D differ off the symmetric subspace
    # (h_a^dag = h_{a'}); the quadratic forms agree for every symmetric dw
    rng = np.random.default_rng(0)
    for _ in range(4):
        dG = rng.standard_normal((ops["m"], ops["m"])); dG = dG + dG.T
        dw = dG.reshape(-1)
        assert abs(dw @ M0 @ dw - dw @ ops["S_inf"] @ dw) < 1e-10 * max(1.0, dw @ ops["S_inf"] @ dw)
    w = np.linalg.eigvalsh(0.5 * (ops["S_inf"] + ops["S_inf"].T))
    assert w.min() > -1e-12
    # u: 0.5 on the m number operators (a = k*m + k), 0 elsewhere (d20: 5 pairs in 10 levels)
    u = ops["u"]
    diag = [k * ops["m"] + k for k in range(ops["m"])]
    assert np.allclose(u[diag], 0.5) and np.allclose(np.delete(u, diag), 0.0)


def test_train_step_energy_aux_split_and_pmap(ctx):
    """The pmap step receives be = [E | aux]; only E reaches the network; the
    step runs on CPU (1 device) and the loss is finite."""
    from flax.training import train_state as fts  # noqa: PLC0415
    import optax  # noqa: PLC0415
    ops = ctx["ops"]; engine = ctx["engine"]
    labels, logits, f2, E, G, rho, vecs = _batch(ctx, B=4, seed=6)
    iu = np.triu_indices(ops["A"])
    Mp = np.stack([OI.cov_thermal_numpy(ops, rho[b])[0][iu] for b in range(4)]).astype(np.float32)
    be = np.concatenate([E, Mp], axis=1)
    e_, a_ = TS.split_be(jnp.asarray(be))
    assert e_.shape == (4, 1) and a_.shape == (4, 5050)
    cfg = resolve(V2Config(), state_type="thermal", label_size=55, m=10, arch="ogn")
    mdl = MDL.build_v2_model("ogn", 55, 1, True, cfg, ops)
    v = mdl.init(jax.random.PRNGKey(0), jnp.asarray(f2), jnp.asarray(E), training=False)

    class TrainState(fts.TrainState):
        batch_stats: dict = None
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=mdl.apply, params=v["params"], tx=tx,
                              batch_stats=v.get("batch_stats", {}))
    step = TS.create_train_step_v2(cfg, ops, 100, "M", False, engine.g_gen)
    nd = jax.local_device_count()
    rep = lambda t: jax.device_put_replicated(t, jax.local_devices())
    p_state = rep(state)
    S_t = rep(jnp.asarray(ops["S"], jnp.float32))
    base_e = rep(jnp.asarray(engine.base_e_params_global, jnp.float32))
    r1 = rep(jnp.asarray(engine.d_rho_1_diag_global, jnp.float32))
    it = rep(jnp.asarray(engine.d_inter_tensor_global, jnp.float32))
    bx = jnp.asarray(f2).reshape(nd, 4 // nd, 10, 10, 1)
    bbe = jnp.asarray(be).reshape(nd, 4 // nd, -1)
    by = jnp.asarray(labels).reshape(nd, 4 // nd, 55)
    p_state, loss = step(p_state, bx, bbe, by, S_t, base_e, r1, it)
    assert np.isfinite(float(np.asarray(loss)[0]))
    # perturbing the aux slice changes the loss but not the network output
    bbe2 = bbe.at[..., 1:].multiply(3.0)
    _p2, loss2 = step(p_state, bx, bbe2, by, S_t, base_e, r1, it)
    assert float(np.asarray(loss2)[0]) != float(np.asarray(loss)[0])
    out1 = mdl.apply(v, jnp.asarray(f2), jnp.asarray(E), training=False)
    assert np.array_equal(np.asarray(out1), np.asarray(mdl.apply(v, jnp.asarray(f2), jnp.asarray(E), training=False)))


def test_engine_hooks_published_path_and_delegation(ctx):
    engine = ctx["engine"]; ops = ctx["ops"]
    # the published factory still builds the published step (no v2 in the engine yet is fine)
    step = engine.create_train_step("gram", True, 1.0)
    assert step is not None
    # requesting v2 before install raises
    cfg = resolve(V2Config(), state_type="thermal", label_size=55, m=10, arch="ogn")
    engine.ns.pop("create_train_step_v2", None)
    try:
        engine.create_train_step("metric", True, 1.0, total_steps=10, v2=cfg)
        raise AssertionError("delegation without install did not raise")
    except RuntimeError:
        pass
    # install, then the engine's create_train_step delegates
    import ognrepro.v2 as V2  # noqa: PLC0415
    notes = V2.install(engine, cfg, ops, total_steps=10, aux_kind="M", is_gs=False)
    assert "create_train_step_v2" in notes["train_step"]
    step2 = engine.create_train_step("metric", True, 1.0, total_steps=10, v2=cfg)
    assert step2 is not None
    # build_model rebind: v2 kwarg -> v2 module, none -> original class
    m_v2 = engine.build_model("ogn", 55, 1, True, v2=cfg)
    assert type(m_v2).__name__ == "PhysicsOrbitalGraphNet" and m_v2.proj_gauge and m_v2.proj_shell == "euclid"
    m_orig = engine.build_model("ogn", 55, 1, True)
    assert not getattr(m_orig, "proj_gauge", False)
    # restore the original factory for the other tests
    engine.ns["build_model"] = engine.ns["_build_model_orig_v2"]
