"""v2 model tests: flag-off parity with the published classes, exactness of
the affine readout on clean and population-noise inputs, the raw-input rule,
the low-dim-head refusal, the energy-free noE arm."""
import numpy as np
import jax
import jax.numpy as jnp

from ... import arch_variants, std_arch, readout


def _tree_shapes(params):
    from flax.traverse_util import flatten_dict
    return {"/".join(k): tuple(v.shape) for k, v in flatten_dict(params).items()}


def _n_params(params):
    return int(sum(np.prod(s) for s in _tree_shapes(params).values()))


def _rho_from_labels(ctx, labels, beta_eff):
    engine = ctx["engine"]
    G = np.asarray(engine.g_gen.reconstruct(jnp.asarray(labels)), np.float64)
    B = len(G)
    e_params = jnp.array(np.tile(engine.U_ENERGY_SEED[0], (B, 1)))
    H = np.asarray(engine.two_body_hamiltonian_dense(
        e_params, jnp.array(G, jnp.float32), engine.d_rho_1_diag_global,
        engine.d_inter_tensor_global), np.float64)
    vals, vecs = np.linalg.eigh(H)
    p = np.exp(-beta_eff * (vals - vals.min(1, keepdims=True)))
    p /= p.sum(1, keepdims=True)
    rho = np.einsum("bnk,bk,bmk->bnm", vecs, p, vecs)
    return G, vals, vecs, p, rho


def _features(ctx, rho):
    dense = np.asarray(ctx["engine"].rho_2_kkbar_arrays.todense())
    return np.einsum("bnm,klmn->bkl", rho, dense, optimize=True)


def _random_inputs(ctx, B=3, seed=0, beta=1.0):
    key = jax.random.PRNGKey(seed)
    _, labels = ctx["engine"].g_gen.generate(key)
    labels = np.asarray(labels[:B])
    G, vals, vecs, p, rho = _rho_from_labels(ctx, labels, beta)
    f2 = _features(ctx, rho)
    E = (p * vals).sum(1)
    return f2.astype(np.float32)[..., None], E.astype(np.float32)[:, None], G


def _shell_target(ops, x, E):
    rho = x[..., 0].astype(np.float64)
    return np.einsum("k,bkk->b", ops["pair_energies"], rho) - E[:, 0].astype(np.float64), rho


def _reconstruct(ctx, out):
    return np.asarray(ctx["engine"].g_gen.reconstruct(jnp.asarray(out, jnp.float32)), np.float64)


# ---------------------------------------------------------------- flag-off parity
def test_flags_off_matches_engine_ogn(ctx):
    engine = ctx["engine"]
    x, E, _ = _random_inputs(ctx)
    ref = engine.build_model("ogn", 55, 3, True)
    ours = arch_variants.build_ogn(label_size=55, res=3, include_energy=True)
    v_ref = ref.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(E), training=False)
    v_our = ours.init(jax.random.PRNGKey(0), jnp.asarray(x), jnp.asarray(E), training=False)
    assert _tree_shapes(v_ref["params"]) == _tree_shapes(v_our["params"])
    assert _n_params(v_our["params"]) == 17_756_929, _n_params(v_our["params"])
    o_ref = np.asarray(ref.apply(v_ref, jnp.asarray(x), jnp.asarray(E), training=False))
    o_our = np.asarray(ours.apply(v_our, jnp.asarray(x), jnp.asarray(E), training=False))
    assert np.array_equal(o_ref, o_our), np.abs(o_ref - o_our).max()


def test_std_class_unit_stats_matches_plain(ctx):
    """StdPhysicsOrbitalGraphNet with mu=0, sigma=1 is the plain class bit for
    bit (x_std == x_sym exactly), so the edited std class keeps its tree."""
    x, E, _ = _random_inputs(ctx, seed=1)
    m = 10
    mu_t = tuple(tuple(0.0 for _ in range(m)) for _ in range(m))
    sg_t = tuple(tuple(1.0 for _ in range(m)) for _ in range(m))
    std = std_arch.StdPhysicsOrbitalGraphNet(label_size=55, res=3, std_mu=mu_t, std_sigma=sg_t)
    plain = arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=3)
    v_s = std.init(jax.random.PRNGKey(3), jnp.asarray(x), jnp.asarray(E), training=False)
    v_p = plain.init(jax.random.PRNGKey(3), jnp.asarray(x), jnp.asarray(E), training=False)
    assert _tree_shapes(v_s["params"]) == _tree_shapes(v_p["params"])
    assert _n_params(v_s["params"]) == 17_756_929
    o_s = np.asarray(std.apply(v_s, jnp.asarray(x), jnp.asarray(E), training=False))
    o_p = np.asarray(plain.apply(v_p, jnp.asarray(x), jnp.asarray(E), training=False))
    assert np.array_equal(o_s, o_p), np.abs(o_s - o_p).max()


# ---------------------------------------------------------------- the affine readout
def _std_stats_random(x, seed=0):
    rng = np.random.default_rng(seed)
    m = x.shape[1]
    mu = rng.normal(0.4, 0.1, (m, m)); mu = 0.5 * (mu + mu.T)
    sg = np.abs(rng.normal(0.2, 0.05, (m, m))) + 1e-3; sg = 0.5 * (sg + sg.T)
    return std_arch.std_stats_tuples(mu, sg)


def test_readout_exact_on_clean_inputs(ctx):
    ops = ctx["ops"]
    x, E, _ = _random_inputs(ctx, B=4, seed=2)
    t, rho = _shell_target(ops, x, E)
    pe = tuple(float(v) for v in ops["pair_energies"])
    mu_t, sg_t = _std_stats_random(x)
    models = {
        "plain": arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=1, proj_gauge=True,
                                                      proj_shell="euclid", pair_energies=pe),
        "std": std_arch.StdPhysicsOrbitalGraphNet(label_size=55, res=1, std_mu=mu_t, std_sigma=sg_t,
                                                  proj_gauge=True, proj_shell="euclid", pair_energies=pe),
    }
    for name, mdl in models.items():
        v = mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
        assert _n_params(v["params"]) == _n_params(
            arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=1).init(
                jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)["params"])
        out = np.asarray(mdl.apply(v, jnp.asarray(x), jnp.asarray(E), training=False), np.float64)
        Gp = _reconstruct(ctx, out)
        dm = np.trace(Gp, axis1=1, axis2=2) / 10
        assert np.abs(dm - 0.55).max() < 1e-4, (name, dm)
        got = np.einsum("bij,bij->b", Gp, rho)
        assert np.abs(got - t).max() < 1e-3 * np.abs(t).max(), (name, got, t)
    # the numpy f64 reference agrees with the (f32, HIGHEST) jax projection on the same G
    G0 = np.asarray(ctx["engine"].g_gen.reconstruct(jnp.asarray(np.zeros((4, 55), np.float32) + 0.5)), np.float64)
    Gj = np.asarray(readout.project_affine(jnp.asarray(G0, jnp.float32), jnp.asarray(rho, jnp.float32),
                                           jnp.asarray(E, jnp.float32), pe, 0.55, True, "euclid"), np.float64)
    Gn = readout.project_affine_numpy(G0, rho, E, pe, 0.55, True, "euclid")
    assert np.abs(Gj - Gn).max() < 1e-5 * max(1.0, np.abs(Gn).max()), np.abs(Gj - Gn).max()
    # idempotent (f64 reference)
    Gnn = readout.project_affine_numpy(Gn, rho, E, pe, 0.55, True, "euclid")
    assert np.abs(Gnn - Gn).max() < 1e-12


def test_readout_exact_under_population_noise(ctx):
    """The measured pair (rho_meas, E_meas) of a finite-shot draw obeys the
    shell identity exactly (same eigenbasis), so the projected output does."""
    ops = ctx["ops"]
    engine = ctx["engine"]
    _, labels = engine.g_gen.generate(jax.random.PRNGKey(9))
    labels = np.asarray(labels[:3])
    G, vals, vecs, p, rho = _rho_from_labels(ctx, labels, 1.0)
    rng = np.random.default_rng(0)
    pe = tuple(float(v) for v in ops["pair_energies"])
    mdl = arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=1, proj_gauge=True,
                                               proj_shell="euclid", pair_energies=pe)
    for N_s in (1e4, 1e8):
        for model in ("gaussian", "multinomial"):
            if model == "gaussian":
                pm = p + rng.normal(0, 1, p.shape) * np.sqrt(p * (1 - p) / N_s)
                pm = np.maximum(pm, 0.0); pm /= pm.sum(1, keepdims=True)
            else:
                pm = np.stack([rng.multinomial(int(N_s), pi / pi.sum()) / N_s for pi in p])
            rho_m = np.einsum("bnk,bk,bmk->bnm", vecs, pm, vecs)
            x = _features(ctx, rho_m).astype(np.float32)[..., None]
            E = (pm * vals).sum(1).astype(np.float32)[:, None]
            t, rho_x = _shell_target(ops, x, E)
            v = mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
            out = np.asarray(mdl.apply(v, jnp.asarray(x), jnp.asarray(E), training=False), np.float64)
            Gp = _reconstruct(ctx, out)
            got = np.einsum("bij,bij->b", Gp, rho_x)
            assert np.abs(got - t).max() < 1e-3 * np.abs(t).max(), (N_s, model, got, t)
            assert np.abs(np.trace(Gp, axis1=1, axis2=2) / 10 - 0.55).max() < 1e-4


def test_std_readout_reads_raw_block(ctx):
    """With standardization on, the shell identity must hold against the RAW
    block; feeding the standardized block to the projection would break it by
    orders of magnitude (the ratio of the two blocks' norms)."""
    ops = ctx["ops"]
    x, E, _ = _random_inputs(ctx, B=3, seed=4)
    t, rho = _shell_target(ops, x, E)
    pe = tuple(float(v) for v in ops["pair_energies"])
    mu_t, sg_t = _std_stats_random(x, seed=5)
    mdl = std_arch.StdPhysicsOrbitalGraphNet(label_size=55, res=1, std_mu=mu_t, std_sigma=sg_t,
                                             proj_gauge=True, proj_shell="euclid", pair_energies=pe)
    v = mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
    out = np.asarray(mdl.apply(v, jnp.asarray(x), jnp.asarray(E), training=False), np.float64)
    Gp = _reconstruct(ctx, out)
    got = np.einsum("bij,bij->b", Gp, rho)
    assert np.abs(got - t).max() < 1e-3 * np.abs(t).max()
    z = (rho - np.asarray(mu_t)) / np.asarray(sg_t)
    got_z = np.einsum("bij,bij->b", Gp, z)
    t_z = np.einsum("k,bkk->b", ops["pair_energies"], z) - E[:, 0]
    assert np.abs(got_z - t_z).max() > 1e-2 * np.abs(t_z).max()


def test_gauge_only_noE_is_energy_free(ctx):
    x, E, _ = _random_inputs(ctx, B=3, seed=6)
    mdl = arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=1, use_energy_input=False,
                                               proj_gauge=True, proj_shell="off")
    v = mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
    shapes = _tree_shapes(v["params"])
    assert not any(s == (1, 32) for k, s in shapes.items() if k.endswith("kernel")), "energy branch present"
    o1 = np.asarray(mdl.apply(v, jnp.asarray(x), jnp.asarray(E), training=False))
    o2 = np.asarray(mdl.apply(v, jnp.asarray(x), jnp.asarray(E) + 7.0, training=False))
    assert np.array_equal(o1, o2)
    Gp = _reconstruct(ctx, o1)
    assert np.abs(np.trace(Gp, axis1=1, axis2=2) / 10 - 0.55).max() < 1e-5
    # with the shell row the arm would consume E (documents the noE rule)
    pe = tuple(float(v_) for v_ in ctx["ops"]["pair_energies"])
    mdl2 = arch_variants.PhysicsOrbitalGraphNet(label_size=55, res=1, use_energy_input=False,
                                                proj_gauge=True, proj_shell="euclid", pair_energies=pe)
    v2 = mdl2.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
    p1 = np.asarray(mdl2.apply(v2, jnp.asarray(x), jnp.asarray(E), training=False))
    p2 = np.asarray(mdl2.apply(v2, jnp.asarray(x), jnp.asarray(E) + 7.0, training=False))
    assert not np.array_equal(p1, p2)


def test_lowdim_head_refuses_projection(ctx):
    x, E, _ = _random_inputs(ctx, B=2, seed=7)
    pe = tuple(float(v) for v in ctx["ops"]["pair_energies"])
    for cls in (arch_variants.PhysicsOrbitalGraphNet, std_arch.StdPhysicsOrbitalGraphNet):
        kw = dict(label_size=1, res=1, proj_gauge=True, proj_shell="euclid", pair_energies=pe)
        if cls is std_arch.StdPhysicsOrbitalGraphNet:
            kw.update(zip(("std_mu", "std_sigma"), _std_stats_random(x)))
        mdl = cls(**kw)
        try:
            mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
        except ValueError:
            continue
        raise AssertionError("%s accepted the projection with a low-dim head" % cls.__name__)
    # and the published low-dim path is untouched: label 1 / 3 build and run
    for ls in (1, 3):
        mdl = arch_variants.PhysicsOrbitalGraphNet(label_size=ls, res=1)
        v = mdl.init(jax.random.PRNGKey(1), jnp.asarray(x), jnp.asarray(E), training=False)
        out = mdl.apply(v, jnp.asarray(x), jnp.asarray(E), training=False)
        assert out.shape == (2, ls)
