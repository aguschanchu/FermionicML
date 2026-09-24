"""ognrepro.v2.model -- build the v2 network and install it in the engine.

The v2 network IS the release class of the arm (arch_variants.PhysicsOrbitalGraphNet
for 'ogn', std_arch.StdPhysicsOrbitalGraphNet for 'ogn_std' -- the manuscript's
STD-1 flagship class) with the [V2] readout fields set from V2Config; the
parameter tree is the published one (the projection has no parameters).
`install_build_model` rebinds engine.ns['build_model'] with the late-binding
pattern of the release trainer (train_lane.py:304-329): the engine's
train_model passes `v2=cfg` (p3_training.py [V2] hook) and receives the v2
module; every other call (v2=None) goes to the original factory.
"""
from .. import arch_variants, std_arch
from .config import V2Config

V2_ARCHS = ("ogn", "ogn_std", "mlp", "mlp_std")


def readout_kwargs(cfg, ops):
    if cfg.readout != "affine":
        return dict(proj_gauge=False, proj_shell="off", gauge_value=float(cfg.gauge_value),
                    pair_energies=())
    return dict(proj_gauge=True, proj_shell=str(cfg.shell), gauge_value=float(cfg.gauge_value),
                pair_energies=tuple(float(v) for v in ops["pair_energies"]))


def build_v2_model(arch, label_size, res, include_energy, cfg, ops, std_stats=None,
                   use_scatter=True, use_reinject=True, use_orb_emb=True, readout_bias=0.55,
                   use_energy_input=True, n_elec=None, mlp_width=None, mlp_blocks=8):
    """std_stats: (mu, sigma) m x m arrays for 'ogn_std'; (mean, std) (55,) for 'mlp_std'."""
    if not isinstance(cfg, V2Config):
        raise TypeError("cfg must be a V2Config")
    if not cfg.resolved:
        raise ValueError("resolve() the config for the lane before building the model")
    rk = readout_kwargs(cfg, ops)
    if arch == "ogn":
        return arch_variants.PhysicsOrbitalGraphNet(
            label_size=int(label_size), res=int(res), include_energy=bool(include_energy),
            use_scatter=use_scatter, use_reinject=use_reinject, use_orb_emb=use_orb_emb,
            readout_bias=float(readout_bias), use_energy_input=bool(use_energy_input), **rk)
    if arch == "ogn_std":
        if std_stats is None or cfg.std_convention != "std1_entry":
            raise ValueError("ogn_std needs STD-1 (mu, sigma) statistics")
        mu_t, sg_t = std_arch.std_stats_tuples(std_stats[0], std_stats[1])
        return std_arch.StdPhysicsOrbitalGraphNet(
            label_size=int(label_size), res=int(res), include_energy=bool(include_energy),
            use_scatter=use_scatter, use_reinject=use_reinject, use_orb_emb=use_orb_emb,
            readout_bias=float(readout_bias), use_energy_input=bool(use_energy_input),
            std_mu=mu_t, std_sigma=sg_t, **rk)
    if arch in ("mlp", "mlp_std"):
        from .. import mlp_variants  # noqa: PLC0415
        if cfg.readout == "affine":
            raise ValueError("the MLP arms keep the published readout (author decision)")
        mean = std = ()
        if arch == "mlp_std":
            if std_stats is None or cfg.std_convention != "triu55":
                raise ValueError("mlp_std needs triu55 (mean, std) statistics")
            mean = tuple(float(v) for v in std_stats[0]); std = tuple(float(v) for v in std_stats[1])
        return mlp_variants.DeepResMLPV2(label_size=int(label_size), res=int(res),
                                         include_energy=bool(include_energy),
                                         readout_bias=float(readout_bias), n_elec=int(n_elec),
                                         std_mean=mean, std_std=std,
                                         width=int(mlp_width or 0), num_blocks=int(mlp_blocks))
    raise ValueError("arch %r not a v2 arch (%s)" % (arch, ", ".join(V2_ARCHS)))


def model_build_record(model, arch, cfg, std_stats_sha=None):
    """The config.json 'model_build' block (enough to rebuild without the engine)."""
    fields = {}
    for k in ("label_size", "res", "include_energy", "use_scatter", "use_reinject", "use_orb_emb",
              "readout_bias", "use_energy_input", "proj_gauge", "proj_shell", "gauge_value",
              "pair_energies", "n_elec", "width", "num_blocks"):
        if hasattr(model, k):
            v = getattr(model, k)
            fields[k] = list(v) if isinstance(v, tuple) else v
    return dict(cls=type(model).__name__, arch=arch, fields=fields,
                std_convention=cfg.std_convention, std_stats_sha256=std_stats_sha)


def install_build_model(engine, cfg, ops, std_stats=None, use_energy_input=True, n_elec=None,
                        mlp_width=None, mlp_blocks=8):
    """Rebind engine.ns['build_model'] so train_model(..., v2=cfg) constructs the
    v2 module; calls without v2 keep the original factory."""
    orig = engine.ns.get("_build_model_orig_v2") or engine.ns["build_model"]
    engine.ns["_build_model_orig_v2"] = orig

    def build_model_v2(arch, label_size, res, include_energy, use_scatter=True,
                       use_reinject=True, use_orb_emb=True, readout_bias=0.55, v2=None, **kw):
        if v2 is None:
            return orig(arch, label_size, res, include_energy, use_scatter=use_scatter,
                        use_reinject=use_reinject, use_orb_emb=use_orb_emb,
                        readout_bias=readout_bias)
        if v2 != cfg:
            raise RuntimeError("[v2.model] installed config differs from the requested one")
        return build_v2_model(arch, label_size, res, include_energy, cfg, ops, std_stats=std_stats,
                              use_scatter=use_scatter, use_reinject=use_reinject,
                              use_orb_emb=use_orb_emb, readout_bias=readout_bias,
                              use_energy_input=use_energy_input, n_elec=n_elec,
                              mlp_width=mlp_width, mlp_blocks=mlp_blocks)

    engine.ns["build_model"] = build_model_v2
    return "build_model_v2(readout=%s, shell=%s, std=%s)" % (cfg.readout, cfg.shell, cfg.std_convention)
