"""ognrepro.v2 -- the production-run v2 configuration of the OGN pipeline.

Three changes, measured on the exploration branch (explore_v2f @ 269bee39,
wave-2 report explore_v2f/report/wave2_report.pdf) and carried into the
release pipeline behind ONE versioned configuration object (config.V2Config):

  1. HIGHEST-precision dataset generation (Hamiltonian build AND pair-block
     contraction) plus a covariance / ground-state sidecar in the cache
     (gen.py, loader.py, stats.py);
  2. the exact affine readout -- gauge + energy-shell projection of the
     network's symmetric output (ognrepro.readout, wired into
     arch_variants.PhysicsOrbitalGraphNet and std_arch.StdPhysicsOrbitalGraphNet);
  3. the covariance-based ("state-aware metric") training loss with a
     log-linear lambda anneal (losses.py, train_step.py).

With every v2 flag off (V2Config.published()) the code paths are the
published ones: the model trees and forwards are byte-identical to the
release classes, and the engine keyword additions default to their
published numerics.  Modules are loaded lazily (importing this package
imports nothing heavy).
"""
import importlib

CFG_VERSION = 2

_MODULES = ("config", "ops_index", "gen", "loader", "stats", "losses",
            "train_step", "model", "registry", "d16_aux", "testenv")

__all__ = list(_MODULES) + ["CFG_VERSION"]


def install(engine, cfg, ops, *, total_steps, aux_kind, is_gs, std_stats=None,
            use_energy_input=True, n_elec=None, mlp_width=None, mlp_blocks=8):
    """Bind the v2 model factory and training step into the engine namespace
    (late binding: the engine's train_model and create_train_step pick them up
    when called with v2=cfg).  Returns a dict of notes for config.json."""
    from . import model as _model, train_step as _ts  # noqa: PLC0415
    notes = dict(build_model=_model.install_build_model(engine, cfg, ops, std_stats=std_stats,
                                                        use_energy_input=use_energy_input,
                                                        n_elec=n_elec, mlp_width=mlp_width,
                                                        mlp_blocks=mlp_blocks),
                 train_step=_ts.install(engine, cfg, ops, total_steps, aux_kind, is_gs))
    return notes


def __getattr__(name):
    if name in _MODULES:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_MODULES))
