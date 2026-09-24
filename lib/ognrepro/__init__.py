"""ognrepro -- the FermionicML notebook-release extracted library.

Modules (loaded LAZILY -- importing the package imports nothing heavy):
    repro_common   bundle root / eval convention / engine + values loaders
    blocked        seniority-blocked exact thermal solver + shard IO
    std_arch       standardized-input OGN (STD-1) + std-stats loader
    arch_variants  campaign5 OGN with the use_energy_input gate (ogn_noE)
    classical      classical estimator numerical cores (WLS/GLS/ridge/
                   secular/prior-ridge/box-MAP/forward-map/input-matched fit)
    restore        CPU-f64 checkpoint restore + forward + d20 registry
    figstyle       shared matplotlib rc
    readout        [V2] the exact affine readout (gauge + energy shell)
    v2             [V2] the production-run v2 configuration package

Extraction date 2026-08-23; per-module provenance headers carry source
paths, source sha256, line ranges, and VERBATIM/ADAPTED tags.

NOTE on import order: modules that import jax (std_arch, arch_variants,
restore) must be imported AFTER repro_common.enable_eval_convention() when
the CPU-f64 convention is wanted.  `blocked` pins single-thread BLAS env
vars at import (mirroring its source stage) -- import it before numpy gets
its thread pool if you want the pins to bind.
"""
import importlib

__version__ = "0.2.0"

_MODULES = ("repro_common", "blocked", "std_arch", "arch_variants",
            "classical", "restore", "figstyle", "readout", "mlp_variants", "v2")

__all__ = list(_MODULES)


def __getattr__(name):                     # PEP 562 lazy submodule access
    if name in _MODULES:
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_MODULES))
