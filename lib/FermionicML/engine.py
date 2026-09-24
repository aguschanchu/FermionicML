"""Engine: notebook-faithful shared namespace.

The original pipeline is a Jupyter notebook where every cell shares one global
namespace (functions reference module-level globals like `basis`,
`rho_2_kkbar_arrays`, `g_gen` at call time). Rather than refactor ~4k lines of
validated physics code, we replicate those semantics exactly: every file in
engine_parts/ is exec'd, in order, into ONE shared dict, so cross-part global
references keep working via late binding.

Usage from task drivers:
    import engine
    engine.init_d20(h_type="random", state_type="thermal", beta=1.0)
    loader = engine.gen_dataset(...)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARTS = [
    "p1_core.py",      # config init, basis/operators, hamiltonians, eigensolvers, GGenerator, dataset pipeline
    "p2_models.py",    # DeepResMLP, CoordResMLP, OGN (+ ablation toggles)
    "p3_training.py",  # gram metric, train/eval/predict, checkpoint save/load
    "p4_gevp.py",      # covariance/GEVP diagnostics suite (patched, sign-fixed)
    "p5_shots.py",     # WLS log-inversion, Gaussian + multinomial finite-shot protocol
    "p6_bcs_rg.py",    # BCS inversions, Richardson-Gaudin canonical inversion
]

ns = {"__name__": "engine_ns", "__file__": __file__}
for _p in _PARTS:
    _path = os.path.join(_HERE, "engine_parts", _p)
    with open(_path) as _f:
        exec(compile(_f.read(), _path, "exec"), ns)


def __getattr__(name):  # PEP 562: engine.foo -> shared namespace
    try:
        return ns[name]
    except KeyError:
        raise AttributeError(f"engine has no attribute {name!r}")
