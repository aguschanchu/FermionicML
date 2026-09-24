"""ognrepro.v2.testenv -- bind the vendored engine for v2 tests and CPU smokes.

Mirrors explore_v2f/env.py + engine_bind.py (@ 269bee39): the canonical
exec-namespace engine is the repository's campaign/ tree (byte-identical to
the release-vendored copy except p6), numba gets a per-process cache dir
(p4_gevp's @njit(cache=True) kernels poison any shared cache), and the engine
is bound ONCE per process (its globals are mutable and read late-bound).

    from ognrepro.v2 import testenv
    testenv.setup_env()            # before the first jax / repro_common import
    engine = testenv.bind("random", "thermal", beta=1.0, gpu_batch_size=8)
    ops = testenv.ops()
"""
import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
ENGINE_DIR = os.path.join(REPO_ROOT, "campaign")
_BOUND = {}


def setup_env():
    os.environ.setdefault("OGNREPRO_ENGINE_DIR", ENGINE_DIR)
    os.environ.setdefault("OGNREPRO_ROOT",
                          os.path.join(REPO_ROOT, "notebook_release", "src"))
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "numba_cache_v2_%d" % os.getpid()))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    lib = os.path.join(REPO_ROOT, "notebook_release", "src", "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    return REPO_ROOT


def get_engine():
    setup_env()
    from .. import repro_common  # noqa: PLC0415
    return repro_common.get_engine()


def bind(h_type="random", state_type="thermal", beta=1.0, gpu_batch_size=8,
         g_init=0.1, g_stop=1.0, verbose=True):
    key = (h_type, state_type, float(beta), float(g_init), float(g_stop))
    if _BOUND:
        if _BOUND["key"] != key:
            raise RuntimeError("[testenv] engine already bound to %r; wanted %r "
                               "(start a fresh process)" % (_BOUND["key"], key))
        return _BOUND["engine"]
    t0 = time.time()
    engine = get_engine()
    engine.init_d20(h_type, state_type, beta=float(beta),
                    gpu_batch_size=int(gpu_batch_size),
                    g_init=float(g_init), g_stop=float(g_stop))
    engine.init_gram()
    from . import ops_index  # noqa: PLC0415
    ops_ = ops_index.build(engine)
    assert engine.g_gen.h_type == h_type
    _BOUND.update(key=key, engine=engine, ops=ops_, h_type=h_type,
                  state_type=state_type, beta=float(beta))
    if verbose:
        print("[testenv] %s/%s beta=%g bound in %.1fs (D=%d, m=%d)"
              % (h_type, state_type, beta, time.time() - t0,
                 int(engine.basis.size), int(engine.M_PAIRS)), flush=True)
    return engine


def ops():
    if not _BOUND:
        raise RuntimeError("[testenv] bind() first")
    return _BOUND["ops"]
