"""ognrepro.repro_common -- shared bundle plumbing (paths, conventions, gates).

PROVENANCE
    Source: NEW FILE (no extracted body).  Conventions it encodes are lifted
    from the sealed evaluation stages:
      * CPU-f64 eval convention (JAX_PLATFORMS=cpu + JAX_ENABLE_X64=1 BEFORE
        the first jax import, matmul precision 'highest'):
        experiments/experiment-campaign2-r2/results/D16PROG-P4-SCORES/
        expc2r2_d16prog_p4_eval_as_run.py lines 115-125 and 1123-1150
        (source sha256 0a9edddc60696917e3d6bc61d4745410ea7139a856c0a55a750aa080f3453c59)
      * sha256_file: expc2r2_d12_blocked_validation.py lines 247-252
        (source sha256 9a14f209b740e6e17607208e3aa7dcc057083e705f1a10c5e74b4e4d14552947)
    Extraction date: 2026-08-23.  Tag: ADAPTED (new module; the two items
    above are the only borrowed bodies, both cited in place).

Tolerance constants:
    T1 = 1e-10   identity-grade (blocked-solver identities, sealed replays)
    T2 = 1e-6    restore-grade (CPU-f64 forward vs sealed scores)
"""
import hashlib
import json
import os
import sys

T1 = 1e-10
T2 = 1e-6

_ROOT_ENV = "OGNREPRO_ROOT"
_DATA_ENV = "OGNREPRO_DATA"
_ENGINE_ENV = "OGNREPRO_ENGINE_DIR"


def bundle_root():
    """Resolve the bundle root: $OGNREPRO_ROOT if set, else walk up from this
    file until a directory containing both `lib` and `data` is found, else the
    directory two levels up from lib/ognrepro (the src/ tree during W1)."""
    env = os.environ.get(_ROOT_ENV)
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        if (os.path.isdir(os.path.join(d, "lib"))
                and os.path.isdir(os.path.join(d, "data"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # fall back to the tree that holds lib/ (src/ during W1: lib exists,
    # data may not yet)
    return os.path.dirname(os.path.dirname(here))


BUNDLE_ROOT = bundle_root()


def enable_eval_convention():
    """Arm the CPU-f64 primary evaluation convention of the sealed records
    (D16PROG-P4 as-run lines 115-125): JAX_PLATFORMS=cpu and JAX_ENABLE_X64=1
    must be in the environment BEFORE the first jax import; then pin the
    default matmul precision to 'highest'.

    Raises RuntimeError if jax was already imported without x64 (the env pins
    would then be advisory only and every downstream comparison meaningless).
    """
    if "jax" in sys.modules:
        import jax
        if not jax.config.read("jax_enable_x64"):
            raise RuntimeError(
                "[ognrepro] jax is already imported WITHOUT x64: "
                "enable_eval_convention() must run before the first jax "
                "import (start a fresh process).")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ["JAX_ENABLE_X64"] = "1"
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ["JAX_ENABLE_X64"] = "1"
        import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError("[ognrepro] eval convention wants backend 'cpu', "
                           "got %r" % backend)
    return dict(backend=backend,
                x64=bool(jax.config.read("jax_enable_x64")),
                matmul_precision="highest", jax_version=jax.__version__)


def print_env():
    """Print the environment receipt (versions + convention flags)."""
    import numpy
    import scipy
    rec = dict(python=sys.version.split()[0],
               executable=sys.executable,
               numpy=numpy.__version__, scipy=scipy.__version__,
               bundle_root=BUNDLE_ROOT,
               env={k: os.environ.get(k) for k in
                    ("JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     _ROOT_ENV, _DATA_ENV, _ENGINE_ENV)})
    if "jax" in sys.modules:
        import jax
        rec["jax"] = jax.__version__
        rec["jax_backend"] = jax.default_backend()
        rec["jax_x64"] = bool(jax.config.read("jax_enable_x64"))
    try:
        import flax
        rec["flax"] = flax.__version__
    except Exception:                                        # noqa: BLE001
        pass
    for k, v in rec.items():
        print("  %-14s %s" % (k, v))
    return rec


def sha256_file(path, chunk=1 << 22):
    """expc2r2_d12_blocked_validation.py:247-252 _sha256_file, verbatim body."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_ENGINE_CACHE = {}


def get_engine():
    """Import the vendored engine (notebook-faithful shared namespace).

    Search order:
      1. $OGNREPRO_ENGINE_DIR (W1 smoke: the staged tree at
         release_bundle_v2/acceptance-public/code/FermionicML-staged/FermionicML)
      2. BUNDLE_ROOT/lib/FermionicML (the bundle's vendored tree)
    The chosen directory is inserted at sys.path[0] and `engine` imported.
    """
    if "engine" in _ENGINE_CACHE:
        return _ENGINE_CACHE["engine"]
    cands = []
    env = os.environ.get(_ENGINE_ENV)
    if env:
        cands.append(os.path.abspath(env))
    cands.append(os.path.join(BUNDLE_ROOT, "lib", "FermionicML"))
    engine_dir = next((d for d in cands
                       if os.path.isfile(os.path.join(d, "engine.py"))), None)
    if engine_dir is None:
        raise FileNotFoundError(
            "[ognrepro] no vendored engine found; looked in %r "
            "(set %s to override)" % (cands, _ENGINE_ENV))
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    import engine                                            # noqa: PLC0415
    if os.path.dirname(os.path.abspath(engine.__file__)) != engine_dir:
        raise RuntimeError(
            "[ognrepro] a DIFFERENT `engine` module is already imported from "
            "%r (wanted %r)" % (engine.__file__, engine_dir))
    _ENGINE_CACHE["engine"] = engine
    return engine


def load_values(name):
    """Load a sealed-values JSON from BUNDLE_ROOT/data/values/<name>.json
    (fallback root: $OGNREPRO_DATA/values/<name>.json)."""
    if not name.endswith(".json"):
        name = name + ".json"
    cands = [os.path.join(BUNDLE_ROOT, "data", "values", name)]
    env = os.environ.get(_DATA_ENV)
    if env:
        cands.append(os.path.join(os.path.abspath(env), "values", name))
    for p in cands:
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError("[ognrepro] values file %r not found in %r"
                            % (name, cands))


def assert_close(got, sealed, tol, label, provenance=None):
    """Compare a computed value/array against a sealed one; print a receipt
    line and raise AssertionError on failure."""
    import numpy as np
    g = np.asarray(got, np.float64)
    s = np.asarray(sealed, np.float64)
    if g.shape != s.shape:
        print("[assert_close] %-40s SHAPE MISMATCH got %r sealed %r"
              % (label, g.shape, s.shape))
        raise AssertionError("%s: shape mismatch %r vs %r"
                             % (label, g.shape, s.shape))
    delta = float(np.max(np.abs(g - s))) if g.size else 0.0
    ok = bool(np.isfinite(delta) and delta <= tol)
    gs = ("%.12g" % float(g)) if g.size == 1 else ("array%r" % (g.shape,))
    ss = ("%.12g" % float(s)) if s.size == 1 else ("array%r" % (s.shape,))
    print("[assert_close] %-40s value=%s sealed=%s maxD=%.3e tol=%.1e %s%s"
          % (label, gs, ss, delta, tol, "PASS" if ok else "FAIL",
             ("  [%s]" % provenance) if provenance else ""))
    if not ok:
        raise AssertionError("%s: max|delta|=%.3e > tol=%.1e"
                             % (label, delta, tol))
    return delta
