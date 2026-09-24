"""Self-contained v2 test runner (no pytest in the pinned env).

    JAX_PLATFORMS=cpu <pinned python> -m ognrepro.v2.tests.run_tests [pattern]

Binds the engine ONCE (random/thermal beta=1, gpu_batch_size 8; ~70 s) and
runs every test_* function of the test modules with a shared context dict
(engine, ops).
"""
import importlib
import sys
import time
import traceback

from .. import testenv

testenv.setup_env()

MODULES = ["test_model", "test_gen", "test_losses", "test_d16", "test_restore", "test_figaux", "test_tierb_stages", "test_tierb2_stages"]


def build_context():
    engine = testenv.bind("random", "thermal", beta=1.0, gpu_batch_size=8)
    return dict(engine=engine, ops=testenv.ops())


def main(argv=None):
    argv = argv or sys.argv[1:]
    pattern = argv[0] if argv else ""
    ctx = build_context()
    n_ok = n_fail = 0
    failures = []
    for modname in MODULES:
        mod = importlib.import_module("ognrepro.v2.tests." + modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or (pattern not in name and pattern not in modname):
                continue
            fn = getattr(mod, name)
            t0 = time.time()
            try:
                fn(ctx)
                n_ok += 1
                print("PASS %-50s %.1fs" % (modname + "." + name, time.time() - t0), flush=True)
            except Exception:  # noqa: BLE001
                n_fail += 1
                failures.append(modname + "." + name)
                print("FAIL %-50s %.1fs" % (modname + "." + name, time.time() - t0), flush=True)
                traceback.print_exc()
    print("\n%d passed, %d failed%s" % (n_ok, n_fail,
                                        (": " + ", ".join(failures)) if failures else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
