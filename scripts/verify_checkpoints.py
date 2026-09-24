#!/usr/bin/env python
"""verify_checkpoints -- integrity + forward verification of the bundle's
checkpoints.

    python verify_checkpoints.py --quick [--bundle-root DIR]
    python verify_checkpoints.py --full  [--bundle-root DIR]

--quick (~2 min): sha256 EVERY file under <bundle>/checkpoints/ and compare
against the release pin tables (loaded path-relatively from
notebook_release/build/expected_hashes.py):
  * d16prog lane files  -> the frozen GCS pull table
    (expected_hashes.D16PROG_FILES, build/copy_lists/checkpoints_gcs.tsv);
  * d20 final_state.msgpack -> the PRV-01 registry rows
    (expected_hashes.PRV01_HASHES == the supplement SM hash table);
  * data/streams/*.npy (bonus) -> PRV01_STREAM_HASHES.

--full: quick, then (CPU-f64 eval convention of the sealed records)
  * restore all 12 d16prog lanes via ognrepro.restore.build_arm_model /
    restore_params_f64 and forward the sealed P1-REF targets (TS-U-1024
    const / TS-V-512 vect, data/percells/D16PROG-P1-REF), comparing
    predictions AND derived error streams to the sealed D16PROG-P4-SCORES
    percell (scores_pred__<lane> / scores_err__*__<lane>) at the T2
    tolerance max|delta| <= 1e-6, reporting per-lane max|delta|
    (W1 measured the std_const lane bit-exact on this box);
  * restore each of the 14 d20-era checkpoints (ognrepro.restore.
    restore_d20) and run a finite forward on zeros (the full heldout-100
    evaluations belong to notebook B, not here).
Sequential, ognrepro.restore.free() + jax cache clearing between lanes
(sized for a 15 GB box).
"""
import argparse
import hashlib
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "lib")):
    sys.path.insert(0, os.path.join(_ROOT, "lib"))
    os.environ.setdefault("OGNREPRO_ROOT", _ROOT)

T2 = 1e-6

D16_LANES = tuple(
    "d16prog%s_%s%s" % (kind, sector, "" if seed == 42 else "_s%d" % seed)
    for sector in ("const_gs", "vect_gs") for kind in ("orig", "std")
    for seed in (42, 43, 44))
D16_SECTOR = {ln: ("const_gs" if "const_gs" in ln else "vect_gs")
              for ln in D16_LANES}
D16_LABEL_SIZE = {"const_gs": 1, "vect_gs": 3}
D16_N_PARAMS = {"const_gs": 17_771_458, "vect_gs": 17_772_228}


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _load_expected_hashes(bundle_root):
    """Import expected_hashes.py path-relatively.  Search: next to this
    script, notebook_release/build (src layout: ../../build), bundle
    tools/."""
    cands = [
        os.path.join(_HERE, "expected_hashes.py"),
        os.path.join(os.path.dirname(_ROOT), "build", "expected_hashes.py"),
        os.path.join(os.path.dirname(os.path.dirname(_ROOT)), "build",
                     "expected_hashes.py"),
        os.path.join(bundle_root, "tools", "expected_hashes.py"),
        os.path.join(bundle_root, "build", "expected_hashes.py"),
    ]
    for p in cands:
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("expected_hashes",
                                                          p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("[verify] expected_hashes loaded from %s "
                  "(%d d16prog final_state pins, %d PRV-01 d20 pins)"
                  % (p, len(mod.D16PROG_FINAL_STATE),
                     len(mod.PRV01_HASHES)))
            return mod
    raise SystemExit("[verify] expected_hashes.py not found (searched %r); "
                     "it must ship next to its copy_lists/ pull table."
                     % cands)


def _resolve_bundle_root(arg):
    cands = [arg, _ROOT, os.getcwd()]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "checkpoints")):
            return os.path.abspath(c)
    raise SystemExit("[verify] no checkpoints/ under any of %r; pass "
                     "--bundle-root" % [c for c in cands if c])


# ================================================================ quick
def phase_quick(bundle_root, EH):
    ck = os.path.join(bundle_root, "checkpoints")
    rows, n_ok, n_bad, n_unpinned = [], 0, 0, 0
    for dirpath, _dirs, files in sorted(os.walk(ck)):
        for fn in sorted(files):
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ck)
            parts = rel.split(os.sep)
            got = _sha256_file(path)
            want, source = None, None
            if len(parts) == 3 and parts[0] == "d16prog":
                ent = EH.D16PROG_FILES.get((parts[1], parts[2]))
                if ent:
                    want, source = ent[0], "checkpoints_gcs.tsv"
            elif (len(parts) == 3 and parts[0] == "d20"
                  and parts[2] == "final_state.msgpack"):
                want = EH.PRV01_HASHES.get(parts[1])
                source = "PRV-01"
            if want is None:
                status = "UNPINNED"
                n_unpinned += 1
            elif got == want:
                status = "OK"
                n_ok += 1
            else:
                status = "MISMATCH"
                n_bad += 1
            rows.append((rel, status, source or "-", got))
    print("\n[verify --quick] checkpoints/ vs pin tables "
          "(%d files: %d OK, %d MISMATCH, %d unpinned):"
          % (len(rows), n_ok, n_bad, n_unpinned))
    print("  %-55s %-9s %-20s sha256" % ("file", "status", "pin source"))
    for rel, status, source, got in rows:
        print("  %-55s %-9s %-20s %s" % (rel, status, source, got[:16]))

    # bonus: the 3 sealed evaluation-stream files
    sdir = os.path.join(bundle_root, "data", "streams")
    if os.path.isdir(sdir):
        print("\n[verify --quick] data/streams vs PRV-01 stream rows:")
        for fn in sorted(EH.PRV01_STREAM_HASHES):
            p = os.path.join(sdir, fn)
            if not os.path.isfile(p):
                print("  %-45s MISSING" % fn)
                n_bad += 1
                continue
            got = _sha256_file(p)
            ok = got == EH.PRV01_STREAM_HASHES[fn]
            n_ok += ok
            n_bad += (not ok)
            print("  %-45s %-9s %s" % (fn, "OK" if ok else "MISMATCH",
                                       got[:16]))
    print("\n[verify --quick] %s (%d OK / %d MISMATCH / %d unpinned)"
          % ("PASS" if n_bad == 0 else "FAIL", n_ok, n_bad, n_unpinned))
    return n_bad == 0


# ================================================================ full
def phase_full(bundle_root, EH):
    from ognrepro import repro_common as RC
    conv = RC.enable_eval_convention()      # BEFORE the first jax import
    print("\n[verify --full] eval convention armed: %s" % conv)
    import jax
    import numpy as np
    from ognrepro import restore as R

    ck = os.path.join(bundle_root, "checkpoints")
    ref_npz = os.path.join(bundle_root, "data", "percells",
                           "D16PROG-P1-REF",
                           "expc2r2_d16prog_p1_classical_percell.npz")
    p4_npz = os.path.join(bundle_root, "data", "percells",
                          "D16PROG-P4-SCORES",
                          "expc2r2_d16prog_p4_eval_percell.npz")
    zr = np.load(ref_npz)
    zs = np.load(p4_npz)
    refs = {
        "const_gs": dict(
            X=np.asarray(zr["ref_TS_U_1024_rho2"], np.float64)[..., None],
            E=np.asarray(zr["ref_TS_U_1024_energy"], np.float64)[:, None],
            T=np.asarray(zr["ref_TS_U_1024_labels"], np.float64)),
        "vect_gs": dict(
            X=np.asarray(zr["ref_TS_V_512_rho2"], np.float64)[..., None],
            E=np.asarray(zr["ref_TS_V_512_energy"], np.float64)[:, None],
            T=np.asarray(zr["ref_TS_V_512_labels"], np.float64)),
    }

    def _free(*objs):
        R.free(*objs)
        try:
            jax.clear_caches()
        except Exception:                                    # noqa: BLE001
            pass

    all_ok = True
    print("\n[verify --full] 12 d16prog lanes: sealed P1-REF forward vs "
          "sealed P4-SCORES (T2 = %.0e):" % T2)
    print("  %-28s %-8s %-12s %-12s %s"
          % ("lane", "n_par", "maxD(pred)", "maxD(err)", "verdict"))
    for lane in D16_LANES:
        sector = D16_SECTOR[lane]
        d = os.path.join(ck, "d16prog", lane)
        is_std = "std" in lane.split("_")[0]
        mu = sigma = None
        if is_std:
            with np.load(os.path.join(d, "std_stats.npz")) as zz:
                mu, sigma = np.asarray(zz["mu"]), np.asarray(zz["sigma"])
        mdl, _cls = R.build_arm_model(is_std, D16_LABEL_SIZE[sector],
                                      mu, sigma)
        with open(os.path.join(d, "final_state.msgpack"), "rb") as fh:
            raw = fh.read()
        p64, b64, n_par = R.restore_params_f64(mdl, raw, 8)
        np_ok = (n_par == D16_N_PARAMS[sector])
        Rf = refs[sector]
        preds, det = R.forward_f64(mdl, p64, b64, Rf["X"], Rf["E"])
        sealed_pred = np.asarray(zs["scores_pred__%s" % lane], np.float64)
        d_pred = float(np.max(np.abs(preds - sealed_pred)))
        # error streams, recomputed the P4 way [:1262-1269] and compared
        dp = preds - Rf["T"]
        d_err = 0.0
        if sector == "const_gs":
            errs = {"abs_coupling_error": np.abs(dp[:, 0])}
        else:
            l2 = np.linalg.norm(dp, axis=1)
            errs = {"vect_label_l2_error": l2,
                    "vect_label_rms_error": l2 / np.sqrt(dp.shape[1])}
        for mname, ev in errs.items():
            sealed_err = np.asarray(zs["scores_err__%s__%s"
                                       % (mname, lane)], np.float64)
            d_err = max(d_err, float(np.max(np.abs(ev - sealed_err))))
        ok = np_ok and det and d_pred <= T2 and d_err <= T2
        all_ok = all_ok and ok
        print("  %-28s %-8d %-12.3e %-12.3e %s%s"
              % (lane, n_par, d_pred, d_err,
                 "PASS" if ok else "FAIL",
                 "" if np_ok else " [n_params != %d]"
                 % D16_N_PARAMS[sector]))
        _free(mdl, p64, b64, preds)
        mdl = p64 = b64 = preds = None

    print("\n[verify --full] 14 d20-era checkpoints: restore + finite "
          "forward on zeros:")
    d20_dir = os.path.join(ck, "d20")
    for name in sorted(R.D20_MODELS):
        out = R.restore_d20(name, d20_dir)
        m = out["m"]
        X = np.zeros((2, m, m, 1))
        E = np.zeros((2, 1))
        preds, det = R.forward_f64(out["model"], out["params"],
                                   out["batch_stats"], X, E)
        finite = bool(np.isfinite(preds).all())
        ok = finite and det and preds.shape == (2,
                                                out["spec"]["label_size"])
        all_ok = all_ok and ok
        print("  %-28s n_par=%-10d out=%s det=%s finite=%s %s"
              % (name, out["n_params"], preds.shape, det, finite,
                 "PASS" if ok else "FAIL"))
        _free(out, preds)
        out = preds = None
    print("\n[verify --full] %s" % ("PASS" if all_ok else "FAIL"))
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--bundle-root", default=None,
                    help="bundle root holding checkpoints/ and data/ "
                    "(default: this script's parent directory)")
    a = ap.parse_args(argv)
    if not (a.quick or a.full):
        ap.error("pick --quick or --full")
    bundle_root = _resolve_bundle_root(a.bundle_root)
    print("[verify] bundle root: %s" % bundle_root)
    EH = _load_expected_hashes(bundle_root)
    ok = phase_quick(bundle_root, EH)
    if a.full:
        ok = phase_full(bundle_root, EH) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
