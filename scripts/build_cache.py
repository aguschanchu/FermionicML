#!/usr/bin/env python
"""build_cache -- deterministic rebuild of the two certified d16 GS caches.

    python build_cache.py const_gs --labels-dir DIR [--cache-root DIR]
                          [--nproc N] [--selftest]
    python build_cache.py vect_gs  [--cache-root DIR] [--nproc N]
                          [--selftest]

Rebuilds ds_d16n8_const_gs / ds_d16n8_vect_gs with the extracted
ognrepro.blocked numeric core (the pinned d12-stage bodies verbatim),
replicating the certified builds of
experiments/experiment-campaign2-r2/stages/expc2r2_d16prog_p3_cachebuild.py:

  * geometry: splits half0/half1/val = 7813/7813/782 shards x 64 rows
    (cachebuild:105-108), shard names shard_%05d.npz, npz keys
    labels/energy/features (f32; energy (64,1), features (64,8,8,1));
  * targets: exact blocked f64 solve at the beta=100 GS convention
    (p1_core.py:413), h_type const / vect (cachebuild:560-610);
  * label streams (cachebuild:119-135, :344-346, :616-716):
      const_gs REUSES the TPU-minted, digest-gated d14-extract label
        arrays: --labels-dir must hold labels_{half0,half1,val}.npy, each
        gated BIT-EQUAL to the recorded digest before any solve.  A CPU
        replay can NEVER reproduce this stream (1-ULP class difference),
        so there is no mint fallback for a certified const build.
      vect_gs MINTS the cpu-blocked-class stream by the pinned
        replay_labels chain (per-split seeds 42/43/1007, ndev 4, gen_bs
        64, label_size 3, sorted descending), replayed in an ISOLATED
        interpreter (twice, bit-equality required) so this parent process
        never imports jax before forking workers (the documented
        fork-after-jax hazard, cachebuild:688-695).

After the build, per-split labels/features/energy sha256 roll-ups are
computed from disk and verified against the SAME embedded certificate
constants train_lane.py gates on (shared module d16prog_certs); the
certificate-comparison receipt is printed and written next to the cache.

--selftest: 2 shards per split (6 shards, 384 rows), digest-of-selftest
printed with per-shard wall receipts and a full-build extrapolation.  NO
certificate claim is made (prefix digests are not the certificate).
"""
import os

# Single-thread BLAS BEFORE numpy (the builder's own policy; the fork pool
# provides the parallelism) [cachebuild:66-70]
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse           # noqa: E402
import concurrent.futures  # noqa: E402
import json               # noqa: E402
import multiprocessing    # noqa: E402
import subprocess         # noqa: E402
import sys                # noqa: E402
import tempfile           # noqa: E402
import time               # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "lib")):
    sys.path.insert(0, os.path.join(_ROOT, "lib"))
sys.path.insert(0, _HERE)

import numpy as np        # noqa: E402

import d16prog_certs as CERT  # noqa: E402

SELFTEST_SHARDS = 2


# ---------------------------------------------------------------- labels
def _mint_labels_isolated(split, htype, hi=None, timeout_s=3600):
    """Replay the production label chain in a SEPARATE INTERPRETER (mirror
    of cachebuild._mint_labels_isolated:616-655): the child imports jax,
    the parent never does, so the later fork pool is safe."""
    nb = CERT.SPLIT_NB[split]
    hi_eff = nb if hi is None else int(hi)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "labels.npy")
        code = (
            "import os,sys\n"
            "for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',"
            "'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',"
            "'VECLIB_MAXIMUM_THREADS'):\n"
            "    os.environ.setdefault(k,'1')\n"
            "os.environ.setdefault('JAX_PLATFORMS','cpu')\n"
            "sys.path.insert(0,%r)\nsys.path.insert(0,%r)\n"
            "import numpy as np\nimport d16prog_certs as C\n"
            "a=C.mint_labels(%r,%r,hi=%r)\n"
            "np.save(%r, np.asarray(a, np.float32))\n"
            % (os.path.join(_ROOT, "lib"), _HERE, split, htype, hi_eff, out))
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True,
                           timeout=timeout_s)
        if p.returncode != 0 or not os.path.isfile(out):
            raise SystemExit("[build_cache] isolated label mint failed for "
                             "%s (rc=%s)\nstderr: %s"
                             % (split, p.returncode, p.stderr[-2000:]))
        arr = np.asarray(np.load(out), np.float32)
    want = (hi_eff * CERT.GEN_BS, CERT.LABEL_SIZE[htype])
    if arr.shape != want:
        raise SystemExit("[build_cache] mint returned %r, expected %r"
                         % (arr.shape, want))
    return arr


def acquire_labels(sector, split, labels_dir, n_shards):
    """(labels f32, provenance dict).  const: REUSE + digest gate;
    vect: isolated CPU mint, twice, bit-equality required
    [cachebuild acquire_labels:658-716]."""
    htype = CERT.H_TYPE[sector]
    ds_key = CERT.DS_KEYS[sector]
    if sector == "const_gs":
        if not labels_dir:
            raise SystemExit(
                "[build_cache] const_gs REUSES the digest-gated d14-extract "
                "label stream and needs --labels-dir with "
                "labels_{half0,half1,val}.npy.  " + CERT.CONST_LABEL_STREAM_NOTE)
        arr = CERT.load_reuse_labels(labels_dir, split, ds_key)
        prov = dict(mode="REUSE", split=split,
                    path=os.path.join(labels_dir, "labels_%s.npy" % split),
                    array_sha256=CERT.arr_sha(arr), bit_equal=True,
                    stream_class="tpu-extract (d14, digest-gated)")
        return arr[:n_shards * CERT.GEN_BS], prov
    hi = n_shards if n_shards < CERT.SPLIT_NB[split] else None
    lab = _mint_labels_isolated(split, htype, hi=hi)
    lab2 = _mint_labels_isolated(split, htype, hi=hi)
    got, got2 = CERT.arr_sha(lab), CERT.arr_sha(lab2)
    if got != got2:
        raise SystemExit("[build_cache] CPU label replay not reproducible "
                         "on this host (%s != %s)" % (got, got2))
    if lab.shape[1] > 1 and not bool(np.all(np.diff(lab, axis=1) <= 0)):
        raise SystemExit("[build_cache] minted vect rows are not sorted "
                         "DESCENDING (GGenerator convention)")
    full = hi is None
    prov = dict(mode="MINT", split=split, seed=CERT.SPLIT_SEEDS[split],
                n_batches=CERT.SPLIT_NB[split] if full else hi,
                gen_bs=CERT.GEN_BS, ndev=CERT.NDEV,
                label_size=CERT.LABEL_SIZE[htype], htype=htype,
                g_init=CERT.G_INIT, g_stop=CERT.G_STOP,
                array_sha256=got, replay_reproducible=True,
                stream_class="cpu-blocked-class")
    if full:
        want = CERT.LABEL_DIGESTS[ds_key][split]
        if got != want:
            raise SystemExit(
                "[build_cache] minted %s %s label digest %s != certificate "
                "%s -- the replay chain drifted; REFUSING to solve."
                % (ds_key, split, got, want))
        prov["matches_certificate_label_digest"] = True
    return lab, prov


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(htype, beta, labels_by_split, root):
    from ognrepro import blocked as B
    _W["htype"] = htype
    _W["beta"] = float(beta)
    _W["labels"] = labels_by_split
    _W["root"] = root
    _W["eps"] = B.ladder(CERT.D16_M)
    _W["blocks"] = B.build_blocks(CERT.D16_M, CERT.D16_N_ELEC)


def _run_job(job):
    split, lo, hi = job
    return dict(split=split, **CERT.build_shard_range_d16(
        _W["labels"][split], os.path.join(_W["root"], split), lo, hi,
        _W["htype"], _W["blocks"], _W["eps"], beta=_W["beta"]))


def _plan_ranges(nb, nw):
    """Contiguous near-even partition of [0, nb) into <= nw ranges."""
    nw = max(1, min(int(nw), int(nb)))
    step = -(-nb // nw)
    return [(lo, min(lo + step, nb)) for lo in range(0, nb, step)]


def _pmap(jobs, procs, init_args):
    if int(procs) <= 1 or len(jobs) <= 1:
        _init_worker(*init_args)
        return [_run_job(j) for j in jobs]
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(procs), len(jobs)), mp_context=ctx,
            initializer=_init_worker, initargs=init_args) as ex:
        return list(ex.map(_run_job, jobs))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sector", choices=sorted(CERT.DS_KEYS)
                    + sorted(CERT.DS_KEYS.values()),
                    help="const_gs / vect_gs (or the full ds key)")
    ap.add_argument("--cache-root", default="./d16_caches",
                    help="cache lands at <cache-root>/<ds_key>/")
    ap.add_argument("--labels-dir", default=None,
                    help="const_gs: directory holding the digest-gated "
                    "labels_{half0,half1,val}.npy arrays (REQUIRED)")
    ap.add_argument("--nproc", type=int,
                    default=multiprocessing.cpu_count(),
                    help="worker processes (default: cpu_count)")
    ap.add_argument("--selftest", action="store_true",
                    help="%d shards per split, digests printed, NO "
                    "certificate claim" % SELFTEST_SHARDS)
    ap.add_argument("--aux", choices=("psi0",), default=None,
                    help="[V2] after the certificate walk, build the psi0 "
                    "sidecars (shard_%%05d_psi0.npy beside the untouched "
                    "shards; double build; psi0_certificate.json).  If the "
                    "cache already verifies, only the sidecars are built.")
    a = ap.parse_args(argv)

    sector = CERT.SECTOR_OF_KEY.get(a.sector, a.sector)
    ds_key = CERT.DS_KEYS[sector]
    htype = CERT.H_TYPE[sector]
    root = os.path.join(os.path.abspath(a.cache_root),
                        ds_key + ("__selftest" if a.selftest else ""))
    if "results" in os.path.realpath(root).split(os.sep):
        raise SystemExit("[build_cache] REFUSED: --cache-root resolves "
                         "under a **/results/ tree (sealed-record guard)")
    t0 = time.time()
    n_shards = {s: (SELFTEST_SHARDS if a.selftest else CERT.SPLIT_NB[s])
                for s in CERT.CACHE_SPLITS}
    if a.aux == "psi0" and not a.selftest:
        try:
            CERT.verify_cache(os.path.abspath(a.cache_root), ds_key)
            print("[build_cache] %s already certified; building the psi0 "
                  "sidecars only" % ds_key, flush=True)
            return _build_psi0(root, ds_key, htype, n_shards, a.nproc)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            print("[build_cache] cache not (yet) certified (%s); full build "
                  "first" % str(exc).splitlines()[0][:120], flush=True)
    print("[build_cache] %s -> %s  (%s; shards %r; nproc %d)"
          % (ds_key, root, "SELFTEST" if a.selftest else "FULL BUILD",
             n_shards, a.nproc), flush=True)

    # ---- labels first (digest-gated / reproducibility-gated), pre-fork
    labels, prov = {}, {}
    for split in CERT.CACHE_SPLITS:
        labels[split], prov[split] = acquire_labels(sector, split,
                                                    a.labels_dir,
                                                    n_shards[split])
        print("[build_cache] labels %-6s %-5s sha %s.. rows %d"
              % (split, prov[split]["mode"],
                 prov[split]["array_sha256"][:16], labels[split].shape[0]),
              flush=True)

    # ---- shard jobs over the fork pool
    jobs = []
    for split in CERT.CACHE_SPLITS:
        jobs += [(split, lo, hi)
                 for lo, hi in _plan_ranges(n_shards[split], a.nproc)]
    out = _pmap(jobs, a.nproc, (htype, CERT.BETA_GS, labels, root))
    wrote = sum(o["n_written"] for o in out)
    skipped = sum(o["n_skipped"] for o in out)
    rows = sum(o["n_rows"] for o in out)
    nd = max(o["number_identity_dev_max"] for o in out)
    ed = max(o["energy_identity_dev_max"] for o in out)
    wall_cpu = sum(o["wall_s"] for o in out)
    wall = time.time() - t0
    print("[build_cache] solve done: wrote %d shards (%d rows, %d skipped) "
          "in %.1fs wall / %.1fs worker-cpu; number-identity max dev "
          "%.3e (tol 1e-12), energy-identity %.3e (tol 1e-10)"
          % (wrote, rows, skipped, wall, wall_cpu, nd, ed), flush=True)
    if nd > 1e-12 or ed > 1e-10:
        raise SystemExit("[build_cache] IDENTITY GATE FAILED (number %.3e / "
                         "energy %.3e)" % (nd, ed))

    # ---- digest walk + certificate comparison receipt
    label_size = CERT.LABEL_SIZE[htype]
    per = {s: CERT.split_array_digests_streaming(root, s, n_shards[s],
                                                 label_size)
           for s in CERT.CACHE_SPLITS}
    roll = CERT.manifest_digest16(per)
    receipt = dict(ds_key=ds_key, sector=sector, root=os.path.realpath(root),
                   selftest=bool(a.selftest), n_shards=n_shards,
                   per_split_array_digests=per, array_digest16=roll,
                   label_provenance=prov,
                   number_identity_dev_max=nd, energy_identity_dev_max=ed,
                   wall_s=wall, worker_cpu_s=wall_cpu, nproc=int(a.nproc))
    print("\n[build_cache] certificate-comparison receipt (%s):" % ds_key)
    if a.selftest:
        rows_built = sum(n_shards.values()) * CERT.GEN_BS
        per_row_ms = 1000.0 * wall_cpu / max(rows_built, 1)
        full_rows = sum(CERT.SPLIT_ROWS.values())
        est = full_rows * per_row_ms / 1000.0
        receipt["certificate_claim"] = False
        receipt["selftest_note"] = (
            "prefix digests over %d shards/split; NOT the certificate "
            "(which binds the full 16,408-shard walk)" % SELFTEST_SHARDS)
        receipt["extrapolation"] = dict(
            measured_ms_per_row=per_row_ms, full_rows=full_rows,
            est_full_build_cpu_s=est,
            est_full_build_wall_s_at_nproc=est / max(1, a.nproc))
        for s in CERT.CACHE_SPLITS:
            print("  %-6s labels %s..  features %s..  energy %s..  "
                  "(%d rows)" % (s, per[s]["labels_sha256"][:16],
                                 per[s]["features_sha256"][:16],
                                 per[s]["energy_sha256"][:16],
                                 per[s]["n_rows"]))
        print("  selftest digest16 (of the %d-shard prefix): %s -- NO "
              "certificate claim" % (SELFTEST_SHARDS, roll))
        print("  measured %.2f ms/row -> full build (%d rows) est. "
              "%.0f s cpu, ~%.0f s wall at nproc=%d"
              % (per_row_ms, full_rows, est, est / max(1, a.nproc),
                 a.nproc))
        # labels-vs-certificate spot check (prefix rows of REUSE streams
        # are bit-slices of the certified arrays; MINT prefixes replayed
        # from the pinned chain) -- informational only.
    else:
        want = CERT.D16PROG_CACHE_SPLIT_DIGESTS[ds_key]
        ok = True
        for s in CERT.CACHE_SPLITS:
            for f in ("labels_sha256", "features_sha256", "energy_sha256"):
                match = per[s][f] == want[s][f]
                ok = ok and match
                print("  %-6s %-16s %s  %s" % (s, f, per[s][f][:16] + "..",
                                               "MATCH" if match
                                               else "MISMATCH (want %s..)"
                                               % want[s][f][:16]))
        roll_ok = roll == CERT.D16PROG_CACHE_DIGESTS[ds_key]
        print("  roll-up digest16 %s  %s (certificate %s)"
              % (roll, "MATCH" if roll_ok else "MISMATCH",
                 CERT.D16PROG_CACHE_DIGESTS[ds_key]))
        receipt["matches_certificate"] = bool(ok and roll_ok)
        if not (ok and roll_ok):
            _write_receipt(root, receipt)
            raise SystemExit(
                "[build_cache] CERTIFICATE COMPARISON FAILED for %s -- the "
                "rebuilt cache is NOT the certified cache; do not train on "
                "it.  (const: wrong/missing label arrays?  vect: stream or "
                "solver drift?)" % ds_key)
        print("  => rebuilt cache is BIT-IDENTICAL to the certified build "
              "(labels+features+energy, all splits)")
    _write_receipt(root, receipt)
    print("[build_cache] receipt written: %s"
          % os.path.join(root, "build_cache_receipt.json"))
    if a.aux == "psi0":
        return _build_psi0(root, ds_key, htype, n_shards, a.nproc,
                           selftest=a.selftest)
    return 0


def _build_psi0(root, ds_key, htype, n_shards, nproc, selftest=False):
    """[V2] psi0 sidecars for every split of an existing cache (double build
    + certificate; ognrepro.v2.d16_aux)."""
    from ognrepro.v2 import d16_aux
    t0 = time.time()
    cert = d16_aux.build_psi0_cache(root, htype, dict(n_shards), nproc=nproc,
                                    double=True, ds_key=ds_key)
    print("[build_cache] psi0 sidecars %s in %.0fs (double build %s):"
          % (ds_key, time.time() - t0,
             "AGREES" if cert.get("run_b_equal") else "n/a"))
    for s_, d in cert["per_split"].items():
        print("  %-6s psi0 %s..  rows %d  |E-e0|max %.2e  gap_min %.3e"
              % (s_, d["psi0_sha256"][:16], d["n_rows"], d["e0_dev_max"],
                 d["gap_min"]))
    if selftest:
        print("  (selftest prefix: NO certificate claim)")
    else:
        print("  certificate: %s" % os.path.join(root, "psi0_certificate.json"))
        want = CERT.PSI0_DIGESTS.get(ds_key) or {}
        for s_, d in cert["per_split"].items():
            if want.get(s_):
                print("  %-6s vs transcribed constant: %s"
                      % (s_, "MATCH" if want[s_] == d["psi0_sha256"]
                         else "MISMATCH"))
    return 0


def _write_receipt(root, receipt):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "build_cache_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
