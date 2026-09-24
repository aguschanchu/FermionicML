"""ognrepro.v2.d16_aux -- the psi0 sidecar of the certified d16 GS caches.

The certified caches (ds_d16n8_{const,vect}_gs; d16prog_certs) are never
modified: for every shard_%05d.npz a sibling shard_%05d_psi0.npy (64, 70) f32
holds the seniority-zero ground state of each row, computed numpy-only from
the row's LABELS with the pinned blocked solver bodies (ognrepro.blocked:
v_matrix + FreeSetBlock(8, 8, ()).h_batch, 70x70 eigh).  Per row the cache's
stored energy must satisfy |E - e0| < E0_TOL (the beta=100 state IS the
ground state); a violation is a hard refusal.  Sign rule = gen.py's device
kernel: the largest-|component| entry is made positive.

Certificate: build_psi0_cache runs the build TWICE (run B into a scratch
directory) and issues <root>/psi0_certificate.json only when the per-split
sha256 digests of the two runs agree; verify_psi0 re-walks the digests
against the certificate (and against the transcribed constants when given).
"""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import time

import numpy as np

from .. import blocked as B

D16_M, D16_N = 8, 8
S0_DIM = 70
GEN_BS = 64
E0_TOL = 1e-4
PSI0_SUFFIX = "_psi0.npy"
BETA_GS = 100.0


def s0_block(m=D16_M, n=D16_N):
    return B.FreeSetBlock(m, n, ())


def sidecar_path(shard_path):
    assert shard_path.endswith(".npz"), shard_path
    return shard_path[:-4] + PSI0_SUFFIX


def psi0_rows(labels, htype, blk=None, eps_lvl=None, m=D16_M):
    """(psi0 (B, dim) f64 sign-fixed, e0 (B,), gap (B,)) from the v=0 block."""
    blk = blk or s0_block(m, D16_N)
    eps_lvl = B.ladder(m) if eps_lvl is None else eps_lvl
    V = np.stack([B.v_matrix(htype, m, row) for row in np.asarray(labels)])
    H = blk.h_batch(eps_lvl, V, 0.0)
    vals, vecs = np.linalg.eigh(H)
    psi = vecs[:, :, 0]
    amax = np.argmax(np.abs(psi), axis=1)
    sgn = np.sign(psi[np.arange(len(psi)), amax])
    sgn[sgn == 0] = 1.0
    return psi * sgn[:, None], vals[:, 0], vals[:, 1] - vals[:, 0]


def _atomic_save(path, arr):
    tmp = path + ".PROVISIONAL"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def build_psi0_range(split_dir, lo, hi, htype, out_dir=None, tol=E0_TOL):
    """Shards [lo, hi) of one split -> psi0 sidecars (in out_dir if given).
    Existing well-formed sidecars in the target directory are kept."""
    blk = s0_block()
    eps = B.ladder(D16_M)
    out_dir = out_dir or split_dir
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_rows = n_written = n_skipped = 0
    dev_max, gap_min = 0.0, np.inf
    worst = None
    for i in range(int(lo), int(hi)):
        p = os.path.join(split_dir, B.shard_name(i))
        out = sidecar_path(os.path.join(out_dir, B.shard_name(i)))
        with np.load(p) as z:
            lab = np.asarray(z["labels"])
            E = np.asarray(z["energy"], np.float64).reshape(-1)
        psi, e0, gap = psi0_rows(lab, htype, blk, eps)
        dev = np.abs(E - e0)
        j = int(np.argmax(dev))
        if dev[j] > dev_max:
            dev_max, worst = float(dev[j]), dict(shard=i, row=j, E=float(E[j]), e0=float(e0[j]), gap=float(gap[j]))
        gap_min = min(gap_min, float(gap.min()))
        if dev[j] >= tol:
            raise RuntimeError("[d16_aux] |E - e0| = %.3e >= %.1e at %s row %d (gap %.3e): the stored "
                               "beta=100 state is not the seniority-zero ground state"
                               % (dev[j], tol, p, j, gap[j]))
        if os.path.isfile(out):
            try:
                if np.load(out, mmap_mode="r").shape == psi.shape:
                    n_skipped += 1
                    n_rows += len(psi)
                    continue
            except Exception:  # noqa: BLE001
                pass
        _atomic_save(out, psi.astype(np.float32))
        n_written += 1
        n_rows += len(psi)
    return dict(lo=int(lo), hi=int(hi), n_rows=int(n_rows), n_written=int(n_written),
                n_skipped=int(n_skipped), e0_dev_max=float(dev_max), gap_min=float(gap_min),
                worst=worst, wall_s=time.time() - t0)


def _plan_ranges(n, k):
    k = max(1, min(int(k), int(n)))
    step = -(-n // k)
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


def _job(args):
    split_dir, lo, hi, htype, out_dir = args
    return build_psi0_range(split_dir, lo, hi, htype, out_dir=out_dir)


def psi0_digest_streaming(sidecar_dir, n_shards):
    h = hashlib.sha256()
    n_rows = 0
    for i in range(int(n_shards)):
        p = sidecar_path(os.path.join(sidecar_dir, B.shard_name(i)))
        a = np.load(p)
        if a.ndim != 2 or a.shape[1] != S0_DIM or a.dtype != np.float32:
            raise RuntimeError("[d16_aux] sidecar %s malformed %r %s" % (p, a.shape, a.dtype))
        h.update(np.ascontiguousarray(a).tobytes())
        n_rows += a.shape[0]
    return dict(psi0_sha256=h.hexdigest(), n_rows=int(n_rows), n_shards=int(n_shards))


def build_psi0_cache(root, htype, splits, nproc=None, double=True, verbose=True, ds_key=None):
    """splits: {split: n_shards}.  Builds the sidecars beside the shards, then
    (double=True) rebuilds into <root>/<split>/.psi0_runB and requires digest
    equality; writes <root>/psi0_certificate.json.  Returns the certificate."""
    nproc = nproc or multiprocessing.cpu_count()
    t0 = time.time()
    cert = dict(ds_key=ds_key, root=os.path.realpath(root), htype=htype, e0_tol=E0_TOL,
                sign_rule="largest |component| positive (gen.py kernel)", basis="FreeSetBlock(8,8,()) cfgs",
                per_split={}, double_build=bool(double), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    runs = ["A", "B"] if double else ["A"]
    digests = {r: {} for r in runs}
    stats = {}
    for run in runs:
        jobs = []
        for split, n_sh in splits.items():
            sd = os.path.join(root, split)
            od = None if run == "A" else os.path.join(sd, ".psi0_runB")
            jobs += [(sd, lo, hi, htype, od) for lo, hi in _plan_ranges(int(n_sh), nproc)]
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as ex:
            outs = list(ex.map(_job, jobs))
        for split, n_sh in splits.items():
            sd = os.path.join(root, split)
            od = sd if run == "A" else os.path.join(sd, ".psi0_runB")
            digests[run][split] = psi0_digest_streaming(od, n_sh)
            if run == "A":
                mine = [o for o, j in zip(outs, jobs) if j[0] == sd]
                stats[split] = dict(e0_dev_max=max(o["e0_dev_max"] for o in mine),
                                    gap_min=min(o["gap_min"] for o in mine),
                                    n_written=sum(o["n_written"] for o in mine),
                                    n_skipped=sum(o["n_skipped"] for o in mine),
                                    worker_cpu_s=sum(o["wall_s"] for o in mine))
        if verbose:
            print("[d16_aux] run %s done (%.0fs): %s" % (run, time.time() - t0,
                  {s: d["psi0_sha256"][:16] for s, d in digests[run].items()}), flush=True)
    for split in splits:
        cert["per_split"][split] = dict(digests["A"][split], **stats[split])
    if double:
        eq = all(digests["A"][s]["psi0_sha256"] == digests["B"][s]["psi0_sha256"] for s in splits)
        cert["run_b_equal"] = bool(eq)
        for split in splits:
            shutil.rmtree(os.path.join(root, split, ".psi0_runB"), ignore_errors=True)
        if not eq:
            raise RuntimeError("[d16_aux] DOUBLE BUILD DISAGREES: %r vs %r" % (digests["A"], digests["B"]))
    cert["wall_s"] = time.time() - t0
    with open(os.path.join(root, "psi0_certificate.json"), "w") as fh:
        json.dump(cert, fh, indent=1)
    return cert


def verify_psi0(root, splits, want=None, verbose=True):
    """Re-walk the sidecar digests of <root>/<split> and compare with the
    cache's psi0_certificate.json (and the transcribed constants `want`
    {split: sha256} when given).  Raises on any mismatch."""
    cp = os.path.join(root, "psi0_certificate.json")
    if not os.path.isfile(cp):
        raise RuntimeError("[d16_aux] no psi0_certificate.json under %s (build_cache.py --aux psi0)" % root)
    cert = json.load(open(cp))
    if not cert.get("run_b_equal", not cert.get("double_build")):
        raise RuntimeError("[d16_aux] certificate %s was not issued by an agreeing double build" % cp)
    got, bad = {}, []
    for split, n_sh in splits.items():
        g = psi0_digest_streaming(os.path.join(root, split), n_sh)
        got[split] = g
        c = cert["per_split"].get(split)
        if c is None or c["psi0_sha256"] != g["psi0_sha256"] or int(c["n_rows"]) != g["n_rows"]:
            bad.append("%s: got %s/%d, certificate %s" % (split, g["psi0_sha256"][:16], g["n_rows"],
                                                          (c or {}).get("psi0_sha256", "?")[:16]))
        if want and want.get(split) and want[split] != g["psi0_sha256"]:
            bad.append("%s: got %s != transcribed %s" % (split, g["psi0_sha256"][:16], want[split][:16]))
    if bad:
        raise RuntimeError("[d16_aux] PSI0 CERTIFICATE GATE FAILED under %s:\n  %s" % (root, "\n  ".join(bad)))
    if verbose:
        print("[d16_aux] psi0 sidecars match the certificate%s: %s"
              % (" + transcribed constants" if want else "", {s: g["psi0_sha256"][:16] for s, g in got.items()}))
    return dict(root=os.path.realpath(root), per_split=got, matches_certificate=True,
                matches_transcribed=bool(want), e0_tol=cert.get("e0_tol"))
