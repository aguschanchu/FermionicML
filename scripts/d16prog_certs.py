"""d16prog_certs -- shared d16prog cache-certificate constants + helpers.

Used by BOTH train_lane.py (pre-training cache gate) and build_cache.py
(post-build certificate comparison), so the two scripts can never disagree
about what the certified caches look like.

PROVENANCE (transcription date 2026-08-23; every value copied, none derived)
  * D16PROG_CACHE_SPLIT_DIGESTS: transcribed from
    experiments/experiment-campaign2-r2/stages/t_d16prog_lanes.py
    D16PROG_CREATED_CACHE_SPLIT_DIGESTS lines 437-503 (const_gs block
    :438-463, vect_gs block :478-503), itself transcribed from the two build
    certificates; re-checked here against the certificates directly:
      results/D16PROG-P3-GS-CONST/ds_d16n8_const_gs_cache_certificate.json
        (double build, runA == runB, 2026-08-07; "per_split" block)
      results/T-d16progorig_vect_gs/T-d16progorig_vect_gs.json
        cache_certificate.per_split_array_digests (lane full-walk record of
        results/D16PROG-P3-VECT-GS's certificate; roll-up ce920a840b2ea511)
  * D16PROG_CACHE_DIGESTS (16-hex roll-ups): t_d16prog_lanes.py:523-529
    ("a44b60ad9412b08c" const / "ce920a840b2ea511" vect); derived from the
    per-split blocks by manifest_digest16 below and CROSS-CHECKED AT IMPORT
    (_SELFCHECK), so the two constants can never disagree silently.
  * manifest_digest16: byte-for-byte
    expc2r2_d16prog_p3_cachebuild.py manifest_digest16 (:817) followed by
    the [:16] truncation applied at its only call site (:1472).
  * SPLIT_SEEDS / SPLIT_NB / SPLIT_ROWS / GEN_BS / NDEV / G_INIT / G_STOP:
    expc2r2_d16prog_p3_cachebuild.py:101-116 and :344-346.
  * Label-stream facts: cachebuild :119-135 (const REUSES the digest-gated
    d14 TPU-extract labels_<split>.npy arrays; a CPU replay is a DIFFERENT,
    disclosed stream class -- MATERIAL FINDING 1, 26.4% of rows 1 ULP off)
    and :344 + t_d16prog_lanes.py:577-596 (vect MINTS the cpu-blocked-class
    stream by replay_labels at seeds 42/43/1007, ndev 4, gen_bs 64).
  * Gram-metric pins: t_d16prog_lanes.py:369-376 (file sha / array sha /
    shape / dtype of recon/d16_prototype/adversarial/d16_gram_S_64x64.npy).
  * solve/shard-writer bodies: mirrors of
    expc2r2_d16prog_p3_cachebuild.solve_rows (:588-610) and
    build_shard_range_p3 (:725-757), re-expressed over the extracted
    ognrepro.blocked primitives (v_matrix / blocked_solve / save_shard /
    shard_name / shard_wellformed are the pinned d12-stage bodies verbatim;
    ognrepro.blocked.build_shard_range itself asserts the d12 basis and is
    deliberately NOT reused here -- see its header note).
"""
import hashlib
import json
import os

import numpy as np

# ---------------------------------------------------------------- era pins
# [cachebuild:101-116]
D16_D_SP, D16_N_ELEC, D16_M = 16, 8, 8
D16_BASIS = 12870
GEN_BS = 64
NDEV = 4
G_INIT, G_STOP = 0.1, 1.0
BETA_GS = 100.0
CACHE_SPLITS = ("half0", "half1", "val")
SPLIT_NB = {"half0": 7813, "half1": 7813, "val": 782}
SPLIT_ROWS = {"half0": 500032, "half1": 500032, "val": 50048}
SPLIT_SEEDS = {"half0": 42, "half1": 43, "val": 1007}     # [cachebuild:344]
LABEL_SIZE = {"const": 1, "vect": 3}

DS_KEYS = {"const_gs": "ds_d16n8_const_gs", "vect_gs": "ds_d16n8_vect_gs"}
SECTOR_OF_KEY = {v: k for k, v in DS_KEYS.items()}
H_TYPE = {"const_gs": "const", "vect_gs": "vect"}

# ------------------------------------------------- certificate constants
# [t_d16prog_lanes.py:437-503; re-checked against the build certificates]
D16PROG_CACHE_SPLIT_DIGESTS = {
    "ds_d16n8_const_gs": {
        "half0": {
            "labels_sha256":
                "317983939a0a0f7d9cd8b60aebdba3b12475d3d725010e4c488b31aaf944858c",
            "features_sha256":
                "98476cf122edc07ebaf5893e796ebebc7b91611d12e9a3acc3631c56ee0ef564",
            "energy_sha256":
                "474c902f4953f7d5eaeec13b518ac8635682a938b02f06b5afc46fd1af0f4f05",
            "n_rows": 500032, "n_shards": 7813},
        "half1": {
            "labels_sha256":
                "23d3879cd5ace14a1f9cc8d2f2e9ecc03a8ac4e32595e7fda9305b22809fae40",
            "features_sha256":
                "57f285c2b05bea57bb51330b47c52fcab6d9bdf58f6487d357f7bb8ce8860950",
            "energy_sha256":
                "3dafd29c085bc95a72eece18e3eb8c65c5ab7d0274defee2207cdbf872a9bb49",
            "n_rows": 500032, "n_shards": 7813},
        "val": {
            "labels_sha256":
                "8e74fe348de1f21a840863f9076da2ecdcfbb9a7e24ab2e41955992df520a82c",
            "features_sha256":
                "c60192465db8ba064592cc9f50f1fcf0736d8379c9cb143894c7b97280b49d48",
            "energy_sha256":
                "cddbb972ad2ed11a87859d2b5c2c46985e2b2e241149216ec5c2e0c1cebf2da3",
            "n_rows": 50048, "n_shards": 782},
    },
    "ds_d16n8_vect_gs": {
        "half0": {
            "labels_sha256":
                "14995f99812c002c2a10279087b1224d211c8bd2f7f51c363cc4e432c51d61e0",
            "features_sha256":
                "a24b322b7c74ea6b24d78bc1bb350873069ab65b1fd0b3fcdf889dd10b0ad205",
            "energy_sha256":
                "32123b872bd65526195e58a7483d41ffcef437f5707960c32c05ffb3e7fdeece",
            "n_rows": 500032, "n_shards": 7813},
        "half1": {
            "labels_sha256":
                "cbd90a92c0cf8861888ea21cdc485e7d924772ee113e7a5589b52a9601134341",
            "features_sha256":
                "8107cd5175d2039e48dd9afa15da9350feda4d82447ed6b35b838f56be42d355",
            "energy_sha256":
                "d178e7b0810118a06e35b152b0371d83e9221c47136e5fe500f365301548a3ad",
            "n_rows": 500032, "n_shards": 7813},
        "val": {
            "labels_sha256":
                "b1df5f638267d82a0c1925825c1ab9b81f5e60d859a975ad41d119069bb2f477",
            "features_sha256":
                "0f163257c757d3b43cfa79ce43c404f54da1af432fcab1302d0f80d169d739db",
            "energy_sha256":
                "ee8e802c127c0eadff3b09a271bc7a141355c314ee9e8f29ee2cae180b6bab4a",
            "n_rows": 50048, "n_shards": 782},
    },
}
# 16-hex roll-ups [t_d16prog_lanes.py:523-529] -- re-derived at import below.
D16PROG_CACHE_DIGESTS = {
    "ds_d16n8_const_gs": "a44b60ad9412b08c",
    "ds_d16n8_vect_gs": "ce920a840b2ea511",
}

# Per-split label digests per sector (the label-stream identity binding).
# const == the d14 TPU-extract stream; vect == the cpu-blocked-class mint.
LABEL_DIGESTS = {
    k: {s: d[s]["labels_sha256"] for s in CACHE_SPLITS}
    for k, d in D16PROG_CACHE_SPLIT_DIGESTS.items()}

CONST_LABEL_STREAM_NOTE = (
    "ds_d16n8_const_gs REUSES the TPU-minted, digest-gated d14-extract "
    "label arrays (labels_<split>.npy).  A CPU replay of the same chain is "
    "a DIFFERENT, disclosed stream class (MATERIAL FINDING 1: "
    "jax.random.uniform's affine map rounds differently on CPU; 26.4% of "
    "rows exactly 1 ULP apart) and can NEVER reproduce the certificate "
    "digests.  A certified const rebuild therefore requires the label "
    "arrays themselves (--labels-dir).")
VECT_LABEL_STREAM_NOTE = (
    "ds_d16n8_vect_gs labels are MINTED on CPU by the pinned replay_labels "
    "chain (per-split seeds 42/43/1007, ndev 4, gen_bs 64, label_size 3, "
    "h_type vect, sorted descending) -- the cpu-blocked-class stream of "
    "chair ruling E-P3-A; a faithful CPU replay IS the certified stream.")

# ------------------------------------------------- gram-metric pins
# [t_d16prog_lanes.py:369-376]
D16_GRAM_S_FILE_SHA256 = (
    "bece0615be0d437ce14e0848515956dec87dae759e8c38c1563ba989a8a31623")
D16_GRAM_S_ARRAY_SHA256 = (
    "733219c449e518a8b59ba88cb21e2c41686921bdd52086a189449742f534230a")
D16_GRAM_S_SHAPE = (64, 64)
D16_GRAM_S_DTYPE = "float64"
D16_GRAM_S_BASENAME = "d16_gram_S_64x64.npy"
D16_GRAM_S_RESEARCH_REL = os.path.join(
    "experiments", "experiment-campaign2-r2", "recon", "d16_prototype",
    "adversarial", D16_GRAM_S_BASENAME)


# ================================================================ helpers
def arr_sha(a):
    """expc2r2_d12_blocked_validation._arr_sha (:255-257), verbatim body."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def manifest_digest16(per_split):
    """THE P3 CACHE-IDENTITY RECIPE, byte-for-byte cachebuild
    manifest_digest16 (:817-822) + the [:16] truncation of its only call
    site (:1472): sha256 over the sorted per-split map of the three
    *_sha256 fields (non-sha keys dropped exactly as the builder drops
    them)."""
    blob = json.dumps({s: {k: v for k, v in d.items() if k.endswith("sha256")}
                       for s, d in per_split.items()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def split_array_digests_streaming(root, split, n_shards, label_size,
                                  m=D16_M, gen_bs=GEN_BS):
    """The builder's split_array_digests (whole-split array-payload sha256
    per field, shard order), computed WITHOUT holding the split in memory.
    For C-order arrays the bytes of the axis-0 concatenation ARE the
    concatenation of each part's bytes, so the streaming update is
    bit-identical (t_d16prog_lanes._split_array_digests_streaming
    :3091-3121 proves the equality on a fixture in its heavy selftest).
    Also enforces the builder's shard well-formedness predicate."""
    from ognrepro import blocked as B
    hs = {f: hashlib.sha256() for f in ("labels", "features", "energy")}
    n_rows = 0
    d = os.path.join(root, split)
    for i in range(int(n_shards)):
        p = os.path.join(d, B.shard_name(i))
        if not B.shard_wellformed(p, gen_bs, int(label_size), int(m)):
            raise RuntimeError(
                "[d16prog_certs] shard %s missing or not well-formed for "
                "label_size=%d / m=%d (the builder's own predicate); a "
                "wrong-sector or corrupt cache is a hard refusal."
                % (p, label_size, m))
        with np.load(p) as z:
            for f in ("labels", "features", "energy"):
                hs[f].update(np.ascontiguousarray(np.asarray(z[f])).tobytes())
            n_rows += int(np.asarray(z["labels"]).shape[0])
    return {"labels_sha256": hs["labels"].hexdigest(),
            "features_sha256": hs["features"].hexdigest(),
            "energy_sha256": hs["energy"].hexdigest(),
            "n_shards": int(n_shards), "n_rows": int(n_rows)}


def verify_cache(cache_root, ds_key, verbose=True):
    """Full walk of <cache_root>/<ds_key>/{half0,half1,val} against the
    embedded certificate constants.  Returns the receipt dict on success;
    raises RuntimeError on ANY mismatch (train_lane refuses to train)."""
    want = D16PROG_CACHE_SPLIT_DIGESTS[ds_key]
    sector = SECTOR_OF_KEY[ds_key]
    label_size = LABEL_SIZE[H_TYPE[sector]]
    root = os.path.join(cache_root, ds_key)
    per, bad = {}, []
    for split in CACHE_SPLITS:
        got = split_array_digests_streaming(root, split,
                                            want[split]["n_shards"],
                                            label_size)
        per[split] = got
        for f in ("labels_sha256", "features_sha256", "energy_sha256",
                  "n_rows", "n_shards"):
            if got[f] != want[split][f]:
                bad.append("%s.%s: got %r != certificate %r"
                           % (split, f, got[f], want[split][f]))
    roll = manifest_digest16(per)
    if roll != D16PROG_CACHE_DIGESTS[ds_key]:
        bad.append("roll-up: got %s != certificate %s"
                   % (roll, D16PROG_CACHE_DIGESTS[ds_key]))
    if bad:
        raise RuntimeError(
            "[d16prog_certs] CACHE CERTIFICATE GATE FAILED for %s at %s:\n"
            "  %s\nREFUSING to proceed (rebuild the cache with "
            "build_cache.py or fix --cache-dir)." % (ds_key, root,
                                                     "\n  ".join(bad)))
    receipt = {"ds_key": ds_key, "root": os.path.realpath(root),
               "per_split_array_digests": per, "array_digest16": roll,
               "matches_certificate": True,
               "certificate_sources": [
                   "t_d16prog_lanes.py:437-529 (transcribed constants)",
                   "D16PROG-P3-GS-CONST/ds_d16n8_const_gs_cache_"
                   "certificate.json",
                   "T-d16progorig_vect_gs cache_certificate "
                   "(roll-up ce920a840b2ea511)"]}
    if verbose:
        print("[d16prog_certs] cache %s: array digest16 %s == certificate; "
              "all per-split labels/features/energy digests match"
              % (ds_key, roll))
    return receipt


# ------------------------------------------------- [V2] psi0 sidecar certificate
# Per-split sha256 of the psi0 sidecars (shard_%05d_psi0.npy, (64,70) f32,
# shard order), transcribed from the first agreeing double build
# (ognrepro.v2.d16_aux.build_psi0_cache).  Empty until issued: verify_cache_v2
# then gates on the cache's own psi0_certificate.json only.
PSI0_DIGESTS = {
    # issued 2026-09-09 on tpu-v4-ondemand-3 (agreeing double build, 132 s, |E-e0|max 9.54e-7, gap_min 2.005);
    # cross-checked against the workstation double build
    "ds_d16n8_const_gs": {
        "half0": "56f22c42b6c059eb3cc767a9b5520da996456b27334a9ea66a0580126f2f1cf5",
        "half1": "57c1333a40aa7a46e366d67581282ed73026139cb653de032bdd5e1baacf08ce",
        "val":   "98a686fb30efb7d0494b27159dcbd6db9a918c125b4c8e6bb3f4d415a7491499",
    },
    # issued 2026-09-09 (workstation, agreeing double build, |E-e0|max 9.54e-7, gap_min 2.003; == od-3 build)
    "ds_d16n8_vect_gs": {
        "half0": "5be7507380834dbc142553953ca32a943be0770f817892da7bd18b609dfb0b65",
        "half1": "44bb99afeb64bf2340ee3e2fbb2971678c95445cdf3e14e742d79f7fa8ca4e8a",
        "val":   "4da6e5c7ad2debf4ca5aa2640c7de6895342b8f083a9683acd76f00e8f24e61c",
    },
}


def verify_cache_v2(cache_root, ds_key, verbose=True):
    """[V2] the published 3-array certificate walk (verify_cache, unchanged
    constants) + the psi0 sidecar digests (double-build certificate and, once
    transcribed, PSI0_DIGESTS).  Returns the combined receipt."""
    from ognrepro.v2 import d16_aux
    rec = verify_cache(cache_root, ds_key, verbose=verbose)
    want = {k: v for k, v in PSI0_DIGESTS.get(ds_key, {}).items() if v}
    rec["psi0"] = d16_aux.verify_psi0(os.path.join(cache_root, ds_key), dict(SPLIT_NB),
                                      want=(want or None), verbose=verbose)
    rec["schema"] = "v2: labels/features/energy certificate + psi0 sidecar certificate"
    return rec


# ------------------------------------------------- label acquisition
def load_reuse_labels(labels_dir, split, ds_key="ds_d16n8_const_gs"):
    """REUSE mode (const sector): read labels_<split>.npy and gate it
    BIT-EQUAL to the recorded digest [cachebuild acquire_labels:671-687]."""
    path = os.path.join(labels_dir, "labels_%s.npy" % split)
    arr = np.asarray(np.load(path), np.float32)
    got = arr_sha(arr)
    want = LABEL_DIGESTS[ds_key][split]
    if got != want:
        raise RuntimeError(
            "[d16prog_certs] REFUSED: %s array digest %s != the recorded "
            "d14-extract digest %s.  %s" % (path, got, want,
                                            CONST_LABEL_STREAM_NOTE))
    if arr.shape != (SPLIT_ROWS[split], 1):
        raise RuntimeError("[d16prog_certs] %s shape %r != %r"
                           % (path, arr.shape, (SPLIT_ROWS[split], 1)))
    return arr


def mint_labels(split, htype, n_batches=None, hi=None):
    """MINT mode: replay the production label chain via the PINNED
    ognrepro.blocked.replay_labels (seeds 42/43/1007, ndev 4, gen_bs 64
    [cachebuild:344-346]).  `hi` limits to the first `hi` batches (a strict
    prefix of the canonical stream -- the key chain is identical).
    NOTE: imports jax; do not call before forking worker processes."""
    from ognrepro import blocked as B
    nb = SPLIT_NB[split] if n_batches is None else int(n_batches)
    return np.asarray(
        B.replay_labels(SPLIT_SEEDS[split], nb, gen_bs=GEN_BS, ndev=NDEV,
                        label_size=LABEL_SIZE[htype], htype=htype,
                        g_init=G_INIT, g_stop=G_STOP, lo=0, hi=hi),
        np.float32)


# ------------------------------------------------- d16 solve + shard write
def solve_rows_d16(labels, htype, blocks, eps_lvl, beta=BETA_GS,
                   m=D16_M, chunk=64):
    """cachebuild solve_rows (:588-610) over the ognrepro.blocked
    primitives: blocked f64 solve of a label block -> (f2, energy) with the
    whole-cache number/energy identities accumulated and the d16 state
    accounting asserted (12,870 -- NOT the d12 assert baked into
    ognrepro.blocked.build_shard_range, which is why that function is not
    reused here)."""
    from ognrepro import blocked as B
    n = labels.shape[0]
    f2 = np.empty((n, m, m), np.float64)
    en = np.empty((n,), np.float64)
    num_dev, en_dev = 0.0, 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        V = np.stack([B.v_matrix(htype, m, row) for row in labels[s:e]])
        r2, ee, _z, _e0, nst, sg = B.blocked_solve(blocks, eps_lvl, V, m,
                                                   beta)
        if nst != D16_BASIS:
            raise RuntimeError("[d16prog_certs] state accounting %d != %d"
                               % (nst, D16_BASIS))
        num_dev = max(num_dev, float(np.max(np.abs(
            B.number_identity(r2, sg) - D16_N_ELEC))))
        en_dev = max(en_dev, float(np.max(np.abs(
            B.energy_identity(r2, sg, eps_lvl, V) - ee))))
        f2[s:e] = r2
        en[s:e] = ee
    return f2, en, num_dev, en_dev


def build_shard_range_d16(labels, split_dir, lo, hi, htype, blocks, eps_lvl,
                          beta=BETA_GS, gen_bs=GEN_BS):
    """cachebuild build_shard_range_p3 (:725-757) over ognrepro.blocked
    save_shard/shard_name/shard_wellformed: compute and write shards
    [lo, hi) of one split; shard i holds label rows [i*gen_bs,(i+1)*gen_bs).
    Existing well-formed shards are never rewritten."""
    import time
    from ognrepro import blocked as B
    os.makedirs(split_dir, exist_ok=True)
    written = skipped = rows = 0
    num_dev = en_dev = 0.0
    t0 = time.time()
    for i in range(int(lo), int(hi)):
        path = os.path.join(split_dir, B.shard_name(i))
        if os.path.exists(path) and B.shard_wellformed(
                path, gen_bs, labels.shape[1], D16_M):
            skipped += 1
            continue
        lab = np.ascontiguousarray(labels[i * gen_bs:(i + 1) * gen_bs])
        assert lab.shape[0] == gen_bs, lab.shape
        f2, en, nd, ed = solve_rows_d16(lab, htype, blocks, eps_lvl, beta)
        num_dev = max(num_dev, nd)
        en_dev = max(en_dev, ed)
        B.save_shard(path, lab.astype(np.float32),
                     en.astype(np.float32)[:, None],
                     f2.astype(np.float32)[..., None])
        written += 1
        rows += gen_bs
    return dict(lo=int(lo), hi=int(hi), n_written=int(written),
                n_skipped=int(skipped), n_rows=int(rows),
                number_identity_dev_max=float(num_dev),
                energy_identity_dev_max=float(en_dev),
                wall_s=time.time() - t0)


# ------------------------------------------------- gram-S artifact loader
def find_gram_s(bundle_root=None, explicit=None):
    """Locate the pinned d16 gram-metric artifact d16_gram_S_64x64.npy.
    Search order: explicit path, $OGNREPRO_D16_GRAM_S, bundle data/gram/,
    bundle data/, then the research-repo path walking up from bundle_root.
    Returns the first existing candidate (identity is gated by
    load_gram_s, not here)."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("OGNREPRO_D16_GRAM_S")
    if env:
        cands.append(env)
    if bundle_root:
        cands.append(os.path.join(bundle_root, "data", "gram",
                                  D16_GRAM_S_BASENAME))
        cands.append(os.path.join(bundle_root, "data", D16_GRAM_S_BASENAME))
        d = os.path.abspath(bundle_root)
        for _ in range(4):
            d = os.path.dirname(d)
            cands.append(os.path.join(d, D16_GRAM_S_RESEARCH_REL))
    for p in cands:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "[d16prog_certs] d16 gram-metric artifact %s not found (searched "
        "%r).  Pass --gram-s / set OGNREPRO_D16_GRAM_S, or place the "
        "artifact under <bundle>/data/gram/." % (D16_GRAM_S_BASENAME, cands))


def load_gram_s(path):
    """Load + gate the 64x64 f64 gram metric S on BOTH pinned identities
    (file sha of the .npy bytes AND array sha of the contiguous payload --
    t_d16prog_lanes.py:366-375's dual-hash convention)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    fsha = hashlib.sha256(raw).hexdigest()
    S = np.load(path)
    asha = arr_sha(S)
    if (S.shape != D16_GRAM_S_SHAPE or str(S.dtype) != D16_GRAM_S_DTYPE
            or fsha != D16_GRAM_S_FILE_SHA256
            or asha != D16_GRAM_S_ARRAY_SHA256):
        raise RuntimeError(
            "[d16prog_certs] GRAM GATE FAILED for %s: shape %r dtype %s "
            "file sha %s.. array sha %s.. (pins %s.. / %s..)"
            % (path, S.shape, S.dtype, fsha[:16], asha[:16],
               D16_GRAM_S_FILE_SHA256[:16], D16_GRAM_S_ARRAY_SHA256[:16]))
    if not np.array_equal(S, S.T):
        raise RuntimeError("[d16prog_certs] gram S not exactly symmetric")
    return np.asarray(S, np.float64)


# ------------------------------------------------- import-time self-check
def _selfcheck():
    for key, roll in D16PROG_CACHE_DIGESTS.items():
        got = manifest_digest16(D16PROG_CACHE_SPLIT_DIGESTS[key])
        if got != roll:
            raise RuntimeError(
                "[d16prog_certs] EMBEDDED CONSTANTS INCONSISTENT: "
                "manifest_digest16(%s per-split block) = %s but the "
                "transcribed roll-up is %s" % (key, got, roll))
        for s in CACHE_SPLITS:
            d = D16PROG_CACHE_SPLIT_DIGESTS[key][s]
            assert d["n_shards"] == SPLIT_NB[s], (key, s)
            assert d["n_rows"] == SPLIT_ROWS[s] == SPLIT_NB[s] * GEN_BS, \
                (key, s)


_selfcheck()

if __name__ == "__main__":
    print("d16prog_certs: embedded certificate constants self-check PASS")
    for key in sorted(D16PROG_CACHE_DIGESTS):
        print("  %-22s roll-up %s  (%d shards / %d rows)"
              % (key, D16PROG_CACHE_DIGESTS[key],
                 sum(SPLIT_NB.values()), sum(SPLIT_ROWS.values())))
