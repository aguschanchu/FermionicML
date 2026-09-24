"""ognrepro.blocked -- the seniority-blocked exact solver + shard IO.

PROVENANCE
    Source: experiments/experiment-campaign2-r2/stages/
            expc2r2_d12_blocked_validation.py
    Source file sha256:
            9a14f209b740e6e17607208e3aa7dcc057083e705f1a10c5e74b4e4d14552947
    Line ranges extracted:
        63-71     THREAD_ENV_VARS single-thread BLAS pins (module level in the
                  source, mirrored at module level here)
        87-88     STAGE/TAG (TAG is referenced by v_matrix's refusal message)
        101-105   era pins D12_D_SP/D12_N_ELEC/D12_M, D12_BASIS, BETA,
                  G_INIT/G_STOP, GEN_BS
        106-108   NDEV (emulated device count for replay_labels)
        93        PROVISIONAL suffix (used by save_shard)
        278-279   shard_name
        739-942   ladder, v_matrix, sector_table, FreeSetBlock, build_blocks,
                  blocked_solve, number_identity, energy_identity,
                  replay_labels
        948-1021  save_shard, shard_wellformed, build_shard_range, read_split
    Extraction date: 2026-08-23.
    Tag: VERBATIM for every function/class body above (no matmul<->einsum
    swap, no reassociation, no reordering -- the frozen-expression-tree rule
    of recon/d16_probe_plan.md sec 2 is preserved), with these deliberate
    module-level diffs only:
      * ERAS dict (ADAPTED ADDITION, below): the era table, transcribed from
        experiments/experiment-campaign2-r2/stages/t_d16probe.py lines
        130-138 (ERA_TABLE / SENIORITY_ZERO_STATES; source sha256
        73f3f3113d3b2d720e41e4999c03a902c3ee0c00d9ecca0ffe9bdc4d69f24b18)
        and experiments/experiment-campaign2-r2/stages/
        expc2r2_d16prog_p3_cachebuild.py lines 101-104 (D16_BASIS 12870,
        D16_N_BLOCKS 128, D16_S0_DIM 70; source sha256
        04f4ef18dc9b811ddda947e6b1adc944c98fc0c82f4e8461a76ccab1fc2da4d7).
      * The source's other module constants (registered digests, sealed
        record pins, gate rosters) are NOT carried: they belong to the
        validation stage, not the numeric core.
      * import list trimmed to what the extracted bodies use (os, itertools,
        math, time, numpy; jax stays LOCAL to replay_labels exactly as in
        the source).
    NOTE build_shard_range asserts nstates == D12_BASIS (line 996 of the
    source): it is the d12 cache builder, verbatim; do not reuse it for
    other eras (the d16 builder lives in expc2r2_d16prog_p3_cachebuild.py).
"""
import os

# Single-thread BLAS policy: must be in the environment BEFORE the first
# numpy import, or the pinned values are advisory only.
# [VERBATIM expc2r2_d12_blocked_validation.py:65-71 -- the source sets these
#  at module level, so this module mirrors it.]
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS")
for _k in THREAD_ENV_VARS:
    os.environ.setdefault(_k, "1")

import itertools
import math
import time

import numpy as np

STAGE = "expc2r2_d12_blocked_validation"          # [:87] (provenance label)
TAG = "[%s]" % STAGE                              # [:88]
PROVISIONAL = ".PROVISIONAL"                      # [:93]

# ---------------------------------------------------------------- era pins
# [VERBATIM :101-108]
D12_D_SP, D12_N_ELEC, D12_M = 12, 6, 6
D12_BASIS = 924
BETA = 1.0
G_INIT, G_STOP = 0.1, 1.0
GEN_BS = 512
NDEV = 4                       # emulated; 512 % 4 == 0 AND 512 % 8 == 0, so a
                               # divisibility-only guard would NOT catch ndev=8

# ---------------------------------------------------------------- ERAS
# [ADAPTED ADDITION -- see header.  tag -> era record; `m` is the SOLVER
#  dimension (M_PAIRS = D_SP//2 pair levels), the m argument of ladder /
#  v_matrix / build_blocks / blocked_solve; `n` is N_ELEC, the n argument of
#  build_blocks / sector_table.  Sources: t_d16probe.py ERA_TABLE:130-136 +
#  SENIORITY_ZERO_STATES:138 + N_BLOCKS:110-111; p3_cachebuild.py:101-104;
#  expc2r2_d12_blocked_validation.py:101-102,:115,:117.]
ERAS = {
    "d12":   dict(d_sp=12, n_elec=6, m=6, m_pairs=6, basis=924,
                  pairs=False, s0_block_dim=20, n_blocks=32),
    "d14n6": dict(d_sp=14, n_elec=6, m=7, m_pairs=7, basis=3003,
                  pairs=False),
    "d14n7": dict(d_sp=14, n_elec=7, m=7, m_pairs=7, basis=3432,
                  pairs=False),
    "d16n8": dict(d_sp=16, n_elec=8, m=8, m_pairs=8, basis=12870,
                  pairs=False, s0_block_dim=70, n_blocks=128),
}


def shard_name(i):
    return "shard_%05d.npz" % int(i)


# ======================================================================
# THE BLOCKED SOLVER -- frozen expression tree, lifted verbatim from
# recon/d16_prototype/proto_blocked.py (FreeSetBlock / build_blocks /
# blocked_solve).  No matmul<->einsum swap, no reassociation, no reordering:
# d16_probe_plan.md sec 2 ("Freeze the expression tree when lifting").
# ======================================================================
def ladder(m, scale=1.0):
    """F-LADDER (t_d14probe.py:1109-1110; == init_d12 at half filling): the
    PER-LEVEL energies; the mode energies are np.repeat(levels, 2)."""
    levels = np.arange(0, m) - m // 2 + 0.5
    return (levels / scale).astype(np.float64)


def v_matrix(htype, m, lab_row):
    """GGenerator.reconstruct (p1_core.py:493-495,:535-549) for one label."""
    if htype == "const":
        return np.full((m, m), float(lab_row[0]))
    if htype == "vect":
        expanded = np.concatenate([[0.0], np.asarray(lab_row, np.float64)])
        vals = np.repeat(expanded, 2)[:m]
        idx = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
        return vals[idx]
    raise SystemExit("%s h_type %r is outside the capability predicate"
                     % (TAG, htype))


def sector_table(m, n):
    out, total = [], 0
    for v in range(0, min(m, n) + 1):
        if (n - v) % 2 or (n - v) // 2 > m - v or n - v < 0:
            continue
        npair = (n - v) // 2
        dim = math.comb(m - v, npair)
        sets_ = math.comb(m, v)
        out.append(dict(v=v, blocked_sets=sets_, member_mult=2 ** v,
                        block_dim=dim, states=sets_ * (2 ** v) * dim))
        total += sets_ * (2 ** v) * dim
    return out, total


class FreeSetBlock:
    """All label-independent structure of ONE blocked-set S (v = |S| levels
    carrying a single occupied member; the 2^v member copies are bit-identical
    and are collapsed into the integer multiplier `mult`)."""

    def __init__(self, m, n, blocked):
        self.blocked = tuple(blocked)
        self.free = tuple(l for l in range(m) if l not in blocked)
        self.v = len(blocked)
        self.mult = 2 ** self.v
        self.npair = (n - self.v) // 2
        cfgs = list(itertools.combinations(self.free, self.npair))
        self.cfgs = cfgs
        self.dim = len(cfgs)
        idx = {c: i for i, c in enumerate(cfgs)}
        m_ = m
        O = np.zeros((self.dim, m_), np.float64)
        for i, c in enumerate(cfgs):
            for l in c:
                O[i, l] = 1.0
        self.O = O
        ty, tx, ta, tb = [], [], [], []
        for i, c in enumerate(cfgs):
            cs = set(c)
            for a in c:
                for b in self.free:
                    if b in cs:
                        continue
                    y = tuple(sorted(cs - {a} | {b}))
                    ty.append(idx[y])
                    tx.append(i)
                    ta.append(a)
                    tb.append(b)
        self.ty = np.asarray(ty, np.intp)
        self.tx = np.asarray(tx, np.intp)
        self.ta = np.asarray(ta, np.intp)
        self.tb = np.asarray(tb, np.intp)

    def h_batch(self, eps_lvl, V_batch, e_off):
        """(B, dim, dim) f64.
        diag = e_off + sum_{l in cfg} 2 eps_l - sum_{l in cfg} V[l,l]
        hop  = H[y,x] -= V[b,a]  (matrix element of T proven exactly +1)."""
        B = V_batch.shape[0]
        H = np.zeros((B, self.dim, self.dim), np.float64)
        d0 = self.O @ (2.0 * eps_lvl)
        vdiag = np.einsum("cl,bll->bc", self.O, V_batch)
        di = np.arange(self.dim)
        H[:, di, di] = e_off + d0[None, :] - vdiag
        if len(self.tx):
            H[:, self.ty, self.tx] -= V_batch[:, self.tb, self.ta]
        return H


def build_blocks(m, n):
    """Frozen enumeration order: v ascending, blocked-set lexicographic."""
    blocks = []
    for v in range(0, min(m, n) + 1):
        if (n - v) % 2 or (n - v) // 2 > m - v or n - v < 0:
            continue
        for blocked in itertools.combinations(range(m), v):
            blocks.append(FreeSetBlock(m, n, blocked))
    return blocks


def blocked_solve(blocks, eps_lvl, V_batch, m, beta=BETA):
    """Exact canonical thermal solve from the blocked decomposition.

    Returns (rho2, energy, z_shifted, e0, nstates, single_occ):
      rho2      (B,m,m) f64 RAW storage, f2[a,b] = Tr(rho T[a,b])
      energy    (B,)    f64
      z_shifted (B,)    f64 -- the E0-SHIFTED normalization; NOT Z (naming
                        guard, Sol 1.3): log Z_phys = log z_shifted - beta*e0
      e0        (B,)    f64 global per-sample min over ALL blocks
      nstates   int     state accounting (must equal the basis size)
      single_occ(B,m)   f64 s_a, the per-level single-occupancy probability
                        (needed by the G-6 particle-number/energy identities)
    """
    B = V_batch.shape[0]
    per = []
    nstates = 0
    for blk in blocks:
        e_off = float(sum(eps_lvl[l] for l in blk.blocked))
        H = blk.h_batch(eps_lvl, V_batch, e_off)
        if blk.dim == 1:
            vals = H[:, 0, :].copy()
            vecs = np.ones((B, 1, 1))
        else:
            vals, vecs = np.linalg.eigh(H)
        per.append((blk, vals, vecs))
        nstates += blk.mult * blk.dim
    e0 = np.min([vals.min(axis=1) for _, vals, _ in per], axis=0)
    Z = np.zeros(B)
    num_E = np.zeros(B)
    num_f2 = np.zeros((B, m, m))
    num_s = np.zeros((B, m))
    for blk, vals, vecs in per:
        w = np.exp(-beta * (vals - e0[:, None])) * blk.mult
        Z += w.sum(axis=1)
        num_E += (w * vals).sum(axis=1)
        rho = np.matmul(vecs * w[:, None, :], np.swapaxes(vecs, 1, 2))
        d = np.einsum("bcc->bc", rho)
        occ = d @ blk.O
        di = np.arange(m)
        num_f2[:, di, di] += occ
        if blk.v:
            wsum = w.sum(axis=1)
            for l in blk.blocked:
                num_s[:, l] += wsum
        if len(blk.tx):
            vals_t = rho[:, blk.tx, blk.ty]
            flat = np.zeros((B, m * m))
            np.add.at(flat.T, blk.ta * m + blk.tb, vals_t.T)
            num_f2 += flat.reshape(B, m, m)
    rho2 = num_f2 / Z[:, None, None]
    energy = num_E / Z
    single = num_s / Z[:, None]
    return rho2, energy, Z, e0, nstates, single


def number_identity(rho2, single):
    """2 * sum_a f2[a,a] + sum_a s_a == N exactly (the CORRECT G-6 invariant;
    sum_a f2[a,a] alone is <N_pair> and is NOT constant -- PANEL-O1 C-2)."""
    diag = np.einsum("bii->bi", rho2)
    return 2.0 * diag.sum(axis=1) + single.sum(axis=1)


def energy_identity(rho2, single, eps_lvl, V_batch):
    """E = sum_a eps_a (2 f2_aa + s_a) - sum_ab V_ab f2_ab (Sol sec 2.4)."""
    diag = np.einsum("bii->bi", rho2)
    return (2.0 * diag + single) @ eps_lvl - np.einsum("bij,bij->b", V_batch,
                                                       rho2)


# ======================================================================
# label stream replay (p1_core.py:864-869, :888, :583-590)
# ======================================================================
def replay_labels(seed, n_batches, gen_bs=GEN_BS, ndev=NDEV, label_size=1,
                  htype="const", g_init=G_INIT, g_stop=G_STOP, lo=0, hi=None):
    """Replay gen_dataset's label chain.  Emulated ndev is asserted EQUAL to 4
    (a divisibility-only guard passes for ndev=8 too: 512 % 8 == 0) and the
    warm-up split is performed and its keys discarded (dropping it shifts every
    batch)."""
    import jax
    import jax.numpy as jnp
    assert int(ndev) == 4, "emulated ndev must be exactly 4, got %r" % (ndev,)
    assert gen_bs % ndev == 0, "gen_bs must divide the device count"
    db = gen_bs // ndev
    hi = n_batches if hi is None else int(hi)
    key = jax.random.PRNGKey(int(seed))
    key, *warmup_keys = jax.random.split(key, ndev + 1)      # consumed
    assert len(warmup_keys) == ndev, "warm-up split did not occur"
    out = []
    for i in range(n_batches):
        key, *subkeys = jax.random.split(key, ndev + 1)
        if i < lo:
            continue                       # fast-forward: key consumed only
        if i >= hi:
            break
        rows = []
        for sk in subkeys:
            key_g, key_e = jax.random.split(sk)              # INNER split
            lab = jax.random.uniform(key_g, shape=(db, label_size),
                                     minval=g_init, maxval=g_stop,
                                     dtype=jnp.float32)
            if htype == "vect":
                lab = jnp.sort(lab, axis=1)[:, ::-1]
            rows.append(np.asarray(lab))
        out.append(np.concatenate(rows, axis=0).reshape(gen_bs, label_size))
    return np.concatenate(out, axis=0) if out else np.zeros((0, label_size),
                                                            np.float32)


# ======================================================================
# shard IO (mirrors expc2r2_d14_build._save_shard / _shard_wellformed)
# ======================================================================
def save_shard(path, labels, energy, features):
    tmp = path + PROVISIONAL
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, labels=labels, energy=energy,
                            features=features)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def shard_wellformed(path, gen_bs, label_dim, m):
    try:
        with np.load(path) as d:
            lab = np.asarray(d["labels"])
            feat = np.asarray(d["features"])
            en = np.asarray(d["energy"])
        return (lab.shape == (gen_bs, label_dim)
                and en.shape == (gen_bs, 1)
                and feat.shape == (gen_bs, m, m, 1)     # RANK-4, not [:3]
                and lab.dtype == np.float32
                and en.dtype == np.float32
                and feat.dtype == np.float32)
    except Exception:                                        # noqa: BLE001
        return False


def build_shard_range(labels, split_dir, lo, hi, m, n, eps_lvl, blocks,
                      htype="const", beta=BETA, gen_bs=GEN_BS, chunk=64):
    """Compute and write shards [lo, hi) of one split.  Row range of shard i is
    [i*gen_bs, (i+1)*gen_bs) of the pinned label array."""
    os.makedirs(split_dir, exist_ok=True)
    written, skipped, rows = 0, 0, 0
    t0 = time.time()
    for i in range(lo, hi):
        path = os.path.join(split_dir, shard_name(i))
        if os.path.exists(path) and shard_wellformed(path, gen_bs,
                                                     labels.shape[1], m):
            skipped += 1                       # already_present: never rewritten
            continue
        lab = np.ascontiguousarray(labels[i * gen_bs:(i + 1) * gen_bs])
        assert lab.shape[0] == gen_bs
        f2 = np.empty((gen_bs, m, m), np.float64)
        en = np.empty((gen_bs,), np.float64)
        for s in range(0, gen_bs, chunk):
            e = min(s + chunk, gen_bs)
            V = np.stack([v_matrix(htype, m, r) for r in lab[s:e]])
            r2, ee, _z, _e0, nst, _sg = blocked_solve(blocks, eps_lvl, V, m,
                                                      beta)
            assert nst == D12_BASIS
            f2[s:e] = r2
            en[s:e] = ee
        save_shard(path, lab.astype(np.float32),
                   en.astype(np.float32)[:, None],
                   f2.astype(np.float32)[..., None])
        written += 1
        rows += gen_bs
    return dict(lo=int(lo), hi=int(hi), n_written=int(written),
                n_skipped=int(skipped), n_rows=int(rows),
                wall_s=time.time() - t0,
                ms_per_label=(1000.0 * (time.time() - t0) / rows
                              if rows else None))


def read_split(split_dir, want_shards):
    names = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    expected = [shard_name(i) for i in range(want_shards)]
    labs, feats, ens = [], [], []
    for nm in names:
        with np.load(os.path.join(split_dir, nm)) as z:
            labs.append(np.asarray(z["labels"]))
            feats.append(np.asarray(z["features"]))
            ens.append(np.asarray(z["energy"]))
    return (np.concatenate(labs, 0), np.concatenate(feats, 0),
            np.concatenate(ens, 0), names, expected)
