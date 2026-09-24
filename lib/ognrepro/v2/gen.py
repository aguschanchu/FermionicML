"""ognrepro.v2.gen -- v2 dataset generation (HIGHEST precision + sidecars).

PROVENANCE
    Source: explore_v2f/data.py @ v2-F 269bee39 (sha256
            23412982cca3c0c6036263d5469c379c381bef344b65e5f205cf781d4fb6afdc):
            _make_kernel (:228-357, single-channel branch), gen_dataset_v2f
            (:360-529), read_schema / cache_complete / labels_digest /
            shard_path / sidecar_path (:74-118, :532-540); the multi-channel,
            Kubo-Mori and noisy-copy paths are not carried.
    Tag: ADAPTED -- driven by config.V2Config, explicit `precision=` on the
    engine's Hamiltonian / feature kernels (p1_core.py [V2] keywords), schema
    version 2 with the precision + sidecar tags, per-array certificates.

The generator keeps the engine's LABEL KEY CHAIN verbatim (PRNGKey(seed) ->
one warm-up split of n_devices+1 keys -> one split per batch; g_gen configured
with device_batch = gen_bs // n_devices; p1_core.py:819, :864-866, :888-889),
so every shard's labels are identical to the published caches and streams for
the same (h_type, seed, n_devices, gen_bs).  The features are the engine's
forward expressions with the matmul precision of BOTH the Hamiltonian build
and the state / pair-block contractions set by cfg.data_precision ('highest'
= exact float32; 'default' = the published bf16-pass numerics on TPU).

Sidecars (cfg.aux): 'psi0' -> the sign-fixed ground-state vector (D,) f32
stored inside the shard npz; 'M' -> the centered symmetric covariance of the
A = m*m coupling operators in the state, packed upper triangle (A(A+1)/2,)
stored as an uncompressed sidecar shard_%05d_M.npy in cfg.m_dtype.  Shards
keep the engine keys labels / energy / features, so engine.NumpyLoader reads a
v2 cache unchanged (sidecar files are invisible to its glob).
"""
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np

from .config import V2Config, CFG_VERSION

SCHEMA_VERSION = 2
GENERATOR = "ognrepro.v2.gen.gen_dataset_v2"


# ----------------------------------------------------------------- paths
def shard_path(cache_path, i):
    return os.path.join(cache_path, "shard_%05d.npz" % i)


def sidecar_path(cache_path, i):
    return os.path.join(cache_path, "shard_%05d_M.npy" % i)


def dataset_key(era, h_type, state_type, cfg, tag=None):
    """Cache key that carries precision + sidecar: no published `ds_*` directory
    can ever be picked up by a v2 lane."""
    k = "dsv2_%s_%s_%s__%s" % (era, h_type, state_type, cfg.cache_tag())
    if tag:
        k += "_" + str(tag)
    return k


def aux_width(aux, A, D):
    if aux is None:
        return 0
    if aux == "psi0":
        return int(D)
    if aux == "M":
        return int(A * (A + 1) // 2)
    raise ValueError(aux)


# ----------------------------------------------------------------- schema
def read_schema(cache_path):
    p = os.path.join(cache_path, "schema.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def cache_complete(cache_path, num_samples, gen_bs, cfg, h_type=None, state_type=None,
                   beta=None, seed=None):
    """Our own, complete, matching cache?  Schema-gated: never a size heuristic,
    never a foreign or partial cache."""
    sc = read_schema(cache_path)
    if not sc or sc.get("schema_version") != SCHEMA_VERSION:
        return False
    if sc.get("cfg_version") != CFG_VERSION:
        return False
    if sc.get("aux") != cfg.aux or int(sc.get("gen_bs", -1)) != int(gen_bs):
        return False
    if sc.get("matmul_precision") != cfg.data_precision:
        return False
    if cfg.aux == "M" and sc.get("aux_dtype") != cfg.m_dtype:
        return False
    if h_type is not None and sc.get("h_type") != h_type:
        return False
    if state_type is not None and sc.get("state_type") != state_type:
        return False
    if beta is not None and abs(float(sc.get("beta", -1.0)) - float(beta)) > 0:
        return False
    if seed is not None and int(sc.get("seed", -1)) != int(seed):
        return False
    if int(sc.get("num_samples", -1)) < int(num_samples):
        return False
    if not sc.get("complete"):
        return False
    # host class: a cache minted on another backend / device count is a different
    # precision class and a different label chain (n_devices) -- never reused silently
    import jax  # noqa: PLC0415
    if sc.get("backend") != jax.default_backend() or int(sc.get("n_devices", -1)) != int(jax.local_device_count()):
        return False
    files = sorted(glob.glob(os.path.join(cache_path, "shard_*.npz")))
    n_batches = math.ceil(num_samples / gen_bs)
    if len(files) < n_batches:
        return False
    if cfg.aux == "M":
        side = sorted(glob.glob(os.path.join(cache_path, "shard_*_M.npy")))
        if len(side) < n_batches:
            return False
    return True


def labels_digest(cache_path, n_shards=2):
    h = hashlib.sha256()
    for i in range(n_shards):
        p = shard_path(cache_path, i)
        if not os.path.isfile(p):
            break
        with np.load(p) as z:
            h.update(np.ascontiguousarray(z["labels"]).tobytes())
    return h.hexdigest()


def certify_cache(cache_path, n_rows=None):
    """Per-array sha256 over every shard in file order (streaming; the
    d16prog_certs.split_array_digests_streaming recipe): labels, features,
    energy and the sidecar / psi0 when present.  Returns the certificate dict."""
    files = sorted(glob.glob(os.path.join(cache_path, "shard_*.npz")))
    hs = {k: hashlib.sha256() for k in ("labels", "features", "energy")}
    h_aux = None
    rows = 0
    for i, f in enumerate(files):
        with np.load(f) as z:
            for k in hs:
                hs[k].update(np.ascontiguousarray(z[k]).tobytes())
            rows += len(z["labels"])
            if "psi0" in z.files:
                h_aux = h_aux or hashlib.sha256()
                h_aux.update(np.ascontiguousarray(z["psi0"]).tobytes())
        sp = sidecar_path(cache_path, i)
        if os.path.isfile(sp):
            h_aux = h_aux or hashlib.sha256()
            h_aux.update(np.ascontiguousarray(np.load(sp)).tobytes())
    cert = {k + "_sha256": v.hexdigest() for k, v in hs.items()}
    cert["aux_sha256"] = h_aux.hexdigest() if h_aux is not None else None
    cert["n_rows"] = int(rows)
    cert["n_shards"] = len(files)
    return cert


# ----------------------------------------------------------------- kernel
def _precision_of(cfg):
    import jax  # noqa: PLC0415
    return jax.lax.Precision.HIGHEST if cfg.data_precision == "highest" else None


def _make_kernel(engine, ops, h_type, beta, is_thermal, aux, device_batch, cfg,
                 m_host=False):
    """Fused per-device kernel: labels -> G -> H -> eigh -> state -> features
    (+ sidecar).  Precision per cfg.data_precision on the H build, the state
    build and the pair-block contraction (explicit kwargs on the engine's [V2]
    kernels, plus the trace-time default_matmul_precision context set by the
    caller as belt and braces)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    P = _precision_of(cfg)
    D, A = ops["D"], ops["A"]
    pair_idx = jnp.asarray(ops["pair_idx"])          # (A, A, D) int32
    iu = np.triu_indices(A)
    iu_r, iu_c = jnp.asarray(iu[0]), jnp.asarray(iu[1])
    g_gen = engine.GGenerator(engine.basis, h_type, device_batch,
                              g_init=engine.g_gen.g_init, g_stop=engine.g_gen.g_stop)
    two_body = engine.two_body_hamiltonian_dense
    effective_beta = float(beta) if is_thermal else 100.0   # p1_core.py:413
    m_chunk = int(cfg.m_chunk)

    def _cov_chunk(rho_chunk, hbar_chunk):
        # rho_chunk (c, D, D), hbar (c, A) -> packed M (c, A(A+1)/2)
        c = rho_chunk.shape[0]
        flat = jnp.concatenate([rho_chunk.reshape(c, D * D),
                                jnp.zeros((c, 1), rho_chunk.dtype)], axis=1)
        M2 = jnp.take(flat, pair_idx.reshape(-1), axis=1)      # (c, A*A*D)
        M2 = M2.reshape(c, A, A, D).sum(-1)                     # Tr(rho h_a h_b)
        M = 0.5 * (M2 + jnp.swapaxes(M2, 1, 2)) - hbar_chunk[:, :, None] * hbar_chunk[:, None, :]
        return M[:, iu_r, iu_c]

    def kernel(key_shard, base_energies_shard, rho_1_d, rho_2_i, rho_t):
        _, labels = g_gen.generate(key_shard)
        G = g_gen.reconstruct(labels)
        H = two_body(base_energies_shard, G, rho_1_d, rho_2_i, precision=P)
        vals, vecs = jnp.linalg.eigh(H)
        probs = jax.nn.softmax(-effective_beta * vals, axis=-1)
        rho = jnp.matmul(vecs * probs[..., None, :], jnp.swapaxes(vecs.conj(), -1, -2),
                         precision=P)
        energy = jnp.sum(probs * vals, axis=-1)
        f2 = engine.compute_rdm_trace(rho, rho_t, precision=P)   # (b, m, m, 1)
        out = dict(labels=labels, features=f2.real, energy=energy.real)
        if aux == "psi0":
            psi = vecs[..., 0]
            amax = jnp.argmax(jnp.abs(psi), axis=-1)
            sgn = jnp.sign(jnp.take_along_axis(psi, amax[:, None], axis=1))
            out["psi0"] = (psi * sgn).real
        elif aux == "M" and m_host:
            out["rho"] = rho.real                                 # host-side covariance (CPU smoke)
        elif aux == "M":
            hbar = f2.real.reshape(f2.shape[0], A)                # <h_a> = f2 flattened (symmetric)
            n = rho.shape[0]
            nchunk = max(1, n // m_chunk)
            assert n % nchunk == 0, (n, nchunk)
            rho_c = rho.real.reshape(nchunk, n // nchunk, D, D)
            hb_c = hbar.reshape(nchunk, n // nchunk, A)
            Mp = jax.lax.map(lambda t: _cov_chunk(t[0], t[1]), (rho_c, hb_c))
            out["M"] = Mp.reshape(n, -1)
        return out

    return kernel


def _cov_host(ops, rho_h, f2_h):
    """Host (numpy f64) covariance for the CPU path; packed upper triangle."""
    gen_bs = rho_h.shape[0]
    iu = np.triu_indices(ops["A"])
    flat_pad = np.concatenate([rho_h.reshape(gen_bs, -1), np.zeros((gen_bs, 1))], axis=1)
    Mp = np.empty((gen_bs, len(iu[0])), np.float32)
    for r in range(gen_bs):
        M2 = flat_pad[r][ops["pair_idx"]].sum(-1)
        M = 0.5 * (M2 + M2.T) - np.outer(f2_h[r], f2_h[r])
        Mp[r] = M[iu]
    return Mp


# ----------------------------------------------------------------- generation
def gen_dataset_v2(engine, ops, cache_path, *, h_type, state_type, beta, num_samples,
                   gen_bs, seed, cfg, include_energy=True, verbose=True,
                   force_device_m=False, era=None):
    """Generate (or reuse) a schema-v2 cache under cfg.  Returns the path."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if not isinstance(cfg, V2Config):
        raise TypeError("cfg must be a V2Config")
    cache_path = os.path.abspath(cache_path)
    if cache_complete(cache_path, num_samples, gen_bs, cfg, h_type=h_type,
                      state_type=state_type, beta=beta, seed=seed):
        if verbose:
            print("[v2.gen] cache complete: %s" % cache_path, flush=True)
        return cache_path
    sc = read_schema(cache_path)
    if os.path.isdir(cache_path):
        if sc is None and glob.glob(os.path.join(cache_path, "shard_*.npz")):
            raise RuntimeError("[v2.gen] %s holds shards without a v2 schema; "
                               "refusing to overwrite a foreign cache" % cache_path)
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    devices = jax.local_devices()
    n_devices = len(devices)
    if gen_bs % n_devices != 0:
        raise ValueError("gen_bs %d not divisible by n_devices %d" % (gen_bs, n_devices))
    device_batch = gen_bs // n_devices
    is_thermal = (state_type == "thermal")
    aux = cfg.aux

    def replicate(arr):
        if hasattr(arr, "todense"):
            arr = arr.todense()
        arr = np.asarray(arr)
        arr = arr.astype(np.complex64) if np.iscomplexobj(arr) else arr.astype(np.float32)
        return jax.device_put_sharded([jnp.array(arr)] * n_devices, devices)

    rho_1_np = engine._ensure_dense(engine.rho_1_arrays)
    rho_1_diag_np = (np.einsum("kknn->kn", rho_1_np) if rho_1_np.ndim == 4
                     else np.diagonal(rho_1_np, axis1=1, axis2=2))
    target_np = engine._ensure_dense(engine.rho_2_kkbar_arrays)
    p_rho_1_diag = replicate(rho_1_diag_np)
    p_rho_2_inter = replicate(target_np)
    p_rho_target = replicate(target_np)
    base_energies_np = engine.U_ENERGY_SEED[0]
    p_base_energies = replicate(np.tile(base_energies_np, (device_batch, 1)))

    m_host = (aux == "M" and jax.default_backend() == "cpu" and not force_device_m)
    kernel = _make_kernel(engine, ops, h_type, beta, is_thermal, aux, device_batch, cfg,
                          m_host=m_host)
    p_step = jax.pmap(kernel, axis_name="batch")
    global_prec = cfg.data_precision

    # ---- key chain VERBATIM p1_core.py:864-866 / :888-889
    rng_key = jax.random.PRNGKey(seed)
    rng_key, *warmup_keys = jax.random.split(rng_key, n_devices + 1)
    t0 = time.time()
    with jax.default_matmul_precision(global_prec):
        _ = p_step(jnp.array(warmup_keys), p_base_energies, p_rho_1_diag,
                   p_rho_2_inter, p_rho_target)
    if verbose:
        print("[v2.gen] kernel compiled in %.1fs (precision=%s, aux=%s, n_devices=%d, gen_bs=%d)"
              % (time.time() - t0, cfg.data_precision, aux, n_devices, gen_bs), flush=True)

    num_batches = math.ceil(num_samples / gen_bs)
    schema = dict(schema_version=SCHEMA_VERSION, cfg_version=CFG_VERSION, generator=GENERATOR,
                  era=era, h_type=h_type, state_type=state_type, beta=float(beta),
                  effective_beta=(float(beta) if is_thermal else 100.0),
                  gen_bs=int(gen_bs), n_devices=int(n_devices), seed=int(seed),
                  num_samples=int(num_samples), n_batches=int(num_batches),
                  D=int(ops["D"]), A=int(ops["A"]), m=int(ops["m"]),
                  label_size=int(engine.g_gen.label_size()),
                  aux=aux, aux_width=aux_width(aux, ops["A"], ops["D"]),
                  aux_dtype=(cfg.m_dtype if aux == "M" else ("float32" if aux else None)),
                  aux_layout=({"M": "packed_triu_AxA", "psi0": "psi_full"}.get(aux)),
                  matmul_precision=cfg.data_precision, h_precision=cfg.data_precision,
                  feat_precision=cfg.data_precision,
                  include_energy=bool(include_energy), complete=False,
                  backend=jax.default_backend(), device_kind=devices[0].device_kind,
                  jax_version=jax.__version__, v2config=cfg.to_json(),
                  created=time.time())
    with open(os.path.join(cache_path, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)

    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = []
    label_len = engine.g_gen.label_size()

    def save_task(i, out):
        fn = shard_path(cache_path, i)
        d = dict(labels=out["labels"],
                 energy=(out["energy"] if out["energy"].ndim == 2 else out["energy"][:, None]),
                 features=out["features"])
        if "psi0" in out:
            d["psi0"] = out["psi0"]
        tmp = fn + ".tmp.npz"
        np.savez(tmp, **d)
        os.replace(tmp, fn)
        if "M" in out:
            sp = sidecar_path(cache_path, i)
            np.save(sp + ".tmp.npy", np.asarray(out["M"], dtype=cfg.m_dtype))
            os.replace(sp + ".tmp.npy", sp)

    t0 = time.time()
    for i in range(num_batches):
        rng_key, *subkeys = jax.random.split(rng_key, n_devices + 1)
        with jax.default_matmul_precision(global_prec):
            res = p_step(jnp.array(subkeys), p_base_energies, p_rho_1_diag,
                         p_rho_2_inter, p_rho_target)
        res["features"].block_until_ready()
        host = {}
        for k, v in res.items():
            arr = np.asarray(v)
            host[k] = arr.reshape((gen_bs,) + arr.shape[2:])
        host["labels"] = host["labels"].reshape(gen_bs, label_len)
        if m_host:
            rho_h = host.pop("rho").astype(np.float64)
            f2_h = host["features"].reshape(gen_bs, ops["A"]).astype(np.float64)
            host["M"] = _cov_host(ops, rho_h, f2_h)
        futures.append(io_pool.submit(save_task, i, host))
        if len(futures) > 8:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                f.result()
            futures = [f for f in futures if not f.done()]
        if verbose and (i % 50 == 0 or i == num_batches - 1):
            el = time.time() - t0
            print("[v2.gen] %d/%d batches  %.1fs  (%.1f samples/s)"
                  % (i + 1, num_batches, el, (i + 1) * gen_bs / max(el, 1e-9)), flush=True)
    for f in concurrent.futures.wait(futures)[0]:
        f.result()
    io_pool.shutdown()
    schema["complete"] = True
    schema["wall_s"] = time.time() - t0
    schema["labels_digest_first2"] = labels_digest(cache_path, n_shards=2)
    with open(os.path.join(cache_path, "schema.json"), "w") as fh:
        json.dump(schema, fh, indent=1)
    return cache_path


# ----------------------------------------------------------------- lane caches
TRAIN_SEEDS = (42, 43)      # campaign/config.py TRAIN_SEEDS: half0 / half1
VAL_SEED = 1007             # campaign/config.py VAL_SEED


def split_plan(num_samples, gen_bs, val_frac=0.05):
    """(half0, half1, val) sample counts of the published convention
    (t_train.py:141-146): halves of num_samples//2, val = round(5 %)."""
    n_half = max(1, int(num_samples) // 2)
    n_val = max(int(gen_bs), int(round(num_samples * val_frac)))
    return n_half, n_half, n_val


def ensure_caches(engine, ops, root, *, era, h_type, state_type, beta, num_samples, gen_bs,
                  cfg, verbose=True, force_device_m=False):
    """half0/half1 (seeds 42/43) + val (seed 1007) under <root>/{half0,half1,val};
    returns (train_dirs, val_dir, info).  Cross-process safety: an atomic mkdir
    lock on <root>.lock (the t_train.py:150-176 pattern); the winner generates,
    losers wait until the caches are schema-complete."""
    n0, n1, nv = split_plan(num_samples, gen_bs)
    half_dirs = [os.path.join(root, "half0"), os.path.join(root, "half1")]
    val_dir = os.path.join(root, "val")
    jobs = ((half_dirs[0], TRAIN_SEEDS[0], n0), (half_dirs[1], TRAIN_SEEDS[1], n1),
            (val_dir, VAL_SEED, nv))
    os.makedirs(root, exist_ok=True)
    lock = root + ".lock"

    def _ready():
        return all(cache_complete(cp, n, gen_bs, cfg, h_type=h_type, state_type=state_type,
                                  beta=beta, seed=sd) for cp, sd, n in jobs)

    STALE_LOCK_S, WAIT_MAX_S, waited = 3 * 3600, 6 * 3600, 0
    while not _ready():
        try:
            os.mkdir(lock)
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > STALE_LOCK_S:
                    os.rmdir(lock); continue
            except OSError:
                pass
            if waited >= WAIT_MAX_S:
                raise RuntimeError("timed out waiting for cache %s (lock held elsewhere)" % root)
            time.sleep(20); waited += 20
            continue
        try:
            for cp, sd, n in jobs:
                gen_dataset_v2(engine, ops, cp, h_type=h_type, state_type=state_type, beta=beta,
                               num_samples=n, gen_bs=gen_bs, seed=sd, cfg=cfg, verbose=verbose,
                               force_device_m=force_device_m, era=era)
        finally:
            try:
                os.rmdir(lock)
            except OSError:
                pass
        break
    info = dict(root=root, train_seeds=list(TRAIN_SEEDS), val_seed=VAL_SEED,
                n_train=n0 + n1, n_val=nv, gen_batch_size=int(gen_bs),
                schemas={os.path.basename(cp): read_schema(cp) for cp, _s, _n in jobs})
    return half_dirs, val_dir, info
