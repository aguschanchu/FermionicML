"""ognrepro.v2.stats -- input-standardization statistics of the v2 lanes.

Two conventions, both of the published record, kept separate:

  std1_entry_stats  -- STD-1 per-entry m x m mean / sigma of the SYMMETRIZED
      pair block over the first n_max TRAINING rows in file order
      (shuffle=False), f64 first/second-moment accumulation with the
      >= n_max overshoot-inclusive break, sigma floored at 1e-8, exact
      symmetry asserted.  This is the manuscript's standardized flagship
      (experiments/experiment-campaign2/stages/t_std2.py:860-905 `_std_stats`,
      itself tasks6/t_train6.py:230-261); consumed by
      std_arch.StdPhysicsOrbitalGraphNet (std_mu / std_sigma).
  triu55_stats      -- campaign10 55-feature sqrt(2)-weighted-triu mean / std
      (tasks10/t_train10.py:691-728 `_feature_stats`; std < 1e-6 -> 1) used by
      the standardized MLP arm only (explore_v2f/model.py feature_stats
      numerics @ 269bee39).

Both read the lane's OWN v2 cache halves through V2Loader (aux=None,
shuffle=False, batch = gen_bs), never a statistics file of another cache.
"""
import numpy as np

from .loader import V2Loader

SIGMA_FLOOR = 1e-8


def std1_entry_stats(half_dirs, gen_bs, m, n_max=300_000, sigma_floor=SIGMA_FLOOR, max_rows_per_dir=None):
    loader = V2Loader(half_dirs, batch_size=int(gen_bs), shuffle=False, aux=None, max_rows_per_dir=max_rows_per_dir)
    n = 0
    s = np.zeros((m, m), np.float64)
    ss = np.zeros((m, m), np.float64)
    for bx, _be, _by in loader:
        X = np.asarray(bx, np.float64)
        if X.ndim == 4:
            X = X[..., 0]
        Xs = 0.5 * (X + np.swapaxes(X, 1, 2))
        s += Xs.sum(0); ss += (Xs * Xs).sum(0); n += len(Xs)
        if n >= n_max:
            break
    if n == 0:
        raise RuntimeError("[v2.stats] standardization: training loader empty")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    sigma = np.maximum(np.sqrt(var), sigma_floor)
    mu32 = mean.astype(np.float32)
    sig32 = sigma.astype(np.float32)
    assert np.array_equal(mu32, mu32.T) and np.array_equal(sig32, sig32.T), (
        "[v2.stats] std stats not exactly symmetric")
    assert float(sig32.min()) >= sigma_floor * 0.999
    return mu32, sig32, int(n)


def _triu_mask(m):
    r, c = np.triu_indices(m)
    mask = np.where(r == c, 1.0, np.sqrt(2.0)).astype(np.float32)
    return r, c, mask


def triu55_stats(half_dirs, gen_bs, m, n_max=300_000, max_rows_per_dir=None):
    r, c, mask = _triu_mask(m)
    loader = V2Loader(half_dirs, batch_size=int(gen_bs), shuffle=False, aux=None, max_rows_per_dir=max_rows_per_dir)
    n = 0
    s = ss = None
    for bx, _be, _by in loader:
        X = np.asarray(bx, np.float64)[..., 0]
        Xs = 0.5 * (X + np.swapaxes(X, 1, 2))
        F = Xs[:, r, c] * mask
        if s is None:
            s = np.zeros(F.shape[1], np.float64); ss = np.zeros(F.shape[1], np.float64)
        s += F.sum(0); ss += (F * F).sum(0); n += len(F)
        if n >= n_max:
            break
    if n == 0:
        raise RuntimeError("[v2.stats] standardization: training loader empty")
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), int(n)


def save_std_stats(path, kind, mean, std, n, source):
    """Write the lane's standardization sidecar.  kind 'std1_entry' -> keys
    mu/sigma (the published std_stats.npz schema); 'triu55' -> keys mean/std."""
    if kind == "std1_entry":
        np.savez(path, mu=np.asarray(mean, np.float32), sigma=np.asarray(std, np.float32),
                 n=np.int64(n), kind=np.array(kind), source=np.array(str(source)))
    elif kind == "triu55":
        np.savez(path, mean=np.asarray(mean, np.float32), std=np.asarray(std, np.float32),
                 n=np.int64(n), kind=np.array(kind), source=np.array(str(source)))
    else:
        raise ValueError(kind)
    return path


def load_std_stats(path):
    with np.load(path) as z:
        kind = str(z["kind"]) if "kind" in z.files else ("std1_entry" if "mu" in z.files else "triu55")
        if kind == "std1_entry":
            return kind, np.asarray(z["mu"]), np.asarray(z["sigma"]), int(z["n"]) if "n" in z.files else None
        return kind, np.asarray(z["mean"]), np.asarray(z["std"]), int(z["n"]) if "n" in z.files else None
