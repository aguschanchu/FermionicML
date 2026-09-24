"""ognrepro.v2.loader -- engine-compatible loader with the aux channel.

PROVENANCE
    Source: explore_v2f/data.py @ v2-F 269bee39, class V2FLoader (:544-628);
    ADAPTED: single channel, no population-noise mixing; `max_rows_per_dir`
    added for the learning-curve lanes (a prefix of each production half in
    file order = the same label stream truncated).

NumpyLoader semantics (p1_core.py:684-723) are replicated exactly: file-order
shuffle then per-file index permutation through the GLOBAL numpy RNG,
partial chunks dropped, so a lane's sample order is a pure function of
(init_seed, epoch) exactly as the engine's.  Yields the engine 3-tuple
(bx, be_packed, by) with be_packed = concat([energy (b,1), aux_flat], axis=1);
the training step splits energy (the only tensor the network receives) from
aux (loss only).
"""
import glob
import os

import numpy as np


class V2Loader:
    def __init__(self, dirs, batch_size, shuffle=True, aux=None, max_rows_per_dir=None):
        dirs = [dirs] if isinstance(dirs, str) else list(dirs)
        files = []
        for d in dirs:
            fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
            if max_rows_per_dir is not None and fs:
                with np.load(fs[0]) as z:
                    per = len(z["labels"])
                n_files = int(np.ceil(float(max_rows_per_dir) / per))
                fs = fs[:n_files]
            files.extend(fs)
        self.files = files
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.aux = aux
        if aux not in ("M", "psi0", None):
            raise ValueError("aux %r" % (aux,))
        if not self.files:
            print("Warning: No files found in %r" % (dirs,))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _sidecar(f):
        return f[:-4] + "_M.npy"

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            np.random.shuffle(files)
        for f in files:
            with np.load(f) as d:
                Y = d["labels"]
                E = d["energy"]
                X = d["features"]
                aux_arr = None
                if self.aux == "psi0" and "psi0" in d.files:
                    aux_arr = np.asarray(d["psi0"])
            if self.aux == "M":
                aux_arr = np.load(self._sidecar(f), mmap_mode="r")
            elif self.aux == "psi0" and aux_arr is None:
                # certified d16 caches: psi0 lives in a sibling sidecar (shards untouched)
                aux_arr = np.load(f[:-4] + "_psi0.npy", mmap_mode="r")
            N = len(Y)
            indices = np.arange(N)
            if self.shuffle:
                np.random.shuffle(indices)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx = indices[start:end]
                if len(idx) < self.batch_size:
                    continue
                bx = X[idx]
                if bx.ndim == 3:
                    bx = bx[..., None]
                be = E[idx].astype(np.float32)
                if be.ndim == 1:
                    be = be[:, None]
                if aux_arr is not None:
                    idx_sorted = np.sort(idx)          # mmap-friendly gather
                    order = np.argsort(np.argsort(idx))
                    a = np.asarray(aux_arr[idx_sorted], np.float32)[order]
                    be = np.concatenate([be, a.reshape(len(idx), -1)], axis=1)
                yield bx, be, Y[idx]
