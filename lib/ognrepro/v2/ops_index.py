"""ognrepro.v2.ops_index -- operator index tables for the pair operators.

PROVENANCE
    Source: explore_v2f/ops_index.py @ v2-F 269bee39
    Source file sha256:
            3483772c06c5463a4061e834cb9c0e8ffb7f75ae01d5a2f081344f9921e057f5
    build() and the numpy references are VERBATIM (the Kubo-Mori reference
    kmb_numpy is deliberately not carried: no v2 lane uses it).

The engine's pair-block operator tensor (fermionic_mbody.rho_2_kkbar_gen,
p1_core.py:959) is a scipy COO of shape (m, m, D, D) with op[k,l][r,c] in
{0, 1}: every op[k,l] is a 0/1 PARTIAL PERMUTATION of the D = C(m, m/2)
seniority-zero configurations (at most one nonzero per row; verified by
build()).  The engine assembles H_I[r,c] = sum_ij G[i,j] op[j,i][r,c]
(p1_core.py:158), so we define the coupling operators

    h_a := op[j,i]   for a = i*m + j   (C-order flattening of G)

and H = H0 - sum_a G_a h_a with G symmetric (G_a = G_{a^T}).  The pair block
seen by the network is f2[k,l] = Tr(rho op[k,l]) = <h_{(l,k)}> = <h_{(k,l)}>.

Tables (all numpy, host side; cast to jnp where used):
    src_pad (A, D) int32   (h_a psi)[r] = psi_pad[src_pad[a, r]], psi_pad =
                           concat(psi, [0]); src_pad = D where h_a has no
                           entry in row r.
    pair_idx (A, A, D) int32  Tr(rho h_a h_b) = sum_r rho_flat_pad[pair_idx[a,b,r]]
                           with rho_flat_pad = concat(rho.reshape(D*D), [0])
                           (index tgt*D + r where tgt = src_b[src_a[r]]).
    u (A,) f64             Tr(h_a)/D  (0.5 on the m number operators, else 0)
    S (A, A) f64           engine.S_matrix_np = Tr(h_a h_b)/D  (Gram second moment)
    S_inf (A, A) f64       S - u u^T = Cov in the maximally mixed state (beta->0)
    pair_energies (m,) f64 2*eps_k in engine units (energies.reshape(m,2).sum(1)),
                           so that E = sum_k pair_energies[k] rho_kk - <G, rho>_F
                           (the energy-shell identity; verified to 8e-8 in f32).
"""
import numpy as np


def _coo_arrays(op):
    coords = np.asarray(op.coords)
    data = np.asarray(op.data)
    return coords, data


def build(engine):
    op = engine.rho_2_kkbar_arrays
    m = int(engine.M_PAIRS)
    D = int(engine.basis.size)
    A = m * m
    coords, data = _coo_arrays(op)
    if not np.all(data == 1.0):
        raise AssertionError("[ops_index] operator data not all +1")
    k, l, r, c = (coords[i] for i in range(4))
    # h_a = op[j,i] for a = i*m + j  <=>  op[k,l] = h_{(l,k)}  -> a = l*m + k
    a_idx = l * m + k
    src = np.full((A, D), -1, np.int64)
    if np.any(src[a_idx, r] != -1):
        raise AssertionError("[ops_index] more than one nonzero in a row")
    src[a_idx, r] = c
    valid = src >= 0
    src_pad = np.where(valid, src, D).astype(np.int32)

    # pair table: tgt_ab[r] = src_b[src_a[r]] if both valid
    src_a = src[:, None, :]                           # (A,1,D)
    src_a_safe = np.where(src_a >= 0, src_a, 0)
    tgt = np.take_along_axis(np.broadcast_to(src[None, :, :], (A, A, D)),
                             np.broadcast_to(src_a_safe, (A, A, D)), axis=2)
    ok = (src_a >= 0) & (tgt >= 0)
    pair_idx = np.where(ok, tgt * D + np.arange(D)[None, None, :], D * D)
    pair_idx = pair_idx.astype(np.int32)

    u = valid.sum(1).astype(np.float64) / D
    # diagonal entries only: h_a has row r -> r when r's config has pair k occupied
    diag_count = np.array([(src[a] == np.arange(D)).sum() for a in range(A)],
                          np.float64)
    u = diag_count / D
    S = np.asarray(engine.S_matrix_np, np.float64)
    S_inf = S - np.outer(u, u)
    energies = np.asarray(engine.energies, np.float64)
    pair_energies = energies.reshape(m, 2).sum(1)
    return dict(m=m, D=D, A=A, src_pad=src_pad, pair_idx=pair_idx, u=u, S=S,
                S_inf=S_inf, pair_energies=pair_energies,
                n_pairs=int(engine.N_ELEC) // 2)


# ---------------------------------------------------------------- numpy refs
def apply_h(ops, a, psi):
    """(h_a psi) for a state vector psi (numpy reference)."""
    psi_pad = np.concatenate([np.asarray(psi, np.float64), [0.0]])
    return psi_pad[ops["src_pad"][a]]


def h_dense(ops, a):
    D = ops["D"]
    H = np.zeros((D, D), np.float64)
    rows = np.arange(D)
    src = ops["src_pad"][a]
    ok = src < D
    H[rows[ok], src[ok]] = 1.0
    return H


def cov_gs_numpy(ops, psi, dw_flat):
    """Var_psi(sum_a dw_a h_a) -- reference implementation."""
    v = np.zeros(ops["D"], np.float64)
    for a in range(ops["A"]):
        if dw_flat[a] != 0.0:
            v += dw_flat[a] * apply_h(ops, a, psi)
    psi = np.asarray(psi, np.float64)
    return float(v @ v - (psi @ v) ** 2)


def second_moment_numpy(ops, rho):
    """M2[a,b] = Tr(rho h_a h_b) via the pair table (numpy reference)."""
    rho_flat_pad = np.concatenate([np.asarray(rho, np.float64).reshape(-1), [0.0]])
    return rho_flat_pad[ops["pair_idx"]].sum(-1)


def cov_thermal_numpy(ops, rho):
    """Centered symmetric covariance M = sym(M2) - hbar hbar^T in state rho."""
    M2 = second_moment_numpy(ops, rho)
    D = ops["D"]
    rho_flat_pad = np.concatenate([np.asarray(rho, np.float64).reshape(-1), [0.0]])
    # <h_a> = sum_r rho[src_a[r], r]
    src = ops["src_pad"].astype(np.int64)
    idx = np.where(src < D, src * D + np.arange(D)[None, :], D * D)
    hbar = rho_flat_pad[idx].sum(-1)
    return 0.5 * (M2 + M2.T) - np.outer(hbar, hbar), hbar


# ----------------------------------------------------------------- d16 (seniority-zero block)
def build_d16_s0(S_pinned, m=8, n_pairs=4):
    """Operator tables for the d16n8 GS lanes.

    q_M (ground-state variance) needs the pair-hop tables on the v = 0 block
    only: the beta=100 ground state of the attractive pairing Hamiltonian is
    seniority-zero and every h_a = P^dag P preserves seniority, so the 70-dim
    block (C(8,4) configurations, ognrepro.blocked.FreeSetBlock(m, n, ())
    enumeration order = the psi0 sidecar's basis order) is closed under dH.
    The anchor S_inf = S - u u^T is the beta->0 covariance over the FULL
    12,870-dim basis, S being the pinned d16 Gram metric the published d16
    lanes trained with (Tr(h_a^dag h_b)/D_full; d16_gram_S_64x64.npy) and
    u_a = Tr(h_a)/D_full (nonzero on the m number operators; asserted equal
    to the pinned diagonal).  Index convention mirrors build(): h_a for
    a = i*m + j is P_j^dag P_i (remove pair i, add pair j); quadratic forms are
    evaluated on symmetric dw, where the a <-> a' convention is immaterial.
    """
    import math  # noqa: PLC0415
    from .. import blocked as B  # noqa: PLC0415
    n = 2 * n_pairs
    blk = B.FreeSetBlock(m, n, ())
    D, A = blk.dim, m * m
    idx = {c: i for i, c in enumerate(blk.cfgs)}
    src = np.full((A, D), -1, np.int64)
    for a_ in range(A):
        i, j = divmod(a_, m)
        for x, c in enumerate(blk.cfgs):
            cs = set(c)
            if i not in cs or (j != i and j in cs):
                continue
            y = x if j == i else idx[tuple(sorted((cs - {i}) | {j}))]
            if src[a_, y] != -1:
                raise AssertionError("[ops_index.d16] two sources for one row")
            src[a_, y] = x
    valid = src >= 0
    src_pad = np.where(valid, src, D).astype(np.int32)
    src_a = src[:, None, :]
    src_a_safe = np.where(src_a >= 0, src_a, 0)
    tgt = np.take_along_axis(np.broadcast_to(src[None, :, :], (A, A, D)),
                             np.broadcast_to(src_a_safe, (A, A, D)), axis=2)
    ok = (src_a >= 0) & (tgt >= 0)
    pair_idx = np.where(ok, tgt * D + np.arange(D)[None, None, :], D * D).astype(np.int32)
    S = np.asarray(S_pinned, np.float64)
    if S.shape != (A, A) or not np.array_equal(S, S.T):
        raise ValueError("[ops_index.d16] pinned S must be a symmetric (%d,%d) array" % (A, A))
    D_full = math.comb(2 * m, n)
    u_kk = math.comb(2 * m - 2, n - 2) / D_full          # Tr(n_k)/D_full
    u = np.zeros(A, np.float64)
    diag = [k * m + k for k in range(m)]
    u[diag] = u_kk
    if not np.allclose(S[diag, diag], u_kk, rtol=0, atol=1e-12):
        raise AssertionError("[ops_index.d16] pinned S diagonal %r != Tr(n_k)/D %g"
                             % (S[diag, diag], u_kk))
    S_inf = S - np.outer(u, u)
    eps_lvl = B.ladder(m)
    return dict(m=m, D=D, A=A, src_pad=src_pad, pair_idx=pair_idx, u=u, S=S, S_inf=S_inf,
                pair_energies=2.0 * eps_lvl, n_pairs=int(n_pairs), D_full=int(D_full),
                block_cfgs=[tuple(int(v) for v in c) for c in blk.cfgs])
