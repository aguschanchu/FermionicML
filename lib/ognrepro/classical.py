"""ognrepro.classical -- classical estimator numerical cores (d=20 seniority
zero + d=16 input-matched fit), extracted from the campaign stages WITHOUT the
campaign gate/record machinery (no argparse, no checkpoints/jsonl, no gates,
no sealed-tree IO).

PROVENANCE (per block; every function cites its source lines in place)
    campaign4/compute.py
        sha256 0cec7b396133112d1ca466f1aa15656cb41fcc3bc6abda309af66ba8c4a4fa5b
        lines 31-50 (BLAS pins + era constants + energies), 53-75
        (heldout_G_batch), 78-114 (seniority-zero sector + h_s0), 117-138
        (SYMM_IDX + reconstruct_G_from_symm), 140-152 (wls_rows), 154-167
        (wls_solve), 169-191 (gls_solve), 193-205 (shifted_error /
        draw_gaussian), 207 (SWEEP16), 311-320 (prepare_systems), 385-431
        (the Hellinger forward-map fit core inside _fwd_fit_one ->
        fwdmap_fit), 544-550 (SWEEP50 / draw_multinomial), 579-606 (the
        CV-ridge lambda-selection core inside stage_wls50 -> cv_ridge).
    campaign6/stages/s6_secular_protocol.py
        sha256 114e4588b899e57341d7432d6548e47b961660e332a78064558168bedad562da
        lines 114-118 (COND/NULL_RTOL/D/NP55/IU_M,IU_N), 135-228 (the four
        arm solvers: offdiag_system, offdiag_solution,
        commutant_gauged_error, diag_rows_weighted, solve_full_hard,
        solve_full_soft, x_true_symm), 232-239 (_build_system ->
        build_system).  Wrapped by secular_solve(arm=...).
    campaign5/stages/s_prior_ridge.py
        sha256 c0d6157afc7bdfe4cb2b2e274f2e9134645b978c016572c06d26678115ddc688
        lines 140-146 (X_PRIOR / SUPPORT_DROP / BOX_LB / BOX_UB), 160-185
        (_solve -> prior_ridge), 188-205 (_solve_box -> box_map).
    experiments/experiment-campaign2-r2/stages/x22_fitu_pilot.py
        sha256 d916bcb44ae89fc958d32714fc16785690e9eaef385a612cf1fcec494fe45d79
        lines 156-177 (d16 era + fit-declaration constants), 409-485
        (_fwd_setup / forward_serialized / loss_batch / fit_one_target ->
        input_matched_setup / input_matched_forward / input_matched_loss /
        input_matched_fit).
    Extraction date: 2026-08-23.
    Tag: ADAPTED.  Deliberate diffs (beyond dropping campaign machinery):
      * n_states / n_noise / nproc / seed are PARAMETERS here; no
        multiprocessing pools, no shards, no jsonl checkpoints -- callers
        parallelize if they need to.
      * wls_solve/gls_solve keep their exact source bodies but the shared
        weighted-lstsq assembly of wls_solve is also exposed as wls_core
        (returns the raw 56-vector solution) so a synthetic system of any
        column count can be verified against a direct lstsq.
      * cv_ridge is the stage_wls50 lambda-selection loop (SeedSequence
        [1, stream, i_s, ci, t], stream 2=gaussian / 3=multinomial, 50-node
        SWEEP50 default, argmin of per-lambda medians) refactored to take
        the sweep / grid / CV budget as arguments.
      * fwdmap_fit is the per-trial core of compute.stage_fwdmap's
        _fwd_fit_one with the env-var switches (C4_FWD_MAP / C4_FWD_PRIOR_SIG
        / C4_FWD_TOL / C4_FWD_NFEV) turned into keyword arguments with the
        same defaults; the seeded draw SeedSequence([0, stream, i_s, si, t])
        is kept (stream parameterized; the source's fwdmap ran stream 2).
      * secular_solve / prior_ridge / box_map keep the exact solver bodies;
        module-name references (c4.X) are rebound to this module's globals.
      * input_matched_fit replaces the source's dynamically-verified d12
        module load (_d12()/_fwd_setup module cache) with an explicit
        (eps, blocks) argument pair built by input_matched_setup via
        ognrepro.blocked (itself the VERBATIM solver); the per-target fit
        body (grid + golden section, tolerances, convergence rule) is
        otherwise verbatim.
      * NOT extracted (too entangled / out of numerical-core scope,
        documented here per the extraction order): compute.py's
        stage_fast seniority-weight census (:210-266), the bootstrap /
        ratio assembly of stage_wls50 (:607-656), s6's checkpoint/bootstrap
        /finalize blocks (:300-622), s_prior_ridge's CV-cache and
        bootstrap (:209-575), and x22's leg drivers/gates (:525-964).
"""
import itertools
import math
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                 # [compute.py:32-33]

import numpy as np
import scipy.linalg

# ============================================================ campaign4 core
# [compute.py:37-50]
BETA = 1.0
MP = 10          # pair levels (d=20)
NPAIR = 5        # N/2
DS0 = 252        # C(10,5)

levels = np.arange(0, 10) - 10 // 2 + 0.5                    # [-4.5..4.5]
pair_jitter = np.linspace(-1e-4, 1e-4, MP).astype(np.float32)
levels_jittered = levels + pair_jitter
energies_sp = (np.repeat(levels_jittered, 2).astype(np.float32) / 10.0)
E_PAIR = (energies_sp[0::2] + energies_sp[1::2]).astype(np.float64)  # 2e_k


def heldout_G_batch(n_states, seed=0):
    """GGenerator('random') draws, verbatim conventions [p1_core l.443-620].
    [compute.py:53-75]"""
    import jax, jax.numpy as jnp
    key = jax.random.PRNGKey(4242 + int(seed))
    key_g, _key_e = jax.random.split(key)
    L = MP * (MP + 1) // 2
    raw = jax.random.uniform(key_g, shape=(int(n_states), L),
                             minval=0.1, maxval=1.0, dtype=jnp.float32)
    r_full, c_full = np.triu_indices(MP)
    out = jnp.zeros((int(n_states), MP, MP), dtype=jnp.float32)
    out = out.at[:, r_full, c_full].set(raw)
    diag_idx = jnp.arange(MP)
    diags = out[:, diag_idx, diag_idx]
    shift = 0.55 - jnp.mean(diags, axis=1, keepdims=True)
    out = out.at[:, diag_idx, diag_idx].set(diags + shift)
    labels = out[:, r_full, c_full]
    # reconstruct via _symmetric (k=0 branch: M + M.T with diag halved)
    full = jnp.zeros((int(n_states), MP, MP), dtype=jnp.float32)
    full = full.at[:, r_full, c_full].set(labels)
    full = full + jnp.swapaxes(full, 1, 2)
    dg = full[:, diag_idx, diag_idx]
    full = full.at[:, diag_idx, diag_idx].set(dg * 0.5)
    return np.array(full).astype(np.float64)


# ------------------------------------------- seniority-zero (hard-core) rep
# [compute.py:78-114]
S0_STATES = list(itertools.combinations(range(MP), NPAIR))
S0_INDEX = {s: i for i, s in enumerate(S0_STATES)}
assert len(S0_STATES) == DS0


def build_sector(levels_list, npair):
    """Hard-core pair basis + P^dag P operators for a level subset.
    [compute.py:89-105]"""
    states = list(itertools.combinations(range(len(levels_list)), npair))
    idx = {s: i for i, s in enumerate(states)}
    D_ = len(states)
    m = len(levels_list)
    P = np.zeros((m, m, D_, D_))
    for si, s in enumerate(states):
        occ = set(s)
        for J in s:                      # annihilate pair at J
            for I in range(m):           # create pair at I
                if I == J:
                    P[I, J, si, si] += 1.0        # n_I diagonal
                elif I not in occ:
                    t = tuple(sorted((occ - {J}) | {I}))
                    P[I, J, idx[t], si] += 1.0    # <t| P_I^dag P_J |s>
    return states, P


S0_BASIS, S0_P = build_sector(list(range(MP)), NPAIR)   # (10,10,252,252)
H0_DIAG_S0 = np.array([sum(E_PAIR[k] for k in s) for s in S0_STATES])


def h_s0(G):
    """H = H0 - sum_{IJ} G_IJ P_I^dag P_J on the seniority-zero sector.
    [compute.py:110-114]"""
    H = -np.einsum('ij,ijab->ab', G, S0_P)
    H[np.diag_indices(DS0)] += H0_DIAG_S0
    return 0.5 * (H + H.T)


# --------------------------------------------------------------- WLS pieces
# [compute.py:117-138]
TRIU_R, TRIU_C = np.triu_indices(MP)


def sym_col_index():
    symm_idx = np.zeros((MP, MP), dtype=int)
    idx = 0
    for I in range(MP):
        symm_idx[I, I] = idx; idx += 1
    for I in range(MP):
        for J in range(I + 1, MP):
            symm_idx[I, J] = symm_idx[J, I] = idx; idx += 1
    return symm_idx


SYMM_IDX = sym_col_index()
NPARAMS = MP * (MP + 1) // 2


def reconstruct_G_from_symm(w):
    G = np.zeros((MP, MP))
    idx = 0
    for I in range(MP):
        G[I, I] = w[idx]; idx += 1
    for I in range(MP):
        for J in range(I + 1, MP):
            G[I, J] = G[J, I] = w[idx]; idx += 1
    return G


def wls_rows(E, V):
    """Diagonal-row system: one row <k| . |k> per eigenstate [p5_shots].
    [compute.py:140-152]"""
    # A[k, col(I,J)] = <k| P_I^dag P_J + (I<->J off-diag) |k>
    # accumulate over ordered (I,J) into symmetric column index
    PV = np.tensordot(S0_P, V, axes=([3], [0]))       # (10,10,252,252k)
    diag = np.einsum('ak,ijak->ijk', V, PV, optimize=True)
    A = np.zeros((DS0, NPARAMS))
    for I in range(MP):
        for J in range(MP):
            A[:, SYMM_IDX[I, J]] += diag[I, J, :]
    W = np.hstack([A, -np.ones((DS0, 1))])            # c column (-state@state)
    h0d = np.einsum('ak,a,ak->k', V, H0_DIAG_S0, V)
    return W, h0d


def wls_core(W_rows, h0_diag, p_meas, ridge=0.0, support_drop=1e-12,
             cond=1e-9):
    """[ADAPTED SPLIT of compute.py:154-166] The weighted-lstsq assembly of
    wls_solve, column-count agnostic; returns the raw solution vector x."""
    mask = p_meas > support_drop
    p = p_meas[mask]
    w = np.sqrt(p)
    Wm = W_rows[mask] * w[:, None]
    Tm = (h0_diag[mask] + (1.0 / BETA) * np.log(p)) * w
    if ridge > 0:
        n = Wm.shape[1]
        Wm = np.vstack([Wm, math.sqrt(ridge) * np.eye(n)])
        Tm = np.concatenate([Tm, np.zeros(n)])
    x, _, _, _ = scipy.linalg.lstsq(Wm, Tm, cond=cond)
    return x


def wls_solve(W_rows, h0_diag, p_meas, ridge=0.0, support_drop=1e-12,
              cond=1e-9):
    """[compute.py:154-167; ops identical, assembly via wls_core]"""
    x = wls_core(W_rows, h0_diag, p_meas, ridge=ridge,
                 support_drop=support_drop, cond=cond)
    G = reconstruct_G_from_symm(x[:-1])
    return 0.5 * (G + G.T)


def gls_solve(W_rows, h0_diag, p_meas, N_shots, support_drop=1e-12,
              cond=1e-9):
    """Full-covariance GLS: whiten with the implemented clip-renormalize
    covariance (manuscript Eq. 20) propagated through the log (delta method),
    pseudo-inverted on the retained support.  [compute.py:169-191]"""
    mask = p_meas > support_drop
    p = p_meas[mask]
    sig2 = np.clip(p_meas * (1.0 - p_meas), 0.0, None) / N_shots
    s2 = sig2[mask]
    # Eq. 20: Cov(p_i,p_j) = sig_i^2 d_ij - p_j sig_i^2 - p_i sig_j^2
    #                        + p_i p_j sum_k sig_k^2
    S2 = sig2.sum()
    Cp = np.diag(s2) - np.outer(p, s2) - np.outer(s2, p) + np.outer(p, p) * S2
    Dinv = 1.0 / p
    Cb = (Cp * Dinv[:, None] * Dinv[None, :]) / BETA**2   # Cov of (1/b)ln p
    lam, Q = np.linalg.eigh(0.5 * (Cb + Cb.T))
    keep = lam > max(lam.max(), 0) * 1e-12
    Whalf = Q[:, keep] / np.sqrt(lam[keep])[None, :]      # C^{-1/2} cols
    Wm = Whalf.T @ W_rows[mask]
    Tm = Whalf.T @ (h0_diag[mask] + (1.0 / BETA) * np.log(p))
    x, _, _, _ = scipy.linalg.lstsq(Wm, Tm, cond=cond)
    G = reconstruct_G_from_symm(x[:-1])
    return 0.5 * (G + G.T)


def shifted_error(G_pred, G_true):
    """[compute.py:193-197] diag-mean gauge-aligned relative Frobenius error."""
    shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
    Gs = G_pred.copy()
    np.fill_diagonal(Gs, np.diag(Gs) + shift)
    return float(np.linalg.norm(Gs - G_true) / np.linalg.norm(G_true))


def draw_gaussian(p_true, N_shots, rng):
    """[compute.py:199-205]"""
    noise = rng.standard_normal(len(p_true)) * np.sqrt(
        p_true * (1.0 - p_true) / N_shots)
    p_raw = p_true + noise
    counts = np.maximum(p_raw * N_shots, 0.0)
    s = counts.sum()
    return counts / s if s > 0 else p_true.copy()


def draw_multinomial(p_true, N_shots, rng):
    """[compute.py:546-550]"""
    p_safe = p_true * (1.0 - 1e-10)
    counts = rng.multinomial(int(round(N_shots)), p_safe)
    tot = counts.sum()
    return counts / float(tot) if tot > 0 else p_true.copy()


SWEEP16 = np.geomspace(1e10, 1e2, 16)                  # [compute.py:207]
SWEEP50 = np.geomspace(1e10, 1e2, 50)                  # [compute.py:544]


def prepare_systems(n_states, seed=0):
    """[compute.py:311-320]"""
    Gb = heldout_G_batch(n_states, seed=seed)
    systems = []
    for G in Gb:
        E, V = scipy.linalg.eigh(h_s0(G))
        p = np.exp(-BETA * (E - E.min()))
        p /= p.sum()
        W, h0d = wls_rows(E, V)
        systems.append(dict(G=G, E=E, V=V, p=p, W=W, h0=h0d))
    return systems


# ============================================================== CV ridge
def cv_ridge(noise_model="gaussian", sweep=None, lam_grid=None,
             cv_n=10, cv_t=5, cv_seed=1, cv_systems=None, verbose=False):
    """[ADAPTED from compute.py stage_wls50:586-606] Per-shot-node CV
    selection of the ridge strength for wls_solve, on the seed-(4242+cv_seed)
    CV stream (default seed 1 -> PRNGKey(4243), the campaign convention).

    noise_model 'gaussian' (stream 2) or 'multinomial' (stream 3); draws use
    SeedSequence([cv_seed, stream, i_s, ci, t]) exactly as the source.
    Returns (lam_star_per_node list, scores per node per lambda).
    """
    sweep = SWEEP50 if sweep is None else np.asarray(sweep, np.float64)
    if lam_grid is None:
        lam_grid = [0.0] + [10.0 ** e for e in range(-9, 1)]   # [:586]
    stream, drawer = {"gaussian": (2, draw_gaussian),
                      "multinomial": (3, draw_multinomial)}[noise_model]
    cv = (prepare_systems(cv_n, seed=cv_seed)
          if cv_systems is None else cv_systems)
    lams, all_scores = [], []
    for i_s, N in enumerate(sweep):
        scores = []
        for lam in lam_grid:
            es = []
            for ci, s in enumerate(cv):
                for t in range(cv_t):
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [cv_seed, stream, i_s, ci, t]))
                    pm = drawer(s['p'], N, rng)
                    es.append(shifted_error(
                        wls_solve(s['W'], s['h0'], pm, ridge=lam), s['G']))
            scores.append(np.median(es))
        lams.append(lam_grid[int(np.argmin(scores))])
        all_scores.append([float(x) for x in scores])
        if verbose:
            print('cv_ridge %s node %d/%d N=%.2e lam*=%g'
                  % (noise_model, i_s + 1, len(sweep), N, lams[-1]),
                  flush=True)
    return lams, all_scores


# ============================================================ forward-map fit
def fwdmap_fit(G_true, E, V, p_true, i_s, si, trials, N, stream=2,
               map_mode=False, prior_sig=0.26, tol=1e-10, max_nfev=4000):
    """[ADAPTED from compute.py _fwd_fit_one:385-431] Hellinger forward-map
    fit (no log, no support drop) for one system at one shot count.

    min_G sum_n (sqrt(q_n(G)) - sqrt(p_meas_n))^2 in the supplied exact
    eigenbasis V; x0 = prior mean G=0.55; draws SeedSequence
    ([0, stream, i_s, si, t]) (the source's fwdmap used stream 2).
    map_mode adds the N-scaled Gaussian prior penalty rows (round-B MAP).
    Returns (rows [shifted_error per trial], nev_tot).
    """
    import scipy.optimize
    res_rows = []
    x_prior = np.zeros(NPARAMS)
    Gp = np.full((MP, MP), 0.55)
    x_prior[:MP] = np.diag(Gp)
    k = MP
    for I in range(MP):
        for J in range(I + 1, MP):
            x_prior[k] = Gp[I, J]; k += 1
    nev_tot = 0
    _map = bool(map_mode)
    _sig = float(prior_sig)
    for t in trials:
        rng = np.random.default_rng(np.random.SeedSequence(
            [0, int(stream), int(i_s), int(si), int(t)]))
        pm = draw_gaussian(p_true, N, rng) if stream == 2 else \
            draw_multinomial(p_true, N, rng)
        sq = np.sqrt(pm)
        _pw = (1.0 / (2.0 * _sig * np.sqrt(N))) if _map else 0.0

        def resid(x):
            G = reconstruct_G_from_symm(x)
            Ei, Ui = np.linalg.eigh(h_s0(G))
            pi = np.exp(-BETA * (Ei - Ei.min()))
            pi /= pi.sum()
            M2 = (Ui.T @ V) ** 2          # |<u_m|Psi_n>|^2
            q = pi @ M2
            r = np.sqrt(np.maximum(q, 0.0)) - sq
            if _map:
                return np.concatenate([r, _pw * (x - x_prior)])
            return r

        sol = scipy.optimize.least_squares(resid, x_prior, method='trf',
                                           xtol=tol, ftol=tol, gtol=tol,
                                           max_nfev=max_nfev)
        nev_tot += sol.nfev * (NPARAMS + 1)   # 2-point jac evals approx
        Gf = reconstruct_G_from_symm(sol.x)
        res_rows.append(shifted_error(0.5 * (Gf + Gf.T), G_true))
    return res_rows, nev_tot


# ===================================================== secular (campaign6)
# [s6_secular_protocol.py:114-118]
COND = 1e-9            # published lstsq cond; also the off-diag SVD cutoffs
NULL_RTOL = 1e-9       # sigma <= NULL_RTOL*sigma_max -> off-diag null space
D = DS0                # 252
NP55 = NPARAMS         # 55
IU_M, IU_N = np.triu_indices(D, k=1)             # 31626 off-diagonal pairs


def offdiag_system(s):
    """Exact off-diagonal secular block for one prepared system.
    [s6_secular_protocol.py:135-153]"""
    V = s['V']
    T1 = np.tensordot(S0_P, V, axes=([3], [0]))        # (10,10,252,252n)
    B = np.einsum('am,ijan->ijmn', V, T1, optimize=True)  # (10,10,252,252)
    A_cols = np.empty((NP55, D, D))
    for I in range(MP):
        A_cols[SYMM_IDX[I, I]] = B[I, I]
        for J in range(I + 1, MP):
            A_cols[SYMM_IDX[I, J]] = B[I, J] + B[J, I]
    A_od = A_cols[:, IU_M, IU_N].T                        # (31626, 55)
    Hod = V.T @ (H0_DIAG_S0[:, None] * V)                 # <m|H0|n>
    t_od = Hod[IU_M, IU_N]
    U, svals, Vt = scipy.linalg.svd(A_od, full_matrices=False)
    return dict(A_od=A_od, t_od=t_od, U=U, svals=svals, Vt=Vt)


def offdiag_solution(od):
    """Min-norm truncated-SVD solution x_p (55,) of A_od x = t_od at rel.
    cutoff COND, and null-space basis Z (55, n_null) at NULL_RTOL.
    [s6_secular_protocol.py:156-165]"""
    U, svals, Vt = od['U'], od['svals'], od['Vt']
    smax = svals[0]
    keep = svals > COND * smax
    x_p = Vt[keep].T @ ((U[:, keep].T @ od['t_od']) / svals[keep])
    null = svals <= NULL_RTOL * smax
    Z = Vt[null].T                                        # (55, n_null)
    return x_p, Z


def commutant_gauged_error(G_pred, G_true):
    """[s6_secular_protocol.py:168-180] Frobenius error after removing the
    best-fit component along the analytic off-diag kernel span{I, H_true}."""
    K1 = np.eye(MP)
    K2 = np.diag(E_PAIR) - G_true
    dG = G_pred - G_true
    M = np.array([[np.sum(K1 * K1), np.sum(K1 * K2)],
                  [np.sum(K2 * K1), np.sum(K2 * K2)]])
    b = np.array([np.sum(K1 * dG), np.sum(K2 * dG)])
    ab, *_ = np.linalg.lstsq(M, b, rcond=None)
    R = dG - ab[0] * K1 - ab[1] * K2
    return float(np.linalg.norm(R) / np.linalg.norm(G_true))


def diag_rows_weighted(s, p_meas):
    """[s6_secular_protocol.py:183-191] Published diagonal-row pieces for one
    draw, exactly as wls_solve: support drop p<=1e-12, sqrt(p) weights."""
    mask = p_meas > 1e-12
    p = p_meas[mask]
    w = np.sqrt(p)
    W_d = s['W'][mask] * w[:, None]
    T_d = (s['h0'][mask] + (1.0 / BETA) * np.log(p)) * w
    return W_d, T_d


def solve_full_hard(s, od_cache, p_meas):
    """Off-diag rows as hard constraints; diag rows fix only (y, c).
    [s6_secular_protocol.py:194-204]"""
    x_p, Z = od_cache
    W_d, T_d = diag_rows_weighted(s, p_meas)
    W_G, w_c = W_d[:, :NP55], W_d[:, NP55]
    Msmall = np.column_stack([W_G @ Z, w_c])              # (k, n_null+1)
    rhs = T_d - W_G @ x_p
    y, *_ = scipy.linalg.lstsq(Msmall, rhs, cond=COND)
    x55 = x_p + Z @ y[:-1]
    G = reconstruct_G_from_symm(x55)
    return 0.5 * (G + G.T)


def solve_full_soft(s, od, p_meas):
    """Stacked lstsq(cond=1e-9): unit-weight off-diag rows (via SVD row
    reduction R = diag(s)Vt) + published weighted diag rows.
    [s6_secular_protocol.py:207-218]"""
    W_d, T_d = diag_rows_weighted(s, p_meas)
    R_od = od['svals'][:, None] * od['Vt']                # (55, 55)
    q_od = od['U'].T @ od['t_od']                         # (55,)
    top = np.hstack([R_od, np.zeros((NP55, 1))])          # zero c column
    Wm = np.vstack([top, W_d])
    Tm = np.concatenate([q_od, T_d])
    x, *_ = scipy.linalg.lstsq(Wm, Tm, cond=COND)
    G = reconstruct_G_from_symm(x[:-1])
    return 0.5 * (G + G.T)


def x_true_symm(G):
    """[s6_secular_protocol.py:221-228]"""
    x = np.zeros(NP55)
    x[:MP] = np.diag(G)
    k = MP
    for I in range(MP):
        for J in range(I + 1, MP):
            x[k] = G[I, J]; k += 1
    return x


def build_system(G):
    """Rebuild one prepared system from G exactly as prepare_systems.
    [s6_secular_protocol.py:232-239]"""
    E, V = scipy.linalg.eigh(h_s0(G))
    p = np.exp(-BETA * (E - E.min()))
    p /= p.sum()
    W, h0d = wls_rows(E, V)
    return dict(G=G, E=E, V=V, p=p, W=W, h0=h0d)


def secular_solve(s, p_meas=None, arm="full_hard", od=None):
    """[ADAPTED WRAPPER] One arm of the campaign6 secular protocol for one
    prepared system `s` (from prepare_systems/build_system).

    arm 'published'    -> wls_solve on the diagonal-log-population rows
        'full_hard'    -> off-diag rows as hard constraints (min-norm SVD
                          x_p + null basis Z) + weighted diagonal rows
        'full_soft'    -> single stacked lstsq (unit-weight off-diag SVD
                          reduction + weighted diagonal rows)
        'offdiag_only' -> G from x_p alone, ZERO shots (p_meas unused)
    `od` (the offdiag_system dict) is computed on demand and can be passed
    in to amortize the SVD across draws.  Returns (G_pred, od).
    """
    if arm == "published":
        if p_meas is None:
            raise ValueError("published arm needs p_meas")
        return wls_solve(s['W'], s['h0'], p_meas), od
    if od is None:
        od = offdiag_system(s)
    if arm == "offdiag_only":
        x_p, _Z = offdiag_solution(od)
        G = reconstruct_G_from_symm(x_p)
        return 0.5 * (G + G.T), od
    if p_meas is None:
        raise ValueError("%s arm needs p_meas" % arm)
    if arm == "full_hard":
        return solve_full_hard(s, offdiag_solution(od), p_meas), od
    if arm == "full_soft":
        return solve_full_soft(s, od, p_meas), od
    raise ValueError("unknown arm %r" % (arm,))


# ================================================ prior ridge / box MAP (c5)
# [s_prior_ridge.py:140-146]
X_PRIOR = np.full(NPARAMS, 0.55)
SUPPORT_DROP = 1e-12
BOX_LB = np.concatenate([np.full(MP, -0.26),
                         np.full(NPARAMS - MP, 0.1), [-np.inf]])
BOX_UB = np.concatenate([np.full(MP, 1.36),
                         np.full(NPARAMS - MP, 1.0), [np.inf]])


def prior_ridge(W, h0d, pm, lam=0.0, x_prior=None, energy=None):
    """Published WLS core (identical ops to wls_solve for lam=0/energy=None)
    + optional prior-centered ridge rows (55 G params only, c free) and/or
    the weighted E_meas row.  [s_prior_ridge.py _solve:160-185; ADAPTED:
    x_prior=None defaults to X_PRIOR]"""
    if x_prior is None:
        x_prior = X_PRIOR
    mask = pm > SUPPORT_DROP
    p = pm[mask]
    w = np.sqrt(p)
    blocks = [W[mask] * w[:, None]]
    tgts = [(h0d[mask] + (1.0 / BETA) * np.log(p)) * w]
    if energy is not None:
        evals, _N = energy
        aE = (pm @ W).astype(np.float64)
        aE[-1] = 0.0                                  # c column
        E_meas = float(pm @ evals)
        varE = float(pm @ (evals - E_meas) ** 2)
        wE = 1.0 / math.sqrt(max(varE, 1e-30))
        blocks.append(aE[None, :] * wE)
        tgts.append(np.array([(float(pm @ h0d) - E_meas) * wE]))
    if lam > 0.0:
        blocks.append(np.hstack([math.sqrt(lam) * np.eye(NPARAMS),
                                 np.zeros((NPARAMS, 1))]))
        tgts.append(math.sqrt(lam) * x_prior)
    x, _, _, _ = scipy.linalg.lstsq(np.vstack(blocks), np.concatenate(tgts),
                                    cond=COND)
    G = reconstruct_G_from_symm(x[:-1])
    return 0.5 * (G + G.T)


def box_map(W, h0d, pm):
    """MAP under the training box prior = box-constrained WLS (no cond
    truncation, no ridge, no energy row).  [s_prior_ridge.py _solve_box:
    188-205]"""
    from scipy.optimize import lsq_linear
    mask = pm > SUPPORT_DROP
    p = pm[mask]
    w = np.sqrt(p)
    A = W[mask] * w[:, None]
    T = (h0d[mask] + (1.0 / BETA) * np.log(p)) * w
    try:
        res = lsq_linear(A, T, bounds=(BOX_LB, BOX_UB), method='bvls')
    except Exception:
        res = lsq_linear(A, T, bounds=(BOX_LB, BOX_UB), method='trf',
                         lsq_solver='exact')
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError('box solve returned non-finite x')
    G = reconstruct_G_from_symm(res.x[:-1])
    return 0.5 * (G + G.T)


# ====================================== input-matched fit (x22, d16 blocked)
# [x22_fitu_pilot.py:156-177]
D16_D_SP, D16_N_ELEC, D16_M = 16, 8, 8
D16_BASIS = 12_870                            # C(16,8)
G_INIT, G_STOP = 0.1, 1.0
GRID_N = 65
XTOL = 1e-10
MAXITER = 200
TOL_OBJ = 1e-10
INVPHI = (math.sqrt(5.0) - 1.0) / 2.0
ENERGY_NORMALIZER_MIN = 1.0                   # gate: |e_t| must exceed this


def input_matched_setup(m=D16_M, n=D16_N_ELEC):
    """[ADAPTED from x22 _fwd_setup:409-414] Build (eps, blocks) for the
    blocked forward map via ognrepro.blocked (the VERBATIM solver the x22
    stage loaded from the verified d12 stage file)."""
    from . import blocked as S
    return S.ladder(m), S.build_blocks(m, n)


def input_matched_forward(g_batch, eps, blocks, m=D16_M, basis=D16_BASIS,
                          beta=BETA):
    """The declared exact blocked forward map + f32 serialization.
    g_batch (B,) f64 -> (x_m (B,m,m) f64-of-f32, e_m (B,) f64-of-f32).
    [x22 forward_serialized:417-430; ADAPTED: (eps, blocks) explicit]"""
    from . import blocked as S
    g = np.asarray(g_batch, np.float64).reshape(-1)
    V = np.stack([S.v_matrix("const", m, g[i:i + 1])
                  for i in range(g.shape[0])])
    r2, ee, _z, _e0, nst, _sg = S.blocked_solve(blocks, eps, V, m, beta)
    if nst != basis:
        raise RuntimeError("input_matched_forward: nstates %d != %d"
                           % (nst, basis))
    x_m = np.float32(r2).astype(np.float64)
    e_m = np.float32(ee).astype(np.float64)
    return x_m, e_m


def input_matched_loss(g_batch, x_t, e_t, eps, blocks, m=D16_M,
                       basis=D16_BASIS, beta=BETA):
    """Dimensionless declared loss for a batch of couplings vs ONE target.
    [x22 loss_batch:433-439]"""
    x_m, e_m = input_matched_forward(g_batch, eps, blocks, m, basis, beta)
    xn = float(np.sum(x_t * x_t))
    pair = np.sum((x_m - x_t[None]) ** 2, axis=(1, 2)) / xn
    en = (e_m - e_t) ** 2 / (e_t * e_t)
    return pair + en


def input_matched_fit(x_t, e_t, eps, blocks, m=D16_M, basis=D16_BASIS,
                      beta=BETA, g_init=G_INIT, g_stop=G_STOP,
                      grid_n=GRID_N, xtol=XTOL, maxiter=MAXITER,
                      tol_obj=TOL_OBJ):
    """Truth-blind deterministic fit: grid_n-point coarse grid (one batched
    solve) + golden-section refinement on the bracketing neighbors of the
    grid argmin.  [x22 fit_one_target:442-485; ADAPTED: (eps, blocks) and
    the declared constants are keyword arguments with the pilot's values]"""
    import time
    if abs(float(e_t)) < ENERGY_NORMALIZER_MIN:
        raise RuntimeError("input_matched_fit REFUSED: |e_t| below the "
                           "declared normalizer gate")
    t0w, t0c = time.perf_counter(), time.process_time()
    grid = np.linspace(g_init, g_stop, grid_n)
    gl = input_matched_loss(grid, x_t, e_t, eps, blocks, m, basis, beta)
    n_evals = grid_n
    i = int(np.argmin(gl))
    best_g, best_l = float(grid[i]), float(gl[i])
    a = float(grid[max(i - 1, 0)])
    b = float(grid[min(i + 1, grid_n - 1)])

    state = dict(n_evals=n_evals, best_g=best_g, best_l=best_l)

    def f(g):
        l = float(input_matched_loss(np.array([g]), x_t, e_t, eps, blocks,
                                     m, basis, beta)[0])
        state["n_evals"] += 1
        if l < state["best_l"]:
            state["best_g"], state["best_l"] = float(g), l
        return l

    c = b - (b - a) * INVPHI
    d = a + (b - a) * INVPHI
    fc, fd = f(c), f(d)
    it = 0
    while (b - a) > xtol and it < maxiter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * INVPHI
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * INVPHI
            fd = f(d)
        it += 1
    term = "xtol" if (b - a) <= xtol else "maxiter"
    conv = bool(term == "xtol" and state["best_l"] <= tol_obj)
    return dict(ghat=state["best_g"], loss_final=state["best_l"],
                n_evals=state["n_evals"], golden_iters=it,
                bracket_width=float(b - a), term=term, converged=conv,
                wall_s=time.perf_counter() - t0w,
                cpu_s=time.process_time() - t0c)
