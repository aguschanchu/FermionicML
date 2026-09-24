# =============================================================================
# p5_shots.py — engine part 5: finite-shot noise protocol (manuscript Sec V.C)
#
# WLS log-inversion vs OGN under Gaussian-CLT and multinomial shot noise.
# Exec'd into the shared engine namespace AFTER p1_core..p4_gevp; relies on
# late-bound era globals set by the p1_core init:
#   basis, rho_1_arrays, rho_2_kkbar_arrays (or rho_2_arrays), U_ENERGY_SEED,
#   BETA, g_gen, GGenerator, two_body_hamiltonian_sp,
#   _prepare_hamiltonian_for_eigsh, compute_rho_m, eval_step
#
# Sources (provenance markers inline):
#   - notebook export FermionicML_thermal_2body.py l.2620-2686
#     ([REGEN-EXTRACT wls_system]: build_thermal_linear_system,
#      compute_thermal_pseudoinverse)
#   - export l.4455-4722 (published Sec V.C protocol,
#     plot_asymptotic_shot_noise_resilience_wls)
#   - .claude/cells/cell_63.py (multinomial discrete-counting variant)
#   - .claude/cells/cell_64.py (cell form of the published protocol)
#   - campaign/reference/values_prev.json 'shots' block (output schema; the
#     analysis logic mirrors the prior-campaign regen driver that produced it)
# =============================================================================

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

# ----------------------------------------------------------------- constants
# Held-out label seed base: distinct from dataset seeds 42/43 (train halves)
# and 1007 (validation); the protocol Hamiltonians are FRESH draws, never seen
# by any loader.
SHOTS_HELDOUT_SEED = 4242
# Deterministic noise streams: SeedSequence([seed, stream, shot, state, trial])
# Stream ids match the prior campaign (2 = gaussian, 3 = multinomial) so that
# seed=0 reproduces the values_prev.json draws exactly.
SHOTS_NOISE_STREAM = {"gaussian": 2, "multinomial": 3}
SHOTS_WLS_DROP_P = 1e-12         # drop rows with p <= 1e-12 (published)
SHOTS_WLS_CUTOFF = 1e-9          # lstsq spectral cutoff (published cond=1e-9)
SHOTS_DEPARTURE_TOL = 0.05       # |median ratio - 1| > 5% = "departure"


# ------------------------------------------------------------- small helpers
def _safe_dense(arr):
    # [from export l.4467-4470]
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)


def reconstruct_G_from_symm(w_symm, M_pairs):
    # [from export l.4472-4482]
    G = np.zeros((M_pairs, M_pairs), dtype=w_symm.dtype)
    idx = 0
    for I in range(M_pairs):
        G[I, I] = w_symm[idx]
        idx += 1
    for I in range(M_pairs):
        for J in range(I + 1, M_pairs):
            G[I, J] = G[J, I] = w_symm[idx]
            idx += 1
    return G


def compute_W_symm_pairing_nd(state, rho_2_kkbar):
    # [from export l.2831-2856] verbatim duplicate (needed at call time by the
    # WLS system; identical to the copy carried by the inversion parts).
    M_pairs = rho_2_kkbar.shape[0]
    D_N = rho_2_kkbar.shape[2]
    num_symm = M_pairs * (M_pairs + 1) // 2

    W_symm = np.zeros((D_N, num_symm), dtype=np.float64)
    symm_idx = np.zeros((M_pairs, M_pairs), dtype=int)

    idx = 0
    for I in range(M_pairs):
        symm_idx[I, I] = idx
        idx += 1
    for I in range(M_pairs):
        for J in range(I + 1, M_pairs):
            symm_idx[I, J] = symm_idx[J, I] = idx
            idx += 1

    coords = rho_2_kkbar.coords
    data = rho_2_kkbar.data
    J_coords, I_coords = coords[0], coords[1]
    r_coords, c_coords = coords[2], coords[3]

    cols = symm_idx[I_coords, J_coords]
    vals = data * np.asarray(state)[c_coords]
    np.add.at(W_symm, (r_coords, cols), vals.real)
    return W_symm, symm_idx


# ----------------------------------------------------- WLS thermal system
# [REGEN-EXTRACT-BEGIN wls_system]  [from export l.2621-2668]
def build_thermal_linear_system(evals, evecs, H0_dense, rho_2_kkbar, beta, truncate_tol=1e-12):
    """
    Constructs the diagonal-row W and T matrices for the Thermal Oracle.

    [REGEN-PATCH P3] One scalar equation per populated eigenstate (the
    eigenstate projection <k|.|k>), matching manuscript Sec. III.C.2:
    "252 rows ... against 56 unknowns" (55 G params + partition scalar).
    The previous version stacked the full 252-dim blocks (W_aug_k, target_k),
    silently imposing the noise-free OFF-diagonal constraints that the text
    explicitly excludes ("The off-diagonal equations are not fitted: with
    oracle eigenvectors they carry no shot noise and would determine G
    independently of N_shots").
    """
    # 1. Compute thermal probabilities (Boltzmann weights)
    E_shifted = evals - evals.min()
    probs = np.exp(-beta * E_shifted)
    probs /= np.sum(probs)

    # 2. Filter states with negligible population to prevent log(0) blowups
    valid_mask = probs > truncate_tol
    probs = probs[valid_mask]
    vecs = evecs[:, valid_mask]

    W_stacked, T_stacked = [], []

    # 3. Iterate over populated microstates
    for k in range(len(probs)):
        p_k = probs[k]
        state_k = vecs[:, k]

        # A. Action matrix for this eigenstate
        W_symm_k, _ = compute_W_symm_pairing_nd(state_k, rho_2_kkbar)

        # B. Augment W to absorb the partition function shift (c)
        # We append state_k as a column to fit the scalar variable 'c'
        W_aug_k = np.hstack([W_symm_k, state_k.reshape(-1, 1)])

        # C. Target Vector: (H0 + 1/beta * ln(p_k)) |k>
        target_k = (H0_dense @ state_k) + (1.0 / beta) * np.log(p_k) * state_k

        # D. Eigenstate projection -> one weighted scalar row  [REGEN-PATCH P3]
        W_row_k = state_k @ W_aug_k        # shape (n_params + 1,)
        t_row_k = state_k @ target_k       # scalar
        weight = np.sqrt(p_k)
        W_stacked.append(weight * W_row_k)
        T_stacked.append(weight * t_row_k)

    return np.vstack(W_stacked), np.asarray(T_stacked)
# [REGEN-EXTRACT-END wls_system]


def compute_thermal_pseudoinverse(evals, evecs, H0_dense, rho_2_kkbar, beta, M_pairs, truncate_tol=1e-12):
    # [from export l.2671-2686]
    """
    Solves the Inverse Problem using the Thermal Ensemble.
    """
    W_thermal, T_thermal = build_thermal_linear_system(evals, evecs, H0_dense, rho_2_kkbar, beta, truncate_tol)

    # Massively overdetermined solve
    x_pinv, res, rank, s = scipy.linalg.lstsq(W_thermal, T_thermal, cond=1e-9)

    # Extract structural parameters and reconstruct (drop the last 'c' shift element)
    w_pinv = x_pinv[:-1]

    G_pinv = reconstruct_G_from_symm(w_pinv, M_pairs).real
    G_pinv = 0.5 * (G_pinv + G_pinv.T)

    return G_pinv, rank, s


# ------------------------------------------------------ protocol primitives
def _shots_shifted_error(G_pred, G_true, norm_g):
    # [from export l.4631-4635] gauge-aligned ||dG||_F / ||G||_F
    shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
    G_shifted = G_pred.copy()
    np.fill_diagonal(G_shifted, np.diag(G_shifted) + shift)
    return float(np.linalg.norm(G_shifted - G_true) / norm_g)


def _shots_wls_solve(sys_data, p_meas, beta, M_pairs, support_mask=None):
    # [from export l.4609-4624] sqrt(p)-weighted WLS on the diagonal-row
    # system with the published p > 1e-12 row drop and cond=1e-9 spectral
    # cutoff. support_mask hook = noiseless-gate solve (full support).
    if support_mask is None:
        # 1. Isolate the strictly valid subspace (Filters zero-probability states)
        support_mask = p_meas > SHOTS_WLS_DROP_P
    p_valid = p_meas[support_mask]

    # Variance of ln(p_k) is ~1/p_k. Optimal weight is proportional to sqrt(p_k).
    weight = np.sqrt(p_valid)

    # [REGEN-PATCH P3] Diagonal-row system: (<=252) x 56, one
    # eigenstate-projected equation per retained microstate.
    W_thermal = sys_data['W_rows'][support_mask] * weight[:, None]
    T_thermal = (sys_data['h0_diag'][support_mask]
                 + (1.0 / beta) * np.log(p_valid)) * weight

    w_log_wls, _, _, _ = scipy.linalg.lstsq(W_thermal, T_thermal, cond=SHOTS_WLS_CUTOFF)
    G_log_wls = 0.5 * (reconstruct_G_from_symm(w_log_wls[:-1], M_pairs).real
                       + reconstruct_G_from_symm(w_log_wls[:-1], M_pairs).real.T)
    return G_log_wls, int(np.sum(~support_mask))


def _shots_draw_gaussian(p_true, N_shots, rng):
    # [from export l.4570-4583] Asymptotic measurement using Central Limit
    # Theorem; eq (noise_model) Gaussian, var p(1-p)/N.
    # [REGEN-PATCH P4] no sum-zero recentering; the renormalization below
    # already restores sum(p) = 1.
    noise = rng.standard_normal(len(p_true)) * np.sqrt(p_true * (1.0 - p_true) / N_shots)
    p_raw = p_true + noise

    # Zero-clipped normalization (No Laplace Smoothing)
    pseudo_counts = np.maximum(p_raw * N_shots, 0.0)
    sum_counts = np.sum(pseudo_counts)

    # Normalize or fallback if statistical noise wipes out everything
    p_meas = pseudo_counts / sum_counts if sum_counts > 0 else p_true.copy()
    return p_meas


def _shots_draw_multinomial(p_true, N_shots, rng):
    # [from cell_63 l.146-154] exact multinomial counts / N_shots via numpy
    # Generator.multinomial; no further noise. The (1 - 1e-10) scaling
    # guarantees sum(p) < 1, bypassing the C-level multinomial crash.
    p_safe_draw = p_true * (1.0 - 1e-10)
    counts = rng.multinomial(int(round(N_shots)), p_safe_draw)

    # Renormalize by actual drawn counts to ensure trace = 1.0
    total_counts = np.sum(counts)
    p_meas = counts / float(total_counts) if total_counts > 0 else p_true.copy()
    return p_meas


# --------------------------------------------------- held-out true systems
def _shots_h_dense64(G_mat):
    """float64 dense Hamiltonian from the era sparse builder.

    The float32 two_body_hamiltonian_dense path of the published function caps
    the noiseless diagonal-row recovery at ~1e-5 and fails the p3_noiseless
    gate (values_prev.json: median 2.9e-13); host-side diagnostics therefore
    use the float64 sparse builder (same construction as the covariance suite).
    """
    M_pairs = basis.d // 2
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
    e_arr64 = U_ENERGY_SEED[0:1].astype(np.float64)
    H_sp = two_body_hamiltonian_sp(basis, e_arr64,
                                   np.asarray([G_mat], dtype=np.float64),
                                   rho_1_arrays, target_rho2_coo,
                                   g_gen.h_type)[0]
    H = _prepare_hamiltonian_for_eigsh(H_sp)
    H = H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
    return 0.5 * (H + H.T)


def _shots_heldout_G_batch(n_states, seed=0):
    """Fresh held-out interaction matrices: labels drawn via the era GGenerator
    config (g_gen.h_type / g_init / g_stop) from PRNGKey(SHOTS_HELDOUT_SEED+seed).

    [spec edit] Replaces the published predict_and_load(val_dataset, ...)
    extraction: the protocol Hamiltonians are held out from every dataset."""
    gen_holdout = GGenerator(basis, g_gen.h_type, int(n_states),
                             g_init=g_gen.g_init, g_stop=g_gen.g_stop)
    key = jax.random.PRNGKey(SHOTS_HELDOUT_SEED + int(seed))
    _, g_true_labels = gen_holdout.generate(key)
    G_true_batch = np.array(gen_holdout.reconstruct(jnp.array(g_true_labels))
                            ).astype(np.float64)
    return G_true_batch


def prepare_heldout_systems(n_states, seed=0, with_wls_rows=True, progress=True):
    """Build the per-Hamiltonian exact thermal data used by the protocol.

    For each held-out G_true: exact eigh (float64), Boltzmann populations at
    the era BETA, and (optionally) the precomputed diagonal-row WLS system
    (W_rows, h0_diag).  [from export l.4524-4550, data flow preserved]
    """
    M_pairs = basis.d // 2
    D_N = basis.size
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
    beta = float(BETA)

    G_true_batch = _shots_heldout_G_batch(n_states, seed)
    H0_dense = _shots_h_dense64(np.zeros((M_pairs, M_pairs)))

    true_systems = []
    it = range(int(n_states))
    if progress:
        it = tqdm(it, desc="shots precompute", leave=False)
    for i in it:
        G_true = G_true_batch[i]
        E_true, V_true = scipy.linalg.eigh(_shots_h_dense64(G_true))
        p_true = np.exp(-beta * (E_true - E_true.min())).astype(np.float64)
        p_true /= np.sum(p_true)

        sys_data = {
            'G_true': G_true, 'E_true': E_true, 'V_true': V_true,
            'p_true': p_true, 'norm_g_true': np.linalg.norm(G_true),
        }
        if with_wls_rows:
            # [REGEN-PATCH P3] Precompute the DIAGONAL eigenstate projections
            # only: one row <k|.|k> per eigenstate (manuscript Sec. III.C.2;
            # the off-diagonal equations are noise-free and must not be
            # fitted).  [from export l.4536-4549]
            sys_W_rows = []
            sys_h0_diag = []
            for k in range(D_N):
                state_k = V_true[:, k]
                W_symm_k, _ = compute_W_symm_pairing_nd(state_k, target_rho2_coo)
                W_aug_k = np.hstack([W_symm_k, -state_k.reshape(-1, 1)])
                sys_W_rows.append(state_k @ W_aug_k)              # (n_params + 1,)
                sys_h0_diag.append(float(state_k @ (H0_dense @ state_k)))
            sys_data['W_rows'] = np.array(sys_W_rows)
            sys_data['h0_diag'] = np.array(sys_h0_diag)
        true_systems.append(sys_data)
    return true_systems, H0_dense


# --------------------------------------------------------- public protocol
def run_shots_protocol(state_model, n_states=100, n_noise=100, n_shots_sweep=None,
                       noise_model='gaussian', seed=0, include_energy=True):
    """Sec V.C finite-shot protocol on held-out Hamiltonians at the era BETA.

    Per (shot point, state, draw) the SAME noisy population vector p_meas is
    supplied to BOTH estimators: the OGN (2-RDM + energy of the measured
    ensemble) and the sqrt(p)-weighted WLS log inversion. Returns median-ready
    per-(shots, state, draw) error arrays (float64, unclipped) for both.

    noise_model: 'gaussian' (CLT noise + zero-clip + renormalize, published)
                 or 'multinomial' (exact counts/N_shots, cell_63 variant).
    """
    if noise_model not in SHOTS_NOISE_STREAM:
        raise ValueError(f"noise_model must be one of {list(SHOTS_NOISE_STREAM)}")
    stream_id = SHOTS_NOISE_STREAM[noise_model]
    drawer = (_shots_draw_gaussian if noise_model == 'gaussian'
              else _shots_draw_multinomial)

    M_pairs = basis.d // 2
    beta = float(BETA)
    target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
    target_rho2_dense = _safe_dense(target_rho2)

    if n_shots_sweep is None:
        # Focus the computational sweep strictly on the realistic experimental
        # sparsity frontier  [from export l.4513]
        n_shots_sweep = np.geomspace(1e10, 1e2, 50)
    N_SHOTS_SWEEP = np.asarray(n_shots_sweep, dtype=np.float64)
    EPSILONS = 1.0 / np.sqrt(N_SHOTS_SWEEP)

    print(f"[1/3] Preparing {n_states} held-out Hamiltonians "
          f"(seed base {SHOTS_HELDOUT_SEED}+{seed}) and tensor actions...")
    true_systems, _H0 = prepare_heldout_systems(n_states, seed=seed,
                                                with_wls_rows=True)

    # Noiseless diagonal-row recovery (gate p3_noiseless input): full-support
    # WLS on the exact populations must reproduce G to numerical precision.
    noiseless_wls_err = np.full(int(n_states), np.nan)
    for i, sys_data in enumerate(true_systems):
        try:
            G_rec, _ = _shots_wls_solve(
                sys_data, sys_data['p_true'], beta, M_pairs,
                support_mask=np.ones_like(sys_data['p_true'], dtype=bool))
            noiseless_wls_err[i] = _shots_shifted_error(
                G_rec, sys_data['G_true'], sys_data['norm_g_true'])
        except Exception:
            pass

    shape = (len(N_SHOTS_SWEEP), int(n_states), int(n_noise))
    err_wls = np.full(shape, np.nan)
    err_ogn = np.full(shape, np.nan)
    n_censored = np.zeros(shape, dtype=np.int32)

    print(f"[2/3] Simulating {noise_model} shot noise & WLS filtering...")
    for idx_sweep, N_shots in enumerate(tqdm(N_SHOTS_SWEEP,
                                             desc=f"Sweeping Finite Shots ({noise_model})",
                                             leave=False)):
        rdm_batch, e_batch, meta_batch = [], [], []

        for state_idx, sys_data in enumerate(true_systems):
            p_true = sys_data['p_true']
            V_true = sys_data['V_true']

            for trial in range(int(n_noise)):
                # [REGEN-PATCH SEED] collision-free deterministic stream per
                # (shot, state, trial); same draw feeds both estimators.
                # [from export l.4566-4568; stream_id 2/3 = gaussian/multinomial]
                rng = np.random.default_rng(np.random.SeedSequence(
                    [int(seed), stream_id, idx_sweep, state_idx, trial]))

                p_meas = drawer(p_true, N_shots, rng)

                # [from export l.4585-4591] measured ensemble -> OGN inputs
                rho_meas = (V_true * p_meas) @ V_true.T
                RDM_meas = compute_rho_m(np.array([rho_meas]), target_rho2_dense, 1)[0]
                E_meas = np.sum(p_meas * sys_data['E_true'])

                rdm_batch.append(RDM_meas)
                e_batch.append([E_meas])
                meta_batch.append({'p_meas': p_meas, 'state_idx': state_idx,
                                   'trial': trial, 'sys_data': sys_data})

        # [from export l.4593-4599] batched OGN inference on the same draws.
        # [engine fix 2026-06-11] CHUNKED: a single 10000-sample eval_step
        # OOMs TPU HBM (edge tensors ~ B x M x M x 1537 channels); 512-sample
        # chunks keep peak activation memory under ~2 GB with identical output.
        bx_all = np.asarray(rdm_batch).real
        if bx_all.ndim == 3:
            bx_all = bx_all[..., np.newaxis]
        be_all = np.asarray(e_batch).real if include_energy else None
        EVAL_CHUNK = 512
        g_parts = []
        for s in range(0, len(rdm_batch), EVAL_CHUNK):
            e = min(s + EVAL_CHUNK, len(rdm_batch))
            bx = jnp.array(bx_all[s:e])
            be = jnp.array(be_all[s:e]) if include_energy else None
            by_dummy = jnp.zeros((e - s, g_gen.label_size()))
            logits = eval_step(state_model, bx, be, by_dummy)
            g_parts.append(np.array(g_gen.reconstruct(logits)))
        G_ml_batch = np.concatenate(g_parts, axis=0)

        for i, meta in enumerate(meta_batch):
            p_meas = meta['p_meas']
            sys_data = meta['sys_data']
            state_idx, trial = meta['state_idx'], meta['trial']
            G_true, norm_g = sys_data['G_true'], sys_data['norm_g_true']

            # WEIGHTED LEAST SQUARES (INVERSE-VARIANCE WEIGHTING)
            # [from export l.4609-4626]
            try:
                G_log_wls, nc = _shots_wls_solve(sys_data, p_meas, beta, M_pairs)
            except Exception:
                G_log_wls = np.zeros((M_pairs, M_pairs))
                nc = -1
            n_censored[idx_sweep, state_idx, trial] = nc

            # Fair Gauge Shift & Error Evaluation  [from export l.4629-4641]
            try:
                err_ogn[idx_sweep, state_idx, trial] = _shots_shifted_error(
                    G_ml_batch[i], G_true, norm_g)
            except Exception:
                pass
            try:
                err_wls[idx_sweep, state_idx, trial] = _shots_shifted_error(
                    G_log_wls, G_true, norm_g)
            except Exception:
                pass

    print("[3/3] Protocol sweep complete.")
    # [REGEN-PATCH STATS] all statistics downstream operate on UNCLIPPED
    # errors; any display clip is applied at plot time only.
    return {
        'noise_model': noise_model,
        'seed': int(seed),
        'beta': beta,
        'include_energy': bool(include_energy),
        'n_states': int(n_states),
        'n_noise': int(n_noise),
        'n_shots_sweep': N_SHOTS_SWEEP,
        'epsilons': EPSILONS,
        'err_wls': err_wls,                  # (n_shots, n_states, n_noise)
        'err_ogn': err_ogn,                  # (n_shots, n_states, n_noise)
        'n_censored': n_censored,            # rows dropped by p <= 1e-12 (-1 = solve failed)
        'noiseless_wls_err': noiseless_wls_err,
        'p0_per_state': np.array([s['p_true'][0] for s in true_systems]),
        'norm_g_true': np.array([s['norm_g_true'] for s in true_systems]),
        'heldout_seed_base': SHOTS_HELDOUT_SEED,
    }


def p0_statistics(n_states, seed=0):
    """Ground-state Boltzmann weight p0 at the era BETA over the held-out
    ensemble (referee request: quote p0 at beta=1). Returns median/min/max."""
    beta = float(BETA)
    M_pairs = basis.d // 2
    G_true_batch = _shots_heldout_G_batch(n_states, seed)
    p0 = np.empty(int(n_states), dtype=np.float64)
    p_min = np.empty(int(n_states), dtype=np.float64)
    for i in range(int(n_states)):
        E_true = scipy.linalg.eigh(_shots_h_dense64(G_true_batch[i]),
                                   eigvals_only=True)
        p_true = np.exp(-beta * (E_true - E_true.min())).astype(np.float64)
        p_true /= np.sum(p_true)
        p0[i] = p_true[0]
        p_min[i] = p_true.min()
    return {
        'beta': beta,
        'n_states': int(n_states),
        'seed': int(seed),
        'heldout_seed_base': SHOTS_HELDOUT_SEED,
        'p0_median': float(np.median(p0)),
        'p0_min': float(p0.min()),
        'p0_max': float(p0.max()),
        'p0_mean': float(p0.mean()),
        'p_smallest_median': float(np.median(p_min)),  # smallest eigenstate weight
        'p0_per_state': p0.tolist(),
    }


# ----------------------------------------------------------------- analysis
def _shots_median_curves(d):
    n_pts = len(np.asarray(d['n_shots_sweep']))
    def stat(key, fn, q=None):
        a = np.asarray(d[key], dtype=np.float64).reshape(n_pts, -1)
        return fn(a, axis=1) if q is None else fn(a, q, axis=1)
    return {
        'med_wls': stat('err_wls', np.nanmedian),
        'med_ogn': stat('err_ogn', np.nanmedian),
        'q25_wls': stat('err_wls', np.nanquantile, 0.25),
        'q75_wls': stat('err_wls', np.nanquantile, 0.75),
        'q25_ogn': stat('err_ogn', np.nanquantile, 0.25),
        'q75_ogn': stat('err_ogn', np.nanquantile, 0.75),
    }


def _shots_crossover_inrange(epsilons, m_wls, m_ogn):
    """Log-log interpolated eps where the WLS and OGN medians cross, or None
    if no sign change occurs within the tested sweep. [prior-campaign logic]"""
    m_wls = np.asarray(m_wls, float); m_ogn = np.asarray(m_ogn, float)
    fin = np.isfinite(m_wls) & np.isfinite(m_ogn) & (m_wls > 0) & (m_ogn > 0)
    eps_f = np.asarray(epsilons, float)[fin]
    if eps_f.size < 2:
        return None
    d = np.log(m_wls[fin]) - np.log(m_ogn[fin])
    sgn = np.sign(d)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size == 0:
        return None
    j = int(idx[0])
    x0, x1 = np.log(eps_f[j]), np.log(eps_f[j + 1])
    f = d[j] / (d[j] - d[j + 1])
    eps_c = float(np.exp(x0 + f * (x1 - x0)))
    return {'eps': eps_c, 'N_shots': float(1.0 / eps_c**2)}


def _shots_slope_window(n_censored, n_pts):
    """Largest contiguous run of shot points with zero censored rows
    (max across trials; fallback: median across trials). [v4 logic]"""
    ncc = np.asarray(n_censored).reshape(n_pts, -1)
    max_cens = ncc.max(axis=1)
    ok = max_cens == 0
    criterion = 'max n_censored == 0 across all trials'
    if not np.any(ok):
        med_cens = np.median(ncc, axis=1)
        ok = med_cens == 0
        criterion = 'fallback: median n_censored == 0 (no all-trials-zero window)'
    best = (0, 0)
    s = None
    for i, o in enumerate(list(ok) + [False]):
        if o and s is None:
            s = i
        if not o and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return best[0], best[1], criterion


def analyze_curves(gauss_dict, multi_dict):
    """Median curves + derived Sec V.C numbers (values_prev.json 'shots'
    schema): OGN plateau, WLS log-log slope over the uncensored high-shot
    window, crossover (in-range if the medians cross within the tested sweep,
    otherwise extrapolated from the slope fit and flagged), OGN advantage
    window, and Gaussian-vs-multinomial median discrepancy statistics."""
    sweep = np.asarray(gauss_dict['n_shots_sweep'], dtype=np.float64)
    eps = np.asarray(gauss_dict['epsilons'], dtype=np.float64)
    assert np.allclose(sweep, np.asarray(multi_dict['n_shots_sweep'], float)), \
        "gaussian and multinomial runs must share the same shot sweep"
    n_pts = len(sweep)

    g = _shots_median_curves(gauss_dict)
    m = _shots_median_curves(multi_dict)

    out = {
        'n_shots_sweep': sweep.tolist(),
        'epsilons': eps.tolist(),
        'median_wls_gauss': g['med_wls'].tolist(),
        'median_ogn_gauss': g['med_ogn'].tolist(),
        'median_wls_multi': m['med_wls'].tolist(),
        'median_ogn_multi': m['med_ogn'].tolist(),
        'q25_wls_gauss': g['q25_wls'].tolist(),
        'q75_wls_gauss': g['q75_wls'].tolist(),
        'q25_ogn_gauss': g['q25_ogn'].tolist(),
        'q75_ogn_gauss': g['q75_ogn'].tolist(),
    }

    # ---- WLS slope over the uncensored high-shot window (gaussian run)
    w0, w1, criterion = _shots_slope_window(gauss_dict['n_censored'], n_pts)
    out['slope_window_criterion'] = criterion
    slope_fit = None
    if w1 - w0 >= 3:
        x = np.log10(eps[w0:w1])
        y = np.log10(g['med_wls'][w0:w1])
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() >= 3:
            cfs, cov = np.polyfit(x[fin], y[fin], 1, cov=True)
            slope_fit = cfs
            out['wls_slope'] = [float(cfs[0]), float(np.sqrt(cov[0, 0]))]
            out['slope_window_shots'] = [float(sweep[w0:w1].min()),
                                         float(sweep[w0:w1].max())]
            powerlaw = lambda e: 10**(cfs[1]) * e**cfs[0]
            ratio = g['med_wls'] / powerlaw(eps)
            above = np.where(ratio > 2.0)[0]
            out['onset_eps'] = float(eps[above.min()]) if above.size else None
    if slope_fit is None:
        out['wls_slope'] = None
        out['slope_window_shots'] = None
        out['onset_eps'] = None

    # ---- OGN plateau (high-shot asymptote of the gaussian OGN median)
    top = sweep >= 1e9
    if not np.any(top):  # reduced/smoke sweeps
        top = sweep >= np.quantile(sweep, 0.8)
    ogn_plateau = float(np.nanmedian(g['med_ogn'][top]))
    out['ogn_plateau'] = ogn_plateau

    # ---- crossover (gaussian): in-range if the medians cross within the
    # tested sweep, else extrapolated from the slope fit (flagged)
    def _crossover_block(curves, prefix):
        cr = _shots_crossover_inrange(eps, curves['med_wls'], curves['med_ogn'])
        if cr is not None:
            return {f'{prefix}Nshots': cr['N_shots'], f'{prefix}eps': cr['eps'],
                    'extrapolated': False}
        # extrapolate the fitted WLS power law down to the OGN plateau
        if slope_fit is None or not np.isfinite(ogn_plateau) or ogn_plateau <= 0:
            return {f'{prefix}Nshots': None, f'{prefix}eps': None,
                    'extrapolated': None}
        x = np.log10(eps[w0:w1])
        y = np.log10(curves['med_wls'][w0:w1])
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() < 3:
            return {f'{prefix}Nshots': None, f'{prefix}eps': None,
                    'extrapolated': None}
        cfs = np.polyfit(x[fin], y[fin], 1)
        plateau = float(np.nanmedian(curves['med_ogn'][top]))
        eps_x = float(10.0 ** ((np.log10(plateau) - cfs[1]) / cfs[0]))
        return {f'{prefix}Nshots': float(1.0 / eps_x**2), f'{prefix}eps': eps_x,
                'extrapolated': True}

    cg = _crossover_block(g, 'crossover_')
    out['crossover_Nshots'] = cg['crossover_Nshots']
    out['crossover_eps'] = cg['crossover_eps']
    out['crossover_extrapolated'] = cg['extrapolated']

    cm = _crossover_block(m, 'crossover_')
    out['crossover_Nshots_multinomial'] = cm['crossover_Nshots']
    out['crossover_eps_multinomial'] = cm['crossover_eps']
    out['crossover_multinomial_extrapolated'] = cm['extrapolated']

    # ---- OGN advantage window (shots range where OGN median < WLS median)
    adv = np.isfinite(g['med_ogn']) & np.isfinite(g['med_wls']) \
        & (g['med_ogn'] < g['med_wls'])
    out['advantage_window_shots'] = ([float(sweep[adv].min()),
                                      float(sweep[adv].max())]
                                     if np.any(adv) else None)

    # ---- Gaussian-vs-multinomial median discrepancy per estimator
    hi = sweep >= 1e8
    if not np.any(hi):
        hi = sweep >= np.quantile(sweep, 0.8)
    out['multi_departure_criterion'] = (
        f'|median_multinomial / median_gaussian - 1| > {SHOTS_DEPARTURE_TOL}; '
        'departure_Nshots = largest such N_shots')
    for est in ('wls', 'ogn'):
        ratio = m[f'med_{est}'] / g[f'med_{est}']
        fin = np.isfinite(ratio)
        if not np.any(fin):
            out[f'multi_departure_Nshots_{est}'] = None
            out[f'multi_max_ratio_{est}'] = None
            out[f'multi_max_ratio_Nshots_{est}'] = None
            out[f'multi_highshot_max_dev_{est}'] = None
            continue
        dev = np.abs(ratio - 1.0)
        depart = fin & (dev > SHOTS_DEPARTURE_TOL)
        out[f'multi_departure_Nshots_{est}'] = (float(sweep[depart].max())
                                                if np.any(depart) else None)
        k = int(np.nanargmax(np.where(fin, ratio, -np.inf)))
        out[f'multi_max_ratio_{est}'] = float(ratio[k])
        out[f'multi_max_ratio_Nshots_{est}'] = float(sweep[k])
        out[f'multi_highshot_max_dev_{est}'] = (float(np.nanmax(dev[hi & fin]))
                                                if np.any(hi & fin) else None)

    out['n_censored_median_per_shot'] = np.median(
        np.asarray(gauss_dict['n_censored']).reshape(n_pts, -1), axis=1).tolist()

    # ---- gates (acceptance criteria from the prior campaign)
    e0 = np.asarray(gauss_dict['noiseless_wls_err'], dtype=np.float64)
    e0f = e0[np.isfinite(e0)]
    n_states = int(gauss_dict['n_states'])
    gates = {
        'p3_noiseless': {
            'n_below_1e-8': int(np.sum(e0f < 1e-8)),
            'median': float(np.median(e0f)) if e0f.size else None,
            'max': float(e0f.max()) if e0f.size else None,
            'pass': bool(e0f.size and np.sum(e0f < 1e-8) >= 0.99 * n_states),
        },
        'same_draw': {
            'enforced': True,
            'note': ('single p_meas per (state, draw, shots) feeds both '
                     'estimators by construction (one rng draw per tuple)'),
        },
        'plateau': {
            'value': ogn_plateau,
            'pass': bool(np.isfinite(ogn_plateau) and 5e-4 <= ogn_plateau <= 2e-3),
        },
        'multinomial_agree': {
            'max_rel_dev_wls': out['multi_highshot_max_dev_wls'],
            'max_rel_dev_ogn': out['multi_highshot_max_dev_ogn'],
            'pass': bool(out['multi_highshot_max_dev_wls'] is not None
                         and out['multi_highshot_max_dev_ogn'] is not None
                         and out['multi_highshot_max_dev_wls'] < 0.2
                         and out['multi_highshot_max_dev_ogn'] < 0.2),
        },
    }
    out['gates'] = gates
    return out


# ------------------------------------------------------------------- figure
def plot_shots_figure(gauss_dict, multi_dict, analysis, out_prefix):
    """Regenerated Sec V.C figure (style of log_inversion_wls.png): log-log
    median parameter relative error vs eps_shot (bottom) / N_shots (top),
    OGN vs WLS, Gaussian solid + multinomial dashed overlays, IQR bands,
    crossover annotated. Saves <out_prefix>.png (300 dpi) and <out_prefix>.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.asarray(analysis['epsilons'], dtype=np.float64)
    med_gw = np.asarray(analysis['median_wls_gauss'], dtype=np.float64)
    med_go = np.asarray(analysis['median_ogn_gauss'], dtype=np.float64)
    med_mw = np.asarray(analysis['median_wls_multi'], dtype=np.float64)
    med_mo = np.asarray(analysis['median_ogn_multi'], dtype=np.float64)
    q25_w = np.asarray(analysis['q25_wls_gauss'], dtype=np.float64)
    q75_w = np.asarray(analysis['q75_wls_gauss'], dtype=np.float64)
    q25_o = np.asarray(analysis['q25_ogn_gauss'], dtype=np.float64)
    q75_o = np.asarray(analysis['q75_ogn_gauss'], dtype=np.float64)

    # [REGEN-PATCH STATS] display clip only; all statistics are unclipped
    _clip = lambda a: np.clip(a, 1e-8, 10.0)

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(11, 7))

    # [from export l.4674-4682] IQR bands + thick median lines (gaussian)
    ax.fill_between(eps, _clip(q25_w), _clip(q75_w),
                    color='forestgreen', alpha=0.18, lw=0, zorder=2)
    ax.fill_between(eps, _clip(q25_o), _clip(q75_o),
                    color='navy', alpha=0.18, lw=0, zorder=3)
    ax.plot(eps, _clip(med_gw), color='forestgreen', lw=4,
            label='Log Inversion (WLS)', zorder=4)
    ax.plot(eps, _clip(med_go), color='navy', lw=4,
            label='OGN Prediction', zorder=5)

    # multinomial overlays (dashed + markers)
    ax.plot(eps, _clip(med_mw), color='forestgreen', lw=2.5, ls='--',
            marker='s', markersize=4, markevery=max(1, len(eps) // 16),
            label='Log Inversion (multinomial)', zorder=4)
    ax.plot(eps, _clip(med_mo), color='navy', lw=2.5, ls='--',
            marker='o', markersize=4, markevery=max(1, len(eps) // 16),
            label='OGN (multinomial)', zorder=5)

    # theoretical 1/sqrt(N) guide
    ax.plot(eps, eps, 'k--', lw=2, alpha=0.5,
            label=r'$\propto 1/\sqrt{N_{shots}}$', zorder=0)

    # Top Axis for Number of Shots  [from export l.4685-4686]
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1 / (x**2),
                                                  lambda x: 1 / np.sqrt(x)))
    ax_top.set_xlabel('Number of Experimental Measurement Shots ($N_{shots}$)',
                      fontweight='bold', fontsize=15, labelpad=12)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(eps.min() * 0.9, eps.max() * 1.1)
    lo = max(1e-8, 0.5 * float(np.nanmin(_clip(med_gw))))
    hi_data = float(np.nanmax(_clip(np.concatenate([med_gw, med_go]))))
    ax.set_ylim(lo, max(0.6, min(10.0, 1.5 * hi_data)))

    ax.set_xlabel(r'Statistical Measurement Noise '
                  r'($\epsilon_{shot} = 1/\sqrt{N_{shots}}$)',
                  fontweight='bold', fontsize=16)
    ax.set_ylabel('Parameter Relative Error\n'
                  r'($||\Delta G||_F / ||G_{true}||_F$)',
                  fontweight='bold', fontsize=16)

    # crossover annotation
    cr_N = analysis.get('crossover_Nshots')
    cr_eps = analysis.get('crossover_eps')
    cr_ext = analysis.get('crossover_extrapolated')
    if cr_N is not None and cr_eps is not None:
        if cr_ext is False and eps.min() <= cr_eps <= eps.max():
            ax.axvline(cr_eps, color='crimson', ls=':', lw=2, zorder=1)
            ax.annotate(f'crossover\n$N_{{shots}} \\approx {cr_N:.2g}$',
                        xy=(cr_eps, lo * 3), xycoords='data',
                        color='crimson', fontsize=12, fontweight='bold',
                        ha='left', va='bottom')
        else:
            ax.text(0.02, 0.97,
                    f'WLS/OGN crossover (extrapolated):\n'
                    f'$N_{{shots}} \\approx {cr_N:.2g}$ '
                    f'($\\epsilon \\approx {cr_eps:.2g}$)',
                    transform=ax.transAxes, fontsize=12, va='top', ha='left',
                    color='crimson',
                    bbox=dict(boxstyle='round', fc='white', ec='crimson',
                              alpha=0.9))

    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle='--', alpha=0.15)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=12)

    plt.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches='tight')
    plt.close(fig)
    return str(out_prefix) + ".png"
