# =============================================================================
# p6_bcs_rg.py — engine part 6: BCS mean-field inversions (uniform + vectorial
# d=12 panels) and the Richardson-Gaudin exact canonical inversion.
#
# Exec'd into the shared engine namespace AFTER p1_core..p5_shots; relies on
# late-bound era globals set by p1_core's init_d12():
#   basis (924-dim FixedBasis), rho_1_arrays, rho_2_kkbar_arrays, levels,
#   U_ENERGY_SEED, BETA, STATE_TYPE, N_ELEC,
#   two_body_hamiltonian_sp, _prepare_hamiltonian_for_eigsh, thermal_state
#
# Sources (provenance markers inline):
#   - .claude/cells/cell_33.py  uniform-panel BCS kernels + scalar inversion
#     (gap_eq_residual_t0/finite_t, get_bcs_rho_t0/finite_t,
#      get_bcs_energy_exact, solve_gap_scipy, invert_bcs_direct,
#      get_order_parameter) and the inversion-loop flow
#   - .claude/cells/cell_35.py  vectorial BCS variational driver
#     (solve_gap_eq_stateless, calc_bcs_observables, bcs_variational_driver)
#   - .claude/cells/cell_36.py  reconstruct_vec + accuracy panel
#   - .claude/cells/cell_37.py  parallel_robust_task_v2, vector-form
#     get_order_parameter, physical_observables_dashboard flow
#   - FermionicML-satelite/regen_figures.py  regen_t3 (l.944-1136): RG exact
#     canonical inversion on the uniform model; its validated output stats
#     live in campaign/reference/values_prev.json 'rg' block
#     (median_abs_err = 3.9e-10 over 1024 targets, n_not_converged = 0)
#
# Parallelism: the cells' ray.remote workers and regen's ray chunks are
# replaced by concurrent.futures.ProcessPoolExecutor (fork start method) with
# a plain serial-loop fallback, per the engine contract. On hosts where TPU
# chips are visible (/dev/accel*) the default is SERIAL, because forking a
# process after the TPU runtime initialized can deadlock (the prior campaign
# used freshly *spawned* ray workers for exactly this reason). Override with
# the n_workers argument or P6_FORCE_POOL=1.
# =============================================================================

import os
import math
import time
import multiprocessing
import concurrent.futures

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from tqdm.auto import tqdm

try:
    from numba import njit  # [from cell_33] the kernels are @njit'd; m=6
except ImportError:          # arrays are tiny, so a transparent no-op
    def njit(*args, **kwargs):  # fallback is safe for environments w/o numba
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

# ----------------------------------------------------------------- constants
# value-from-code for manuscript l.120: gamma weighting the energy term of the
# composite BCS objective (regularized branch of invert_bcs_direct; literal
# 1e-2 at cell_33 l.190: err += 1e-2 * |E_target - E_BCS|).
BCS_GAMMA = 1e-2
# [from cell_37 l.8-9] vectorial-driver hyperparameters
L_ENERGY = 1.0
L_SMOOTH = 0.05
EPSILON = 1e-12  # [from cell_35 l.11]
# [from regen_figures.py l.1132-1133] RG gate thresholds
RG_GATE_MEDIAN = 1e-6
RG_GATE_MAX = 1e-4


# =============================================================================
# 1. Uniform-panel physics kernels  [from cell_33 l.19-137]
# =============================================================================

@njit()
def gap_eq_residual_t0(delta, g_val, energies, e_mean):
    """Zero-Temperature Gap Equation Residual: Sum(1/2E) - 1/G"""
    xi = energies - e_mean
    E_k = np.sqrt(xi**2 + delta**2)
    E_safe = np.maximum(E_k, 1e-12)
    sum_term = np.sum(1.0 / (2.0 * E_safe))

    inv_g = 1.0 / g_val if g_val > 1e-9 else 1e9
    return sum_term - inv_g

@njit()
def gap_eq_residual_finite_t(delta, g_val, energies, e_mean, beta):
    """Finite-Temperature Gap Equation Residual"""
    xi = energies - e_mean
    E_k = np.sqrt(xi**2 + delta**2)
    E_safe = np.maximum(E_k, 1e-12)

    arg = 0.5 * beta * E_k
    th = np.empty_like(arg)
    for i in range(len(arg)):
        if arg[i] > 20.0: th[i] = 1.0
        else: th[i] = np.tanh(arg[i])

    sum_term = np.sum(th / (2.0 * E_safe))
    inv_g = 1.0 / g_val if g_val > 1e-9 else 1e9
    return sum_term - inv_g

@njit()
def get_bcs_rho_t0(delta, energies, e_mean):
    """Constructs T=0 BCS RDM"""
    m = len(energies)
    xi = energies - e_mean
    E_k = np.sqrt(xi**2 + delta**2)
    E_safe = np.maximum(E_k, 1e-12)

    # u*v = Delta / 2E
    ukvk = delta / (2.0 * E_safe)
    # n_k = 1/2 * (1 - xi/E)
    nk = 0.5 * (1.0 - xi / E_safe)

    rho = np.outer(ukvk, ukvk)
    for k in range(m):
        rho[k, k] = nk[k]
    return rho

@njit()
def get_bcs_rho_finite_t(delta, energies, e_mean, beta):
    """Constructs Finite-T BCS RDM"""
    m = len(energies)
    xi = energies - e_mean
    E_k = np.sqrt(xi**2 + delta**2)
    E_safe = np.maximum(E_k, 1e-12)

    arg = 0.5 * beta * E_k
    th = np.empty_like(arg)
    for i in range(len(arg)):
         if arg[i] > 20.0: th[i] = 1.0
         else: th[i] = np.tanh(arg[i])

    ukvk = (delta / (2.0 * E_safe)) * th
    nk = 0.5 * (1.0 - (xi / E_safe) * th)

    rho = np.outer(ukvk, ukvk)
    for k in range(m):
        rho[k, k] = nk[k]
    return rho

@njit()
def get_bcs_energy_exact(delta, g_val, energies, beta):
    """
    Calculates the expectation value <H>_BCS including finite-size corrections.
    H = Sum(2*eps*n) - G * P^dag P
    """
    # 1. Setup
    if g_val < 1e-9: g_val = 1e-9

    xi = energies - np.mean(energies) # Ensure centered levels
    E_k = np.sqrt(xi**2 + delta**2)
    E_safe = np.maximum(E_k, 1e-12)

    # 2. Thermal Factors
    if beta > 1000:
        th = np.ones_like(E_k)
    else:
        arg = 0.5 * beta * E_k
        th = np.empty_like(arg)
        for i in range(len(arg)):
            if arg[i] > 20.0: th[i] = 1.0
            else: th[i] = np.tanh(arg[i])

    # 3. Densities
    # v_k^2 = <n_k>
    v2 = 0.5 * (1.0 - (xi / E_safe) * th)
    # u_k v_k = <P_k>
    uv = (delta / (2.0 * E_safe)) * th

    # 4. Energy Calculation
    # Kinetic: Sum 2 * eps * v^2
    e_kin = np.sum(2.0 * energies * v2)

    # Interaction: -G * <P^dag P>
    # <P^dag P> = Sum_{i,j} <c_i^dag c_ibar^dag c_jbar c_j>
    #           = (Sum uv)^2 - Sum(uv)^2 + Sum(v^2)
    # The term Sum(v^2) is the diagonal (Hartree-Fock) term.
    # In the Normal State (uv=0), this term gives E = -G * N_pairs,
    # which is crucial for matching the Exact Diagonalization energy.
    sum_uv = np.sum(uv)
    sum_uv2 = np.sum(uv**2)
    sum_v2 = np.sum(v2)

    pair_expectation = (sum_uv**2) - sum_uv2 + sum_v2
    e_int = -g_val * pair_expectation

    return e_kin + e_int


# -----------------------------------------------------------------------------
# 2. Inversion Solvers  [from cell_33 l.139-218]
# -----------------------------------------------------------------------------

def solve_gap_scipy(G, energies, e_mean, beta, is_gs):
    """
    Gap Equation Solver
    """
    if G <= 1e-6: return 0.0

    if is_gs:
        resid_fn = lambda d: gap_eq_residual_t0(d, G, energies, e_mean)
    else:
        resid_fn = lambda d: gap_eq_residual_finite_t(d, G, energies, e_mean, beta)

    # If Residual(0) < 0, it means 1/G > Sum(1/2E), i.e., G is too small.
    if resid_fn(0.0) < 0:
        return 0.0

    try:
        sol = scipy.optimize.root_scalar(
            resid_fn, bracket=[1e-9, 20.0], method='brentq', xtol=1e-7
        )
        return sol.root
    except ValueError:
        return 0.0

def invert_bcs_direct(rho_target, energy_target, energies, e_mean, beta, is_gs, alt_loss = False):
    """
    Optimizes G to minimize || rho_target - rho_BCS(G) ||.
    alt_loss=False: energy-regularized composite objective (gamma = BCS_GAMMA > 0).
    alt_loss=True : unregularized variant (gamma = 0; gap-eq consistency term only).
    """
    if rho_target.ndim == 3: rho_target = rho_target.squeeze()

    def loss_fn(g_val):
        # Solve Delta(G)
        delta = solve_gap_scipy(g_val, energies, e_mean, beta, is_gs)

        # Construct State
        if is_gs:
            rho_pred = get_bcs_rho_t0(delta, energies, e_mean)
            # If Delta > 0, this is ~0. If Delta = 0, this is |Sum(1/2|xi|) - 1/G|
            resid = gap_eq_residual_t0(delta, g_val, energies, e_mean)
        else:
            rho_pred = get_bcs_rho_finite_t(delta, energies, e_mean, beta)
            resid = gap_eq_residual_finite_t(delta, g_val, energies, e_mean, beta)

        # Loss Calculation
        err = np.linalg.norm(rho_pred - rho_target)

        # Energy loss
        if not alt_loss:
            # [engine edit] literal 1e-2 (cell_33 l.190) lifted into BCS_GAMMA
            # so the manuscript can cite the value-from-code (main.tex l.120).
            err += BCS_GAMMA * np.abs(energy_target - get_bcs_energy_exact(delta, g_val, energies, beta))
            return err

        else:
            # Physical Regularization: Gap Equation Consistency
            err += 1e-3 * (resid**2)
            return err

        # Regularization for low G noise
        if delta == 0:
            return err # + 1e-5 * g_val

        return err

    # Bounded optimization
    res = scipy.optimize.minimize_scalar(
        loss_fn, bounds=(0.0, 5.0), method='bounded', options={'xatol': 1e-5}
    )

    G_opt = res.x
    Delta_opt = solve_gap_scipy(G_opt, energies, e_mean, beta, is_gs)

    return G_opt, Delta_opt

def get_order_parameter(rho_matrix, g):
        # [from cell_33 l.214-218] scalar-G form. NOTE: redefined below with the
        # cell_37 vector form, mirroring notebook execution order; the two are
        # numerically identical for scalar g and 2D rho (norm == abs).
        rho_abs = np.abs(rho_matrix).copy().squeeze()
        np.fill_diagonal(rho_abs, 0.0)
        coherence_sum = np.sum(rho_abs)
        return np.abs(g) * np.sqrt(coherence_sum)


# =============================================================================
# 3. d=12 exact canonical forward map (Richardson-Gaudin context)
#    [from FermionicML-satelite/regen_figures.py regen_t3, l.944-1018]
# =============================================================================

_T3_CTX = {}
_D12_CTX_CACHE = {}


def _t3_rho_can(G_val, ctx):
    # [from regen_figures.py l.947-957] exact canonical (ground-state) rho2
    import scipy.sparse.linalg as _spla
    H = ctx['H0'] - G_val * ctx['V_int']
    _, v = _spla.eigsh(H, k=1, which='SA', tol=1e-9)
    psi = v[:, 0]
    m = ctx['m_pairs']
    rho = np.empty((m, m))
    for a in range(m):
        for b in range(m):
            rho[a, b] = psi @ ctx['R'][a][b].dot(psi)
    return 0.5 * (rho + rho.T)


def _t3_fit_one(payload):
    # [from regen_figures.py l.960-968] single-target RG fit (serial form)
    import scipy.optimize as _sopt
    idx, rho_t, g_hi = payload
    ctx = _T3_CTX
    fun = lambda g: float(np.sum((_t3_rho_can(g, ctx) - rho_t)**2))
    r = _sopt.minimize_scalar(fun, bounds=(0.0, 1.2 * g_hi),
                              method='bounded',
                              options={'xatol': 1e-9})
    return idx, float(r.x), float(r.fun), bool(getattr(r, 'success', True))


def _t3_fit_chunk(payload):
    """Chunked RG fits for the process pool.
    [from regen_figures.py _t3_ray_chunk l.1046-1069; ray.remote replaced by
    ProcessPool per engine contract. The context arrives through the
    fork-inherited module global _T3_CTX instead of ray.put handles.]"""
    chunk, g_hi_ = payload
    out = []
    for idx, rho_t in chunk:
        fun = lambda g: float(np.sum((_t3_rho_can(g, _T3_CTX) - rho_t) ** 2))
        res = scipy.optimize.minimize_scalar(fun, bounds=(0.0, 1.2 * g_hi_),
                                             method='bounded',
                                             options={'xatol': 1e-9})
        out.append((idx, float(res.x), float(res.fun),
                    bool(getattr(res, 'success', True))))
    return out


def _rg_build_ctx():
    """Build the d=12 exact canonical forward-map context from the CURRENT era
    globals (init_d12): H(g) = H0 - g*V_int for the uniform ('const') model,
    plus the pair-block operators R[a][b] realizing <P_a^dag P_b>.

    [from regen_figures.py regen_t3 l.977-1018; the basis and energies now come
    from the init_d12 era globals (basis, U_ENERGY_SEED, rho_1_arrays,
    rho_2_kkbar_arrays) instead of a locally rebuilt jittered/rescaled basis.]
    Cached per (basis size, energy seed)."""
    assert basis.d == 12 and basis.size == 924, (basis.d, basis.size)
    e_row = np.asarray(U_ENERGY_SEED[0], dtype=np.float64)
    key = (int(basis.size), e_row.tobytes())
    if key in _D12_CTX_CACHE:
        return _D12_CTX_CACHE[key]

    m_pairs = basis.d // 2
    e_arr = e_row[None, :]
    ENERG = e_row[::2]                 # ENERG_BCS = U_ENERGY_SEED[0][::2]

    # H(g) = H0 - g * V ; precompute the two sparse pieces once
    G0 = np.zeros((1, m_pairs, m_pairs))
    H0_sp = _prepare_hamiltonian_for_eigsh(
        two_body_hamiltonian_sp(basis, e_arr, G0, rho_1_arrays, rho_2_kkbar_arrays,
                                'const')[0])
    G1 = np.ones((1, m_pairs, m_pairs))
    H1_sp = _prepare_hamiltonian_for_eigsh(
        two_body_hamiltonian_sp(basis, e_arr, G1, rho_1_arrays, rho_2_kkbar_arrays,
                                'const')[0])
    import scipy.sparse as _sp
    H0_csr = (H0_sp if _sp.issparse(H0_sp) else _sp.csr_matrix(H0_sp)).tocsr()
    H1_csr = (H1_sp if _sp.issparse(H1_sp) else _sp.csr_matrix(H1_sp)).tocsr()
    V_int = (H0_csr - H1_csr)          # so H(g) = H0 - g*V_int

    # pair-block operators R[a][b] (csr) for <P_a^dag P_b>
    co = rho_2_kkbar_arrays.coords
    da = rho_2_kkbar_arrays.data
    D_N = basis.size
    R = [[None] * m_pairs for _ in range(m_pairs)]
    for a in range(m_pairs):
        for b in range(m_pairs):
            sel = (co[0] == a) & (co[1] == b)
            R[a][b] = _sp.coo_array(
                (da[sel], (co[2][sel], co[3][sel])),
                shape=(D_N, D_N)).tocsr()
    # orientation check: compute_rho_m uses rho_tensor[J, I]; match the
    # dataset pipeline by evaluating <P^dag P> with the same index order.
    ctx = {'H0': H0_csr, 'V_int': V_int, 'R': R, 'm_pairs': m_pairs,
           'ENERG': ENERG, 'e_mean': float(np.mean(ENERG))}
    _D12_CTX_CACHE[key] = ctx
    return ctx


def _d12_exact_target(H, ctx, beta, is_gs):
    """Exact ED (rho2, energy) for one many-body Hamiltonian H (csr).
    Ground-state branch follows _t3_rho_can (eigsh k=1, psi-quadratic-form);
    thermal branch uses p1_core's thermal_state on the dense H and the trace
    orientation of compute_rho_m: rho2[a,b] = Tr(rho_state @ R[a][b])."""
    m = ctx['m_pairs']
    rho = np.empty((m, m))
    if is_gs:
        e, v = scipy.sparse.linalg.eigsh(H, k=1, which='SA', tol=1e-9)
        psi = v[:, 0]
        for a in range(m):
            for b in range(m):
                rho[a, b] = psi @ ctx['R'][a][b].dot(psi)
        return 0.5 * (rho + rho.T), float(e[0])
    energies_t, mats = thermal_state(float(beta), H.toarray()[None, ...])
    rho_state = np.asarray(mats[0].real, dtype=np.float64)
    for a in range(m):
        for b in range(m):
            # Tr(rho @ R) = sum_ij R[i,j] * rho[j,i]
            rho[a, b] = ctx['R'][a][b].multiply(rho_state.T).sum()
    return 0.5 * (rho + rho.T), float(energies_t[0])


# =============================================================================
# 4. Process-pool infrastructure (ray replacement, engine contract)
# =============================================================================

def _register_engine_ns_for_pickle():
    """ProcessPoolExecutor pickles submitted callables by module + qualname.
    The engine parts are exec'd into a plain dict with __name__='engine_ns',
    which is not an importable module, so engine-namespace functions are not
    picklable by default. Registering (the references of) that namespace as a
    real module in sys.modules, combined with the 'fork' start method (children
    inherit sys.modules and the namespace memory), makes them picklable."""
    import sys, types
    name = globals().get('__name__', 'engine_ns')
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    mod.__dict__.update(globals())
    return mod


def _p6_default_workers():
    """min(32, cpu_count) on CPU hosts; 1 (serial) when TPU chips are visible,
    because fork-after-TPU-init can deadlock. Override via the n_workers
    argument of the drivers or env P6_FORCE_POOL=1."""
    if os.environ.get("P6_FORCE_POOL", "0") == "1":
        return min(32, os.cpu_count() or 1)
    import glob as _glob
    if _glob.glob("/dev/accel*"):
        return 1
    return min(32, os.cpu_count() or 1)


def _pool_map(fn, payloads, n_workers, desc):
    """Order-preserving map over payloads: ProcessPoolExecutor
    (max_workers=min(32, os.cpu_count()), fork context) with a plain serial
    loop fallback. Any pool-level failure falls back to serial."""
    if n_workers is None:
        n_workers = _p6_default_workers()
    n_workers = max(1, int(n_workers))
    results = [None] * len(payloads)
    if n_workers > 1 and len(payloads) > 1:
        try:
            _register_engine_ns_for_pickle()
            mp_ctx = multiprocessing.get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(n_workers, len(payloads)),
                    mp_context=mp_ctx) as ex:
                futs = {ex.submit(fn, pl): i for i, pl in enumerate(payloads)}
                for fut in tqdm(concurrent.futures.as_completed(futs),
                                total=len(futs), desc=desc):
                    results[futs[fut]] = fut.result()
            return results
        except Exception as e:
            print(f"[p6] ProcessPool path failed ({e!r}); "
                  f"falling back to serial loop")
    for i, pl in enumerate(tqdm(payloads, desc=f"{desc} (serial)")):
        results[i] = fn(pl)
    return results


# =============================================================================
# 5. SPEC 1 — uniform-panel BCS inversions against exact ED targets
# =============================================================================

def bcs_uniform_inversions(g_targets, beta=None, is_gs=True):
    """Uniform-model BCS inversion panel (cell_33 flow, exact-ED targets).

    For each target coupling G: exact ED target rho2 + energy from the current
    d12 era globals (H(g) = H0 - g*V_int, 924-dim basis), then BOTH BCS
    inversions — energy-regularized (gamma = BCS_GAMMA) and unregularized
    (alt_loss) — plus order parameters and condensation eigenvalues.

    g_targets: array of couplings, or int n -> np.linspace(0.1, 1.0, n).
    beta:      inverse temperature for the finite-T branch (era BETA if None).
    is_gs:     ground-state branch (TEMP_BETA forced to inf, as in cell_33).
    Returns a dict of host-side float64 arrays + a 'summary' block.
    """
    t0 = time.time()
    # [from cell_33 l.8-17] cell-level config, evaluated here at call time
    ENERG_BCS = np.asarray(U_ENERGY_SEED[0], dtype=np.float64)[::2]
    E_MEAN_BCS = np.mean(ENERG_BCS)
    TEMP_BETA = float(BETA) if beta is None else float(beta)
    IS_GS = bool(is_gs)
    if IS_GS:
        TEMP_BETA = np.inf

    if np.isscalar(g_targets):
        g_true_sorted = np.linspace(0.1, 1.0, int(g_targets))
    else:
        g_true_sorted = np.sort(np.asarray(g_targets, dtype=np.float64).ravel())
    n = len(g_true_sorted)

    # exact ED targets (replaces cell_33's predict_and_load dataset targets:
    # the BCS control is model-independent; CNN overlays go in at plot time)
    ctx = _rg_build_ctx()
    m = ctx['m_pairs']
    rho_sorted = np.empty((n, m, m))
    energies_sorted = np.empty(n)
    for i, g in enumerate(tqdm(g_true_sorted, desc='uniform ED targets')):
        H = ctx['H0'] - float(g) * ctx['V_int']
        rho_sorted[i], energies_sorted[i] = _d12_exact_target(
            H, ctx, TEMP_BETA, IS_GS)

    # Inversion Loop  [from cell_33 l.259-304]
    g_bcs_list = []
    delta_bcs_list = []
    delta_theo_list = []
    lambda_bcs_list = []
    lambda_true_list = []
    psi_bcs_list = []
    psi_bcs_list_alt = []
    psi_exact_list = []
    g_bcs_list_alt = []
    delta_bcs_list_alt = []

    print(f"Running Inversion on {len(g_true_sorted)} samples...")

    for i in tqdm(range(len(g_true_sorted))):
        # Invert G
        g_inv, d_inv = invert_bcs_direct(rho_sorted[i], energies_sorted[i], ENERG_BCS, E_MEAN_BCS, TEMP_BETA, IS_GS)
        g_inv_a, d_inv_a = invert_bcs_direct(rho_sorted[i], energies_sorted[i], ENERG_BCS, E_MEAN_BCS, TEMP_BETA, IS_GS, alt_loss = True)

        # Exact Order Parameter
        # Psi = Sum_{k < k'} |<P_k P_k'>|
        psi_exact_list.append(get_order_parameter(rho_sorted[i], g_true_sorted[i]))

        # Eigenvalues for Condensation Analysis
        if IS_GS:
            rho_bcs = get_bcs_rho_t0(d_inv, ENERG_BCS, E_MEAN_BCS)
            rho_bcs_a = get_bcs_rho_t0(d_inv_a, ENERG_BCS, E_MEAN_BCS)
        else:
            rho_bcs = get_bcs_rho_finite_t(d_inv, ENERG_BCS, E_MEAN_BCS, TEMP_BETA)
            # [engine edit] cell_33 built rho_bcs_a only in the IS_GS branch
            # (latent NameError at finite T); completed symmetrically here.
            rho_bcs_a = get_bcs_rho_finite_t(d_inv_a, ENERG_BCS, E_MEAN_BCS, TEMP_BETA)

        # BCS Order Parameter
        psi_bcs_list.append(get_order_parameter(rho_bcs, g_inv))
        psi_bcs_list_alt.append(get_order_parameter(rho_bcs_a, g_inv_a))

        lam_bcs = np.linalg.eigvalsh(rho_bcs).max()
        lam_true = np.linalg.eigvalsh(rho_sorted[i].squeeze()).max()

        d_theo = solve_gap_scipy(g_true_sorted[i], ENERG_BCS, E_MEAN_BCS, TEMP_BETA, IS_GS)

        g_bcs_list.append(g_inv)
        g_bcs_list_alt.append(g_inv_a)
        delta_bcs_list.append(d_inv)
        delta_theo_list.append(d_theo)
        delta_bcs_list_alt.append(d_inv_a)
        lambda_bcs_list.append(lam_bcs)
        lambda_true_list.append(lam_true)

    g_bcs = np.asarray(g_bcs_list, dtype=np.float64)
    g_bcs_alt = np.asarray(g_bcs_list_alt, dtype=np.float64)
    abs_err_reg = np.abs(g_bcs - g_true_sorted)
    abs_err_unreg = np.abs(g_bcs_alt - g_true_sorted)

    return {
        'g_true': g_true_sorted,
        'g_bcs': g_bcs,
        'g_bcs_alt': g_bcs_alt,
        'delta_bcs': np.asarray(delta_bcs_list, dtype=np.float64),
        'delta_bcs_alt': np.asarray(delta_bcs_list_alt, dtype=np.float64),
        'delta_theo': np.asarray(delta_theo_list, dtype=np.float64),
        'psi_exact': np.asarray(psi_exact_list, dtype=np.float64),
        'psi_bcs': np.asarray(psi_bcs_list, dtype=np.float64),
        'psi_bcs_alt': np.asarray(psi_bcs_list_alt, dtype=np.float64),
        'lambda_bcs': np.asarray(lambda_bcs_list, dtype=np.float64),
        'lambda_true': np.asarray(lambda_true_list, dtype=np.float64),
        'energy_targets': energies_sorted,
        'rho_targets': rho_sorted,
        'beta': TEMP_BETA, 'is_gs': IS_GS,
        'summary': {
            'n_targets': int(n),
            'bcs_gamma': float(BCS_GAMMA),
            'median_abs_err_reg': float(np.median(abs_err_reg)),
            'max_abs_err_reg': float(abs_err_reg.max()),
            'median_abs_err_unreg': float(np.median(abs_err_unreg)),
            'max_abs_err_unreg': float(abs_err_unreg.max()),
            'wall_s': time.time() - t0,
        },
    }


# =============================================================================
# 6. SPEC 2 — Richardson-Gaudin exact canonical inversion
#    [from regen_figures.py regen_t3 l.971-1136]
# =============================================================================

def rg_canonical_inversion(g_targets, seed=0, g_lo=0.1, g_hi=1.0,
                           n_workers=None, with_bcs_baseline=True):
    """Richardson-Gaudin canonical inversion on the uniform (const) model:
    one-parameter exact-canonical fit via ED (d=12, N=6, fully connected
    basis without pairing restriction, per manuscript appendix).

    g_targets: int n -> n targets drawn ~U(g_lo, g_hi), sorted, with the
               prior-campaign stream SeedSequence([seed, 4, 0]) (seed=0 +
               n=1024 reproduces the values_prev.json 'rg' protocol);
               or an explicit array of couplings (used as given).
    Returns a dict with per-target g_rg/abs_err/losses/converged arrays, the
    unregularized BCS baseline (g_bcs_alt) and the values_prev-style summary
    (median_abs_err, max_abs_err, n_not_converged, gate_pass).
    """
    t0 = time.time()
    ctx = _rg_build_ctx()
    m_pairs = ctx['m_pairs']
    _T3_CTX.clear()
    _T3_CTX.update({'H0': ctx['H0'], 'V_int': ctx['V_int'], 'R': ctx['R'],
                    'm_pairs': m_pairs})

    # targets from the exact canonical (ground-state) pipeline
    # [from regen_t3 l.1020-1026; GLOBAL_SEED -> seed argument]
    if np.isscalar(g_targets):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4, 0]))
        g_true = np.sort(rng.uniform(g_lo, g_hi, int(g_targets)))
    else:
        g_true = np.asarray(g_targets, dtype=np.float64).ravel()
        g_hi = max(float(g_hi), float(g_true.max()))
    n_targets = len(g_true)

    targets = np.empty((n_targets, m_pairs, m_pairs))
    for i, g in enumerate(tqdm(g_true, desc='rg targets')):
        targets[i] = _t3_rho_can(float(g), _T3_CTX)

    # Parallel fits: ProcessPool chunks (CH=8, as the prior campaign's ray
    # chunks); serial fallback inside _pool_map.
    CH = 8
    payloads = [([(i, targets[i]) for i in range(c0, min(c0 + CH, n_targets))],
                 g_hi)
                for c0 in range(0, n_targets, CH)]
    results = [None] * n_targets
    for out_chunk in _pool_map(_t3_fit_chunk, payloads, n_workers, 'rg fits'):
        for idx, gfit, loss, okf in out_chunk:
            results[idx] = (gfit, loss, okf)

    g_rg = np.array([r[0] for r in results])
    losses = np.array([r[1] for r in results])
    flags = np.array([r[2] for r in results], dtype=bool)
    abs_err = np.abs(g_rg - g_true)

    # BCS inversion baseline for the overlay panel (published unregularized
    # control)  [from regen_t3 l.1093-1101]
    g_bcs_alt = np.full(n_targets, np.nan)
    if with_bcs_baseline:
        ENERG = ctx['ENERG']
        e_mean = ctx['e_mean']
        vals = []
        for i, g in enumerate(tqdm(g_true, desc='rg bcs baselines')):
            rho_t = targets[i]
            gi, _ = invert_bcs_direct(rho_t, None, ENERG, e_mean, np.inf, True,
                                      alt_loss=True)
            vals.append(gi)
        g_bcs_alt = np.asarray(vals, dtype=np.float64)

    return {
        'n_targets': int(n_targets),
        'median_abs_err': float(np.median(abs_err)),
        'max_abs_err': float(abs_err.max()),
        'n_not_converged': int(np.sum(~flags)),
        'g_range': [float(g_lo), float(g_hi)],
        'basis': {'d': int(basis.d), 'N': int(N_ELEC), 'pairs': False,
                  'size': int(basis.size)},
        'gate_pass': bool(np.median(abs_err) < RG_GATE_MEDIAN
                          and abs_err.max() < RG_GATE_MAX),
        'wall_s': time.time() - t0,
        'g_true': g_true, 'g_rg': g_rg, 'abs_err': abs_err,
        'losses': losses, 'converged': flags, 'g_bcs_alt': g_bcs_alt,
    }


# =============================================================================
# 7. Vectorial-panel physics kernels  [from cell_35 l.40-201]
# =============================================================================

#@njit(cache=True, fastmath=True)
def solve_gap_eq_stateless(g_params, energ, m_pairs, vrepeat, beta,
                           tol=1e-7, max_iter=2000, damping=0.6):
    """
    Reconstructs G from scalar params and finds Delta.
    """
    # Reconstruct G vector
    G_full = np.zeros(m_pairs, dtype=np.float64)
    n_params = len(g_params)

    # Map optimization parameters to distance vector
    for k in range(n_params):
        start_idx = (k + 1) * vrepeat
        end_idx = min(start_idx + vrepeat, m_pairs)
        if start_idx < m_pairs:
            G_full[start_idx:end_idx] = g_params[k]

    # Initialize with a macroscopic seed to avoid getting stuck in Delta=0
    delta = np.ones(m_pairs, dtype=np.float64) * 0.1
    delta_new = np.empty_like(delta)
    converged = False

    for _ in range(max_iter):
        E_k = np.sqrt(energ**2 + delta**2)

        if beta > 0:
            arg = np.abs(0.5 * beta * E_k)
            th_factor = np.where(arg < 20.0, np.tanh(arg), 1.0)
        else:
            th_factor = 1.0

        # ukvk = (Delta / 2E) * tanh(...)
        ukvk = (delta / (2.0 * np.maximum(E_k, 1e-12))) * th_factor

        # Gap Equation: Delta = G * ukvk
        delta_new[:] = 0.0
        for i in range(m_pairs):
            acc = 0.0
            for j in range(m_pairs):
                dist = abs(i - j)
                acc += G_full[dist] * ukvk[j]
            delta_new[i] = acc

        # Convergence check
        if np.max(np.abs(delta_new - delta)) < tol:
            converged = True
            break

        delta = damping * delta_new + (1.0 - damping) * delta

    return delta, converged

#@njit(cache=True, fastmath=True)
def calc_bcs_observables(delta, energ, m_pairs, beta):
    """Computes RDM diagonal (n_k) and Energy."""
    E_k = np.sqrt(energ**2 + delta**2)

    if beta > 0:
        arg = np.abs(0.5 * beta * E_k)
        th_factor = np.where(arg < 20.0, np.tanh(arg), 1.0)
    else:
        th_factor = 1.0

    # Off-diagonal (u*v)
    ukvk = (delta / (2.0 * np.maximum(E_k, 1e-12))) * th_factor

    # Diagonal occupation (n_k)
    # n_k = 0.5 * (1 - xi/E * tanh)
    xi_over_E = (energ / np.maximum(E_k, 1e-12)) * th_factor
    nk = 0.5 * (1.0 - xi_over_E)

    # Energy
    E_bcs = 2.0 * np.sum(energ * nk) - np.sum(delta * ukvk)

    return nk, ukvk, E_bcs

# ==============================================================================
# Inversion driver  [from cell_35 l.120-201]
# ==============================================================================

def bcs_variational_driver(rho_target, state_type, beta, energy_target,
                           energ_array, label_size, vrepeat,
                           lambda_smooth=0.1, lambda_energy=1.0):

    m_pairs = len(energ_array)

    rho_target = np.asarray(rho_target, dtype=np.float64).squeeze()
    energ_array = np.asarray(energ_array, dtype=np.float64)

    # Extract Targets
    if rho_target.ndim == 2:
        nk_target = np.diag(rho_target)
        rho_full_target = rho_target
    else:
        nk_target = rho_target
        rho_full_target = None

    # --- Cost Function ---
    def cost_fn(g_params):
        # Forward Pass
        delta_sc, converged = solve_gap_eq_stateless(
            g_params, energ_array, m_pairs, vrepeat, beta
        )

        if not converged: return 1e6

        # Observables
        nk_pred, ukvk_pred, E_pred = calc_bcs_observables(
            delta_sc, energ_array, m_pairs, beta
        )

        # Loss Terms
        # A. Diagonal Loss (Occupation)
        loss = np.mean((nk_pred - nk_target)**2)

        # B. Off-Diagonal Loss (Structure)
        if rho_full_target is not None:
            rho_pred_block = np.outer(ukvk_pred, ukvk_pred)
            np.fill_diagonal(rho_pred_block, nk_pred)
            loss += np.mean((rho_pred_block - rho_full_target)**2)

        # C. Energy Constraint
        if energy_target is not None:
             loss += lambda_energy * ((E_pred - energy_target) / m_pairs)**2

        # D. Smoothness
        if len(g_params) > 1:
            loss += lambda_smooth * np.mean(np.diff(g_params)**2)

        return loss

    # --- Initialization Strategy  ---

    # Grid Search
    grid_vals = np.linspace(0.2, 3.0, 8)
    best_scalar = 0.5
    best_loss = np.inf

    for g_val in grid_vals:
        x_grid = np.full(label_size, g_val)
        l = cost_fn(x_grid)
        if l < best_loss:
            best_loss = l
            best_scalar = g_val

    # Full Gradient Optimization
    x0 = np.full(label_size, best_scalar)
    bounds = [(0.0, 10.0) for _ in range(label_size)]

    res = scipy.optimize.minimize(
        cost_fn, x0, method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 150}
    )

    g_final = res.x
    delta_final, _ = solve_gap_eq_stateless(g_final, energ_array, m_pairs, vrepeat, beta)

    return g_final, delta_final, np.sqrt(res.fun)


def reconstruct_vec(labels, vr, m):
    # [from cell_36 l.5-11] label vector -> distance vector G(|i-j|)
    vec = np.zeros(m)
    for k, val in enumerate(labels):
        start = (k + 1) * vr
        end = min(start + vr, m)
        if start < m: vec[start:end] = val
    return vec


def parallel_robust_task_v2(rho_i, state_type, beta, actual_energy, energ_array, l_size, vr):
    # [from cell_37 l.12-18; @ray.remote removed per engine contract — the
    # signature is preserved and the call is dispatched via _pool_map]
    return bcs_variational_driver(
        rho_i, state_type, beta, actual_energy,
        energ_array, l_size, vr,
        lambda_smooth=L_SMOOTH, lambda_energy=L_ENERGY
    )


def _vect_task(payload):
    """Pool wrapper around parallel_robust_task_v2; returns the error string on
    failure (same contract as cell_35's parallel_task collector)."""
    try:
        return parallel_robust_task_v2(*payload)
    except Exception as e:
        return str(e)


def get_order_parameter(rho_matrix, g_vector):
        # [from cell_37 l.36-46] vector-G form; this redefinition shadows the
        # cell_33 scalar form above, mirroring notebook execution order (both
        # agree for scalar g and 2D rho: ||g|| == |g|).
        # 1. Take absolute value
        rho_abs = np.abs(rho_matrix).copy()

        # 2. Zero out the diagonal (Remove particle number contribution)
        # We only want off-diagonal coherence: sum_{i!=j} <P_i P_j>
        np.fill_diagonal(rho_abs, 0.0)

        # 3. Calculate metric
        coherence_sum = np.sum(rho_abs)
        return np.linalg.norm(g_vector) * np.sqrt(coherence_sum)


# =============================================================================
# 8. SPEC 3 — vectorial-panel inversions against exact ED targets
# =============================================================================

def bcs_vect_inversions(n_targets, seed, beta=None, n_workers=None,
                        g_init=0.1, g_stop=1.0, vrepeat=2):
    """Vectorial-model BCS variational inversion panel (cell_35/36/37 flow).

    Draws n_targets vectorial labels ~U(g_init, g_stop), sorted descending per
    row (the GGenerator 'vect' ensemble; SeedSequence([seed, 6, 0])), builds
    exact ED targets (rho2 + energy) from the current d12 era globals, runs the
    BCS variational inversion (lambda_smooth=L_SMOOTH, lambda_energy=L_ENERGY)
    ProcessPool-parallel, and assembles the panel observables of
    physical_observables_dashboard: accuracy ||G_opt - G_true||,
    renormalization Z = ||G_BCS||/||G_true||, order-parameter branches
    (exact vs BCS), occupancies and their MSE.

    beta defaults to the era BETA (passed straight into the driver, as the
    notebook did); pass np.inf for a strict T=0 BCS side. Exact targets follow
    STATE_TYPE: eigsh ground states for 'gs', thermal_state otherwise.
    """
    t0 = time.time()
    n = int(n_targets)
    # [from cell_35 l.9-10, l.231-233] cell-level config at call time
    ENERG_VECT = levels
    M_PAIRS = len(ENERG_VECT)
    ENERG_VECT_F64 = np.asarray(ENERG_VECT, dtype=np.float64)
    VREPEAT_REF = int(vrepeat)
    # [from GGenerator.label_size 'vect'] computed locally so the driver does
    # not depend on the era g_gen's h_type
    L_SIZE_REF = math.ceil(M_PAIRS / VREPEAT_REF) - 1
    beta_used = float(BETA) if beta is None else float(beta)
    try:
        st = STATE_TYPE
    except NameError:
        st = 'gs'
    is_gs = (st == 'gs')

    # --- exact ED targets (replaces predict_and_load dataset targets) ---
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 6, 0]))
    labels = rng.uniform(g_init, g_stop, size=(n, L_SIZE_REF))
    labels = np.sort(labels, axis=1)[:, ::-1]   # descending, per generate()

    # numpy mirror of GGenerator.reconstruct 'vect': G_ij = vec(|i-j|)
    idx_np = np.abs(np.arange(M_PAIRS)[:, None] - np.arange(M_PAIRS)[None, :])
    ctx = _rg_build_ctx()
    R = ctx['R']
    rho_targets = np.empty((n, M_PAIRS, M_PAIRS))
    e_targets = np.empty(n)
    for i in tqdm(range(n), desc='vect ED targets'):
        vec = reconstruct_vec(labels[i], VREPEAT_REF, M_PAIRS)
        G_mat = vec[idx_np]
        # H = H0 - sum_ij G[i,j] * R[j][i]: identical to
        # two_body_hamiltonian_sp(basis, e, G, rho_1_arrays, rho_2_kkbar_arrays)
        # (HI contraction 'bij,jirc'), assembled from the precomputed
        # pair-block operators (verified bit-exact against the direct builder).
        HI = None
        for a in range(M_PAIRS):
            for b in range(M_PAIRS):
                if G_mat[a, b] == 0.0:
                    continue
                term = G_mat[a, b] * R[b][a]
                HI = term if HI is None else HI + term
        H = ctx['H0'] - HI if HI is not None else ctx['H0'].copy()
        rho_targets[i], e_targets[i] = _d12_exact_target(
            H.tocsr(), ctx, beta_used, is_gs)

    # 3. Sort results  [from cell_35 l.30-38]
    g_norms_true = np.linalg.norm(labels, axis=1)
    g_ids = g_norms_true.argsort()
    g_true_sort = labels[g_ids]
    energy_sort = e_targets[g_ids]
    rho_actual_sort = rho_targets[g_ids]
    g_norms_sort = g_norms_true[g_ids]

    # --- inversion (cell_37 re-run with corrected constraints) ---
    payloads = [(rho_actual_sort[i], st, beta_used, energy_sort[i],
                 ENERG_VECT_F64, L_SIZE_REF, VREPEAT_REF)
                for i in range(len(g_true_sort))]
    raw = _pool_map(_vect_task, payloads, n_workers, 'vect inversions')

    # Collect  [from cell_37 l.25-34]
    g_opt_list_v2, delta_opt_list_v2 = [], []
    for res in raw:
        if isinstance(res, str):
            g_opt_list_v2.append(np.full(L_SIZE_REF, np.nan))
            delta_opt_list_v2.append(np.full(M_PAIRS, np.nan))
        else:
            g_opt_list_v2.append(res[0])
            delta_opt_list_v2.append(res[1])

    g_opt_arr_v2 = np.array(g_opt_list_v2)
    delta_opt_arr_v2 = np.array(delta_opt_list_v2)
    valid_mask_v2 = ~np.isnan(g_opt_arr_v2).any(axis=1)

    print(f"\nFinal success rate: {np.sum(valid_mask_v2)}/{len(valid_mask_v2)}")

    # --- panel observables  [from cell_37 physical_observables_dashboard
    #     l.48-126, data assembly only; plotting lives in
    #     plot_bcs_vect_panels] ---
    idx_valid = np.where(valid_mask_v2)[0]
    nk_exact_list, nk_bcs_list = [], []
    psi_exact_list, psi_bcs_list = [], []
    norms_true, norms_phys = [], []
    for i in idx_valid:
        rho = rho_actual_sort[i]
        if rho.ndim == 3 and rho.shape[-1] == 1: rho = rho.squeeze(-1)
        if rho.ndim == 1: rho = rho.reshape(M_PAIRS, M_PAIRS)
        nk_exact_list.append(np.diag(rho))

        g_params = g_opt_arr_v2[i]
        delta, _ = solve_gap_eq_stateless(g_params, ENERG_VECT_F64, M_PAIRS,
                                          VREPEAT_REF, beta_used)
        nk_b, ukvk_b, _ = calc_bcs_observables(delta, ENERG_VECT_F64, M_PAIRS,
                                               beta_used)
        nk_bcs_list.append(nk_b)

        psi_exact_list.append(get_order_parameter(rho, g_true_sort[i]))
        rhobcs = np.outer(ukvk_b, ukvk_b)
        np.fill_diagonal(rhobcs, nk_b)
        psi_bcs_list.append(get_order_parameter(rhobcs, g_params))

        norms_true.append(np.linalg.norm(g_true_sort[i]))
        norms_phys.append(np.linalg.norm(g_params))

    nk_exact_arr = np.array(nk_exact_list, dtype=np.float64)
    nk_bcs_arr = np.array(nk_bcs_list, dtype=np.float64)
    psi_exact_arr = np.array(psi_exact_list, dtype=np.float64)
    psi_bcs_arr = np.array(psi_bcs_list, dtype=np.float64)
    norms_true = np.array(norms_true, dtype=np.float64)
    norms_phys = np.array(norms_phys, dtype=np.float64)

    # Renormalization Z  [from cell_37 l.144-147]
    valid_norm = norms_true > 1e-9
    z_factor = np.zeros_like(norms_phys)
    z_factor[valid_norm] = norms_phys[valid_norm] / norms_true[valid_norm]

    # Occupancy MSE  [from cell_37 l.181]
    occ_error = (np.mean((nk_exact_arr - nk_bcs_arr)**2, axis=1)
                 if len(idx_valid) else np.zeros(0))

    # Accuracy  [from cell_36 l.24]
    err_phys = (np.linalg.norm(g_opt_arr_v2[valid_mask_v2]
                               - g_true_sort[valid_mask_v2], axis=1)
                if valid_mask_v2.any() else np.zeros(0))

    return {
        'g_true_labels': g_true_sort,
        'g_norms_true': g_norms_sort,
        'g_opt': g_opt_arr_v2,
        'delta_opt': delta_opt_arr_v2,
        'valid_mask': valid_mask_v2,
        'energy_targets': energy_sort,
        'rho_targets': rho_actual_sort,
        # arrays below are restricted to valid_mask rows (dashboard convention)
        'nk_exact': nk_exact_arr, 'nk_bcs': nk_bcs_arr,
        'psi_exact': psi_exact_arr, 'psi_bcs': psi_bcs_arr,
        'norms_true': norms_true, 'norms_phys': norms_phys,
        'z_factor': z_factor, 'occ_mse': occ_error, 'err_phys': err_phys,
        'beta': beta_used, 'state_type': st,
        'vrepeat': VREPEAT_REF, 'label_size': L_SIZE_REF,
        'summary': {
            'n_targets': int(n),
            'n_valid': int(np.sum(valid_mask_v2)),
            'median_err_phys': float(np.median(err_phys)) if err_phys.size else float('nan'),
            'max_err_phys': float(err_phys.max()) if err_phys.size else float('nan'),
            'median_z': float(np.median(z_factor)) if z_factor.size else float('nan'),
            'median_occ_mse': float(np.median(occ_error)) if occ_error.size else float('nan'),
            'lambda_energy': float(L_ENERGY), 'lambda_smooth': float(L_SMOOTH),
            'wall_s': time.time() - t0,
        },
    }


# =============================================================================
# 9. Manuscript figure functions (Agg backend; save .png dpi=300 + .pdf)
# =============================================================================

_BCS_RC = {  # [from cell_33 l.310-318]
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'figure.autolayout': True,
}


def _save_fig(fig, out_prefix, name):
    fig.savefig(f"{out_prefix}_{name}.png", dpi=300)
    fig.savefig(f"{out_prefix}_{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [f"{out_prefix}_{name}.png", f"{out_prefix}_{name}.pdf"]


def plot_bcs_uniform_panels(res, out_prefix, g_pred=None, rg=None):
    """Uniform-panel manuscript figures from a bcs_uniform_inversions result:
    {out_prefix}_interaction and {out_prefix}_order_parameter (.png + .pdf).
    g_pred: optional CNN predictions aligned with res['g_true'] (scatter
    overlay, cell_33 styling); rg: optional rg_canonical_inversion result for
    the Richardson-Gaudin overlay curve (regen_t3 preview styling)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update(_BCS_RC)
    written = []
    g_true_sorted = res['g_true']

    # --- Plot 1: Interaction Strength ---  [from cell_33 l.320-331]
    fig = plt.figure(figsize=(10, 8))
    plt.plot(g_true_sorted, g_true_sorted, 'k--', alpha=0.5, label='Identity')
    if g_pred is not None:
        plt.scatter(g_true_sorted, np.asarray(g_pred).ravel(), s=50, alpha=0.6,
                    color='tab:blue', label='CNN Prediction')
    plt.plot(g_true_sorted, res['g_bcs'], 'r-', label='BCS Inversion (Reg)')
    plt.plot(g_true_sorted, res['g_bcs_alt'], 'g-', label='BCS Inversion')
    if rg is not None:  # [from regen_t3 l.1111-1112]
        plt.plot(rg['g_true'], rg['g_rg'], color='darkorange', lw=3,
                 label='Canonical inversion (Richardson-Gaudin)')
    plt.xlabel(r"$G_{true}$")
    plt.ylabel(r"Predicted $G$")
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.grid(True, alpha=0.3, linestyle='--')
    written += _save_fig(fig, out_prefix, "interaction")

    # --- Plot 2: Order Parameter ---  [from cell_33 l.334-344]
    fig = plt.figure(figsize=(10, 8))
    plt.plot(g_true_sorted, res['psi_exact'], 'k--', label=r'Theoretical')
    plt.plot(g_true_sorted, res['psi_bcs'], 'r-', label=r'BCS Inversion' + ' (Reg)')
    plt.plot(g_true_sorted, res['psi_bcs_alt'], 'g-', label=r'BCS Inversion')
    plt.xlabel(r"$G_{true}$")
    plt.ylabel(r"Order Parameter")
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.grid(True, alpha=0.3, linestyle='--')
    written += _save_fig(fig, out_prefix, "order_parameter")
    return written


def plot_bcs_vect_panels(res, out_prefix, err_ml=None, psi_ml=None):
    """Vectorial-panel manuscript figures from a bcs_vect_inversions result:
    {out_prefix}_accuracy, {out_prefix}_renormalization,
    {out_prefix}_order_parameter_vect, {out_prefix}_occupancy_mse
    (.png + .pdf each). err_ml/psi_ml: optional CNN branches aligned with the
    valid rows (cell_36/37 overlays)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update(_BCS_RC)
    written = []
    nt = res['norms_true']

    # --- Accuracy ---  [from cell_36 l.26-40]
    fig = plt.figure(figsize=(10, 8))
    if err_ml is not None:
        plt.scatter(nt, np.asarray(err_ml).ravel(), alpha=0.6, s=50,
                    color='tab:blue', label='CNN Prediction')
    plt.scatter(nt, res['err_phys'], marker='x', c='tab:red', s=60,
                alpha=0.6, label='BCS Inversion')
    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"$||G_{pred} - G_{true}||$")
    plt.yscale('log')
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    written += _save_fig(fig, out_prefix, "accuracy")

    # --- Renormalization Z ---  [from cell_37 l.142-155]
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(nt, res['z_factor'], alpha=0.6, s=50, color='tab:blue')
    plt.axhline(1.0, color='k', ls='--', linewidth=2.5)
    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"$||G_{BCS}|| / ||G_{true}||$")
    plt.grid(True, alpha=0.3, linestyle='--')
    written += _save_fig(fig, out_prefix, "renormalization")

    # --- Order Parameter Fidelity ---  [from cell_37 l.158-176]
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(nt, res['psi_bcs'], c='tab:red', alpha=0.5, s=50,
                label='BCS Inversion')
    if psi_ml is not None:
        plt.scatter(nt, np.asarray(psi_ml).ravel(), c='tab:blue', alpha=0.5,
                    s=50, marker='x', label='CNN Prediction')
    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"Order Parameter")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    written += _save_fig(fig, out_prefix, "order_parameter_vect")

    # --- Occupancy Error vs Strength ---  [from cell_37 l.179-188]
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(nt, res['occ_mse'], c='tab:red', alpha=0.5, marker='x', s=60)
    plt.yscale('log')
    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"Occupancy MSE")
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    written += _save_fig(fig, out_prefix, "occupancy_mse")
    return written
