# ============================================================================
# p4_gevp.py -- covariance / GEVP diagnostics suite (patched, sign-fixed).
#
# Exec'd FOURTH into the shared engine namespace (see campaign/engine.py).
# Names from earlier parts (basis, g_gen, rho_1_arrays, rho_2_kkbar_arrays,
# U_ENERGY_SEED, GGenerator, two_body_hamiltonian_sp,
# _prepare_hamiltonian_for_eigsh, _to_dense_array from p1; S_matrix_np /
# get_gram_matrix / predict_and_load / GPU_BATCH_SIZE from p3) are resolved
# at CALL time via late binding -- exactly like the original notebook.
# Do not import other parts, do not import engine.
#
# Sources (verbatim transplants, edits marked with "[engine edit]"):
#   FermionicML_thermal_2body.py l.1944-2463 -> the PATCHED covariance suite
#       ([REGEN-EXTRACT covariance_suite]: get_w_N_pairing,
#       evaluate_subspace_projection, compute_gevp_observables,
#       compute_gevp_observables_diag, background_split_metrics,
#       build_background_vectors, plot_averaged_scree,
#       evaluate_ml_covariance_batch).  Canonical, preferred over cell_46.
#   .claude/cells/cell_40.py -> numba Wick machinery (m_index_map,
#       get_rdm_*_hcb, compute_M / compute_M_paired_hcb,
#       covariance_matrix(_paired), get_permutation_maps, compute_coords,
#       random_h).
#   .claude/cells/cell_43.py -> null-space check flow (adapted inside
#       null_mode_audit; the cell is a top-level script and cannot run at
#       exec time).
#   .claude/cells/cell_44.py -> get_pairing_symmetry_vectors (verbatim).
#
# Campaign drivers added at the bottom (drive_panel / sweep_stability /
# plot_scree_qk / null_mode_audit) follow evaluate_ml_covariance_batch's
# internals but take (g_pred, g_true) ARRAYS instead of running a model.
# All host-side diagnostics are float64 numpy.  numba njit transplants
# compile lazily (first call), nothing heavy runs at exec time.
# ============================================================================

import os
import sys
import itertools
from itertools import combinations
from functools import lru_cache

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse as sp
import scipy.sparse.linalg
import sparse
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')  # [engine edit] headless campaign rendering
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from numba import njit, prange, typed, types
from numba.typed import Dict

import fermionic_mbody as fmb
import openfermion as of


# ===========================================================================
# Numba Wick machinery  [from cell_40]
# ===========================================================================

def m_index_map(d, m):  # [from cell_40]
    """
    Creates a mapping between m-body creation operator strings and their
    index in the FixedBasis(d, m).
    """
    pairs = list(combinations(range(d), m))
    idx_basis = fmb.FixedBasis(d, m)

    mapping = {}
    for cre in pairs:
        term = of.FermionOperator(([(x, 1) for x in cre]))
        mapping[cre] = idx_basis.opr_to_idx(term)

    pairs_sorted = sorted(pairs, key=lambda cre: mapping[cre])

    return mapping, pairs_sorted

# --- Numba helper functions ---

@njit(cache=True)
def get_sorting_sign(indices):  # [from cell_40]
    """
    Calculates the sign required to sort the indices (fermionic reordering)
    """
    n = len(indices)
    if n <= 1:
        return 1

    # Check for duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if indices[i] == indices[j]:
                return 0

    # Calculate inversions
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if indices[i] > indices[j]:
                inversions += 1

    return 1 if inversions % 2 == 0 else -1

@njit(cache=True)
def get_rdm_2_element(creators, annihilators, rho_2, quad_map):  # [from cell_40]
    """
    Calculates <c_i^+ c_j^+ c_k c_l> using the 2-RDM matrix
    """
    sign_cre = get_sorting_sign(creators)
    if sign_cre == 0: return 0.0
    sign_ann = get_sorting_sign(annihilators)
    if sign_ann == 0: return 0.0

    if creators[0] > creators[1]:
        sorted_cre = (creators[1], creators[0])
    else:
        sorted_cre = creators

    if annihilators[0] > annihilators[1]:
        sorted_ann = (annihilators[1], annihilators[0])
    else:
        sorted_ann = annihilators

    if sorted_cre not in quad_map or sorted_ann not in quad_map:
        return 0.0

    idx_cre = quad_map[sorted_cre]
    idx_ann = quad_map[sorted_ann]

    sign = sign_cre * sign_ann

    return -1 * sign * rho_2[idx_ann, idx_cre]

@njit(cache=True)
def get_rdm_3_element(creators, annihilators, rho_3, trip_map):  # [from cell_40]
    """
    Calculates <c_i^+ c_j^+ c_k^+ c_l c_m c_n> using the 3-RDM matrix
    """
    sign_cre = get_sorting_sign(creators)
    if sign_cre == 0: return 0.0
    sign_ann = get_sorting_sign(annihilators)
    if sign_ann == 0: return 0.0

    cre_arr = np.array(creators)
    cre_arr.sort()
    sorted_cre = (cre_arr[0], cre_arr[1], cre_arr[2])

    ann_arr = np.array(annihilators)
    ann_arr.sort()
    sorted_ann = (ann_arr[0], ann_arr[1], ann_arr[2])

    if sorted_cre not in trip_map or sorted_ann not in trip_map:
        return 0.0

    idx_cre = trip_map[sorted_cre]
    idx_ann = trip_map[sorted_ann]

    sign = sign_cre * sign_ann

    return -1 * sign * rho_3[idx_ann, idx_cre]

@njit(cache=True)
def get_rdm_4_element(creators, annihilators, rho_4, quar_map):  # [from cell_40]
    """
    Calculates <c_i^+... c_p> using the 4-RDM matrix
    """
    sign_cre = get_sorting_sign(creators)
    if sign_cre == 0: return 0.0
    sign_ann = get_sorting_sign(annihilators)
    if sign_ann == 0: return 0.0

    cre_arr = np.array(creators)
    cre_arr.sort()
    sorted_cre = (cre_arr[0], cre_arr[1], cre_arr[2], cre_arr[3])

    ann_arr = np.array(annihilators)
    ann_arr.sort()
    sorted_ann = (ann_arr[0], ann_arr[1], ann_arr[2], ann_arr[3])

    if sorted_cre not in quar_map or sorted_ann not in quar_map:
        return 0.0

    idx_cre = quar_map[sorted_cre]
    idx_ann = quar_map[sorted_ann]

    sign = sign_cre * sign_ann

    return sign * rho_4[idx_ann, idx_cre]

# --- Numba helper functions for reduced space (Pairing) ---

@njit(cache=True)
def get_rdm_1_hcb(i, j, rho_1):  # [from cell_40]
    """ Calculates <B_i^+ B_j>. Convention: rho_1[j, i] = <B_i^+ B_j>"""
    # Bounds check required for Numba safety
    if i >= rho_1.shape[1] or j >= rho_1.shape[0]:
        return 0.0
    return rho_1[j, i]

@njit(cache=True)
def get_rdm_2_hcb(i, j, k, l, rho_2, map_2):  # [from cell_40]
    """ Calculates <B_i^+ B_j^+ B_k B_l>"""
    if i == j or k == l:
        return 0.0 # Hard-core constraint (B_i^+ B_i^+ = 0)

    # Sort creators and annihilators (bosonic symmetry, no signs).
    sorted_cre = (i, j) if i < j else (j, i)
    sorted_ann = (k, l) if k < l else (l, k)

    # Check existence in the map (required for Numba typed.Dict)
    if sorted_cre not in map_2 or sorted_ann not in map_2:
        return 0.0

    idx_cre = map_2[sorted_cre]
    idx_ann = map_2[sorted_ann]

    return rho_2[idx_ann, idx_cre]

# --- Optimized Wick expansion ---

@njit(cache=True)
def calculate_L1_L2_expectation_numba(i, j, k, l, p, q, r, s,
                                      rho_2, rho_3, rho_4,
                                      quad_map, trip_map, quar_map):  # [from cell_40]
    """
    Calculates < (c_i^+ c_j^+ c_k c_l) (c_p^+ c_q^+ c_r c_s) >
    """
    value = 0.0

    # --- 4-body term ---
    value += get_rdm_4_element((i, j, p, q), (k, l, r, s), rho_4, quar_map)

    # --- 3-body terms ---

    # Term: +delta_lq <c_i^+ c_j^+ c_p^+ c_k c_r c_s>
    if l == q:
        value += get_rdm_3_element((i, j, p), (k, r, s), rho_3, trip_map)

    # Term: -delta_lp <c_i^+ c_j^+ c_q^+ c_k c_r c_s>
    if l == p:
        value -= get_rdm_3_element((i, j, q), (k, r, s), rho_3, trip_map)

    # Term: +delta_kp <c_i^+ c_j^+ c_q^+ c_l c_r c_s>
    if k == p:
        value += get_rdm_3_element((i, j, q), (l, r, s), rho_3, trip_map)

    # Term: -delta_kq <c_i^+ c_j^+ c_p^+ c_l c_r c_s>
    if k == q:
        value -= get_rdm_3_element((i, j, p), (l, r, s), rho_3, trip_map)

    # --- 2-body terms  ---

    coeff_2 = 0
    if (l == p) and (k == q):
        coeff_2 += 1
    if (l == q) and (k == p):
        coeff_2 -= 1

    if coeff_2 != 0:
        value += coeff_2 * get_rdm_2_element((i, j), (r, s), rho_2, quad_map)

    return value

# --- Main computation loop ---

@njit(parallel=True)
def compute_M(all_indices, rho_2, rho_3, rho_4, quad_map, trip_map, quar_map):  # [from cell_40]
    """
    Calculates the covariance matrix
    M_ab = <L_a^\dag L_b> - <L_a^\dag><L_b>
    L_a = c_i^+ c_j^+ c_k c_l.
    """
    M_dim = len(all_indices)
    M = np.zeros((M_dim, M_dim))

    # Pre-calculate expectations <L_i>
    expectations = np.zeros(M_dim)
    for idx in prange(M_dim):
        i, j, k, l = all_indices[idx]
        expectations[idx] = get_rdm_2_element((i, j), (k, l), rho_2, quad_map)

    # Calculate the covariance matrix M
    for idx1 in prange(M_dim):
        i1, j1, k1, l1 = all_indices[idx1]
        exp_L1_dag = get_rdm_2_element((l1, k1), (j1, i1), rho_2, quad_map)

        for idx2 in range(idx1, M_dim):
            i2, j2, k2, l2 = all_indices[idx2]
            exp_L2 = expectations[idx2]

            # Calculate <L1^\dag L2>.
            # L1^\dag = c_k1^+ c_l1^+ c_i1 c_j1
            exp_L1dag_L2 = calculate_L1_L2_expectation_numba(
                l1, k1, j1, i1,  # Indices for L1^\dag
                i2, j2, k2, l2,  # Indices for L2
                rho_2, rho_3, rho_4, quad_map, trip_map, quar_map)

            M_val = exp_L1dag_L2 - exp_L1_dag * exp_L2

            M[idx1, idx2] = M_val
            if idx1 != idx2:
                M[idx2, idx1] = M_val

    return M

def convert_to_numba_dict(py_dict, key_type, value_type):  # [from cell_40]
    """Helper to convert Python dict to Numba typed dict"""
    numba_dict = Dict.empty(key_type=key_type, value_type=value_type)
    for k, v in py_dict.items():
        typed_key = tuple(int(x) for x in k)
        numba_dict[typed_key] = int(v)
    return numba_dict

def covariance_matrix(basis, rho_2, rho_3, rho_4, quad_map, trip_map, quar_map, quad_pairs):  # [from cell_40]
    """
    Calculates the covariance matrix M
    """
    int_type = types.int64
    key_type_2 = types.UniTuple(int_type, 2)
    key_type_3 = types.UniTuple(int_type, 3)
    key_type_4 = types.UniTuple(int_type, 4)
    value_type = int_type

    rho_2_np = np.asarray(rho_2, dtype=np.float64)
    rho_3_np = np.asarray(rho_3, dtype=np.float64)
    rho_4_np = np.asarray(rho_4, dtype=np.float64)

    quad_map_numba = convert_to_numba_dict(quad_map, key_type_2, value_type)
    trip_map_numba = convert_to_numba_dict(trip_map, key_type_3, value_type)
    quar_map_numba = convert_to_numba_dict(quar_map, key_type_4, value_type)

    all_indices = []

    for (i, j), (k, l) in itertools.product(quad_pairs, repeat=2):
        all_indices.append((int(i), int(j), int(k), int(l)))

    tuple_type_4 = types.UniTuple(int_type, 4)

    try:
        all_indices_numba = typed.List.empty_list(tuple_type_4)
    except AttributeError:
        all_indices_numba = typed.List()

    for item in all_indices:
        all_indices_numba.append(item)

    print(f"Computing covariance matrix (Dim={len(all_indices)}x{len(all_indices)})...")
    M = compute_M(all_indices_numba, rho_2_np, rho_3_np, rho_4_np,
                        quad_map_numba, trip_map_numba, quar_map_numba)

    return M

# --- Main computation loop (Reduced case) ---

#@njit(parallel=True)
def compute_M_paired_hcb(rho_1, rho_2, map_2):  # [from cell_40]
    """
    Calculates the covariance matrix for 1-body operators (B_i^+ B_j)
    """
    d_pair = rho_1.shape[0]
    M_dim = d_pair * d_pair
    dtype = rho_1.dtype # Use dtype of input RDM (handles complex if necessary)
    M = np.zeros((M_dim, M_dim), dtype=dtype)

    # Iterate over the flattened indices L1 = B_i^+ B_j, L2 = B_k^+ B_l
    for idx1 in prange(M_dim):
        i, j = divmod(idx1, d_pair)

        # <L1^\dag> = <B_j^\dag B_i>
        exp_L1_dag = get_rdm_1_hcb(j, i, rho_1)

        for idx2 in range(idx1, M_dim):
            k, l = divmod(idx2, d_pair)

            # <L2> = <B_k^\dag B_l>
            exp_L2 = get_rdm_1_hcb(k, l, rho_1)

            # Calculate <L1^\dag L2> = <B_j^\dag B_i B_k^\dag B_l>
            exp_L1dag_L2 = 0.0

            if i != k:
                # i != k: Bosonic commutation -> <B_j^\dag B_k^\dag B_i B_l>
                exp_L1dag_L2 = get_rdm_2_hcb(j, k, i, l, rho_2, map_2)
            else:
                # i = k: HCB relation -> <B_j^\dag B_l> - <B_j^\dag B_i^\dag B_i B_l>
                term1 = get_rdm_1_hcb(j, l, rho_1)
                # term2 handles hard-core constraints internally.
                term2 = get_rdm_2_hcb(j, i, i, l, rho_2, map_2)
                exp_L1dag_L2 = term1 - term2

            M_val = exp_L1dag_L2 - exp_L1_dag * exp_L2

            M[idx1, idx2] = M_val
            if idx1 != idx2:
                # Ensure hermiticity
                M[idx2, idx1] = np.conj(M_val)

    return M

def covariance_matrix_paired(basis, rho_1_pair, rho_2_pair):  # [from cell_40]
    """
    Calculates the covariance matrix M restricted to the pairing subspace
    """
    if not basis.pairs:
        raise ValueError("This function is intended for basis with pairs=True.")

    d_pair = basis.d // 2

    # Generate the 2-body map for the reduced (bosonic) system.
    # Requires m_index_map and convert_to_numba_dict to be in scope.
    map_2_red, _ = m_index_map(d_pair, 2)

    # Setup Numba types and convert inputs.
    int_type = types.int64
    key_type_2 = types.UniTuple(int_type, 2)

    rho_1_np = np.asarray(rho_1_pair, dtype=np.float64)
    rho_2_np = np.asarray(rho_2_pair, dtype=np.float64)

    map_2_numba = convert_to_numba_dict(map_2_red, key_type_2, int_type)

    M_dim = d_pair**2
    # Assuming rho_1_np and rho_2_np are the dense numpy arrays
    M = compute_M_paired_hcb(rho_1_np, rho_2_np, map_2_numba)

    return M

# --- Helper function for permutation maps ---

def get_permutation_maps(quad_map, quad_pairs):  # [from cell_40]
    """
    Calculates the permutation maps between Lexicographical (L) and Bitmask (B) ordering
    for the flattened 2-body basis indices.

    L_IJ = c_i+ c_j+ c_k c_l (i<j, k<l)
    R_JI = C_I+ C_J = (c_i+ c_j+) (c_l c_k) = -(c_i+ c_j+ c_k c_l) = -L_IJ.
    """
    D2 = len(quad_pairs)

    # 1D Map: P_1D[I_L] = I_B. Maps Lex index to Bitmask index.
    P_1D = np.array([quad_map[pair] for pair in quad_pairs], dtype=int)

    # 2D Flattened Map (L to B): a_B = P_L_to_B[a_L].
    # L_B is derived from R^T, so indices are (J_B, I_B). Flattening (row-major) gives a_B = J_B*D2 + I_B.
    # We map a_L=(I_L, J_L) to this a_B.
    # J_B = P_1D[I_L], I_B = P_1D[J_L].
    # a_B = P_1D[I_L]*D2 + P_1D[J_L].

    J_B_map = P_1D * D2
    I_B_map = P_1D

    # Permutation_2D[I_L, J_L] = a_B
    Permutation_2D = J_B_map[:, None] + I_B_map[None, :]
    P_L_to_B = Permutation_2D.ravel()

    # Inverse 2D Flattened Map (B to L): a_L = P_B_to_L[a_B].
    P_B_to_L = np.argsort(P_L_to_B)

    return P_1D, P_L_to_B, P_B_to_L

# --- Coordinate calculation  ---

def compute_coords(h, basis, rho_2_arrays, quad_map, quad_pairs):  # [from cell_40]
    """
    Calculates the coordinates (w) of Hamiltonian h.

    NOTE (campaign): this lstsq route is the coordinate map for the UNPAIRED
    2-body basis (cell_43 flow).  For the paired/seniority basis used by the
    d=20 panels the coordinate map reduces to direct row-major flattening of
    the G coefficient matrix (random_h Case 1: coefs = G.reshape(-1)), which
    is what the panel drivers below use -- identical to the
    evaluate_ml_covariance_batch call-site assembly.
    """
    D2 = rho_2_arrays.shape[0]
    M_dim = D2**2
    N_dim = basis.size
    N_dim_sq = N_dim**2

    # Define the operator basis in bitmask order
    # L_B corresponds to rows of M
    R_T = rho_2_arrays.transpose((1, 0, 2, 3))
    L_B_tensor = -R_T
    L_B = L_B_tensor.reshape((M_dim, N_dim, N_dim))

    # Calculate Gram Matrix S_B
    Y_coo = L_B.reshape((M_dim, N_dim_sq))
    Y_csr = Y_coo.to_scipy_sparse().tocsr().astype(np.float64)

    print("Calculating Gram matrix S_B = Y @ Y^T")
    S_B_sparse = Y_csr @ Y_csr.T
    S_B = S_B_sparse.toarray()

    # Calculate b_B
    if scipy.sparse.issparse(h):
        H_flat = h.toarray().reshape(-1)
    elif isinstance(h, sparse.COO):
        H_flat = h.todense().reshape(-1)
    else:
        H_flat = np.asarray(h).reshape(-1)

    H_flat = H_flat.astype(np.float64)
    b_B = Y_csr @ H_flat

    # Solve S_B w = b_B directly
    print("Solving the system S_B w = b_B...")
    w, residuals, rank, s_values = np.linalg.lstsq(S_B, b_B, rcond=1e-10)

    res_sum = np.sum(residuals) if residuals.size > 0 else 0.0
    print(f"System solved. Rank={rank}/{M_dim}. Residuals={res_sum:.4e}")

    return w, residuals, rank

# --- Hamiltonian generation ---

def random_h(basis, rho_gen, quad_map=None, quad_pairs=None):  # [from cell_40]
    """
    Generates a random Hermitian Hamiltonian H and its coefficients.

    Supports 3 cases:
    1. Restricted Basis (basis.pairs=True):
       Returns coefficients for the reduced basis (Size d/2)
    2. Pairing H in 2-body basis(basis.pairs=False, rho_gen is kkbar_gen):
       Returns H built efficiently, with coefficients mapped to the full 2-body basis (Size d^2/2).
    3. Full Random H (basis.pairs=False, rho_gen is full 2-body):
       Returns coefficients for the full 2-body basis.
    """

# --- Case 1: Hard-Core Boson Pairing basis ---
    if basis.pairs:
        d_pair = basis.d // 2
        # Negative (Attractive) interaction
        G = -np.random.uniform(0.1, 5, (d_pair, d_pair))
        G = 0.5 * (G + G.T)
        coefs = G.reshape(-1)

        h = scipy.sparse.coo_array(G).tensordot(rho_gen, axes=([0, 1], [1, 0]))

        return h, coefs

    # --- Cases 2 & 3: Full 2-Body Basis ---
    else:
        if quad_map is None or quad_pairs is None:
            raise ValueError("quad_map and quad_pairs required for fermionic basis")

        D2 = len(quad_pairs)
        shape_dim0 = rho_gen.shape[0]

        # Detect Case 3: Generator is for Pairing (kkbar) block only
        is_pairing_gen = (shape_dim0 == basis.d // 2)

        if is_pairing_gen:
            # --- Case 3: Pairing Hamiltonian in Full Basis ---
            d_pair = shape_dim0
            G = -np.random.uniform(0.1, 5, (d_pair, d_pair)) # Attractive
            G = 0.5 * (G + G.T)

            h = scipy.sparse.coo_array(G).tensordot(rho_gen, axes=([0, 1], [0, 1]))

            # Map G coefficients to the full coefficient vector
            coefs = np.zeros(D2**2)

            pair_indices = []
            for k in range(d_pair):
                key = tuple(sorted((2*k, 2*k+1)))
                if key not in quad_map:
                    raise ValueError(f"Pair {key} not found in quad_map")
                pair_indices.append(quad_map[key])

            pair_indices = np.array(pair_indices)

            # W matrix in Bitmask order
            W_B = np.zeros((D2, D2))
            rows = pair_indices[:, None]
            cols = pair_indices[None, :]

            W_B[rows, cols] = -G
            coefs = W_B.reshape(-1)

            return h, coefs

        else:
            # --- Case 2: Full random 2-Body Hamiltonian ---
            M_dim = D2**2
            N_dim = basis.size

            # Generate coefficients directly in Bitmask order
            W_B = np.random.uniform(0.1, 5, (D2, D2))
            W_B = 0.5 * (W_B + W_B.T)
            coefs_B = W_B.reshape(-1)

            R_T = rho_gen.transpose((1, 0, 2, 3))
            L_B_tensor = -R_T
            L_B = L_B_tensor.reshape((M_dim, N_dim, N_dim))

            h = scipy.sparse.coo_array(coefs_B).tensordot(L_B, axes=([0], [0]))

            return h, coefs_B


# ===========================================================================
# Local SU(2) pairing symmetry vectors  [from cell_44]
# ===========================================================================

def get_pairing_symmetry_vectors(basis, quad_map, quad_pairs):  # [from cell_44]
    """
    Generates the null space vectors corresponding to the Local SU(2) Physical Spin
    symmetries of the Pairing Hamiltonian.

    Theory:
    The ground state |Psi> of a Pairing Hamiltonian is a Seniority-Zero state
    (a product of singlets). It is annihilated by the local spin operators
    for every pair k:
        1. S_k^z = 0.5 * (n_2k - n_2k+1)   [Spin Imbalance]
        2. S_k^+ = c_2k^dag c_2k+1         [Spin Flip Up]
        3. S_k^- = c_2k+1^dag c_2k         [Spin Flip Down]

    Since M is a 2-body operator matrix, we 'lift' these 1-body symmetries
    to the 2-body sector by multiplying by the Number Operator N_hat:
        O_2body = N_hat * O_1body
    Since O_1body |Psi> = 0, O_2body |Psi> = 0, so these lie in the Null Space of M.
    """
    if basis.d % 2 != 0:
            raise ValueError("Basis dimension must be even for pairing.")

    d = basis.d
    n_pairs = d // 2
    D2 = len(quad_pairs)
    M_dim = D2 * D2

    symmetry_vectors = []
    labels = []

    def add_lifted_term(w, i, j, k, l, val):
        # 1. Sort Creators (Bitmask order usually implies i < j)
        cre = tuple(sorted((i, j)))
        sign_c = -1 if (i > j) else 1

        # 2. Sort Annihilators
        ann = tuple(sorted((k, l)))
        sign_a = -1 if (k > l) else 1

        # 3. Map to Index
        # Since quad_map and M are now both Bitmask, this is now 1:1
        if cre in quad_map and ann in quad_map:
            idx_cre = quad_map[cre]
            idx_ann = quad_map[ann]
            flat_idx = idx_cre * D2 + idx_ann
            w[flat_idx] += val * sign_c * sign_a

    for p in range(n_pairs):
        up = 2 * p
        dn = 2 * p + 1

        # --- 1. S_p^z Symmetry ---
        w_z = np.zeros(M_dim)
        for r in range(d):
            if r != up:
                add_lifted_term(w_z, up, r, r, up, 0.5)
            if r != dn:
                add_lifted_term(w_z, dn, r, r, dn, -0.5)

        if np.linalg.norm(w_z) > 1e-9:
            symmetry_vectors.append(w_z / np.linalg.norm(w_z))
            labels.append(f"S_{p}^z")

        # --- 2. S_p^+ Symmetry ---
        w_plus = np.zeros(M_dim)
        for r in range(d):
             if r != up and r != dn:
                 add_lifted_term(w_plus, r, up, r, dn, -1.0)

        if np.linalg.norm(w_plus) > 1e-9:
            symmetry_vectors.append(w_plus / np.linalg.norm(w_plus))
            labels.append(f"S_{p}^+")

        # --- 3. S_p^- Symmetry ---
        w_minus = np.zeros(M_dim)
        for r in range(d):
            if r != up and r != dn:
                add_lifted_term(w_minus, r, dn, r, up, -1.0)

        if np.linalg.norm(w_minus) > 1e-9:
            symmetry_vectors.append(w_minus / np.linalg.norm(w_minus))
            labels.append(f"S_{p}^-")

    return np.array(symmetry_vectors).T, labels


# ===========================================================================
# [REGEN-EXTRACT covariance_suite]  (patched suite, transplanted verbatim
# from FermionicML_thermal_2body.py l.1953-2415; edits marked)
# ===========================================================================
# =============================================================================
# 1. Subspace Projection (Gram-Schmidt)
# =============================================================================

def get_w_N_pairing(basis):  # [from export l.1959]
    """Constructs the Number Operator vector representation w_N for the pairing case."""
    d_pair = basis.d // 2
    w_N_mat = np.zeros((d_pair, d_pair))
    # N = 2 * sum(P_k^\dag P_k)
    np.fill_diagonal(w_N_mat, 2.0)
    return w_N_mat.flatten()

def evaluate_subspace_projection(w_pred, w_true, w_N, S):  # [from export l.1967]
    """
    Decomposes the ML prediction w_pred into the orthogonal subspace
    spanned by {H_true, N} using the invariant physical metric S.
    """
    w_pred = np.asarray(w_pred, dtype=np.float64).flatten()
    w_true = np.asarray(w_true, dtype=np.float64).flatten()
    w_N = np.asarray(w_N, dtype=np.float64).flatten()

    # Enforce symmetric metric
    S_sym = 0.5 * (S + S.T)

    def inner_S(u, v): return u.T @ S_sym @ v
    def norm_S(u): return np.sqrt(np.maximum(inner_S(u, u), 1e-16))

    # 1. Orthonormal Basis for {H_true, N}
    norm_H = norm_S(w_true)
    e_H = w_true / norm_H if norm_H > 1e-10 else np.zeros_like(w_true)

    # Orthogonalize N against H_true
    proj_N_on_H = inner_S(e_H, w_N)
    u_N = w_N - proj_N_on_H * e_H

    norm_u_N = norm_S(u_N)
    e_N = u_N / norm_u_N if norm_u_N > 1e-10 else np.zeros_like(u_N)

    # 2. Project ML Prediction
    norm_pred_sq = np.maximum(inner_S(w_pred, w_pred), 1e-16)

    c_H = inner_S(e_H, w_pred)
    c_N = inner_S(e_N, w_pred)

    # 3. Compute Fractional Distributions
    fraction_H = (c_H**2) / norm_pred_sq
    fraction_N = (c_N**2) / norm_pred_sq
    fraction_err = max(0.0, 1.0 - fraction_H - fraction_N)
    # [REGEN-PATCH P2] The raw prediction is what gets projected downstream
    # (manuscript Sec. III.B.4, Eq. qk). The old commented-out subtraction and
    # the "isolate the orthogonal error" framing were dead code / misleading.
    w_ortho = w_pred

    return fraction_H, fraction_N, fraction_err, w_ortho

# =============================================================================
# 2. GEVP & Plotting Functions
# =============================================================================

def compute_gevp_observables(M, S, w_eval, w_ref=None):  # [from export l.2014]
    """
    Solves the GEVP and returns sorted quantum variances and ML projections.
    Normalizes the projections against the Reference True Hamiltonian.
    """
    if w_ref is None:
        w_ref = w_eval

    M_sym = 0.5 * (M + M.T)
    S_sym = 0.5 * (S + S.T)
    M_dim = M.shape[0]

    w_eval = np.asarray(w_eval, dtype=np.float64).flatten()
    w_ref = np.asarray(w_ref, dtype=np.float64).flatten()

    # Regularize S to securely handle kinematic null space
    S_reg = S_sym + np.eye(M_dim) * 1e-9

    evals_gevp, V_gevp = scipy.linalg.eigh(M_sym, b=S_reg)

    # Filter out kinematic constraints
    v_S_v = np.sum(V_gevp * (S_sym @ V_gevp), axis=0)
    valid_mask = v_S_v > 1e-7

    V_phys = V_gevp[:, valid_mask]
    v_S_v_phys = v_S_v[valid_mask]

    # Strict S-Orthonormalization
    V_phys = V_phys / np.sqrt(v_S_v_phys)

    # Absolute Quantum Variance
    lambdas_phys = np.sum(V_phys * (M_sym @ V_phys), axis=0)

    # Sort physical modes ascending (softest to hardest)
    idx_phys = np.argsort(lambdas_phys)
    lambdas_phys = lambdas_phys[idx_phys]
    V_phys = V_phys[:, idx_phys]

    # Metric-Weighted Projections (Normalized relative to the TRUE Hamiltonian)
    norm_ref_sq = w_ref.T @ S_sym @ w_ref
    c_k_gevp = V_phys.T @ S_sym @ w_eval
    O_k_gevp = (c_k_gevp**2) / np.maximum(norm_ref_sq, 1e-16)

    return lambdas_phys, O_k_gevp


# [REGEN-PATCH P5] Diagnostic-enriched GEVP solve (manuscript Sec. III.B.4,
# Eqs. qk / gevp_residual / two_sided_bound). Same numerics as
# compute_gevp_observables (ridge 1e-9, cutoff v'Sv > 1e-7 by default), but
# also returns the per-sample diagnostics promised for Table tab:gevp_diag.
def compute_gevp_observables_diag(M, S, w_eval, w_ref=None,
                                  lam_ridge=1e-9, cutoff=1e-7,
                                  lam_pos_threshold=1e-12):  # [from export l.2064]
    if w_ref is None:
        w_ref = w_eval

    M_sym = 0.5 * (M + M.T)
    S_sym = 0.5 * (S + S.T)
    M_dim = M.shape[0]

    w_eval = np.asarray(w_eval, dtype=np.float64).flatten()
    w_ref = np.asarray(w_ref, dtype=np.float64).flatten()

    S_reg = S_sym + np.eye(M_dim) * lam_ridge

    evals_gevp, V_gevp = scipy.linalg.eigh(M_sym, b=S_reg)

    v_S_v = np.sum(V_gevp * (S_sym @ V_gevp), axis=0)
    valid_mask = v_S_v > cutoff

    V_phys = V_gevp[:, valid_mask]
    v_S_v_phys = v_S_v[valid_mask]
    V_phys = V_phys / np.sqrt(v_S_v_phys)

    lambdas_phys = np.sum(V_phys * (M_sym @ V_phys), axis=0)
    idx_phys = np.argsort(lambdas_phys)
    lambdas_phys = lambdas_phys[idx_phys]
    V_phys = V_phys[:, idx_phys]

    norm_eval_sq = max(float(w_eval @ S_sym @ w_eval), 1e-16)
    norm_ref_sq = max(float(w_ref @ S_sym @ w_ref), 1e-16)
    c_k = V_phys.T @ S_sym @ w_eval
    q_k = (c_k**2) / norm_ref_sq          # Eq. (qk) when w_ref defaults to w_eval

    # 'positive' lambda threshold = the null-band zero_tol (1e-12) used by
    # the scree grey band and n_null, so lam_min_pos/kappa/R never pick up
    # numerical-noise null modes (eigsh tol 1e-8 -> noise floor ~1e-16).
    # [engine edit] hard-coded 1e-12 promoted to the lam_pos_threshold kwarg
    # (default unchanged) so the campaign drivers can thread
    # C.LAMBDA_POS_THRESHOLD through one knob.
    tiny = lam_pos_threshold
    pos = lambdas_phys[lambdas_phys > tiny]
    lam_min_pos = float(pos.min()) if pos.size else np.nan
    lam_max = float(lambdas_phys.max()) if lambdas_phys.size else np.nan
    kappa = lam_max / lam_min_pos if (pos.size and lam_min_pos > 0) else np.nan

    # Eq. (gevp_residual): fixed-scale truncation residual
    quad_form = float(w_eval @ M_sym @ w_eval)
    recon = float(np.sum((c_k**2) * lambdas_phys))
    r = abs(quad_form - recon) / max(lam_max * norm_eval_sq, 1e-300)

    gevp_diag = {
        'lam_min_pos': lam_min_pos,
        'lam_max': lam_max,
        'kappa': kappa,
        'n_discarded': int(np.sum(~valid_mask)),
        'cond_S': float(np.linalg.cond(S_sym)),   # raw S; ridge reported separately
        'r': float(r),
        'quad_form': quad_form,
        'recon_sum': recon,
        'norm_eval_sq': norm_eval_sq,
        'V_phys': V_phys,
        'S_sym': S_sym,
        'c_k': c_k,
        # [engine edit] keep the DISCARDED (v'Sv <= cutoff) eigenvectors
        # instead of dropping them, so disc_err_weight (drive_panel) can audit
        # how much of the error hides in the discarded small-S-norm
        # directions (referee two-report finding (ii), main report Sec. 2).
        'V_disc': V_gevp[:, ~valid_mask],
        'v_S_v_disc': v_S_v[~valid_mask],
    }
    return lambdas_phys, q_k, gevp_diag


# [REGEN-PATCH P1] S-orthogonal background/disorder split of manuscript
# Sec. III.B.3, Eq. (eps_I). w_H0 / w_bg are built by
# build_background_vectors() consistent with the "FIX 1" call-site assembly.
def background_split_metrics(w_pred, w_true, w_H0, w_bg, S):  # [from export l.2131]
    """f_parallel and eps_I of manuscript Sec. III.B.3 / Eq. (eps_I)."""
    S_sym = 0.5 * (S + S.T)
    inner = lambda u, v: float(u @ S_sym @ v)
    e1 = w_H0 / np.sqrt(max(inner(w_H0, w_H0), 1e-16))
    u2 = w_bg - inner(e1, w_bg) * e1
    e2 = u2 / np.sqrt(max(inner(u2, u2), 1e-16))
    P_par  = lambda v: inner(e1, v) * e1 + inner(e2, v) * e2
    P_perp = lambda v: v - P_par(v)
    f_par = inner(P_par(w_true), P_par(w_true)) / max(inner(w_true, w_true), 1e-16)
    dw = w_pred - w_true
    den = inner(P_perp(w_true), P_perp(w_true))
    eps_I = np.sqrt(inner(P_perp(dw), P_perp(dw)) / den) if den > 1e-14 else np.nan
    return f_par, eps_I


# [from export l.2151 build_background_vectors -- SIGN FIX applied]
# [engine edit -- SIGN FIX, referee report item 4(f) / publication-blocking
# item 1]: under the manuscript convention H_I = -sum_{kk'} G_{kk'} P_k^+ P_k'
# (eq:h_pair, l.324) the exact split is
#     H_I = -<G> sum_{ij} P_i^+ P_j  -  sum_{ij} (G_ij - <G>) P_i^+ P_j ,
# i.e. H_bg = -<G> sum_{ij} P_i^+ P_j (the manuscript's l.623 "+<G>" is the
# sign error). The coordinate map used by the panel call site is G -> -G
# flattened (plus the H0 diagonal), so the corrected w_bg is the image of the
# uniform matrix <G>*ones under the SAME map: w_bg = -<G> * ones.flatten().
# w_H0 sign VERIFIED correct: H = H0 - HI puts +pair_energies on the diagonal
# (matches the FIX-1 call-site assembly w[diag] += pair_energies).
# The export's "+ones, scale-invariant projector" shortcut spans the same
# line, so P_par/P_perp (hence f_par, eps_I) are numerically unchanged, but
# the components are now built from the literally correct H_bg coefficients
# (the cross-terms 2<w_0,w_bg>_S and 2<w_bg,dw_I>_S quoted in the manuscript
# carry the corrected sign).
def build_background_vectors(pair_energies_true, m_pairs, g_mean=None):
    w_H0 = np.zeros((m_pairs, m_pairs), dtype=np.float64)
    np.fill_diagonal(w_H0, np.asarray(pair_energies_true, dtype=np.float64))
    if g_mean is None:
        # legacy export behaviour (projector-equivalent span)
        w_bg = np.ones((m_pairs, m_pairs), dtype=np.float64)
    else:
        # corrected H_bg = -<G> sum_{ij} P_i^+ P_j, <G> = diag-included mean
        w_bg = -float(g_mean) * np.ones((m_pairs, m_pairs), dtype=np.float64)
    return w_H0.flatten(), w_bg.flatten()


def plot_averaged_scree(lambdas_phys, O_k_mean, O_k_std, num_samples, title_suffix="", ortho_err_pct=None):  # [from export l.2158]
    """Plots the batch-averaged Covariance Scree plot."""
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    k_indices = np.arange(len(lambdas_phys))
    lambdas_safe = np.maximum(lambdas_phys, 1e-16)

    # Physical Spectrum (Line Plot)
    ax2.plot(k_indices, lambdas_safe, color='tab:red', lw=3.0, zorder=3,
             label=r'Geometric Mean Spectrum ($\langle \lambda_k \rangle$)')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Quantum Variance ($\lambda_k$)', color='tab:red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # ML Deposition (Bar Chart)
    ax1.bar(k_indices, O_k_mean, capsize=3, ecolor='black',
            color='tab:blue', alpha=0.6, width=1.0, zorder=2,
            label=r'Avg Spectral Weight ($\langle q_k \rangle$, Eq. $q_k$)')  # [REGEN-PATCH P2]

    ax1.set_ylabel(r'Spectral Weight ($q_k$)', color='tab:blue', fontweight='bold')  # [REGEN-PATCH P2]
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Normal Mode Rank Index ($k$)', fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 2.0)
    ax1.set_xlim(-2, len(k_indices) + 2)

    # Exact Symmetry Manifold Shading
    zero_tol = 1e-12
    n_null = np.sum(lambdas_phys < zero_tol)
    if n_null > 0:
        ax1.axvspan(-0.5, n_null - 0.5, color='gray', alpha=0.15, zorder=1,
                    label=f'Exact Symmetries (~{n_null})')

    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    if title_suffix:
        pass
        #plt.title(f"Covariance Error Projection ({title_suffix})", fontweight='bold', pad=15)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', framealpha=0.9, fontsize=12)

    if ortho_err_pct is not None:
    # Format text and color based on performance
        err_text = f"Orthogonal Error: {ortho_err_pct:.2f}%"
        box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        #ax1.text(0.03, 0.95, err_text,
        #         transform=ax1.transAxes, fontsize=13, fontweight='bold',
        #         verticalalignment='top', bbox=box_props, zorder=5)

    fig.tight_layout()
    plt.close(fig)  # [engine edit] notebook plt.show() -> close (Agg backend)

# =============================================================================
# 3. Main Evaluation Loop
# =============================================================================

# [REGEN-PATCH P5] kwargs lam_ridge/cutoff expose the GEVP grid for the
# stability sweep; make_plot/return_diag let the regeneration driver collect
# the Table tab:gevp_diag quantities without altering the legacy call sites.
def evaluate_ml_covariance_batch(val_loader, state_model, basis, current_h_type, inc_energy=True, num_samples=15, title_suffix='',
                                 lam_ridge=1e-9, cutoff=1e-7, make_plot=True, return_diag=False):  # [from export l.2220]
    print("\n" + "="*80)
    print(f" BATCH GEOMETRIC EVALUATION (N={num_samples}) | Type: {current_h_type.upper()}")
    print("="*80)

    if not basis.pairs:
        raise ValueError("This pipeline is explicitly streamlined for the pairing basis only.")

    print(f"[1/4] Loading {num_samples} validation samples through ML Model...")
    _, _, g_true_arr, g_pred_arr = predict_and_load(
        loader=val_loader, state=state_model, num_samples=num_samples,
        gpu_batch_size=GPU_BATCH_SIZE, include_energy=inc_energy
    )

    print("[2/4] Constructing Universal Gram Matrix S & w_N...")
    active_rho_gen = fmb.rho_m_gen(basis, 1, n_workers=1)
    S = get_gram_matrix(active_rho_gen)
    w_N = get_w_N_pairing(basis)

    all_lambdas = []
    all_O_k = []          # holds q_k per sample after [REGEN-PATCH P2]
    all_subspace = []
    # [REGEN-PATCH P5] per-sample diagnostics for Table tab:gevp_diag
    all_diag = {k: [] for k in ('r', 'n_discarded', 'lam_min_pos', 'lam_max',
                                'kappa', 'cond_S', 'R', 'rel_S_err', 'ratio_S',
                                'f_par', 'eps_I', 'sum_qk', 'quad_form',
                                'recon_sum', 'norm_eval_sq', 'n_null')}

    print(f"[3/4] Solving GEVP local geometry iteratively...")

    for idx in tqdm(range(num_samples), desc="Processing Batch"):
        g_true = np.array([g_true_arr[idx]])
        g_pred = np.array([g_pred_arr[idx]])

        g_gen_eval = GGenerator(basis, current_h_type, 1)

        if current_h_type == 'randomenerg':
            e_true, V_true = [np.array(x) for x in g_gen_eval.reconstruct(jnp.array(g_true))]
            e_pred, V_pred = [np.array(x) for x in g_gen_eval.reconstruct(jnp.array(g_pred))]
            if e_true.shape[1] * 2 == basis.d:
                e_true, e_pred = np.repeat(e_true, 2, axis=1), np.repeat(e_pred, 2, axis=1)
        else:
            V_true = np.array(g_gen_eval.reconstruct(jnp.array(g_true)))
            V_pred = np.array(g_gen_eval.reconstruct(jnp.array(g_pred)))
            e_true = e_pred = U_ENERGY_SEED[0:1]

        inter_tensor = rho_2_block_arrays if current_h_type in ['blockgen', 'blockgensimp', 'randomenerg'] else (rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays)

        H_true_sp = two_body_hamiltonian_sp(basis, e_true, V_true, rho_1_arrays, inter_tensor, current_h_type)[0]

        # Ground State
        H_true_csr = _prepare_hamiltonian_for_eigsh(H_true_sp)
        try:
            _, v_true = scipy.sparse.linalg.eigsh(H_true_csr, k=1, which='SA', tol=1e-8)
            gs_state = v_true[:, 0]
        except Exception:
            continue

        # Local Covariance M
        rho_1_gs = fmb.rho_m_direct(basis, 1, gs_state)
        rho_2_gs = fmb.rho_m_direct(basis, 2, gs_state)
        M = covariance_matrix_paired(basis, np.asarray(rho_1_gs), np.asarray(rho_2_gs))

        # Extract Hamiltonian Tensors
        w_pred, w_true_extract = -V_pred[0].copy(), -V_true[0].copy()

        # ---> THE FIX 1: Add H0 to the diagonal to evaluate the FULL Hamiltonian
        m_pairs = basis.d // 2
        pair_energies_pred = e_pred.reshape(1, m_pairs, 2).sum(axis=-1)[0]
        pair_energies_true = e_true.reshape(1, m_pairs, 2).sum(axis=-1)[0]
        diag_idx = np.arange(m_pairs)

        w_pred[diag_idx, diag_idx] += pair_energies_pred
        w_true_extract[diag_idx, diag_idx] += pair_energies_true

        w_eval_pred = w_pred.flatten()
        w_eval_true = w_true_extract.flatten()

        # ---> THE FIX 2: Evaluate Absolute Subspace Compositon of the Raw ML Prediction
        f_H, f_N, f_err, w_ortho = evaluate_subspace_projection(w_eval_pred, w_eval_true, w_N, S)
        all_subspace.append((f_H, f_N, f_err))

        # [REGEN-PATCH P2] Project the RAW prediction with the default
        # w_ref = w_eval, yielding exactly q_k = |c_k|^2 / ||w_pred||_S^2
        # (manuscript Eq. qk). The old w_err binding and the w_ref=w_eval_true
        # override produced the deprecated true-normalized O_k convention.
        lambdas_phys, q_k_pred, gevp_diag = compute_gevp_observables_diag(
            M, S, w_eval=w_eval_pred, lam_ridge=lam_ridge, cutoff=cutoff)

        all_lambdas.append(lambdas_phys)
        all_O_k.append(q_k_pred)

        # [REGEN-PATCH P5] Table tab:gevp_diag accumulations
        S_sym_d = gevp_diag['S_sym']
        V_phys_d = gevp_diag['V_phys']
        dw = w_eval_pred.astype(np.float64) - w_eval_true.astype(np.float64)
        dw_S_dw = float(dw @ S_sym_d @ dw)
        true_S_true = max(float(w_eval_true @ S_sym_d @ w_eval_true), 1e-16)
        c_dw = V_phys_d.T @ S_sym_d @ dw
        pos_mask = lambdas_phys > 1e-12   # same null-band zero_tol as n_null
        # Eq. (orientation_ratio): stiff-mode share of the error, same retained V_phys
        R_val = float(np.sum(c_dw[pos_mask]**2) / max(dw_S_dw, 1e-300))
        all_diag['R'].append(R_val)
        all_diag['rel_S_err'].append(dw_S_dw / true_S_true)
        all_diag['ratio_S'].append(np.sqrt(gevp_diag['norm_eval_sq'] / true_S_true))

        # [REGEN-PATCH P1] background/disorder split (eps_I defined for
        # disordered panels only; const yields P_perp w_true ~ 0 -> NaN)
        w_H0_vec, w_bg_vec = build_background_vectors(pair_energies_true, m_pairs)
        f_par_s, eps_I_s = background_split_metrics(
            w_eval_pred.astype(np.float64), w_eval_true.astype(np.float64),
            w_H0_vec, w_bg_vec, S)
        all_diag['f_par'].append(f_par_s)
        all_diag['eps_I'].append(eps_I_s)

        for key in ('r', 'n_discarded', 'lam_min_pos', 'lam_max', 'kappa',
                    'cond_S', 'quad_form', 'recon_sum', 'norm_eval_sq'):
            all_diag[key].append(gevp_diag[key])
        all_diag['sum_qk'].append(float(np.sum(q_k_pred)))
        # same criterion as the plot's grey band (zero_tol = 1e-12 axvspan)
        all_diag['n_null'].append(int(np.sum(lambdas_phys < 1e-12)))

    print("\n[4/4] Averaging distributions and Rendering Plot...")

    # Handle slight possible variations in valid rank
    min_len = min([len(l) for l in all_lambdas])
    if min_len > 0:
        all_lambdas = np.array([l[:min_len] for l in all_lambdas])
        all_O_k = np.array([o[:min_len] for o in all_O_k])

        # Averages (q_k arithmetic mean; lambda geometric mean)
        O_k_mean = np.mean(all_O_k, axis=0)
        O_k_std = np.std(all_O_k, axis=0)
        lambdas_mean = np.exp(np.mean(np.log(np.maximum(all_lambdas, 1e-16)), axis=0))
        subspace_mean = np.mean(all_subspace, axis=0)
        ortho_err_pct = subspace_mean[2] * 100

        if make_plot:  # [REGEN-PATCH P5]
            plot_averaged_scree(lambdas_mean, O_k_mean, O_k_std, num_samples,
                                title_suffix=title_suffix,
                                ortho_err_pct=ortho_err_pct)

    else:
        print("Warning: Could not compute GEVP for any sample.")
        lambdas_mean, O_k_mean, O_k_std = None, None, None

    subspace_mean = np.mean(all_subspace, axis=0)

    # [REGEN-PATCH P5] aggregate the Table tab:gevp_diag quantities
    if return_diag:
        n_disc = np.asarray(all_diag['n_discarded'])
        eps_arr = np.asarray(all_diag['eps_I'], dtype=np.float64)
        diag = {
            'f_err_mean': float(subspace_mean[2]),
            'f_H_mean': float(subspace_mean[0]),
            'f_N_mean': float(subspace_mean[1]),
            'f_par_mean': float(np.mean(all_diag['f_par'])),
            'eps_I_mean': (float(np.nanmean(eps_arr))
                           if np.any(np.isfinite(eps_arr)) else float('nan')),
            'eps_I_nan_count': int(np.sum(~np.isfinite(eps_arr))),
            'ratio_S_median': float(np.median(all_diag['ratio_S'])),
            'ratio_S_mean': float(np.mean(all_diag['ratio_S'])),
            'lam_min_pos_geo': float(np.exp(np.nanmean(np.log(np.maximum(
                np.asarray(all_diag['lam_min_pos']), 1e-300))))),
            'lam_max_geo': float(np.exp(np.nanmean(np.log(np.maximum(
                np.asarray(all_diag['lam_max']), 1e-300))))),
            'kappa_geo': float(np.exp(np.nanmean(np.log(np.maximum(
                np.asarray(all_diag['kappa']), 1e-300))))),
            'r_max': float(np.nanmax(all_diag['r'])),
            'r_med': float(np.nanmedian(all_diag['r'])),
            'n_discarded_mode': int(np.bincount(n_disc).argmax()),
            'n_discarded_range': [int(n_disc.min()), int(n_disc.max())],
            'cond_S': float(np.median(all_diag['cond_S'])),
            'R_mean': float(np.mean(all_diag['R'])),
            'rel_S_err_mean': float(np.mean(all_diag['rel_S_err'])),
            'n_null_mode': int(np.bincount(np.asarray(all_diag['n_null'])).argmax()),
            'n_null_range': [int(np.min(all_diag['n_null'])),
                             int(np.max(all_diag['n_null']))],
            'sum_qk_max': float(np.max(all_diag['sum_qk'])),
            'lam_ridge': lam_ridge,
            'cutoff': cutoff,
            'per_sample': all_diag,
            'O_k_std': O_k_std,
            'all_lambdas_geo': lambdas_mean,
        }
        if O_k_mean is not None:
            above3 = np.where(O_k_mean > 1e-3)[0]
            above5 = np.where(O_k_mean > 1e-5)[0]
            diag['k_last_q_gt_1e3'] = int(above3.max()) if above3.size else -1
            diag['k_last_q_gt_1e5'] = int(above5.max()) if above5.size else -1
        return lambdas_mean, O_k_mean, subspace_mean, diag

    return lambdas_mean, O_k_mean, subspace_mean

# [REGEN-EXTRACT-END covariance_suite]


# ===========================================================================
# Campaign panel drivers (model-free: take (g_pred, g_true) label ARRAYS).
# These follow evaluate_ml_covariance_batch's internals exactly; the only
# behavioural deviations are marked [engine edit] below.
# ===========================================================================

_PANEL_S_CACHE = {}


def _panel_gram_matrix(rho_tensor):
    """[from cell_16 get_gram_matrix] Exact Hilbert-Schmidt metric
    S_ab = Tr(L_a^dag L_b) / D (private fallback copy; p3's get_gram_matrix
    is the same code and is preferred when init_gram() has run)."""
    rho_dense = np.asarray(rho_tensor.todense() if hasattr(rho_tensor, 'todense') else rho_tensor)
    m1, m2, N_dim, _ = rho_dense.shape

    # Align transposition to match how two_body_hamiltonian_dense constructs H
    # G[i, j] multiplies rho_tensor[j, i]
    L_tensor = rho_dense.transpose((1, 0, 2, 3))
    L_flat = L_tensor.reshape((m1 * m2, N_dim**2))

    Y_csr = sp.csr_matrix(L_flat.astype(np.float64))
    S_sparse = Y_csr @ Y_csr.T
    S = S_sparse.toarray()

    S_norm = S / float(N_dim)

    return 0.5 * (S_norm + S_norm.T)


def _panel_gram_S():
    """Gram metric S for the panels: prefer the init_gram() global S_matrix_np
    (p3), else fall back to the covariance-suite construction
    get_gram_matrix(rho_m_gen(basis, 1)) [export l.2236-2237].  For the
    paired basis rho_2_kkbar_gen(basis) == rho_m_gen(basis, 1) (verified),
    so both routes give the identical metric."""
    g = globals()
    S = g.get('S_matrix_np', None)
    if S is not None:
        return np.asarray(S, dtype=np.float64)
    key = (int(basis.d), int(getattr(basis, 'num', 0) or 0), bool(basis.pairs))
    if key not in _PANEL_S_CACHE:
        active_rho_gen = fmb.rho_m_gen(basis, 1, n_workers=1)
        _PANEL_S_CACHE[key] = np.asarray(
            _panel_gram_matrix(active_rho_gen), dtype=np.float64)
    return _PANEL_S_CACHE[key]


def _panel_reconstruct(labels, n):
    """Labels -> (V, e) host arrays for the first n samples.

    [engine edit] the export re-created GGenerator(basis, h_type, 1) INSIDE
    the per-sample loop, which retriggers jit compilation for every sample
    (GGenerator.reconstruct is jit'ed with self static).  Here reconstruction
    is chunked through one generator per chunk -- math identical to the
    export's per-sample calls."""
    current_h_type = g_gen.h_type
    labels = np.asarray(labels)[:n]
    chunks_V, chunks_e = [], []
    for s in range(0, len(labels), 512):
        chunk = labels[s:s + 512]
        g_gen_eval = GGenerator(basis, current_h_type, len(chunk))
        rec = g_gen_eval.reconstruct(jnp.array(chunk))
        if current_h_type == 'randomenerg':
            e_c, V_c = [np.array(x) for x in rec]
            if e_c.shape[1] * 2 == basis.d:
                e_c = np.repeat(e_c, 2, axis=1)
            chunks_e.append(e_c)
        else:
            V_c = np.array(rec)
        chunks_V.append(V_c)
    V = np.concatenate(chunks_V, axis=0)
    if current_h_type == 'randomenerg':
        e = np.concatenate(chunks_e, axis=0)
    else:
        e = np.repeat(np.asarray(U_ENERGY_SEED[0:1]), len(V), axis=0)
    return V, e


def _panel_sample_geometry(V_true_i, e_true_i, V_pred_i, e_pred_i):
    """One held-out sample -> covariance matrix M of the exact ground state
    of H(G_true) plus the FIX-1 coefficient vectors w_pred / w_true
    [follows evaluate_ml_covariance_batch l.2267-2297]."""
    current_h_type = g_gen.h_type
    inter_tensor = rho_2_block_arrays if current_h_type in ['blockgen', 'blockgensimp', 'randomenerg'] else (rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays)

    H_true_sp = two_body_hamiltonian_sp(basis, np.asarray(e_true_i)[None],
                                        np.asarray(V_true_i)[None],
                                        rho_1_arrays, inter_tensor,
                                        current_h_type)[0]

    # Ground State
    # [engine edit] exact GS by float64 DENSE eigh (spec item 2) instead of
    # the export's eigsh(tol=1e-8): removes the ~1e-16 ARPACK noise floor
    # that polluted lam_min_pos before the corrected (1e-12 threshold) pass.
    H_dense = np.asarray(_to_dense_array(_prepare_hamiltonian_for_eigsh(H_true_sp)),
                         dtype=np.float64)
    H_dense = 0.5 * (H_dense + H_dense.T)
    _, evecs = np.linalg.eigh(H_dense)
    gs_state = np.asarray(evecs[:, 0], dtype=np.float64)

    # Local Covariance M (HCB pair RDMs in the seniority basis)
    rho_1_gs = fmb.rho_m_direct(basis, 1, gs_state)
    rho_2_gs = fmb.rho_m_direct(basis, 2, gs_state)
    M = covariance_matrix_paired(basis, np.asarray(rho_1_gs), np.asarray(rho_2_gs))

    # Extract Hamiltonian Tensors (coordinate map of the paired basis:
    # G -> -G row-major flatten, cf. random_h Case 1 / compute_coords note)
    w_pred = -np.asarray(V_pred_i, dtype=np.float64).copy()
    w_true_extract = -np.asarray(V_true_i, dtype=np.float64).copy()

    # ---> THE FIX 1: Add H0 to the diagonal to evaluate the FULL Hamiltonian
    m_pairs = basis.d // 2
    pair_energies_pred = np.asarray(e_pred_i, dtype=np.float64).reshape(m_pairs, 2).sum(axis=-1)
    pair_energies_true = np.asarray(e_true_i, dtype=np.float64).reshape(m_pairs, 2).sum(axis=-1)
    diag_idx = np.arange(m_pairs)

    w_pred[diag_idx, diag_idx] += pair_energies_pred
    w_true_extract[diag_idx, diag_idx] += pair_energies_true

    return {
        'M': np.asarray(M, dtype=np.float64),
        'w_pred': w_pred.flatten(),
        'w_true': w_true_extract.flatten(),
        'pair_energies_true': pair_energies_true,
        # diag-included <G> of the TRUE interaction matrix (sign-fix input)
        'g_mean_true': float(np.mean(np.asarray(V_true_i, dtype=np.float64))),
    }


def _iter_panel_samples(g_pred, g_true, n_samples, pred_energies=None,
                        progress=True, desc="GEVP panel"):
    """Generator over per-sample geometries for the first n_samples pairs."""
    if not basis.pairs:
        raise ValueError("GEVP covariance panels require the paired "
                         "(seniority) basis -- run init_d20 first.")
    n = int(min(n_samples, len(g_pred), len(g_true)))
    V_true, e_true = _panel_reconstruct(g_true, n)
    V_pred, e_pred = _panel_reconstruct(g_pred, n)
    if pred_energies is not None:
        e_pred = np.asarray(pred_energies, dtype=np.float64)[:n]
        if e_pred.ndim == 2 and e_pred.shape[1] * 2 == basis.d:
            e_pred = np.repeat(e_pred, 2, axis=1)
    it = range(n)
    if progress:
        it = tqdm(it, desc=desc, file=sys.stdout, mininterval=1.0)
    for i in it:
        try:
            yield _panel_sample_geometry(V_true[i], e_true[i],
                                         V_pred[i], e_pred[i])
        except np.linalg.LinAlgError as exc:
            print(f"[p4_gevp] sample {i} skipped ({exc})", flush=True)
            continue


def discarded_error_weight(gevp_diag, dw):
    """[NEW -- referee two-report finding (ii)] S-weight of the error
    dw = w_pred - w_true inside the DISCARDED (v'Sv <= cutoff) GEVP
    directions: sum_k |<v_k, dw>_S|^2 / ||dw||_S^2 over the S-normalized
    discarded eigenvectors (kept by compute_gevp_observables_diag rather
    than dropped).  Quantifies how much error could hide in small-S-norm
    directions of large variance per S-norm (manuscript l.199 caveat)."""
    V_disc = gevp_diag.get('V_disc')
    if V_disc is None or V_disc.size == 0:
        return 0.0
    S_sym = gevp_diag['S_sym']
    v_S_v = np.asarray(gevp_diag['v_S_v_disc'], dtype=np.float64)
    keep = v_S_v > 1e-300
    if not np.any(keep):
        return 0.0
    Vn = V_disc[:, keep] / np.sqrt(v_S_v[keep])
    dw = np.asarray(dw, dtype=np.float64)
    c = Vn.T @ (S_sym @ dw)
    dw_S = float(dw @ S_sym @ dw)
    return float(np.sum(c**2) / max(dw_S, 1e-300))


def drive_panel(g_pred, g_true, n_samples, lam_ridge=1e-9, cutoff=1e-7,
                lam_pos_threshold=1e-12, pred_energies=None, progress=True,
                **_ignored):
    """One covariance panel (values_prev.json `covariance.*` block shape).

    Model-free: g_pred/g_true are (N, label_size) label arrays on the
    CURRENT era (init_d20 h_type via the g_gen global).  Per sample the
    exact ground state of H(G_true) is solved (float64 dense eigh), the HCB
    covariance matrix M is built (cell_40 Wick machinery), and the GEVP
    M v = lambda (S + ridge I) v is solved against the init_gram metric S.

    Aggregation (manuscript l.612): fractions (f_err/f_H/f_N/f_par/R/
    rel_S_err) are arithmetic MEANS; scale-like quantities (lam_min_pos,
    lam_max, kappa, r_med, cond_S, ratio_S_median) are MEDIANS; r_max /
    sum_qk_max are maxima.  eps_I is None when >50% of samples are undefined
    (P_perp w_true ~ 0, e.g. the const panel); per-sample eps_I keeps NaN.
    pred_energies optionally overrides the predicted single-particle
    energies (rows of length d, or m repeated like the randomenerg flow).
    """
    S = _panel_gram_S()
    w_N = get_w_N_pairing(basis)
    m_pairs = basis.d // 2

    keys = ('f_err', 'f_H', 'f_N', 'f_par', 'eps_I', 'dw_S_sq', 'ratio_S',
            'R', 'rel_S_err', 'r', 'lam_min_pos', 'lam_max', 'kappa',
            'n_discarded', 'n_null', 'sum_qk', 'cond_S', 'disc_err_weight')
    per = {k: [] for k in keys}
    all_lambdas, all_qk = [], []

    for geo in _iter_panel_samples(g_pred, g_true, n_samples,
                                   pred_energies=pred_energies,
                                   progress=progress, desc="drive_panel"):
        M = geo['M']
        w_eval_pred = geo['w_pred']
        w_eval_true = geo['w_true']

        f_H, f_N, f_err, _ = evaluate_subspace_projection(
            w_eval_pred, w_eval_true, w_N, S)

        lambdas_phys, q_k_pred, gevp_diag = compute_gevp_observables_diag(
            M, S, w_eval=w_eval_pred, lam_ridge=lam_ridge, cutoff=cutoff,
            lam_pos_threshold=lam_pos_threshold)
        all_lambdas.append(lambdas_phys)
        all_qk.append(q_k_pred)

        # Table tab:gevp_diag accumulations [export l.2313-2341]
        S_sym_d = gevp_diag['S_sym']
        V_phys_d = gevp_diag['V_phys']
        dw = w_eval_pred - w_eval_true
        dw_S_dw = float(dw @ S_sym_d @ dw)
        true_S_true = max(float(w_eval_true @ S_sym_d @ w_eval_true), 1e-16)
        c_dw = V_phys_d.T @ S_sym_d @ dw
        pos_mask = lambdas_phys > lam_pos_threshold
        R_val = float(np.sum(c_dw[pos_mask]**2) / max(dw_S_dw, 1e-300))

        # corrected background/disorder split (SIGN FIX -- referee item 4(f))
        w_H0_vec, w_bg_vec = build_background_vectors(
            geo['pair_energies_true'], m_pairs, g_mean=geo['g_mean_true'])
        f_par_s, eps_I_s = background_split_metrics(
            w_eval_pred, w_eval_true, w_H0_vec, w_bg_vec, S)

        per['f_err'].append(float(f_err))
        per['f_H'].append(float(f_H))
        per['f_N'].append(float(f_N))
        per['f_par'].append(float(f_par_s))
        per['eps_I'].append(float(eps_I_s))
        per['dw_S_sq'].append(dw_S_dw)
        per['ratio_S'].append(float(np.sqrt(gevp_diag['norm_eval_sq'] / true_S_true)))
        per['R'].append(R_val)
        per['rel_S_err'].append(dw_S_dw / true_S_true)
        per['r'].append(gevp_diag['r'])
        per['lam_min_pos'].append(gevp_diag['lam_min_pos'])
        per['lam_max'].append(gevp_diag['lam_max'])
        per['kappa'].append(gevp_diag['kappa'])
        per['n_discarded'].append(gevp_diag['n_discarded'])
        per['n_null'].append(int(np.sum(lambdas_phys < lam_pos_threshold)))
        per['sum_qk'].append(float(np.sum(q_k_pred)))
        per['cond_S'].append(gevp_diag['cond_S'])
        per['disc_err_weight'].append(discarded_error_weight(gevp_diag, dw))

    N_s = len(per['f_err'])
    if N_s == 0:
        raise RuntimeError("drive_panel: no sample completed the GEVP flow")

    # scree data (sorted ascending by lambda already; trim to common rank)
    min_len = min(len(l) for l in all_lambdas)
    lam_arr = np.array([l[:min_len] for l in all_lambdas], dtype=np.float64)
    qk_arr = np.array([o[:min_len] for o in all_qk], dtype=np.float64)
    lambdas_geo = np.exp(np.mean(np.log(np.maximum(lam_arr, 1e-16)), axis=0))
    q_k_mean = np.mean(qk_arr, axis=0)
    q_k_std = np.std(qk_arr, axis=0)

    eps_arr = np.asarray(per['eps_I'], dtype=np.float64)
    n_eps_undef = int(np.sum(~np.isfinite(eps_arr)))
    eps_I_agg = (None if n_eps_undef > 0.5 * N_s
                 else float(np.nanmean(eps_arr)))

    n_disc = np.asarray(per['n_discarded'], dtype=int)
    n_null = np.asarray(per['n_null'], dtype=int)

    panel = {
        'N_s': N_s,
        'f_err': float(np.mean(per['f_err'])),
        'f_H': float(np.mean(per['f_H'])),
        'f_N': float(np.mean(per['f_N'])),
        'f_par': float(np.mean(per['f_par'])),
        'eps_I': eps_I_agg,
        'eps_I_nan_count': n_eps_undef,
        'ratio_S_median': float(np.median(per['ratio_S'])),
        'ratio_S_mean': float(np.mean(per['ratio_S'])),
        'lam_min_pos': float(np.nanmedian(per['lam_min_pos'])),
        'lam_max': float(np.nanmedian(per['lam_max'])),
        'kappa': float(np.nanmedian(per['kappa'])),
        'r_max': float(np.nanmax(per['r'])),
        'r_med': float(np.nanmedian(per['r'])),
        'n_discarded': int(np.bincount(n_disc).argmax()),
        'n_discarded_range': [int(n_disc.min()), int(n_disc.max())],
        'cond_S': float(np.median(per['cond_S'])),
        'R': float(np.mean(per['R'])),
        'rel_S_err': float(np.mean(per['rel_S_err'])),
        'null_mode_count': int(np.bincount(n_null).argmax()),
        'null_mode_range': [int(n_null.min()), int(n_null.max())],
        'sum_qk_max': float(np.max(per['sum_qk'])),
        'disc_err_weight': float(np.mean(per['disc_err_weight'])),
        'disc_err_weight_max': float(np.max(per['disc_err_weight'])),
        'ridge': float(lam_ridge),
        'cutoff': float(cutoff),
        'lam_pos_threshold': float(lam_pos_threshold),
        'lambda_threshold_note': (
            f"lam_min_pos/kappa/R from the corrected pass "
            f"(positive-lambda threshold {lam_pos_threshold:g})"),
        'scree': {
            'lambda_geo_mean': lambdas_geo,
            'q_k_mean': q_k_mean,
            'q_k_std': q_k_std,
            'n_modes': int(min_len),
            'note': ("modes sorted ascending by lambda; lambda geometric "
                     "mean / q_k arithmetic mean+std over samples"),
        },
        'per_sample': per,
    }
    above3 = np.where(q_k_mean > 1e-3)[0]
    above5 = np.where(q_k_mean > 1e-5)[0]
    panel['k_last_q_gt_1e3'] = int(above3.max()) if above3.size else -1
    panel['k_last_q_gt_1e5'] = int(above5.max()) if above5.size else -1
    return panel


def sweep_stability(g_pred, g_true, ridges, cutoffs, n_samples=256,
                    lam_pos_threshold=1e-12, default_ridge=1e-9,
                    default_cutoff=1e-7, pred_energies=None, progress=True,
                    **_ignored):
    """One-at-a-time ridge/cutoff sweeps around the defaults (1e-9, 1e-7).

    Per sweep point reports {f_err, null_mode_count, lam_min_pos} (f_err
    arithmetic mean -- ridge/cutoff independent by construction, reported for
    the values_prev 'stability' shape; null_mode_count = per-sample mode;
    lam_min_pos = per-sample median) plus deltas vs the default point.
    Top-level 'f_err' / 'null_count' / 'lam_min_pos' are [min, max] ranges
    over all sweep points (values_prev.json `stability` shape).
    """
    S = _panel_gram_S()
    w_N = get_w_N_pairing(basis)

    # precompute the ridge/cutoff-independent per-sample geometry ONCE
    geos = list(_iter_panel_samples(g_pred, g_true, n_samples,
                                    pred_energies=pred_energies,
                                    progress=progress,
                                    desc="sweep_stability precompute"))
    if not geos:
        raise RuntimeError("sweep_stability: no sample completed")
    f_err_mean = float(np.mean([
        evaluate_subspace_projection(g['w_pred'], g['w_true'], w_N, S)[2]
        for g in geos]))

    def _point(r, c):
        lam_min_list, n_null_list = [], []
        for g in geos:
            lambdas_phys, _q, _d = compute_gevp_observables_diag(
                g['M'], S, w_eval=g['w_pred'], lam_ridge=r, cutoff=c,
                lam_pos_threshold=lam_pos_threshold)
            pos = lambdas_phys[lambdas_phys > lam_pos_threshold]
            lam_min_list.append(float(pos.min()) if pos.size else np.nan)
            n_null_list.append(int(np.sum(lambdas_phys < lam_pos_threshold)))
        return {
            'ridge': float(r),
            'cutoff': float(c),
            'f_err': f_err_mean,
            'null_mode_count': int(np.bincount(np.asarray(n_null_list)).argmax()),
            'lam_min_pos': float(np.nanmedian(lam_min_list)),
        }

    default_pt = _point(default_ridge, default_cutoff)

    def _with_deltas(pt, sweep):
        pt = dict(pt)
        pt['sweep'] = sweep
        pt['d_f_err'] = pt['f_err'] - default_pt['f_err']
        pt['d_null_mode_count'] = pt['null_mode_count'] - default_pt['null_mode_count']
        pt['d_lam_min_pos'] = pt['lam_min_pos'] - default_pt['lam_min_pos']
        return pt

    points = []
    for r in ridges:
        points.append(_with_deltas(_point(float(r), default_cutoff), 'ridge'))
    for c in cutoffs:
        points.append(_with_deltas(_point(default_ridge, float(c)), 'cutoff'))

    f_errs = [p['f_err'] for p in points]
    nulls = [p['null_mode_count'] for p in points]
    lmins = [p['lam_min_pos'] for p in points]
    return {
        # values_prev.json 'stability' shape: [min, max] over sweep points
        'f_err': [float(np.min(f_errs)), float(np.max(f_errs))],
        'null_count': [int(np.min(nulls)), int(np.max(nulls))],
        'lam_min_pos': [float(np.nanmin(lmins)), float(np.nanmax(lmins))],
        'defaults': default_pt,
        'points': points,
        'ridges': [float(r) for r in ridges],
        'cutoffs': [float(c) for c in cutoffs],
        'n_samples': len(geos),
        'lam_pos_threshold': float(lam_pos_threshold),
    }


def plot_scree_qk(panel_dict, out_prefix, title=None):
    """Scree figure in the q_k normalization (manuscript Eq. qk) from a
    drive_panel dict: red lambda curve (log, right axis), blue q_k bars
    (log, left axis), grey-shaded null manifold (lambda <= lam_pos_threshold)
    and the norm ratio ||w_pred||_S/||w_true||_S in the axes text box.
    Saves <out_prefix>.png (dpi=300) AND <out_prefix>.pdf; returns the .png
    path.  [layout from export plot_averaged_scree l.2158]"""
    scree = panel_dict.get('scree') or {}
    if 'lambda_geo_mean' not in scree or 'q_k_mean' not in scree:
        raise ValueError("plot_scree_qk: panel_dict has no 'scree' block "
                         "(run drive_panel first)")
    lambdas_phys = np.asarray(scree['lambda_geo_mean'], dtype=np.float64)
    O_k_mean = np.asarray(scree['q_k_mean'], dtype=np.float64)
    zero_tol = float(panel_dict.get('lam_pos_threshold', 1e-12))
    ratio_S = panel_dict.get('ratio_S_median')

    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    k_indices = np.arange(len(lambdas_phys))
    lambdas_safe = np.maximum(lambdas_phys, 1e-16)

    # Physical Spectrum (Line Plot)
    ax2.plot(k_indices, lambdas_safe, color='tab:red', lw=3.0, zorder=3,
             label=r'Geometric Mean Spectrum ($\langle \lambda_k \rangle$)')
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Quantum Variance ($\lambda_k$)', color='tab:red',
                   fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # ML Deposition (Bar Chart)
    ax1.bar(k_indices, O_k_mean, capsize=3, ecolor='black',
            color='tab:blue', alpha=0.6, width=1.0, zorder=2,
            label=r'Avg Spectral Weight ($\langle q_k \rangle$, Eq. $q_k$)')
    ax1.set_ylabel(r'Spectral Weight ($q_k$)', color='tab:blue',
                   fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Normal Mode Rank Index ($k$)', fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 2.0)
    ax1.set_xlim(-2, len(k_indices) + 2)

    # Exact Symmetry Manifold Shading (lambda <= lam_pos_threshold)
    n_null = int(np.sum(lambdas_phys <= zero_tol))
    if n_null > 0:
        ax1.axvspan(-0.5, n_null - 0.5, color='gray', alpha=0.15, zorder=1,
                    label=f'Exact Symmetries (~{n_null})')

    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    if title:
        ax1.set_title(title, fontweight='bold', pad=15)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right',
               framealpha=0.9, fontsize=12)

    if ratio_S is not None:
        box_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                         alpha=0.9, edgecolor='gray')
        ax1.text(0.03, 0.95,
                 r'$\|w_{\mathrm{pred}}\|_S/\|w_{\mathrm{true}}\|_S'
                 r' = %.4f$' % float(ratio_S),
                 transform=ax1.transAxes, fontsize=13, fontweight='bold',
                 verticalalignment='top', bbox=box_props, zorder=5)

    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=300)
    fig.savefig(out_prefix + '.pdf')
    plt.close(fig)
    return out_prefix + '.png'


def _s_orthonormalize(D, S_sym, drop_tol=1e-10):
    """S-orthonormal basis of span(columns of D) (modified Gram-Schmidt in
    the S inner product, two re-orthogonalization passes; near-dependent
    columns dropped).  Returns (M_dim, n_kept)."""
    Q = []
    for j in range(D.shape[1]):
        v = np.asarray(D[:, j], dtype=np.float64).copy()
        nrm0 = np.sqrt(max(float(v @ S_sym @ v), 0.0))
        for _pass in range(2):
            for q in Q:
                v = v - float(q @ S_sym @ v) * q
        nrm = np.sqrt(max(float(v @ S_sym @ v), 0.0))
        if nrm > drop_tol * max(nrm0, 1e-30):
            Q.append(v / nrm)
    if not Q:
        return np.zeros((D.shape[0], 0))
    return np.stack(Q, axis=1)


def null_mode_audit(g_pred, g_true, n_samples=256, lam_ridge=1e-9,
                    cutoff=1e-7, lam_pos_threshold=1e-12,
                    pred_energies=None, progress=True, **_ignored):
    """Const-panel null-mode audit (values_prev.json `null_audit_const`).

    Per sample: the numerically-resolved null manifold = retained GEVP modes
    with lambda < lam_pos_threshold (zero_tol of the scree grey band) is
    projected onto the S-orthonormalized symmetry dictionary
    {w_H0, w_N, w_true_const, P0..P9 diag pseudospins} (the seniority-basis
    counterpart of the cell_43 null-space check / cell_44
    get_pairing_symmetry_vectors flow: in the paired basis the local SU(2)
    generators reduce to the diagonal pseudospin occupations B_k^+ B_k).
    f_dict per null mode = ||P_dict v||_S^2 (coverage fraction).  Also
    reports the prediction's S-weight inside the null manifold
    (sum of q_k over null modes)."""
    S = _panel_gram_S()
    m_pairs = basis.d // 2
    dict_names = ['w_H0', 'w_N', 'w_true_const'] + \
                 [f'P{k}_diag' for k in range(m_pairs)]
    w_N_vec = get_w_N_pairing(basis)

    f_dict_all = []
    n_null_list = []
    pred_null_w = []
    proj_dict_rows = []   # cell_43-flow: ||V_null^T S d_hat||^2 per dict op

    for geo in _iter_panel_samples(g_pred, g_true, n_samples,
                                   pred_energies=pred_energies,
                                   progress=progress, desc="null_mode_audit"):
        lambdas_phys, q_k, gevp_diag = compute_gevp_observables_diag(
            geo['M'], S, w_eval=geo['w_pred'], lam_ridge=lam_ridge,
            cutoff=cutoff, lam_pos_threshold=lam_pos_threshold)
        S_sym = gevp_diag['S_sym']
        V_phys = gevp_diag['V_phys']

        null_mask = lambdas_phys < lam_pos_threshold
        n_null = int(null_mask.sum())
        n_null_list.append(n_null)
        # prediction's weight inside the null manifold (q_k normalization)
        pred_null_w.append(float(np.sum(q_k[null_mask])))
        if n_null == 0:
            continue
        V_null = V_phys[:, null_mask]      # S-orthonormal columns

        # symmetry dictionary in the flattened m^2 coordinate space
        w_H0_vec, _ = build_background_vectors(geo['pair_energies_true'],
                                               m_pairs)
        D_cols = [w_H0_vec, w_N_vec,
                  np.asarray(geo['w_true'], dtype=np.float64)]
        for k in range(m_pairs):
            e_k = np.zeros(m_pairs * m_pairs, dtype=np.float64)
            e_k[k * m_pairs + k] = 1.0     # B_k^+ B_k (diag pseudospin)
            D_cols.append(e_k)
        D = np.stack(D_cols, axis=1)
        Q = _s_orthonormalize(D, S_sym)

        # coverage fraction of each null mode inside the dictionary span
        proj = Q.T @ (S_sym @ V_null)
        f_dict = np.sum(proj**2, axis=0)
        f_dict_all.extend(float(x) for x in f_dict)

        # cell_43 flow (reverse direction): projection magnitude^2 of every
        # S-normalized dictionary operator onto the null manifold
        row = []
        for j in range(D.shape[1]):
            dj = D[:, j]
            nrm = np.sqrt(max(float(dj @ S_sym @ dj), 1e-16))
            cj = V_null.T @ (S_sym @ (dj / nrm))
            row.append(float(np.sum(cj**2)))
        proj_dict_rows.append(row)

    if not n_null_list:
        raise RuntimeError("null_mode_audit: no sample completed")

    n_null_arr = np.asarray(n_null_list, dtype=int)
    f_arr = np.asarray(f_dict_all, dtype=np.float64)
    n_worst = 20
    worst = sorted(f_arr.tolist())[:n_worst]
    count_mode = int(np.bincount(n_null_arr).argmax())

    audit = {
        'criterion': (
            f"retained GEVP modes with lambdas_phys < {lam_pos_threshold:g} "
            f"(zero_tol of plot_averaged_scree axvspan), ridge "
            f"{lam_ridge:g}, cutoff v'Sv > {cutoff:g}"),
        'count': count_mode,
        'null_mode_count': count_mode,
        'count_range': [int(n_null_arr.min()), int(n_null_arr.max())],
        'n_modes_total': int(f_arr.size),
        'f_dict_min': (float(f_arr.min()) if f_arr.size else None),
        'f_dict_med': (float(np.median(f_arr)) if f_arr.size else None),
        'count_above_0.99': int(np.sum(f_arr > 0.99)),
        'dictionary': dict_names,
        'n_worst_listed': n_worst,
        'worst_f_dict': worst,
        # cell_43-flow per-operator column (mean over samples), aligned with
        # 'dictionary'
        'proj_dict_onto_null_mean': (
            [float(x) for x in np.mean(proj_dict_rows, axis=0)]
            if proj_dict_rows else None),
        # the trained prediction's S-weight inside the null manifold
        'pred_null_weight_mean': float(np.mean(pred_null_w)),
        'pred_null_weight_median': float(np.median(pred_null_w)),
        'pred_null_weight_max': float(np.max(pred_null_w)),
        'pred_null_weight_per_sample': [float(x) for x in pred_null_w],
        'n_samples': int(n_null_arr.size),
        'ridge': float(lam_ridge),
        'cutoff': float(cutoff),
        'lam_pos_threshold': float(lam_pos_threshold),
    }
    return audit
