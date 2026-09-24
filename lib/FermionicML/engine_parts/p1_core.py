# ============================================================================
# p1_core.py -- config init (d=20 / d=12 eras), basis & RDM operator arrays,
# Hamiltonian builders, eigensolver custom_vjp, solver kernels, GGenerator,
# RDM trace helpers, NumpyLoader and the pmap fused dataset pipeline.
#
# Exec'd FIRST into the shared engine namespace (see campaign/engine.py).
# Later parts reference the globals set here (D_SP, basis, g_gen, ...) via
# late binding at CALL time -- exactly like the original notebook. Do not
# import other parts, do not import engine.
#
# Sources (verbatim transplants, edits marked with "[engine edit]"):
#   .claude/cells/cell_03.py  -> d=20 config & basis / RDM arrays (inside init_d20)
#   .claude/cells/cell_04.py  -> _ensure_dense, two_body_hamiltonian_sp,
#                                two_body_hamiltonian_dense, state_energy
#   .claude/cells/cell_05.py  -> eigensolve_and_build_rho (full custom_vjp)
#   .claude/cells/cell_06.py  -> pure_state, k_safe, thermal_state,
#                                compute_rdm_trace, solve_batch_kernel(+_sp)
#   .claude/cells/cell_11.py  -> GGenerator
#   .claude/cells/cell_12.py  -> compute_rho_m, _to_dense_jax
#   .claude/cells/cell_13.py  -> BETA/SCALE_FACTOR/M_PAIRS/U_ENERGY_SEED config
#                                (inside init_d20), NumpyLoader,
#                                check_existing_dataset, gen_dataset (pmap
#                                pipeline; the ray-based gen_dataset_sp is
#                                intentionally NOT transplanted)
#   campaign/reference/j31_lean.py -> d=12 era configuration (inside init_d12)
#
# Edits required by the spec:
#   * init_d20 determinism patch: pair_jitter uses np.linspace instead of
#     np.random.uniform (see comment at the site).
#   * NumpyLoader.__init__ accepts a str OR a list of directories.
#   * gen_dataset signature default train_batch_size=None (resolved to
#     GPU_BATCH_SIZE at call time; in the notebook the default was evaluated
#     against a pre-existing global at def time).
#   * jax.lib.xla_bridge.get_backend().platform -> jax.default_backend()
#     (cosmetic print only; xla_bridge was removed in newer JAX).
# ============================================================================

import os
import sys
import math
import glob
import shutil
import concurrent.futures
from functools import partial
from typing import Union, Optional, Literal

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
import sparse
import jax
import jax.numpy as jnp
from tqdm import tqdm
import fermionic_mbody as fmb


# Type definitions for validation  # [from cell_13]
ValidHType = Literal['const', 'gaussian', 'vect', 'random', 'randomenerg', 'blockgen', 'blockgensimp']
ValidStateType = Literal['thermal', 'gs']
ValidInputType = Literal['rho2', 'rho1', 'rho1+rho2', 'rho2kkbar', 'rho2block']


# ---------------------------------------------------------------------------
# Hamiltonian builders  [from cell_04]
# ---------------------------------------------------------------------------

def _ensure_dense(arr):  # [from cell_04]
        if hasattr(arr, 'todense'):
            return arr.todense()
        return np.asarray(arr)

def two_body_hamiltonian_sp(basis: fmb.FixedBasis,
                            energy_seed: Union[np.ndarray],
                            int_mat: Union[np.ndarray],
                            rho_1_arrays: sp.coo_array,
                            rho_2_arrays: sp.coo_array,
                            h_type: Optional[str] = None) -> sp.coo_array:
    """
    Constructs the many-body Hamiltonian H = H0 - HI in the N-body basis.

    H0(b) = Sum_i E_i(b) * M(c_i^dag c_i)
    HI(b) = Sum_{I,J} V_{I,J}(b) * M(C_I^dag C_J)
    """
    d = basis.d
    DN = basis.size
    is_paired_optimization = (basis.pairs and d % 2 == 0)

    # --- H0 (One-body energy) ---
    # Extract number operators M(c_i^dag c_i). Shape (d, DN, DN)
    if is_paired_optimization:
        # H0 = Sum_k E_k^Pair * N_k^Pair
        m_pairs = d // 2
        B = energy_seed.shape[0]
        pair_energies = energy_seed.reshape(B, m_pairs, 2).sum(axis=-1)
        H0_energy = sp.coo_array(pair_energies)
        indices = np.arange(m_pairs, dtype=int)

    # Extract standard number operators M(c_i^dag c_i)
    else:
        d_r1 = rho_1_arrays.shape[0]
        indices = np.arange(d_r1, dtype=int)
        pair_energies = energy_seed

    rho_1_arrays = _ensure_dense(rho_1_arrays)
    rho_1_diag = rho_1_arrays[indices, indices]

    # Calculate H0
    H0_energy_sparse = sp.coo_array(pair_energies) if not isinstance(pair_energies, sp.coo_array) else pair_energies
    if H0_energy_sparse.ndim == 2:
        # Einsum: 'bi,irc->brc' (b=batch, i=index (pair or sp), r,c=many-body indices)
        h0_arr = H0_energy_sparse.tensordot(rho_1_diag, axes=([1], [0]))
    elif H0_energy_sparse.ndim == 1:
        h0_arr = H0_energy_sparse.tensordot(rho_1_diag, axes=([0], [0]))
    else:
        raise ValueError(f"Unexpected dimensions for H0 energy input: {H0_energy_sparse.ndim}")

    # --- HI (Two-body interaction) ---

    # Ensure int_mat is sparse
    int_mat_sparse = sp.coo_array(np.asarray(int_mat))
    # Calculate H_I. We contract V[i, j] with R[j, i, r, c].
    if int_mat_sparse.ndim == 3:
        # Replaced einsum: 'bij,jirc->brc'
        # i = axis 1 (int_mat) / axis 1 (rho_2), j = axis 2 (int_mat) / axis 0 (rho_2)
        hi_arr = int_mat_sparse.tensordot(rho_2_arrays, axes=([1, 2], [1, 0]))
    elif int_mat_sparse.ndim == 2:
        # Replaced einsum: 'ij,jirc->rc'
        hi_arr = int_mat_sparse.tensordot(rho_2_arrays, axes=([0, 1], [1, 0]))
    else:
        raise ValueError(f"Unexpected dimensions for int_mat")

    return h0_arr - hi_arr

@partial(jax.jit, static_argnames=['precision'])
def two_body_hamiltonian_dense(  # [from cell_04]
                            energy_params: jnp.ndarray,
                            interaction_params: jnp.ndarray,
                            rho_1_diag: jnp.ndarray,
                            rho_2_tensor: jnp.ndarray,
                            precision=None,
                            ) -> jnp.ndarray:
    """
    Constructs a batch of Hamiltonians H = H0 - HI

    [V2] precision: jax.lax.Precision for the H0 dot and the HI contraction
    (None = the backend default, the published numerics; HIGHEST = exact f32,
    the production-run v2 generation convention).
    """
    # --- 1. H0 Construction (Optimized) ---
    # H0 is diagonal in the Fock basis. We compute the diagonal vector directly.
    # (Batch, D_SP) @ (D_SP, N) -> (Batch, N)
    if energy_params.shape[-1] == rho_1_diag.shape[0] * 2:
        energy_params = energy_params.reshape(energy_params.shape[0], -1, 2).sum(axis=-1)

    h0_diag_vals = jnp.dot(energy_params, rho_1_diag, precision=precision)

    # --- 2. HI Construction (Tensor Contraction) ---
    # Case A: Matrix-style params (B, I, J). We contract V_ij * Op_ji
    if interaction_params.ndim == 3:
        # Note: GGenerator typically outputs (I, J).
        # We assume Op is (J, I, N, N) to match original sparse einsum 'bij,jirc'
        HI = jnp.einsum('bij,jirc->brc', interaction_params, rho_2_tensor,
                        precision=precision)

    # Case B: Vector-style params (B, K). Contract V_k * Op_k
    else:
        HI = jnp.tensordot(interaction_params, rho_2_tensor, axes=(1, 0),
                           precision=precision)

    # --- 3. Combine: H = -HI + H0 ---
    H_total = -HI

    N = h0_diag_vals.shape[-1]
    indices = jnp.arange(N)

    # 1. RESTORE H_0
    H_total = H_total.at[:, indices, indices].add(h0_diag_vals)

    # 2. Enforce Hermiticity to clean up numerical noise
    H_total = 0.5 * (H_total + jnp.swapaxes(H_total.conj(), 1, 2))

    return H_total

def state_energy(state: Union[np.ndarray, sparse.COO], h_arr: Union[np.ndarray, sparse.COO]) -> np.ndarray:  # [from cell_04]
    """
    Calculates the energy E = Tr(rho * H) for a batch of states and Hamiltonians.
    """
    if isinstance(state, np.ndarray) and state.ndim == 2 and isinstance(h_arr, np.ndarray):
            # <psi|H|psi> = sum(psi.conj * (H @ psi))
            # H @ psi: (B, N, N) @ (B, N, 1) -> (B, N)
            h_psi = np.einsum('bij,bj->bi', h_arr, state)
            return np.sum(state.conj() * h_psi, axis=1).real

    # Fallback / Density Matrices
    energy = sparse.einsum('bij,bji->b', state, h_arr)
    if hasattr(energy, 'todense'):
        return energy.todense()
    return np.asarray(energy)


# ---------------------------------------------------------------------------
# TPU-OPTIMIZED CUSTOM VJP (PURE FLOAT32)  [from cell_05]
# ---------------------------------------------------------------------------

@jax.custom_vjp
def eigensolve_and_build_rho(H_batch, beta):
    """
    Computes the thermal density matrix rho = e^{-beta H} / Z and expected energy.
    Runs natively in float32 for maximum TPU speed. NO float64 emulation!
    """
    vals, vecs = jnp.linalg.eigh(H_batch)

    # jax.nn.softmax natively handles max-shifting for numerical stability
    probs = jax.nn.softmax(-beta * vals, axis=-1)

    state_rho = jnp.matmul(vecs * probs[..., None, :], jnp.swapaxes(vecs.conj(), -1, -2))
    energy_out = jnp.sum(probs * vals, axis=-1)

    return state_rho, energy_out

# Define the forward pass
def eigensolve_and_build_rho_fwd(H_batch, beta):
    N = H_batch.shape[-1]
    sym_breaker = 1e-5 * ((jnp.arange(N, dtype=jnp.float32) % 2) * 2.0 - 1.0)
    indices = jnp.arange(N)
    H_safe = H_batch.at[..., indices, indices].add(sym_breaker)

    vals, vecs = jnp.linalg.eigh(H_safe)

    probs = jax.nn.softmax(-beta * vals, axis=-1)

    state_rho = jnp.matmul(vecs * probs[..., None, :], jnp.swapaxes(vecs.conj(), -1, -2))
    energy_out = jnp.sum(probs * vals, axis=-1)

    res = (vals, vecs, probs, energy_out, beta)
    return (state_rho, energy_out), res

# Define the backward pass
def eigensolve_and_build_rho_bwd(res, g):
    vals, vecs, probs, energy_out, beta = res
    g_state_rho, g_energy = g

    N = vecs.shape[-1]

    # Construct C matrix
    D_diff = vals[..., :, None] - vals[..., None, :]
    P_diff = probs[..., :, None] - probs[..., None, :]

    # Float32-friendly masking threshold to catch degeneracies
    mask = jnp.abs(D_diff) > 1e-5
    safe_D_diff = jnp.where(mask, D_diff, 1.0)

    # L'Hôpital's analytical limit safely bridges degenerate eigenvalue differences
    limit_C = -beta * (probs[..., :, None] + probs[..., None, :]) / 2.0
    C = jnp.where(mask, P_diff / safe_D_diff, limit_C)

    # Build G = V^dagger g_rho V
    vecs_adj = jnp.swapaxes(vecs.conj(), -1, -2)
    G = jnp.matmul(vecs_adj, jnp.matmul(g_state_rho, vecs))

    # Compute Trace(G * P)
    G_diag = jnp.diagonal(G, axis1=-2, axis2=-1)
    Tr_GP = jnp.sum(G_diag * probs, axis=-1)

    # Construct Hamiltonian gradient in eigenbasis
    H_bar_V = G * C

    # Diagonal corrections
    diag_correction = beta * probs * Tr_GP[..., None]
    diag_correction = diag_correction + g_energy[..., None] * probs * (1.0 - beta * (vals - energy_out[..., None]))

    eye = jnp.eye(N, dtype=H_bar_V.dtype)
    H_bar_V = H_bar_V + diag_correction[..., None] * eye

    # Rotate back to standard operator basis
    grad_H = jnp.matmul(vecs, jnp.matmul(H_bar_V, vecs_adj))

    # Enforce strict Hermiticity to scrub floating point noise
    grad_H = 0.5 * (grad_H + jnp.swapaxes(grad_H.conj(), -1, -2))

    return (grad_H, None)

eigensolve_and_build_rho.defvjp(eigensolve_and_build_rho_fwd, eigensolve_and_build_rho_bwd)


# ---------------------------------------------------------------------------
# Solver kernels  [from cell_06]
# ---------------------------------------------------------------------------

def _to_dense_array(H_sp):  # [from cell_06]
    if scipy.sparse.issparse(H_sp):
        return H_sp.toarray()
    elif hasattr(H_sp, 'todense'):
        return H_sp.todense()
    return np.asarray(H_sp)

def _prepare_hamiltonian_for_eigsh(H_i: Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix]) -> Union[np.ndarray, scipy.sparse.spmatrix]:  # [from cell_06]
    """Helper to convert Hamiltonian to a format suitable for scipy eigensolvers"""
    if isinstance(H_i, sparse.COO):
        return H_i.to_scipy_sparse().tocsr()
    elif scipy.sparse.issparse(H_i):
        return H_i.tocsr() if hasattr(H_i, 'tocsr') else H_i
    else:
        return np.asarray(H_i)


def pure_state(h: Union[np.ndarray, sp.coo_array]):  # [from cell_06]
    """
    Calculates the ground state density matrix rho = |psi_0><psi_0| for a batch of Hamiltonians.
    """
    B, DN, _ = h.shape
    mat = np.zeros((B, DN, DN), dtype=np.complex128)
    energies = np.zeros(B, dtype=np.float64)

    if isinstance(h, np.ndarray) or (hasattr(h, 'todense') and DN < 512):
            if not isinstance(h, np.ndarray): h = h.todense()
            vals, vecs = np.linalg.eigh(h)
            fund = vecs[:, :, 0]
            mat = np.einsum('bi,bj->bij', fund, fund.conj())
            return vals[:, 0], mat.real

    for i in range(B):
        H_i_sp = _prepare_hamiltonian_for_eigsh(h[i])
        e, v = scipy.sparse.linalg.eigsh(H_i_sp, k=1, which='SA', tol=1e-8)
        fund = v[:, 0]
        mat[i, :, :] = np.outer(fund, fund.conj())
        energies[i] = e[0]

    return energies, mat.real

def k_safe(N: int, beta: float, eps: float = 1e-8, eta: float = 4.0, Delta_eff: float = 1.0) -> int:  # [from cell_06]
    """
    Heuristic calculation for the number of eigenstates required for a thermal state approximation.
    """
    if N <= 1:
        return 1

    # Calculation based on Boltzmann weight decay
    val = eta * (1.0 / (beta * max(Delta_eff, 1e-12))) * math.log(max(N/eps, 1.0))

    # Ensure the result is within valid bounds [1, N-1]
    return min(N - 1, max(1, math.ceil(val)))


def thermal_state(beta: float, h: Union[np.ndarray, sp.coo_array]):  # [from cell_06]
    """
    Calculates the thermal density matrix rho = exp(-beta*H)/Z for a batch of Hamiltonians
    """
    B, N, _ = h.shape
    mat = np.zeros((B, N, N), dtype=np.complex128)

    k0 = k_safe(N, beta)
    use_dense = (k0 > 0.25 * N)
    energies = np.zeros(B, dtype=np.float64)

    for i in range(B):
        H_i_scipy = _prepare_hamiltonian_for_eigsh(h[i])

        try:
            if use_dense:
                H_i_dense = _to_dense_array(H_i_scipy)
                E, V = np.linalg.eigh(H_i_dense)
            else:
                E, V = scipy.sparse.linalg.eigsh(H_i_scipy, k=k0, which="SA", tol=1e-8, return_eigenvectors=True)

        except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as e:
             print(f"Warning: ARPACK failed (k={k0}) for index {i}. Falling back to dense.")
             try:
                 H_i_dense = _to_dense_array(H_i_scipy)
                 E, V = np.linalg.eigh(H_i_dense)
             except Exception as e_dense:
                 print(f"Dense fallback failed for index {i}: {e_dense}")
                 mat[i, :, :] = np.nan
                 continue

        # Calculate weights
        E_shifted = E - E.min()
        weights = np.exp(-beta * E_shifted)
        Z = weights.sum()

        if Z < 1e-12:
            print(f"Warning: Partition function Z near zero for index {i}.")
            mat[i, :, :] = np.nan
            continue

        weights /= Z
        energies[i] = np.sum(weights * E)

        # Construct the density matrix: rho = V @ diag(weights) @ V.H
        mat[i, :, :] = (V * weights[None, :]) @ V.conj().T

    return energies, mat.real

def compute_rdm_trace(state, ops, precision=None):  # [from cell_06]
    """
    RDM = Tr(rho * Op) -> Einsum 'bij, kij -> bk'

    [V2] precision: jax.lax.Precision of the pair-block contraction (None =
    the backend default, the published numerics).
    """

    # Flatten spatial dims: (M, M, N, N) -> (M*M, N, N)
    orig_shape = ops.shape[:-2]
    ops_flat = ops.reshape(-1, ops.shape[-2], ops.shape[-1])

    # Compute trace: sum(rho_nm * op_mn) -> einsum 'bnm, knm'
    res = jnp.einsum('bnm,kmn->bk', state, ops_flat, precision=precision)
    return res.reshape((state.shape[0],) + orig_shape + (1,))

@partial(jax.jit, static_argnames=['is_thermal', 'precision'])
def solve_batch_kernel(  # [from cell_06]
    e_params, i_params, beta, is_thermal,
    rho_1_diag, rho_2_inter, rho_target, rho_1_full, precision=None):
    """
    Performs the full features extraction on GPU
    Params -> Hamiltonian -> Eigensolve -> State -> RDM Features

    [V2] precision=None reproduces the published kernel (custom-VJP state
    build, default-precision contractions); a jax.lax.Precision value applies
    to the Hamiltonian build, the state build and BOTH feature contractions
    (the state is then built inline: eigh -> softmax -> matmul at that
    precision; the custom-VJP path is only needed by the rdm loss).
    """

    # 1. Build Hamiltonian Batch
    H_batch = two_body_hamiltonian_dense(e_params, i_params, rho_1_diag, rho_2_inter,
                                         precision=precision)

    effective_beta = beta if is_thermal else 100.0
    if precision is None:
        state_rho, energy_out = eigensolve_and_build_rho(H_batch, effective_beta)
    else:
        vals, vecs = jnp.linalg.eigh(H_batch)
        probs = jax.nn.softmax(-effective_beta * vals, axis=-1)
        state_rho = jnp.matmul(vecs * probs[..., None, :],
                               jnp.swapaxes(vecs.conj(), -1, -2), precision=precision)
        energy_out = jnp.sum(probs * vals, axis=-1)

    # 4. Compute Features (RDMs)
    f1_out = compute_rdm_trace(state_rho, rho_1_full, precision=precision)

    # Feature 2: Target Rho (kkbar or block)
    f2_out = compute_rdm_trace(state_rho, rho_target, precision=precision)

    return f1_out.real, f2_out.real, energy_out.real

def solve_batch_kernel_sp(basis, e_params, i_params, beta, is_thermal, rho_1_arrays, rho_2_inter, rho_target):  # [from cell_06]

    H_batch = two_body_hamiltonian_sp(basis, e_params, i_params, rho_1_arrays, rho_2_inter)

    if is_thermal:
        energy_out, state = thermal_state(beta, H_batch)
    else:
        energy_out, state = pure_state(H_batch)

    f1_out = fmb.rho_m(state, rho_1_arrays)
    f2_out = fmb.rho_m(state, rho_target)

    return f1_out.real, f2_out.real, energy_out.real


# ---------------------------------------------------------------------------
# GGenerator  [from cell_11]
# ---------------------------------------------------------------------------

class GGenerator:
    """
    Generates and reconstructs parameterized Hamiltonian matrices
    """
    def __init__(self, basis, h_type: str, batch_size: int, g_init=0.0, g_stop=1.0, *, vrepeat=2, justenerg=False):
        # Handle basis object or integer input
        self.d = basis.d if hasattr(basis, 'd') else basis
        self.m = self.d // 2

        self.h_type = h_type
        self.batch_size = int(batch_size)
        self.g_init = float(g_init)
        self.g_stop = float(g_stop)
        self.vrepeat = int(vrepeat)
        self.justenerg = bool(justenerg)

        # Precompute sizes
        self.opsize = self._get_opsize()
        self._initialize_indices()

    def _get_opsize(self):
        if self.h_type in ['const', 'random', 'vect', 'gaussian', 'blockgensimp', 'randomenerg']:
            return self.m
        elif self.h_type == 'blockgen':
            return self.m ** 2
        raise ValueError(f"Unsupported h_type: {self.h_type}")

    def label_size(self):
        m = self.m
        if self.h_type == 'const': return 1
        elif self.h_type == 'random': return (m * (m + 1)) // 2
        elif self.h_type == 'vect': return math.ceil(m / self.vrepeat) - 1
        elif self.h_type == 'gaussian': return 2
        elif self.h_type == 'blockgen': return (m**4 + m**2) // 2
        elif self.h_type == 'blockgensimp': return (m**2 + m) // 2
        elif self.h_type == 'randomenerg':
            le = m
            li = (m**2 + m) // 2
            return le if self.justenerg else (le + li)
        return 0

    def _initialize_indices(self):
        # Precompute indices using NumPy
        # k=1 (strict upper triangle) for random, k=0 (includes diagonal) for others
        self._k = 0 if self.h_type in ['random', 'blockgensimp', 'randomenerg'] else 0

        r, c = np.triu_indices(self.opsize, k=self._k)
        self._triu_r = jnp.array(r)
        self._triu_c = jnp.array(c)

        if self.h_type == 'vect':
            idx_np = np.abs(np.arange(self.opsize)[:, None] - np.arange(self.opsize)[None, :])
            self._vect_idx = jnp.array(idx_np)

        if self.h_type == 'gaussian':
            i = np.arange(self.opsize, dtype=np.float32)
            self._gauss_diff2 = jnp.array((i[:, None] - i[None, :]) ** 2)

        if self.h_type == 'random':
            self._triu_r = jnp.array(r)
            self._triu_c = jnp.array(c)

    @partial(jax.jit, static_argnums=(0,))
    def _symmetric(self, h_labels):
        """Reconstructs symmetric matrices from flattened upper triangular values."""
        B = h_labels.shape[0]
        m = self.opsize

        # Initialize zero batch
        out = jnp.zeros((B, m, m), dtype=h_labels.dtype)

        # Fill upper triangle: out[:, r, c] = labels
        out = out.at[:, self._triu_r, self._triu_c].set(h_labels)

        # Symmetrize: M + M.T
        out_t = jnp.swapaxes(out, 1, 2)
        out = out + out_t

        if self._k == 0:
            diag_idx = jnp.arange(m)
            diags = out[:, diag_idx, diag_idx]
            out = out.at[:, diag_idx, diag_idx].set(diags * 0.5)

        return out

    @partial(jax.jit, static_argnums=(0,))
    def reconstruct(self, h_labels):
        """
        Reconstruct Hamiltonian matrices from labels.
        """
        B = h_labels.shape[0]

        if self.h_type == 'const':
            val = h_labels.reshape(B, 1, 1)
            return jnp.tile(val, (1, self.opsize, self.opsize))

        elif self.h_type in ['random', 'blockgen']:
            return self._symmetric(h_labels)

        elif self.h_type == 'vect':
            zeros = jnp.zeros((B, 1))
            # Pad, Repeat, Slice
            expanded = jnp.concatenate([zeros, h_labels], axis=1)
            expanded_rep = jnp.repeat(expanded, self.vrepeat, axis=1)
            vals = expanded_rep[:, :self.opsize]
            # Gather using precomputed distance indices
            return vals[:, self._vect_idx]

        elif self.h_type == 'gaussian':
            G = h_labels[:, 0]
            sigma_sq = jnp.maximum(jnp.abs(h_labels[:, 1]), 1e-4)
            exponent = -self._gauss_diff2[None, :, :] / (2.0 * sigma_sq[:, None, None])
            return G[:, None, None] * jnp.exp(exponent)

        elif self.h_type == 'blockgensimp':
            V = self._symmetric(h_labels)
            V_flat = V.reshape(B, -1)
            # Batched outer product
            return jnp.einsum('bi,bj->bij', V_flat, V_flat)

        elif self.h_type == 'randomenerg':
            energ = h_labels[:, :self.m]
            if self.justenerg:
                return energ

            int_labels = h_labels[:, self.m:]
            V = self._symmetric(int_labels)
            V_flat = V.reshape(B, -1)
            mat = jnp.einsum('bi,bj->bij', V_flat, V_flat)
            return energ, mat

        return jnp.zeros((B, self.opsize, self.opsize))

    @partial(jax.jit, static_argnums=(0,))
    def generate(self, key):
        """Helper for dataset generation (Pure JAX/TPU execution)"""
        L = self.label_size()
        B = self.batch_size

        # Split key for deterministic, stateless RNG
        key_g, key_e = jax.random.split(key)

        # Use JAX native random uniform
        h_labels = jax.random.uniform(
            key_g, shape=(B, L),
            minval=self.g_init, maxval=self.g_stop,
            dtype=jnp.float32
        )

        if self.h_type == 'vect':
            # jnp.sort sorts ascending, slicing [::-1] reverses it to descending
            h_labels = jnp.sort(h_labels, axis=1)[:, ::-1]

        elif self.h_type == 'random':
            raw_labels = jax.random.uniform(
                key_g, shape=(B, L),
                minval=self.g_init, maxval=self.g_stop,
                dtype=jnp.float32
            )

            m = self.opsize
            out = jnp.zeros((B, m, m), dtype=jnp.float32)

            r_full, c_full = jnp.triu_indices(m)
            out = out.at[:, r_full, c_full].set(raw_labels)

            diag_idx = jnp.arange(m)
            diags = out[:, diag_idx, diag_idx]
            target_mean = (self.g_init + self.g_stop) / 2.0
            shift = target_mean - jnp.mean(diags, axis=1, keepdims=True)
            out = out.at[:, diag_idx, diag_idx].set(diags + shift)

            h_labels = out[:, self._triu_r, self._triu_c]
            return L, h_labels

        elif self.h_type == 'randomenerg':
            ene = jax.random.uniform(
                key_e, shape=(B, self.m),
                minval=0.1, maxval=10.0,
                dtype=jnp.float32
            )
            ene -= jnp.mean(ene, axis=1, keepdims=True)
            ene = jnp.sort(ene, axis=1)[:, ::-1]

            if self.justenerg:
                h_labels = ene
            else:
                # JAX requires .at[...].set(...) for array mutations (no in-place assignment)
                h_labels = h_labels.at[:, :self.m].set(ene)

        elif self.h_type == 'blockgensimp':
             V_temp = self._symmetric(h_labels)
             flip = V_temp.sum(axis=(1, 2)) < 0
             h_labels = jnp.where(flip[:, None], -h_labels, h_labels)

        return L, h_labels


# ---------------------------------------------------------------------------
# Auxiliary RDM helpers  [from cell_12]
# ---------------------------------------------------------------------------

def compute_rho_m(state_batch, op_tensor, batch_size):  # [from cell_12]
        # Fallback for sparse ops
        if not isinstance(op_tensor, np.ndarray):
            res = fmb.rho_m(state_batch, op_tensor)
            if hasattr(res, 'todense'): res = res.todense()
            return res[..., np.newaxis].astype(np.float32)

        # Dense path
        # Flatten operators: (..., N, N) -> (K, N, N)
        op_shape = op_tensor.shape
        D_N = op_shape[-1]
        ops_flat = op_tensor.reshape(-1, D_N, D_N)

        if state_batch.ndim == 2: # State Vectors (B, N)
            # RDM[b, k] = <psi_b | O_k | psi_b>
            # Einstein sum: state[b, i]* . ops[k, i, j] . state[b, j]
            # (B, N) conj . (K, N, N) . (B, N) -> (B, K)
            rdm_flat = np.einsum('bi,kij,bj->bk', state_batch.conj(), ops_flat, state_batch)

        elif state_batch.ndim == 3: # Density Matrices (B, N, N)
            # Tr(rho * O) = sum(rho[b, j, i] * ops[k, i, j])
            rdm_flat = np.einsum('bji,kij->bk', state_batch, ops_flat)

        return rdm_flat.reshape((batch_size,) + op_shape[:-2] + (1,)).astype(np.float32)

def _to_dense_jax(arr):  # [from cell_12]
    """
    Helper to convert sparse/numpy arrays to dense JAX arrays on GPU.
    """
    if arr is None: return None
    if hasattr(arr, 'todense'):
        return jnp.array(arr.todense())
    return jnp.array(arr)


# ---------------------------------------------------------------------------
# Dataset loader & generation pipeline  [from cell_13]
# ---------------------------------------------------------------------------

class NumpyLoader:
    """Iterates over .npz shard files, slicing them into exact batch_size chunks."""
    def __init__(self, data_dir, batch_size, shuffle=True):
        # [engine edit] data_dir may be a single directory (str) or a list of
        # directories; the shard file lists are concatenated.
        dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.files = sorted(
            f for d in dirs for f in glob.glob(os.path.join(d, "shard_*.npz")))
        self.batch_size = batch_size
        self.shuffle = shuffle
        if not self.files: print(f"Warning: No files found in {data_dir}")

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.files)
        for f in self.files:
            try:
                with np.load(f) as d:
                    Y = d['labels']
                    E = d['energy'] if 'energy' in d else None
                    if 'features' in d: X = d['features']; is_tuple = False
                    else:
                        keys = sorted([k for k in d.files if k.startswith('feat_')])
                        X = [d[k] for k in keys]; is_tuple = True

                    # Slicing loop: Breaks large files into small batches
                    N = len(Y)
                    indices = np.arange(N)
                    if self.shuffle: np.random.shuffle(indices)

                    for start in range(0, N, self.batch_size):
                        end = min(start + self.batch_size, N)
                        idx = indices[start:end]
                        if len(idx) < self.batch_size: continue # Skip partial batches

                        by = Y[idx]
                        be = E[idx] if E is not None else None
                        bx = tuple(x[idx] for x in X) if is_tuple else X[idx]
                        if not is_tuple and bx.ndim == 3: bx = bx[..., None]
                        yield bx, be, by
            except Exception as e: print(f"Error loading {f}: {e}")

def check_existing_dataset(cache_path, num_samples, batch_size):  # [from cell_13]
    """Regenerates only if dataset is missing or too small."""
    if not os.path.exists(cache_path): return False
    files = glob.glob(os.path.join(cache_path, "shard_*.npz"))
    if not files: return False

    try:
        with np.load(files[0]) as d: shard_size = len(d['labels'])
        total_est = len(files) * shard_size

        if total_est >= num_samples * 0.95:
            print(f"Dataset valid (~{total_est} samples). Loading from disk.")
            return True
        else:
            print(f"Dataset incomplete ({total_est}/{num_samples}). Regenerating...")
            return False
    except: return False


def gen_dataset(h_type, g_init, g_stop, state_type, input_type,
                include_energy, beta, num_samples=100000,
                cache_path="./dataset", batch_size=4096, train_batch_size=None,
                seed=42):  # [from cell_13]
    # [engine edit] notebook default was train_batch_size=GPU_BATCH_SIZE,
    # evaluated at def time against a pre-existing global; here GPU_BATCH_SIZE
    # is set by init_d20/init_d12, so resolve it lazily. (It is unused below,
    # kept for signature compatibility.)
    if train_batch_size is None:
        train_batch_size = GPU_BATCH_SIZE

    if check_existing_dataset(cache_path, num_samples, batch_size):
        return NumpyLoader(cache_path, batch_size)

    # 2. Cleanup & Setup
    if os.path.exists(cache_path): shutil.rmtree(cache_path)
    # This line previously failed because cache_path was an int. Now it's safe.
    os.makedirs(cache_path, exist_ok=True)

    # --- 0. Universal Device Detection (TPU & Multi-GPU) ---
    # jax.local_devices() automatically finds 8 TPUs or N GPUs
    devices = jax.local_devices()
    n_devices = len(devices)
    # [engine edit] jax.lib.xla_bridge.get_backend() was removed in newer JAX;
    # jax.default_backend() is the stable equivalent (print-only value).
    platform = jax.default_backend().upper()

    print(f"Hardware: {n_devices} x {platform} devices detected.")

    if batch_size % n_devices != 0:
        # Auto-adjust batch size to divide evenly
        new_bs = ((batch_size // n_devices) + 1) * n_devices
        print(f"Adjusted batch size {batch_size} -> {new_bs} to fit devices.")
        batch_size = new_bs

    device_batch_size = batch_size // n_devices

    # --- Helper: Replicate Constants to Devices ---
    # This copies the constant Rho matrices to Accelerator Memory ONCE.
    def replicate(arr):
        if hasattr(arr, 'todense'): arr = arr.todense()
        arr = np.asarray(arr)
        # Cast to float32/complex64 for performance
        if np.iscomplexobj(arr): arr = arr.astype(np.complex64)
        else: arr = arr.astype(np.float32)
        # Shard the same copy to all devices efficiently
        return jax.device_put_sharded([jnp.array(arr)] * n_devices, devices)

    print("Moving static operators to accelerator memory...")

    # 1. Rho 1
    rho_1_np = _ensure_dense(rho_1_arrays)
    if rho_1_np.ndim == 4: rho_1_diag_np = np.einsum('kknn->kn', rho_1_np)
    else: rho_1_diag_np = np.diagonal(rho_1_np, axis1=1, axis2=2)

    p_rho_1_full = replicate(rho_1_np)
    p_rho_1_diag = replicate(rho_1_diag_np)

    # 2. Rho 2 & Target
    if h_type in ['blockgen', 'blockgensimp', 'randomenerg']:
        inter_np = _ensure_dense(rho_2_block_arrays)
    else:
        target = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
        inter_np = _ensure_dense(target)

    if 'rho2block' in input_type: target_np = _ensure_dense(rho_2_block_arrays)
    elif 'rho2kkbar' in input_type: target_np = _ensure_dense(rho_2_kkbar_arrays)
    else: target_np = inter_np

    p_rho_2_inter = replicate(inter_np)
    p_rho_target = replicate(target_np)

    # --- 2. Initialize Generator Logic ---
    # We initialize the generator logic. We will use this INSIDE the pmap.
    # Note: We pass 'device_batch_size' so inner dimensions match the split batch.
    g_gen = GGenerator(basis, h_type, device_batch_size, g_init=g_init, g_stop=g_stop)

    # Pre-calculate base energies (Batch, D)
    base_energies_np = U_ENERGY_SEED[0]

    # Replicate base energies to devices: (Devices, Device_Batch, D)
    p_base_energies = replicate(np.tile(base_energies_np, (device_batch_size, 1)))

    # --- 3. Define Fused Kernel (Reconstruct + Solve) ---
    # CRITICAL OPTIMIZATION: We move reconstruction to the TPU.
    # Input: Tiny Labels (float32) -> Output: Huge Features (float32)
    is_thermal = (state_type == 'thermal')

    def fused_step_fn(key_shard, base_energies_shard,
                      rho_1_d, rho_2_i, rho_t, rho_1_f):

        # Because `g_gen` is configured with `device_batch_size`, it generates the shard natively.
        _, labels_shard = g_gen.generate(key_shard)

        # B. Reconstruct Matrix on Device (Fast HBM)
        rec_out = g_gen.reconstruct(labels_shard)

        # Handle randomenerg tuple return
        if h_type == 'randomenerg':
            e_vals, i_vals = rec_out
            if e_vals.shape[1] * 2 == rho_1_d.shape[0]:
                e_vals = jnp.repeat(e_vals, 2, axis=1)
        else:
            e_vals = base_energies_shard
            i_vals = rec_out

        # B. Solve Eigenproblem
        f1, f2, en = solve_batch_kernel(
            e_vals, i_vals, beta, is_thermal,
            rho_1_d, rho_2_i, rho_t, rho_1_f
        )
        return labels_shard, f1, f2, en

    # Compile with PMAP (Parallel Map)
    # This works identically on Multi-GPU and TPU
    print("Compiling fused kernel...")
    p_step = jax.pmap(fused_step_fn, axis_name='batch')

    # --- 4. Warmup ---
    # Generate dummy labels (Batch 0) to trigger compilation
    rng_key = jax.random.PRNGKey(seed)
    # Generate dummy keys (1 per device) to trigger compilation
    rng_key, *warmup_keys = jax.random.split(rng_key, n_devices + 1)

    _ = p_step(jnp.array(warmup_keys), p_base_energies,
               p_rho_1_diag, p_rho_2_inter, p_rho_target, p_rho_1_full)
    print("Compilation complete. Starting high-speed generation...")

    # --- 5. Main Generation Loop ---
    num_batches = math.ceil(num_samples / batch_size)
    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = []

    # Use standard NumPy RNG (Guaranteed CPU execution, no "device" errors)
    rng = np.random.default_rng()
    label_len = g_gen.label_size()

    # Progress bar on stdout to avoid buffering
    pbar = tqdm(range(num_batches), desc="Processing", file=sys.stdout, mininterval=1.0)

    for i in pbar:
        # 1. Generate Random Parameter Vectors (CPU)
        # Fast: Only O(N) data. We replicate 'g_gen.generate' logic manually here
        # to rely purely on NumPy.
        rng_key, *subkeys = jax.random.split(rng_key, n_devices + 1)
        keys_sharded = jnp.array(subkeys)

        # 2. Run FULL Fused Kernel entirely on Accelerator
        labels_shard, f1_shard, f2_shard, en_shard = p_step(
            keys_sharded, p_base_energies,
            p_rho_1_diag, p_rho_2_inter, p_rho_target, p_rho_1_full
        )

        # 4. Async Save
        f1_shard.block_until_ready() # Ensure sync

        # Copy back to host
        raw_labels = np.array(labels_shard).reshape((batch_size, label_len))
        f1_np = np.array(f1_shard).reshape((batch_size,) + f1_shard.shape[2:])
        f2_np = np.array(f2_shard).reshape((batch_size,) + f2_shard.shape[2:])
        en_np = np.array(en_shard).reshape((batch_size,) + en_shard.shape[2:])

        filename = os.path.join(cache_path, f"shard_{i:05d}.npz")

        def save_task(fname, l, e, f1, f2):
            try:
                save_dict = {'labels': l}
                if include_energy: save_dict['energy'] = e[:, None]
                if input_type == 'rho1': save_dict['features'] = f1
                elif input_type == 'rho1+rho2':
                    save_dict['feat_0'] = f1; save_dict['feat_1'] = f2
                else: save_dict['features'] = f2
                np.savez_compressed(fname, **save_dict)
            except Exception as e:
                print(f"Write failed: {e}")

        futures.append(io_pool.submit(save_task, filename, raw_labels, en_np, f1_np, f2_np))

        # Flow control
        if len(futures) > 20:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            futures = [f for f in futures if not f.done()]

    concurrent.futures.wait(futures)
    io_pool.shutdown()

    return NumpyLoader(cache_path, batch_size)


# ---------------------------------------------------------------------------
# Era initializers (set the notebook's config globals in the shared namespace)
# ---------------------------------------------------------------------------

def init_d20(h_type, state_type, beta=1.0, gpu_batch_size=256, g_init=0.1, g_stop=1.0):
    """d=20 production era: paired basis, SCALE_FACTOR=10.0 (cell_03 + cell_13)."""
    global D_SP, N_ELEC, USE_PAIRING_RESTRICTION, GPU_BATCH_SIZE, basis
    global rho_1_arrays, rho_2_arrays, rho_2_kkbar_arrays, rho_2_block_arrays
    global BETA, SCALE_FACTOR, M_PAIRS, levels, energies, U_ENERGY_SEED
    global GEN_GPU_BATCH_SIZE, g_gen, STATE_TYPE

    # Configuration Parameters  # [from cell_03]
    D_SP = 20  # Number of single-particle modes
    N_ELEC = D_SP // 2  # Number of electrons (e.g., Half-filling).
    USE_PAIRING_RESTRICTION = True
    GPU_BATCH_SIZE = gpu_batch_size

    # --- Basis Initialization ---
    basis = fmb.FixedBasis(D_SP, num=N_ELEC, pairs=USE_PAIRING_RESTRICTION)

    # --- RDM Operator Arrays  ---
    rho_1_arrays = fmb.rho_m_gen(basis, 1, n_workers=1)
    rho_2_arrays = fmb.rho_m_gen(basis, 2, n_workers=1)

    # Specialized RDM blocks (for Pairing Hamiltonians)
    if D_SP % 2 == 0:
        rho_2_kkbar_arrays = fmb.rho_2_kkbar_gen(basis, n_workers=1)
        rho_2_block_arrays = fmb.rho_2_block_gen(basis, n_workers=1) if not USE_PAIRING_RESTRICTION else None

    # --- Simulation Configuration ---  # [from cell_13]
    # Temperature
    BETA = beta
    # Energy scale configuration
    SCALE_FACTOR = 10.0 # 4.0

    M_PAIRS = D_SP // 2
    # Calculate the levels (centered)
    levels = np.arange(0, N_ELEC) - N_ELEC // 2 + 1/2
    # Make them doubly degenerate (k, k_bar) and scale
    # [engine edit -- DETERMINISM PATCH] cell_13 used
    # np.random.uniform(-1e-4, 1e-4, M_PAIRS) here; replaced with the
    # deterministic np.linspace variant used by the same notebook's CPU
    # pipeline (gen_dataset_sp), so every run shares one level spectrum.
    pair_jitter = np.linspace(-1e-4, 1e-4, M_PAIRS).astype(np.float32)
    levels_jittered = levels + pair_jitter
    energies = np.repeat(levels_jittered, 2).astype(np.float32) / SCALE_FACTOR
    # Create the batched energy seed
    U_ENERGY_SEED = np.array([energies for _ in range(0, GPU_BATCH_SIZE)])
    GEN_GPU_BATCH_SIZE = 64

    g_gen = GGenerator(basis, h_type, GPU_BATCH_SIZE, g_init=g_init, g_stop=g_stop)
    STATE_TYPE = state_type


def init_d12(h_type, state_type, beta=10.0, gpu_batch_size=64, g_init=0.1, g_stop=1.0):
    """d=12 era (J3.0/J3.1): unpaired 924-dim basis, SCALE_FACTOR=1.0 (j31_lean.py)."""
    global D_SP, N_ELEC, USE_PAIRING_RESTRICTION, GPU_BATCH_SIZE, basis
    global rho_1_arrays, rho_2_arrays, rho_2_kkbar_arrays, rho_2_block_arrays
    global BETA, SCALE_FACTOR, M_PAIRS, levels, energies, U_ENERGY_SEED
    global GEN_GPU_BATCH_SIZE, g_gen, STATE_TYPE

    # [from reference/j31_lean.py l.27-37]
    D_SP = 12; N_ELEC = 6; M_PAIRS = 6
    SCALE_FACTOR = 1.0          # J3.0-resolved era value
    BETA = beta
    GPU_BATCH_SIZE = gpu_batch_size
    USE_PAIRING_RESTRICTION = False

    basis = fmb.FixedBasis(D_SP, num=N_ELEC, pairs=False)
    assert basis.size == 924, basis.size
    rho_1_arrays = fmb.rho_m_gen(basis, 1, n_workers=1)
    rho_2_kkbar_arrays = fmb.rho_2_kkbar_gen(basis)
    levels = np.arange(0, N_ELEC) - N_ELEC // 2 + 0.5
    energies = np.repeat(levels, 2).astype(np.float64) / SCALE_FACTOR

    # d=12 era never built the full rho_2 / block tensors (heavy, unused);
    # cleared here so stale d=20 arrays cannot leak across re-inits.
    rho_2_arrays = None
    rho_2_block_arrays = None

    # Batched energy seed, same shape convention as the d=20 era (cell_13)
    U_ENERGY_SEED = np.array([energies for _ in range(0, GPU_BATCH_SIZE)])
    GEN_GPU_BATCH_SIZE = 64

    g_gen = GGenerator(basis, h_type, GPU_BATCH_SIZE, g_init=g_init, g_stop=g_stop)
    STATE_TYPE = state_type
