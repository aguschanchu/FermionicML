#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os

RUN_ON_CLUSTER = False  
NUM_VMS = 4

if not RUN_ON_CLUSTER:
    from IPython.core.magic import register_cell_magic, register_line_magic
    
    # Intercept %%px to just run the cell normally in the local environment
    @register_cell_magic
    def px(line, cell):
        get_ipython().run_cell(cell)
        
    # Ignore IPyParallel line magics
    @register_line_magic
    def pxconfig(line): pass
    
    @register_line_magic
    def autopx(line): pass
    
    # Create a dummy 'rc' object so `rc.ids[0]` later in the notebook doesn't crash
    class DummyRC:
        ids = [0]
        def __getitem__(self, key): return self
        def push(self, *args, **kwargs): pass
    rc = DummyRC()

else:
    import ipyparallel as ipp
    rc = ipp.Client()
    rc.wait_for_engines(NUM_VMS) 
    print(f"Connected to {len(rc.ids)} TPU VMs") 
    get_ipython().run_line_magic('pxconfig', '--targets all')


# In[18]:


get_ipython().run_cell_magic('px', '', '\nimport os\nimport sys\nimport time\nimport math\nimport glob\nimport shutil\nimport concurrent.futures\nimport functools\nfrom functools import partial\nimport itertools\nfrom itertools import combinations\nimport multiprocessing\nfrom multiprocessing import Pool, cpu_count\nfrom typing import Literal, Optional, Tuple, Dict, List, Sequence, Any, Union\n\n# ==============================================================================\n# 1. INITIALIZE JAX DISTRIBUTED FIRST\n# ==============================================================================\nimport jax\n\nRUN_ON_CLUSTER = globals().get(\'RUN_ON_CLUSTER\', True)\n\nif RUN_ON_CLUSTER:\n    # Cluster Mode: Use TPU and initialize distributed rendezvous\n    jax.config.update("jax_platforms", "tpu,cpu")\n    try:\n        jax.distributed.initialize()\n    except RuntimeError as e:\n        if "already initialized" not in str(e).lower():\n            raise e\n    print(f"Worker {jax.process_index()} initialized: {jax.local_device_count()} local TPU devices.")\nelse:\n    # Local Mode Fix: Force CPU to prevent multi-host TPU rendezvous hang!\n    jax.config.update("jax_platforms", "cpu")\n    print(f"Local machine initialized on CPU: {jax.local_device_count()} local devices detected.")\n\nimport jax.numpy as jnp\nfrom jax import jit, value_and_grad\n\n# ==============================================================================\n# 2. PREVENT HEADLESS MATPLOTLIB HANGS\n# ==============================================================================\nimport matplotlib\nif RUN_ON_CLUSTER:\n    matplotlib.use(\'Agg\') # Forces headless plotting for cluster\nelse:\n    # Forces inline plotting for local VSCode/Jupyter environments\n    get_ipython().run_line_magic(\'matplotlib\', \'inline\') \nimport matplotlib.pyplot as plt\n\n# ==============================================================================\n# 3. THIRD-PARTY IMPORTS\n# ==============================================================================\nimport fermionic_mbody as fmb # https://github.com/aguschanchu/fermionic-mbody\nimport numba as nb\nfrom numba import njit, prange, typed, types\nimport numpy as np\nimport openfermion as of\nfrom openfermion.utils import commutator, count_qubits, hermitian_conjugated\nimport ray\nimport scipy\nimport scipy.sparse as sp\nimport scipy.optimize\nimport scipy.linalg\nfrom sklearn.metrics import mean_squared_error\nimport sparse\nimport sympy\nfrom sympy import symbols, Function, diff, lambdify\nfrom tqdm.auto import tqdm\n\nimport flax\nfrom flax import linen as nn\nfrom flax.training import train_state\nimport optax\n')


# # Setup

# In[19]:


get_ipython().run_cell_magic('px', '', '# Configuration Parameters\nD_SP = 20  # Number of single-particle modes\nN_ELEC = D_SP // 2  # Number of electrons (e.g., Half-filling).\nUSE_PAIRING_RESTRICTION = True  \nGPU_BATCH_SIZE = 256\n\n# --- Basis Initialization ---\nbasis = fmb.FixedBasis(D_SP, num=N_ELEC, pairs=USE_PAIRING_RESTRICTION)\n\n# --- RDM Operator Arrays  ---\nrho_1_arrays = fmb.rho_m_gen(basis, 1, n_workers=1)\nrho_2_arrays = fmb.rho_m_gen(basis, 2, n_workers=1)\n\n# Specialized RDM blocks (for Pairing Hamiltonians)\nif D_SP % 2 == 0:\n    rho_2_kkbar_arrays = fmb.rho_2_kkbar_gen(basis, n_workers=1)\n    rho_2_block_arrays = fmb.rho_2_block_gen(basis, n_workers=1) if not USE_PAIRING_RESTRICTION else None\n')


# In[20]:


get_ipython().run_cell_magic('px', '', 'def _ensure_dense(arr):\n        if hasattr(arr, \'todense\'):\n            return arr.todense()\n        return np.asarray(arr)\n\ndef two_body_hamiltonian_sp(basis: fmb.FixedBasis,\n                            energy_seed: Union[np.ndarray],\n                            int_mat: Union[np.ndarray],\n                            rho_1_arrays: sp.coo_array,\n                            rho_2_arrays: sp.coo_array,\n                            h_type: Optional[str] = None) -> sp.coo_array:\n    """\n    Constructs the many-body Hamiltonian H = H0 - HI in the N-body basis.\n\n    H0(b) = Sum_i E_i(b) * M(c_i^dag c_i)\n    HI(b) = Sum_{I,J} V_{I,J}(b) * M(C_I^dag C_J)\n    """\n    d = basis.d\n    DN = basis.size\n    is_paired_optimization = (basis.pairs and d % 2 == 0)\n\n    # --- H0 (One-body energy) ---\n    # Extract number operators M(c_i^dag c_i). Shape (d, DN, DN)\n    if is_paired_optimization:\n        # H0 = Sum_k E_k^Pair * N_k^Pair\n        m_pairs = d // 2\n        B = energy_seed.shape[0]\n        pair_energies = energy_seed.reshape(B, m_pairs, 2).sum(axis=-1)\n        H0_energy = sp.coo_array(pair_energies)\n        indices = np.arange(m_pairs, dtype=int)\n\n    # Extract standard number operators M(c_i^dag c_i)\n    else:\n        d_r1 = rho_1_arrays.shape[0]\n        indices = np.arange(d_r1, dtype=int)\n        pair_energies = energy_seed\n\n    rho_1_arrays = _ensure_dense(rho_1_arrays)\n    rho_1_diag = rho_1_arrays[indices, indices]\n\n    # Calculate H0\n    H0_energy_sparse = sp.coo_array(pair_energies) if not isinstance(pair_energies, sp.coo_array) else pair_energies\n    if H0_energy_sparse.ndim == 2:\n        # Einsum: \'bi,irc->brc\' (b=batch, i=index (pair or sp), r,c=many-body indices)\n        h0_arr = H0_energy_sparse.tensordot(rho_1_diag, axes=([1], [0]))\n    elif H0_energy_sparse.ndim == 1:\n        h0_arr = H0_energy_sparse.tensordot(rho_1_diag, axes=([0], [0]))\n    else:\n        raise ValueError(f"Unexpected dimensions for H0 energy input: {H0_energy_sparse.ndim}")\n    \n    # --- HI (Two-body interaction) ---\n\n    # Ensure int_mat is sparse\n    int_mat_sparse = sp.coo_array(np.asarray(int_mat))\n    # Calculate H_I. We contract V[i, j] with R[j, i, r, c].\n    if int_mat_sparse.ndim == 3:\n        # Replaced einsum: \'bij,jirc->brc\' \n        # i = axis 1 (int_mat) / axis 1 (rho_2), j = axis 2 (int_mat) / axis 0 (rho_2)\n        hi_arr = int_mat_sparse.tensordot(rho_2_arrays, axes=([1, 2], [1, 0]))\n    elif int_mat_sparse.ndim == 2:\n        # Replaced einsum: \'ij,jirc->rc\'\n        hi_arr = int_mat_sparse.tensordot(rho_2_arrays, axes=([0, 1], [1, 0]))\n    else:\n        raise ValueError(f"Unexpected dimensions for int_mat")\n\n    return h0_arr - hi_arr\n\n@jax.jit\ndef two_body_hamiltonian_dense(\n                            energy_params: jnp.ndarray,     \n                            interaction_params: jnp.ndarray,\n                            rho_1_diag: jnp.ndarray,        \n                            rho_2_tensor: jnp.ndarray       \n                            ) -> jnp.ndarray:\n    """\n    Constructs a batch of Hamiltonians H = H0 - HI \n    """\n    # --- 1. H0 Construction (Optimized) ---\n    # H0 is diagonal in the Fock basis. We compute the diagonal vector directly.\n    # (Batch, D_SP) @ (D_SP, N) -> (Batch, N)\n    if energy_params.shape[-1] == rho_1_diag.shape[0] * 2:\n        energy_params = energy_params.reshape(energy_params.shape[0], -1, 2).sum(axis=-1)\n\n    h0_diag_vals = jnp.dot(energy_params, rho_1_diag)\n    \n    # --- 2. HI Construction (Tensor Contraction) ---\n    # Case A: Matrix-style params (B, I, J). We contract V_ij * Op_ji\n    if interaction_params.ndim == 3: \n        # Note: GGenerator typically outputs (I, J). \n        # We assume Op is (J, I, N, N) to match original sparse einsum \'bij,jirc\'\n        HI = jnp.einsum(\'bij,jirc->brc\', interaction_params, rho_2_tensor)\n        \n    # Case B: Vector-style params (B, K). Contract V_k * Op_k\n    else: \n        HI = jnp.tensordot(interaction_params, rho_2_tensor, axes=(1, 0))\n        \n    # --- 3. Combine: H = -HI + H0 ---\n    H_total = -HI\n    \n    N = h0_diag_vals.shape[-1]\n    indices = jnp.arange(N)\n    \n    # 1. RESTORE H_0 \n    H_total = H_total.at[:, indices, indices].add(h0_diag_vals)\n\n    # 2. Enforce Hermiticity to clean up numerical noise\n    H_total = 0.5 * (H_total + jnp.swapaxes(H_total.conj(), 1, 2))\n    \n    return H_total\ndef state_energy(state: Union[np.ndarray, sparse.COO], h_arr: Union[np.ndarray, sparse.COO]) -> np.ndarray:\n    """\n    Calculates the energy E = Tr(rho * H) for a batch of states and Hamiltonians.\n    """\n    if isinstance(state, np.ndarray) and state.ndim == 2 and isinstance(h_arr, np.ndarray):\n            # <psi|H|psi> = sum(psi.conj * (H @ psi))\n            # H @ psi: (B, N, N) @ (B, N, 1) -> (B, N)\n            h_psi = np.einsum(\'bij,bj->bi\', h_arr, state)\n            return np.sum(state.conj() * h_psi, axis=1).real\n\n    # Fallback / Density Matrices\n    energy = sparse.einsum(\'bij,bji->b\', state, h_arr)\n    if hasattr(energy, \'todense\'):\n        return energy.todense()\n    return np.asarray(energy)\n')


# In[21]:


get_ipython().run_cell_magic('px', '', 'import jax\nimport jax.numpy as jnp\n\n# =====================================================================\n# TPU-OPTIMIZED CUSTOM VJP (PURE FLOAT32)\n# =====================================================================\n\n@jax.custom_vjp\ndef eigensolve_and_build_rho(H_batch, beta):\n    """\n    Computes the thermal density matrix rho = e^{-beta H} / Z and expected energy.\n    Runs natively in float32 for maximum TPU speed. NO float64 emulation!\n    """\n    vals, vecs = jnp.linalg.eigh(H_batch)\n    \n    # jax.nn.softmax natively handles max-shifting for numerical stability\n    probs = jax.nn.softmax(-beta * vals, axis=-1)\n    \n    state_rho = jnp.matmul(vecs * probs[..., None, :], jnp.swapaxes(vecs.conj(), -1, -2))\n    energy_out = jnp.sum(probs * vals, axis=-1)\n    \n    return state_rho, energy_out\n\n# Define the forward pass\ndef eigensolve_and_build_rho_fwd(H_batch, beta):\n    N = H_batch.shape[-1]\n    sym_breaker = 1e-5 * ((jnp.arange(N, dtype=jnp.float32) % 2) * 2.0 - 1.0)\n    indices = jnp.arange(N)\n    H_safe = H_batch.at[..., indices, indices].add(sym_breaker)\n\n    vals, vecs = jnp.linalg.eigh(H_safe)\n    \n    probs = jax.nn.softmax(-beta * vals, axis=-1)\n    \n    state_rho = jnp.matmul(vecs * probs[..., None, :], jnp.swapaxes(vecs.conj(), -1, -2))\n    energy_out = jnp.sum(probs * vals, axis=-1)\n    \n    res = (vals, vecs, probs, energy_out, beta)\n    return (state_rho, energy_out), res\n\n# Define the backward pass\ndef eigensolve_and_build_rho_bwd(res, g):\n    vals, vecs, probs, energy_out, beta = res\n    g_state_rho, g_energy = g\n\n    N = vecs.shape[-1]\n\n    # Construct C matrix\n    D_diff = vals[..., :, None] - vals[..., None, :]\n    P_diff = probs[..., :, None] - probs[..., None, :]\n    \n    # Float32-friendly masking threshold to catch degeneracies\n    mask = jnp.abs(D_diff) > 1e-5\n    safe_D_diff = jnp.where(mask, D_diff, 1.0)\n    \n    # L\'Hôpital\'s analytical limit safely bridges degenerate eigenvalue differences\n    limit_C = -beta * (probs[..., :, None] + probs[..., None, :]) / 2.0\n    C = jnp.where(mask, P_diff / safe_D_diff, limit_C)\n    \n    # Build G = V^dagger g_rho V\n    vecs_adj = jnp.swapaxes(vecs.conj(), -1, -2)\n    G = jnp.matmul(vecs_adj, jnp.matmul(g_state_rho, vecs))\n    \n    # Compute Trace(G * P)\n    G_diag = jnp.diagonal(G, axis1=-2, axis2=-1)\n    Tr_GP = jnp.sum(G_diag * probs, axis=-1)\n    \n    # Construct Hamiltonian gradient in eigenbasis\n    H_bar_V = G * C\n    \n    # Diagonal corrections\n    diag_correction = beta * probs * Tr_GP[..., None]\n    diag_correction = diag_correction + g_energy[..., None] * probs * (1.0 - beta * (vals - energy_out[..., None]))\n        \n    eye = jnp.eye(N, dtype=H_bar_V.dtype)\n    H_bar_V = H_bar_V + diag_correction[..., None] * eye\n    \n    # Rotate back to standard operator basis\n    grad_H = jnp.matmul(vecs, jnp.matmul(H_bar_V, vecs_adj))\n    \n    # Enforce strict Hermiticity to scrub floating point noise\n    grad_H = 0.5 * (grad_H + jnp.swapaxes(grad_H.conj(), -1, -2))\n\n    return (grad_H, None)\n\neigensolve_and_build_rho.defvjp(eigensolve_and_build_rho_fwd, eigensolve_and_build_rho_bwd)\n')


# In[22]:


get_ipython().run_cell_magic('px', '', 'def _to_dense_array(H_sp):\n    if scipy.sparse.issparse(H_sp):\n        return H_sp.toarray()\n    elif hasattr(H_sp, \'todense\'):\n        return H_sp.todense()\n    return np.asarray(H_sp)\n\ndef _prepare_hamiltonian_for_eigsh(H_i: Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix]) -> Union[np.ndarray, scipy.sparse.spmatrix]:\n    """Helper to convert Hamiltonian to a format suitable for scipy eigensolvers"""\n    if isinstance(H_i, sparse.COO):\n        return H_i.to_scipy_sparse().tocsr()\n    elif scipy.sparse.issparse(H_i):\n        return H_i.tocsr() if hasattr(H_i, \'tocsr\') else H_i\n    else:\n        return np.asarray(H_i)\n\n\ndef pure_state(h: Union[np.ndarray, sp.coo_array]):\n    """\n    Calculates the ground state density matrix rho = |psi_0><psi_0| for a batch of Hamiltonians.\n    """\n    B, DN, _ = h.shape\n    mat = np.zeros((B, DN, DN), dtype=np.complex128)\n    energies = np.zeros(B, dtype=np.float64)\n\n    if isinstance(h, np.ndarray) or (hasattr(h, \'todense\') and DN < 512):\n            if not isinstance(h, np.ndarray): h = h.todense()\n            vals, vecs = np.linalg.eigh(h)\n            fund = vecs[:, :, 0]\n            mat = np.einsum(\'bi,bj->bij\', fund, fund.conj())\n            return vals[:, 0], mat.real\n\n    for i in range(B):\n        H_i_sp = _prepare_hamiltonian_for_eigsh(h[i])\n        e, v = scipy.sparse.linalg.eigsh(H_i_sp, k=1, which=\'SA\', tol=1e-8)\n        fund = v[:, 0]\n        mat[i, :, :] = np.outer(fund, fund.conj())\n        energies[i] = e[0]\n\n    return energies, mat.real\n\ndef k_safe(N: int, beta: float, eps: float = 1e-8, eta: float = 4.0, Delta_eff: float = 1.0) -> int:\n    """\n    Heuristic calculation for the number of eigenstates required for a thermal state approximation.\n    """\n    if N <= 1:\n        return 1\n\n    # Calculation based on Boltzmann weight decay\n    val = eta * (1.0 / (beta * max(Delta_eff, 1e-12))) * math.log(max(N/eps, 1.0))\n\n    # Ensure the result is within valid bounds [1, N-1]\n    return min(N - 1, max(1, math.ceil(val)))\n\n\ndef thermal_state(beta: float, h: Union[np.ndarray, sp.coo_array]):\n    """\n    Calculates the thermal density matrix rho = exp(-beta*H)/Z for a batch of Hamiltonians\n    """\n    B, N, _ = h.shape\n    mat = np.zeros((B, N, N), dtype=np.complex128)\n\n    k0 = k_safe(N, beta)\n    use_dense = (k0 > 0.25 * N)\n    energies = np.zeros(B, dtype=np.float64)\n\n    for i in range(B):\n        H_i_scipy = _prepare_hamiltonian_for_eigsh(h[i])\n\n        try:\n            if use_dense:\n                H_i_dense = _to_dense_array(H_i_scipy)\n                E, V = np.linalg.eigh(H_i_dense)\n            else:\n                E, V = scipy.sparse.linalg.eigsh(H_i_scipy, k=k0, which="SA", tol=1e-8, return_eigenvectors=True)\n\n        except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as e:\n             print(f"Warning: ARPACK failed (k={k0}) for index {i}. Falling back to dense.")\n             try:\n                 H_i_dense = _to_dense_array(H_i_scipy)\n                 E, V = np.linalg.eigh(H_i_dense)\n             except Exception as e_dense:\n                 print(f"Dense fallback failed for index {i}: {e_dense}")\n                 mat[i, :, :] = np.nan\n                 continue\n\n        # Calculate weights\n        E_shifted = E - E.min()\n        weights = np.exp(-beta * E_shifted)\n        Z = weights.sum()\n\n        if Z < 1e-12:\n            print(f"Warning: Partition function Z near zero for index {i}.")\n            mat[i, :, :] = np.nan\n            continue\n\n        weights /= Z\n        energies[i] = np.sum(weights * E)\n\n        # Construct the density matrix: rho = V @ diag(weights) @ V.H\n        mat[i, :, :] = (V * weights[None, :]) @ V.conj().T\n\n    return energies, mat.real\n\ndef compute_rdm_trace(state, ops):\n    """\n    RDM = Tr(rho * Op) -> Einsum \'bij, kij -> bk\' \n    """\n\n    # Flatten spatial dims: (M, M, N, N) -> (M*M, N, N)\n    orig_shape = ops.shape[:-2]\n    ops_flat = ops.reshape(-1, ops.shape[-2], ops.shape[-1])\n    \n    # Compute trace: sum(rho_nm * op_mn) -> einsum \'bnm, knm\'\n    res = jnp.einsum(\'bnm,kmn->bk\', state, ops_flat)\n    return res.reshape((state.shape[0],) + orig_shape + (1,))\n\n@partial(jax.jit, static_argnames=[\'is_thermal\'])\ndef solve_batch_kernel(\n    e_params, i_params, beta, is_thermal,\n    rho_1_diag, rho_2_inter, rho_target, rho_1_full):\n    """\n    Performs the full features extraction on GPU\n    Params -> Hamiltonian -> Eigensolve -> State -> RDM Features\n    """\n    \n    # 1. Build Hamiltonian Batch\n    H_batch = two_body_hamiltonian_dense(e_params, i_params, rho_1_diag, rho_2_inter)\n    \n    effective_beta = beta if is_thermal else 100.0\n    state_rho, energy_out = eigensolve_and_build_rho(H_batch, effective_beta)\n\n    # 4. Compute Features (RDMs)\n    f1_out = compute_rdm_trace(state_rho, rho_1_full)\n\n    # Feature 2: Target Rho (kkbar or block)\n    f2_out = compute_rdm_trace(state_rho, rho_target)\n    \n    return f1_out.real, f2_out.real, energy_out.real\n\ndef solve_batch_kernel_sp(basis, e_params, i_params, beta, is_thermal, rho_1_arrays, rho_2_inter, rho_target):\n\n    H_batch = two_body_hamiltonian_sp(basis, e_params, i_params, rho_1_arrays, rho_2_inter)\n\n    if is_thermal:\n        energy_out, state = thermal_state(beta, H_batch)\n    else:\n        energy_out, state = pure_state(H_batch)\n    \n    f1_out = fmb.rho_m(state, rho_1_arrays)\n    f2_out = fmb.rho_m(state, rho_target)\n\n    return f1_out.real, f2_out.real, energy_out.real\n')


# # Machine Learning model
# Basado en matrices densidad de 1 y 2 cuerpos como input, con hamiltoniano como salida

# ### Dataset generation

# Definimos todos los generadores del término de energías e interacción, que luego utilizaremos para generar H mediante two_body_hamiltonian_sp. Tendremos los siguentes tipos:
# 
# *Únicamente bloque kkbar ($G_{kk'}$) $C^\dag_{k'}C^\dag_{\bar{k}'} C_{\bar{k}} C_k$*
# *  const:        $G_{kk'}=G$
# *  random:       $G_{kk'}=G_{kk'}$ random simétrica
# *  vect:         $G_{kk'}=G_{\frac{|k-k'|}{{repeat}}}$
# *  gaussian:     $G_{kk'}=G e^{\frac{|k-k'|^2}{2 \sigma^2}}$ G, $\sigma^2$ parámetros
# 
# *Bloque completo ($G_{\alpha\alpha'}$)*
# *  randomenerg: $H_0 = \epsilon_k (C_k^\dag C_k + C^\dag_{\bar{k}}C_{\bar{k}})$ energías diagonales + interacción random antisimétrica $H_i=G_{\alpha \alpha'}C^\dag_{\alpha'} C_\alpha$
# *  blockgen: $H_i=G_{ijkl}C^\dag_{i} C^\dag_{\bar{j}} C_{\bar{k}}C_l$
# *  blockgensimp: $H_i=G_{ij}G_{kl}C^\dag_{i} C^\dag_{\bar{j}} C_{\bar{k}}C_l$ simplificando la estructura anterior
# 

# A continuación vamos a definir las funciones que generan los arrays de H. Estas las pueden generar de manera random, o dada una semilla (utilizaremos esto último para la reconstrucción)

# In[23]:


get_ipython().run_cell_magic('px', '', 'import jax\nimport jax.numpy as jnp\nimport numpy as np\nimport math\nfrom functools import partial\n\nclass GGenerator:\n    """\n    Generates and reconstructs parameterized Hamiltonian matrices\n    """\n    def __init__(self, basis, h_type: str, batch_size: int, g_init=0.0, g_stop=1.0, *, vrepeat=2, justenerg=False):\n        # Handle basis object or integer input\n        self.d = basis.d if hasattr(basis, \'d\') else basis\n        self.m = self.d // 2\n        \n        self.h_type = h_type\n        self.batch_size = int(batch_size)\n        self.g_init = float(g_init)\n        self.g_stop = float(g_stop)\n        self.vrepeat = int(vrepeat)\n        self.justenerg = bool(justenerg)\n        \n        # Precompute sizes\n        self.opsize = self._get_opsize()\n        self._initialize_indices()\n\n    def _get_opsize(self):\n        if self.h_type in [\'const\', \'random\', \'vect\', \'gaussian\', \'blockgensimp\', \'randomenerg\']:\n            return self.m \n        elif self.h_type == \'blockgen\':\n            return self.m ** 2 \n        raise ValueError(f"Unsupported h_type: {self.h_type}")\n\n    def label_size(self):\n        m = self.m\n        if self.h_type == \'const\': return 1\n        elif self.h_type == \'random\': return (m * (m + 1)) // 2\n        elif self.h_type == \'vect\': return math.ceil(m / self.vrepeat) - 1\n        elif self.h_type == \'gaussian\': return 2\n        elif self.h_type == \'blockgen\': return (m**4 + m**2) // 2\n        elif self.h_type == \'blockgensimp\': return (m**2 + m) // 2\n        elif self.h_type == \'randomenerg\':\n            le = m\n            li = (m**2 + m) // 2\n            return le if self.justenerg else (le + li)\n        return 0\n\n    def _initialize_indices(self):\n        # Precompute indices using NumPy\n        # k=1 (strict upper triangle) for random, k=0 (includes diagonal) for others\n        self._k = 0 if self.h_type in [\'random\', \'blockgensimp\', \'randomenerg\'] else 0    \n\n        r, c = np.triu_indices(self.opsize, k=self._k)\n        self._triu_r = jnp.array(r)\n        self._triu_c = jnp.array(c)\n        \n        if self.h_type == \'vect\':\n            idx_np = np.abs(np.arange(self.opsize)[:, None] - np.arange(self.opsize)[None, :])\n            self._vect_idx = jnp.array(idx_np)\n\n        if self.h_type == \'gaussian\':\n            i = np.arange(self.opsize, dtype=np.float32)\n            self._gauss_diff2 = jnp.array((i[:, None] - i[None, :]) ** 2)\n\n        if self.h_type == \'random\':\n            self._triu_r = jnp.array(r)\n            self._triu_c = jnp.array(c)\n\n    @partial(jax.jit, static_argnums=(0,))\n    def _symmetric(self, h_labels):\n        """Reconstructs symmetric matrices from flattened upper triangular values."""\n        B = h_labels.shape[0]\n        m = self.opsize\n        \n        # Initialize zero batch\n        out = jnp.zeros((B, m, m), dtype=h_labels.dtype)\n        \n        # Fill upper triangle: out[:, r, c] = labels\n        out = out.at[:, self._triu_r, self._triu_c].set(h_labels)\n        \n        # Symmetrize: M + M.T\n        out_t = jnp.swapaxes(out, 1, 2)\n        out = out + out_t\n        \n        if self._k == 0:\n            diag_idx = jnp.arange(m)\n            diags = out[:, diag_idx, diag_idx]\n            out = out.at[:, diag_idx, diag_idx].set(diags * 0.5)\n\n        return out\n\n    @partial(jax.jit, static_argnums=(0,))\n    def reconstruct(self, h_labels):\n        """\n        Reconstruct Hamiltonian matrices from labels.\n        """\n        B = h_labels.shape[0]\n\n        if self.h_type == \'const\':\n            val = h_labels.reshape(B, 1, 1)\n            return jnp.tile(val, (1, self.opsize, self.opsize))\n\n        elif self.h_type in [\'random\', \'blockgen\']:\n            return self._symmetric(h_labels)\n\n        elif self.h_type == \'vect\':\n            zeros = jnp.zeros((B, 1))\n            # Pad, Repeat, Slice\n            expanded = jnp.concatenate([zeros, h_labels], axis=1)\n            expanded_rep = jnp.repeat(expanded, self.vrepeat, axis=1)\n            vals = expanded_rep[:, :self.opsize]\n            # Gather using precomputed distance indices\n            return vals[:, self._vect_idx]\n\n        elif self.h_type == \'gaussian\':\n            G = h_labels[:, 0]\n            sigma_sq = jnp.maximum(jnp.abs(h_labels[:, 1]), 1e-4)\n            exponent = -self._gauss_diff2[None, :, :] / (2.0 * sigma_sq[:, None, None])\n            return G[:, None, None] * jnp.exp(exponent)\n\n        elif self.h_type == \'blockgensimp\':\n            V = self._symmetric(h_labels)\n            V_flat = V.reshape(B, -1)\n            # Batched outer product\n            return jnp.einsum(\'bi,bj->bij\', V_flat, V_flat)\n\n        elif self.h_type == \'randomenerg\':\n            energ = h_labels[:, :self.m]\n            if self.justenerg:\n                return energ\n            \n            int_labels = h_labels[:, self.m:]\n            V = self._symmetric(int_labels)\n            V_flat = V.reshape(B, -1)\n            mat = jnp.einsum(\'bi,bj->bij\', V_flat, V_flat)\n            return energ, mat\n            \n        return jnp.zeros((B, self.opsize, self.opsize))\n\n    @partial(jax.jit, static_argnums=(0,))\n    def generate(self, key):\n        """Helper for dataset generation (Pure JAX/TPU execution)"""\n        L = self.label_size()\n        B = self.batch_size\n        \n        # Split key for deterministic, stateless RNG \n        key_g, key_e = jax.random.split(key)\n        \n        # Use JAX native random uniform\n        h_labels = jax.random.uniform(\n            key_g, shape=(B, L), \n            minval=self.g_init, maxval=self.g_stop, \n            dtype=jnp.float32\n        )\n\n        if self.h_type == \'vect\':\n            # jnp.sort sorts ascending, slicing [::-1] reverses it to descending\n            h_labels = jnp.sort(h_labels, axis=1)[:, ::-1]\n \n        elif self.h_type == \'random\':\n            raw_labels = jax.random.uniform(\n                key_g, shape=(B, L), \n                minval=self.g_init, maxval=self.g_stop, \n                dtype=jnp.float32\n            )\n            \n            m = self.opsize\n            out = jnp.zeros((B, m, m), dtype=jnp.float32)\n            \n            r_full, c_full = jnp.triu_indices(m)\n            out = out.at[:, r_full, c_full].set(raw_labels)\n\n            diag_idx = jnp.arange(m)\n            diags = out[:, diag_idx, diag_idx]\n            target_mean = (self.g_init + self.g_stop) / 2.0\n            shift = target_mean - jnp.mean(diags, axis=1, keepdims=True)\n            out = out.at[:, diag_idx, diag_idx].set(diags + shift)\n            \n            h_labels = out[:, self._triu_r, self._triu_c]\n            return L, h_labels\n\n        elif self.h_type == \'randomenerg\':\n            ene = jax.random.uniform(\n                key_e, shape=(B, self.m), \n                minval=0.1, maxval=10.0, \n                dtype=jnp.float32\n            )\n            ene -= jnp.mean(ene, axis=1, keepdims=True)\n            ene = jnp.sort(ene, axis=1)[:, ::-1]\n            \n            if self.justenerg: \n                h_labels = ene\n            else: \n                # JAX requires .at[...].set(...) for array mutations (no in-place assignment)\n                h_labels = h_labels.at[:, :self.m].set(ene)\n                \n        elif self.h_type == \'blockgensimp\':\n             V_temp = self._symmetric(h_labels)\n             flip = V_temp.sum(axis=(1, 2)) < 0\n             h_labels = jnp.where(flip[:, None], -h_labels, h_labels)\n\n        return L, h_labels\n')


# In[24]:


get_ipython().run_cell_magic('px', '', '"""\nAuxiliary optimizers\n"""\n\ndef compute_rho_m(state_batch, op_tensor, batch_size):\n        # Fallback for sparse ops\n        if not isinstance(op_tensor, np.ndarray):\n            res = fmb.rho_m(state_batch, op_tensor)\n            if hasattr(res, \'todense\'): res = res.todense()\n            return res[..., np.newaxis].astype(np.float32)\n            \n        # Dense path\n        # Flatten operators: (..., N, N) -> (K, N, N)\n        op_shape = op_tensor.shape\n        D_N = op_shape[-1]\n        ops_flat = op_tensor.reshape(-1, D_N, D_N)\n        \n        if state_batch.ndim == 2: # State Vectors (B, N)\n            # RDM[b, k] = <psi_b | O_k | psi_b>\n            # Einstein sum: state[b, i]* . ops[k, i, j] . state[b, j]\n            # (B, N) conj . (K, N, N) . (B, N) -> (B, K)\n            rdm_flat = np.einsum(\'bi,kij,bj->bk\', state_batch.conj(), ops_flat, state_batch)\n            \n        elif state_batch.ndim == 3: # Density Matrices (B, N, N)\n            # Tr(rho * O) = sum(rho[b, j, i] * ops[k, i, j])\n            rdm_flat = np.einsum(\'bji,kij->bk\', state_batch, ops_flat)\n            \n        return rdm_flat.reshape((batch_size,) + op_shape[:-2] + (1,)).astype(np.float32)\n\ndef _to_dense_jax(arr):\n    """\n    Helper to convert sparse/numpy arrays to dense JAX arrays on GPU.\n    """\n    if arr is None: return None\n    if hasattr(arr, \'todense\'):\n        return jnp.array(arr.todense())\n    return jnp.array(arr)\n')


# In[25]:


get_ipython().run_cell_magic('px', '', '# --- Simulation Configuration ---\n\n# Temperature\nBETA = 1\n# Energy scale configuration \nSCALE_FACTOR = 10.0 # 4.0\n\nM_PAIRS = D_SP // 2\n# Calculate the levels (centered)\nlevels = np.arange(0, N_ELEC) - N_ELEC // 2 + 1/2\n# Make them doubly degenerate (k, k_bar) and scale\npair_jitter = np.random.uniform(-1e-4, 1e-4, M_PAIRS).astype(np.float32)\nlevels_jittered = levels + pair_jitter\nenergies = np.repeat(levels_jittered, 2).astype(np.float32) / SCALE_FACTOR\n# Create the batched energy seed \nU_ENERGY_SEED = np.array([energies for _ in range(0, GPU_BATCH_SIZE)])\nGEN_GPU_BATCH_SIZE = 64\n\n# Type definitions for validation\nValidHType = Literal[\'const\', \'gaussian\', \'vect\', \'random\', \'randomenerg\', \'blockgen\', \'blockgensimp\']\nValidStateType = Literal[\'thermal\', \'gs\']\nValidInputType = Literal[\'rho2\', \'rho1\', \'rho1+rho2\', \'rho2kkbar\', \'rho2block\']\n\n\nclass NumpyLoader:\n    """Iterates over .npz shard files, slicing them into exact batch_size chunks."""\n    def __init__(self, data_dir, batch_size, shuffle=True):\n        self.files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))\n        self.batch_size = batch_size\n        self.shuffle = shuffle\n        if not self.files: print(f"Warning: No files found in {data_dir}")\n\n    def __iter__(self):\n        if self.shuffle: np.random.shuffle(self.files)\n        for f in self.files:\n            try:\n                with np.load(f) as d:\n                    Y = d[\'labels\']\n                    E = d[\'energy\'] if \'energy\' in d else None\n                    if \'features\' in d: X = d[\'features\']; is_tuple = False\n                    else: \n                        keys = sorted([k for k in d.files if k.startswith(\'feat_\')])\n                        X = [d[k] for k in keys]; is_tuple = True\n                    \n                    # Slicing loop: Breaks large files into small batches\n                    N = len(Y)\n                    indices = np.arange(N)\n                    if self.shuffle: np.random.shuffle(indices)\n                    \n                    for start in range(0, N, self.batch_size):\n                        end = min(start + self.batch_size, N)\n                        idx = indices[start:end]\n                        if len(idx) < self.batch_size: continue # Skip partial batches\n                        \n                        by = Y[idx]\n                        be = E[idx] if E is not None else None\n                        bx = tuple(x[idx] for x in X) if is_tuple else X[idx]\n                        if not is_tuple and bx.ndim == 3: bx = bx[..., None]\n                        yield bx, be, by\n            except Exception as e: print(f"Error loading {f}: {e}")\n\ndef check_existing_dataset(cache_path, num_samples, batch_size):\n    """Regenerates only if dataset is missing or too small."""\n    if not os.path.exists(cache_path): return False\n    files = glob.glob(os.path.join(cache_path, "shard_*.npz"))\n    if not files: return False\n    \n    try:\n        with np.load(files[0]) as d: shard_size = len(d[\'labels\'])\n        total_est = len(files) * shard_size\n        \n        if total_est >= num_samples * 0.95:\n            print(f"Dataset valid (~{total_est} samples). Loading from disk.")\n            return True\n        else:\n            print(f"Dataset incomplete ({total_est}/{num_samples}). Regenerating...")\n            return False\n    except: return False\n        \n\ndef gen_dataset(h_type, g_init, g_stop, state_type, input_type,\n                include_energy, beta, num_samples=100000,\n                cache_path="./dataset", batch_size=4096, train_batch_size=GPU_BATCH_SIZE,\n                seed=42): \n\n\n    if check_existing_dataset(cache_path, num_samples, batch_size):\n        return NumpyLoader(cache_path, batch_size)\n\n    # 2. Cleanup & Setup\n    if os.path.exists(cache_path): shutil.rmtree(cache_path)\n    # This line previously failed because cache_path was an int. Now it\'s safe.\n    os.makedirs(cache_path, exist_ok=True)\n\n    # --- 0. Universal Device Detection (TPU & Multi-GPU) ---\n    # jax.local_devices() automatically finds 8 TPUs or N GPUs\n    devices = jax.local_devices()\n    n_devices = len(devices)\n    platform = jax.lib.xla_bridge.get_backend().platform.upper()\n    \n    print(f"Hardware: {n_devices} x {platform} devices detected.")\n\n    if batch_size % n_devices != 0:\n        # Auto-adjust batch size to divide evenly\n        new_bs = ((batch_size // n_devices) + 1) * n_devices\n        print(f"Adjusted batch size {batch_size} -> {new_bs} to fit devices.")\n        batch_size = new_bs\n    \n    device_batch_size = batch_size // n_devices\n    \n    # --- Helper: Replicate Constants to Devices ---\n    # This copies the constant Rho matrices to Accelerator Memory ONCE.\n    def replicate(arr):\n        if hasattr(arr, \'todense\'): arr = arr.todense()\n        arr = np.asarray(arr)\n        # Cast to float32/complex64 for performance\n        if np.iscomplexobj(arr): arr = arr.astype(np.complex64)\n        else: arr = arr.astype(np.float32)\n        # Shard the same copy to all devices efficiently\n        return jax.device_put_sharded([jnp.array(arr)] * n_devices, devices)\n\n    print("Moving static operators to accelerator memory...")\n    \n    # 1. Rho 1\n    rho_1_np = _ensure_dense(rho_1_arrays)\n    if rho_1_np.ndim == 4: rho_1_diag_np = np.einsum(\'kknn->kn\', rho_1_np)\n    else: rho_1_diag_np = np.diagonal(rho_1_np, axis1=1, axis2=2)\n\n    p_rho_1_full = replicate(rho_1_np)\n    p_rho_1_diag = replicate(rho_1_diag_np)\n\n    # 2. Rho 2 & Target\n    if h_type in [\'blockgen\', \'blockgensimp\', \'randomenerg\']:\n        inter_np = _ensure_dense(rho_2_block_arrays)\n    else:\n        target = rho_2_kkbar_arrays if \'rho_2_kkbar_arrays\' in globals() else rho_2_arrays\n        inter_np = _ensure_dense(target)\n    \n    if \'rho2block\' in input_type: target_np = _ensure_dense(rho_2_block_arrays)\n    elif \'rho2kkbar\' in input_type: target_np = _ensure_dense(rho_2_kkbar_arrays)\n    else: target_np = inter_np\n\n    p_rho_2_inter = replicate(inter_np)\n    p_rho_target = replicate(target_np)\n\n    # --- 2. Initialize Generator Logic ---\n    # We initialize the generator logic. We will use this INSIDE the pmap.\n    # Note: We pass \'device_batch_size\' so inner dimensions match the split batch.\n    g_gen = GGenerator(basis, h_type, device_batch_size, g_init=g_init, g_stop=g_stop)\n    \n    # Pre-calculate base energies (Batch, D)\n    base_energies_np = U_ENERGY_SEED[0]\n    \n    # Replicate base energies to devices: (Devices, Device_Batch, D)\n    p_base_energies = replicate(np.tile(base_energies_np, (device_batch_size, 1)))\n\n    # --- 3. Define Fused Kernel (Reconstruct + Solve) ---\n    # CRITICAL OPTIMIZATION: We move reconstruction to the TPU.\n    # Input: Tiny Labels (float32) -> Output: Huge Features (float32)\n    is_thermal = (state_type == \'thermal\')\n\n    def fused_step_fn(key_shard, base_energies_shard, \n                      rho_1_d, rho_2_i, rho_t, rho_1_f):\n        \n        # Because `g_gen` is configured with `device_batch_size`, it generates the shard natively.\n        _, labels_shard = g_gen.generate(key_shard)\n        \n        # B. Reconstruct Matrix on Device (Fast HBM)\n        rec_out = g_gen.reconstruct(labels_shard)\n        \n        # Handle randomenerg tuple return\n        if h_type == \'randomenerg\':\n            e_vals, i_vals = rec_out\n            if e_vals.shape[1] * 2 == rho_1_d.shape[0]:\n                e_vals = jnp.repeat(e_vals, 2, axis=1)\n        else:\n            e_vals = base_energies_shard\n            i_vals = rec_out\n            \n        # B. Solve Eigenproblem\n        f1, f2, en = solve_batch_kernel(\n            e_vals, i_vals, beta, is_thermal,\n            rho_1_d, rho_2_i, rho_t, rho_1_f\n        )\n        return labels_shard, f1, f2, en\n\n    # Compile with PMAP (Parallel Map)\n    # This works identically on Multi-GPU and TPU\n    print("Compiling fused kernel...")\n    p_step = jax.pmap(fused_step_fn, axis_name=\'batch\')\n\n    # --- 4. Warmup ---\n    # Generate dummy labels (Batch 0) to trigger compilation\n    rng_key = jax.random.PRNGKey(seed)\n    # Generate dummy keys (1 per device) to trigger compilation\n    rng_key, *warmup_keys = jax.random.split(rng_key, n_devices + 1)\n    \n    _ = p_step(jnp.array(warmup_keys), p_base_energies, \n               p_rho_1_diag, p_rho_2_inter, p_rho_target, p_rho_1_full)\n    print("Compilation complete. Starting high-speed generation...")\n\n    # --- 5. Main Generation Loop ---\n    num_batches = math.ceil(num_samples / batch_size)\n    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)\n    futures = []\n    \n    # Use standard NumPy RNG (Guaranteed CPU execution, no "device" errors)\n    rng = np.random.default_rng()\n    label_len = g_gen.label_size()\n\n    # Progress bar on stdout to avoid buffering\n    pbar = tqdm(range(num_batches), desc="Processing", file=sys.stdout, mininterval=1.0)\n    \n    for i in pbar:\n        # 1. Generate Random Parameter Vectors (CPU)\n        # Fast: Only O(N) data. We replicate \'g_gen.generate\' logic manually here\n        # to rely purely on NumPy.\n        rng_key, *subkeys = jax.random.split(rng_key, n_devices + 1)\n        keys_sharded = jnp.array(subkeys)\n\n        # 2. Run FULL Fused Kernel entirely on Accelerator\n        labels_shard, f1_shard, f2_shard, en_shard = p_step(\n            keys_sharded, p_base_energies,\n            p_rho_1_diag, p_rho_2_inter, p_rho_target, p_rho_1_full\n        )\n        \n        # 4. Async Save\n        f1_shard.block_until_ready() # Ensure sync\n        \n        # Copy back to host\n        raw_labels = np.array(labels_shard).reshape((batch_size, label_len))\n        f1_np = np.array(f1_shard).reshape((batch_size,) + f1_shard.shape[2:])\n        f2_np = np.array(f2_shard).reshape((batch_size,) + f2_shard.shape[2:])\n        en_np = np.array(en_shard).reshape((batch_size,) + en_shard.shape[2:])\n        \n        filename = os.path.join(cache_path, f"shard_{i:05d}.npz")\n        \n        def save_task(fname, l, e, f1, f2):\n            try:\n                save_dict = {\'labels\': l}\n                if include_energy: save_dict[\'energy\'] = e[:, None]\n                if input_type == \'rho1\': save_dict[\'features\'] = f1\n                elif input_type == \'rho1+rho2\':\n                    save_dict[\'feat_0\'] = f1; save_dict[\'feat_1\'] = f2\n                else: save_dict[\'features\'] = f2\n                np.savez_compressed(fname, **save_dict)\n            except Exception as e:\n                print(f"Write failed: {e}")\n\n        futures.append(io_pool.submit(save_task, filename, raw_labels, en_np, f1_np, f2_np))\n        \n        # Flow control\n        if len(futures) > 20:\n            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)\n            futures = [f for f in futures if not f.done()]\n\n    concurrent.futures.wait(futures)\n    io_pool.shutdown()\n    \n    return NumpyLoader(cache_path, batch_size)\n\n@ray.remote(num_cpus=1)\ndef ray_solve_chunk(basis, e_chunk, i_chunk, beta, is_thermal, r1_ref, r2_ref, rt_ref):\n    """\n    Remote Ray task that executes the existing sparse kernel.\n    Ray handles the shared memory for the large rho matrices automatically via ObjectRefs.\n    """\n    # Reuse your existing sparse implementation directly\n    return solve_batch_kernel_sp(\n        basis, e_chunk, i_chunk, beta, is_thermal, r1_ref, r2_ref, rt_ref\n    )\n\ndef gen_dataset_sp(h_type, g_init, g_stop, state_type, input_type,\n                include_energy, beta, num_samples=100000,\n                cache_path="./dataset", batch_size=4096, train_batch_size=256,\n                seed=42): \n    \n    # 1. Check existing dataset\n    if check_existing_dataset(cache_path, num_samples, batch_size):\n        return NumpyLoader(cache_path, batch_size)\n\n    # 2. Setup environment\n    if os.path.exists(cache_path): shutil.rmtree(cache_path)\n    os.makedirs(cache_path, exist_ok=True)\n\n    # Initialize Ray with strict resource limits to prevent CPU choking\n    # runtime_env ensures OMP_NUM_THREADS is set BEFORE libraries load in workers\n    if ray.is_initialized(): ray.shutdown()\n    ray.init(\n        ignore_reinit_error=True, \n        include_dashboard=False,\n        runtime_env={"env_vars": {"OMP_NUM_THREADS": "1", \n                                  "MKL_NUM_THREADS": "1", \n                                  "OPENBLAS_NUM_THREADS": "1"}}\n    )\n    \n    print(f"Ray initialized. Generating {num_samples} samples...")\n\n    # 3. Prepare Static Data (Zero-Copy)\n    s_rho_1 = rho_1_arrays\n    \n    # Select Interaction Operator\n    if h_type in [\'blockgen\', \'blockgensimp\', \'randomenerg\']:\n        s_rho_2_inter = rho_2_block_arrays\n    else:\n        # Default to kkbar (pairing) if available\n        s_rho_2_inter = rho_2_kkbar_arrays if \'rho_2_kkbar_arrays\' in globals() else rho_2_arrays\n        \n    # Select Target Operator\n    if \'rho2block\' in input_type: s_rho_target = rho_2_block_arrays\n    elif \'rho2kkbar\' in input_type: s_rho_target = rho_2_kkbar_arrays\n    else: s_rho_target = s_rho_2_inter\n\n    basis_ref = ray.put(basis)\n    r1_ref = ray.put(s_rho_1)\n    r2_ref = ray.put(s_rho_2_inter)\n    rt_ref = ray.put(s_rho_target)\n    \n    # 4. Generator Setup\n    g_gen = GGenerator(basis, h_type, batch_size, g_init=g_init, g_stop=g_stop)\n    \n    levels = np.arange(0, N_ELEC) - N_ELEC // 2 + 0.5\n    is_thermal = (state_type == \'thermal\')\n    pair_jitter = np.linspace(-1e-4, 1e-4, M_PAIRS)\n    levels_jittered = levels + pair_jitter\n    base_energies_np = np.repeat(levels_jittered, 2).astype(np.float32) / SCALE_FACTOR\n    U_ENERGY_SEED = np.array([base_energies_np for _ in range(0, GPU_BATCH_SIZE)])\n\n    # 5. Main Processing Loop\n    rng_key = jax.random.PRNGKey(seed)\n    num_batches = math.ceil(num_samples / batch_size)\n    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)\n    \n    # Micro-batch size: small chunks keep workers responsive and RAM usage low\n    CHUNK_SIZE = 64 \n    \n    pbar = tqdm(range(num_batches), desc="Processing")\n    \n    for i in pbar:\n        # A. Generate Parameters (Main Process)\n        rng_key, subkey = jax.random.split(rng_key)\n        _, labels = g_gen.generate(subkey)\n        \n        # JAX -> Numpy conversion for Ray transport\n        labels_np = np.array(labels)\n        rec_out = g_gen.reconstruct(labels_np)\n\n        # Handle randomenerg tuple return\n        if h_type == \'randomenerg\':\n            e_vals, i_vals = rec_out\n            if e_vals.shape[1] * 2 == basis.d:\n                e_vals = jnp.repeat(e_vals, 2, axis=1)\n        else:\n            e_vals = np.tile(base_energies_np, (len(labels_np), 1))\n            i_vals = rec_out\n\n        e_vals = np.array(e_vals)\n        i_vals = np.array(i_vals)\n\n        # B. Dispatch to Ray (Scatter)\n        futures = []\n        \n        # Split the large file-batch into worker-friendly micro-chunks\n        for start in range(0, len(labels), CHUNK_SIZE):\n            end = min(start + CHUNK_SIZE, len(labels))\n            \n            # Pass REFERENCES for static data, NUMPY ARRAYS for dynamic data\n            fut = ray_solve_chunk.remote(\n                basis_ref,\n                e_vals[start:end],\n                i_vals[start:end],\n                beta,\n                is_thermal,\n                r1_ref,\n                r2_ref,\n                rt_ref\n            )\n            futures.append(fut)\n            \n        # C. Gather Results (Blocking)\n        try:\n            results = ray.get(futures)\n        except Exception as e:\n            print(f"Ray Task Error: {e}")\n            continue\n\n        # D. Aggregate & Save\n        f1_list, f2_list, en_list = zip(*results)\n        \n        f1_arr = np.concatenate(f1_list, axis=0)\n        f2_arr = np.concatenate(f2_list, axis=0)\n        en_arr = np.concatenate(en_list, axis=0)\n        \n        # Helper to reshape flattened features to CNN format\n        def reshape_feat(f):\n            if f.ndim == 2: # (B, Flat) -> (B, D, D, 1)\n                d = int(np.sqrt(f.shape[1]))\n                return f.reshape(-1, d, d, 1)\n            elif f.ndim == 3: # (B, D, D) -> (B, D, D, 1)\n                return f[..., None]\n            return f\n\n        f1_save = reshape_feat(f1_arr)\n        f2_save = reshape_feat(f2_arr)\n        en_save = en_arr[:, None] if en_arr.ndim == 1 else en_arr\n        \n        fname = os.path.join(cache_path, f"shard_{i:05d}.npz")\n        \n        def save_task(fn, l, e, f1, f2):\n            d = {\'labels\': l}\n            if include_energy: d[\'energy\'] = e\n            if input_type == \'rho1\': d[\'features\'] = f1\n            elif input_type == \'rho1+rho2\': d[\'feat_0\'] = f1; d[\'feat_1\'] = f2\n            else: d[\'features\'] = f2\n            np.savez_compressed(fn, **d)\n\n        io_pool.submit(save_task, fname, labels_np, en_save, f1_save, f2_save)\n\n    io_pool.shutdown()\n    ray.shutdown()\n    \n    return NumpyLoader(cache_path, batch_size)\n')


# ### CNN architecture

# #### Model definition

# In[26]:


get_ipython().run_cell_magic('px', '', 'class ResMLPBlock(nn.Module):\n    hidden_dim: int\n\n    @nn.compact\n    def __call__(self, x, training: bool = True):\n        shortcut = x\n        x = nn.LayerNorm()(x)\n        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)\n        x = nn.gelu(x)\n        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros_init())(x)\n        return shortcut + x\n\nclass DeepResMLP(nn.Module):\n    """\n    Flattened Deep Residual MLP.\n    Provides a unique parameter pathway for every orbital coordinate, avoiding\n    the permutation bias of Transformers and the translation bias of CNNs.\n    """\n    label_size: int\n    res: int = 2\n    include_energy: bool = True\n\n    @nn.compact\n    def __call__(self, x, energy=None, training: bool = True):\n        b, m, _, _ = x.shape\n        x_mat = x.squeeze(-1)\n        \n        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))\n                    \n        r, c = jnp.triu_indices(m)\n        x_flat = x_sym[:, r, c] \n        \n        mask = jnp.where(r == c, 1.0, jnp.sqrt(2.0))\n        x_flat = x_flat * mask\n                \n        if self.include_energy and energy is not None:\n            if energy.ndim == 1: energy = energy[:, None]\n            energy_scaled = energy / N_ELEC  \n            \n            e_proj = nn.Dense(8)(energy_scaled)\n            e_proj = nn.gelu(e_proj)\n            x_flat = jnp.concatenate([x_flat, e_proj], axis=-1)\n            \n        hidden_dim = 256 * self.res\n        \n        x = nn.Dense(hidden_dim)(x_flat)\n        x = nn.gelu(x)\n        \n        for _ in range(8):\n            x = ResMLPBlock(hidden_dim)(x, training=training)\n            \n        x = nn.LayerNorm()(x)\n        x = nn.Dense(hidden_dim // 2)(x)\n        x = nn.gelu(x)\n        \n        out = nn.Dense(\n            self.label_size,\n            kernel_init=nn.initializers.normal(stddev=1e-4),\n            bias_init=nn.initializers.constant(0.55)\n        )(x)\n        return out\n')


# In[27]:


get_ipython().run_cell_magic('px', '', 'class ResBlock(nn.Module):\n    """Residual MLP Block: x + Dense(GELU(Dense(LayerNorm(x))))"""\n    hidden_dim: int\n\n    @nn.compact\n    def __call__(self, x):\n        shortcut = x\n        x = nn.LayerNorm()(x)\n        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)\n        x = nn.gelu(x)\n        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros_init())(x)\n        return shortcut + x\n\nclass CoordResMLP(nn.Module):\n    """\n    Flax implementation of the Coordinate-Injected ResMLP.\n    \n    Params:\n        label_size: Output dimension (number of Hamiltonian parameters).\n        res: Resolution multiplier for CNN width (default=1).\n        include_energy: Whether to inject scalar energy input.\n    """\n    label_size: int\n    res: int = 1\n    include_energy: bool = True\n\n    @nn.compact\n    def __call__(self, x, energy=None, training: bool = True):\n        # x shape: (Batch, H, W, 1)\n        b, h, w, c = x.shape\n\n        # 1. Coordinate Injection\n        # Normalized grids [-1, 1]\n        i_c = jnp.linspace(-1.0, 1.0, h)\n        j_c = jnp.linspace(-1.0, 1.0, w)\n        ii, jj = jnp.meshgrid(i_c, j_c, indexing=\'ij\')\n        \n        # Expand and concat: (B, H, W, 1) + (B, H, W, 2) -> (B, H, W, 3)\n        coords = jnp.stack([ii, jj], axis=-1)\n        coords = jnp.tile(coords[None, ...], (b, 1, 1, 1))\n        x = jnp.concatenate([x, coords], axis=-1)\n\n        # 2. CNN Feature Extraction\n        # Width scaled by self.res\n        x = nn.Conv(features=64 * self.res, kernel_size=(3, 3), padding=\'SAME\')(x)\n        x = nn.BatchNorm(use_running_average=not training)(x)\n        x = nn.gelu(x)\n\n        x = nn.Conv(features=128 * self.res, kernel_size=(3, 3), padding=\'SAME\')(x)\n        x = nn.BatchNorm(use_running_average=not training)(x)\n        x = nn.gelu(x)\n\n        x = x.reshape((b, -1)) # Flatten\n\n        # 3. Energy Injection\n        if self.include_energy:\n            if energy is None: raise ValueError("Model expects energy input")\n            # Ensure energy is (Batch, 1)\n            if energy.ndim == 1: energy = energy[:, None]\n            energy = energy / N_ELEC\n            \n            e_vec = nn.Dense(64 * self.res)(energy)\n            e_vec = nn.gelu(e_vec)\n            x = jnp.concatenate([x, e_vec], axis=-1)\n\n        # 4. ResMLP Head\n        hidden_dim = 1024 * self.res\n        x = nn.Dense(hidden_dim)(x)\n        x = nn.gelu(x)\n\n        for _ in range(3):\n            x = ResBlock(hidden_dim)(x)\n\n        x = nn.LayerNorm()(x)\n        return nn.Dense(self.label_size)(x)\n')


# In[28]:


get_ipython().run_cell_magic('px', '', 'import jax.numpy as jnp\nimport flax.linen as nn\n\nclass MatrixInteractionBlock(nn.Module):\n    """\n    Pre-Norm Message Passing Block with Bare-RDM Injection and Left/Right Scattering.\n    """\n    hidden_dim: int\n\n    @nn.compact\n    def __call__(self, h, e, bare_rdm):\n        B, M, _ = h.shape\n        \n        # =================================================================\n        # 1. PRE-LAYERNORM (Crucial for unblocking deep gradient flow)\n        # =================================================================\n        h_norm = nn.LayerNorm()(h)\n        e_norm = nn.LayerNorm()(e)\n        \n        # =================================================================\n        # 2. NODE UPDATE (Mean + Max Aggregation)\n        # =================================================================\n        e_mean = jnp.mean(e_norm, axis=2)\n        e_max = jnp.max(e_norm, axis=2) # Max-pooling catches dominant random interactions\n        h_in = jnp.concatenate([h_norm, e_mean, e_max], axis=-1)\n        \n        h_update = nn.Dense(self.hidden_dim)(h_in)\n        h_update = nn.gelu(h_update)\n        h_update = nn.Dense(h.shape[-1], kernel_init=nn.initializers.zeros_init())(h_update)\n        \n        h = h + h_update # Clean UNNORMALIZED residual stream\n        \n        # =================================================================\n        # 3. EDGE UPDATE & SCATTERING LOOP\n        # =================================================================\n        h_norm2 = nn.LayerNorm()(h)\n        h_i = jnp.broadcast_to(h_norm2[:, :, None, :], (B, M, M, h.shape[-1]))\n        h_j = jnp.broadcast_to(h_norm2[:, None, :, :], (B, M, M, h.shape[-1]))\n        \n        h_sum = h_i + h_j\n        h_prod = h_i * h_j\n        \n        # Project into Left/Right bases to prevent feature rank collapse\n        e_left = nn.Dense(self.hidden_dim // 2)(e_norm)\n        e_right = nn.Dense(self.hidden_dim // 2)(e_norm)\n        \n        # Native matrix multiplication over the hidden interaction channels\n        contracted = jnp.einsum(\'bikf,bkjf->bijf\', e_left, e_right) / jnp.sqrt(M)\n        \n        # =================================================================\n        # 4. BARE RDM INJECTION (The Anchor)\n        # =================================================================\n        # Injecting `bare_rdm` completely eliminates graph over-smoothing.\n        e_in = jnp.concatenate([\n            e_norm, h_sum, h_prod, contracted, bare_rdm[..., None]\n        ], axis=-1)\n        \n        # Deeper 2-Layer MLP for highly non-linear matrix inversions\n        e_update = nn.Dense(self.hidden_dim)(e_in)\n        e_update = nn.gelu(e_update)\n        e_update = nn.Dense(self.hidden_dim)(e_update)\n        e_update = nn.gelu(e_update)\n        e_update = nn.Dense(e.shape[-1], kernel_init=nn.initializers.zeros_init())(e_update)\n        \n        # Force strict Bosonic/Hermitian interaction symmetry\n        e_update = 0.5 * (e_update + jnp.swapaxes(e_update, 1, 2))\n        \n        e = e + e_update \n        \n        return h, e\n\n\nclass PhysicsOrbitalGraphNet(nn.Module):\n    """\n    V2 Topology: Pre-Norm Relational Matrix Graph\n    """\n    label_size: int\n    res: int = 3\n    include_energy: bool = True\n\n    @nn.compact\n    def __call__(self, x, energy=None, training: bool = True):\n        b, m, _, _ = x.shape\n        x_mat = x.squeeze(-1)\n\n        num_triu = (m * (m + 1)) // 2\n        htype_random = (self.label_size == num_triu)\n        \n        # Strict Symmetrization of the input\n        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))\n        \n        # A. NODE INITIALIZATION\n        diag_idx = jnp.arange(m)\n        n_k = x_sym[:, diag_idx, diag_idx][..., None]\n        \n        # Widen embedding slightly for richer single-particle identity\n        orb_emb = self.param(\'orb_emb\', nn.initializers.normal(stddev=0.1), (m, 32))\n        orb_emb_batch = jnp.broadcast_to(orb_emb[None, :, :], (b, m, 32))\n        \n        node_features = [n_k, orb_emb_batch]\n        \n        if self.include_energy and energy is not None:\n            if energy.ndim == 1: energy = energy[:, None]\n            e_ctx = nn.Dense(32)(energy / m)\n            e_ctx = nn.gelu(e_ctx)\n            e_ctx_batch = jnp.broadcast_to(e_ctx[:, None, :], (b, m, 32))\n            node_features.append(e_ctx_batch)\n            \n        h = jnp.concatenate(node_features, axis=-1)\n        h = nn.Dense(128 * self.res)(h)\n        \n        # B. EDGE INITIALIZATION\n        e = jnp.expand_dims(x_sym, -1)\n        e = nn.Dense(128 * self.res)(e)\n        \n        # C. MESSAGE PASSING (Deepened safely due to Pre-Norm)\n        hidden_dim = 256 * self.res\n        for _ in range(5): \n            h, e = MatrixInteractionBlock(hidden_dim)(h, e, bare_rdm=x_sym)\n            \n        # D. PHYSICAL READOUT\n        # Final LayerNorm before readout (Standard requirement for Pre-LN networks)\n        e_final = nn.LayerNorm()(e)\n        \n        out_mat = nn.Dense(\n            1, \n            kernel_init=nn.initializers.normal(stddev=1e-3), # Safer, smaller init\n            bias_init=nn.initializers.constant(0.55) # Match target mean\n        )(e_final).squeeze(-1)\n        \n        out_mat = 0.5 * (out_mat + jnp.swapaxes(out_mat, 1, 2))\n        \n        r, c = jnp.triu_indices(m)\n        out_features = out_mat[:, r, c]\n\n        if htype_random:\n            return out_features\n        else:\n            x_proj = nn.Dense(128 * self.res)(out_features)\n            x_proj = nn.gelu(x_proj)\n            return nn.Dense(\n                self.label_size,\n                kernel_init=nn.initializers.normal(stddev=1e-3),\n                bias_init=nn.initializers.constant(0.55)\n            )(x_proj)\n\n        return \n')


# #### Model Training

# In[29]:


get_ipython().run_cell_magic('px', '', 'from flax.jax_utils import replicate, unreplicate\nfrom functools import partial\nimport scipy.sparse as sp\n\n# =========================================================================\n# Global Precomputations for Physics Losses\n# =========================================================================\nprint("Initializing global physics tensors...")\n\n# A. Operators for the \'eigh\' Variational Loss\nrho_1_np = _ensure_dense(rho_1_arrays)\nif rho_1_np.ndim == 4: \n    d_rho_1_diag_global = jnp.array(np.einsum(\'kknn->kn\', rho_1_np))\nelse: \n    d_rho_1_diag_global = jnp.array(np.diagonal(rho_1_np, axis1=1, axis2=2))\n    \ntarget_inter_np = _ensure_dense(rho_2_kkbar_arrays if \'rho_2_kkbar_arrays\' in globals() else rho_2_arrays)\nd_inter_tensor_global = jnp.array(target_inter_np)\nbase_e_params_global = jnp.array(U_ENERGY_SEED[0])\n\n# B. Operators for the \'gram\' Metric Loss\ndef get_gram_matrix(rho_tensor):\n    """Computes the exact Hilbert-Schmidt Metric Tensor S_ab = Tr(L_a^dag L_b)"""\n    rho_dense = np.asarray(rho_tensor.todense() if hasattr(rho_tensor, \'todense\') else rho_tensor)\n    m1, m2, N_dim, _ = rho_dense.shape\n    \n    # Align transposition to match how two_body_hamiltonian_dense constructs H\n    # G[i, j] multiplies rho_tensor[j, i]\n    L_tensor = rho_dense.transpose((1, 0, 2, 3))\n    L_flat = L_tensor.reshape((m1 * m2, N_dim**2))\n    \n    print(f"Computing exact {m1*m2}x{m1*m2} Gram Matrix S...")\n    Y_csr = sp.csr_matrix(L_flat.astype(np.float64))\n    S_sparse = Y_csr @ Y_csr.T\n    S = S_sparse.toarray()\n    \n    S_norm = S / float(N_dim)\n    \n    return 0.5 * (S_norm + S_norm.T)\n\nS_matrix_np = get_gram_matrix(target_inter_np)\n\n# Export to JAX (using float32 for TPU speed)\nS_tensor_global = jnp.array(S_matrix_np, dtype=jnp.float32)\n\n')


# In[30]:


get_ipython().run_cell_magic('px', '', 'is_master = (jax.process_index() == 0)\n\ndef print_model_summary(model, input_shape, include_energy):\n    if not is_master:\n        return None\n    print("\\n" + "="*80)\n    res_info = f"(res={model.res})" if hasattr(model, \'res\') else ""\n    print(f" Model Summary: {model.__class__.__name__} {res_info}")\n    print("="*80)\n    \n    rng = jax.random.PRNGKey(0)\n    try:\n        dummy_x = jnp.ones(input_shape)\n        dummy_e = jnp.ones((input_shape[0], 1)) if include_energy else None\n        print(model.tabulate(rng, dummy_x, dummy_e, training=False))\n    except Exception as e:\n        print(f"Summary unavailable: {e}")\n        # Fallback to parameter counting if tabulate fails\n        try:\n            dummy_x = jnp.ones(input_shape)\n            dummy_e = jnp.ones((input_shape[0], 1)) if include_energy else None\n            variables = model.init(rng, dummy_x, dummy_e, training=False)\n            flat = flax.traverse_util.flatten_dict(variables[\'params\'], sep=\'/\')\n            total_params = sum(x.size for x in flat.values())\n            print(f"Total Parameters: {total_params:,}")\n        except:\n            pass\n    print("="*80 + "\\n")\n\nclass TrainState(train_state.TrainState):\n    batch_stats: Any\n\ndef create_train_step(loss_type=\'gram\', is_thermal=False, beta=100.0):\n    @partial(jax.pmap, axis_name=\'batch\')\n    def p_train_step(state, bx, be, by, S_tensor, base_e_params, rho_1_diag, inter_tensor): \n        def loss_fn(params):\n            logits, updates = state.apply_fn(\n                {\'params\': params, \'batch_stats\': state.batch_stats},\n                bx, be, training=True, mutable=[\'batch_stats\']\n            )\n            \n            logits_f64 = logits.astype(jnp.float64)\n            by_f64 = by.astype(jnp.float64)\n            \n            H_pred_mat = g_gen.reconstruct(logits_f64)\n            H_true_mat = g_gen.reconstruct(by_f64)\n            \n            base_loss = jnp.mean(jnp.sum(jnp.square(logits_f64 - by_f64), axis=-1))\n\n            if loss_type == \'gram\':\n                B_size = H_pred_mat.shape[0]\n                H_pred_flat = H_pred_mat.reshape((B_size, -1))\n                H_true_flat = H_true_mat.reshape((B_size, -1))\n                delta_H = H_pred_flat - H_true_flat\n                \n                # 1. PHYSICAL LOSS\n                phys_variances = jnp.einsum(\'bi,ij,bj->b\', delta_H, S_tensor, delta_H)\n                phys_loss = jnp.mean(phys_variances)\n                \n                # 2. TIKHONOV REGULARIZATION\n                norm_penalty = jnp.mean(jnp.sum(jnp.square(delta_H), axis=-1))\n\n                # 3. TRACE LOSS (Gauge Fixing)\n                # GGenerator fixes the mean of the diagonal to 0.55. \n                # We penalize arbitrary G -> G + cI shifts by matching the diagonal mean.\n                pred_diag_mean = jnp.mean(jnp.diagonal(H_pred_mat, axis1=1, axis2=2), axis=1)\n                true_diag_mean = jnp.mean(jnp.diagonal(H_true_mat, axis1=1, axis2=2), axis=1)\n                trace_loss = jnp.mean(jnp.square(pred_diag_mean - true_diag_mean))\n                \n                # Add trace_loss to the total loss\n                total_loss = 1.0 * phys_loss + 1.0 * trace_loss + 1e-2 * norm_penalty\n                                \n                return total_loss.astype(jnp.float32), updates\n\n            if loss_type == \'rdm\':\n                B = bx.shape[0]\n                e_params = jnp.tile(base_e_params.astype(jnp.float32), (B, 1))\n                \n                # 1. Reconstruct full H strictly in float32\n                H_batch = two_body_hamiltonian_dense(\n                    e_params, H_pred_mat.astype(jnp.float32), \n                    d_rho_1_diag_global.astype(jnp.float32), \n                    d_inter_tensor_global.astype(jnp.float32)\n                )\n\n                effective_beta = beta if is_thermal else 100.0\n                state_rho_pred, _ = eigensolve_and_build_rho(H_batch, effective_beta)\n                \n                # 3. Extract predicted RDM using the Einsum trace\n                target_op = d_inter_tensor_global.astype(jnp.float32)\n                orig_shape = target_op.shape[:-2]\n                ops_flat = target_op.reshape(-1, target_op.shape[-2], target_op.shape[-1])\n                \n                rdm_pred_flat = jnp.einsum(\'bnm,kmn->bk\', state_rho_pred, ops_flat)\n                rdm_pred = rdm_pred_flat.reshape((B,) + orig_shape + (1,))\n                \n                # 1. RDM OBSERVABLE ERROR (The Zero-Temperature Physics)\n                rdm_mse = jnp.mean(jnp.square(rdm_pred.real - bx.astype(jnp.float32).real))\n                \n                # 2. GAUGE FIX (Trace matching to break global shift degeneracy)\n                pred_diag_mean = jnp.mean(jnp.diagonal(H_pred_mat, axis1=1, axis2=2), axis=1)\n                true_diag_mean = jnp.mean(jnp.diagonal(H_true_mat, axis1=1, axis2=2), axis=1)\n                trace_loss = jnp.mean(jnp.square(pred_diag_mean - true_diag_mean))\n                \n                # 3. TIE-BREAKER (Tikhonov Regularizer)\n                # We use a weak base_loss (1e-2) as a gentle anchor. Out of the infinite \n                # compatible Hamiltonians that share this GS, this pulls the network \n                # towards the minimum-norm solution matching your dataset distribution.\n                total_loss = (rdm_mse * 100.0) + (trace_loss * 1.0) + (base_loss * 1e-2)\n                \n                return total_loss.astype(jnp.float32), updates\n                \n            return base_loss.astype(jnp.float32), updates\n\n        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)\n        \n        # Cross-TPU Gradient Synchronization\n        grads = jax.lax.pmean(grads, axis_name=\'batch\')\n        loss = jax.lax.pmean(loss, axis_name=\'batch\')\n        \n        new_state = state.apply_gradients(grads=grads)\n        new_batch_stats = updates.get(\'batch_stats\', state.batch_stats)\n        return new_state.replace(batch_stats=new_batch_stats), loss\n\n    return p_train_step\n\n@jax.jit\ndef eval_step(state, bx, be, by):\n    logits = state.apply_fn(\n        {\'params\': state.params, \'batch_stats\': state.batch_stats},\n        bx, be, training=False\n    )\n    return logits\n\ndef train_model(dataset, label_size, input_type, M_PAIRS, include_energy, \n                total_samples, batch_size, epochs=20, res=1, loss_type=\'gram\',\n                is_thermal=False, beta=100.0):\n    \n    dim = M_PAIRS**2 if input_type == \'rho2block\' else M_PAIRS \n    input_shape = (1, dim, dim, 1)\n    \n    # ----------------------------------------------------\n    # Model topologies\n    #model = DeepResMLP(label_size, res=res, include_energy=include_energy)\n    #model = CoordResMLP(label_size, res=res, include_energy=include_energy)\n    model = PhysicsOrbitalGraphNet(label_size, res=res, include_energy=include_energy)\n    # ----------------------------------------------------\n    \n    print_model_summary(model, input_shape, include_energy)\n    \n    rng = jax.random.PRNGKey(42)\n    dummy_x = jnp.ones(input_shape)\n    dummy_e = jnp.ones((1, 1)) if include_energy else None\n    variables = model.init(rng, dummy_x, dummy_e, training=False)\n    \n    steps_per_epoch = total_samples // batch_size\n    total_steps = steps_per_epoch * epochs\n    warmup_steps = min(3000, int(0.05 * total_steps)) \n    \n    sched = optax.warmup_cosine_decay_schedule(\n        init_value=1e-6,       \n        peak_value=3e-4,      \n        warmup_steps=warmup_steps,\n        decay_steps=total_steps,\n        end_value=5e-6         \n    )\n\n    tx = optax.chain(\n        optax.clip_by_global_norm(1.0), \n        optax.adamw(sched, weight_decay=1e-4) \n    )\n\n    state = TrainState.create(\n        apply_fn=model.apply, params=variables[\'params\'], tx=tx, \n        batch_stats=variables.get(\'batch_stats\', {})\n    )\n    \n    # === SETUP FOR 8 TPUS ===\n    n_devices = jax.local_device_count()\n    device_batch = batch_size // n_devices\n    \n    p_state = replicate(state)\n    p_S_tensor = replicate(S_tensor_global)\n    p_base_e_params = replicate(base_e_params_global)\n    p_rho_1_diag = replicate(d_rho_1_diag_global)\n    p_inter_tensor = replicate(d_inter_tensor_global)\n    \n    # Compile the specific step based on the requested loss\n    p_train_step = create_train_step(loss_type, is_thermal, beta)\n    \n    history = []\n    print(f"Starting Training | Loss Mode: {loss_type.upper()} | TPUs: {n_devices}")\n    \n    for ep in range(epochs):\n        loss_acc, steps = 0.0, 0\n        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep+1}", leave=False, disable=not is_master)\n        \n        for big_x, big_e, big_y in dataset:\n            chunk_size = big_x.shape[0]\n            for start in range(0, chunk_size, batch_size):\n                end = min(start + batch_size, chunk_size)\n                if (end - start) != batch_size: continue\n\n                bx = jnp.array(big_x[start:end]).reshape(n_devices, device_batch, *big_x.shape[1:])\n                by = jnp.array(big_y[start:end]).reshape(n_devices, device_batch, *big_y.shape[1:])\n                be = jnp.array(big_e[start:end]).reshape(n_devices, device_batch, *big_e.shape[1:]) if include_energy else None\n                \n                # ---> P_STATE IS OVERWRITTEN ITERATIVELY <---\n                p_state, p_loss = p_train_step(\n                    p_state, bx, be, by, p_S_tensor, p_base_e_params, p_rho_1_diag, p_inter_tensor\n                )\n                \n                loss_val = np.array(p_loss)[0].item()\n                loss_acc += loss_val\n                steps += 1\n                pbar.update(1)\n                pbar.set_postfix(loss=f"{loss_val:.5f}")\n                \n                if steps >= steps_per_epoch: break\n            if steps >= steps_per_epoch: break\n        \n        pbar.close()\n        avg_loss = loss_acc / steps if steps > 0 else 0\n        history.append(avg_loss)\n        if is_master:\n            print(f"Epoch {ep+1} | Loss: {avg_loss:.6f}")\n        \n    return unreplicate(p_state), history\n\ndef evaluate_universal_metrics(state, dataset, include_energy, S_matrix_np, gpu_batch_size=256, is_thermal=False, beta=100.0):\n    """\n    Evaluates the model across Legacy Parameter Error, \n    Universal Physical Operator Distance (Gauge Invariant),\n    and directly via the Exact RDM Forward Distance.\n    """\n    print("\\n" + "="*60)\n    print(" RUNNING UNIVERSAL PHYSICAL EVALUATION")\n    print("="*60)\n    \n    all_preds, all_actual, all_rdm_inputs = [], [], []\n    count = 0\n    \n    for batch_x, batch_e, batch_y in dataset:\n        if count >= 20: break \n        \n        chunk_size = batch_x.shape[0]\n        \n        for start in range(0, chunk_size, gpu_batch_size):\n            end = min(start + gpu_batch_size, chunk_size)\n            \n            bx = jnp.array(batch_x[start:end])\n            by = jnp.array(batch_y[start:end])\n            be = jnp.array(batch_e[start:end]) if include_energy else None\n            \n            logits = eval_step(state, bx, be, by)\n            \n            if logits.ndim == 3:  \n                logits = logits[0]\n                by = by[0]\n                bx = bx[0]\n            \n            # Prevent single precision NaNs from blowing up the evaluation\n            logits = jnp.nan_to_num(logits)\n                \n            all_preds.append(np.array(logits))\n            all_actual.append(np.array(by))\n            all_rdm_inputs.append(np.array(bx))\n            \n        count += 1\n        \n    if not all_preds: return 0.0\n    \n    pred = np.concatenate(all_preds)\n    actual = np.concatenate(all_actual)\n    true_rdms = np.concatenate(all_rdm_inputs)\n    \n    # Chunk the JAX Reconstruction\n    pred_mat_list = []\n    actual_mat_list = []\n    \n    for start in range(0, len(pred), gpu_batch_size):\n        end = min(start + gpu_batch_size, len(pred))\n        pred_mat_list.append(np.array(g_gen.reconstruct(pred[start:end])))\n        actual_mat_list.append(np.array(g_gen.reconstruct(actual[start:end])))\n        \n    pred_mat = np.concatenate(pred_mat_list, axis=0)\n    actual_mat = np.concatenate(actual_mat_list, axis=0)\n    \n    # Flatten them into (Batch, M*M) vectors for distance calculation\n    # CRITICAL: Upcast to float64 to prevent np.einsum overflow!\n    pred_flat = pred_mat.reshape((pred_mat.shape[0], -1)).astype(np.float64)\n    actual_flat = actual_mat.reshape((actual_mat.shape[0], -1)).astype(np.float64)\n    \n    # STATISTICALLY RIGOROUS BASELINE: Predict the global mean of the batch\n    mean_mat = np.mean(actual_flat, axis=0, keepdims=True)\n    baseline_flat = np.repeat(mean_mat, actual_flat.shape[0], axis=0)\n\n    # --- 1. Legacy Metric ---\n    param_error = np.mean(np.linalg.norm(pred_flat - actual_flat, axis=1))\n    param_rand = np.mean(np.linalg.norm(baseline_flat - actual_flat, axis=1))\n    \n    # --- 2. Universal Physical Metric ---\n    delta_H = pred_flat - actual_flat\n    phys_variances = np.einsum(\'bi,ij,bj->b\', delta_H, S_matrix_np.astype(np.float64), delta_H)\n    phys_error = np.mean(np.sqrt(np.maximum(phys_variances, 0.0) + 1e-12))\n    \n    delta_H_rand = baseline_flat - actual_flat\n    rand_variances = np.einsum(\'bi,ij,bj->b\', delta_H_rand, S_matrix_np.astype(np.float64), delta_H_rand)\n    rand_phys_error = np.mean(np.sqrt(np.maximum(rand_variances, 0.0) + 1e-12))\n    \n    # --- 3. RDM OBSERVABLE ERROR (End-to-End) ---\n    print("   Reconstructing RDM observables for baseline comparison...")\n    \n    baseline_labels = np.repeat(np.mean(actual, axis=0, keepdims=True), actual.shape[0], axis=0)\n    \n    @jax.jit\n    def get_rdm_observable(interaction_params):\n        B = interaction_params.shape[0]\n        e_params = jnp.tile(base_e_params_global.astype(jnp.float32), (B, 1))\n        \n        H_batch = two_body_hamiltonian_dense(\n            e_params, interaction_params.astype(jnp.float32), \n            d_rho_1_diag_global.astype(jnp.float32), \n            d_inter_tensor_global.astype(jnp.float32)\n        )\n        \n        effective_beta = beta if is_thermal else 100.0\n        state_rho, _ = eigensolve_and_build_rho(H_batch, effective_beta)\n        \n        target_op = d_inter_tensor_global.astype(jnp.float32)\n        orig_shape = target_op.shape[:-2]\n        ops_flat = target_op.reshape(-1, target_op.shape[-2], target_op.shape[-1])\n        \n        rdm_flat = jnp.einsum(\'bnm,kmn->bk\', state_rho, ops_flat)\n        return rdm_flat.reshape((B,) + orig_shape + (1,)).real\n\n    pred_rdms_list = []\n    base_rdms_list = []\n    \n    # Reconstruct the interaction matrices of the mean baseline\n    baseline_mat_list = []\n    for start in range(0, len(baseline_labels), gpu_batch_size):\n        end = min(start + gpu_batch_size, len(baseline_labels))\n        baseline_mat_list.append(np.array(g_gen.reconstruct(baseline_labels[start:end])))\n    baseline_mat = np.concatenate(baseline_mat_list, axis=0)\n    \n    for start in range(0, len(pred_mat), gpu_batch_size):\n        end = min(start + gpu_batch_size, len(pred_mat))\n        \n        # Pass the reconstructed matrix (interaction_params) to the solver\n        pr = np.array(get_rdm_observable(pred_mat[start:end]))\n        br = np.array(get_rdm_observable(baseline_mat[start:end]))\n        \n        pred_rdms_list.append(pr)\n        base_rdms_list.append(br)\n        \n    pred_rdms = np.concatenate(pred_rdms_list, axis=0)\n    base_rdms = np.concatenate(base_rdms_list, axis=0)\n    \n    # Clean up trailing channel dimensions for distance calculation\n    if true_rdms.ndim == 4 and true_rdms.shape[-1] == 1:\n        true_rdms = true_rdms.squeeze(-1)\n    if pred_rdms.ndim == 4 and pred_rdms.shape[-1] == 1:\n        pred_rdms = pred_rdms.squeeze(-1)\n    if base_rdms.ndim == 4 and base_rdms.shape[-1] == 1:\n        base_rdms = base_rdms.squeeze(-1)\n        \n    axis_tuple = tuple(range(1, true_rdms.ndim))\n    rdm_model_error = np.mean(np.sqrt(np.mean((pred_rdms - true_rdms)**2, axis=axis_tuple)))\n    rdm_rand_error = np.mean(np.sqrt(np.mean((base_rdms - true_rdms)**2, axis=axis_tuple)))\n\n    # ---------------------------------------------------------\n    # REPORTING\n    # ---------------------------------------------------------\n    print(f"1. NAIVE PARAMETER ERROR (Susceptible to Gauge Symmetries)")\n    print(f"   Model Raw Error : {param_error:.6f}")\n    print(f"   Random Baseline : {param_rand:.6f}")\n    print(f"   Ratio           : {param_rand / param_error if param_error > 1e-9 else 0.0:.2f}x\\n")\n    \n    print(f"2. UNIVERSAL PHYSICAL ERROR (Exact Operator Distance)")\n    print(f"   Model Phys Error: {phys_error:.6f}")\n    print(f"   Random Baseline : {rand_phys_error:.6f}")\n    ratio = rand_phys_error / phys_error if phys_error > 1e-9 else 0.0\n    print(f"   Improvement     : {ratio:.2f}x better than random\\n")\n    \n    print(f"3. RDM OBSERVABLE ERROR (End-to-End Prediction)")\n    print(f"   Model RDM Error : {rdm_model_error:.6f}")\n    print(f"   Random Baseline : {rdm_rand_error:.6f}")\n    rdm_ratio = rdm_rand_error / rdm_model_error if rdm_model_error > 1e-9 else 0.0\n    print(f"   Improvement     : {rdm_ratio:.2f}x better than random")\n    print("="*60 + "\\n")\n    \n    return ratio\n\ndef predict_and_load(loader, state, num_samples, gpu_batch_size=256, include_energy=True):\n    """\n    Iterates through the NumpyLoader, performs JAX inference in mini-batches, \n    and accumulates results into numpy arrays. \n    """\n    print(f"Loading and predicting on {num_samples} validation samples...")\n    \n    all_rdms = []\n    all_energies = []\n    all_g_true = []\n    all_g_pred = []\n    \n    count = 0\n    \n    # Iterate over the loader (which yields large chunks, e.g., 4096)\n    # We create a new iterator to ensure we start fresh if needed, \n    # but reusing the existing \'dataset\' object is fine if it resets automatically.\n    for big_x, big_e, big_y in loader:\n        if count >= num_samples:\n            break\n            \n        chunk_size = big_x.shape[0]\n        \n        # Inner loop: Slice large chunk into GPU-friendly mini-batches\n        for start in range(0, chunk_size, gpu_batch_size):\n            end = min(start + gpu_batch_size, chunk_size)\n            \n            # Prepare Inputs (JAX Arrays)\n            bx = jnp.array(big_x[start:end])\n            by = jnp.array(big_y[start:end])\n            be = jnp.array(big_e[start:end]) if include_energy else None\n            \n            # Inference (Flax)\n            # eval_step signature: (state, bx, be, by) -> logits\n            logits = eval_step(state, bx, be, by)\n            \n            # Store Results (Host Memory / Numpy)\n            all_rdms.append(np.array(bx)) \n            # Handle energy potentially being None or shaped\n            if be is not None:\n                all_energies.append(np.array(be))\n            else:\n                all_energies.append(np.zeros((len(bx), 1)))\n                \n            all_g_true.append(np.array(by))\n            all_g_pred.append(np.array(logits))\n            \n        count += chunk_size\n\n    if not all_rdms:\n        raise ValueError("Dataset loader returned no data.")\n\n    # Concatenate and Slice to exact num_samples\n    rdm_arr = np.concatenate(all_rdms, axis=0)[:num_samples]\n    energy_arr = np.concatenate(all_energies, axis=0)[:num_samples]\n    g_true_arr = np.concatenate(all_g_true, axis=0)[:num_samples]\n    g_pred_arr = np.concatenate(all_g_pred, axis=0)[:num_samples]\n    \n    # Squeeze last dimension of RDMs if necessary (N, N, 1) -> (N, N)\n    if rdm_arr.ndim == 4 and rdm_arr.shape[-1] == 1:\n        rdm_arr = rdm_arr.squeeze(-1)\n        \n    return rdm_arr, energy_arr, g_true_arr, g_pred_arr\n\ndef rho_reconstruction(g_preds, h_type, state_type, input_type, basis,\n                       u_energy_seed, beta, rho_1_dense,\n                       rho_2_arrays, rho_2_kkbar, rho_2_block):\n    """\n    Reconstructs the Density Matrices (RDM) from the predicted Hamiltonian parameters.\n    Uses the JAX kernels (solve_batch_kernel) defined earlier.\n    """\n    batch_size = len(g_preds)\n    print(f"Reconstructing physics for {batch_size} predictions...")\n    \n    # 1. Prepare Base Energies (B, D_SP)\n    base_energies = u_energy_seed[0] \n    e_params = jnp.tile(base_energies, (batch_size, 1))\n    \n    # 2. Prepare Operators (Move to JAX GPU)\n    # Ensure dense arrays\n    d_rho_1_full = jnp.array(rho_1_dense)\n    \n    # Extract diagonal: (D, D) -> (D,) or (K, K, N, N) -> (K, N)\n    if rho_1_dense.ndim == 3:\n        d_rho_1_diag = jnp.array(np.diagonal(rho_1_dense, axis1=1, axis2=2))\n    else:\n        d_rho_1_diag = jnp.array(np.einsum(\'kknn->kn\', rho_1_dense))\n    \n    # Select Interaction Tensor based on problem type\n    if h_type in [\'blockgen\', \'blockgensimp\', \'randomenerg\']:\n         inter_tensor = rho_2_block\n    else:\n         inter_tensor = rho_2_kkbar\n            \n    d_inter_tensor = jnp.array(_ensure_dense(inter_tensor))\n    d_rho_target = d_inter_tensor \n    \n    # 3. Reconstruct H parameters (Labels -> Matrix/Vector)\n    # Use temporary generator\n    temp_gen = GGenerator(basis, h_type, batch_size)\n    d_labels = jnp.array(g_preds)\n    \n    # JIT-compiled reconstruction step\n    @jax.jit\n    def _recon_step(lbls, es):\n        rec_out = temp_gen.reconstruct(lbls)\n        \n        if h_type == \'randomenerg\':\n            e_vals, i_vals = rec_out\n            if e_vals.shape[1] * 2 == d_rho_1_diag.shape[0]:\n                 e_vals = jnp.repeat(e_vals, 2, axis=1)\n        else:\n            e_vals = es\n            i_vals = rec_out\n\n        # 4. Solve for RDMs\n        is_thermal = (state_type == \'thermal\')\n        _, f2, _ = solve_batch_kernel(\n            e_vals, i_vals, beta, is_thermal,\n            d_rho_1_diag, d_inter_tensor, d_rho_target, d_rho_1_full\n        )\n        return f2\n\n    # Run in chunks to match memory constraints\n    chunk = 256\n    results = []\n    for i in range(0, batch_size, chunk):\n        end = min(i + chunk, batch_size)\n        res = _recon_step(d_labels[i:end], e_params[i:end])\n        results.append(np.array(res))\n\n    return np.concatenate(results, axis=0)\n\n')


# ## Main

# Config

# In[31]:


get_ipython().run_cell_magic('px', '', '# --- Configuration ---\nnum_samples = 5000000\nh_type: ValidHType = \'random\'\ng_init = 0.1\ng_stop = 1.0\nstate_type: ValidStateType = \'thermal\' \ninput_type: ValidInputType = \'rho2kkbar\' \ninclude_energy = True\nnum_epochs = 25\nres = 3\ntrain_batch_size = GPU_BATCH_SIZE\n\n# Initialize GGenerator for the configuration\ng_gen = GGenerator(basis, h_type, GPU_BATCH_SIZE, g_init=g_init, g_stop=g_stop)\nlabel_size = g_gen.label_size()\n\n# Workers config\nnum_hosts = jax.process_count()\nhost_id = jax.process_index()\nlocal_num_samples = num_samples // num_hosts\nlocal_batch_size = train_batch_size # // num_hosts\n\nprint(f\'\\nConfiguration: h_type={h_type}, input_type={input_type}, Label size={label_size}\')\n\n# Constants \nD2_SIZE = math.comb(D_SP, 2)\nis_thermal = False if state_type == \'gs\' else True\n\n# =====================================================================\n# Dataset generation and training\n# =====================================================================\n\n\n# Generate dataset\ndataset = gen_dataset(h_type, g_init, g_stop, state_type, input_type, include_energy, BETA, \n                        local_num_samples, cache_path=f"/home/agus/workspace/dataset_random_w{host_id}", batch_size=2048, \n                        train_batch_size=GPU_BATCH_SIZE, seed=42 + host_id)\n\nfinal_state, hist = train_model(\n        dataset=dataset, \n        label_size=g_gen.label_size(), \n        input_type=input_type, \n        M_PAIRS=M_PAIRS, \n        include_energy=True, \n        total_samples=local_num_samples, \n        batch_size=local_batch_size, \n        epochs=num_epochs, \n        res=res,\n        loss_type=\'gram\',\n        is_thermal=is_thermal,\n        beta=BETA)\n\n# Evaluate\nval_dataset = gen_dataset(\n    h_type, g_init, g_stop, state_type, input_type, include_energy, BETA, \n    num_samples=int(0.05 * num_samples),\n    cache_path="/home/agus/workspace/dataset_val",   \n    batch_size=2048, \n    train_batch_size=GPU_BATCH_SIZE,\n    seed=43 + host_id\n)\n\nevaluate_universal_metrics(\n    final_state, \n    val_dataset, \n    include_energy=True, \n    S_matrix_np=S_matrix_np,\n    is_thermal=is_thermal,\n    beta=BETA  \n)\n')


# # Alternative methods

# Execution configuration

# In[16]:


get_ipython().run_cell_magic('px', '', '\nfrom flax import serialization\nimport numpy as np\nimport os\n\ndef save_model_and_history(state, hist, save_dir="./home/agus/workspace/model"):\n    """Securely serializes the TrainState and history to disk."""\n    os.makedirs(save_dir, exist_ok=True)\n    state_path = os.path.join(save_dir, "final_state.msgpack")\n    hist_path = os.path.join(save_dir, "hist.npy")\n    \n    # 1. Save TrainState using Flax\'s built-in msgpack serialization (Secure, no pickle)\n    # This securely dumps params, batch_stats, and opt_state into a secure byte-string.\n    with open(state_path, "wb") as f:\n        f.write(serialization.to_bytes(state))\n        \n    # 2. Save history (NumPy binary format, allow_pickle=False ensures safety)\n    np.save(hist_path, np.array(hist), allow_pickle=False)\n    \n    print(f"Saved state to {state_path} and history to {hist_path}")\n\nsave_model_and_history(final_state, hist)\n')


# In[32]:


from flax import serialization
import jax
import jax.numpy as jnp
import optax
import numpy as np
import os

def load_model_and_history(label_size, res, include_energy, input_shape, tx, load_dir="/home/agus/workspace/model"):
    """Loads model state and training history into a fresh instance."""
    state_path = os.path.join(load_dir, "final_state.msgpack")
    hist_path = os.path.join(load_dir, "hist.npy")
    
    # 1. Re-instantiate the precise model topology
    model = PhysicsOrbitalGraphNet(label_size=label_size, res=res, include_energy=include_energy)

    # 2. Initialize dummy inputs to extract the PyTree shapes
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones(input_shape)
    dummy_e = jnp.ones((1, 1)) if include_energy else None

    # 3. Get structural variables (params and batch_stats)
    variables = model.init(rng, dummy_x, dummy_e, training=False)

    # 4. Create the empty custom TrainState template
    template_state = TrainState.create(
        apply_fn=model.apply, 
        params=variables['params'], 
        tx=tx, 
        batch_stats=variables.get('batch_stats', {})
    )

    # 5. Read the bytes from disk
    with open(state_path, "rb") as f:
        byte_data = f.read()

    # 6. Securely map the bytes from disk into the empty state template
    restored_state = serialization.from_bytes(template_state, byte_data)

    # 7. Load the history
    restored_hist = np.load(hist_path, allow_pickle=False).tolist()
    
    print(f"Loaded state from {state_path} and history from {hist_path}")
    return restored_state, restored_hist

sched = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, peak_value=3e-4, warmup_steps=3000, decay_steps=10000, end_value=5e-6
)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))

# Setup the precise input shape your model expects
dim = M_PAIRS**2 if input_type == 'rho2block' else M_PAIRS
input_shape = (1, dim, dim, 1)

final_state, hist = load_model_and_history(
    label_size=g_gen.label_size(),
    res=res,
    include_energy=include_energy,
    input_shape=input_shape,
    tx=tx
)

val_dataset = gen_dataset(
    h_type, g_init, g_stop, state_type, input_type, include_energy, BETA, 
    num_samples=int(0.05 * num_samples),
    cache_path="/home/agus/workspace/dataset_val",   
    batch_size=2048, 
    train_batch_size=GPU_BATCH_SIZE,
    seed=43 + host_id
)

print("Model successfully loaded and ready for JIT inference!")


# ## BCS Comparison

# ##### Uniform G (G = G)

# In[ ]:


import scipy.optimize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numba
from numba import njit
import numpy as np

CURRENT_STATE_TYPE = state_type if 'state_type' in locals() else 'gs' 
TEMP_BETA = BETA

IS_GS = (CURRENT_STATE_TYPE == 'gs')
if IS_GS:
    TEMP_BETA = np.inf

ENERG_BCS = U_ENERGY_SEED[0][::2] 
E_MEAN_BCS = np.mean(ENERG_BCS)
M_BCS = len(ENERG_BCS)

# -----------------------------------------------------------------------------
# Physics Kernels
# -----------------------------------------------------------------------------

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
# 2. Inversion Solvers
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
            err += 1e-2 * np.abs(energy_target - get_bcs_energy_exact(delta, g_val, energies, beta))
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
        rho_abs = np.abs(rho_matrix).copy().squeeze()
        np.fill_diagonal(rho_abs, 0.0)
        coherence_sum = np.sum(rho_abs)
        return np.abs(g) * np.sqrt(coherence_sum)

# -----------------------------------------------------------------------------
# 3. Execution
# -----------------------------------------------------------------------------

print("\nProcessing Validation Dataset (JAX)...")

# 1. Load and Predict using new helper
# We use the existing 'dataset' loader
NUM_EVAL = 5000

rho_arr, energies_arr, g_true_arr, g_pred_arr = predict_and_load(
    loader=dataset,
    state=final_state,
    num_samples=NUM_EVAL,
    gpu_batch_size=GPU_BATCH_SIZE,
    include_energy=include_energy
)

# 2. Flatten and Sort
# Handle scalar vs vector labels
if g_true_arr.ndim > 1 and g_true_arr.shape[1] > 1:
    sort_keys = np.linalg.norm(g_true_arr, axis=1)
    g_true_flat = g_true_arr 
    g_pred_flat = g_pred_arr
else:
    sort_keys = g_true_arr.flatten()
    g_true_flat = g_true_arr.flatten()
    g_pred_flat = g_pred_arr.flatten()

energies_flat = energies_arr.flatten()

sort_idx = np.argsort(sort_keys)

g_true_sorted = g_true_flat[sort_idx]
g_pred_sorted = g_pred_flat[sort_idx]
rho_sorted = rho_arr[sort_idx]
energies_sorted = energies_flat[sort_idx]


# Inversion Loop
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

# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'figure.autolayout': True
})

# --- Plot 1: Interaction Strength ---
plt.figure(figsize=(10, 8))
plt.plot(g_true_sorted, g_true_sorted, 'k--', alpha=0.5, label='Identity')
plt.scatter(g_true_sorted, g_pred_sorted, s=50, alpha=0.6, color='tab:blue', label='CNN Prediction')
plt.plot(g_true_sorted, g_bcs_list, 'r-', label='BCS Inversion (Reg)')
plt.plot(g_true_sorted, g_bcs_list_alt, 'g-', label='BCS Inversion')

plt.xlabel(r"$G_{true}$")
plt.ylabel(r"Predicted $G$")
plt.legend(frameon=True, fancybox=True, framealpha=0.8)
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig("interaction.png", dpi=300)
plt.show()

# --- Plot 2: Order Parameter ---
plt.figure(figsize=(10, 8))
plt.plot(g_true_sorted, psi_exact_list, 'k--', label=r'Theoretical')
plt.plot(g_true_sorted, psi_bcs_list, 'r-', label=r'BCS Inversion' + ' (Reg)')
plt.plot(g_true_sorted, psi_bcs_list_alt, 'g-', label=r'BCS Inversion')

plt.xlabel(r"$G_{true}$")
plt.ylabel(r"Order Parameter")
plt.legend(frameon=True, fancybox=True, framealpha=0.8)
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig("order_parameter.png", dpi=300)
plt.show()


# ##### Vectorial G = G(k-k')

# In[ ]:


import numpy as np
import scipy.optimize
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm.auto import tqdm
import ray

ENERG_VECT = levels
M_PAIRS = len(ENERG_VECT)
EPSILON = 1e-12 

samples = 1000
print("Loading data and predicting with ML model...")

# 1. Load and Predict
rdm_input, energy_input, actual_values_raw, predictions = predict_and_load(
    loader=dataset, 
    state=final_state, 
    num_samples=samples, 
    gpu_batch_size=GPU_BATCH_SIZE,
    include_energy=include_energy
)

# 2. Reshape Logic (Preserved from original code)
label_size = g_gen.label_size()
values_shape = (-1, label_size) if label_size > 1 else (-1, 1)
actual_values = actual_values_raw.reshape(values_shape)

# 3. Sort results
g_norms_true = np.linalg.norm(actual_values, axis=1)
g_ids = g_norms_true.argsort()

predictions_sort = predictions[g_ids]
g_true_sort = actual_values[g_ids]
energy_sort = energy_input[g_ids]
rho_actual_sort = rdm_input[g_ids]
g_norms_sort = g_norms_true[g_ids]

# ==============================================================================
# Physics Kernels
# ==============================================================================

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
# Inversion driver
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

# ==============================================================================
# Execution 
# ==============================================================================

print("Starting Inversion...")

ray.shutdown()
ray.init(num_cpus=16, ignore_reinit_error=True)

@ray.remote
def parallel_task(rho_i, state_type, beta, actual_energy, energ_array, l_size, vr):
    try:
        e_target = None
        if actual_energy is not None:
            try:
                val = float(actual_energy)
                if not np.isnan(val): e_target = val
            except: pass
            
        return bcs_variational_driver(
            rho_i, state_type, beta, e_target, 
            energ_array, l_size, vr,
            lambda_smooth=0.05, lambda_energy=1.0
        )
    except Exception as e:
        return str(e) 

# Setup data
ENERG_VECT_F64 = np.asarray(ENERG_VECT, dtype=np.float64)
L_SIZE_REF = int(g_gen.label_size())
VREPEAT_REF = int(g_gen.vrepeat)

# Launch
futures = [parallel_task.remote(
                rho_actual_sort[i], state_type, BETA, 
                energy_sort[i], ENERG_VECT_F64, L_SIZE_REF, VREPEAT_REF
           ) for i in range(len(g_true_sort))]

# Collect
g_opt_list, delta_opt_list = [], []
errors_printed = 0

for future in tqdm(futures, total=len(futures), desc="Optimizing"):
    res = ray.get(future)
    if isinstance(res, str): 
        if errors_printed < 3: 
            print(f"Task Failed: {res}")
            errors_printed += 1
        g_opt_list.append(np.full(L_SIZE_REF, np.nan))
        delta_opt_list.append(np.full(M_PAIRS, np.nan))
    else:
        g_opt_list.append(res[0])
        delta_opt_list.append(res[1])

g_opt_arr = np.array(g_opt_list)
valid_mask = ~np.isnan(g_opt_arr).any(axis=1)

print(f"\nFinal success rate: {np.sum(valid_mask)}/{len(valid_mask)}")


# In[ ]:


# ==============================================================================
# Visualization
# ==============================================================================

def reconstruct_vec(labels, vr, m):
    vec = np.zeros(m)
    for k, val in enumerate(labels):
        start = (k + 1) * vr
        end = min(start + vr, m)
        if start < m: vec[start:end] = val
    return vec

if np.sum(valid_mask) > 0:
    idx_map = np.where(valid_mask)[0]
    norms = np.linalg.norm(g_true_sort[idx_map], axis=1)
    mid_idx = idx_map[np.argsort(norms)[len(norms)//2]]
    
    v_true = reconstruct_vec(g_true_sort[mid_idx], VREPEAT_REF, M_PAIRS)
    v_ml   = reconstruct_vec(predictions_sort[mid_idx], VREPEAT_REF, M_PAIRS)
    v_phys = reconstruct_vec(g_opt_arr[mid_idx], VREPEAT_REF, M_PAIRS)


    err_ml = np.linalg.norm(predictions_sort[valid_mask] - g_true_sort[valid_mask], axis=1)
    err_phys = np.linalg.norm(g_opt_arr[valid_mask] - g_true_sort[valid_mask], axis=1)
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18})

    plt.scatter(g_norms_sort[valid_mask], err_ml, alpha=0.6, s=50, 
                color='tab:blue', label='CNN Prediction')
    plt.scatter(g_norms_sort[valid_mask], err_phys, marker='x', c='tab:red', s=60, 
                alpha=0.6, label='BCS Inversion')
    
    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"$||G_{pred} - G_{true}||$")
    plt.yscale('log')
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    
    plt.savefig("accuracy.png", dpi=300)
    plt.show()
else:
    print("No valid data points generated.")


# In[ ]:


# ==============================================================================
# Re-Run Optimization with Corrected Constraints
# ==============================================================================
ray.shutdown()
ray.init(num_cpus=16, ignore_reinit_error=True)

# Define hyperparameters
L_ENERGY = 1.0
L_SMOOTH = 0.05

# Redefine the worker to use the new lambdas
@ray.remote
def parallel_robust_task_v2(rho_i, state_type, beta, actual_energy, energ_array, l_size, vr):
    return bcs_variational_driver(
        rho_i, state_type, beta, actual_energy,
        energ_array, l_size, vr,
        lambda_smooth=L_SMOOTH, lambda_energy=L_ENERGY
    )

futures = [parallel_robust_task_v2.remote(
                rho_actual_sort[i], state_type, BETA,
                energy_sort[i], ENERG_VECT_F64, L_SIZE_REF, VREPEAT_REF
           ) for i in range(len(g_true_sort))]

g_opt_list_v2, delta_opt_list_v2 = [], []

for future in tqdm(futures, total=len(futures), desc="Optimizing"):
    res = ray.get(future)
    if isinstance(res, str):
        g_opt_list_v2.append(np.full(L_SIZE_REF, np.nan))
        delta_opt_list_v2.append(np.full(M_PAIRS, np.nan))
    else:
        g_opt_list_v2.append(res[0])
        delta_opt_list_v2.append(res[1])
        
def get_order_parameter(rho_matrix, g_vector):
        # 1. Take absolute value
        rho_abs = np.abs(rho_matrix).copy()
        
        # 2. Zero out the diagonal (Remove particle number contribution)
        # We only want off-diagonal coherence: sum_{i!=j} <P_i P_j>
        np.fill_diagonal(rho_abs, 0.0)
        
        # 3. Calculate metric
        coherence_sum = np.sum(rho_abs)
        return np.linalg.norm(g_vector) * np.sqrt(coherence_sum)

def physical_observables_dashboard(rho_input_batch, g_opt_batch, g_true_batch,
                                   energ_array, valid_mask, g_gen, rho_ml_batch=None):

    idx_valid = np.where(valid_mask)[0]
    if len(idx_valid) == 0: return

    nk_exact_list = []
    nk_bcs_list = []

    psi_exact_list = []
    psi_bcs_list = []
    psi_ml_list = []

    norms_true = []
    norms_phys = []

    m_pairs = len(energ_array)
    vr = int(g_gen.vrepeat)

    print("Computing physical observables...")

    for i in idx_valid:
        # --- 1. Exact Observables ---
        rho = rho_input_batch[i]

        # Robust shape handling for Ground Truth (Existing logic)
        if rho.ndim == 3 and rho.shape[-1] == 1: rho = rho.squeeze(-1)
        if rho.ndim == 1: rho = rho.reshape(m_pairs, m_pairs)

        # Occupancy (Diagonal)
        nk_ex = np.diag(rho)
        nk_exact_list.append(nk_ex)

        # --- 2. BCS Observables ---
        g_params = g_opt_batch[i]
        delta, _ = solve_gap_eq_stateless(g_params, energ_array, m_pairs, vr, BETA)
        nk_b, ukvk_b, _ = calc_bcs_observables(delta, energ_array, m_pairs, BETA)

        nk_bcs_list.append(nk_b)

        # Exact Order Parameter
        psi_ex = get_order_parameter(rho, g_true_batch[i])
        psi_exact_list.append(psi_ex)

        # ML Order Parameter
        if rho_ml_batch is not None:
            rho_ml = rho_ml_batch[i]
            
            # --- FIX: Squeeze channel dimension (M, M, 1) -> (M, M) ---
            if rho_ml.ndim == 3 and rho_ml.shape[-1] == 1: 
                rho_ml = rho_ml.squeeze(-1)
            # Handle flattened case if present
            if rho_ml.ndim == 1: 
                rho_ml = rho_ml.reshape(m_pairs, m_pairs)
            
            psi_ml = get_order_parameter(rho_ml, g_true_batch[i])
            psi_ml_list.append(psi_ml)

        # BCS Pairing Order Parameter
        rhobcs = np.outer(ukvk_b, ukvk_b)
        np.fill_diagonal(rhobcs, nk_b)
        psi_b = get_order_parameter(rhobcs, g_params)
        psi_bcs_list.append(psi_b)

        # Norms
        norms_true.append(np.linalg.norm(g_true_batch[i]))
        norms_phys.append(np.linalg.norm(g_params))

    # Convert to arrays
    nk_exact_arr = np.array(nk_exact_list)
    nk_bcs_arr = np.array(nk_bcs_list)
    psi_exact_arr = np.array(psi_exact_list)
    psi_bcs_arr = np.array(psi_bcs_list)

    if rho_ml_batch is not None:
        psi_ml_arr = np.array(psi_ml_list)

    norms_true = np.array(norms_true)
    norms_phys = np.array(norms_phys)

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 2.5,
        'figure.autolayout': True
    })

    # --- Plot 1: Renormalization Z ---
    plt.figure(figsize=(10, 8))
    # Filter to avoid division by zero
    valid_norm = norms_true > 1e-9
    z_factor = np.zeros_like(norms_phys)
    z_factor[valid_norm] = norms_phys[valid_norm] / norms_true[valid_norm]

    plt.scatter(norms_true, z_factor, alpha=0.6, s=50, color='tab:blue')
    plt.axhline(1.0, color='k', ls='--', linewidth=2.5)

    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"$||G_{BCS}|| / ||G_{true}||$")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig("renormalization.png", dpi=300)
    plt.show()

    # --- Plot 2: Order Parameter Fidelity ---
    plt.figure(figsize=(10, 8))
    
    plt.scatter(norms_true, psi_bcs_arr, c='tab:red', alpha=0.5, s=50,
                label='BCS Inversion')

    if rho_ml_batch is not None:
        plt.scatter(norms_true, psi_ml_arr, c='tab:blue', alpha=0.5, s=50,
                    marker='x', label='CNN Prediction')

    mx = max(psi_exact_arr.max(), psi_bcs_arr.max())
    if rho_ml_batch is not None:
        mx = max(mx, psi_ml_arr.max())

    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"Order Parameter")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(frameon=True, fancybox=True, framealpha=0.8)
    plt.savefig("order_parameter_vect.png", dpi=300)
    plt.show()

    # --- Plot 3: Occupancy Error vs Strength ---
    plt.figure(figsize=(10, 8))
    occ_error = np.mean((nk_exact_arr - nk_bcs_arr)**2, axis=1)

    plt.scatter(norms_true, occ_error, c='tab:red', alpha=0.5, marker='x', s=60)
    plt.yscale('log')

    plt.xlabel(r"$||G_{true}||$")
    plt.ylabel(r"Occupancy MSE")
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.show()

# ==============================================================================
# Execution
# ==============================================================================

# Ensure dense inputs for reconstruction
rho1_d = _ensure_dense(rho_1_arrays)
rho2_d = _ensure_dense(rho_2_arrays)
# Use safe access for optional arrays
rho2_kk_d = _ensure_dense(rho_2_kkbar_arrays) if 'rho_2_kkbar_arrays' in globals() else rho2_d
rho2_bl_d = _ensure_dense(rho_2_block_arrays) if 'rho_2_block_arrays' in globals() else rho2_d

print("Reconstructing ML RDM from predictions...")
rho_ml_batch = rho_reconstruction(
     predictions_sort, h_type, state_type, input_type, basis,
     U_ENERGY_SEED, BETA, rho1_d,
     rho2_d, rho2_kk_d, rho2_bl_d
)

g_opt_arr_v2 = np.array(g_opt_list_v2)
valid_mask_v2 = ~np.isnan(g_opt_arr_v2).any(axis=1)

physical_observables_dashboard(
    rho_actual_sort,
    g_opt_arr_v2,
    g_true_sort,
    ENERG_VECT_F64,
    valid_mask_v2,
    g_gen,
    rho_ml_batch=rho_ml_batch
)


# ## Covariance matrix comparison

# Based on https://arxiv.org/abs/1712.01850 

# In[33]:


from numba import njit, prange, typed, types
from numba.typed import Dict
from functools import lru_cache
import scipy.sparse

def m_index_map(d, m):
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
def get_sorting_sign(indices):
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
def get_rdm_2_element(creators, annihilators, rho_2, quad_map):
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
def get_rdm_3_element(creators, annihilators, rho_3, trip_map):
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
def get_rdm_4_element(creators, annihilators, rho_4, quar_map):
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
def get_rdm_1_hcb(i, j, rho_1):
    """ Calculates <B_i^+ B_j>. Convention: rho_1[j, i] = <B_i^+ B_j>"""
    # Bounds check required for Numba safety
    if i >= rho_1.shape[1] or j >= rho_1.shape[0]:
        return 0.0
    return rho_1[j, i]

@njit(cache=True)
def get_rdm_2_hcb(i, j, k, l, rho_2, map_2):
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
                                      quad_map, trip_map, quar_map):
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
def compute_M(all_indices, rho_2, rho_3, rho_4, quad_map, trip_map, quar_map):
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

def convert_to_numba_dict(py_dict, key_type, value_type):
    """Helper to convert Python dict to Numba typed dict"""
    numba_dict = Dict.empty(key_type=key_type, value_type=value_type)
    for k, v in py_dict.items():
        typed_key = tuple(int(x) for x in k)
        numba_dict[typed_key] = int(v)
    return numba_dict

def covariance_matrix(basis, rho_2, rho_3, rho_4, quad_map, trip_map, quar_map, quad_pairs):
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
def compute_M_paired_hcb(rho_1, rho_2, map_2):
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

def covariance_matrix_paired(basis, rho_1_pair, rho_2_pair):
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

def get_permutation_maps(quad_map, quad_pairs):
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

def compute_coords(h, basis, rho_2_arrays, quad_map, quad_pairs):
    """
    Calculates the coordinates (w) of Hamiltonian h.
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

def random_h(basis, rho_gen, quad_map=None, quad_pairs=None):
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


# In[18]:


"""
Requierments
"""
# Basis
d, m = 20, 10
USE_PAIRING_RESTRICTION = True  
basis = fmb.FixedBasis(d, m, pairs=USE_PAIRING_RESTRICTION)

# Mappings
quad_map, quad_pairs = m_index_map(basis.d, 2)
trip_map, trip_pairs = m_index_map(basis.d, 3)
quar_map, quar_pairs = m_index_map(basis.d, 4)
M_dim = len(list(combinations(range(basis.d), 2)))**2

# Indices
all_indices = []
for alpha in itertools.product(quad_pairs, repeat=2):
    i, j, k, l = [int(x) for tup in alpha for x in tup]
    all_indices.append((i, j, k, l))


if USE_PAIRING_RESTRICTION:
    # Pairing case
    rho_p_arrays = fmb.rho_m_gen(basis, 1, n_workers=1)
    h, coefs = random_h(basis, rho_p_arrays )
    e, v = scipy.sparse.linalg.eigsh(h, k=1, which='SA', tol=1e-8)
    state = v[:, 0]
    # RDMs
    rho_1_pair = fmb.rho_m_direct(basis, 1, state)
    rho_2_pair = fmb.rho_m_direct(basis, 2, state)

    # Pairing covariance matrix
    M = covariance_matrix_paired(basis, np.asarray(rho_1_pair), np.asarray(rho_2_pair))
    
else:
    # General Case
    quad_map, quad_pairs = m_index_map(basis.d, 2)
    # Use fmb.rho_m_gen(basis, 2) for 2-body general Hamiltonian
    rho_gen = fmb.rho_2_kkbar_gen(basis)
    h, coefs = random_h(basis, rho_gen, quad_map, quad_pairs)
    e, v = scipy.sparse.linalg.eigsh(h, k=1, which='SA', tol=1e-8)
    state = v[:, 0]
    # RDMs
    rho_2 = fmb.rho_m_direct(basis, 2, state)
    rho_3 = fmb.rho_m_direct(basis, 3, state)
    rho_4 = fmb.rho_m_direct(basis, 4, state)

    # Covariance matrix
    M = covariance_matrix(basis, rho_2, rho_3, rho_4, quad_map, trip_map, quar_map, quad_pairs)


# ##### General analysis (random H)

# In[22]:


import numpy as np
import itertools

M_np = np.asarray(M)
M_dim = M_np.shape[0]

print(f"\n--- Verification of the Null Space (Pairs={USE_PAIRING_RESTRICTION}) ---")

# 1. Construct w_N (Coefficient vector for Number Operator)
w_N = np.zeros(M_dim)

if USE_PAIRING_RESTRICTION:
    d_pair = basis.d // 2
    print(f"Dimensions: d_pair={d_pair}, M_dim={M_dim}")
    
    # In the HCB basis, operators are P_i^dag P_j flattened in row-major order.
    # The Particle Number operator is N = 2 * Sum_k P_k^dag P_k.
    # This corresponds to diagonal elements (i=j) with coefficient 2.0.
    for k in range(d_pair):
        idx = k * d_pair + k
        w_N[idx] = 2.0
else:
    D2 = len(quad_pairs)
    print(f"Dimensions: D2={D2}, M_dim={M_dim}")
    
    # In general basis, N is represented by a specific sum of 2-body terms
    # w_N corresponds to sum_{i!=j} c_i^dag c_j^dag c_j c_i
    coeff_val = -2.0 / (basis.num - 1)
    idx = 0
    for pair1, pair2 in itertools.product(quad_pairs, repeat=2):
        if pair1 == pair2:
            w_N[idx] = coeff_val
        idx += 1

# Normalize w_N
if np.linalg.norm(w_N) > 0:
    w_N_norm = w_N / np.linalg.norm(w_N)
else:
    w_N_norm = w_N

# 2. Variance Check
# w_N is a conserved quantity (scalar in the eigenspace), so its variance should be 0.
variance_N = w_N_norm.T @ M_np @ w_N_norm
variance_H = coefs.T @ M_np @ coefs

print(f"\nVariance Check:")
print(f"Variance of H (random)    : {variance_H:.6e}")
print(f"Variance of N_hat rep.    : {variance_N:.6e} (Should be ~0)")

# 3. Null Space Analysis
print("\nAnalyzing the null space projections...")

# Use eigh for symmetric matrices
M_sym = 0.5 * (M_np + M_np.T)
eigenvalues, eigenvectors = np.linalg.eigh(M_sym)

tol = 1e-12
idx_sort = np.argsort(np.abs(eigenvalues))
eigenvalues = eigenvalues[idx_sort]
eigenvectors = eigenvectors[:, idx_sort]

null_indices = np.where(np.abs(eigenvalues) < tol)[0]
print(f"Number of near-zero eigenvalues found: {len(null_indices)}")

if len(null_indices) > 0:
    null_space_vectors = eigenvectors[:, null_indices]

    # Normalize vectors
    norm_coefs = coefs / np.linalg.norm(coefs)

    # Calculate projections
    # H (eigenstate variance=0) should project significantly onto the null space
    proj_coefs = null_space_vectors.T @ norm_coefs
    mag_sq_proj_coefs = np.linalg.norm(proj_coefs)**2
    print(f"Projection magnitude^2 of H (original) onto null space: {mag_sq_proj_coefs:.6f}")

    # N_hat (variance=0) should project perfectly (1.0) onto the null space
    proj_w_N = null_space_vectors.T @ w_N_norm
    mag_sq_proj_w_N = np.linalg.norm(proj_w_N)**2
    print(f"Projection magnitude^2 of N_hat rep. onto null space   : {mag_sq_proj_w_N:.6f}")

    # Check linear independence
    overlap = np.abs(np.dot(w_N_norm, norm_coefs))
    print(f"Overlap between H and N_hat coefficients               : {overlap:.4f}")
else:
    print("No null space found.")


# In[ ]:


def get_pairing_symmetry_vectors(basis, quad_map, quad_pairs):
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

# --- Verification Block to Add to Your Script ---

if not USE_PAIRING_RESTRICTION:
    print("\n--- Verifying Pairing Symmetries (Local SU(2)) ---")
    sym_vecs, sym_labels = get_pairing_symmetry_vectors(basis, quad_map, quad_pairs)
    print(f"Generated {sym_vecs.shape[1]} local symmetry vectors.")
    
    # 1. Variance Check (Should be ~0)
    # Variance = w^T M w
    variances = np.diag(sym_vecs.T @ M_np @ sym_vecs)
    max_var = np.max(np.abs(variances))
    print(f"Max Variance of Symmetries         : {max_var:.6e}")
    
    # 2. Projection onto Null Space
    if len(null_indices) > 0:
        # Project symmetry vectors onto the null space basis V
        # Projection = || V.T @ w ||^2 (if V is orthonormal)
        # null_space_vectors is (M_dim, n_null)
        projections = np.linalg.norm(null_space_vectors.T @ sym_vecs, axis=0)**2
        min_proj = np.min(projections)
        mean_proj = np.mean(projections)
        print(f"Min Projection onto M's Null Space : {min_proj:.6f}")
        print(f"Mean Projection onto M's Null Space: {mean_proj:.6f}")


# ##### ML Analysis (dataset)

# In[ ]:


import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys, os
from tqdm.auto import tqdm

# [REGEN-EXTRACT-BEGIN covariance_suite]  (regen_figures.py executes this
# region verbatim on the compute engines; keep it self-contained)
# =============================================================================
# 1. Subspace Projection (Gram-Schmidt)
# =============================================================================

def get_w_N_pairing(basis):
    """Constructs the Number Operator vector representation w_N for the pairing case."""
    d_pair = basis.d // 2
    w_N_mat = np.zeros((d_pair, d_pair))
    # N = 2 * sum(P_k^\dag P_k)
    np.fill_diagonal(w_N_mat, 2.0)
    return w_N_mat.flatten()

def evaluate_subspace_projection(w_pred, w_true, w_N, S):
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

def compute_gevp_observables(M, S, w_eval, w_ref=None):
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
                                  lam_ridge=1e-9, cutoff=1e-7):
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
    tiny = 1e-12
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
    }
    return lambdas_phys, q_k, gevp_diag


# [REGEN-PATCH P1] S-orthogonal background/disorder split of manuscript
# Sec. III.B.3, Eq. (eps_I). w_H0 / w_bg are built by
# build_background_vectors() consistent with the "FIX 1" call-site assembly.
def background_split_metrics(w_pred, w_true, w_H0, w_bg, S):
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


# [REGEN-PATCH P1] Background vectors: w_H0 is the flattened M_p x M_p matrix
# with diag = pair_energies_true (the H0 diagonal added at the call site);
# w_bg is the flattened all-ones matrix (sum_{kk'} P_k^dag P_k'); the
# projector is scale-invariant so no <G> prefactor is applied.
def build_background_vectors(pair_energies_true, m_pairs):
    w_H0 = np.zeros((m_pairs, m_pairs), dtype=np.float64)
    np.fill_diagonal(w_H0, np.asarray(pair_energies_true, dtype=np.float64))
    w_bg = np.ones((m_pairs, m_pairs), dtype=np.float64)
    return w_H0.flatten(), w_bg.flatten()


def plot_averaged_scree(lambdas_phys, O_k_mean, O_k_std, num_samples, title_suffix="", ortho_err_pct=None):
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
    plt.show()

# =============================================================================
# 3. Main Evaluation Loop
# =============================================================================

# [REGEN-PATCH P5] kwargs lam_ridge/cutoff expose the GEVP grid for the
# stability sweep; make_plot/return_diag let the regeneration driver collect
# the Table tab:gevp_diag quantities without altering the legacy call sites.
def evaluate_ml_covariance_batch(val_loader, state_model, basis, current_h_type, inc_energy=True, num_samples=15, title_suffix='',
                                 lam_ridge=1e-9, cutoff=1e-7, make_plot=True, return_diag=False):
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


# 1. Random Check
avg_lambdas, avg_O_k, sub_random = evaluate_ml_covariance_batch(
    val_loader=val_dataset,      
    state_model=final_state,     
    basis=basis,
    current_h_type='random',
    inc_energy=include_energy,
    num_samples=4096,
    title_suffix='Random'
)

# 2. Const Check
avg_lambdas_const, avg_O_k_const, sub_const = evaluate_ml_covariance_batch(
    val_loader=val_dataset_const,      
    state_model=final_state_const,     
    basis=basis,
    current_h_type='const',
    inc_energy=include_energy,
    num_samples=4096,
    title_suffix='Constant'
)

# 3. Untrained Network Check
avg_lambdas_non, avg_O_k_non, sub_untrained = evaluate_ml_covariance_batch(
    val_loader=val_dataset_non,      
    state_model=final_state_non,     
    basis=basis,
    current_h_type='random',
    inc_energy=include_energy,
    num_samples=2048,
    title_suffix='Untrained'
)

# 4. Comparative Subspace Plotting
results = {
    'Random Trained': sub_random,
    'Const Trained': sub_const,
    'Untrained Network': sub_untrained
}

results


# ## Pseudoinverse comparison (GS)

# Given how the 2-body operators acts on the state, it's possible to invert the eigenvalues equation

# In[18]:


import numpy as np
import sparse
import openfermion as of
from itertools import combinations
import time
import scipy.sparse.linalg as spla
from scipy.sparse import hstack, csc_matrix

d = 20
basis = fmb.FixedBasis(d, num=d//2, pairs=True)
quad_map, quad_pairs = m_index_map(basis.d, 2)
rho_2_arrays = fmb.rho_m_gen(basis, 2, n_workers = 48) 

M_dim = len(list(combinations(range(basis.d), 2)))**2


# In[19]:


def bitmask_to_lex(T_bm, axes, quad_map, quad_pairs):
    """
    Permutes the specified axes of a sparse tensor T_bm from Bitmask (bm) order to Lexicographical (lex) order
    """
    D2 = len(quad_pairs)
    perm = np.zeros(D2, dtype=int)
    for i, pair in enumerate(quad_pairs):
        perm[i] = quad_map[pair]
    
    inv_perm = np.argsort(perm)

    coords = T_bm.coords
    data = T_bm.data
    new_coords = coords.copy()

    for axis in axes:
        idx_bm = coords[axis]
        idx_lex = inv_perm[idx_bm]
        new_coords[axis] = idx_lex
    
    return sparse.COO(new_coords, data, shape=T_bm.shape)

def indexation_remap(W, quad_map, quad_pairs):
    """
    Permutes the last two dimensions of a 3D sparse array W.
    Converts from FixedBasis index (bm) -> quad_pairs index (lex).
    """
    coords = W.coords
    data = W.data
    r_idx = coords[0]
    I_idx_bm = coords[1]
    J_idx_bm = coords[2]

    # Calculate permutation P: lex -> bm
    perm = np.zeros(len(quad_pairs), dtype=int)
    for i, pair in enumerate(quad_pairs):
        perm[i] = quad_map[pair]
    
    # Calculate inverse permutation P^-1: bm -> lex
    inv_perm = np.argsort(perm)

    # Map the indices to the new ordering (bm to lex)
    I_lex = inv_perm[I_idx_bm]
    J_lex = inv_perm[J_idx_bm]

    new_coords = np.stack([r_idx, I_lex, J_lex])
    
    return sparse.COO(new_coords, data, shape=W.shape)

def compute_W(state, rho_2_arrays, quad_map, quad_pairs):
    """
    Computes the matrix W. The output W matrix columns are indexed in Lexicographical order.
    """
    D2 = len(quad_pairs)
    DN = len(state)
    
    perm = np.zeros(D2, dtype=int)
    for i, pair in enumerate(quad_pairs):
        perm[i] = quad_map[pair]
    inv_perm = np.argsort(perm)
    
    # Extract native SciPy 1.12+ coordinates safely
    coords = rho_2_arrays.coords
    data = rho_2_arrays.data
    J_bm, I_bm, r_idx, c_idx = coords[0], coords[1], coords[2], coords[3]
    
    # Map from bitmask to lexicographical
    J_lex = inv_perm[J_bm]
    I_lex = inv_perm[I_bm]
    
    col_idx = I_lex * D2 + J_lex
    
    # H_I = - \sum W_{IJ} C^\dagger_I C_J, so W_IJ's action is -C^\dagger_I C_J
    w_data = -data * np.asarray(state)[c_idx] 
    
    W_sparse = sp.coo_array((w_data, (r_idx, col_idx)), shape=(DN, D2**2))
    W_sparse.sum_duplicates()
    return W_sparse.tocsr()

def construct_W_sym(W):
    """
    Constructs the matrix W_sym corresponding to the basis of symmetric operators
    The resulting hamiltonian must be hermitian, therefore the present generators
    """
    D2 = len(quad_pairs)
    W_csc = W.tocsc()
    cols_sym = []
       
    for I in range(D2):
        cols_sym.append(W_csc[:, I * D2 + I])

    for I in range(D2):
        for J in range(I + 1, D2):
            col = W_csc[:, I * D2 + J] + W_csc[:, J * D2 + I]
            cols_sym.append(col)

    return sp.hstack(cols_sym).tocsr()

def random_h(basis, rho_2_arrays, quad_map, quad_pairs):
    """
    Generates a random two-body Hamiltonian H and its coefficients in Lexicographical order
    """
    D2 = len(quad_pairs)
    DN = basis.size
    
    W_matrix = np.random.uniform(0.1, 5, (D2, D2))
    W_matrix = 0.5 * (W_matrix + W_matrix.T)
    coefs = W_matrix.reshape(-1)
    
    perm = np.zeros(D2, dtype=int)
    for i, pair in enumerate(quad_pairs):
        perm[i] = quad_map[pair]
    inv_perm = np.argsort(perm)
    
    coords = rho_2_arrays.coords
    data = rho_2_arrays.data
    J_bm, I_bm, r_idx, c_idx = coords[0], coords[1], coords[2], coords[3]
    
    J_lex = inv_perm[J_bm]
    I_lex = inv_perm[I_bm]
    
    # Fast vectorized contraction mapping (incorporating the minus sign)
    h_data = -data * W_matrix[I_lex, J_lex]
    
    h_sparse = sp.coo_array((h_data, (r_idx, c_idx)), shape=(DN, DN))
    h_sparse.sum_duplicates()
    
    # Scrub floating point noise to guarantee exact Hermitianity
    h_csr = h_sparse.tocsr()
    h_csr = 0.5 * (h_csr + h_csr.T) 
    
    return h_csr, coefs

# [REGEN-EXTRACT-BEGIN wls_system]
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


# ##### Ground states

# In[19]:


h, coefs = random_h(basis, rho_2_arrays, quad_map, quad_pairs)
e, v = scipy.sparse.linalg.eigsh(h, k=1, which='SA', tol=1e-8)
state = v[:, 0]
epsilon = e[0]
D2 = len(quad_pairs)

W = compute_W(state, rho_2_arrays, quad_map, quad_pairs)

# Verification check
residual = np.linalg.norm(W @ coefs - epsilon * state)
print(f"Consistency check residual ||W*a - e*|psi>||: {residual}")

W_symm = construct_W_sym(W)


# In[40]:


import scipy.linalg
import scipy.sparse
import numpy as np
import time

def compute_null_space_optimized_WTW(W_sparse, tol_zero=1e-9):
    """
    Computes the null space by forming W^T W using sparse multiplication
    and then using a dense parallel eigensolver
    """
    print("--- Null space calculation ---")

    if not isinstance(W_sparse, (scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)):
        try:
            W_sparse = W_sparse.tocsr()
        except AttributeError:
            W_sparse = scipy.sparse.csr_matrix(W_sparse)

    # Compute W^T W 
    print(f"Computing W^T @ W (Shape: {W_sparse.shape[1]}x{W_sparse.shape[1]})...")
    WTW_sparse = W_sparse.T @ W_sparse
    
    # Convert W^T W to dense
    WTW_dense = WTW_sparse.todense()

    # Dense parallel diagonalization
    print("Starting dense parallel eigh...")
    # This step is highly parallelized.
    evals, evecs = scipy.linalg.eigh(WTW_dense)

    # Analyze eigenvalues and filter
    print(f"\n--- Eigenvalue Analysis ---")
    print(f"Smallest 5 eigenvalues:\n{evals[:5]}")
    
    mask = evals < tol_zero
    kernel = evecs[:, mask]
    
    print(f"Nullity found with tolerance {tol_zero}: {kernel.shape[1]}")
    return kernel, evals

ker_symm, evals = compute_null_space_optimized_WTW(W, tol_zero=1e-9)
ker_symm


# In[ ]:


import numpy as np
import scipy.linalg
import time
import scipy.sparse

def find_compatible_hamiltonians(W_sparse, state, tol_zero=1e-9):
    """
    Finds the subspace of Hamiltonians H such that H|psi> = epsilon|psi> for some epsilon.
    Computed via diagonalization of A^T A = W^T Q W = W^T W - v v^T.
    
    Args:
        W_sparse: The operator action matrix (e.g., W_symm).
        state: The eigenstate |psi>
    """
    print("\n--- Finding compatible hamiltonians ---")

    if not scipy.sparse.isspmatrix_csr(W_sparse) and not scipy.sparse.isspmatrix_csc(W_sparse):
        try:
            W_sparse = W_sparse.tocsr()
        except:
            W_sparse = scipy.sparse.csr_matrix(W_sparse)

    state = state / np.linalg.norm(state)

    # Compute W^T W (Sparse)
    print(f"Computing W^T @ W (Shape: {W_sparse.shape[1]}x{W_sparse.shape[1]})...")
    WTW_sparse = W_sparse.T @ W_sparse

    # Compute v = W^T |psi>
    v = W_sparse.T @ state
    
    # Construct A^T A = W^T W - v v^T (Dense)
    print("Constructing A^T A (Dense)...")
    # Convert WTW to dense
    ATA_dense = WTW_sparse.todense()
    # Subtract the rank-1 update v v^T
    ATA_dense -= np.outer(v, v)

    # Diagonalize A^T A
    print("Starting dense parallel eigh on A^T A...")
    evals, evecs = scipy.linalg.eigh(ATA_dense)

    # Analyze eigenvalues and filter
    print(f"\n--- Eigenvalue analysis (A^T A) ---")
    print(f"Smallest 5 eigenvalues:\n{evals[:5]}")
    
    mask = evals < tol_zero
    S_comp = evecs[:, mask]
    
    # Expected dimension: dim(Ker(W)) + 1. (Should be 2 for the Hermitian case).
    print(f"Dimension of compatible hamiltonians found: {S_comp.shape[1]}") 
    
    return S_comp

W_symm = construct_W_sym(W)
S_comp_symm = find_compatible_hamiltonians(W_symm, state)


# In[37]:


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp

# =========================================================================
# FAST VECTORIZED ACTION MATRIX 
# =========================================================================
def compute_W_symm_pairing_nd(state, rho_2_kkbar):
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

def reconstruct_G_from_symm(w_symm, M_pairs):
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

# =========================================================================
# EXACT INVERSION VS. ML COMPARISON
# =========================================================================
def visualize_exact_inversion_vs_ml(val_loader, state_model, basis, g_gen, u_energy_seed, rho_1_arrays, rho_2_kkbar):
    print("\n" + "="*80)
    print(" EXACT FULL-STATE INVERSION VS. OGN RDM INFERENCE")
    print("="*80)

    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    
    # 1. Use the pre-existing predict_and_load to ensure correct ML inference!
    NUM_SAMPLES = 10
    print(f"[1/4] Loading pristine samples from validation dataset...")
    rdm_arr, energy_arr, g_true_arr, g_pred_arr = predict_and_load(
        loader=val_loader, 
        state=state_model, 
        num_samples=NUM_SAMPLES, 
        gpu_batch_size=min(NUM_SAMPLES, 256), 
        include_energy=True
    )
    
    results = []
    
    # Pick two representative samples from the batch
    indices_to_plot = [0, 1] 
    
    for count, idx in enumerate(indices_to_plot):
        print(f"\nProcessing Sample {count+1}/2...")
        
        G_true = np.array(g_gen.reconstruct(jnp.array([g_true_arr[idx]])))[0]
        G_pred = np.array(g_gen.reconstruct(jnp.array([g_pred_arr[idx]])))[0]
        
        print("      Computing exact ground state vector...")
        H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true]), 
                                            rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H_true_csr = _prepare_hamiltonian_for_eigsh(H_true_sp)
        H_true_csr = 0.5 * (H_true_csr + H_true_csr.T)
        
        e_true, v_true = scipy.sparse.linalg.eigsh(H_true_csr, k=1, which='SA', tol=1e-8)
        gs_state, E_gs = v_true[:, 0], e_true[0]

        print(f"      Constructing Operator Action Matrix W ({len(gs_state)} dimensions) and solving Pseudoinverse...")
        W_symm, symm_idx = compute_W_symm_pairing_nd(gs_state, rho_2_kkbar)
        
        H0_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([np.zeros((M_pairs, M_pairs))]), 
                                        rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H0_csr = _prepare_hamiltonian_for_eigsh(H0_sp)
        H0_csr = 0.5 * (H0_csr + H0_csr.T)
        target_vec = (H0_csr @ gs_state) - E_gs * gs_state
        
        # Math Pseudoinverse (Scipy)
        w_pinv, res, rank, s = scipy.linalg.lstsq(W_symm, target_vec, cond=1e-9)
        G_pinv = reconstruct_G_from_symm(w_pinv, M_pairs).real
        G_pinv = 0.5 * (G_pinv + G_pinv.T)
        
        # Fair Gauge Shift: The pseudo-inverse is blind to total particle number conservation
        # We align the diagonal mean to isolate the pure structural error
        shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pinv))
        G_pinv_shifted = G_pinv.copy()
        np.fill_diagonal(G_pinv_shifted, np.diag(G_pinv_shifted) + shift)
        
        dist_ml = np.linalg.norm(G_pred - G_true)
        dist_pinv = np.linalg.norm(G_pinv_shifted - G_true)
        
        results.append({
            'title': f"Validation Sample {count+1}",
            'G_true': G_true, 'G_pred': G_pred, 'G_pinv': G_pinv_shifted,
            'dist_ml': dist_ml, 'dist_pinv': dist_pinv,
            'rank': rank, 'state_dim': len(gs_state)
        })
        print(f"      --> Math Rank: {rank}/55 | Oracle Error: {dist_pinv:.3f} | ML Error: {dist_ml:.3f}")

    # ================= PLOTTING =================
    print("\n[4/4] Rendering Visualization...")
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for row_idx, res in enumerate(results):
        G_t, G_m, G_p = res['G_true'], res['G_pred'], res['G_pinv']
        
        matrices = [G_t, G_m, G_p]
        titles = [f"True Hamiltonian\n($G_{{true}}$)", 
                  f"OGN ML Prediction (RDM Input)\n(Error: {res['dist_ml']:.3f})", 
                  f"Exact Algebraic Oracle (Full State)\n(Error: {res['dist_pinv']:.3f})"]
        
        vmax = max(np.abs(G_t).max(), np.abs(G_m).max(), np.abs(G_p).max())
        vmin = -vmax

        for i in range(3):
            ax = axes[row_idx, i]
            im = ax.imshow(matrices[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
                
            ax.set_title(titles[i], fontweight='bold')
            ax.set_xlabel("Pair Index $j$")
            
            if i == 0: 
                ax.set_ylabel(f"{res['title']}\n\nPair Index $i$", fontweight='bold', fontsize=14)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    plt.tight_layout(pad=3.0)
    plt.savefig("operator_inversion_oracle_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# EXECUTE
visualize_exact_inversion_vs_ml(val_dataset, final_state, basis, g_gen, U_ENERGY_SEED, rho_1_arrays, 
                                rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays)


# In[22]:


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm.auto import tqdm

def compute_W_symm_pairing_nd(state, rho_2_kkbar):
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

def reconstruct_G_from_symm(w_symm, M_pairs):
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

def visualize_numerical_and_physical_breakdown(val_loader, state_model, basis, g_gen, u_energy_seed, rho_1_arrays, rho_2_kkbar):
    print("\n" + "="*80)
    print(" UNCOVERING THE TWO BREAKDOWNS OF ALGEBRAIC INVERSION")
    print("="*80)

    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    
    # 1. Grab ONE representative sample
    print("[1/3] Loading a pristine sample...")
    rdm_arr, energy_arr, g_true_arr, g_pred_arr = predict_and_load(
        loader=val_loader, state=state_model, num_samples=1, gpu_batch_size=1, include_energy=True)
    
    G_base = np.array(g_gen.reconstruct(jnp.array([g_true_arr[0]])))[0]
    G_base = G_base / np.linalg.norm(G_base) # Normalize so we can scale it cleanly

    # =========================================================================
    # PART A: THE NUMERICAL NOISE EXPLOSION (SVD SPECTRUM)
    # =========================================================================
    print("[2/3] Analyzing Numerical Noise (Rank 54 vs 55)...")
    G_true_strong = G_base * 4.0 # Strong coupling
    
    H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true_strong]), 
                                        rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
    H_true_csr = _prepare_hamiltonian_for_eigsh(H_true_sp)
    H_true_csr = 0.5 * (H_true_csr + H_true_csr.T)
    e_true, v_true = scipy.sparse.linalg.eigsh(H_true_csr, k=1, which='SA', tol=1e-8)
    gs_state_strong, E_gs_strong = v_true[:, 0], e_true[0]

    W_symm_strong, _ = compute_W_symm_pairing_nd(gs_state_strong, rho_2_kkbar)
    U, S_vals_strong, Vh = scipy.linalg.svd(W_symm_strong, full_matrices=False)

    # =========================================================================
    # PART B: THE PHYSICAL PHASE TRANSITION (SWEEPING G)
    # =========================================================================
    print("[3/3] Simulating Continuous Physical Phase Transition (Sweeping G)...")
    scales = np.linspace(0.01, 3.5, 256)
    g_norms, ml_errors, oracle_errors, sigma_54_list = [], [], [], []
    
    for scale in tqdm(scales):
        G_true = G_base * scale
        
        H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true]), 
                                            rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H_true_csr = _prepare_hamiltonian_for_eigsh(H_true_sp)
        H_true_csr = 0.5 * (H_true_csr + H_true_csr.T)
        
        try:
            e_true, v_true = scipy.sparse.linalg.eigsh(H_true_csr, k=1, which='SA', tol=1e-8)
            gs_state, E_gs = v_true[:, 0], e_true[0]
        except: continue
            
        W_symm, _ = compute_W_symm_pairing_nd(gs_state, rho_2_kkbar)
        
        # SVD Spectrum to track the weakest physical mode
        S_vals = scipy.linalg.svdvals(W_symm)
        sigma_54_list.append(S_vals[53]) # The 54th singular value
        
        # Target Vector
        H0_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([np.zeros((M_pairs, M_pairs))]), 
                                        rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H0_csr = _prepare_hamiltonian_for_eigsh(H0_sp)
        H0_csr = 0.5 * (H0_csr + H0_csr.T)
        target_vec = (H0_csr @ gs_state) - E_gs * gs_state
        
        # Math Pseudoinverse (using strict 1e-8 cond to isolate the real physics, preventing the Rank 55 bug)
        w_pinv, res, rank, s = scipy.linalg.lstsq(W_symm, target_vec, cond=1e-8)
        G_pinv = reconstruct_G_from_symm(w_pinv, M_pairs).real
        G_pinv = 0.5 * (G_pinv + G_pinv.T)
        
        # Fair Gauge Shift
        shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pinv))
        G_pinv_shifted = G_pinv.copy()
        np.fill_diagonal(G_pinv_shifted, np.diag(G_pinv_shifted) + shift)
        
        # ML Prediction (we must generate the RDM for the scaled G)
        rho_2_gs = fmb.rho_m(state, rho_2_kkbar_arrays)
        bx_new = np.asarray(rho_2_gs).reshape(1, M_pairs, M_pairs, 1)
        be_new = np.array([[E_gs]]) 
        w_dummy = np.zeros(W_symm.shape[1]) 
        
        logits_new = eval_step(state_model, jnp.array(bx_new), jnp.array(be_new), jnp.array([w_dummy]))
        G_pred = np.array(g_gen.reconstruct(jnp.array([logits_new[0]])))[0]

        g_norms.append(np.linalg.norm(G_true))
        ml_errors.append(np.linalg.norm(G_pred - G_true))
        oracle_errors.append(np.linalg.norm(G_pinv_shifted - G_true))

    # =========================================================================
    # PLOTTING
    # =========================================================================
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- PANEL 1: SVD Spectrum (The Rank 54/55 Illusion) ---
    k_indices = np.arange(1, len(S_vals_strong) + 1)
    ax1.plot(k_indices, S_vals_strong, 'ko-', lw=2, markersize=6)
    
    ax1.axhline(1e-8, color='red', linestyle='--', lw=2, label='Eigensolver Noise Floor (tol=1e-8)')
    ax1.axvspan(54.5, 55.5, color='red', alpha=0.15)
    
    ax1.annotate('Physical Dimensions\n(True Rank 54)', xy=(25, S_vals_strong[25]), xytext=(5, S_vals_strong[25]*100),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold')
    ax1.annotate('Numerical Noise\n(The 55th Mode)', xy=(55, S_vals_strong[-1]), xytext=(30, S_vals_strong[-1]/1000),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='tab:red')
                 
    ax1.set_yscale('log')
    ax1.set_ylim(1e-12, 1e2)
    ax1.set_xlabel("Singular Value Index ($k$)")
    ax1.set_ylabel("Singular Value Magnitude ($\sigma_k$)")
    ax1.set_title("1. The Numerical Illusion (Rank 54 vs 55)", fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- PANEL 2: The Physical Phase Transition ---
    ax2.plot(g_norms, oracle_errors, color='tab:red', marker='X', markersize=7, linestyle='-', linewidth=2.5, 
             label='Algebraic Oracle Error')
    ax2.plot(g_norms, ml_errors, color='tab:blue', marker='o', markersize=7, linestyle='-', linewidth=2.5, 
             label='OGN ML Prediction Error')
    
    ax2.set_xlabel("True Interaction Strength ($||G_{true}||$)")
    ax2.set_ylabel("Reconstruction Error", fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(g_norms, sigma_54_list, color='tab:green', linestyle='--', linewidth=3, 
             label=r'Weakest Physical Mode ($\sigma_{54}$)')
    ax2_twin.axhline(1e-8, color='black', linestyle=':', lw=2, label='Eigensolver Noise Floor')
    ax2_twin.set_ylabel(r'Information Gap ($\sigma_{54}$)', color='tab:green', fontweight='bold')
    ax2_twin.set_yscale('log')
    ax2_twin.tick_params(axis='y', labelcolor='tab:green')
    
    ax2.set_title("2. The True Kinematic Collapse ($G \to 0$)", fontweight='bold')
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', framealpha=0.95, fontsize=11)

    plt.tight_layout(pad=3.0)
    plt.savefig("algebraic_breakdowns.png", dpi=300)
    plt.show()

# EXECUTE
visualize_numerical_and_physical_breakdown(val_dataset, final_state, basis, g_gen, U_ENERGY_SEED, rho_1_arrays, 
                                           rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays)


# In[24]:


import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import pandas as pd
from matplotlib.lines import Line2D
import traceback

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)

def plot_noise_resilience_scatter(basis, u_energy_seed, rho_1_arrays, target_rho2, g_gen, state_model, include_energy=True):
    print("="*80)
    print(" HIGH-DENSITY NOISE SCATTER: OGN DENOISING VS ORACLE")
    print("="*80)
    
    LSTSQ_COND = 1e-9  
    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    
    target_rho2_dense = _safe_dense(target_rho2)
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else target_rho2
    
    # ---------------------------------------------------------
    # Bulletproof JAX globals for Dense Physics evaluation
    # ---------------------------------------------------------
    r1_dense = _safe_dense(rho_1_arrays)
    r1_diag = np.diagonal(r1_dense, axis1=1, axis2=2) if r1_dense.ndim == 3 else np.einsum('kknn->kn', r1_dense)
    d_rho_1_diag = jnp.array(r1_diag)
    d_inter_tensor = jnp.array(target_rho2_dense)
    d_base_energies = jnp.array([base_energies])
    
    def get_rdm_from_G(G_matrix):
        """100% Robust Dense JAX Solver to bypass SciPy Sparse crashes"""
        G_jnp = jnp.array([G_matrix])
        H_batch = two_body_hamiltonian_dense(d_base_energies, G_jnp, d_rho_1_diag, d_inter_tensor)
        H_dense = np.array(H_batch[0])
        H_dense = 0.5 * (H_dense + H_dense.conj().T) # Strict Hermiticity
        
        evals, evecs = scipy.linalg.eigh(H_dense)
        return compute_rho_m(np.array([evecs[:, 0]]), target_rho2_dense, 1)[0]

    # Precompute Target Vector H0
    H0_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([np.zeros((M_pairs, M_pairs))]), d_rho_1_diag, d_inter_tensor)
    H0_dense = np.array(H0_batch[0])
    H0_dense = 0.5 * (H0_dense + H0_dense.T)

    scales = [0.5, 2.5, 6.0]
    titles = ["Weak Coupling ($||G|| \\approx 0.5$)", 
              "ML Training Region ($||G|| \\approx 2.5$)", 
              "Strong Coupling ($||G|| \\approx 6.0$)"]
    
    N_STATES = 8          
    N_NOISE = 3           
    EPSILONS = np.geomspace(1e-6, 1e-1, 25) 
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    err_print_count = 0

    for ax_idx, scale in enumerate(scales):
        print(f"\nEvaluating Regime {ax_idx+1}/3: {titles[ax_idx].replace(chr(10), ' ')}")
        ax = axes[ax_idx]
        
        # Pre-generate the true states
        true_states_data = []
        for s_idx in range(N_STATES):
            _, h_labels = g_gen.generate(jax.random.PRNGKey(1000 + ax_idx * 100 + s_idx))
            G_base = np.array(g_gen.reconstruct(h_labels))[0]
            G_true = G_base * (scale / np.linalg.norm(G_base))
            
            H_true_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([G_true]), d_rho_1_diag, d_inter_tensor)
            H_true_dense = np.array(H_true_batch[0])
            H_true_dense = 0.5 * (H_true_dense + H_true_dense.T)
            
            e_true, v_true = scipy.linalg.eigh(H_true_dense)
            gs_true = v_true[:, 0]
            
            RDM_true = compute_rho_m(np.array([gs_true]), target_rho2_dense, 1)[0]
            true_states_data.append({
                'G_true': G_true, 'H_true_dense': H_true_dense, 'gs_true': gs_true, 
                'RDM_true': RDM_true, 'norm_rdm_true': np.linalg.norm(RDM_true)
            })
            
        plot_eps, plot_ml, plot_pinv = [], [], []
        
        for eps in tqdm(EPSILONS, desc="Sweeping Noise (\u03B5)", leave=False):
            rdm_batch, e_batch, meta_batch = [], [], []
            
            # 1. Generate Noisy States
            for s_data in true_states_data:
                gs_true = s_data['gs_true']
                for trial in range(N_NOISE):
                    np.random.seed(int(eps * 1e8) % 12345 + trial * 777)
                    noise = np.random.randn(len(gs_true))
                    if np.iscomplexobj(gs_true): noise = noise + 1j * np.random.randn(len(gs_true))
                    
                    # Orthogonalize noise for a pure state perturbation
                    noise = noise - np.vdot(gs_true, noise) * gs_true
                    noise /= np.linalg.norm(noise)

                    gs_noisy = gs_true + eps * noise
                    gs_noisy /= np.linalg.norm(gs_noisy)
                    
                    RDM_noisy = compute_rho_m(np.array([gs_noisy]), target_rho2_dense, 1)[0]
                    E_noisy = np.vdot(gs_noisy, s_data['H_true_dense'] @ gs_noisy).real
                    
                    rdm_batch.append(RDM_noisy)
                    e_batch.append([E_noisy])
                    meta_batch.append({'gs_noisy': gs_noisy, 'E_noisy': E_noisy, 's_data': s_data})
                    
            # 2. Fast Batched OGN Inference
            bx = jnp.array(rdm_batch).real
            if bx.ndim == 3: bx = bx[..., jnp.newaxis]
            be = jnp.array(e_batch).real if include_energy else None
            by_dummy = jnp.zeros((len(rdm_batch), g_gen.label_size()))
            
            logits = eval_step(state_model, bx, be, by_dummy)
            G_ml_batch = np.array(g_gen.reconstruct(logits))
            
            # 3. Process Oracle and Evaluate Errors
            for i, meta in enumerate(meta_batch):
                gs_noisy, E_noisy, s_data = meta['gs_noisy'], meta['E_noisy'], meta['s_data']
                
                # Oracle pseudo-inverse
                w_out = compute_W_symm_pairing_nd(gs_noisy, target_rho2_coo)
                W_symm = w_out[0] if isinstance(w_out, tuple) else w_out 
                target_vec = (H0_dense @ gs_noisy) - E_noisy * gs_noisy
                
                try:
                    w_pinv, _, _, _ = scipy.linalg.lstsq(W_symm, target_vec, cond=LSTSQ_COND)
                    G_pinv = 0.5 * (reconstruct_G_from_symm(w_pinv, M_pairs).real + reconstruct_G_from_symm(w_pinv, M_pairs).real.T)
                except Exception:
                    G_pinv = np.zeros((M_pairs, M_pairs))
                    
                # Evaluate Errors
                try:
                    RDM_ml = get_rdm_from_G(G_ml_batch[i])
                    err_ml = np.linalg.norm(RDM_ml - s_data['RDM_true']) / s_data['norm_rdm_true']
                except Exception as e:
                    if err_print_count < 1:
                        print(f"\n[CRITICAL ML ERROR] {e}")
                        traceback.print_exc()
                        err_print_count += 1
                    err_ml = np.nan
                
                try:
                    RDM_pinv = get_rdm_from_G(G_pinv)
                    err_pinv = np.linalg.norm(RDM_pinv - s_data['RDM_true']) / s_data['norm_rdm_true']
                except Exception as e:
                    if err_print_count < 2:
                        print(f"\n[CRITICAL ORACLE ERROR] {e}")
                        err_print_count += 1
                    err_pinv = np.nan
                
                # Jitter eps horizontally for scatter visibility
                jittered_eps = eps * np.random.uniform(0.95, 1.05)
                plot_eps.append(jittered_eps)
                plot_ml.append(err_ml)
                plot_pinv.append(err_pinv)

        plot_eps = np.array(plot_eps)
        plot_ml = np.where(np.isnan(plot_ml), np.nan, np.clip(plot_ml, 1e-6, 3.0))
        plot_pinv = np.where(np.isnan(plot_pinv), np.nan, np.clip(plot_pinv, 1e-6, 3.0))
        
        df = pd.DataFrame({'eps_orig': np.repeat(EPSILONS, N_STATES * N_NOISE), 
                           'ml': plot_ml, 'pinv': plot_pinv})
        medians = df.groupby('eps_orig').median(numeric_only=True).reset_index()
        
        # Plot Scatter Clouds
        valid_pinv = ~np.isnan(plot_pinv)
        ax.scatter(plot_eps[valid_pinv], plot_pinv[valid_pinv], color='tab:red', marker='^', s=45, alpha=0.35, zorder=1)
        #ax.plot(medians['eps_orig'], medians['pinv'], 'darkred', lw=4, label='Oracle Median Trend', zorder=3)
        
        valid_ml = ~np.isnan(plot_ml)
        ax.scatter(plot_eps[valid_ml], plot_ml[valid_ml], color='tab:blue', marker='o', s=45, alpha=0.45, zorder=2)
        #ax.plot(medians['eps_orig'], medians['ml'], 'navy', lw=4, label='OGN Median Trend', zorder=4)
        
        # Baseline Reference
        #ax.plot(EPSILONS, EPSILONS, 'k--', lw=2.5, alpha=0.6, label='Injected State Noise ($\epsilon$)', zorder=0)
        
        # Styling
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(5e-5, 3.0)
        ax.set_xlim(EPSILONS.min() * 0.8, EPSILONS.max() * 1.2)
        ax.set_xlabel('State Measurement Noise ($\epsilon$)', fontweight='bold', fontsize=15)
        ax.set_title(titles[ax_idx], fontweight='bold', pad=15, fontsize=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Observable Relative Error\n($||\\Delta \\rho_2||_F / ||\\rho_{2, true}||_F$)', fontweight='bold', fontsize=15)
            custom_lines = [
                Line2D([0], [0], color='w', marker='^', markerfacecolor='tab:red', markersize=10, alpha=0.6, label='Algebraic Pseudoinversion'),
                Line2D([0], [0], color='w', marker='o', markerfacecolor='tab:blue', markersize=10, alpha=0.6, label='ML Inversion'),
            ]
            ax.legend(handles=custom_lines, loc='lower right', framealpha=0.95, fontsize=12)

    #plt.suptitle("Information Resilience: OGN Denoising vs. Oracle Full-State Inversion", fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("high_density_scatter_resilience.png", dpi=300, bbox_inches='tight')
    plt.show()

# Execute 
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
plot_noise_resilience_scatter(basis, U_ENERGY_SEED, rho_1_arrays, target_rho2, g_gen, final_state, include_energy=include_energy)


# ##### Thermal states

# In[20]:


import numpy as np
import scipy.linalg

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)
    
def build_thermal_linear_system(evals, evecs, H0_dense, rho_2_kkbar, beta, truncate_tol=1e-12):
    """
    Constructs the Augmented W and T matrices for the Thermal Oracle.
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
        
        # D. Thermal Weighting: sqrt(p_k) safely pulls noise to zero as p_k -> 0
        weight = np.sqrt(p_k)
        W_stacked.append(W_aug_k * weight)
        T_stacked.append(target_k * weight)
        
    return np.vstack(W_stacked), np.concatenate(T_stacked)

def compute_thermal_pseudoinverse(evals, evecs, H0_dense, rho_2_kkbar, beta, M_pairs, truncate_tol=1e-12):
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

def visualize_exact_inversion_vs_ml(val_loader, state_model, basis, g_gen, u_energy_seed, rho_1_arrays, rho_2_kkbar, beta=1.0):
    print("\n" + "="*80)
    print(" THERMAL ORACLE INVERSION VS. OGN RDM INFERENCE")
    print("="*80)

    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    num_symm = M_pairs * (M_pairs + 1) // 2
    
    print(f"[1/4] Loading pristine samples from validation dataset...")
    rdm_arr, energy_arr, g_true_arr, g_pred_arr = predict_and_load(
        loader=val_loader, state=state_model, num_samples=10, gpu_batch_size=10, include_energy=True)
    
    results = []
    indices_to_plot = [0, 1] 
    
    for count, idx in enumerate(indices_to_plot):
        print(f"\nProcessing Sample {count+1}/2...")
        
        G_true = np.array(g_gen.reconstruct(jnp.array([g_true_arr[idx]])))[0]
        G_pred = np.array(g_gen.reconstruct(jnp.array([g_pred_arr[idx]])))[0]
        
        print("      Computing Full Thermal Spectrum...")
        H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true]), 
                                            rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H_true_dense = 0.5 * (_safe_dense(H_true_sp) + _safe_dense(H_true_sp).T)
        evals, evecs = scipy.linalg.eigh(H_true_dense)

        # Baseline H0
        H0_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([np.zeros((M_pairs, M_pairs))]), 
                                        rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H0_dense = 0.5 * (_safe_dense(H0_sp) + _safe_dense(H0_sp).T)
        
        print(f"      Solving Massively Overdetermined Thermal Pseudoinverse...")
        G_pinv, rank, _ = compute_thermal_pseudoinverse(evals, evecs, H0_dense, rho_2_kkbar, beta, M_pairs)
        
        # Fair Gauge Shift (Corrects Particle Number Symmetry collinearity)
        shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pinv))
        G_pinv_shifted = G_pinv.copy()
        np.fill_diagonal(G_pinv_shifted, np.diag(G_pinv_shifted) + shift)
        
        dist_ml = np.linalg.norm(G_pred - G_true)
        dist_pinv = np.linalg.norm(G_pinv_shifted - G_true)
        
        results.append({
            'title': f"Validation Sample {count+1}",
            'G_true': G_true, 'G_pred': G_pred, 'G_pinv': G_pinv_shifted,
            'dist_ml': dist_ml, 'dist_pinv': dist_pinv, 'rank': rank
        })
        print(f"      --> Math Rank: {rank}/{num_symm+1} | Oracle Error: {dist_pinv:.3e} | ML Error: {dist_ml:.3f}")

    # ================= PLOTTING =================
    print("\n[4/4] Rendering Visualization...")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for row_idx, res in enumerate(results):
        matrices = [res['G_true'], res['G_pred'], res['G_pinv']]
        titles = [f"True Hamiltonian\n($G_{{true}}$)", 
                  f"OGN ML Prediction (RDM Input)\n(Error: {res['dist_ml']:.3f})", 
                  f"Thermal Algebraic Oracle\n(Error: {res['dist_pinv']:.3e})"]
        
        vmax = max(np.abs(matrices[0]).max(), np.abs(matrices[1]).max(), np.abs(matrices[2]).max())
        for i in range(3):
            ax = axes[row_idx, i]
            im = ax.imshow(matrices[i], cmap='coolwarm', vmin=-vmax, vmax=vmax)
            ax.set_title(titles[i], fontweight='bold')
            ax.set_xlabel("Pair Index $j$")
            if i == 0: ax.set_ylabel(f"{res['title']}\n\nPair Index $i$", fontweight='bold', fontsize=14)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    plt.tight_layout(pad=3.0)
    plt.savefig("operator_inversion_oracle_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def visualize_numerical_and_physical_breakdown(val_loader, state_model, basis, g_gen, u_energy_seed, rho_1_arrays, rho_2_kkbar, beta=1.0):
    print("\n" + "="*80)
    print(" RESOLVING THE ALGEBRAIC BREAKDOWNS WITH THE THERMAL ORACLE")
    print("="*80)

    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    num_symm = M_pairs * (M_pairs + 1) // 2
    
    print("[1/3] Loading a pristine sample...")
    rdm_arr, energy_arr, g_true_arr, g_pred_arr = predict_and_load(
        loader=val_loader, state=state_model, num_samples=1, gpu_batch_size=1, include_energy=True)
    
    G_base = np.array(g_gen.reconstruct(jnp.array([g_true_arr[0]])))[0]
    G_base = G_base / np.linalg.norm(G_base) 

    # Baseline H0
    H0_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([np.zeros((M_pairs, M_pairs))]), 
                                    rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
    H0_dense = 0.5 * (_safe_dense(H0_sp) + _safe_dense(H0_sp).T)

    # =========================================================================
    # PART A: THE THERMAL SVD SPECTRUM
    # =========================================================================
    print("[2/3] Analyzing Augmented Thermal Spectrum...")
    G_true_strong = G_base * 4.0 
    
    H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true_strong]), 
                                        rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
    H_true_dense = 0.5 * (_safe_dense(H_true_sp) + _safe_dense(H_true_sp).T)
    evals, evecs = scipy.linalg.eigh(H_true_dense)

    W_thermal_strong, _ = build_thermal_linear_system(evals, evecs, H0_dense, rho_2_kkbar, beta)
    U, S_vals_strong, Vh = scipy.linalg.svd(W_thermal_strong, full_matrices=False)
    
    # Compare with GS Action matrix
    W_symm_gs, _ = compute_W_symm_pairing_nd(evecs[:, 0], rho_2_kkbar)
    S_vals_gs = scipy.linalg.svdvals(W_symm_gs)

    # =========================================================================
    # PART B: SWEEPING G 
    # =========================================================================
    print("[3/3] Simulating Continuous Phase Transition (Sweeping G)...")
    scales = np.linspace(0.1, 10.0, 256)
    g_norms, ml_errors, oracle_errors, sigma_tracked_list = [], [], [], []
    
    for scale in tqdm(scales):
        G_true = G_base * scale
        
        H_true_sp = two_body_hamiltonian_sp(basis, np.array([base_energies]), np.array([G_true]), 
                                            rho_1_arrays, rho_2_kkbar, g_gen.h_type)[0]
        H_true_dense = 0.5 * (_safe_dense(H_true_sp) + _safe_dense(H_true_sp).T)
        evals, evecs = scipy.linalg.eigh(H_true_dense)
        
        # 1. Thermal Oracle Solve
        W_thermal, T_thermal = build_thermal_linear_system(evals, evecs, H0_dense, rho_2_kkbar, beta)
        
        # Track the weakest physical mode (The second-to-last singular value of the augmented matrix)
        # Because the last mode is the exact gauge symmetry (Particle Number conservation)
        S_vals = scipy.linalg.svdvals(W_thermal)
        sigma_tracked_list.append(S_vals[-2]) 
        
        x_pinv, res, rank, s = scipy.linalg.lstsq(W_thermal, T_thermal, cond=1e-8)
        G_pinv = reconstruct_G_from_symm(x_pinv[:-1], M_pairs).real
        G_pinv = 0.5 * (G_pinv + G_pinv.T)
        
        shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pinv))
        G_pinv_shifted = G_pinv.copy()
        np.fill_diagonal(G_pinv_shifted, np.diag(G_pinv_shifted) + shift)
        
        # 2. ML Prediction
        probs = np.exp(-beta * (evals - evals.min()))
        probs /= probs.sum()
        thermal_state_dense = (evecs * probs) @ evecs.T
        
        RDM_thermal = compute_rho_m(np.array([thermal_state_dense]), _safe_dense(rho_2_kkbar), 1)[0]
        E_thermal = np.sum(probs * evals)
        
        bx_new = RDM_thermal.real.reshape(1, M_pairs, M_pairs, 1)
        be_new = np.array([[E_thermal]]) 
        w_dummy = np.zeros(g_gen.label_size()) 
        
        logits_new = eval_step(state_model, jnp.array(bx_new), jnp.array(be_new), jnp.array([w_dummy]))
        G_pred = np.array(g_gen.reconstruct(jnp.array([logits_new[0]])))[0]

        # FIX 1: Apply the exact same Fair Gauge Shift to the ML model!
        shift_ml = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
        G_pred_shifted = G_pred.copy()
        np.fill_diagonal(G_pred_shifted, np.diag(G_pred_shifted) + shift_ml)

        g_norms.append(np.linalg.norm(G_true))
        
        # Calculate error on the physically aligned prediction
        ml_errors.append(np.linalg.norm(G_pred_shifted - G_true))
        oracle_errors.append(np.linalg.norm(G_pinv_shifted - G_true))

    # =========================================================================
    # PLOTTING
    # =========================================================================
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- PANEL 1: Cured Thermal SVD Spectrum ---
    k_indices = np.arange(1, len(S_vals_gs) + 1)
    ax1.plot(k_indices, S_vals_gs, 'o-', color='tab:red', lw=2, markersize=6, label='GS Action Matrix ($W_{gs}$)')
    ax1.plot(np.arange(1, len(S_vals_strong) + 1), S_vals_strong, '^-', color='tab:green', lw=2, markersize=6, label='Thermal Action Matrix ($W_{thermal}$)')
    
    ax1.axhline(1e-8, color='black', linestyle='--', lw=2, label='Eigensolver Noise Floor (tol=1e-8)')
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-15, 1e3)
    ax1.set_xlabel("Singular Value Index ($k$)")
    ax1.set_ylabel("Singular Value Magnitude ($\sigma_k$)")
    ax1.set_title("1. Resolution of the Numerical Illusion", fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- PANEL 2: The Physical Phase Transition ---
    ax2.plot(g_norms, oracle_errors, color='tab:red', marker='X', markersize=7, linestyle='-', linewidth=2.5, label='Thermal Oracle Error')
    ax2.plot(g_norms, ml_errors, color='tab:blue', marker='o', markersize=7, linestyle='-', linewidth=2.5, label='OGN ML Prediction Error')
    
    ax2.set_xlabel("True Interaction Strength ($||G_{true}||$)")
    ax2.set_ylabel("Reconstruction Error", fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(g_norms, sigma_tracked_list, color='tab:green', linestyle='--', linewidth=3, label=f'Weakest Physical Mode ($\sigma_{{{num_symm}}}$)')
    ax2_twin.axhline(1e-8, color='black', linestyle=':', lw=2, label='Eigensolver Noise Floor')
    ax2_twin.set_ylabel(f'Information Gap ($\sigma_{{{num_symm}}}$)', color='tab:green', fontweight='bold')
    ax2_twin.set_yscale('log')
    ax2_twin.tick_params(axis='y', labelcolor='tab:green')
    
    ax2.set_title("2. Immunity to Kinematic Collapse ($G \to 0$)", fontweight='bold')
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', framealpha=0.95, fontsize=11)

    plt.tight_layout(pad=3.0)
    plt.savefig("thermal_algebraic_resolution.png", dpi=300)
    plt.show()

# EXECUTION CALLS:
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
visualize_exact_inversion_vs_ml(val_dataset, final_state, basis, g_gen, U_ENERGY_SEED, rho_1_arrays, target_rho2, beta=BETA)
visualize_numerical_and_physical_breakdown(val_dataset, final_state, basis, g_gen, U_ENERGY_SEED, rho_1_arrays, target_rho2, beta=BETA)


# In[25]:


import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import pandas as pd
from matplotlib.lines import Line2D
import traceback

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)

# =========================================================================
# THERMAL ORACLE BUILDER
# =========================================================================
def build_thermal_linear_system_noisy(p_noisy, v_noisy, H0_dense, rho_2_kkbar, beta, truncate_tol=1e-10):
    """Safely builds the Oracle system from a noisy diagonalized density matrix"""
    valid_mask = p_noisy > truncate_tol
    probs = p_noisy[valid_mask]
    probs /= np.sum(probs) # Strict renormalization
    vecs = v_noisy[:, valid_mask]
    
    W_stacked, T_stacked = [], []
    for k in range(len(probs)):
        p_k = probs[k]
        state_k = vecs[:, k]
        
        W_symm_k, _ = compute_W_symm_pairing_nd(state_k, rho_2_kkbar)
        
        # Augment with state vector to absorb the Partition Function shift
        W_aug_k = np.hstack([W_symm_k, -state_k.reshape(-1, 1)])
        
        # Thermal Target: H0 + (1/beta)*ln(p_k)
        target_k = (H0_dense @ state_k) + (1.0 / beta) * np.log(p_k) * state_k
        
        # Noise-suppressing weight
        weight = np.sqrt(p_k)
        W_stacked.append(W_aug_k * weight)
        T_stacked.append(target_k * weight)
        
    return np.vstack(W_stacked), np.concatenate(T_stacked)

# =========================================================================
# EVALUATION ROUTINE
# =========================================================================
def plot_thermal_noise_resilience_scatter(val_dataset, basis, u_energy_seed, rho_1_arrays, target_rho2, g_gen, state_model, beta=1.0, include_energy=True):
    print("="*80)
    print(" THERMAL NOISE SCATTER: OGN DENOISING VS THERMAL ORACLE FRAGILITY")
    print("="*80)
    
    LSTSQ_COND = 1e-9  
    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    D_N = basis.size
    
    target_rho2_dense = _safe_dense(target_rho2)
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else target_rho2
    
    r1_dense = _safe_dense(rho_1_arrays)
    r1_diag = np.diagonal(r1_dense, axis1=1, axis2=2) if r1_dense.ndim == 3 else np.einsum('kknn->kn', r1_dense)
    d_rho_1_diag = jnp.array(r1_diag)
    d_inter_tensor = jnp.array(target_rho2_dense)
    d_base_energies = jnp.array([base_energies])
    
    def get_rdm_from_G_thermal(G_matrix):
        """Computes the exact Thermal RDM for the predicted Hamiltonian"""
        G_jnp = jnp.array([G_matrix])
        H_batch = two_body_hamiltonian_dense(d_base_energies, G_jnp, d_rho_1_diag, d_inter_tensor)
        H_dense = 0.5 * (np.array(H_batch[0]) + np.array(H_batch[0]).conj().T) 
        
        evals, evecs = scipy.linalg.eigh(H_dense)
        probs = np.exp(-beta * (evals - evals.min()))
        probs /= np.sum(probs)
        rho_th = (evecs * probs) @ evecs.T
        return compute_rho_m(np.array([rho_th]), target_rho2_dense, 1)[0]

    # Precompute Target Vector H0
    H0_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([np.zeros((M_pairs, M_pairs))]), d_rho_1_diag, d_inter_tensor)
    H0_dense = 0.5 * (np.array(H0_batch[0]) + np.array(H0_batch[0]).T)

    N_STATES = 12          
    N_NOISE = 4           
    EPSILONS = np.geomspace(1e-6, 5e-2, 30) 
    
    print(f"[1/4] Extracting {N_STATES} native samples from the validation dataset...")
    # Load pristine samples directly from your validation loader
    _, _, g_true_labels, _ = predict_and_load(
        loader=val_dataset, state=state_model, num_samples=N_STATES, 
        gpu_batch_size=min(N_STATES, 256), include_energy=include_energy
    )
    
    G_true_batch = np.array(g_gen.reconstruct(jnp.array(g_true_labels)))
    
    true_states_data = []
    for i in range(N_STATES):
        G_true = G_true_batch[i]
        H_true_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([G_true]), d_rho_1_diag, d_inter_tensor)
        H_true_dense = 0.5 * (np.array(H_true_batch[0]) + np.array(H_true_batch[0]).T)
        
        e_true, v_true = scipy.linalg.eigh(H_true_dense)
        p_true = np.exp(-beta * (e_true - e_true.min()))
        p_true /= np.sum(p_true)
        rho_full_true = (v_true * p_true) @ v_true.T
        
        RDM_true = compute_rho_m(np.array([rho_full_true]), target_rho2_dense, 1)[0]
        true_states_data.append({
            'G_true': G_true, 'H_true_dense': H_true_dense, 'rho_full_true': rho_full_true, 
            'RDM_true': np.squeeze(RDM_true).real, 'norm_rdm_true': np.linalg.norm(np.squeeze(RDM_true).real)
        })
        
    plot_eps, plot_ml, plot_pinv = [], [], []
    err_print_count = 0
    
    print(f"[2/4] Injecting Noise and Evaluating Oracle & OGN...")
    for eps in tqdm(EPSILONS, desc="Sweeping Measurement Noise (\u03B5)", leave=False):
        rdm_batch, e_batch, meta_batch = [], [], []
        
        # 1. Inject Noise into the Full Thermal State
        for s_data in true_states_data:
            rho_full_true = s_data['rho_full_true']
            for trial in range(N_NOISE):
                np.random.seed(int(eps * 1e8) % 12345 + trial * 777)
                
                # Real symmetric noise representing physical state tomography error
                noise_mat = np.random.randn(D_N, D_N)
                noise_mat = 0.5 * (noise_mat + noise_mat.T)
                noise_mat -= np.trace(noise_mat) / D_N * np.eye(D_N) 
                noise_mat /= np.linalg.norm(noise_mat)

                state_norm = np.linalg.norm(rho_full_true)
                rho_full_noisy = rho_full_true + eps * state_norm * noise_mat
                
                # Project back to a valid PSD density matrix
                p_noisy, v_noisy = scipy.linalg.eigh(rho_full_noisy)
                p_noisy = np.maximum(p_noisy, 1e-12)
                p_noisy /= np.sum(p_noisy)
                rho_noisy_valid = (v_noisy * p_noisy) @ v_noisy.T
                
                RDM_noisy = compute_rho_m(np.array([rho_noisy_valid]), target_rho2_dense, 1)[0]
                E_noisy = np.trace(s_data['H_true_dense'] @ rho_noisy_valid).real
                
                rdm_batch.append(RDM_noisy)
                e_batch.append([E_noisy])
                meta_batch.append({'p_noisy': p_noisy, 'v_noisy': v_noisy, 's_data': s_data})
                
        # 2. Fast Batched OGN Inference 
        bx = jnp.array(rdm_batch).real
        if bx.ndim == 3: bx = bx[..., jnp.newaxis]
        be = jnp.array(e_batch).real if include_energy else None
        by_dummy = jnp.zeros((len(rdm_batch), g_gen.label_size()))
        
        logits = eval_step(state_model, bx, be, by_dummy)
        G_ml_batch = np.array(g_gen.reconstruct(logits))
        
        # 3. Process Oracle & Evaluate Errors
        for i, meta in enumerate(meta_batch):
            p_noisy, v_noisy, s_data = meta['p_noisy'], meta['v_noisy'], meta['s_data']
            
            # A. Oracle pseudo-inverse (Requires FULL NOISY SPECTRUM)
            try:
                W_th, T_th = build_thermal_linear_system_noisy(p_noisy, v_noisy, H0_dense, target_rho2_coo, beta)
                w_pinv, _, _, _ = scipy.linalg.lstsq(W_th, T_th, cond=LSTSQ_COND)
                G_pinv = 0.5 * (reconstruct_G_from_symm(w_pinv[:-1], M_pairs).real + reconstruct_G_from_symm(w_pinv[:-1], M_pairs).real.T)
            except Exception:
                G_pinv = np.zeros((M_pairs, M_pairs))
                
            # B. Evaluate Observable Errors safely using .flatten()
            try:
                RDM_ml = np.squeeze(get_rdm_from_G_thermal(G_ml_batch[i])).real
                err_ml = float(np.linalg.norm(RDM_ml.flatten() - s_data['RDM_true'].flatten()) / s_data['norm_rdm_true'])
            except Exception as e:
                if err_print_count < 1: 
                    print(f"\n[ML Error Caught safely]: {e}")
                    traceback.print_exc()
                    err_print_count += 1
                err_ml = np.nan
            
            try:
                RDM_pinv = np.squeeze(get_rdm_from_G_thermal(G_pinv)).real
                err_pinv = float(np.linalg.norm(RDM_pinv.flatten() - s_data['RDM_true'].flatten()) / s_data['norm_rdm_true'])
            except Exception:
                err_pinv = np.nan
            
            jittered_eps = eps * np.random.uniform(0.95, 1.05)
            plot_eps.append(jittered_eps)
            plot_ml.append(err_ml)
            plot_pinv.append(err_pinv)

    print(f"[3/4] Processing Error Distributions...")
    plot_eps = np.array(plot_eps)
    plot_ml = np.where(np.isnan(plot_ml), np.nan, np.clip(plot_ml, 1e-6, 3.0))
    plot_pinv = np.where(np.isnan(plot_pinv), np.nan, np.clip(plot_pinv, 1e-6, 3.0))
    
    # Calculate Median Trends
    df = pd.DataFrame({'eps_orig': np.repeat(EPSILONS, N_STATES * N_NOISE), 'ml': plot_ml, 'pinv': plot_pinv})
    medians = df.groupby('eps_orig').median(numeric_only=True).reset_index()
    
    print(f"[4/4] Rendering Plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    eps_vals = medians['eps_orig'].values
    ml_vals = medians['ml'].values
    pinv_vals = medians['pinv'].values

    # Highlight ML Supremacy Region
    valid_fill = ~(np.isnan(ml_vals) | np.isnan(pinv_vals))
    ax.fill_between(eps_vals[valid_fill], ml_vals[valid_fill], pinv_vals[valid_fill], 
                    where=(pinv_vals[valid_fill] > ml_vals[valid_fill]), 
                    interpolate=True, color='lime', alpha=0.15, label='Regime of Neural Supremacy')

    # Plot Scatter Clouds
    valid_pinv = ~np.isnan(plot_pinv)
    ax.scatter(plot_eps[valid_pinv], plot_pinv[valid_pinv], color='tab:red', marker='^', s=60, alpha=0.35, zorder=1)
    valid_ml = ~np.isnan(plot_ml)
    ax.scatter(plot_eps[valid_ml], plot_ml[valid_ml], color='tab:blue', marker='o', s=60, alpha=0.45, zorder=2)
    
    ax.plot(eps_vals, pinv_vals, 'darkred', lw=4, label='Thermal Oracle Median', zorder=3)
    ax.plot(eps_vals, ml_vals, 'navy', lw=4, label='OGN Median', zorder=4)

    # Baselines
    ax.axhline(1e-4, color='gray', linestyle=':', lw=2.5, alpha=0.8, label='Theoretical Float32 NN Precision', zorder=0)
    ax.plot(EPSILONS, EPSILONS, 'k--', lw=2.5, alpha=0.6, label='State Measurement Noise ($\epsilon$)', zorder=0)
    
    # Styling
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(2e-5, 3.0) 
    ax.set_xlim(EPSILONS.min() * 0.8, EPSILONS.max() * 1.2)
    ax.set_xlabel('Relative State Measurement Noise ($\epsilon$)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Observable Relative Error\n($||\\Delta \\rho_2||_F / ||\\rho_{2, true}||_F$)', fontweight='bold', fontsize=16)
    ax.set_title("Information Resilience: OGN Denoising vs Thermal Oracle Fragility", fontweight='bold', pad=15, fontsize=18)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    custom_lines = [
        Line2D([0], [0], color='tab:red', marker='^', lw=4, markersize=10, alpha=0.8, label=f'Algebraic Oracle (Inputs: {D_N**2})'),
        Line2D([0], [0], color='navy', marker='o', lw=4, markersize=10, alpha=0.8, label=f'OGN Neural Denoiser (Inputs: {M_pairs**2})'),
        plt.Rectangle((0,0),1,1, color='lime', alpha=0.3, label='Regime of Neural Supremacy')
    ]
    ax.legend(handles=custom_lines, loc='lower right', framealpha=0.95, fontsize=14)

    plt.tight_layout()
    plt.savefig("thermal_noise_scatter_resilience_native.png", dpi=300, bbox_inches='tight')
    plt.show()

# EXECUTE
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
plot_thermal_noise_resilience_scatter(val_dataset, basis, U_ENERGY_SEED, rho_1_arrays, target_rho2, g_gen, final_state, beta=BETA, include_energy=include_energy)


# In[26]:


import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import pandas as pd
from matplotlib.lines import Line2D
import traceback

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)
    
def plot_thermal_noise_resilience_scatter_bare_G(val_dataset, basis, u_energy_seed, rho_1_arrays, target_rho2, g_gen, state_model, beta=1.0, include_energy=True):
    print("="*80)
    print(" THERMAL NOISE SCATTER (BARE G): OGN DENOISING VS THERMAL ORACLE FRAGILITY")
    print("="*80)
    
    LSTSQ_COND = 1e-9  
    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    D_N = basis.size
    
    target_rho2_dense = _safe_dense(target_rho2)
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else target_rho2
    
    r1_dense = _safe_dense(rho_1_arrays)
    r1_diag = np.diagonal(r1_dense, axis1=1, axis2=2) if r1_dense.ndim == 3 else np.einsum('kknn->kn', r1_dense)
    d_rho_1_diag = jnp.array(r1_diag)
    d_inter_tensor = jnp.array(target_rho2_dense)
    d_base_energies = jnp.array([base_energies])

    # Precompute Target Vector H0
    H0_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([np.zeros((M_pairs, M_pairs))]), d_rho_1_diag, d_inter_tensor)
    H0_dense = 0.5 * (np.array(H0_batch[0]) + np.array(H0_batch[0]).T)

    N_STATES = 6          
    N_NOISE = 4           
    EPSILONS = np.geomspace(1e-6, 5e-2, 60) 
    
    print(f"[1/4] Extracting {N_STATES} native samples from the validation dataset...")
    # Load pristine samples directly from your validation loader
    _, _, g_true_labels, _ = predict_and_load(
        loader=val_dataset, state=state_model, num_samples=N_STATES, 
        gpu_batch_size=min(N_STATES, 256), include_energy=include_energy
    )
    
    G_true_batch = np.array(g_gen.reconstruct(jnp.array(g_true_labels)))
    
    true_states_data = []
    for i in range(N_STATES):
        G_true = G_true_batch[i]
        H_true_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([G_true]), d_rho_1_diag, d_inter_tensor)
        H_true_dense = 0.5 * (np.array(H_true_batch[0]) + np.array(H_true_batch[0]).T)
        
        e_true, v_true = scipy.linalg.eigh(H_true_dense)
        p_true = np.exp(-beta * (e_true - e_true.min()))
        p_true /= np.sum(p_true)
        rho_full_true = (v_true * p_true) @ v_true.T
        
        # Save the true G matrix and its norm for direct relative errors
        true_states_data.append({
            'G_true': G_true, 
            'H_true_dense': H_true_dense, 
            'rho_full_true': rho_full_true, 
            'norm_g_true': np.linalg.norm(G_true) 
        })
        
    plot_eps, plot_ml, plot_pinv = [], [], []
    err_print_count = 0
    
    print(f"[2/4] Injecting Noise and Evaluating Oracle & OGN...")
    for eps in tqdm(EPSILONS, desc="Sweeping Measurement Noise (\u03B5)", leave=False):
        rdm_batch, e_batch, meta_batch = [], [], []
        
        # 1. Inject Noise into the Full Thermal State
        for s_data in true_states_data:
            rho_full_true = s_data['rho_full_true']
            for trial in range(N_NOISE):
                np.random.seed(int(eps * 1e8) % 12345 + trial * 777)
                
                # Real symmetric noise representing physical state tomography error
                noise_mat = np.random.randn(D_N, D_N)
                noise_mat = 0.5 * (noise_mat + noise_mat.T)
                noise_mat -= np.trace(noise_mat) / D_N * np.eye(D_N) 
                noise_mat /= np.linalg.norm(noise_mat)

                state_norm = np.linalg.norm(rho_full_true)
                rho_full_noisy = rho_full_true + eps * state_norm * noise_mat
                
                # Project back to a valid PSD density matrix
                p_noisy, v_noisy = scipy.linalg.eigh(rho_full_noisy)
                p_noisy = np.maximum(p_noisy, 1e-12)
                p_noisy /= np.sum(p_noisy)
                rho_noisy_valid = (v_noisy * p_noisy) @ v_noisy.T
                
                RDM_noisy = compute_rho_m(np.array([rho_noisy_valid]), target_rho2_dense, 1)[0]
                E_noisy = np.trace(s_data['H_true_dense'] @ rho_noisy_valid).real
                
                rdm_batch.append(RDM_noisy)
                e_batch.append([E_noisy])
                meta_batch.append({'p_noisy': p_noisy, 'v_noisy': v_noisy, 's_data': s_data})
                
        # 2. Fast Batched OGN Inference 
        bx = jnp.array(rdm_batch).real
        if bx.ndim == 3: bx = bx[..., jnp.newaxis]
        be = jnp.array(e_batch).real if include_energy else None
        by_dummy = jnp.zeros((len(rdm_batch), g_gen.label_size()))
        
        logits = eval_step(state_model, bx, be, by_dummy)
        G_ml_batch = np.array(g_gen.reconstruct(logits))
        
        # 3. Process Oracle & Evaluate Parameter Errors
        for i, meta in enumerate(meta_batch):
            p_noisy, v_noisy, s_data = meta['p_noisy'], meta['v_noisy'], meta['s_data']
            G_true = s_data['G_true']
            norm_g_true = s_data['norm_g_true']
            
            # A. Oracle pseudo-inverse
            try:
                W_th, T_th = build_thermal_linear_system_noisy(p_noisy, v_noisy, H0_dense, target_rho2_coo, beta)
                w_pinv, _, _, _ = scipy.linalg.lstsq(W_th, T_th, cond=LSTSQ_COND)
                G_pinv = 0.5 * (reconstruct_G_from_symm(w_pinv[:-1], M_pairs).real + reconstruct_G_from_symm(w_pinv[:-1], M_pairs).real.T)
            except Exception:
                G_pinv = np.zeros((M_pairs, M_pairs))
                
            # B. Fair Gauge Shift & Error Evaluation on the Bare Parameters directly
            try:
                G_ml = G_ml_batch[i]
                shift_ml = np.mean(np.diag(G_true)) - np.mean(np.diag(G_ml))
                G_ml_shifted = G_ml.copy()
                np.fill_diagonal(G_ml_shifted, np.diag(G_ml_shifted) + shift_ml)
                
                err_ml = float(np.linalg.norm(G_ml_shifted - G_true) / norm_g_true)
            except Exception as e:
                if err_print_count < 1: 
                    print(f"\n[ML Error Caught safely]: {e}")
                    traceback.print_exc()
                    err_print_count += 1
                err_ml = np.nan
            
            try:
                shift_pinv = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pinv))
                G_pinv_shifted = G_pinv.copy()
                np.fill_diagonal(G_pinv_shifted, np.diag(G_pinv_shifted) + shift_pinv)
                
                err_pinv = float(np.linalg.norm(G_pinv_shifted - G_true) / norm_g_true)
            except Exception:
                err_pinv = np.nan
            
            jittered_eps = eps * np.random.uniform(0.95, 1.05)
            plot_eps.append(jittered_eps)
            plot_ml.append(err_ml)
            plot_pinv.append(err_pinv)

    print(f"[3/4] Processing Error Distributions...")
    plot_eps = np.array(plot_eps)
    
    # We clip parameters slightly higher since algebraic inversion errors explode 
    # worse in unbounded parameter space compared to bounded probability spaces
    plot_ml = np.where(np.isnan(plot_ml), np.nan, np.clip(plot_ml, 1e-6, 10.0))
    plot_pinv = np.where(np.isnan(plot_pinv), np.nan, np.clip(plot_pinv, 1e-6, 10.0))
    
    # Calculate Median Trends
    df = pd.DataFrame({'eps_orig': np.repeat(EPSILONS, N_STATES * N_NOISE), 'ml': plot_ml, 'pinv': plot_pinv})
    medians = df.groupby('eps_orig').median(numeric_only=True).reset_index()
    
    print(f"[4/4] Rendering Plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    eps_vals = medians['eps_orig'].values
    ml_vals = medians['ml'].values
    pinv_vals = medians['pinv'].values

    # Plot Scatter Clouds
    valid_pinv = ~np.isnan(plot_pinv)
    ax.scatter(plot_eps[valid_pinv], plot_pinv[valid_pinv], color='tab:red', marker='^', s=60, alpha=0.35, zorder=1)
    valid_ml = ~np.isnan(plot_ml)
    ax.scatter(plot_eps[valid_ml], plot_ml[valid_ml], color='tab:blue', marker='o', s=60, alpha=0.45, zorder=2)
    
    ax.plot(eps_vals, pinv_vals, 'darkred', lw=4, label='Algebraic Pseudoinversion Median', zorder=3)
    ax.plot(eps_vals, ml_vals, 'navy', lw=4, label='OGN Median', zorder=4)
    
    # Styling
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(2e-5, 10.0) 
    ax.set_xlim(EPSILONS.min() * 0.8, EPSILONS.max() * 1.2)
    
    ax.set_xlabel('State Measurement Noise ($\epsilon$)', fontweight='bold', fontsize=16)
    
    # Updated Y-Axis Label
    ax.set_ylabel('Parameter Relative Error\n($||\\Delta G||_F / ||G_{true}||_F$)', fontweight='bold', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    custom_lines = [
        Line2D([0], [0], color='tab:red', marker='^', lw=4, markersize=10, alpha=0.8, label=f'Algebraic Pseudoinverse'),
        Line2D([0], [0], color='navy', marker='o', lw=4, markersize=10, alpha=0.8, label=f'OGN Prediction'),
    ]
    ax.legend(handles=custom_lines, loc='lower right', framealpha=0.95, fontsize=14)

    plt.tight_layout()
    plt.savefig("thermal_noise_scatter_resilience_bare_G.png", dpi=300, bbox_inches='tight')
    plt.show()

# EXECUTE
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
plot_thermal_noise_resilience_scatter_bare_G(val_dataset, basis, U_ENERGY_SEED, rho_1_arrays, target_rho2, g_gen, final_state, beta=BETA, include_energy=include_energy)


# ## Logarithm based inversion

# In[34]:


import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import pandas as pd
from matplotlib.lines import Line2D

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)

# =========================================================================
# 1. EXACT LOG INVERSION BUILDER
# =========================================================================

def precompute_A_matrix(rho_2_kkbar, M_pairs, D_N):
    """
    Precomputes the geometric projection matrix A for the Log Inversion.
    Maps the interaction parameters G_IJ and the scalar shift to the Hamiltonian space.
    """
    num_symm = M_pairs * (M_pairs + 1) // 2
    A = np.zeros((D_N * D_N, num_symm + 1), dtype=np.float64)
    
    rho_2_coo = rho_2_kkbar.tocoo() if hasattr(rho_2_kkbar, 'tocoo') else scipy.sparse.coo_matrix(rho_2_kkbar)
    coords = rho_2_coo.coords
    data = rho_2_coo.data
    
    J_coords, I_coords = coords[0], coords[1]
    r_coords, c_coords = coords[2], coords[3]
    
    symm_idx = np.zeros((M_pairs, M_pairs), dtype=int)
    idx = 0
    for I in range(M_pairs):
        symm_idx[I, I] = idx
        idx += 1
    for I in range(M_pairs):
        for J in range(I + 1, M_pairs):
            symm_idx[I, J] = symm_idx[J, I] = idx
            idx += 1
            
    # Matrix A expects H_I = - G_IJ C^\dagger_I C_J
    row_indices = r_coords * D_N + c_coords
    col_indices = symm_idx[I_coords, J_coords]
    np.add.at(A, (row_indices, col_indices), -data.real)
    
    # Last column is the Identity matrix (to absorb Partition Function shift)
    idx_identity = np.arange(D_N)
    A[idx_identity * D_N + idx_identity, -1] = 1.0
    return A

def reconstruct_G_from_symm(w_symm, M_pairs):
    """Rebuilds the symmetric G matrix from the flattened parameters"""
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

# =========================================================================
# 2. EVALUATION ROUTINE (FINITE SHOT NOISE)
# =========================================================================

def plot_shot_noise_resilience(val_dataset, basis, u_energy_seed, rho_1_arrays, target_rho2, g_gen, state_model, beta=1.0, include_energy=True):
    print("="*80)
    print(" EXPERIMENTAL SHOT NOISE: OGN ROBUSTNESS VS LOG CATASTROPHE")
    print("="*80)
    
    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    D_N = basis.size
    
    target_rho2_dense = _safe_dense(target_rho2)
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else target_rho2
    
    r1_dense = _safe_dense(rho_1_arrays)
    r1_diag = np.diagonal(r1_dense, axis1=1, axis2=2) if r1_dense.ndim == 3 else np.einsum('kknn->kn', r1_dense)
    d_rho_1_diag = jnp.array(r1_diag)
    d_inter_tensor = jnp.array(target_rho2_dense)
    d_base_energies = jnp.array([base_energies])

    # 1. Precompute Baseline H0
    H0_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([np.zeros((M_pairs, M_pairs))]), d_rho_1_diag, d_inter_tensor)
    H0_dense = 0.5 * (np.array(H0_batch[0]) + np.array(H0_batch[0]).T)

    # 2. Precompute Log Inversion A Matrix & Pseudo-inverse
    print("[1/4] Precomputing Log Inversion Projection Matrix A_pinv...")
    A_matrix = precompute_A_matrix(target_rho2_coo, M_pairs, D_N)
    A_pinv = scipy.linalg.pinv(A_matrix)

    N_STATES = 8          
    N_NOISE = 3           
    
    # Sweep N_shots from 100 Million down to 100
    # Plotted as Statistical Error epsilon = 1 / sqrt(N_shots)
    N_SHOTS_SWEEP = np.geomspace(1e12, 1e6, 60).astype(int) 
    EPSILONS = 1.0 / np.sqrt(N_SHOTS_SWEEP)
    
    print(f"[2/4] Extracting {N_STATES} pristine Hamiltonians...")
    _, _, g_true_labels, _ = predict_and_load(
        loader=val_dataset, state=state_model, num_samples=N_STATES, 
        gpu_batch_size=min(N_STATES, 256), include_energy=include_energy
    )
    G_true_batch = np.array(g_gen.reconstruct(jnp.array(g_true_labels)))
    
    true_systems = []
    for i in range(N_STATES):
        G_true = G_true_batch[i]
        H_true_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([G_true]), d_rho_1_diag, d_inter_tensor)
        H_true_dense = 0.5 * (np.array(H_true_batch[0]) + np.array(H_true_batch[0]).T)
        
        E_true, V_true = scipy.linalg.eigh(H_true_dense)
        p_true = np.exp(-beta * (E_true - E_true.min())).astype(np.float64)
        p_true /= np.sum(p_true) 
        
        true_systems.append({
            'G_true': G_true, 
            'E_true': E_true,
            'V_true': V_true,
            'p_true': p_true,
            'norm_g_true': np.linalg.norm(G_true) 
        })
        
    plot_eps, plot_ml, plot_log = [], [], []
    
    print(f"[3/4] Simulating Multinomial Measurement Shots and Evaluating Solvers...")
    for idx_sweep, N_shots in enumerate(tqdm(N_SHOTS_SWEEP, desc="Sweeping Finite Shots", leave=False)):
        eps = EPSILONS[idx_sweep]
        rdm_batch, e_batch, meta_batch = [], [], []
        
        for sys_data in true_systems:
            p_true = sys_data['p_true']
            V_true = sys_data['V_true']
            
            for trial in range(N_NOISE):
                np.random.seed(int(eps * 1e8) % 12345 + trial * 777)
                
                # --- EXPERIMENTAL MEASUREMENT (WITH FLOAT64 FIX) ---
                # We scale probabilities down by a microscopic amount (1e-10). 
                # This guarantees sum(p) < 1.0, totally bypassing the C-level multinomial crash.
                p_safe_draw = p_true * (1.0 - 1e-10)
                counts = np.random.multinomial(N_shots, p_safe_draw)
                
                # Renormalize by actual drawn counts to ensure trace = 1.0
                total_counts = np.sum(counts)
                p_meas = counts / float(total_counts) if total_counts > 0 else counts / float(N_shots)
                # ---------------------------------------------------
                
                # 2. Empirical Measured State
                rho_meas = (V_true * p_meas) @ V_true.T
                RDM_meas = compute_rho_m(np.array([rho_meas]), target_rho2_dense, 1)[0]
                E_meas = np.sum(p_meas * sys_data['E_true'])
                
                rdm_batch.append(RDM_meas)
                e_batch.append([E_meas])
                meta_batch.append({'p_meas': p_meas, 'V_true': V_true, 'sys_data': sys_data})
                
        # Fast Batched OGN Inference 
        bx = jnp.array(rdm_batch).real
        if bx.ndim == 3: bx = bx[..., jnp.newaxis]
        be = jnp.array(e_batch).real if include_energy else None
        by_dummy = jnp.zeros((len(rdm_batch), g_gen.label_size()))
        
        logits = eval_step(state_model, bx, be, by_dummy)
        G_ml_batch = np.array(g_gen.reconstruct(logits))
        
        for i, meta in enumerate(meta_batch):
            p_meas, V_true = meta['p_meas'], meta['V_true']
            sys_data = meta['sys_data']
            G_true, norm_g = sys_data['G_true'], sys_data['norm_g_true']
            
            # -----------------------------------------------------------------
            # Log Inversion (Requires artificial smoothing for p=0)
            # -----------------------------------------------------------------
            try:
                # Experimental Fix: Add a tiny Laplace smoothing to prevent -infinity
                p_safe = np.maximum(p_meas, 1e-12)
                p_safe /= np.sum(p_safe)
                
                H_eff = - (1.0 / beta) * (V_true * np.log(p_safe)) @ V_true.T
                b_vec = (H_eff - H0_dense).real.flatten()
                
                w_log = A_pinv @ b_vec
                G_log = 0.5 * (reconstruct_G_from_symm(w_log[:-1], M_pairs).real + reconstruct_G_from_symm(w_log[:-1], M_pairs).real.T)
            except Exception:
                G_log = np.zeros((M_pairs, M_pairs))

            # -----------------------------------------------------------------
            # Fair Gauge Shift & Error Evaluation
            # -----------------------------------------------------------------
            def compute_shifted_error(G_pred):
                shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
                G_shifted = G_pred.copy()
                np.fill_diagonal(G_shifted, np.diag(G_shifted) + shift)
                return float(np.linalg.norm(G_shifted - G_true) / norm_g)
            
            try: err_ml = compute_shifted_error(G_ml_batch[i])
            except: err_ml = np.nan

            try: err_log = compute_shifted_error(G_log)
            except: err_log = np.nan
            
            # Jitter slightly for scatter visualization
            jittered_eps = eps * np.random.uniform(0.95, 1.05)
            plot_eps.append(jittered_eps)
            plot_ml.append(err_ml)
            plot_log.append(err_log)

    print(f"[4/4] Processing Error Distributions and Rendering Plot...")
    plot_eps = np.array(plot_eps)
    
    # Cap errors so Log Inversion's massive failures stay safely on the plot
    plot_ml = np.where(np.isnan(plot_ml), np.nan, plot_ml)
    plot_log = np.where(np.isnan(plot_log), np.nan, np.clip(plot_log, 1e-6, 10.0))
    
    df = pd.DataFrame({'eps_orig': np.repeat(EPSILONS, N_STATES * N_NOISE), 
                       'ml': plot_ml, 'log': plot_log})
    medians = df.groupby('eps_orig').median(numeric_only=True).reset_index()
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    eps_vals = medians['eps_orig'].values
    ml_vals = medians['ml'].values
    log_vals = medians['log'].values

    # Highlight where OGN beats the Algebraic method
    valid_fill = ~(np.isnan(ml_vals) | np.isnan(log_vals))
    ax.fill_between(eps_vals[valid_fill], ml_vals[valid_fill], log_vals[valid_fill], 
                    where=(log_vals[valid_fill] > ml_vals[valid_fill]), 
                    interpolate=True, color='lime', alpha=0.15, label='Regime of Absolute Neural Supremacy')

    # Scatter Clouds
    valid_log = ~np.isnan(plot_log)
    ax.scatter(plot_eps[valid_log], plot_log[valid_log], color='tab:green', marker='s', s=45, alpha=0.35, zorder=1)
    
    valid_ml = ~np.isnan(plot_ml)
    ax.scatter(plot_eps[valid_ml], plot_ml[valid_ml], color='tab:blue', marker='o', s=45, alpha=0.45, zorder=2)
    
    # Medians
    ax.plot(eps_vals, log_vals, 'darkgreen', lw=4, label='Log Inversion Median', zorder=4)
    ax.plot(eps_vals, ml_vals, 'navy', lw=4, label='OGN Median', zorder=5)
    
    # Theoretical Baseline (1/sqrt(N) scaling)
    ax.plot(EPSILONS, EPSILONS, 'k--', lw=2.5, alpha=0.6, label='Theoretical Statistical Error Scaling ($\propto 1/\sqrt{N_{shots}}$)', zorder=0)
    
    # Top X-axis for Number of Shots
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1/(x**2), lambda x: 1/np.sqrt(x)))
    ax_top.set_xlabel('Number of Experimental Measurement Shots ($N_{shots}$)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Bottom Styling
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-3, 10.0) 
    ax.set_xlim(EPSILONS.min() * 0.8, EPSILONS.max() * 1.2)
    
    ax.set_xlabel('Statistical Measurement Noise ($\epsilon_{shot} = 1/\sqrt{N_{shots}}$)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Parameter Relative Error\n($||\\Delta G||_F / ||G_{true}||_F$)', fontweight='bold', fontsize=16)
    ax.set_title("The Finite Sampling Catastrophe:\nOGN vs Mathematical Log Inversion", fontweight='bold', pad=30, fontsize=18)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    custom_lines = [
        Line2D([0], [0], color='tab:green', marker='s', lw=4, markersize=10, alpha=0.8, label='Log Inversion (Mathematical Projection)'),
        Line2D([0], [0], color='navy', marker='o', lw=4, markersize=10, alpha=0.8, label='OGN Prediction (2-RDM Input)'),
        plt.Rectangle((0,0),1,1, color='lime', alpha=0.3, label='Regime of Neural Supremacy')
    ]
    ax.legend(handles=custom_lines, loc='lower right', framealpha=0.95, fontsize=14)

    plt.tight_layout()
    plt.savefig("finite_shot_noise_catastrophe.png", dpi=300, bbox_inches='tight')
    plt.show()

# =========================================================================
# 3. EXECUTION
# =========================================================================
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays

# Execute the benchmark
plot_shot_noise_resilience(
    val_dataset, basis, U_ENERGY_SEED, rho_1_arrays, target_rho2, 
    g_gen, final_state, beta=BETA, include_energy=include_energy
)


# In[ ]:


import numpy as np
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import pandas as pd
from matplotlib.lines import Line2D

def _safe_dense(arr):
    if scipy.sparse.issparse(arr): return arr.toarray()
    if hasattr(arr, 'todense'): return np.array(arr.todense())
    return np.asarray(arr, dtype=np.float64)

def reconstruct_G_from_symm(w_symm, M_pairs):
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

# =========================================================================
# EVALUATION ROUTINE (WLS vs OGN)
# =========================================================================

def plot_asymptotic_shot_noise_resilience_wls(val_dataset, basis, u_energy_seed, rho_1_arrays, target_rho2, g_gen, state_model, beta=1.0, include_energy=True):
    print("="*80)
    print(" ASYMPTOTIC SHOT NOISE: WLS LOG INVERSION AND GEOMETRIC REGULARIZATION")
    print("="*80)
    
    M_pairs = basis.d // 2
    base_energies = u_energy_seed[0]
    D_N = basis.size
    
    target_rho2_dense = _safe_dense(target_rho2)
    target_rho2_coo = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else target_rho2
    
    r1_dense = _safe_dense(rho_1_arrays)
    r1_diag = np.diagonal(r1_dense, axis1=1, axis2=2) if r1_dense.ndim == 3 else np.einsum('kknn->kn', r1_dense)
    d_rho_1_diag = jnp.array(r1_diag)
    d_inter_tensor = jnp.array(target_rho2_dense)
    d_base_energies = jnp.array([base_energies])

    H0_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([np.zeros((M_pairs, M_pairs))]), d_rho_1_diag, d_inter_tensor)
    H0_dense = 0.5 * (np.array(H0_batch[0]) + np.array(H0_batch[0]).T)

    N_STATES = 100          
    N_NOISE = 100          
    
    # Focus the computational sweep strictly on the realistic experimental sparsity frontier
    N_SHOTS_SWEEP = np.geomspace(1e10, 1e2, 50) 
    EPSILONS = 1.0 / np.sqrt(N_SHOTS_SWEEP)
    
    print(f"[1/3] Extracting {N_STATES} pristine Hamiltonians and precomputing tensor actions...")
    _, _, g_true_labels, _ = predict_and_load(
        loader=val_dataset, state=state_model, num_samples=N_STATES, 
        gpu_batch_size=min(N_STATES, 256), include_energy=include_energy
    )
    G_true_batch = np.array(g_gen.reconstruct(jnp.array(g_true_labels)))
    
    true_systems = []
    for i in range(N_STATES):
        G_true = G_true_batch[i]
        H_true_batch = two_body_hamiltonian_dense(d_base_energies, jnp.array([G_true]), d_rho_1_diag, d_inter_tensor)
        H_true_dense = 0.5 * (np.array(H_true_batch[0]) + np.array(H_true_batch[0]).T)
        
        E_true, V_true = scipy.linalg.eigh(H_true_dense)
        p_true = np.exp(-beta * (E_true - E_true.min())).astype(np.float64)
        p_true /= np.sum(p_true) 
        
        # [REGEN-PATCH P3] Precompute the DIAGONAL eigenstate projections only:
        # one row <k|.|k> per eigenstate (manuscript Sec. III.C.2; the
        # off-diagonal equations are noise-free and must not be fitted).
        sys_W_rows = []
        sys_h0_diag = []
        for k in range(D_N):
            state_k = V_true[:, k]
            W_symm_k, _ = compute_W_symm_pairing_nd(state_k, target_rho2_coo)
            W_aug_k = np.hstack([W_symm_k, -state_k.reshape(-1, 1)])
            sys_W_rows.append(state_k @ W_aug_k)              # shape (n_params + 1,)
            sys_h0_diag.append(float(state_k @ (H0_dense @ state_k)))

        true_systems.append({
            'G_true': G_true, 'E_true': E_true, 'V_true': V_true,
            'p_true': p_true, 'norm_g_true': np.linalg.norm(G_true),
            'W_rows': np.array(sys_W_rows),
            'h0_diag': np.array(sys_h0_diag),
        })
        
    plot_eps, plot_ml, plot_log_wls = [], [], []
    
    print(f"[2/3] Simulating Asymptotic Shot Noise (CLT) & WLS Filtering...")
    for idx_sweep, N_shots in enumerate(tqdm(N_SHOTS_SWEEP, desc="Sweeping Finite Shots", leave=False)):
        eps = EPSILONS[idx_sweep]
        rdm_batch, e_batch, meta_batch = [], [], []
        
        for state_idx, sys_data in enumerate(true_systems):
            p_true = sys_data['p_true']
            V_true = sys_data['V_true']

            for trial in range(N_NOISE):
                # [REGEN-PATCH SEED] collision-free deterministic stream per
                # (shot, state, trial); same draw feeds both estimators.
                rng = np.random.default_rng(np.random.SeedSequence(
                    [(GLOBAL_SEED if 'GLOBAL_SEED' in globals() else 0),
                     2, idx_sweep, state_idx, trial]))

                # Asymptotic measurement using Central Limit Theorem
                # [REGEN-PATCH P4] removed the undocumented sum-zero
                # recentering (absent from Eq. noise_model /
                # Eq. implemented_covariance); the renormalization below
                # already restores sum(p) = 1.
                noise = rng.standard_normal(len(p_true)) * np.sqrt(p_true * (1.0 - p_true) / N_shots)
                p_raw = p_true + noise
                
                # Zero-clipped normalization (No Laplace Smoothing)
                pseudo_counts = np.maximum(p_raw * N_shots, 0.0)
                sum_counts = np.sum(pseudo_counts)
                
                # Normalize or fallback if statistical noise wipes out everything
                p_meas = pseudo_counts / sum_counts if sum_counts > 0 else p_true.copy()
                
                rho_meas = (V_true * p_meas) @ V_true.T
                RDM_meas = compute_rho_m(np.array([rho_meas]), target_rho2_dense, 1)[0]
                E_meas = np.sum(p_meas * sys_data['E_true'])
                
                rdm_batch.append(RDM_meas)
                e_batch.append([E_meas])
                meta_batch.append({'p_meas': p_meas, 'sys_data': sys_data})
                
        bx = jnp.array(rdm_batch).real
        if bx.ndim == 3: bx = bx[..., jnp.newaxis]
        be = jnp.array(e_batch).real if include_energy else None
        by_dummy = jnp.zeros((len(rdm_batch), g_gen.label_size()))
        
        logits = eval_step(state_model, bx, be, by_dummy)
        G_ml_batch = np.array(g_gen.reconstruct(logits))
        
        for i, meta in enumerate(meta_batch):
            p_meas = meta['p_meas']
            sys_data = meta['sys_data']
            G_true, norm_g = sys_data['G_true'], sys_data['norm_g_true']
            
            # -----------------------------------------------------------------
            # WEIGHTED LEAST SQUARES (INVERSE-VARIANCE WEIGHTING)
            # -----------------------------------------------------------------
            try:
                # 1. Isolate the strictly valid subspace (Filters zero-probability states)
                valid_mask = p_meas > 1e-12
                p_valid = p_meas[valid_mask]

                # Variance of ln(p_k) is ~1/p_k. Optimal weight is proportional to sqrt(p_k).
                weight = np.sqrt(p_valid)

                # [REGEN-PATCH P3] Diagonal-row system: (<=252) x 56, one
                # eigenstate-projected equation per retained microstate.
                W_thermal = sys_data['W_rows'][valid_mask] * weight[:, None]
                T_thermal = (sys_data['h0_diag'][valid_mask]
                             + (1.0 / beta) * np.log(p_valid)) * weight

                w_log_wls, _, _, _ = scipy.linalg.lstsq(W_thermal, T_thermal, cond=1e-9)
                G_log_wls = 0.5 * (reconstruct_G_from_symm(w_log_wls[:-1], M_pairs).real + reconstruct_G_from_symm(w_log_wls[:-1], M_pairs).real.T)
            except Exception:
                G_log_wls = np.zeros((M_pairs, M_pairs))

            # -----------------------------------------------------------------
            # Fair Gauge Shift & Error Evaluation
            # -----------------------------------------------------------------
            def compute_shifted_error(G_pred):
                shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
                G_shifted = G_pred.copy()
                np.fill_diagonal(G_shifted, np.diag(G_shifted) + shift)
                return float(np.linalg.norm(G_shifted - G_true) / norm_g)
            
            try: err_ml = compute_shifted_error(G_ml_batch[i])
            except: err_ml = np.nan

            try: err_log_wls = compute_shifted_error(G_log_wls)
            except: err_log_wls = np.nan

            # [REGEN-PATCH SEED] display jitter removed (no scatter plotted
            # here; the unseeded global draw broke determinism)
            plot_eps.append(eps)
            plot_ml.append(err_ml)
            plot_log_wls.append(err_log_wls)

    print(f"[3/3] Processing Error Distributions and Rendering Plot...")
    plot_eps = np.array(plot_eps)
    # [REGEN-PATCH STATS] All statistics are computed on UNCLIPPED errors;
    # the display clip (floor raised 1e-6 -> 1e-8 so the high-shot WLS median
    # is not floor-limited) is applied at plot time only.
    plot_ml = np.asarray(plot_ml, dtype=float)
    plot_log_wls = np.asarray(plot_log_wls, dtype=float)

    df = pd.DataFrame({
        'eps_orig': np.repeat(EPSILONS, N_STATES * N_NOISE),
        'ml': plot_ml,
        'log_wls': plot_log_wls
    })
    grouped = df.groupby('eps_orig')
    medians = grouped.median(numeric_only=True).reset_index()
    q25 = grouped.quantile(0.25, numeric_only=True).reset_index()
    q75 = grouped.quantile(0.75, numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(11, 7))

    eps_vals = medians['eps_orig'].values
    _clip = lambda a: np.clip(a, 1e-8, 10.0)   # [REGEN-PATCH STATS] display only
    ml_vals = _clip(medians['ml'].values)
    log_wls_vals = _clip(medians['log_wls'].values)

    # [REGEN-PATCH STATS] 25-75% IQR bands around both medians
    ax.fill_between(eps_vals, _clip(q25['log_wls'].values), _clip(q75['log_wls'].values),
                    color='forestgreen', alpha=0.18, lw=0, zorder=2)
    ax.fill_between(eps_vals, _clip(q25['ml'].values), _clip(q75['ml'].values),
                    color='navy', alpha=0.18, lw=0, zorder=3)

    # Plot pure, thick data lines with explicit markers
    ax.plot(eps_vals, log_wls_vals, color='forestgreen', lw=4, label='Log Inversion', zorder=4)
    ax.plot(eps_vals, ml_vals, color='navy', lw=4, label='OGN Prediction', zorder=5)
    
    # Top Axis for Number of Shots
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1/(x**2), lambda x: 1/np.sqrt(x)))
    ax_top.set_xlabel('Number of Experimental Measurement Shots ($N_{shots}$)', fontweight='bold', fontsize=15, labelpad=12)
    
    # Strictly Bound the Plot to the Experimental Domain
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # The X-limits now tightly frame the relevant experimental sparsity regime
    ax.set_xlim(EPSILONS.min() * 0.9, EPSILONS.max() * 1.1) 
    ax.set_ylim(1e-4, 0.6)  
    
    ax.set_xlabel('Statistical Measurement Noise ($\epsilon_{shot} = 1/\sqrt{N_{shots}}$)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Parameter Relative Error\n($||\\Delta G||_F / ||G_{true}||_F$)', fontweight='bold', fontsize=16)
        
    # Clean grid styling
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle='--', alpha=0.15)
    
    # Minimalist Legend
    ax.legend(loc='lower right', framealpha=0.95, fontsize=14)

    plt.tight_layout()
    plt.savefig("wls_vs_ogn_experimental_domain.png", dpi=300, bbox_inches='tight')
    plt.show()

# =========================================================================
# EXECUTE
# =========================================================================
target_rho2 = rho_2_kkbar_arrays if 'rho_2_kkbar_arrays' in globals() else rho_2_arrays
plot_asymptotic_shot_noise_resilience_wls(
    val_dataset, basis, U_ENERGY_SEED, rho_1_arrays, target_rho2, 
    g_gen, final_state, beta=BETA, include_energy=include_energy
)


# 
# # Test cases

# In[ ]:


jax.config.update("jax_enable_x64", True)

def verify_const_pairing_integrity(batch_size=32):
    """
    Generates a small batch for h_type='const' and verifies:
    1. Homogeneity of the Interaction Matrix (G_ij = const)
    2. Macroscopic Condensate (Max Eigenvalue of Pair Matrix > 1)
    3. Phase Coherence (Positivity of Off-Diagonal correlations)
    4. Energy Consistency (Label 'g' vs Feature 'P_sum')
    """
    print(f"\n{'='*80}")
    print(f"RUNNING CONST PAIRING INTEGRITY CHECK (Batch: {batch_size})")
    print(f"{'='*80}")
    
    # 1. Configuration
    n_pairs = N_ELEC // 2
    m_pairs = D_SP // 2
    print(f"Config: D_SP={D_SP}, Pairs={n_pairs}, Type='const'")
    
    # 2. Generate Data
    print("\n[1/3] Generating Constant Pairing parameters...")
    # Initialize generator for 'const'. 
    # g_init must be > 0 to see pairing effects clearly.
    g_gen_test = GGenerator(basis, h_type='const', batch_size=batch_size, g_init=0.1, g_stop=2.5)
    
    key = jax.random.PRNGKey(101)
    _, h_labels = g_gen_test.generate(key)
    g_arr = g_gen_test.reconstruct(h_labels) 
    
    # Energy Seed
    e_arr = U_ENERGY_SEED[:batch_size] 
    
    # 3. Construct Operators & Solve
    print("[2/3] Constructing Hamiltonians and solving for Ground States...")
    t_rho1 = rho_1_arrays #fmb.rho_m_gen(basis, 1)
    t_rho2_kk = rho_2_kkbar_arrays #fmb.rho_2_kkbar_gen(basis)
    
    # Build H (ensure we pass 'const' if your builder optimizes for it)
    h_arr = two_body_hamiltonian_sp(basis, e_arr, g_arr, t_rho1, t_rho2_kk, h_type='const')
    E, states = pure_state(h_arr)
    rho2_kk = compute_rho_m(states, t_rho2_kk, batch_size)
    energies_solver = state_energy(states, h_arr)


    # GPU based method
    levels = np.arange(0, N_ELEC) - N_ELEC // 2 + 0.5
    base_energies = U_ENERGY_SEED[0][::2]
    e_vals = jnp.tile(base_energies, (batch_size, 1))

    d_rho_1_diag = jnp.einsum('kknn->kn', t_rho1.todense())

    r1, r2, en = solve_batch_kernel(e_vals, g_arr, 1, False, d_rho_1_diag, t_rho2_kk.todense(), t_rho2_kk.todense(), t_rho1.todense())
    #r1, r2, en = solve_batch_kernel_sp(basis, e_vals, g_arr, 1, False, rho_1_arrays, t_rho2_kk, t_rho2_kk)
    
    rho2_kk = r2
    energies_solver = en

    # 4. Compute Features & Validate
    print("[3/3] Computing RDMs and validating Physics...")
    
    
    
    valid_count = 0
    
    # Table Header
    print(f"\n{'Idx':<4} | {'Status':<6} | {'g_val':<6} | {'MaxEig':<8} | {'OffDiag>0':<9} | {'E_Err':<8}")
    print("-" * 75)
    
    # Store data for visualization
    history_g = []
    history_max_eig = []
    
    for i in range(batch_size):
        P_mat = rho2_kk[i].squeeze()
        G_mat = g_arr[i]
        
        # --- Check 1: Interaction Homogeneity (Specific to 'const') ---
        # The G matrix should be uniform. Check standard deviation.
        g_std = np.std(G_mat)
        is_uniform = g_std < 1e-6
        
        # Extract the scalar g (mean of the matrix)
        scalar_g = np.mean(G_mat)
        
        # --- Check 2: The "Condensate" Check (Eigenvalue > 1) ---
        # In a pairing Hamiltonian, the Pair Matrix has a dominant eigenvalue > 1 (ODLRO).
        # Free fermions have max_eig = 1.
        evals = np.linalg.eigvalsh(P_mat)
        max_eig = np.max(evals)
        
        # Store for plotting
        history_g.append(scalar_g)
        history_max_eig.append(max_eig)
        
        # We strictly expect pairing if g is non-zero.
        # If g is very small, max_eig is close to 1. If g > 0.1, it should be > 1.
        condensate_check = max_eig > (1.0 + 1e-4)
        
        # --- Check 3: Phase Consistency (Positivity) ---
        # For attractive pairing, off-diagonal correlations should be positive
        mask = ~np.eye(P_mat.shape[0], dtype=bool)
        min_off_diag = np.min(P_mat[mask])
        # Allow tiny numerical noise, but generally >= 0
        phase_valid = min_off_diag > -1e-5
        
        # --- Check 4: Energy Reconstruction ---
        # E_int = g * Sum(P_mat) 
        # (Since G is constant, we sum the entire pair matrix)
        sum_P = np.sum(P_mat).real
        e_int_recon = scalar_g * sum_P
        
        # 1-Body Energy
        n_sp_occ = np.diag(r1[i].squeeze()).real
        e_1body = np.sum(e_vals[i] * n_sp_occ)
        
        e_recon = e_1body - e_int_recon
        e_error = abs(energies_solver[i] - e_recon)
        
        # Validates flawlessly to < 1e-12 with float64
        energy_valid = e_error < 1e-3

        
        # --- Overall Status ---
        # We relax the condensate check if g is very weak (< 0.05)
        phy_check = condensate_check if scalar_g > 0.05 else True
        
        is_valid = is_uniform and energy_valid and phy_check and phase_valid
        status = "PASS" if is_valid else "FAIL"
        
        phase_str = "YES" if phase_valid else "NO"
        
        print(f"{i:<4} | {status:<6} | {scalar_g:<6.3f} | {max_eig:<8.4f} | {phase_str:<9} | {e_error:<8.2e}")
        
        if is_valid:
            valid_count += 1

    print("-" * 75)
    print(f"Result: {valid_count}/{batch_size} samples verified.")
    
    if valid_count < batch_size:
        print("WARNING: Some samples failed. Check if 'g' is too small or solver failed.")
    else:
        print("SUCCESS: Constant Pairing physics verified.")

    # --- Visualization: The Phase Transition ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        
        # Plot 1: Condensate Formation
        # Shows how the "Cooper Pair" eigenvalue grows with interaction strength
        plt.subplot(1, 2, 1)
        plt.scatter(history_g, history_max_eig, c='purple', alpha=0.7, edgecolors='k')
        plt.xlabel("Interaction Strength (g)")
        plt.ylabel("Largest Eigenvalue of Pair Matrix")
        plt.title("Condensate Formation Check")
        plt.axhline(1.0, color='r', linestyle='--', label="Free Particle Limit")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Matrix Structure (should look uniform-ish with positive off-diagonals)
        plt.subplot(1, 2, 2)
        # Take the sample with highest g
        idx_max = np.argmax(history_g)
        plt.imshow(rho2_kk[idx_max].squeeze().real, cmap='viridis')
        plt.colorbar(label=r'$\langle P^\dagger_i P_j \rangle$')
        plt.title(f"Pair Matrix Structure (g={history_g[idx_max]:.2f})")
        plt.xlabel("Pair Index")
        plt.ylabel("Pair Index")
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

# Run the routine
verify_const_pairing_integrity(batch_size=32)


# In[ ]:


import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

def verify_random_pairing_integrity(batch_size=32):
    """
    Generates a small batch for h_type='random' and verifies:
    1. Randomness & Symmetry (G_ij = G_ji, std > 0)
    2. Particle Number Conservation (Tr(P) = n_pairs)
    3. Pauli Exclusion Principle (0 <= P_ii <= 1)
    4. N-Representability / Positivity (min eigenvalue >= 0)
    5. Phase Coherence (Off-diagonals >= 0 since G_ij > 0)
    6. Exact Energy Reconstruction (E_solver = E_1B - E_int)
    """
    print(f"\n{'='*90}")
    print(f"RUNNING RANDOM PAIRING INTEGRITY CHECK (Batch: {batch_size})")
    print(f"{'='*90}")
    
    # 1. Configuration
    n_pairs = N_ELEC // 2
    m_pairs = D_SP // 2
    g_init = 0.1
    g_stop = 2.5
    print(f"Config: D_SP={D_SP}, Pairs={n_pairs}, Type='random', G_Range=[{g_init}, {g_stop}]")
    
    # 2. Generate Data
    print("\n[1/3] Generating Random Pairing parameters...")
    g_gen_test = GGenerator(basis, h_type='random', batch_size=batch_size, g_init=g_init, g_stop=g_stop)
    
    key = jax.random.PRNGKey(42)
    _, h_labels = g_gen_test.generate(key)
    g_arr = np.array(g_gen_test.reconstruct(h_labels))
    
    # Energy Seed (Shape: B, 20)
    e_vals = U_ENERGY_SEED[:batch_size]
    
    # 3. Construct Operators & Solve
    print("[2/3] Constructing Hamiltonians and solving for Ground States...")
    t_rho1 = rho_1_arrays
    t_rho2_kk = rho_2_kkbar_arrays
    
    # GPU based method inputs
    d_rho_1_diag = jnp.array(np.einsum('kknn->kn', t_rho1.todense()))
    d_rho_2_kk = jnp.array(t_rho2_kk.todense())
    d_rho_1_full = jnp.array(t_rho1.todense())
    
    r1, r2, en = solve_batch_kernel(
        jnp.array(e_vals), 
        jnp.array(g_arr), 
        1.0, 
        False, 
        d_rho_1_diag, 
        d_rho_2_kk, 
        d_rho_2_kk, 
        d_rho_1_full
    )
    
    rho1_batch = np.array(r1)
    rho2_kk = np.array(r2)
    energies_solver = np.array(en)

    # 4. Compute Features & Validate
    print("[3/3] Computing RDMs and validating Physics...\n")
    
    valid_count = 0
    
    # Table Header
    print(f"{'Idx':<4} | {'Status':<6} | {'G_Symm':<7} | {'Tr(P)':<6} | {'P>=0':<5} | {'OffDiag>0':<9} | {'E_Err':<8}")
    print("-" * 90)
    
    history_g_std = []
    history_max_eig = []
    
    for i in range(batch_size):
        P_mat = rho2_kk[i].squeeze().real
        G_mat = g_arr[i]
        
        # --- Check 1: Randomness & Exact Symmetry ---
        sym_err = np.max(np.abs(G_mat - G_mat.T))
        is_symmetric = sym_err < 1e-6
        g_std = np.std(G_mat)
        is_random = g_std > 1e-2
        history_g_std.append(g_std)
        
        # --- Check 2: Particle Number & Pauli Exclusion ---
        n_sp_occ = np.diag(rho1_batch[i].squeeze()).real
        trace_P = np.sum(n_sp_occ)
        is_trace_correct = abs(trace_P - n_pairs) < 1e-4
        
        # Diagonal elements (occupancies) must be bounded by Exclusion Principle
        is_pauli = np.all((n_sp_occ >= -1e-5) & (n_sp_occ <= 1.0 + 1e-5))
        
        # --- Check 3: N-Representability (Positivity) ---
        evals = np.linalg.eigvalsh(P_mat)
        min_eig = np.min(evals)
        max_eig = np.max(evals)
        is_psd = min_eig > -1e-5
        history_max_eig.append(max_eig)
        
        # --- Check 4: Phase Coherence (Positivity of Off-Diagonals) ---
        # With purely attractive Random interactions, phase amplitudes are aligned.
        mask = ~np.eye(P_mat.shape[0], dtype=bool)
        min_off_diag = np.min(P_mat[mask])
        is_phase_coherent = min_off_diag > -1e-5
        
        # --- Check 5: Exact Energy Reconstruction ---
        # E_int = Sum_{ij} (G_ij P_ji)
        e_int_recon = np.sum(G_mat * P_mat.T)
        
        # Calculate pair energies properly to contract with n_k
        pair_energies = e_vals[i].reshape(-1, 2).sum(axis=-1)
        e_1body = np.sum(pair_energies * n_sp_occ)
        
        e_recon = e_1body - e_int_recon
        e_error = abs(energies_solver[i] - e_recon)
        is_energy_exact = e_error < 1e-3
        
        # --- Overall Status ---
        is_valid = (is_symmetric and is_random and is_trace_correct and 
                    is_pauli and is_psd and is_phase_coherent and is_energy_exact)
        
        status = "PASS" if is_valid else "FAIL"
        
        sym_str = "YES" if is_symmetric else "NO"
        psd_str = "YES" if is_psd else "NO"
        phase_str = "YES" if is_phase_coherent else "NO"
        
        print(f"{i:<4} | {status:<6} | {sym_err:<7.1e} | {trace_P:<6.3f} | {psd_str:<5} | {phase_str:<9} | {e_error:<8.2e}")
        
        if is_valid:
            valid_count += 1

    print("-" * 90)
    print(f"Result: {valid_count}/{batch_size} samples physically verified.")
    
    if valid_count < batch_size:
        print("WARNING: Some samples failed to reconstruct correctly. Check logic.")
    else:
        print("SUCCESS: Random Pairing exact physics structurally verified.")

    # --- Visualization ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 4))
        
        # Plot 1: Interaction Matrix G
        plt.subplot(1, 3, 1)
        idx_max = np.argmax(history_g_std)
        im = plt.imshow(g_arr[idx_max], cmap='coolwarm')
        plt.colorbar(im, label=r'$G_{ij}$')
        plt.title(f"Random Interaction G (Sample {idx_max})")
        plt.xlabel("Pair Index")
        plt.ylabel("Pair Index")
        
        # Plot 2: Pair Matrix Structure
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(rho2_kk[idx_max].squeeze().real, cmap='viridis')
        plt.colorbar(im2, label=r'$\langle P^\dagger_i P_j \rangle$')
        plt.title("Pair Matrix Structure")
        plt.xlabel("Pair Index")
        plt.ylabel("Pair Index")
        
        # Plot 3: Spectrum of P
        plt.subplot(1, 3, 3)
        plt.scatter(history_g_std, history_max_eig, c='purple', alpha=0.7, edgecolors='k')
        plt.axhline(1.0, color='r', linestyle='--', label="Free Particle Limit")
        plt.xlabel(r"Interaction Variance ($\sigma_G$)")
        plt.ylabel("Largest Eigenvalue of P")
        plt.title("Condensate vs Interaction Disorder")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

# Run the routine
verify_random_pairing_integrity(batch_size=32)

