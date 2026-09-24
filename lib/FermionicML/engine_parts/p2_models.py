# ============================================================================
# p2_models.py -- model zoo: DeepResMLP (mlp), CoordResMLP (cnn),
# PhysicsOrbitalGraphNet (ogn, production model) + referee ablation toggles.
#
# Exec'd into the shared engine namespace AFTER p1_core.py. Cross-part globals
# (e.g. N_ELEC from p1_core) resolve at CALL time via late binding -- exactly
# like the original notebook. Do not import other parts.
#
# Sources (verbatim transplants, edits marked):
#   .claude/cells/cell_16.py  -> ResMLPBlock, DeepResMLP
#   .claude/cells/cell_17.py  -> ResBlock, CoordResMLP
#   .claude/cells/cell_18.py  -> MatrixInteractionBlock, PhysicsOrbitalGraphNet
#                                (production OGN, res=3 => 17,756,929 params)
#
# Ablation toggles (defaults reproduce production exactly):
#   use_scatter  : False => skip einsum('bikf,bkjf->bijf') left/right scattering
#                  contraction; 'contracted' dropped from edge-update concat.
#   use_reinject : False => drop the bare_rdm channel from edge-update concat.
#   use_orb_emb  : False => drop the learned orbital embedding from node init.
#   readout_bias : value used wherever the readout bias_init constant 0.55
#                  appears (all readout heads; also DeepResMLP's head).
# ============================================================================

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


# ---------------------------------------------------------------------------
# DeepResMLP  [from cell_16]
# ---------------------------------------------------------------------------

class ResMLPBlock(nn.Module):  # [from cell_16]
    hidden_dim: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        shortcut = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros_init())(x)
        return shortcut + x


class DeepResMLP(nn.Module):  # [from cell_16]
    """
    Flattened Deep Residual MLP.
    Provides a unique parameter pathway for every orbital coordinate, avoiding
    the permutation bias of Transformers and the translation bias of CNNs.
    """
    label_size: int
    res: int = 2
    include_energy: bool = True
    readout_bias: float = 0.55  # [ablation toggle] readout head bias_init

    @nn.compact
    def __call__(self, x, energy=None, training: bool = True):
        b, m, _, _ = x.shape
        x_mat = x.squeeze(-1)

        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

        r, c = jnp.triu_indices(m)
        x_flat = x_sym[:, r, c]

        mask = jnp.where(r == c, 1.0, jnp.sqrt(2.0))
        x_flat = x_flat * mask

        if self.include_energy and energy is not None:
            if energy.ndim == 1: energy = energy[:, None]
            energy_scaled = energy / N_ELEC  # noqa: F821  (late-bound, p1_core)

            e_proj = nn.Dense(8)(energy_scaled)
            e_proj = nn.gelu(e_proj)
            x_flat = jnp.concatenate([x_flat, e_proj], axis=-1)

        hidden_dim = 256 * self.res

        x = nn.Dense(hidden_dim)(x_flat)
        x = nn.gelu(x)

        for _ in range(8):
            x = ResMLPBlock(hidden_dim)(x, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(hidden_dim // 2)(x)
        x = nn.gelu(x)

        out = nn.Dense(
            self.label_size,
            kernel_init=nn.initializers.normal(stddev=1e-4),
            bias_init=nn.initializers.constant(self.readout_bias)  # [edit: was 0.55]
        )(x)
        return out


# ---------------------------------------------------------------------------
# CoordResMLP  [from cell_17]
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):  # [from cell_17]
    """Residual MLP Block: x + Dense(GELU(Dense(LayerNorm(x))))"""
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros_init())(x)
        return shortcut + x


class CoordResMLP(nn.Module):  # [from cell_17]
    """
    Flax implementation of the Coordinate-Injected ResMLP.

    Params:
        label_size: Output dimension (number of Hamiltonian parameters).
        res: Resolution multiplier for CNN width (default=1).
        include_energy: Whether to inject scalar energy input.
    """
    label_size: int
    res: int = 1
    include_energy: bool = True

    @nn.compact
    def __call__(self, x, energy=None, training: bool = True):
        # x shape: (Batch, H, W, 1)
        b, h, w, c = x.shape

        # 1. Coordinate Injection
        # Normalized grids [-1, 1]
        i_c = jnp.linspace(-1.0, 1.0, h)
        j_c = jnp.linspace(-1.0, 1.0, w)
        ii, jj = jnp.meshgrid(i_c, j_c, indexing='ij')

        # Expand and concat: (B, H, W, 1) + (B, H, W, 2) -> (B, H, W, 3)
        coords = jnp.stack([ii, jj], axis=-1)
        coords = jnp.tile(coords[None, ...], (b, 1, 1, 1))
        x = jnp.concatenate([x, coords], axis=-1)

        # 2. CNN Feature Extraction
        # Width scaled by self.res
        x = nn.Conv(features=64 * self.res, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128 * self.res, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.gelu(x)

        x = x.reshape((b, -1)) # Flatten

        # 3. Energy Injection
        if self.include_energy:
            if energy is None: raise ValueError("Model expects energy input")
            # Ensure energy is (Batch, 1)
            if energy.ndim == 1: energy = energy[:, None]
            energy = energy / N_ELEC  # noqa: F821  (late-bound, p1_core)

            e_vec = nn.Dense(64 * self.res)(energy)
            e_vec = nn.gelu(e_vec)
            x = jnp.concatenate([x, e_vec], axis=-1)

        # 4. ResMLP Head
        hidden_dim = 1024 * self.res
        x = nn.Dense(hidden_dim)(x)
        x = nn.gelu(x)

        for _ in range(3):
            x = ResBlock(hidden_dim)(x)

        x = nn.LayerNorm()(x)
        return nn.Dense(self.label_size)(x)


# ---------------------------------------------------------------------------
# PhysicsOrbitalGraphNet (production OGN)  [from cell_18]
# ---------------------------------------------------------------------------

class MatrixInteractionBlock(nn.Module):  # [from cell_18]
    """
    Pre-Norm Message Passing Block with Bare-RDM Injection and Left/Right Scattering.
    """
    hidden_dim: int
    use_scatter: bool = True   # [ablation toggle] left/right einsum contraction
    use_reinject: bool = True  # [ablation toggle] bare_rdm channel in edge update

    @nn.compact
    def __call__(self, h, e, bare_rdm):
        B, M, _ = h.shape

        # =================================================================
        # 1. PRE-LAYERNORM (Crucial for unblocking deep gradient flow)
        # =================================================================
        h_norm = nn.LayerNorm()(h)
        e_norm = nn.LayerNorm()(e)

        # =================================================================
        # 2. NODE UPDATE (Mean + Max Aggregation)
        # =================================================================
        e_mean = jnp.mean(e_norm, axis=2)
        e_max = jnp.max(e_norm, axis=2) # Max-pooling catches dominant random interactions
        h_in = jnp.concatenate([h_norm, e_mean, e_max], axis=-1)

        h_update = nn.Dense(self.hidden_dim)(h_in)
        h_update = nn.gelu(h_update)
        h_update = nn.Dense(h.shape[-1], kernel_init=nn.initializers.zeros_init())(h_update)

        h = h + h_update # Clean UNNORMALIZED residual stream

        # =================================================================
        # 3. EDGE UPDATE & SCATTERING LOOP
        # =================================================================
        h_norm2 = nn.LayerNorm()(h)
        h_i = jnp.broadcast_to(h_norm2[:, :, None, :], (B, M, M, h.shape[-1]))
        h_j = jnp.broadcast_to(h_norm2[:, None, :, :], (B, M, M, h.shape[-1]))

        h_sum = h_i + h_j
        h_prod = h_i * h_j

        # =================================================================
        # 4. BARE RDM INJECTION (The Anchor)
        # =================================================================
        # Injecting `bare_rdm` completely eliminates graph over-smoothing.
        # [edit: concat assembled conditionally for ablation toggles;
        #  channel ORDER identical to original: e_norm, h_sum, h_prod,
        #  contracted, bare_rdm]
        e_in_parts = [e_norm, h_sum, h_prod]

        if self.use_scatter:
            # Project into Left/Right bases to prevent feature rank collapse
            e_left = nn.Dense(self.hidden_dim // 2)(e_norm)
            e_right = nn.Dense(self.hidden_dim // 2)(e_norm)

            # Native matrix multiplication over the hidden interaction channels
            contracted = jnp.einsum('bikf,bkjf->bijf', e_left, e_right) / jnp.sqrt(M)
            e_in_parts.append(contracted)

        if self.use_reinject:
            e_in_parts.append(bare_rdm[..., None])

        e_in = jnp.concatenate(e_in_parts, axis=-1)

        # Deeper 2-Layer MLP for highly non-linear matrix inversions
        e_update = nn.Dense(self.hidden_dim)(e_in)
        e_update = nn.gelu(e_update)
        e_update = nn.Dense(self.hidden_dim)(e_update)
        e_update = nn.gelu(e_update)
        e_update = nn.Dense(e.shape[-1], kernel_init=nn.initializers.zeros_init())(e_update)

        # Force strict Bosonic/Hermitian interaction symmetry
        e_update = 0.5 * (e_update + jnp.swapaxes(e_update, 1, 2))

        e = e + e_update

        return h, e


class PhysicsOrbitalGraphNet(nn.Module):  # [from cell_18]
    """
    V2 Topology: Pre-Norm Relational Matrix Graph
    """
    label_size: int
    res: int = 3
    include_energy: bool = True
    use_scatter: bool = True    # [ablation toggle] -> MatrixInteractionBlock
    use_reinject: bool = True   # [ablation toggle] -> MatrixInteractionBlock
    use_orb_emb: bool = True    # [ablation toggle] learned orbital embedding
    readout_bias: float = 0.55  # [ablation toggle] readout head bias_init

    @nn.compact
    def __call__(self, x, energy=None, training: bool = True):
        b, m, _, _ = x.shape
        x_mat = x.squeeze(-1)

        num_triu = (m * (m + 1)) // 2
        htype_random = (self.label_size == num_triu)

        # Strict Symmetrization of the input
        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

        # A. NODE INITIALIZATION
        diag_idx = jnp.arange(m)
        n_k = x_sym[:, diag_idx, diag_idx][..., None]

        node_features = [n_k]

        if self.use_orb_emb:  # [edit: ablation gate; default path identical]
            # Widen embedding slightly for richer single-particle identity
            orb_emb = self.param('orb_emb', nn.initializers.normal(stddev=0.1), (m, 32))
            orb_emb_batch = jnp.broadcast_to(orb_emb[None, :, :], (b, m, 32))
            node_features.append(orb_emb_batch)

        if self.include_energy and energy is not None:
            if energy.ndim == 1: energy = energy[:, None]
            e_ctx = nn.Dense(32)(energy / m)
            e_ctx = nn.gelu(e_ctx)
            e_ctx_batch = jnp.broadcast_to(e_ctx[:, None, :], (b, m, 32))
            node_features.append(e_ctx_batch)

        h = jnp.concatenate(node_features, axis=-1)
        h = nn.Dense(128 * self.res)(h)

        # B. EDGE INITIALIZATION
        e = jnp.expand_dims(x_sym, -1)
        e = nn.Dense(128 * self.res)(e)

        # C. MESSAGE PASSING (Deepened safely due to Pre-Norm)
        hidden_dim = 256 * self.res
        for _ in range(5):
            h, e = MatrixInteractionBlock(
                hidden_dim,
                use_scatter=self.use_scatter,    # [edit: toggles passed down]
                use_reinject=self.use_reinject,
            )(h, e, bare_rdm=x_sym)

        # D. PHYSICAL READOUT
        # Final LayerNorm before readout (Standard requirement for Pre-LN networks)
        e_final = nn.LayerNorm()(e)

        out_mat = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1e-3), # Safer, smaller init
            bias_init=nn.initializers.constant(self.readout_bias) # Match target mean [edit: was 0.55]
        )(e_final).squeeze(-1)

        out_mat = 0.5 * (out_mat + jnp.swapaxes(out_mat, 1, 2))

        r, c = jnp.triu_indices(m)
        out_features = out_mat[:, r, c]

        if htype_random:
            return out_features
        else:
            x_proj = nn.Dense(128 * self.res)(out_features)
            x_proj = nn.gelu(x_proj)
            return nn.Dense(
                self.label_size,
                kernel_init=nn.initializers.normal(stddev=1e-3),
                bias_init=nn.initializers.constant(self.readout_bias)  # [edit: was 0.55]
            )(x_proj)


# ---------------------------------------------------------------------------
# Factory + parameter counting (campaign API)
# ---------------------------------------------------------------------------

def build_model(arch, label_size, res, include_energy,
                use_scatter=True, use_reinject=True, use_orb_emb=True,
                readout_bias=0.55):
    """Return the flax module for arch in {'ogn','mlp','cnn'}.

    Defaults reproduce the production models exactly. The ablation toggles
    (use_scatter/use_reinject/use_orb_emb) only affect 'ogn'; readout_bias
    affects 'ogn' and 'mlp' (the 'cnn' head has no biased readout init).
    """
    if arch == 'ogn':
        return PhysicsOrbitalGraphNet(
            label_size=label_size, res=res, include_energy=include_energy,
            use_scatter=use_scatter, use_reinject=use_reinject,
            use_orb_emb=use_orb_emb, readout_bias=readout_bias,
        )
    if arch == 'mlp':
        return DeepResMLP(
            label_size=label_size, res=res, include_energy=include_energy,
            readout_bias=readout_bias,
        )
    if arch == 'cnn':
        return CoordResMLP(
            label_size=label_size, res=res, include_energy=include_energy,
        )
    raise ValueError(f"Unknown arch {arch!r}; expected 'ogn', 'mlp' or 'cnn'")


def count_params(model, input_shape, include_energy, seed=0):
    """Init the model with dummy inputs and return total parameter count (int).

    Dummy input construction mirrors print_model_summary [cell_21]:
    x = zeros(input_shape), energy = zeros((input_shape[0], 1)) when
    include_energy (batch dims must match; with batch 1 this is zeros((1, 1))).
    Counts only the 'params' collection (batch_stats excluded).
    """
    rng = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros(input_shape)
    dummy_e = jnp.zeros((input_shape[0], 1)) if include_energy else None
    variables = model.init(rng, dummy_x, dummy_e, training=False)
    flat = flax.traverse_util.flatten_dict(variables['params'], sep='/')
    return int(sum(x.size for x in flat.values()))
