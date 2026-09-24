"""ognrepro.arch_variants -- the campaign5 OGN with the use_energy_input gate.

Exists to restore the `ogn_noE` d20 checkpoint (V2-F01 energy-channel
ablation): the vendored campaign4 engine's PhysicsOrbitalGraphNet has no
use_energy_input attribute, so the ablated checkpoint cannot deserialize
against it.  With use_energy_input=True (the default) this module reproduces
the production campaign4 module bit-for-bit, including the flax auto-naming
of every Dense, so it also restores every published 'ogn' checkpoint.

PROVENANCE
    Source: campaign5/train/engine5_parts/p2_models5.py
    Source file sha256:
            e3b471237fd3024906d1ca9ab960df860a8f73e1dfa0e4fc6d1cca6c0a5ceda2
    Line ranges extracted:
        199-272   class MatrixInteractionBlock  (the block class the OGN
                  references; VERBATIM.  It is functionally identical to
                  campaign4/engine_parts/p2_models.py's copy -- the only
                  diff there is a comment -- so restoring against either is
                  equivalent; extracting it here keeps this module importable
                  without the vendored engine.)
        275-361   class PhysicsOrbitalGraphNet  (VERBATIM, [C5 EDIT 1]
                  use_energy_input toggle included)
        651-660   OGN_USE_ENERGY_INPUT ambient global + set_ogn_energy_input
                  (VERBATIM)
    Extraction date: 2026-08-23.
    Tag: VERBATIM for the bodies above.  Deliberate module-level diffs:
      * import list trimmed to jax.numpy / flax.linen (the source also
        imports jax and flax at module level for its factory/counting
        helpers, which are not extracted);
      * build_model / count_params and the other architecture classes
        (DeepResMLP, CoordResMLP, MLPOrbFeat, DeepSetsNoMP, EdgeMLPNoMP) are
        NOT extracted -- non-'ogn' archs restore through the vendored engine
        (repro_common.get_engine().build_model);
      * ADAPTED ADDITION build_ogn(...): a thin keyword wrapper replicating
        exactly the arch=='ogn' branch of the source's build_model
        (p2_models5.py:663-681), provided so callers do not re-transcribe
        the default toggle set.
"""
import jax.numpy as jnp
import flax.linen as nn

from .readout import project_affine as _project_affine   # [V2] exact affine readout


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
    use_energy_input: bool = True  # [C5 EDIT 1] V2-F01: False => energy branch
    #                                never created (no energy Dense params;
    #                                forward exactly energy-independent)
    # [V2] exact affine readout (ognrepro.readout.project_affine): gauge row
    # and/or energy-shell row applied to the symmetric output block, reading
    # the RAW symmetrized input and the RAW energy.  Defaults OFF => the
    # published forward and parameter tree, bit for bit (no parameters).
    proj_gauge: bool = False
    proj_shell: str = "off"          # 'off' | 'euclid'
    gauge_value: float = 0.55
    pair_energies: tuple = ()        # (m,) 2*eps_k, engine units (shell row only)

    @nn.compact
    def __call__(self, x, energy=None, training: bool = True):
        b, m, _, _ = x.shape
        x_mat = x.squeeze(-1)

        num_triu = (m * (m + 1)) // 2
        htype_random = (self.label_size == num_triu)

        # [V2] raw energy for the readout, captured before the branch scaling
        proj_on = bool(self.proj_gauge) or self.proj_shell != "off"
        if proj_on and not htype_random:
            raise ValueError("the affine readout is defined for the %d-parameter dense family only "
                             "(label_size %d)" % (num_triu, self.label_size))
        energy_raw = None
        if energy is not None:
            energy_raw = energy if energy.ndim == 2 else energy[:, None]

        # Strict Symmetrization of the input
        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

        # A. NODE INITIALIZATION
        diag_idx = jnp.arange(m)
        n_k = x_sym[:, diag_idx, diag_idx][..., None]

        node_features = [n_k]

        if self.use_orb_emb:
            # Widen embedding slightly for richer single-particle identity
            orb_emb = self.param('orb_emb', nn.initializers.normal(stddev=0.1), (m, 32))
            orb_emb_batch = jnp.broadcast_to(orb_emb[None, :, :], (b, m, 32))
            node_features.append(orb_emb_batch)

        # [C5 EDIT 1] energy-channel ablation gate (default path identical)
        if self.include_energy and self.use_energy_input and energy is not None:
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
                use_scatter=self.use_scatter,
                use_reinject=self.use_reinject,
            )(h, e, bare_rdm=x_sym)

        # D. PHYSICAL READOUT
        # Final LayerNorm before readout (Standard requirement for Pre-LN networks)
        e_final = nn.LayerNorm()(e)

        out_mat = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1e-3), # Safer, smaller init
            bias_init=nn.initializers.constant(self.readout_bias) # Match target mean
        )(e_final).squeeze(-1)

        out_mat = 0.5 * (out_mat + jnp.swapaxes(out_mat, 1, 2))

        # [V2] exact affine readout on the RAW block + RAW energy (no params)
        if proj_on:
            out_mat = _project_affine(out_mat, x_sym, energy_raw, self.pair_energies,
                                      gauge_value=self.gauge_value,
                                      gauge=bool(self.proj_gauge), shell=self.proj_shell)

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
                bias_init=nn.initializers.constant(self.readout_bias)
            )(x_proj)


# ---------------------------------------------------------------------------
# [VERBATIM p2_models5.py:651-660] ambient use_energy_input flag
# ---------------------------------------------------------------------------
OGN_USE_ENERGY_INPUT = True  # ambient default; see set_ogn_energy_input()


def set_ogn_energy_input(flag):
    """Set the ambient use_energy_input picked up by build_model when its
    caller does not pass the kwarg explicitly (campaign4 p3_training call
    sites). Drivers MUST call this per model spec immediately before any
    train_model/load_model/count_params for that model."""
    globals()['OGN_USE_ENERGY_INPUT'] = bool(flag)
    return globals()['OGN_USE_ENERGY_INPUT']


def build_ogn(label_size, res=3, include_energy=True,
              use_scatter=True, use_reinject=True, use_orb_emb=True,
              readout_bias=0.55, use_energy_input=None,
              proj_gauge=False, proj_shell="off", gauge_value=0.55,
              pair_energies=()):
    """[ADAPTED] The arch=='ogn' branch of p2_models5.build_model (:663-681),
    including the ambient-global fallback for use_energy_input=None.
    [V2] proj_gauge / proj_shell / gauge_value / pair_energies select the exact
    affine readout (defaults off = published)."""
    if use_energy_input is None:
        use_energy_input = bool(globals().get('OGN_USE_ENERGY_INPUT', True))
    return PhysicsOrbitalGraphNet(
        label_size=label_size, res=res, include_energy=include_energy,
        use_scatter=use_scatter, use_reinject=use_reinject,
        use_orb_emb=use_orb_emb, readout_bias=readout_bias,
        use_energy_input=use_energy_input,
        proj_gauge=bool(proj_gauge), proj_shell=str(proj_shell),
        gauge_value=float(gauge_value),
        pair_energies=tuple(float(v) for v in pair_energies),
    )
