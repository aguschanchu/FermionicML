"""ognrepro.std_arch -- the standardized-input OGN (STD-1) + stats loader.

PROVENANCE
    Source: experiments/experiment-campaign2-r2/stages/t_stdd12.py
    Source file sha256:
            8c33cddcfc721f478aff36997642f0ce76d44801adaf28968f31fb3253d04b1f
    Line ranges extracted:
        1656      _MatrixInteractionBlock binding (the source binds
                  engine.ns["MatrixInteractionBlock"]; see diffs)
        1659-1769 class StdPhysicsOrbitalGraphNet (VERBATIM body)
        1782-1803 _set_std_stats f32-tuple conversion (its conversion core is
                  reproduced by load_std_stats below; the ambient
                  one-lane-per-process machinery is deliberately NOT carried)
    Extraction date: 2026-08-23.
    Tag: VERBATIM for the StdPhysicsOrbitalGraphNet class body, with EXACTLY
    these deliberate diffs:
      * _MatrixInteractionBlock is imported from ognrepro.arch_variants
        (itself a VERBATIM extract of p2_models5.py:199-272) instead of
        engine.ns["MatrixInteractionBlock"] -- the classes are functionally
        identical and carry the same __name__, so the flax parameter tree
        auto-naming ("MatrixInteractionBlock_k") is unchanged;
      * TASK_TAG (used in the class's dimensional-tripwire assert message) is
        defined locally with the source's value (t_stdd12.py:180);
      * ADAPTED ADDITION load_std_stats(npz_path): loads a std_stats.npz
        sidecar (keys mu/sigma) and applies the _set_std_stats f32-tuple
        conversion (t_stdd12.py:1799-1802, the exact expression), returning
        (mu_tuple, sigma_tuple) hashable nested tuples.  The ambient
        _STD_ACTIVE re-arm refusal is NOT reproduced: per the sealed
        D16PROG-P4 as-run (expc2r2_d16prog_p4_eval_as_run.py:79-94,
        "BATCH SAFETY"), evaluation constructs the class DIRECTLY with
        per-arm stats and never arms module state.
    NOTE the construction call proven training-identical by the sealed P4
    record (as-run :86-94):
        StdPhysicsOrbitalGraphNet(label_size, res=3, include_energy=True,
            use_scatter=True, use_reinject=True, use_orb_emb=True,
            readout_bias=0.55, use_energy_input=True,
            std_mu=mu_t, std_sigma=sg_t)
"""
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from .arch_variants import MatrixInteractionBlock as _MatrixInteractionBlock
from .readout import project_affine as _project_affine   # [V2] exact affine readout

TASK_TAG = "t_stdd12"                     # [t_stdd12.py:180]


class StdPhysicsOrbitalGraphNet(nn.Module):
    """PhysicsOrbitalGraphNet (p2_models5.py:275-361) + STD-1 standardization.

    __call__ is a verbatim copy of p2_models5.py:290-361 with EXACTLY these
    diffs (each marked in-line):
      + compute x_std = (x_sym - mu) / sigma once, right after the production
        symmetrization (after p2_models5.py:299);
      ~ p2_models5.py:303  n_k reads x_std          [STD-1 node_init]
      ~ p2_models5.py:325  edge init reads x_std    [STD-1 edge_init]
      ~ p2_models5.py:335  bare_rdm=x_std           [STD-1 reinjection]
    Everything else is untouched -> the params pytree is IDENTICAL to the
    published d12 OGN (standardization adds no trainable params).

    [R2-T2b] std_mu / std_sigma are m x m nested tuples -- at era d12 that is
    6x6 (M_PAIRS = 6, p1_core.py:995).  The `len(self.std_mu) == m` assert is
    the dimensional tripwire: a d20 10x10 statistic can never be applied here.
    Every trainable d12 arm has label_size != num_triu (1 or 2 vs 21), so the
    LOW-DIM
    HEAD branch at the tail is always taken."""
    label_size: int
    res: int = 3
    include_energy: bool = True
    use_scatter: bool = True
    use_reinject: bool = True
    use_orb_emb: bool = True
    readout_bias: float = 0.55
    use_energy_input: bool = True
    std_mu: tuple = ()       # m x m nested tuple (per-entry mean, f32 values)
    std_sigma: tuple = ()    # m x m nested tuple (per-entry std, floored 1e-8)
    # [V2] exact affine readout (see arch_variants.PhysicsOrbitalGraphNet):
    # reads the RAW symmetrized block x_sym (never x_std) and the RAW energy.
    proj_gauge: bool = False
    proj_shell: str = "off"          # 'off' | 'euclid'
    gauge_value: float = 0.55
    pair_energies: tuple = ()

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

        # Strict Symmetrization of the input            [= p2_models5.py:299]
        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

        # ---- [STD-1] per-entry affine standardization of the pair block ----
        # (single inserted stanza; consumed at the three marked sites below.
        #  The ENERGY channel is NOT standardized -- out of scope.)
        assert len(self.std_mu) == m and len(self.std_sigma) == m, (
            f"[{TASK_TAG}] std stats are {len(self.std_mu)}x? but m={m}")
        mu = jnp.asarray(self.std_mu, x_sym.dtype)
        sigma = jnp.asarray(self.std_sigma, x_sym.dtype)
        x_std = (x_sym - mu) / sigma
        # -------------------------------------------------------------------

        # A. NODE INITIALIZATION
        diag_idx = jnp.arange(m)
        n_k = x_std[:, diag_idx, diag_idx][..., None]  # [STD-1 node_init] was x_sym (p2_models5.py:302-303)

        node_features = [n_k]

        if self.use_orb_emb:
            orb_emb = self.param('orb_emb', nn.initializers.normal(stddev=0.1), (m, 32))
            orb_emb_batch = jnp.broadcast_to(orb_emb[None, :, :], (b, m, 32))
            node_features.append(orb_emb_batch)

        # energy branch verbatim (p2_models5.py:313-319); NOT standardized
        if self.include_energy and self.use_energy_input and energy is not None:
            if energy.ndim == 1: energy = energy[:, None]
            e_ctx = nn.Dense(32)(energy / m)
            e_ctx = nn.gelu(e_ctx)
            e_ctx_batch = jnp.broadcast_to(e_ctx[:, None, :], (b, m, 32))
            node_features.append(e_ctx_batch)

        h = jnp.concatenate(node_features, axis=-1)
        h = nn.Dense(128 * self.res)(h)

        # B. EDGE INITIALIZATION
        e = jnp.expand_dims(x_std, -1)                 # [STD-1 edge_init] was x_sym (p2_models5.py:325)
        e = nn.Dense(128 * self.res)(e)

        # C. MESSAGE PASSING
        hidden_dim = 256 * self.res
        for _ in range(5):
            h, e = _MatrixInteractionBlock(
                hidden_dim,
                use_scatter=self.use_scatter,
                use_reinject=self.use_reinject,
            )(h, e, bare_rdm=x_std)                    # [STD-1 reinjection] was bare_rdm=x_sym (p2_models5.py:335 -> :255-256)

        # D. PHYSICAL READOUT
        e_final = nn.LayerNorm()(e)

        out_mat = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1e-3),
            bias_init=nn.initializers.constant(self.readout_bias)
        )(e_final).squeeze(-1)

        out_mat = 0.5 * (out_mat + jnp.swapaxes(out_mat, 1, 2))

        # [V2] exact affine readout on the RAW block x_sym + RAW energy (no params)
        if proj_on:
            out_mat = _project_affine(out_mat, x_sym, energy_raw, self.pair_energies,
                                      gauge_value=self.gauge_value,
                                      gauge=bool(self.proj_gauge), shell=self.proj_shell)

        r, c = jnp.triu_indices(m)
        out_features = out_mat[:, r, c]

        if htype_random:
            return out_features
        else:
            # [R2-T2b] every d12 arm lands here (label_size 1 or 2 vs 21 triu)
            x_proj = nn.Dense(128 * self.res)(out_features)
            x_proj = nn.gelu(x_proj)
            return nn.Dense(
                self.label_size,
                kernel_init=nn.initializers.normal(stddev=1e-3),
                bias_init=nn.initializers.constant(self.readout_bias)
            )(x_proj)


def std_stats_tuples(mu, sigma):
    """[ADAPTED] The _set_std_stats f32-tuple conversion (t_stdd12.py:
    1799-1802), verbatim expressions, WITHOUT the ambient-state machinery:
    encode f32 stats as hashable nested tuples for the module attributes."""
    mu_t = tuple(tuple(float(v) for v in row)
                 for row in np.asarray(mu, np.float32))
    sigma_t = tuple(tuple(float(v) for v in row)
                    for row in np.asarray(sigma, np.float32))
    return mu_t, sigma_t


def load_std_stats(npz_path):
    """[ADAPTED] Load a lane's std_stats.npz sidecar (keys mu / sigma, m x m
    f32 arrays; schema t_stdd12.py:364 and the d16prog lane sidecars) and
    return (mu_tuple, sigma_tuple) via the _set_std_stats conversion."""
    with np.load(npz_path) as z:
        mu = np.asarray(z["mu"])
        sigma = np.asarray(z["sigma"])
    return std_stats_tuples(mu, sigma)
