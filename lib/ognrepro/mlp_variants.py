"""ognrepro.mlp_variants -- the DeepResMLP arms of the v2 production run.

PROVENANCE
    ResMLPBlock, DeepResMLP body: campaign/engine_parts/p2_models.py:35-96
    (cell_16), VERBATIM except that the late-bound engine global N_ELEC is an
    explicit field (n_elec) and the trunk width / depth are fields defaulting
    to the published 256*res / 8.
    triu55 standardization: campaign6/train/engine6_parts/p2_models6.py:117-123
    (DeepResMLPCap, V2B-F08): z = (x_sym[triu] * mask - mean) / std on the 55
    sqrt(2)-weighted upper-triangle features, BEFORE the energy channel is
    concatenated; the energy keeps its /N_ELEC scaling.  Empty stats = the
    published DeepResMLP bit-for-bit (same submodule names -> same param tree).

The MLP arms keep the published readout (no affine projection; author
decision 2026-09-09).
"""
import jax.numpy as jnp
import flax.linen as nn


class ResMLPBlock(nn.Module):  # [p2_models.py:35-45 verbatim]
    hidden_dim: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        shortcut = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.zeros_init())(x)
        return shortcut + x


class DeepResMLPV2(nn.Module):
    """Flattened Deep Residual MLP [p2_models.py:47-96] with explicit n_elec,
    optional triu55 standardization and configurable width/depth."""
    label_size: int
    res: int = 2
    include_energy: bool = True
    readout_bias: float = 0.55
    n_elec: int = 10
    std_mean: tuple = ()          # (55,) sqrt(2)-triu feature means, or empty
    std_std: tuple = ()
    width: int = 0                # 0 -> 256 * res (published)
    num_blocks: int = 8

    @nn.compact
    def __call__(self, x, energy=None, training: bool = True):
        b, m, _, _ = x.shape
        x_mat = x.squeeze(-1)

        x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

        r, c = jnp.triu_indices(m)
        x_flat = x_sym[:, r, c]

        mask = jnp.where(r == c, 1.0, jnp.sqrt(2.0))
        x_flat = x_flat * mask

        if len(self.std_mean) == x_flat.shape[-1]:
            if len(self.std_std) != len(self.std_mean):
                raise ValueError("std_std must match std_mean")
            mean = jnp.asarray(self.std_mean, x_flat.dtype)
            std = jnp.asarray(self.std_std, x_flat.dtype)
            x_flat = (x_flat - mean) / std
        elif len(self.std_mean):
            raise ValueError("std_mean has %d entries, features have %d"
                             % (len(self.std_mean), int(x_flat.shape[-1])))

        if self.include_energy and energy is not None:
            if energy.ndim == 1:
                energy = energy[:, None]
            energy_scaled = energy / float(self.n_elec)

            e_proj = nn.Dense(8)(energy_scaled)
            e_proj = nn.gelu(e_proj)
            x_flat = jnp.concatenate([x_flat, e_proj], axis=-1)

        hidden_dim = int(self.width) if self.width else 256 * self.res

        x = nn.Dense(hidden_dim)(x_flat)
        x = nn.gelu(x)

        for _ in range(int(self.num_blocks)):
            x = ResMLPBlock(hidden_dim)(x, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(hidden_dim // 2)(x)
        x = nn.gelu(x)

        out = nn.Dense(
            self.label_size,
            kernel_init=nn.initializers.normal(stddev=1e-4),
            bias_init=nn.initializers.constant(self.readout_bias)
        )(x)
        return out
