"""ognrepro.readout -- the exact affine readout (gauge + energy-shell projection).

PROVENANCE
    Source: explore_v2f/model.py @ v2-F 269bee39, V2FOGN._project (:198-238)
            and V2FOGN._direction (:155-169), single-channel 'euclid' branch.
    Source file sha256:
            91ec599e7a11d2058252654e32238ec2dc7b4af0f784f9e28155c7b5102549f0
    Tag: ADAPTED -- the closed-form 2x2 solve is verbatim; the multi-channel,
    'S' / 'W' / 'euclid_std' direction variants are not carried (measured
    equivalent or divergent, WAVE2_REPORT.md 1.12); a determinant guard is
    added for the all-zero dummy inputs of flax init.

Two facts the published network had to learn are linear in G: the data
generator's gauge (mean_k G_kk = gauge_value) and the energy-shell identity
    <G, rho>_F = sum_k pair_energies[k] rho_kk - E,
exact for every state of H(G) and for every population-noise draw (the drawn
state shares the eigenbasis of the true Hamiltonian).  Given the network's
symmetric output G and the RAW symmetrized input block rho with the RAW
energy E, the readout returns G' = G + alpha rho + c I with (alpha, c) the
solution of the 2x2 system that makes both constraints exact:

    [ tr(rho)/m   1     ] [alpha]   [ gauge_value - tr(G)/m ]
    [ ||rho||_F^2 tr rho] [  c  ] = [ t - <G, rho>_F        ],   t = sum_k pe_k rho_kk - E.

No parameters: the flax tree of any module that calls this is unchanged.
"""
import jax
import jax.numpy as jnp

HI = jax.lax.Precision.HIGHEST


def project_affine(G, rho, energy, pair_energies, gauge_value=0.55,
                   gauge=True, shell="euclid"):
    """G, rho: (b, m, m) symmetric blocks (network output / RAW input);
    energy: (b, 1) RAW E (may be None when shell == 'off'); pair_energies:
    length-m sequence of 2*eps_k in engine units.  Returns G'."""
    if not gauge and shell == "off":
        return G
    if shell not in ("euclid", "off"):
        raise ValueError("shell %r (expected 'euclid' or 'off')" % (shell,))
    b, m, _ = G.shape
    eye = jnp.eye(m, dtype=G.dtype)
    diag_mean_G = jnp.trace(G, axis1=1, axis2=2) / m
    b1 = gauge_value - diag_mean_G                                 # (b,)
    if shell == "off":
        return G + b1[:, None, None] * eye
    if energy is None:
        raise ValueError("the energy-shell row needs the raw energy input")
    pe = jnp.asarray(pair_energies, G.dtype)                       # (m,)
    if pe.shape[0] != m:
        raise ValueError("pair_energies has %d entries, m = %d" % (pe.shape[0], m))
    e0 = jnp.einsum("k,bkk->b", pe, rho, precision=HI)
    t = e0 - energy[:, 0]
    P = rho                                                        # 'euclid' direction
    GP = jnp.einsum("bij,bij->b", G, rho, precision=HI)
    b2 = t - GP
    a21 = jnp.einsum("bij,bij->b", P, rho, precision=HI)
    if not gauge:
        alpha = b2 / a21
        return G + alpha[:, None, None] * P
    a11 = jnp.trace(P, axis1=1, axis2=2) / m
    a22 = jnp.trace(rho, axis1=1, axis2=2)
    det = a11 * a22 - a21
    # all-zero dummy inputs (flax init templates) give det == 0 exactly; keep
    # the init finite without touching any real input (det != 0 for every
    # physical pair block: rho is not proportional to the identity)
    det = jnp.where(det == 0, jnp.ones_like(det), det)
    alpha = (b1 * a22 - b2) / det
    c = (a11 * b2 - a21 * b1) / det
    return G + alpha[:, None, None] * P + c[:, None, None] * eye


def project_affine_numpy(G, rho, energy, pair_energies, gauge_value=0.55,
                         gauge=True, shell="euclid"):
    """float64 numpy reference of project_affine (tests / analyses)."""
    import numpy as np  # noqa: PLC0415
    G = np.asarray(G, np.float64); rho = np.asarray(rho, np.float64)
    b, m, _ = G.shape
    eye = np.eye(m)
    b1 = gauge_value - np.trace(G, axis1=1, axis2=2) / m
    if shell == "off":
        return G + b1[:, None, None] * eye if gauge else G
    pe = np.asarray(pair_energies, np.float64)
    t = np.einsum("k,bkk->b", pe, rho) - np.asarray(energy, np.float64)[:, 0]
    b2 = t - np.einsum("bij,bij->b", G, rho)
    a21 = np.einsum("bij,bij->b", rho, rho)
    if not gauge:
        return G + (b2 / a21)[:, None, None] * rho
    a11 = np.trace(rho, axis1=1, axis2=2) / m
    a22 = np.trace(rho, axis1=1, axis2=2)
    det = a11 * a22 - a21
    alpha = (b1 * a22 - b2) / det
    c = (a11 * b2 - a21 * b1) / det
    return G + alpha[:, None, None] * rho + c[:, None, None] * eye
