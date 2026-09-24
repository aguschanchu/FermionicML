"""ognrepro.v2.losses -- the v2 loss family (float32 on device).

PROVENANCE
    Source: explore_v2f/losses.py @ v2-F 269bee39 (sha256
            ca19f28c12106e6109d169e99e549c69792b9449212c5d6824f6ab73670ec169):
            make_lambda / quad / var_gs / unpack_M / var_thermal VERBATIM
            (:29-71); make_loss ADAPTED (branches gram | metric | mse, driven
            by config.V2Config; the kl and uncertainty-head branches are not
            carried).

dw = flatten(G_pred - G_true) in the engine's A = m*m C-order flattening
(p3_training.py:181-190; every symmetric off-diagonal counted twice, so every
quadratic form is evaluated on the symmetric subspace where the HS second
moment Tr(h_a^dag h_b)/D (engine S) and Tr(h_a h_b)/D coincide).

  gram    L = mean(dw^T S dw) + w_gauge * mean((diagmean G_pred - diagmean G_true)^2)
              + w_ridge * mean(||dw||^2)                  (engine production math)
  metric  L = mean(q_M + lam(step) q_Sinf) + w_gauge * gauge + w_ridge * ridge
          q_Sinf = dw^T (S - u u^T) dw = Var_{beta->0}(dH)
          q_M    = Var_{rho_true}(dH): GS from psi0 (gather tables), thermal
                   from the stored packed covariance
          lam(step): log-linear anneal lam0 -> lam1 over anneal_frac * total_steps
  mse     L = mean(sum (logits - by)^2)                   (engine base_loss)
"""
import numpy as np
import jax
import jax.numpy as jnp

HI = jax.lax.Precision.HIGHEST
DEF = jax.lax.Precision.DEFAULT


def make_lambda(cfg, total_steps):
    """lambda(step) of the metric loss; None when the loss has no anchor."""
    if cfg.loss != "metric":
        return None
    lam0, lam1 = float(cfg.lam0), float(cfg.lam1)
    frac = float(cfg.anneal_frac)
    if frac <= 0.0:
        return lambda step: jnp.asarray(lam1, jnp.float32)
    if lam0 == lam1:
        return lambda step: jnp.asarray(lam0, jnp.float32)
    n = max(1.0, frac * float(total_steps))
    la0, la1 = float(np.log(lam0)), float(np.log(lam1))

    def lam(step):
        f = jnp.clip(jnp.asarray(step).astype(jnp.float32) / n, 0.0, 1.0)
        return jnp.exp(la0 + (la1 - la0) * f)
    return lam


def quad(dw, S, prec):
    return jnp.einsum("bi,ij,bj->b", dw, S, dw, precision=prec)


def var_gs(dw, psi, src_pad, prec):
    """Var_psi(sum_a dw_a h_a): psi (B,D), src_pad (A,D) int32 (D = pad)."""
    B, D = psi.shape
    psi_pad = jnp.concatenate([psi, jnp.zeros((B, 1), psi.dtype)], axis=1)
    Vg = jnp.take(psi_pad, src_pad.reshape(-1), axis=1).reshape(B, src_pad.shape[0], D)
    v = jnp.einsum("ba,bar->br", dw, Vg, precision=prec)
    return jnp.sum(v * v, axis=-1) - jnp.sum(psi * v, axis=-1) ** 2


def unpack_M(Mp, A, iu_r, iu_c):
    """packed upper triangle (B, A(A+1)/2) -> symmetric (B, A, A)."""
    B = Mp.shape[0]
    M = jnp.zeros((B, A, A), Mp.dtype)
    M = M.at[:, iu_r, iu_c].set(Mp)
    M = M.at[:, iu_c, iu_r].set(Mp)
    return M


def var_thermal(dw, Mp, A, iu_r, iu_c, prec):
    M = unpack_M(Mp, A, iu_r, iu_c)
    return jnp.einsum("ba,bac,bc->b", dw, M, dw, precision=prec)


def make_loss(cfg, ops, total_steps, aux_kind, is_gs):
    """Returns loss(logits, by, aux_flat, step, g_gen, S_engine) -> (total, parts)."""
    loss_type = cfg.loss
    prec = HI if cfg.loss_precision == "highest" else DEF
    w_gauge = float(cfg.w_gauge)
    w_ridge = float(cfg.w_ridge)
    A, D = int(ops["A"]), int(ops["D"])
    S_inf = jnp.asarray(ops["S_inf"], jnp.float32)
    src_pad = jnp.asarray(ops["src_pad"], jnp.int32)
    iu = np.triu_indices(A)
    iu_r, iu_c = jnp.asarray(iu[0]), jnp.asarray(iu[1])
    lam = make_lambda(cfg, total_steps)
    if loss_type == "metric" and aux_kind not in ("psi0", "M"):
        raise ValueError("metric loss needs aux 'psi0' or 'M', got %r" % (aux_kind,))
    if loss_type == "metric" and ((aux_kind == "psi0") != bool(is_gs)):
        raise ValueError("aux %r does not match is_gs=%r" % (aux_kind, is_gs))

    def loss_fn(logits, by, aux_flat, step, g_gen, S_engine):
        H_pred = g_gen.reconstruct(logits)
        H_true = g_gen.reconstruct(by)
        B = H_pred.shape[0]
        dw = (H_pred - H_true).reshape(B, -1)
        pred_dm = jnp.mean(jnp.diagonal(H_pred, axis1=1, axis2=2), axis=1)
        true_dm = jnp.mean(jnp.diagonal(H_true, axis1=1, axis2=2), axis=1)
        gauge = jnp.mean(jnp.square(pred_dm - true_dm))
        ridge = jnp.mean(jnp.sum(jnp.square(dw), axis=-1))
        parts = dict(gauge=gauge, ridge=ridge)
        if loss_type == "gram":
            phys = jnp.mean(quad(dw, S_engine, prec))
            parts["phys"] = phys
            total = phys + w_gauge * gauge + w_ridge * ridge
        elif loss_type == "metric":
            qS = quad(dw, S_inf, prec)
            if is_gs:
                qM = var_gs(dw, aux_flat[:, :D], src_pad, prec)
            else:
                qM = var_thermal(dw, aux_flat, A, iu_r, iu_c, prec)
            lam_t = lam(step)
            parts.update(qM=jnp.mean(qM), qS=jnp.mean(qS), lam=lam_t)
            total = jnp.mean(qM + lam_t * qS) + w_gauge * gauge + w_ridge * ridge
        elif loss_type == "mse":
            total = jnp.mean(jnp.sum(jnp.square(logits - by), axis=-1))
            parts["mse"] = total
        else:
            raise ValueError(loss_type)
        return total.astype(jnp.float32), parts

    return loss_fn
