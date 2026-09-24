"""ognrepro.v2.train_step -- the v2 pmap training step (engine signature).

PROVENANCE
    Source: explore_v2f/train_step.py @ v2-F 269bee39 (sha256
            a7d1ad7c10eee38a4e672eb502895f239a0b9543e132acc06aad23e3bb0abd88);
            ADAPTED: single energy channel, no kl extras, V2Config-driven,
            installed under engine.ns['create_train_step_v2'] (the engine's
            create_train_step delegates to it when given v2=...).

    p_train_step(state, bx, be, by, S_tensor, base_e_params, rho_1_diag, inter_tensor)

`be` is the V2Loader packed channel [energy | aux_flat]; ONLY `energy` (b,1)
reaches the network (sealed input access), `aux_flat` feeds the loss.
"""
import contextlib
from functools import partial

import jax

from .losses import make_loss

N_E = 1


def split_be(be):
    """(energy (b,1), aux_flat (b, w)) from the packed channel."""
    return be[:, :N_E], be[:, N_E:]


def create_train_step_v2(cfg, ops, total_steps, aux_kind, is_gs, g_gen):
    loss_core = make_loss(cfg, ops, total_steps, aux_kind, is_gs)
    net_prec = cfg.net_precision

    @partial(jax.pmap, axis_name="batch")
    def p_train_step(state, bx, be, by, S_tensor, base_e_params, rho_1_diag, inter_tensor):
        energy, aux_flat = split_be(be)

        def loss_fn(params):
            ctx = jax.default_matmul_precision(net_prec) if net_prec else contextlib.nullcontext()
            with ctx:
                logits, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    bx, energy, training=True, mutable=["batch_stats"])
            total, _parts = loss_core(logits, by, aux_flat, state.step, g_gen, S_tensor)
            return total, updates

        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)
        new_batch_stats = updates.get("batch_stats", state.batch_stats)
        return new_state.replace(batch_stats=new_batch_stats), loss

    return p_train_step


def install(engine, cfg, ops, total_steps, aux_kind, is_gs):
    """Bind engine.ns['create_train_step_v2']; the engine's create_train_step
    (p3_training.py [V2] hook) forwards (loss_type, is_thermal, beta,
    total_steps=, v2=) here.  total_steps given at install time wins over the
    engine's (they are computed identically); the engine's value is checked."""
    g_gen = engine.g_gen

    def factory(loss_type="gram", is_thermal=False, beta=100.0, total_steps_engine=None,
                v2=None, **_gammas):
        if v2 is not None and v2 != cfg:
            raise RuntimeError("[v2.train_step] installed config differs from the requested one")
        if total_steps_engine is not None and int(total_steps_engine) != int(total_steps):
            raise RuntimeError("[v2.train_step] total_steps mismatch: installed %d, engine %d"
                               % (int(total_steps), int(total_steps_engine)))
        if (loss_type == "metric") != (cfg.loss == "metric"):
            raise RuntimeError("[v2.train_step] loss_type %r vs cfg.loss %r" % (loss_type, cfg.loss))
        return create_train_step_v2(cfg, ops, total_steps, aux_kind, is_gs, g_gen)

    engine.ns["create_train_step_v2"] = factory
    return "create_train_step_v2(loss=%s, aux=%s, loss_precision=%s, lam %g->%g over %g)" % (
        cfg.loss, aux_kind, cfg.loss_precision, cfg.lam0, cfg.lam1, cfg.anneal_frac)
