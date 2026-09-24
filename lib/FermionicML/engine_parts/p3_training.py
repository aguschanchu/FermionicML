# ============================================================================
# p3_training.py -- gram-metric initialization, TrainState, the parametrized
# pmap train step, spot-resilient train_model, jitted eval_step, prediction /
# evaluation helpers, checkpoint save/load, and the cell_26 gamma_base
# gradient-norm probe.
#
# Exec'd into the shared engine namespace AFTER p1_core.py / p2_models.py
# (see campaign/engine.py). Cross-part globals (basis, g_gen, M_PAIRS,
# rho_1_arrays, rho_2_kkbar_arrays, U_ENERGY_SEED, _ensure_dense,
# two_body_hamiltonian_dense, eigensolve_and_build_rho, NumpyLoader from p1;
# build_model from p2) resolve at CALL time via late binding -- exactly like
# the original notebook. Do not import other parts.
#
# Sources (verbatim transplants, edits marked with "# [engine edit]"):
#   .claude/cells/cell_20.py -> get_gram_matrix + the gram/physics-loss
#                               globals (wrapped into init_gram() so they are
#                               rebuilt from the CURRENT era set by
#                               init_d20/init_d12)
#   .claude/cells/cell_21.py -> is_master, print_model_summary, TrainState,
#                               eval_step, predict_and_load skeleton
#   .claude/cells/cell_26.py -> create_train_step (parametrized loss weights),
#                               train_model (timing/init_seed/gammas/probe),
#                               evaluate_run_metrics, _global_norm,
#                               get_probe_batch, make_component_probe
#   .claude/cells/cell_29.py -> save_model_and_history
#   .claude/cells/cell_30.py -> load_model (template rebuild + from_bytes)
#
# Spec edits on top of the sources:
#   * train_model: model built via the p2 build_model factory (arch +
#     ablation toggles); optimizer hyperparameters lifted to arguments
#     (peak_lr/weight_decay/clip, defaults = production values); per-epoch
#     np.random.seed(init_seed + epoch) for deterministic loader shuffles;
#     SPOT-RESILIENCE checkpointing (epoch_state.msgpack + meta.json after
#     every epoch, resume on entry, sync_cb after each save).
#   * predict_and_load returns (g_pred, g_true, energies) per campaign API.
#   * evaluate_run_metrics aligns the prediction's diagonal mean to the
#     target's (G -> G + cI gauge) before the Frobenius error.
# ============================================================================

import os
import json
import time
from functools import partial
from typing import Any

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import flax
from flax import serialization
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import optax
from tqdm import tqdm


is_master = (jax.process_index() == 0)  # [from cell_21]


# ---------------------------------------------------------------------------
# Global precomputations for the physics losses  [from cell_20]
# ---------------------------------------------------------------------------

def get_gram_matrix(rho_tensor):  # [from cell_20]
    """Computes the exact Hilbert-Schmidt Metric Tensor S_ab = Tr(L_a^dag L_b)"""
    rho_dense = np.asarray(rho_tensor.todense() if hasattr(rho_tensor, 'todense') else rho_tensor)
    m1, m2, N_dim, _ = rho_dense.shape

    # Align transposition to match how two_body_hamiltonian_dense constructs H
    # G[i, j] multiplies rho_tensor[j, i]
    L_tensor = rho_dense.transpose((1, 0, 2, 3))
    L_flat = L_tensor.reshape((m1 * m2, N_dim**2))

    print(f"Computing exact {m1*m2}x{m1*m2} Gram Matrix S...")
    Y_csr = sp.csr_matrix(L_flat.astype(np.float64))
    S_sparse = Y_csr @ Y_csr.T
    S = S_sparse.toarray()

    S_norm = S / float(N_dim)

    return 0.5 * (S_norm + S_norm.T)


def init_gram():
    """Build the cell_20 gram-metric globals from the CURRENT era globals
    (basis / rho_1_arrays / rho_2_kkbar_arrays / U_ENERGY_SEED, as set by
    init_d20 or init_d12). Sets: S_matrix_np (float64, (M^2, M^2)),
    S_tensor_global (float32), d_rho_1_diag_global, d_inter_tensor_global,
    base_e_params_global.

    [engine edit] cell_20 ran these statements at import time against the
    one-and-only notebook configuration; here they are wrapped so each era
    re-init can rebuild them.
    """
    global S_matrix_np, S_tensor_global
    global d_rho_1_diag_global, d_inter_tensor_global, base_e_params_global

    print("Initializing global physics tensors...")  # [from cell_20]

    # A. Operators for the 'eigh' Variational Loss  [from cell_20]
    rho_1_np = _ensure_dense(rho_1_arrays)
    if rho_1_np.ndim == 4:
        d_rho_1_diag_global = jnp.array(np.einsum('kknn->kn', rho_1_np))
    else:
        d_rho_1_diag_global = jnp.array(np.diagonal(rho_1_np, axis1=1, axis2=2))

    # [engine edit] cell_20 checked 'rho_2_kkbar_arrays' in globals(); here it
    # always exists after an era init but may be None-cleared, so guard both.
    use_kkbar = ('rho_2_kkbar_arrays' in globals()
                 and rho_2_kkbar_arrays is not None)
    target_inter_np = _ensure_dense(rho_2_kkbar_arrays if use_kkbar else rho_2_arrays)
    d_inter_tensor_global = jnp.array(target_inter_np)
    base_e_params_global = jnp.array(U_ENERGY_SEED[0])

    # B. Operators for the 'gram' Metric Loss  [from cell_20]
    S_matrix_np = get_gram_matrix(target_inter_np)

    # Export to JAX (using float32 for TPU speed)
    S_tensor_global = jnp.array(S_matrix_np, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Model summary + TrainState  [from cell_21]
# ---------------------------------------------------------------------------

def print_model_summary(model, input_shape, include_energy):  # [from cell_21]
    if not is_master:
        return None
    print("\n" + "="*80)
    res_info = f"(res={model.res})" if hasattr(model, 'res') else ""
    print(f" Model Summary: {model.__class__.__name__} {res_info}")
    print("="*80)

    rng = jax.random.PRNGKey(0)
    try:
        dummy_x = jnp.ones(input_shape)
        dummy_e = jnp.ones((input_shape[0], 1)) if include_energy else None
        print(model.tabulate(rng, dummy_x, dummy_e, training=False))
    except Exception as e:
        print(f"Summary unavailable: {e}")
        # Fallback to parameter counting if tabulate fails
        try:
            dummy_x = jnp.ones(input_shape)
            dummy_e = jnp.ones((input_shape[0], 1)) if include_energy else None
            variables = model.init(rng, dummy_x, dummy_e, training=False)
            flat = flax.traverse_util.flatten_dict(variables['params'], sep='/')
            total_params = sum(x.size for x in flat.values())
            print(f"Total Parameters: {total_params:,}")
        except:
            pass
    print("="*80 + "\n")


class TrainState(train_state.TrainState):  # [from cell_21]
    batch_stats: Any


# ---------------------------------------------------------------------------
# Parametrized pmap train step  [from cell_26]
# ---------------------------------------------------------------------------

def create_train_step(loss_type='gram', is_thermal=False, beta=100.0,
                      gamma_rdm=100.0, gamma_trace=1.0, gamma_base=1e-2,
                      gram_w_phys=1.0, gram_w_trace=1.0, gram_w_ridge=1e-2,
                      total_steps=None, v2=None):
    """Identical to production, with all loss weights lifted to parameters.
    Defaults reproduce the shipped configuration exactly. Each distinct
    weight set closes over its own compiled pmap step.

    [V2] v2=<ognrepro.v2.config.V2Config> (or loss_type 'metric') delegates to
    the v2 step factory installed at engine.ns['create_train_step_v2'] by
    ognrepro.v2.install(); total_steps is forwarded for its lambda schedule.
    With v2=None the body below is the published step, untouched."""  # [from cell_26]
    # the published body still serves: every call without v2, the 'rdm' loss
    # under v2 (the rdm step is unchanged by the v2 port), and a v2 config with
    # every v2 flag off (the pipeline control lane)
    use_v2 = loss_type == 'metric' or (
        v2 is not None and loss_type != 'rdm'
        and not (hasattr(v2, 'published') and v2.published()))
    globals()['_last_train_step_kind'] = 'v2' if use_v2 else 'published'   # [V2] echo for config.json
    if use_v2:
        fac = globals().get('create_train_step_v2')
        if fac is None:
            raise RuntimeError("create_train_step: v2 requested but ognrepro.v2.install() "
                               "has not bound create_train_step_v2 in this engine")
        return fac(loss_type, is_thermal, beta, total_steps_engine=total_steps, v2=v2,
                   gamma_rdm=gamma_rdm, gamma_trace=gamma_trace, gamma_base=gamma_base,
                   gram_w_phys=gram_w_phys, gram_w_trace=gram_w_trace, gram_w_ridge=gram_w_ridge)

    @partial(jax.pmap, axis_name='batch')
    def p_train_step(state, bx, be, by, S_tensor, base_e_params,
                     rho_1_diag, inter_tensor):
        def loss_fn(params):
            logits, updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                bx, be, training=True, mutable=['batch_stats']
            )
            logits_f64 = logits.astype(jnp.float64)
            by_f64 = by.astype(jnp.float64)

            H_pred_mat = g_gen.reconstruct(logits_f64)
            H_true_mat = g_gen.reconstruct(by_f64)

            base_loss = jnp.mean(jnp.sum(jnp.square(logits_f64 - by_f64), axis=-1))

            if loss_type == 'gram':
                B_size = H_pred_mat.shape[0]
                H_pred_flat = H_pred_mat.reshape((B_size, -1))
                H_true_flat = H_true_mat.reshape((B_size, -1))
                delta_H = H_pred_flat - H_true_flat

                # 1. PHYSICAL LOSS
                phys_variances = jnp.einsum('bi,ij,bj->b', delta_H, S_tensor, delta_H)
                phys_loss = jnp.mean(phys_variances)

                # 2. TIKHONOV REGULARIZATION
                norm_penalty = jnp.mean(jnp.sum(jnp.square(delta_H), axis=-1))

                # 3. TRACE LOSS (Gauge Fixing)
                pred_diag_mean = jnp.mean(jnp.diagonal(H_pred_mat, axis1=1, axis2=2), axis=1)
                true_diag_mean = jnp.mean(jnp.diagonal(H_true_mat, axis1=1, axis2=2), axis=1)
                trace_loss = jnp.mean(jnp.square(pred_diag_mean - true_diag_mean))

                total_loss = (gram_w_phys * phys_loss
                              + gram_w_trace * trace_loss
                              + gram_w_ridge * norm_penalty)
                return total_loss.astype(jnp.float32), updates

            if loss_type == 'rdm':
                B = bx.shape[0]
                e_params = jnp.tile(base_e_params.astype(jnp.float32), (B, 1))

                # 1. Reconstruct full H strictly in float32
                H_batch = two_body_hamiltonian_dense(
                    e_params, H_pred_mat.astype(jnp.float32),
                    d_rho_1_diag_global.astype(jnp.float32),
                    d_inter_tensor_global.astype(jnp.float32)
                )
                effective_beta = beta if is_thermal else 100.0
                state_rho_pred, _ = eigensolve_and_build_rho(H_batch, effective_beta)

                # Extract predicted RDM using the Einsum trace
                target_op = d_inter_tensor_global.astype(jnp.float32)
                orig_shape = target_op.shape[:-2]
                ops_flat = target_op.reshape(-1, target_op.shape[-2], target_op.shape[-1])
                rdm_pred_flat = jnp.einsum('bnm,kmn->bk', state_rho_pred, ops_flat)
                rdm_pred = rdm_pred_flat.reshape((B,) + orig_shape + (1,))

                # 1. RDM OBSERVABLE ERROR (The Zero-Temperature Physics)
                rdm_mse = jnp.mean(jnp.square(rdm_pred.real - bx.astype(jnp.float32).real))

                # 2. GAUGE FIX (Trace matching to break global shift degeneracy)
                pred_diag_mean = jnp.mean(jnp.diagonal(H_pred_mat, axis1=1, axis2=2), axis=1)
                true_diag_mean = jnp.mean(jnp.diagonal(H_true_mat, axis1=1, axis2=2), axis=1)
                trace_loss = jnp.mean(jnp.square(pred_diag_mean - true_diag_mean))

                # 3. TIE-BREAKER (Tikhonov Regularizer)
                total_loss = (gamma_rdm * rdm_mse
                              + gamma_trace * trace_loss
                              + gamma_base * base_loss)
                # FIX 0 (return inside the branch) -- keep this line.
                return total_loss.astype(jnp.float32), updates

            return base_loss.astype(jnp.float32), updates

        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Cross-TPU Gradient Synchronization
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        new_state = state.apply_gradients(grads=grads)
        new_batch_stats = updates.get('batch_stats', state.batch_stats)
        return new_state.replace(batch_stats=new_batch_stats), loss

    return p_train_step


# ---------------------------------------------------------------------------
# Jitted inference  [from cell_21]
# ---------------------------------------------------------------------------

@jax.jit
def eval_step(state, bx, be, by=None, training=False):
    # [from cell_21] signature there was (state, bx, be, by); `by` is unused.
    # [engine edit] `by` made optional and a `training` kwarg tolerated --
    # every caller (drivers, p5_shots) passes the dummy labels positionally,
    # and inference is ALWAYS training=False exactly as in cell_21.
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        bx, be, training=False
    )
    return logits


# ---------------------------------------------------------------------------
# Training driver  [from cell_26, patched per campaign spec]
# ---------------------------------------------------------------------------

def _count_loader_samples(loader):
    """Total-sample estimate from a NumpyLoader's shard inventory
    (first-shard size x file count, exactly the check_existing_dataset
    heuristic). Used only when total_samples is not supplied."""
    files = list(getattr(loader, 'files', []) or [])
    if not files:
        raise ValueError(
            "train_model: cannot infer total_samples (loader has no shard "
            "files); pass total_samples explicitly")
    with np.load(files[0]) as d:
        shard_size = len(d['labels'])
    return shard_size * len(files)


def train_model(train_loader=None, label_size=None, include_energy=True,
                input_shape=None, epochs=20, res=1, loss_type='gram',
                is_thermal=False, beta=100.0, init_seed=42, arch='ogn',
                use_scatter=True, use_reinject=True, use_orb_emb=True,
                readout_bias=0.55, batch_size=256, ckpt_dir=None, resume=True,
                probe_cb=None, sync_cb=None, gammas=None,
                peak_lr=3e-4, weight_decay=1e-4, clip=1.0,
                total_samples=None, v2=None, **_ignored):
    """[from cell_26 train_model] Patched per campaign spec:
      (i)   model built via the p2 build_model factory (arch + toggles);
      (ii)  init_seed threads the model-init PRNGKey AND the per-epoch
            np.random.seed(init_seed + epoch) loader-shuffle determinism;
      (iii) gammas dict forwarded to create_train_step;
      (iv)  per-epoch wall-clock recorded;
      (v)   optional probe_cb(host_state, epoch) -> dict, once per epoch;
      (vi)  SPOT-RESILIENCE: with ckpt_dir set, the unreplicated TrainState is
            flax-serialized to ckpt_dir/epoch_state.msgpack (+ meta.json) after
            EVERY epoch; on entry with resume=True an unfinished checkpoint is
            from_bytes-restored and training continues at epoch+1; sync_cb
            (best-effort) is called after each save.
    Returns (final_state, {'loss', 'epoch_seconds', 'grad_norms'})."""
    # [engine edit] loader aliases: t_train passes train_loader/dataset/loader,
    # the gbase-probe driver only dataset/loader.
    if train_loader is None:
        train_loader = _ignored.get('dataset', _ignored.get('loader'))
    if train_loader is None:
        raise ValueError("train_model: no training loader given "
                         "(train_loader / dataset / loader)")
    dataset = train_loader
    if label_size is None:
        raise ValueError("train_model: label_size is required")

    gammas = dict(gammas or {})  # [from cell_26]

    # [engine edit] drivers forward spec.get(...) values that may be None
    peak_lr = 3e-4 if peak_lr is None else float(peak_lr)
    weight_decay = 1e-4 if weight_decay is None else float(weight_decay)
    clip = 1.0 if clip is None else float(clip)

    if 'S_tensor_global' not in globals():
        raise RuntimeError("train_model: init_gram() must be called after the "
                           "era init (init_d20/init_d12) and before training")

    # [from cell_26] input shape derived from input_type/M_PAIRS when absent
    if input_shape is None:
        input_type = _ignored.get('input_type', 'rho2kkbar')
        m_pairs = int(_ignored['M_PAIRS']) if _ignored.get('M_PAIRS') else int(M_PAIRS)
        dim = m_pairs**2 if input_type == 'rho2block' else m_pairs
        input_shape = (1, dim, dim, 1)
    input_shape = tuple(input_shape)

    # ---- model topology via the p2 factory (late-bound)  [engine edit]
    # [V2] v2=<V2Config> reaches the v2 factory installed by ognrepro.v2.install();
    # with v2=None the call is the published one
    model = build_model(arch, label_size, res, include_energy,
                        use_scatter=use_scatter, use_reinject=use_reinject,
                        use_orb_emb=use_orb_emb, readout_bias=readout_bias,
                        **({'v2': v2} if v2 is not None else {}))
    print_model_summary(model, input_shape, include_energy)

    rng = jax.random.PRNGKey(init_seed)  # [from cell_26] (was hardcoded 42 in cell_21)
    dummy_x = jnp.ones(input_shape)
    dummy_e = jnp.ones((1, 1)) if include_energy else None
    variables = model.init(rng, dummy_x, dummy_e, training=False)

    if total_samples is None:
        total_samples = _count_loader_samples(dataset)

    # [from cell_21/26; engine edit: max(1, ...) guards the degenerate
    # smaller-than-one-batch smoke case]
    steps_per_epoch = max(1, int(total_samples) // int(batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(3000, int(0.05 * total_steps))

    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=peak_lr,
        warmup_steps=warmup_steps, decay_steps=total_steps, end_value=5e-6)
    tx = optax.chain(optax.clip_by_global_norm(clip),
                     optax.adamw(sched, weight_decay=weight_decay))

    state = TrainState.create(apply_fn=model.apply, params=variables['params'],
                              tx=tx, batch_stats=variables.get('batch_stats', {}))

    # ---- SPOT-RESILIENCE: restore an unfinished checkpoint  [engine edit]
    history, epoch_seconds, gradnorm_log = [], [], []
    start_epoch = 0
    state_path = meta_path = None
    if ckpt_dir:
        ckpt_dir = os.path.abspath(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        state_path = os.path.join(ckpt_dir, 'epoch_state.msgpack')
        meta_path = os.path.join(ckpt_dir, 'meta.json')
        if resume and os.path.exists(state_path) and os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                done = int(meta.get('epoch', -1))  # 0-based index of last finished epoch
                if 0 <= done < epochs:
                    with open(state_path, 'rb') as f:
                        state = serialization.from_bytes(state, f.read())
                    h = meta.get('hist') or {}
                    history = list(h.get('loss', []))[:done + 1]
                    epoch_seconds = list(h.get('epoch_seconds', []))[:done + 1]
                    gradnorm_log = list(h.get('grad_norms', []))
                    start_epoch = done + 1
                    if is_master:
                        print(f"Resumed {state_path}: epoch {done + 1}/{epochs} "
                              f"done, continuing at epoch {start_epoch + 1}")
            except Exception as exc:
                if is_master:
                    print(f"[resume warning] could not restore {ckpt_dir}: "
                          f"{exc}; training from scratch")
                history, epoch_seconds, gradnorm_log, start_epoch = [], [], [], 0

    # === SETUP FOR LOCAL DEVICES (pmap; works for n_devices == 1) ===
    # [from cell_21/26]
    n_devices = jax.local_device_count()
    device_batch = batch_size // n_devices

    p_state = replicate(state)
    p_S_tensor = replicate(S_tensor_global)
    p_base_e_params = replicate(base_e_params_global)
    p_rho_1_diag = replicate(d_rho_1_diag_global)
    p_inter_tensor = replicate(d_inter_tensor_global)

    # Compile the specific step based on the requested loss
    # [V2] total_steps / v2 forwarded (ignored by the published gram/rdm/mse step)
    p_train_step = create_train_step(loss_type, is_thermal, beta,
                                     total_steps=total_steps, v2=v2, **gammas)

    if is_master:
        print(f"Starting Training | Loss: {loss_type.upper()} | gammas={gammas} "
              f"| seed={init_seed} | devices(local): {n_devices}")

    for ep in range(start_epoch, epochs):
        # [engine edit -- PER-EPOCH DETERMINISM] NumpyLoader shuffles via the
        # global numpy RNG; reseeding here makes every epoch's shard/sample
        # order a pure function of (init_seed, epoch), so resumed runs replay
        # exactly the stream they would have seen.
        np.random.seed(init_seed + ep)

        t0 = time.time()
        loss_acc, steps = 0.0, 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep+1}", leave=False,
                    disable=not is_master)
        for big_x, big_e, big_y in dataset:
            chunk_size = big_x.shape[0]
            for start in range(0, chunk_size, batch_size):
                end = min(start + batch_size, chunk_size)
                if (end - start) != batch_size:
                    continue
                bx = jnp.array(big_x[start:end]).reshape(n_devices, device_batch, *big_x.shape[1:])
                by = jnp.array(big_y[start:end]).reshape(n_devices, device_batch, *big_y.shape[1:])
                be = (jnp.array(big_e[start:end]).reshape(n_devices, device_batch, *big_e.shape[1:])
                      if include_energy else None)
                # ---> P_STATE IS OVERWRITTEN ITERATIVELY <---
                p_state, p_loss = p_train_step(
                    p_state, bx, be, by, p_S_tensor, p_base_e_params,
                    p_rho_1_diag, p_inter_tensor)
                loss_val = np.array(p_loss)[0].item()
                loss_acc += loss_val
                steps += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.5f}")
                if steps >= steps_per_epoch:
                    break
            if steps >= steps_per_epoch:
                break
        pbar.close()

        epoch_seconds.append(time.time() - t0)
        avg_loss = loss_acc / steps if steps > 0 else float('nan')
        history.append(avg_loss)

        if probe_cb is not None:  # [from cell_26]
            try:
                rec = probe_cb(unreplicate(p_state), ep)
                if rec is not None:
                    gradnorm_log.append(rec)
            except Exception as exc:                      # never kill a run on a probe
                if is_master:
                    print(f"  [probe warning] epoch {ep+1}: {exc}")

        # ---- SPOT-RESILIENCE: serialize after EVERY finite epoch  [engine edit]
        if ckpt_dir and np.isfinite(avg_loss):
            host_state = unreplicate(p_state)
            tmp = state_path + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(serialization.to_bytes(host_state))
            os.replace(tmp, state_path)
            meta = {'epoch': ep,
                    'hist': {'loss': history,
                             'epoch_seconds': epoch_seconds,
                             'grad_norms': gradnorm_log}}
            tmp = meta_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(meta, f)
            os.replace(tmp, meta_path)
            if sync_cb is not None:
                try:
                    sync_cb(ckpt_dir)
                except Exception as exc:
                    if is_master:
                        print(f"  [sync warning] epoch {ep+1}: {exc}")

        if is_master:
            print(f"Epoch {ep+1} | Loss: {avg_loss:.6f} | {epoch_seconds[-1]:.1f}s")
        if not np.isfinite(avg_loss):
            print(f"!! Non-finite loss at epoch {ep+1}; aborting this run.")
            break

    return unreplicate(p_state), {'loss': history,
                                  'epoch_seconds': epoch_seconds,
                                  'grad_norms': gradnorm_log}


# ---------------------------------------------------------------------------
# Checkpoint save / load  [from cell_29 / cell_30]
# ---------------------------------------------------------------------------

def save_model_and_history(state, hist, save_dir):
    """Securely serializes the TrainState and history to disk."""  # [from cell_29]
    # [engine edit] absolute-path-safe (cell_29's default was the relative
    # "./home/..." typo); save_dir is now a required argument.
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    state_path = os.path.join(save_dir, "final_state.msgpack")
    hist_path = os.path.join(save_dir, "hist.npy")

    # 1. Save TrainState using Flax's built-in msgpack serialization (Secure, no pickle)
    with open(state_path, "wb") as f:
        f.write(serialization.to_bytes(state))

    # 2. Save history (NumPy binary format, allow_pickle=False ensures safety)
    # [engine edit] hist is now the cell_26 dict {'loss','epoch_seconds',
    # 'grad_norms'}; np.save(allow_pickle=False) cannot store dicts, so the
    # loss curve keeps the cell_29 hist.npy layout and the full dict is
    # written to hist.json alongside.
    if isinstance(hist, dict):
        loss_hist = hist.get('loss', [])
        try:
            with open(os.path.join(save_dir, "hist.json"), "w") as f:
                json.dump(hist, f, indent=1, default=float)
        except Exception as exc:
            print(f"Warning: hist.json not written: {exc}")
    else:
        loss_hist = list(hist) if hist is not None else []
    np.save(hist_path, np.asarray(loss_hist, dtype=np.float64),
            allow_pickle=False)

    print(f"Saved state to {state_path} and history to {hist_path}")


def load_model(ckpt_dir, arch, label_size, res, include_energy, input_shape,
               use_scatter=True, use_reinject=True, use_orb_emb=True,
               readout_bias=0.55, **_ignored):
    """Loads a trained TrainState from ckpt_dir/final_state.msgpack (fallback:
    ckpt_dir/epoch_state.msgpack, the spot-resilience checkpoint).
    [from cell_30 load_model_and_history; engine edit: model rebuilt via the
    p2 build_model factory with the ablation toggles, dummy tx built inline,
    history loading dropped -- returns the TrainState only.]"""
    ckpt_dir = os.path.abspath(ckpt_dir)
    state_path = os.path.join(ckpt_dir, "final_state.msgpack")
    if not os.path.exists(state_path):
        fallback = os.path.join(ckpt_dir, "epoch_state.msgpack")
        if os.path.exists(fallback):
            state_path = fallback
        else:
            raise FileNotFoundError(
                f"load_model: neither final_state.msgpack nor "
                f"epoch_state.msgpack found in {ckpt_dir}")

    # 1. Re-instantiate the precise model topology
    model = build_model(arch, label_size, res, include_energy,
                        use_scatter=use_scatter, use_reinject=use_reinject,
                        use_orb_emb=use_orb_emb, readout_bias=readout_bias)

    # 2. Initialize dummy inputs to extract the PyTree shapes  [from cell_30]
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones(tuple(input_shape))
    dummy_e = jnp.ones((1, 1)) if include_energy else None

    # 3. Get structural variables (params and batch_stats)
    variables = model.init(rng, dummy_x, dummy_e, training=False)

    # 4. Dummy tx with the SAME pytree structure as the train_model optimizer
    # (chain(clip, adamw(schedule))), so from_bytes can map opt_state.
    # [from cell_30 -- identical dummy schedule values]
    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=3e-4, warmup_steps=3000,
        decay_steps=10000, end_value=5e-6)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=1e-4))

    # 5. Create the empty custom TrainState template
    template_state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables.get('batch_stats', {})
    )

    # 6. Securely map the bytes from disk into the empty state template
    with open(state_path, "rb") as f:
        byte_data = f.read()
    restored_state = serialization.from_bytes(template_state, byte_data)

    print(f"Loaded state from {state_path}")
    return restored_state


# ---------------------------------------------------------------------------
# Prediction + run metrics  [from cell_21 / cell_26]
# ---------------------------------------------------------------------------

def predict_and_load(state_model, loader, max_samples=None, include_energy=True):
    """Iterates the NumpyLoader, performs JAX inference in mini-batches and
    accumulates results into numpy arrays.
    [from cell_21 predict_and_load; engine edit: returns
    (g_pred, g_true, energies) per the campaign API, max_samples=None = all.]"""
    gpu_batch_size = 256  # [from cell_21 default]
    all_energies, all_g_true, all_g_pred = [], [], []
    count = 0

    for big_x, big_e, big_y in loader:
        if max_samples is not None and count >= max_samples:
            break
        chunk_size = big_x.shape[0]

        # Inner loop: Slice large chunk into GPU-friendly mini-batches
        for start in range(0, chunk_size, gpu_batch_size):
            end = min(start + gpu_batch_size, chunk_size)

            bx = jnp.array(big_x[start:end])
            by = jnp.array(big_y[start:end])
            be = (jnp.array(big_e[start:end])
                  if (include_energy and big_e is not None) else None)

            logits = eval_step(state_model, bx, be, by)

            if be is not None:
                all_energies.append(np.array(be))
            else:
                all_energies.append(np.zeros((end - start, 1)))
            all_g_true.append(np.array(by))
            all_g_pred.append(np.array(logits))

        count += chunk_size

    if not all_g_pred:
        raise ValueError("Dataset loader returned no data.")

    g_pred = np.concatenate(all_g_pred, axis=0)
    g_true = np.concatenate(all_g_true, axis=0)
    energies = np.concatenate(all_energies, axis=0)
    if max_samples is not None:
        g_pred = g_pred[:max_samples]
        g_true = g_true[:max_samples]
        energies = energies[:max_samples]
    return g_pred, g_true, energies


def evaluate_run_metrics(state_model=None, loader=None, max_samples=20000,
                         **_ignored):
    """Gauge-aligned relative parameter error (median/mean) + relative
    S-metric error on `loader`'s samples, host-side float64.
    [from cell_26 evaluate_run_metrics; engine edits: consumes an existing
    loader instead of regenerating a val cache; the Frobenius error is taken
    AFTER aligning the prediction's diagonal mean to the target's (the
    G -> G + cI gauge); the rho-forward RMSE block is owned by the ablation
    driver and dropped here.]"""
    # [engine edit] alias resolution for _call_filtered-style drivers
    if state_model is None:
        state_model = _ignored.get('state')
    if loader is None:
        loader = _ignored.get('val_loader', _ignored.get('dataset'))
    if state_model is None or loader is None:
        raise ValueError("evaluate_run_metrics: need state_model and loader")
    include_energy = bool(_ignored.get('include_energy', True))
    chunk = int(_ignored.get('gpu_batch_size') or 256)

    g_pred, g_true, _energies = predict_and_load(
        state_model, loader, max_samples=max_samples,
        include_energy=include_energy)
    g_pred = np.nan_to_num(g_pred)

    # [from cell_26] chunked reconstruction via g_gen (late-bound)
    pm, am = [], []
    for s in range(0, len(g_pred), chunk):
        e = min(s + chunk, len(g_pred))
        pm.append(np.array(g_gen.reconstruct(jnp.array(g_pred[s:e]))))
        am.append(np.array(g_gen.reconstruct(jnp.array(g_true[s:e]))))
    pred_mat = np.concatenate(pm, 0).astype(np.float64)
    true_mat = np.concatenate(am, 0).astype(np.float64)

    # Gauge alignment: shift the prediction's diagonal mean onto the target's
    # before the Frobenius distance (penalizes shape, not the cI gauge mode).
    m = pred_mat.shape[1]
    shift = (np.trace(true_mat, axis1=1, axis2=2)
             - np.trace(pred_mat, axis1=1, axis2=2)) / float(m)
    pred_aligned = pred_mat + shift[:, None, None] * np.eye(m)[None, :, :]

    pf = pred_aligned.reshape((len(pred_mat), -1))
    tf = true_mat.reshape((len(true_mat), -1))
    dn = np.linalg.norm(pf - tf, axis=1)
    tn = np.linalg.norm(tf, axis=1)
    rel = dn / np.maximum(tn, 1e-12)                       # ||dG||_F / ||G||_F

    # [from cell_26] relative S-metric error via the float64 S_matrix_np
    # (raw, un-aligned delta -- the S metric is itself gauge-aware)
    S64 = S_matrix_np.astype(np.float64)
    d = pred_mat.reshape((len(pred_mat), -1)) - tf
    dSd = np.einsum('bi,ij,bj->b', d, S64, d)
    tSt = np.einsum('bi,ij,bj->b', tf, S64, tf)
    relS = np.sqrt(np.maximum(dSd, 0) / np.maximum(tSt, 1e-12))  # NOT f_err; GEVP later

    return {
        'rel_param_err_median': float(np.median(rel)),
        'rel_param_err_mean': float(np.mean(rel)),
        'rel_S_err_median': float(np.median(relS)),
        'rel_S_err_mean': float(np.mean(relS)),
        'n_param': int(len(rel)),
    }


# ---------------------------------------------------------------------------
# gamma_base gradient-norm probe  [from cell_26]
# ---------------------------------------------------------------------------

def _global_norm(tree):  # [from cell_26]
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(l.astype(jnp.float32))) for l in leaves)))


def get_probe_batch(loader=None, probe_path=None, n=64, **_ignored):
    """One fixed batch, saved once, reused by every run -> comparable norms.
    [from cell_26 get_probe_batch; engine edit: takes a loader (or a shard
    cache dir via cache_path/val_cache/val_dir kwargs) instead of calling
    gen_dataset against notebook-global config.]"""
    if probe_path and os.path.exists(probe_path):
        z = np.load(probe_path)
        return z['bx'], (z['be'] if 'be' in z.files else None), z['by']

    include_energy = bool(_ignored.get('include_energy', True))
    if loader is None:
        cache = (_ignored.get('cache_path') or _ignored.get('val_cache')
                 or _ignored.get('val_dir'))
        if cache is None:
            raise ValueError("get_probe_batch: need a loader or a "
                             "cache_path/val_cache/val_dir kwarg")
        loader = NumpyLoader(cache, max(64, int(n)), shuffle=False)

    bx = be = by = None
    for big_x, big_e, big_y in loader:
        bx, by = big_x[:n], big_y[:n]
        be = big_e[:n] if (include_energy and big_e is not None) else None
        break
    if bx is None:
        raise RuntimeError("get_probe_batch: loader yielded no data")
    if probe_path and is_master:
        kw = dict(bx=bx, by=by) if be is None else dict(bx=bx, be=be, by=by)
        np.savez_compressed(probe_path, **kw)
    return bx, be, by


def make_component_probe(loss_type, is_thermal, beta, probe_batch):
    """Per-epoch ||grad L_i|| for each UNWEIGHTED loss component, on one fixed
    batch, eval-mode (training=False), single device on the master host.
    For 'rdm': rdm_mse / trace / base.  For 'gram': phys / trace / ridge."""
    # [from cell_26, verbatim]
    bx_np, be_np, by_np = probe_batch
    bx = jnp.array(bx_np); by = jnp.array(by_np)
    be = jnp.array(be_np) if be_np is not None else None
    beta_eff = beta if is_thermal else 100.0

    def comp_losses(params, batch_stats):
        logits = state_apply(params, batch_stats)
        logits_f64 = logits.astype(jnp.float64)
        by_f64 = by.astype(jnp.float64)
        Hp = g_gen.reconstruct(logits_f64)
        Ht = g_gen.reconstruct(by_f64)
        base = jnp.mean(jnp.sum(jnp.square(logits_f64 - by_f64), axis=-1))
        pdm = jnp.mean(jnp.diagonal(Hp, axis1=1, axis2=2), axis=1)
        tdm = jnp.mean(jnp.diagonal(Ht, axis1=1, axis2=2), axis=1)
        trace = jnp.mean(jnp.square(pdm - tdm))
        if loss_type == 'gram':
            B = Hp.shape[0]
            dH = (Hp.reshape((B, -1)) - Ht.reshape((B, -1)))
            phys = jnp.mean(jnp.einsum('bi,ij,bj->b', dH, S_tensor_global, dH))
            ridge = jnp.mean(jnp.sum(jnp.square(dH), axis=-1))
            return {'phys': phys, 'trace': trace, 'ridge': ridge, 'base': base}
        B = bx.shape[0]
        e_params = jnp.tile(base_e_params_global.astype(jnp.float32), (B, 1))
        H_batch = two_body_hamiltonian_dense(
            e_params, Hp.astype(jnp.float32),
            d_rho_1_diag_global.astype(jnp.float32),
            d_inter_tensor_global.astype(jnp.float32))
        rho_p, _ = eigensolve_and_build_rho(H_batch, beta_eff)
        top = d_inter_tensor_global.astype(jnp.float32)
        osh = top.shape[:-2]
        ofl = top.reshape(-1, top.shape[-2], top.shape[-1])
        rp = jnp.einsum('bnm,kmn->bk', rho_p, ofl).reshape((B,) + osh + (1,))
        rdm_mse = jnp.mean(jnp.square(rp.real - bx.astype(jnp.float32).real))
        return {'rdm_mse': rdm_mse, 'trace': trace, 'base': base}

    state_apply = None  # bound per call via closure rebinding below

    def probe_cb(host_state, epoch):
        if not is_master:
            return None
        nonlocal state_apply
        def _sa(params, batch_stats):
            return host_state.apply_fn({'params': params, 'batch_stats': batch_stats},
                                       bx, be, training=False)
        state_apply = _sa
        names = list(comp_losses(host_state.params, host_state.batch_stats).keys())
        rec = {'epoch': int(epoch)}
        for name in names:
            gfun = jax.grad(lambda p: comp_losses(p, host_state.batch_stats)[name])
            rec[f'gnorm_{name}'] = _global_norm(gfun(host_state.params))
        for name, v in comp_losses(host_state.params, host_state.batch_stats).items():
            rec[f'loss_{name}'] = float(v)
        return rec

    return probe_cb


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def n_params(state):
    """Total trainable parameter count of a TrainState (sum over param leaves)."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
