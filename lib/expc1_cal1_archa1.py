#!/usr/bin/env python3
# =============================================================================
# experiments/experiment-campaign1/stages/expc1_cal1_archa1.py
#
# CPU-track stage for TWO items of panel/FINAL_PLAN.md (v1.0-panel, chair-
# ratified 2026-07-29):
#
#   CAL-1  (FINAL_PLAN.md "### CAL-1", ~:559) — pre-window noise-grid
#          seed-spread calibration: the registered prod triplet
#          (prod_thermal_random seed 42, abl_gram_s43, abl_gram_s44) evaluated
#          on the seed-1007 dev stream noise grid, 6 population-noise budgets
#          x 100 Hamiltonians x 20 Gaussian draws, CPU-float64 convention.
#          Registers per-budget per-seed medians, 3-seed spread/SE and the
#          derived gate thresholds max(10%, 2xSE) for G3 / NOISE-claim and
#          max(5%, 2xSE) for G-NINJ (FINAL_PLAN §3 rule 5, §4 R5/R16, §6).
#
#   ARCH-A1 (FINAL_PLAN.md "### R6. ARCH-A1", ~:346) — CPU-f64 capacity-ladder
#          rescore: campaign6 originals mlp_cap_{1p1M,4p4M,17M}_{b4,b8} PLUS
#          the reproA-20260725 duplicates, plus abl_mlp and the OGN prod
#          triplet, on the seed-1007 tab3_4096 dev rows under the exact rh_c3
#          CPU protocol.  Emits the capacity exponent alpha-hat on
#          1.1M->4.4M->17.4M with (i) per-Hamiltonian cluster-bootstrap eval
#          CI and (ii) a run-noise term sigma_run from the original-vs-reproA
#          duplicate pairs, and the G-A1 decision branch
#          (authorize / close / straddle; FINAL_PLAN §4 R6 + gate table §6).
#
# LINEAGE (mirrored, never reinvented; file:line citations):
#  * CAL-1 noise machinery == the registered Table-III noise-eval path:
#      - Gaussian population-noise drawer VERBATIM
#        campaign4/engine_parts/p5_shots.py:202-216 (_shots_draw_gaussian);
#        runtime bit-identity gate against the engine copy.
#      - draw streams SeedSequence([0, stream 2, i_s(SWEEP50), si, t]) — the
#        registered shots_master streams, p5_shots.py:384-385, reused verbatim
#        by campaign10/stages/c10_tableiii_cpu64.py:361-363 (STAT-04).
#      - budget grid = SWEEP50 indices [0,12,17,24,37,49]
#        (campaign4/compute.py:544 SWEEP50 = np.geomspace(1e10,1e2,50);
#        c10_tableiii_cpu64.py:126 NODES6; Table III / STAT-04 nodes:
#        N_s = 1e10, 1.1e8, 1.7e7, 1.2e6, 9.1e3, 1e2).
#      - measured-ensemble -> features: rho_meas=(V*p_meas)@V.T; pair block
#        einsum('bji,kij->bk') = compute_rho_m (p1_core.py:645-668) WITHOUT
#        its float32 cast; E_meas = sum(p_meas*E_true)
#        (c10_tableiii_cpu64.py:339-371 float64 branch, itself
#        p5_shots.py:389-392 minus the f32 cast).
#      - CPU-f64 convention == the STAT-04 cpu64 deliverable branch
#        (c10_tableiii_cpu64.py:33-36,509-518,575-585): float64 features +
#        float64 params + jax_enable_x64=True + default_matmul_precision
#        'highest' + float64 scoring.
#      - scoring: _shots_shifted_error diag-mean gauge (p5_shots.py:170-175
#        == err_q of campaign10/stages/c10_h0_sensitivity.py:213-219).
#      - DEVIATION FROM STAT-04 (the CAL-1 item itself): the Hamiltonian set
#        is the REGISTERED seed-1007 dev labels rows 0..100 (not the
#        seed-4242 held-out stream), per FINAL_PLAN CAL-1 "dev-1007 noise
#        grid" + §3 rule 5 (100 H x 20 minimum).  Systems are built exactly
#        as p5_shots.prepare_heldout_systems (p5_shots.py:268-313, WLS rows
#        skipped) with the G source swapped to the dev labels.
#  * ARCH-A1 == the rh_c3 seed-1007 CPU arm
#    (campaign11/roundH/stages/rh_c3.py): float32 dataset-pipeline features
#    regenerated from the registered val labels rows 0..4096 through
#    engine6.solve_batch_kernel (rh_c3.py:594-625 build_features_float32,
#    itself rg_h6.py:612-643 / p1_core.py:783-855), float32 forward,
#    float64 err_pair scoring (rh_c3.py:379-387), per-Hamiltonian cluster
#    bootstrap B=1000 / default_rng(20260717) / order stats 25/975
#    (rh_c3.py:178-183,415-446; FINAL_PLAN §3 rule 5 "rh_c3 pins").
#  * checkpoint discipline: byte verification BEFORE deserialization
#    (campaign10 checkpoint_authority contract, self-contained here: the
#    stage hashes the exact bytes it deserializes and gates them against the
#    registered pins); model topologies from c10_tableiii_cpu64.py:273-305
#    (OGN prod), rh_c3.py:546-591 (abl_mlp / mlp_cap), reproA receipts
#    (ladder widths/blocks).
#  * fmb fork-safety: repro_audit/stages/repro_train.py:164-181 pattern —
#    campaign10 fmb_serial10.install_fmb_serial_policy() installed BEFORE the
#    engine import when available; installation status recorded; the CAL-1
#    process pool uses the SPAWN start method only (never os.fork of the
#    jax-holding process) and workers are numpy-only.
#
# CHECKPOINTS OF RECORD (r20 PRV manifest campaign10/results/runs/
# c10-20260715-r20/PRV-01/sha_manifest_full.json, all present_hashed, +
# reproA-20260725 receipts campaign11/results/reproA/T-*/):
#   prod_thermal_random 7b7bc8e9... (== repro_audit/expected_published.json)
#   abl_gram_s43 cf4ce2ba... / abl_gram_s44 a3077969... (the registered
#     seeds-43/44 production-spec checkpoints behind the printed
#     3.77/3.85/3.86e-3 triplet; NOT the reproA-20260725 v4 fresh retrains)
#   abl_mlp 33b7114d... ; mlp_cap ladder per pins below.
#   AUTHORING-TIME FINDING (recorded, gate-relevant): all six reproA ladder
#   duplicates are BIT-IDENTICAL to the campaign6 originals (receipt shas ==
#   r20 rows), so sigma_run is exactly 0 wherever bit-identity holds; the
#   stage verifies this from the local bytes and records it per pair.
#
# CONVENTIONS PRODUCED (documented per task ruling):
#   CAL-1 emits the CPU-f64 convention ONLY (STAT-04 cpu64 branch).  The
#   legacy convention of the printed noise tables is accelerator-bf16/f32 at
#   default matmul and is NOT producible on CPU; the accelerator-style
#   CPU-f32 proxy was explicitly ruled out as not needed.  The result JSON
#   carries this statement under results.conventions.
#   ARCH-A1 emits the CPU-f64 scoring convention of rh_c3 (f32 pipeline
#   features + f32 forward + f64 error arithmetic) — the convention of the
#   registered 1.675e-3 / 1.0168e-3 anchors quoted by FINAL_PLAN §2.3.
#
# GATES (result JSON gates.*.{pass, rationale}; all_gates_pass = AND over
# executed gates AND both parts present AND not smoke):
#   G1_checkpoints_loaded_digests  every requested checkpoint loaded from
#       byte-verified bytes; sha256 recorded and == the registered pins;
#       cross-check vs expected_published.json recorded (match / mismatch /
#       absent) — gate decides on the RECORDED pins.
#   G2_perbudget_se_finite_registered  (cal1) all per-budget 3-seed SEs
#       finite, > 0, and written into the registered results block.
#   G3_thresholds_table_emitted    (cal1) the per-budget threshold table
#       max(10%, 2xSE) [G3 / NOISE-claim] and max(5%, 2xSE) [G-NINJ] emitted.
#   G4_alpha_gA1_emitted           (archa1) alpha-hat + eval CI + run-noise
#       combined CI + G-A1 branch emitted and finite.
#   plus auxiliary hard gates (inputs byte-pinned, drawer identity, panel
#   identity, tab3 anchor rescore identity, x64 active, determinism,
#   finiteness, stats authority) — see the payload gate blocks.
#
# CLI:
#   expc1_cal1_archa1.py <out_dir> [--part cal1|archa1|all] [--nproc N]
#                        [--ckpt-root DIR] [--smoke] [--selftest]
#     --selftest   structural numpy-only checks; no jax/flax/scipy/engine and
#                  no registered inputs required; passes on any python3+numpy
#     --ckpt-root  local checkpoint mirror produced by expc1_fetch_ckpts.sh
#                  (layout <root>/<model>/final_state.msgpack and
#                   <root>/reproA-20260725/<model>/final_state.msgpack);
#                  REQUIRED for any evaluation run
#     --part all   runs cal1 and archa1 as two SUBPROCESSES (cal1 needs
#                  jax_enable_x64=True, archa1 must run without it), then
#                  merges; single parts run in-process and re-merge whatever
#                  payloads exist in <out_dir>
#     --smoke      reduced grid; gates evaluated; all_gates_pass FORCED False
#                  (smoke records are never quotable; AUD-02 precedent)
#
# OUTPUT: <out_dir>/expc1_cal1_archa1_result.json (merged)
#         <out_dir>/expc1_part_cal1.json / expc1_part_archa1.json (payloads)
#         <out_dir>/expc1_cal1_perseed.npz   (per-seed predictions/errors —
#                                             shared with ENS-1 per §7.1.2)
#         <out_dir>/expc1_archa1_persample.npz
# Pinned stack: python 3.10 / jax 0.6.2 CPU / flax 0.10.7 / numpy 2.2.6
# (JAX_PLATFORMS=cpu assumed).  Prints tagged [expc1_cal1].
# =============================================================================
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')                       # BEFORE numpy
os.environ.setdefault('JAX_PLATFORMS', 'cpu')            # BEFORE any jax import

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time

import numpy as np

TAG = '[expc1_cal1]'
STAGE = 'expc1_cal1_archa1'
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

# ------------------------------------------------------------------- pins
MP = 10                                   # pair levels (d=20)
LABEL55 = 55
BETA_THERMAL, G_INIT, G_STOP = 1.0, 0.1, 1.0      # campaign4/config.py:64-65
DIAG_GAUGE_MEAN, DIAG_GAUGE_TOL = 0.55, 5e-7      # rh_c3.py:181 (c10 GP0)

NODES6 = (0, 12, 17, 24, 37, 49)          # c10_tableiii_cpu64.py:126
SWEEP50_SPEC = (1e10, 1e2, 50)            # campaign4/compute.py:544
NODES6_NS_PIN = (1e10, 109854114.19875573, 16768329.3681101,
                 1206792.6406393289, 9102.981779915226, 100.0)
SHOTS_STREAM_GAUSSIAN = 2                 # p5_shots.py:39 SHOTS_NOISE_STREAM
SHOTS_SEED0 = 0                           # p5_shots.py:384 / STAT-04 seed=0
N_H_CAL, N_DRAWS_CAL = 100, 20            # FINAL_PLAN §3 rule 5 / CAL-1
N_H_CAL_SMOKE, N_DRAWS_CAL_SMOKE = 10, 3
N_EVAL_ARCH, N_EVAL_ARCH_SMOKE = 4096, 256   # rh_c3 tab3_4096 rows
EVAL_CHUNK_F64 = 512                      # c10_tableiii_cpu64.py:129
EVAL_CHUNK_F32 = 256                      # rh_c3.py:179 EVAL_BS
BOOT_SEED, BOOT_B = 20260717, 1000        # rh_c3 pins (FINAL_PLAN §3 rule 5)
BOOT_LO, BOOT_HI = int(0.025 * BOOT_B), int(0.975 * BOOT_B)          # 25/975
GA1_LOW, GA1_HIGH = 0.05, 0.08            # FINAL_PLAN R6 G-A1 boundaries

# --- registered inputs (repo-relative, sha256 recomputed 2026-07-29 on the
#     read-only trees; any mismatch is a staging error -> hard fail)
REC = {
    'val_npz': (
        'campaign5/results/boot95_inputs/inputs/'
        'prod_thermal_random_predictions_val.npz',
        'e215e0fa2e6c44d4992cf314f39ae662a0d5e662cfbb3305983b56356a2ea900'),
    'val_first4096': (      # accepted fallback (rh_c3/rg_h6.deps precedent)
        'campaign8/inputs/prod_thermal_random_predictions_val_first4096.npz',
        'fbbb4ce2dde3e584152dfaf82a1eea58a5cafee35fbea88445a36e963e92136a'),
    'c6_capacity_eval': (
        'campaign6/results6/tpu-v5e-spot-us-a/c6_capacity_eval/'
        'c6_capacity_eval.json',
        '112577726714559f848b9dd57fb8ae0fdc3bce9fadfa3e8da5ba70cd3ae5d5d1'),
    'r20_manifest': (
        'campaign10/results/runs/c10-20260715-r20/PRV-01/'
        'sha_manifest_full.json',
        '57a75d5f5ad6094689a2917e4fd45de899e610736cae0cc940d2652ec18adeb2'),
    'config01_config': (
        'campaign10/results/CONFIG-01/mlp_cap_17M_b8_config.json',
        '3d77313701daaad096db3851c533494a61723ae0ae7c9763952f4e73d233defe'),
    'rh_c3_json': (
        'campaign11/results/roundH/C3/rh_c3.json',
        '543f08103bde29917173bfe9cf140b42ff3e84f91434a912aac6621066600cc5'),
}
# expected_published.json — resolved at ~/reproA_code first, then the repo
# copy under repro_audit/ (task ruling); pin = the repro_audit copy.
EXPECTED_PUBLISHED_CANDS = ('~/reproA_code/expected_published.json',
                            'repro_audit/expected_published.json')
EXPECTED_PUBLISHED_SHA = \
    '756614ab8533c42958cba158714ae4095a740b2edd7a616277c0fad1e9da0c86'

# reproA ladder ckpt_config.json copies (standardization stats cross-width
# authority companions; receipts campaign11/results/reproA/T-*/DONE.json)
CKPT_CONFIG_PINS = {
    'mlp_cap_1p1M_b4':
        '6c6ee1224bbedeeb27467b88ac8da3e14c6287e78e1945a660106c160bfe58e2',
    'mlp_cap_1p1M_b8':
        '90ef22a7b52f654132fc001262855148177286426b60afd2dc7fb565f822b4d3',
    'mlp_cap_4p4M_b4':
        '594faf1bee460e654cc34d7f5b279b22999aa77fdb0d312c3839e771d349dfd9',
    'mlp_cap_4p4M_b8':
        '140f2f261acf6a55efc9ca326e20fa06fb03d389e7b78888812692a2093184ab',
    'mlp_cap_17M_b4':
        '9a95fb5063f6a53a10df7dc2ce9187e057b97056a7810bbd47d3e3d06e907527',
    'mlp_cap_17M_b8':
        'cb66ae64dbaac3737819b48f7cb5edd7313a80750b744b51fa330705dc9acfc0',
}
CONFIG01_STATS_SHA = ('8474e6eaa23e8fcc9ced6de861f8a7d727ebad2b9a6b9adaae'
                      '79b68849f83183')     # campaign10/config_authority10.py

# --- checkpoints of record: name -> (sha256, bytes, kind, spec)
#     shas = r20 PRV manifest rows (originals) / reproA-20260725 receipts
#     (duplicates; verified equal to the originals at authoring time).
GCS_BUCKET = 'gs://iflp-486215-fermionicml-campaign/checkpoints'
CKPTS = {
    # OGN production triplet (registered; arch/res per
    # c10_tableiii_cpu64.py:273-305 and reproA c1 specs: arch ogn, res 3)
    'prod_thermal_random': dict(
        sha='7b7bc8e92e8d973a9dba19bd247ebb90a8281aa500615dcf57fd300f8388999b',
        bytes=213094675, kind='ogn', res=3, init_seed=42, n_params=17756929,
        gcs=GCS_BUCKET + '/tpu-v6e-spot-eu-a/prod_thermal_random/'
                         'final_state.msgpack'),
    'abl_gram_s43': dict(
        sha='cf4ce2baf44de432f340b382e1377010678c7c4fafc534422550e2d498da5f65',
        bytes=213094675, kind='ogn', res=3, init_seed=43, n_params=17756929,
        gcs=GCS_BUCKET + '/tpu-v6e-spot-eu-a/abl_gram_s43/'
                         'final_state.msgpack'),
    'abl_gram_s44': dict(
        sha='a307796938f81670896d04c3a631878e116924cbc0fab847a855c3118a8754e5',
        bytes=213094675, kind='ogn', res=3, init_seed=44, n_params=17756929,
        gcs=GCS_BUCKET + '/tpu-v6e-spot-eu-a/abl_gram_s44/'
                         'final_state.msgpack'),
    # unstandardized 17.4M MLP (rh_c3 pins; campaign5/train/config5.py:152-155)
    'abl_mlp': dict(
        sha='33b7114d0a3e184fe458bb0824876050399318e63dff787111f913605388d44c',
        bytes=209173414, kind='mlp', res=4, init_seed=42, n_params=17430599,
        gcs=GCS_BUCKET + '/tpu-v6e-spot-eu-a/abl_mlp/final_state.msgpack'),
    # capacity ladder (campaign6 originals; width/blocks/n_params from the
    # reproA receipts == campaign6/train/config6.py specs)
    'mlp_cap_1p1M_b4': dict(
        sha='e4e0b58abc87ad58bad78246cc3ec689c184cb01cf1c4c0c69aee46436b47343',
        bytes=13401469, kind='mlp_cap', width=356, blocks=4, init_seed=42,
        n_params=1116487, cap='1p1M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_1p1M_b4/'
                         'final_state.msgpack'),
    'mlp_cap_1p1M_b8': dict(
        sha='630d1bfadc2653991e50bf056a01f5aafb7938f083b55612baa1dfb85e9ecb7c',
        bytes=13370245, kind='mlp_cap', width=256, blocks=8, init_seed=42,
        n_params=1113671, cap='1p1M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_1p1M_b8/'
                         'final_state.msgpack'),
    'mlp_cap_4p4M_b4': dict(
        sha='db2486c554df0e9fc95dc3ee6b247e11b6eb9671e3fa59443757c6791ad3da3c',
        bytes=52652626, kind='mlp_cap', width=712, blocks=4, init_seed=42,
        n_params=4387415, cap='4p4M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_4p4M_b4/'
                         'final_state.msgpack'),
    'mlp_cap_4p4M_b8': dict(
        sha='74cc1bffe85240483116b430c05b531f9f5aecaa20153a2f1cf5c6ce5884a231',
        bytes=52685722, kind='mlp_cap', width=512, blocks=8, init_seed=42,
        n_params=4389959, cap='4p4M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_4p4M_b8/'
                         'final_state.msgpack'),
    'mlp_cap_17M_b4': dict(
        sha='b4d5540fdadc514dc1bf1083248f56f0fb8fdea5be630b421178ea6baedbf100',
        bytes=209301370, kind='mlp_cap', width=1426, blocks=4, init_seed=42,
        n_params=17441477, cap='17M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_17M_b4/'
                         'final_state.msgpack'),
    'mlp_cap_17M_b8': dict(
        sha='60fe5002cd6de8ec958d199902be9a28d964c8c12b046fb26381b6a6b1bf1c78',
        bytes=209173414, kind='mlp_cap', width=1024, blocks=8, init_seed=42,
        n_params=17430599, cap='17M',
        gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a/mlp_cap_17M_b8/'
                         'final_state.msgpack'),
}
LADDER = ('mlp_cap_1p1M_b4', 'mlp_cap_1p1M_b8', 'mlp_cap_4p4M_b4',
          'mlp_cap_4p4M_b8', 'mlp_cap_17M_b4', 'mlp_cap_17M_b8')
PROD_TRIPLET = ('prod_thermal_random', 'abl_gram_s43', 'abl_gram_s44')
REPROA_RUN = 'reproA-20260725'
# reproA duplicate pins (receipts campaign11/results/reproA/T-<m>_result.json;
# authoring-time finding: byte-identical to the originals, so the pin sha is
# the SAME hex — the stage still hashes the duplicate bytes independently)
REPROA_DUPS = {m: dict(sha=CKPTS[m]['sha'], bytes=CKPTS[m]['bytes'],
                       gcs=GCS_BUCKET + '/tpu-v5e-spot-us-a-reproA/%s/'
                           'final_state.msgpack' % m)
               for m in LADDER}

# --- registered anchors (reported cross-checks + identity gates)
TAB3_PROD_MED = 0.0037696765212937326   # c6_capacity_eval + rg_h9 G0 anchor
TAB3_CAP_MED = 0.004104421587868957
TAB3_ABL_MED = 0.025662127237181198
TAB3_R_ARCH = 1.088799414136559
RH_C3_SEED1007 = {'mlp_cap_17M_b8': 0.0016662067634666004,   # rh_c3.json
                  'abl_mlp': 0.0133492084192001}             # per_model
RG_H9_D0_MED = 0.001016843457679022     # rg_h9 curve[delta=0] (rows 0..512)
MED_IDENTITY_RTOL = 1e-9                # rh_c3 pin (pure-f64 rescore)
XHOST_MED_RTOL = 5e-3                   # fresh-feature cross-host band: the
# f32 eigh pipeline is host-class dependent at the ~1e-5 relative level
# (rh_c3 G4 note + rg_h6 G5 precedent); 5e-3 is far above that jitter and far
# below every effect scale — a violation means a convention/model error.


# ------------------------------------------------------------------ utilities
def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _arr_sha(a):
    return _sha256_bytes(np.ascontiguousarray(a).tobytes())


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _first_existing(cands, what):
    for c in cands:
        c = os.path.expanduser(c)
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError('%s not found; looked in:\n  %s'
                            % (what, '\n  '.join(c for c in cands if c)))


def _resolve(rel):
    """Registered-input resolution: EXPC1_INPUTS (basename), repo, ~/c11,
    ~, ~/workspace at the repo-relative path (rh_c3._resolve order)."""
    inp = os.environ.get('EXPC1_INPUTS', '')
    cands = []
    if inp:
        cands.append(os.path.join(os.path.expanduser(inp),
                                  os.path.basename(rel)))
    cands += [os.path.join(_REPO_ROOT, rel),
              os.path.expanduser('~/c11/' + rel),
              os.path.expanduser('~/' + rel),
              os.path.expanduser('~/workspace/' + rel)]
    return _first_existing(cands, rel)


def sweep50():
    return np.geomspace(*SWEEP50_SPEC)     # campaign4/compute.py:544


# ------------------------------------------- ported registered pure functions
def err_shifted(G_pred, G_true, norm_g):
    """Diag-mean-gauge rel Frobenius error — VERBATIM
    campaign4/engine_parts/p5_shots.py:170-175 (_shots_shifted_error);
    identical to err_q of c10_h0_sensitivity.py:213-219 (rh_c3 err_pair)."""
    shift = np.mean(np.diag(G_true)) - np.mean(np.diag(G_pred))
    G_shifted = G_pred.copy()
    np.fill_diagonal(G_shifted, np.diag(G_shifted) + shift)
    return float(np.linalg.norm(G_shifted - G_true) / norm_g)


def err_pair(Ghat, G_true):
    """(err_q, err_f) — VERBATIM c10_h0_sensitivity.py:213-219 via
    rh_c3.py:379-387 (ARCH-A1 scoring)."""
    nrm = np.linalg.norm(G_true)
    s = np.mean(np.diag(G_true)) - np.mean(np.diag(Ghat))
    Gs = Ghat.copy()
    np.fill_diagonal(Gs, np.diag(Gs) + s)
    return (float(np.linalg.norm(Gs - G_true) / nrm),
            float(np.linalg.norm(Ghat - G_true) / nrm))


def draw_gaussian(p_true, N_shots, rng):
    """Gaussian population-noise drawer — VERBATIM
    campaign4/engine_parts/p5_shots.py:202-216 (_shots_draw_gaussian);
    bit-identity against the engine copy is gated at runtime."""
    noise = (rng.standard_normal(len(p_true))
             * np.sqrt(p_true * (1.0 - p_true) / N_shots))
    p_raw = p_true + noise
    pseudo_counts = np.maximum(p_raw * N_shots, 0.0)
    sum_counts = np.sum(pseudo_counts)
    p_meas = pseudo_counts / sum_counts if sum_counts > 0 else p_true.copy()
    return p_meas


def panel_matrices(vec55, mp=MP):
    """55 triu labels -> symmetric (B,10,10) float64 G (GGenerator _symmetric
    k=0 branch; VERBATIM rh_c3.py:390-400 _panel_matrices)."""
    v = np.asarray(vec55, dtype=np.float64)
    r, c = np.triu_indices(mp)
    G = np.zeros((len(v), mp, mp))
    G[:, r, c] = v
    G = G + np.swapaxes(G, 1, 2)
    d = np.arange(mp)
    G[:, d, d] *= 0.5
    return G


def score_pred_panel(g_pred, G_true_panel):
    """Float64 per-sample (err_q, err_f) — rh_c3.py:403-412."""
    Gp = np.asarray(g_pred, np.float64)
    n = len(Gp)
    eq = np.empty(n)
    ef = np.empty(n)
    for j in range(n):
        eq[j], ef[j] = err_pair(Gp[j], G_true_panel[j])
    return eq, ef


def boot_median_ci(x, seed=BOOT_SEED, B=BOOT_B):
    """Per-Hamiltonian-cluster bootstrap of the median over a 1-D per-sample
    array; order-statistic interval 25/975 (rh_c3.py:415-426 pins)."""
    x = np.asarray(x, np.float64)
    n = x.size
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        reps[b] = np.median(x[rng.integers(0, n, n)])
    reps.sort()
    return [float(reps[BOOT_LO]), float(reps[BOOT_HI])]


def boot_median_ci_cluster(err_hd, seed=BOOT_SEED, B=BOOT_B):
    """Cluster bootstrap over Hamiltonians for a (n_H, n_draws) cell block:
    resample H indices, keep every draw of a resampled H (FINAL_PLAN §3
    rule 5: per-Hamiltonian cluster bootstrap, B=1000, rh_c3 pins)."""
    e = np.asarray(err_hd, np.float64)
    n_h = e.shape[0]
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_h, n_h)
        reps[b] = np.median(e[idx].ravel())
    reps.sort()
    return [float(reps[BOOT_LO]), float(reps[BOOT_HI])]


def boot_alpha_ci(err_lo, err_hi, log_n_ratio, seed=BOOT_SEED, B=BOOT_B):
    """Draw-paired per-Hamiltonian cluster bootstrap of
    alpha = ln(med(err_lo)/med(err_hi)) / ln(N_hi/N_lo): ONE shared index draw
    per replicate applied to both per-sample arrays (rh_c3 paired-ratio pin,
    rh_c3.py:429-446), then the 25/975 order statistics."""
    a = np.asarray(err_lo, np.float64)
    b_arr = np.asarray(err_hi, np.float64)
    if a.shape != b_arr.shape:
        raise ValueError('paired alpha bootstrap needs equal-length arrays')
    n = a.size
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        reps[b] = (math.log(np.median(a[idx]) / np.median(b_arr[idx]))
                   / log_n_ratio)
    reps.sort()
    return [float(reps[BOOT_LO]), float(reps[BOOT_HI])]


def alpha_hat(med_lo, med_hi, n_lo, n_hi):
    return float(math.log(med_lo / med_hi) / math.log(n_hi / n_lo))


def seed_spread_stats(medians):
    """Per-budget 3-seed statistics + FINAL_PLAN threshold arithmetic.
    SE convention: sd(ddof=1)/sqrt(n) of the per-seed medians; relative SE is
    quoted against the 3-seed mean (the gates are %-improvement thresholds).
    Thresholds: G3 / NOISE-claim = max(10%, 2xSE) (R5); G-NINJ = max(5%,
    2xSE) (R16)."""
    m = np.asarray(medians, np.float64)
    mean = float(np.mean(m))
    sd = float(np.std(m, ddof=1)) if m.size > 1 else float('nan')
    se = sd / math.sqrt(m.size) if m.size > 1 else float('nan')
    se_rel_pct = 100.0 * se / mean if mean > 0 else float('nan')
    return dict(per_seed_medians=[float(v) for v in m],
                mean=mean, spread_abs=float(np.max(m) - np.min(m)),
                spread_rel_pct=100.0 * float(np.max(m) - np.min(m)) / mean,
                sd=sd, se=se, se_rel_pct=se_rel_pct,
                thr_G3_NOISEclaim_pct=max(10.0, 2.0 * se_rel_pct),
                thr_GNINJ_pct=max(5.0, 2.0 * se_rel_pct))


def sigma_run_pair(med_orig, med_dup, bit_identical):
    """Run-noise scale from ONE original-vs-duplicate pair: |Delta|/sqrt(2)
    (a two-run range estimates sqrt(2)*sigma); exactly 0 for bit-identical
    checkpoint bytes (deterministic retrain — the reproA finding)."""
    if bit_identical:
        return 0.0
    return abs(float(med_orig) - float(med_dup)) / math.sqrt(2.0)


def combine_alpha_ci(a_hat, eval_ci, sig_rel_lo, sig_rel_hi, log_n_ratio):
    """alpha-hat +/- (eval CI (+) run-noise): the run-noise term
    d_alpha = sqrt(srel_lo^2 + srel_hi^2)/ln(N_hi/N_lo) is added in
    quadrature to each eval-CI half-width at the 1.96-sigma scale
    (FINAL_PLAN R6: 'alpha is quoted as alpha-hat +/- (eval CI (+)
    run-noise)')."""
    d_alpha = math.sqrt(sig_rel_lo ** 2 + sig_rel_hi ** 2) / log_n_ratio
    h_lo = a_hat - eval_ci[0]
    h_hi = eval_ci[1] - a_hat
    lo = a_hat - math.sqrt(h_lo ** 2 + (1.96 * d_alpha) ** 2)
    hi = a_hat + math.sqrt(h_hi ** 2 + (1.96 * d_alpha) ** 2)
    return [float(lo), float(hi)], float(d_alpha)


def ga1_branch(ci_lo, ci_hi):
    """G-A1 three-branch decision (FINAL_PLAN R6 / gate table §6)."""
    if ci_lo > GA1_HIGH:
        return ('authorize', 'alpha CI lower bound %.4f > %.2f -> authorize '
                'ARCH-A2 (res=2 this window, res=4 next)' % (ci_lo, GA1_HIGH))
    if ci_hi < GA1_LOW:
        return ('close', 'alpha CI upper bound %.4f < %.2f -> capacity '
                'closed; zero further architecture TPU' % (ci_hi, GA1_LOW))
    return ('straddle_no_arch_tpu_this_window',
            'alpha CI [%.4f, %.4f] neither clears %.2f from below nor %.2f '
            'from above -> default branch: no architecture TPU this window; '
            're-measure next window (R6 straddle ruling; the wave-2/3 '
            'alternate simply skips ARCH-A2a)' % (ci_lo, ci_hi,
                                                  GA1_LOW, GA1_HIGH))


# ---------------------------------------------------- CAL-1 pool worker (spawn)
_POOL_STATE = {}


def _cal1_pool_init(ops_path):
    _POOL_STATE['ops'] = np.load(ops_path)


def cal1_features_for_H(task):
    """Per-Hamiltonian noise-feature block for every requested budget node —
    numpy-only (spawn-safe; no jax/engine in workers).
    task = (si, V(252,252)f64, E(252,)f64, p_true(252,)f64, nodes, ns_values,
            n_draws).  Draw stream SeedSequence([0, 2, node, si, t]) —
    p5_shots.py:384-385 / c10_tableiii_cpu64.py:361-363 verbatim; feature
    contraction = compute_rho_m minus the f32 cast
    (c10_tableiii_cpu64.py:349-351)."""
    si, V, E, p_true, nodes, ns_values, n_draws = task
    ops = _POOL_STATE['ops']              # (100, 252, 252) float64
    m = MP
    bx = np.empty((len(nodes), n_draws, m, m), np.float64)
    be = np.empty((len(nodes), n_draws), np.float64)
    for ni, (node, n_s) in enumerate(zip(nodes, ns_values)):
        for t in range(n_draws):
            rng = np.random.default_rng(np.random.SeedSequence(
                [SHOTS_SEED0, SHOTS_STREAM_GAUSSIAN, int(node),
                 int(si), int(t)]))
            p_meas = draw_gaussian(p_true, float(n_s), rng)
            rho_meas = (V * p_meas) @ V.T
            bx[ni, t] = np.einsum('ji,kij->k', rho_meas, ops,
                                  optimize=True).reshape(m, m)
            be[ni, t] = float(np.sum(p_meas * E))
    return si, bx, be


# ----------------------------------------------------------- engine (VM-ONLY)
def install_fmb_policy():
    """repro_audit/stages/repro_train.py:164-181 pattern: campaign10's
    attested no-fork serial policy BEFORE the engine can import/use
    fermionic_mbody.  Recorded, warn-not-fail (repro_train semantics)."""
    status = {'installed': False, 'source': None, 'error': None}
    for cand in (os.path.expanduser('~/c11/campaign10'),
                 os.path.join(_REPO_ROOT, 'campaign10'),
                 os.path.expanduser('~/campaign10')):
        if os.path.isfile(os.path.join(cand, 'fmb_serial10.py')):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import fmb_serial10
                fmb_serial10.install_fmb_serial_policy()
                status.update(installed=True, source=cand)
                print('%s fmb serial (no-fork) policy installed from %s'
                      % (TAG, cand), flush=True)
            except Exception as exc:                          # noqa: BLE001
                status['error'] = repr(exc)
                print('%s WARNING: fmb serial policy not installed: %r'
                      % (TAG, exc), flush=True)
            break
    else:
        status['error'] = 'fmb_serial10.py not found'
        print('%s WARNING: fmb_serial10.py not found (repro_train pattern: '
              'recorded, not fatal)' % TAG, flush=True)
    return status


def load_engine6(want_x64):
    """engine6 (campaign4 engine + campaign5 zoo + p2_models6 mlp_cap +
    p5_shots) via the rh_c3.load_engine6 sibling-layout contract
    (rh_c3.py:459-523).  want_x64: set jax_enable_x64 BEFORE any jax op
    (STAT-04 s9_xhost pattern, c10_tableiii_cpu64.py:118-121)."""
    try:
        import jax
    except Exception as exc:                                  # noqa: BLE001
        sys.exit('%s VM-ONLY: evaluation needs jax/flax + engine6 + the '
                 'checkpoint mirror; not available here (%r). Use --selftest '
                 'locally.' % (TAG, exc))
    if want_x64:
        jax.config.update('jax_enable_x64', True)
    import tempfile
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(
        tempfile.gettempdir(), 'numba_cache_expc1_%d' % os.getpid()))
    fmb_policy = install_fmb_policy()
    eng6_py = _first_existing(
        [os.path.join(_REPO_ROOT, 'campaign6', 'train', 'engine6.py'),
         '~/c11/campaign6/train/engine6.py',
         '~/campaign6/train/engine6.py',
         '~/workspace/campaign6/train/engine6.py'],
        'campaign6/train/engine6.py')
    c6t = os.path.dirname(eng6_py)
    root = os.path.abspath(os.path.join(c6t, '..', '..'))
    for sib in ('campaign4/engine_parts/p1_core.py',
                'campaign4/engine_parts/p5_shots.py',
                'campaign5/train/engine5_parts/p2_models5.py',
                'campaign6/train/engine6_parts/p2_models6.py'):
        if not os.path.isfile(os.path.join(root, sib)):
            sys.exit('%s engine6 sibling layout incomplete: missing %s under '
                     '%s' % (TAG, sib, root))
    for p in (c6t, os.path.join(root, 'campaign5', 'train'),
              os.path.join(root, 'campaign4')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import engine6 as eng
    eng.init_d20(h_type='random', state_type='thermal', beta=BETA_THERMAL,
                 g_init=G_INIT, g_stop=G_STOP)
    assert int(eng.M_PAIRS) == MP and int(eng.g_gen.label_size()) == LABEL55
    assert float(eng.BETA) == BETA_THERMAL
    return eng, fmb_policy, root


def verify_and_restore(path, pin_sha, pin_bytes):
    """Byte verification BEFORE deserialization (campaign10
    checkpoint_authority contract, self-contained): hash the exact bytes,
    gate against the pin, only then msgpack_restore those bytes."""
    from flax import serialization as flax_ser
    with open(path, 'rb') as f:
        data = f.read()
    sha = _sha256_bytes(data)
    row = dict(path=os.path.abspath(path), sha256=sha, bytes=len(data),
               pin_sha256=pin_sha, pin_bytes=pin_bytes,
               match=bool(sha == pin_sha and len(data) == int(pin_bytes)))
    if not row['match']:
        return None, row
    return flax_ser.msgpack_restore(data), row


def build_state(eng, name, spec, raw, config01_stats):
    """Model template + parameter restore.  OGN topology per
    c10_tableiii_cpu64.py:280-289; mlp/mlp_cap per rh_c3.py:546-591;
    set_mlp_cap_config per p2_models6.py:57-75 (called immediately before
    every mlp_cap build so config never leaks across widths)."""
    import jax
    import jax.numpy as jnp
    from flax import serialization as flax_ser
    kind = spec['kind']
    eng.set_ogn_energy_input(True)
    if kind == 'ogn':
        model = eng.build_model('ogn', LABEL55, int(spec['res']), True,
                                use_scatter=True, use_reinject=True,
                                use_orb_emb=True, readout_bias=0.55,
                                use_energy_input=True)
    elif kind == 'mlp':
        model = eng.build_model('mlp', LABEL55, int(spec['res']), True)
    elif kind == 'mlp_cap':
        eng.set_mlp_cap_config(int(spec['width']), int(spec['blocks']),
                               feat_mean=config01_stats['feat_mean'],
                               feat_std=config01_stats['feat_std'],
                               standardize=True)
        model = eng.build_model('mlp_cap', LABEL55, 4, True)
    else:
        raise ValueError(kind)
    variables = model.init(jax.random.PRNGKey(int(spec.get('init_seed', 42))),
                           jnp.zeros((1, MP, MP, 1), jnp.float32),
                           jnp.zeros((1, 1), jnp.float32), training=False)
    raw_params = raw.get('params', raw) if isinstance(raw, dict) else raw
    params = flax_ser.from_state_dict(variables['params'], raw_params)
    bs_t = variables.get('batch_stats', {})
    raw_bs = raw.get('batch_stats', {}) if isinstance(raw, dict) else {}
    batch_stats = flax_ser.from_state_dict(bs_t, raw_bs) if raw_bs else bs_t
    n_par = int(sum(np.asarray(x).size
                    for x in jax.tree_util.tree_leaves(params)))
    if n_par != int(spec['n_params']):
        raise RuntimeError('%s: restored n_params %d != registered %d'
                           % (name, n_par, spec['n_params']))
    return model, params, batch_stats, n_par


def make_apply(model, params, batch_stats):
    """Jitted forward == p3_training.py:263-273 eval_step semantics
    (c10_tableiii_cpu64.py:384-395 make_apply lineage: first call inside the
    matmul-precision context bakes the precision in at trace time)."""
    import jax
    @jax.jit
    def _apply(bx, be):
        return model.apply({'params': params, 'batch_stats': batch_stats},
                           bx, be, training=False)
    return _apply


def load_config01_stats(paths):
    """CONFIG-01 exact bytes (sha-gated in the inputs gate) -> the 55-dim
    feat_mean/feat_std; concat digest must equal the campaign10
    config_authority10 ANCHORS stats_sha256; equality across the six ladder
    ckpt_config.json copies is gated separately (ARCH-A1 'verify CONFIG-01
    standardization vectors across widths')."""
    with open(paths['config01_config']) as f:
        cfg = json.load(f)
    mc = cfg['mlp_cap']
    feat_mean = [float(v) for v in mc['feat_mean']]
    feat_std = [float(v) for v in mc['feat_std']]
    digest = _sha256_bytes(np.asarray(feat_mean + feat_std,
                                      np.float64).astype('<f8').tobytes())
    return dict(feat_mean=feat_mean, feat_std=feat_std,
                stats_sha256=digest,
                stats_sha_match=bool(digest == CONFIG01_STATS_SHA),
                n_stat_samples=int(mc.get('n_stat_samples', -1)))


def resolve_inputs(keys):
    """Resolve + byte-pin the registered inputs; returns (paths, gate)."""
    rows, paths, ok = {}, {}, True
    for key in keys:
        if key == 'expected_published':
            try:
                p = _first_existing(
                    [EXPECTED_PUBLISHED_CANDS[0],
                     os.path.join(_REPO_ROOT, EXPECTED_PUBLISHED_CANDS[1])],
                    'expected_published.json')
                sha = _sha256_file(p)
                m = bool(sha == EXPECTED_PUBLISHED_SHA)
                rows[key] = dict(path=p, sha256=sha,
                                 pin=EXPECTED_PUBLISHED_SHA, match=m)
                paths[key] = p
                ok = ok and m
            except FileNotFoundError as exc:
                rows[key] = dict(path=None, error=str(exc), match=False)
                ok = False
            continue
        rel, pin = REC[key]
        try:
            if key == 'val_npz':
                try:
                    p = _resolve(rel)
                    variant = 'val_npz'
                except FileNotFoundError:
                    p = _resolve(REC['val_first4096'][0])
                    variant = 'val_first4096'
                    pin = REC['val_first4096'][1]
                paths['val_npz'] = p
                paths['val_npz_variant'] = variant
            else:
                p = _resolve(rel)
                paths[key] = p
            sha = _sha256_file(p)
            m = bool(sha == pin)
            rows[key] = dict(path=p, sha256=sha, pin=pin, match=m)
            ok = ok and m
        except FileNotFoundError as exc:
            rows[key] = dict(path=None, error=str(exc), match=False)
            ok = False
    gate = dict(rows=rows, gate_pass=bool(ok),
                rationale='every registered input byte-pinned to its '
                          'authoring-time sha256 (2026-07-29, read-only '
                          'trees); mismatch = staging/protocol error '
                          '(hard fail)')
    return paths, gate


def expected_published_check(paths, digest_rows):
    """Cross-check loaded checkpoint digests vs expected_published.json —
    recorded match/mismatch/absent; the GATE decides on the recorded pins."""
    out = {}
    try:
        with open(paths['expected_published']) as f:
            exp = json.load(f)
    except Exception as exc:                                  # noqa: BLE001
        return {'error': repr(exc)}
    for name, row in digest_rows.items():
        base = name.split('@')[0]
        e = exp.get(base)
        if e is None:
            out[name] = 'absent_from_expected_published'
        else:
            out[name] = ('match' if e.get('expected_sha256') == row['sha256']
                         else 'MISMATCH(expected %s, loaded %s)'
                         % (e.get('expected_sha256'), row['sha256']))
    return out


def env_block(nproc):
    try:
        import jax
        jv, backend = jax.__version__, jax.default_backend()
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:                                         # noqa: BLE001
        jv, backend, x64 = None, None, None
    try:
        import scipy
        sv = scipy.__version__
    except Exception:                                         # noqa: BLE001
        sv = None
    try:
        import flax
        fv = flax.__version__
    except Exception:                                         # noqa: BLE001
        fv = None
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=sv, jax=jv, flax=fv, backend=backend,
                jax_enable_x64=x64,
                jax_platforms=os.environ.get('JAX_PLATFORMS'),
                hostname=platform.node(), platform=platform.platform(),
                nproc=nproc)


def write_payload(out_dir, part, payload):
    p = os.path.join(out_dir, 'expc1_part_%s.json' % part)
    with open(p, 'w') as f:
        json.dump(_jsonable(payload), f, indent=1)
        f.write('\n')
    print('%s wrote %s' % (TAG, p), flush=True)
    return p


# ================================================================ part: CAL-1
def run_cal1(out_dir, ckpt_root, nproc, smoke):
    t0 = time.time()
    n_h = N_H_CAL_SMOKE if smoke else N_H_CAL
    n_d = N_DRAWS_CAL_SMOKE if smoke else N_DRAWS_CAL
    nodes = (NODES6[0], NODES6[-1]) if smoke else NODES6
    gates = {}
    paths, g_in = resolve_inputs(
        ['val_npz', 'expected_published', 'config01_config'])
    gates['C0_registered_inputs'] = g_in
    if not g_in['gate_pass']:
        payload = dict(part='cal1', gates=gates, smoke=smoke,
                       fatal='registered inputs failed',
                       environment=env_block(nproc), wall_s=time.time() - t0)
        write_payload(out_dir, 'cal1', payload)
        return False

    sweep = sweep50()
    ns_values = [float(sweep[i]) for i in nodes]
    grid_ok = bool(np.allclose([float(sweep[i]) for i in NODES6],
                               NODES6_NS_PIN, rtol=1e-12))
    gates['C1_budget_grid_pin'] = dict(
        nodes_sweep_indices=list(nodes), n_s=ns_values,
        pinned_full_grid=list(NODES6_NS_PIN), gate_pass=grid_ok,
        rationale='budget grid = SWEEP50[NODES6] (compute.py:544, '
                  'c10_tableiii_cpu64.py:126), the Table-III nodes')

    zv = np.load(paths['val_npz'])
    labels_full = np.asarray(zv['g_true'], np.float32)     # seed-1007 dev
    labels = np.ascontiguousarray(labels_full[:n_h])
    Gpanel = panel_matrices(labels)
    norm_g = np.linalg.norm(Gpanel, axis=(1, 2))

    # engine (x64 ON before any jax op — STAT-04 convention)
    eng, fmb_policy, code_root = load_engine6(want_x64=True)
    import jax
    import jax.numpy as jnp
    import scipy.linalg
    x64_on = bool(jax.config.jax_enable_x64)
    gates['C2_x64_active'] = dict(
        jax_enable_x64=x64_on,
        default_float=str(jnp.zeros(1).dtype),
        gate_pass=bool(x64_on and jnp.zeros(1).dtype == jnp.float64),
        rationale='CPU-f64 convention (STAT-04 cpu64 branch) requires '
                  'jax_enable_x64 before any jax array op')

    # label-panel identity: pure-numpy panel == engine GGenerator reconstruct
    g_rec = np.asarray(eng.g_gen.reconstruct(jnp.asarray(labels[:4])),
                       np.float64)
    panel_id = bool(np.array_equal(g_rec, Gpanel[:4]))
    diag_dev = float(np.max(np.abs(
        Gpanel[:, np.arange(MP), np.arange(MP)].mean(axis=1)
        - DIAG_GAUGE_MEAN)))
    gates['C3_panel_identity_gauge'] = dict(
        reconstruct_equal=panel_id, diag_gauge_max_dev=diag_dev,
        diag_gauge_tol=DIAG_GAUGE_TOL,
        gate_pass=bool(panel_id and diag_dev < DIAG_GAUGE_TOL),
        rationale='numpy triu panel must equal the engine GGenerator '
                  'reconstruction exactly (rh_c3 precedent) and carry the '
                  '0.55 diagonal gauge (c10 GP0 tolerance)')

    # drawer identity: ported drawer bit-equals the engine copy
    drawer_ok = True
    ptest = np.abs(np.random.default_rng(1).standard_normal(252))
    ptest /= ptest.sum()
    for probe_ns in (1e10, 9102.981779915226, 1e2):
        r1 = np.random.default_rng(np.random.SeedSequence([9, 9, 9]))
        r2 = np.random.default_rng(np.random.SeedSequence([9, 9, 9]))
        a = draw_gaussian(ptest, probe_ns, r1)
        b = eng._shots_draw_gaussian(ptest, probe_ns, r2)
        drawer_ok = drawer_ok and bool(np.array_equal(a, b))
    gates['C4_drawer_port_identity'] = dict(
        gate_pass=bool(drawer_ok),
        rationale='ported draw_gaussian must be bit-identical to the engine '
                  '_shots_draw_gaussian (p5_shots.py:202-216) on identical '
                  'SeedSequence streams — port, never reinvent')

    # thermal systems from the dev-1007 labels (prepare_heldout_systems port,
    # p5_shots.py:268-313, WLS rows skipped; G source = dev labels)
    print('%s building %d dev-1007 thermal systems (eigh 252x252 f64)...'
          % (TAG, n_h), flush=True)
    D_N = int(eng.basis.size)
    V_all = np.empty((n_h, D_N, D_N), np.float64)
    E_all = np.empty((n_h, D_N), np.float64)
    p_all = np.empty((n_h, D_N), np.float64)
    for i in range(n_h):
        H = eng._shots_h_dense64(Gpanel[i])
        E_true, V_true = scipy.linalg.eigh(H)
        p_true = np.exp(-BETA_THERMAL * (E_true - E_true.min()))
        p_true = (p_true / np.sum(p_true)).astype(np.float64)
        V_all[i], E_all[i], p_all[i] = V_true, E_true, p_true

    # pair-block operator tensor (float64, compute_rho_m contraction source)
    rho2_dense = eng._safe_dense(eng.rho_2_kkbar_arrays)
    ops_flat = np.asarray(rho2_dense, np.float64).reshape(
        -1, rho2_dense.shape[-1], rho2_dense.shape[-1])
    ops_path = os.path.join(out_dir, '_expc1_ops_flat.npy')
    np.save(ops_path, ops_flat)

    # noise features: 6 budgets x n_h x n_d (spawn pool, numpy-only workers)
    print('%s drawing noise grid: %d budgets x %d H x %d draws (nproc=%d)...'
          % (TAG, len(nodes), n_h, n_d, nproc), flush=True)
    tasks = [(si, V_all[si], E_all[si], p_all[si], list(nodes), ns_values,
              n_d) for si in range(n_h)]
    BX = np.empty((len(nodes), n_h, n_d, MP, MP), np.float64)
    BE = np.empty((len(nodes), n_h, n_d), np.float64)
    if nproc > 1:
        import multiprocessing as mp_mod
        ctx = mp_mod.get_context('spawn')     # NEVER fork the jax process
        with ctx.Pool(processes=nproc, initializer=_cal1_pool_init,
                      initargs=(ops_path,)) as pool:
            for si, bx, be in pool.imap_unordered(cal1_features_for_H, tasks):
                BX[:, si], BE[:, si] = bx, be
    else:
        _cal1_pool_init(ops_path)
        for task in tasks:
            si, bx, be = cal1_features_for_H(task)
            BX[:, si], BE[:, si] = bx, be
    # pool determinism: recompute H0 serially, must be byte-identical
    _cal1_pool_init(ops_path)
    si0, bx0, be0 = cal1_features_for_H(tasks[0])
    feat_det = bool(np.array_equal(bx0, BX[:, 0])
                    and np.array_equal(be0, BE[:, 0]))
    try:
        os.remove(ops_path)
    except OSError:
        pass

    # checkpoints: registered prod triplet, byte-verified before restore
    config01_stats = load_config01_stats(paths)
    digest_rows, seed_states = {}, {}
    g1_ok = True
    for name in PROD_TRIPLET:
        spec = CKPTS[name]
        path = os.path.join(ckpt_root, name, 'final_state.msgpack')
        raw, row = verify_and_restore(path, spec['sha'], spec['bytes'])
        digest_rows[name] = row
        if raw is None:
            g1_ok = False
            continue
        model, params, bstats, n_par = build_state(eng, name, spec, raw,
                                                   config01_stats)
        # f64 params + highest (STAT-04 cpu64 branch)
        params64 = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, jnp.float64), params)
        bstats64 = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, jnp.float64), bstats)
        seed_states[name] = (make_apply(model, params64, bstats64), n_par)
        print('%s   loaded %s (%s...) n_params=%d'
              % (TAG, name, row['sha256'][:16], n_par), flush=True)
    exp_check = expected_published_check(paths, digest_rows)
    gates['G1_checkpoints_loaded_digests'] = dict(
        rows=digest_rows, expected_published_crosscheck=exp_check,
        gate_pass=bool(g1_ok and len(seed_states) == len(PROD_TRIPLET)),
        rationale='every requested checkpoint byte-verified (sha256+bytes) '
                  'against the r20 PRV manifest pins BEFORE deserialization; '
                  'expected_published.json comparison recorded '
                  '(match/mismatch/absent), gate decides on the recorded '
                  'pins')
    if not gates['G1_checkpoints_loaded_digests']['gate_pass']:
        payload = dict(part='cal1', gates=gates, smoke=smoke,
                       fatal='checkpoint verification failed',
                       environment=env_block(nproc), wall_s=time.time() - t0)
        write_payload(out_dir, 'cal1', payload)
        return False

    # forward passes (f64 features, f64 params, matmul highest, f64 scoring)
    n_cells = len(nodes) * n_h * n_d
    bx_flat = BX.reshape(n_cells, MP, MP)[..., None]
    be_flat = BE.reshape(n_cells, 1)
    err = {}
    logits_store = {}
    det_ok = True
    for name in PROD_TRIPLET:
        apply_f, _np_ = seed_states[name]
        outs = []
        with jax.default_matmul_precision('highest'):
            for s in range(0, n_cells, EVAL_CHUNK_F64):
                e = min(s + EVAL_CHUNK_F64, n_cells)
                logits = np.asarray(apply_f(
                    jnp.asarray(bx_flat[s:e], jnp.float64),
                    jnp.asarray(be_flat[s:e], jnp.float64)))
                if logits.ndim == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                outs.append(logits)
            # determinism: repeat first chunk
            l0 = np.asarray(apply_f(
                jnp.asarray(bx_flat[:min(EVAL_CHUNK_F64, n_cells)],
                            jnp.float64),
                jnp.asarray(be_flat[:min(EVAL_CHUNK_F64, n_cells)],
                            jnp.float64)))
            if l0.ndim == 3:
                l0 = l0.reshape(-1, l0.shape[-1])
            det_ok = det_ok and bool(np.array_equal(l0, outs[0]))
        logits_all = np.concatenate(outs, 0)
        G_hat = np.asarray(eng.g_gen.reconstruct(jnp.asarray(logits_all)),
                           np.float64)
        e_arr = np.empty(n_cells, np.float64)
        h_idx = np.repeat(np.tile(np.arange(n_h), len(nodes)), n_d)
        for j in range(n_cells):
            si = int(h_idx[j])
            e_arr[j] = err_shifted(G_hat[j], Gpanel[si], norm_g[si])
        err[name] = e_arr.reshape(len(nodes), n_h, n_d)
        logits_store[name] = logits_all.reshape(len(nodes), n_h, n_d,
                                                LABEL55)
        for ni, node in enumerate(nodes):
            print('%s   %-20s N_s=%.3g median=%.6e'
                  % (TAG, name, ns_values[ni],
                     float(np.median(err[name][ni].ravel()))), flush=True)
    gates['C5_determinism'] = dict(
        pool_feature_block_identical=feat_det,
        repeated_forward_chunk_identical=bool(det_ok),
        gate_pass=bool(feat_det and det_ok),
        rationale='spawn-pool feature block must equal the serial recompute '
                  'byte-for-byte; repeated jitted f64 forward byte-identical '
                  '(single-host determinism, rg_h6 G5 convention)')

    # per-budget statistics + thresholds
    budgets = []
    ses_ok, thr_ok = True, True
    for ni, node in enumerate(nodes):
        med = {name: float(np.median(err[name][ni].ravel()))
               for name in PROD_TRIPLET}
        cis = {name: boot_median_ci_cluster(err[name][ni])
               for name in PROD_TRIPLET}
        st = seed_spread_stats([med[n] for n in PROD_TRIPLET])
        ses_ok = ses_ok and bool(np.isfinite(st['se']) and st['se'] > 0)
        thr_ok = thr_ok and bool(
            np.isfinite(st['thr_G3_NOISEclaim_pct'])
            and st['thr_G3_NOISEclaim_pct'] >= 10.0
            and np.isfinite(st['thr_GNINJ_pct'])
            and st['thr_GNINJ_pct'] >= 5.0)
        budgets.append(dict(
            node=int(node), N_s=ns_values[ni],
            per_seed=dict(
                (name, dict(median=med[name], median_ci95=cis[name],
                            n_H=n_h, n_draws=n_d)) for name in PROD_TRIPLET),
            seed_stats=st))
    finite_ok = all(bool(np.isfinite(err[n]).all()) for n in PROD_TRIPLET)
    gates['C6_finiteness'] = dict(
        n_cells_per_seed=n_cells, all_finite=finite_ok,
        gate_pass=bool(finite_ok),
        rationale='every (budget, H, draw, seed) error finite')
    gates['G2_perbudget_se_finite_registered'] = dict(
        gate_pass=bool(ses_ok and finite_ok),
        n_budgets=len(budgets),
        rationale='all per-budget 3-seed SEs finite and > 0, computed from '
                  'the per-seed medians (sd(ddof=1)/sqrt(3), quoted relative '
                  'to the 3-seed mean) and registered in '
                  'results.budgets[*].seed_stats')
    gates['G3_thresholds_table_emitted'] = dict(
        gate_pass=bool(thr_ok),
        thresholds=[dict(N_s=b['N_s'],
                         thr_G3_NOISEclaim_pct=b['seed_stats']
                         ['thr_G3_NOISEclaim_pct'],
                         thr_GNINJ_pct=b['seed_stats']['thr_GNINJ_pct'])
                    for b in budgets],
        rationale='per-budget thresholds max(10%%, 2xSE) [G3 / NOISE-claim, '
                  'FINAL_PLAN R5] and max(5%%, 2xSE) [G-NINJ, R16] emitted; '
                  'registered BEFORE the TPU window per CAL-1')

    # per-seed prediction arrays for ENS-1 (FINAL_PLAN §7.1.2 sharing)
    npz_path = os.path.join(out_dir, 'expc1_cal1_perseed.npz')
    store = dict(labels_dev1007=labels,
                 G_true_panel=Gpanel,
                 nodes=np.asarray(nodes, np.int64),
                 N_s=np.asarray(ns_values, np.float64),
                 bx64=BX, be64=BE)
    for name in PROD_TRIPLET:
        store['logits55_%s' % name] = logits_store[name]
        store['err_%s' % name] = err[name]
    np.savez_compressed(npz_path, **store)
    print('%s wrote %s' % (TAG, npz_path), flush=True)

    payload = dict(
        part='cal1',
        item='CAL-1 (FINAL_PLAN.md ~:559; power F4)',
        protocol=dict(
            stream='registered seed-1007 dev labels rows 0..%d '
                   '(%s, byte-pinned); systems built exactly as '
                   'p5_shots.prepare_heldout_systems (p5_shots.py:268-313, '
                   'WLS rows skipped) with the G source swapped to the dev '
                   'labels' % (n_h, paths.get('val_npz_variant', 'val_npz')),
            budgets='SWEEP50 indices %s (compute.py:544; Table III nodes)'
                    % (list(nodes),),
            draws='SeedSequence([0, 2, node, si, t]) — registered '
                  'shots_master gaussian stream (p5_shots.py:384-385), '
                  'si = dev-1007 row index; %d H x %d draws' % (n_h, n_d),
            noise_model='gaussian only (p5_shots.py:202-216, ported verbatim '
                        '+ bit-identity gate); multinomial deferred per R5',
            convention='cpu-f64 == STAT-04 cpu64 deliverable branch: float64 '
                       'features (compute_rho_m contraction minus the f32 '
                       'cast), float64 params, jax_enable_x64, matmul '
                       'highest, float64 scoring '
                       '(c10_tableiii_cpu64.py:33-36)',
            legacy_companion='NOT PRODUCED: the legacy convention of the '
                             'printed noise tables is accelerator-bf16/f32 '
                             'at default matmul and cannot be produced on '
                             'CPU; the accelerator-style CPU-f32 proxy was '
                             'ruled not needed for CAL-1 (task ruling). '
                             'CAL-1 SEs are therefore CPU-f64-convention '
                             'measurements; the FINAL_PLAN G3/NOISE-claim '
                             'gates decide under the legacy convention and '
                             'inherit these thresholds per CAL-1.',
            scoring='diag-mean gauge (p5_shots.py:170-175)',
            bootstrap='per-Hamiltonian cluster bootstrap of the median, '
                      'B=%d, default_rng(%d), order stats %d/%d (rh_c3 '
                      'pins; FINAL_PLAN §3 rule 5)'
                      % (BOOT_B, BOOT_SEED, BOOT_LO, BOOT_HI)),
        results=dict(budgets=budgets,
                     conventions=dict(produced=['cpu-f64'],
                                      legacy_companion='not produced '
                                      '(see protocol.legacy_companion)'),
                     perseed_npz=os.path.basename(npz_path),
                     perseed_npz_sha256=_sha256_file(npz_path)),
        checkpoints=digest_rows,
        fmb_rho_policy=fmb_policy,
        gates=gates, smoke=bool(smoke),
        environment=env_block(nproc),
        script_sha256=_sha256_file(os.path.abspath(__file__)),
        wall_s=time.time() - t0)
    write_payload(out_dir, 'cal1', payload)
    return all(bool(g.get('gate_pass')) for g in gates.values())


# ============================================================== part: ARCH-A1
def build_features_float32(eng, labels_f32):
    """Labels -> float32 dataset-pipeline features/energies — VERBATIM
    rh_c3.build_features_float32 (rh_c3.py:594-625; rg_h6.py:612-643;
    p1_core.py:783-855 numerics)."""
    import jax.numpy as jnp
    rho_1_np = eng._ensure_dense(eng.rho_1_arrays)
    if rho_1_np.ndim == 4:
        rho_1_diag_np = np.einsum('kknn->kn', rho_1_np)
    else:
        rho_1_diag_np = np.diagonal(rho_1_np, axis1=1, axis2=2)
    target_np = eng._ensure_dense(eng.rho_2_kkbar_arrays)
    inter_np = target_np                                    # h_type 'random'
    cast = lambda a: jnp.array(np.asarray(a).astype(np.float32))  # noqa: E731
    p_rho_1_full = cast(rho_1_np)
    p_rho_1_diag = cast(rho_1_diag_np)
    p_rho_2_inter = cast(inter_np)
    p_rho_target = cast(target_np)
    base_e = np.asarray(eng.U_ENERGY_SEED[0], np.float32)
    n = len(labels_f32)
    bx = np.empty((n, MP, MP, 1), np.float32)
    be = np.empty((n, 1), np.float32)
    for s in range(0, n, EVAL_CHUNK_F32):
        e = min(s + EVAL_CHUNK_F32, n)
        lab = jnp.array(labels_f32[s:e])
        i_vals = eng.g_gen.reconstruct(lab)
        e_vals = jnp.array(np.tile(base_e, (e - s, 1)))
        _f1, f2, en = eng.solve_batch_kernel(
            e_vals, i_vals, float(eng.BETA), True,
            p_rho_1_diag, p_rho_2_inter, p_rho_target, p_rho_1_full)
        bx[s:e] = np.asarray(f2, np.float32)
        be[s:e] = np.asarray(en, np.float32).reshape(-1, 1)
    return bx, be


def run_archa1(out_dir, ckpt_root, nproc, smoke):
    t0 = time.time()
    n_eval = N_EVAL_ARCH_SMOKE if smoke else N_EVAL_ARCH
    gates = {}
    paths, g_in = resolve_inputs(
        ['val_npz', 'c6_capacity_eval', 'r20_manifest', 'config01_config',
         'rh_c3_json', 'expected_published'])
    # ladder ckpt_config.json copies (stats-across-widths authority)
    cfg_rows, cfg_ok = {}, True
    for m in LADDER:
        rel = 'campaign11/results/reproA/T-%s/ckpt_config.json' % m
        try:
            p = _resolve(rel)
            sha = _sha256_file(p)
            ok = bool(sha == CKPT_CONFIG_PINS[m])
            cfg_rows[m] = dict(path=p, sha256=sha, pin=CKPT_CONFIG_PINS[m],
                               match=ok)
            cfg_ok = cfg_ok and ok
        except FileNotFoundError as exc:
            cfg_rows[m] = dict(path=None, error=str(exc), match=False)
            cfg_ok = False
    g_in['rows']['ladder_ckpt_configs'] = cfg_rows
    g_in['gate_pass'] = bool(g_in['gate_pass'] and cfg_ok)
    gates['A0_registered_inputs'] = g_in
    if not g_in['gate_pass']:
        payload = dict(part='archa1', gates=gates, smoke=smoke,
                       fatal='registered inputs failed',
                       environment=env_block(nproc), wall_s=time.time() - t0)
        write_payload(out_dir, 'archa1', payload)
        return False

    # standardization stats: CONFIG-01 bytes + equality across all widths
    config01_stats = load_config01_stats(paths)
    stats_eq = True
    for m in LADDER:
        with open(cfg_rows[m]['path']) as f:
            mc = json.load(f)['mlp_cap']
        stats_eq = stats_eq and bool(
            [float(v) for v in mc['feat_mean']] == config01_stats['feat_mean']
            and [float(v) for v in mc['feat_std']]
            == config01_stats['feat_std']
            and int(mc['width']) == CKPTS[m]['width']
            and int(mc['num_blocks']) == CKPTS[m]['blocks']
            and bool(mc['standardize']) is True)
    gates['A1_stats_authority_across_widths'] = dict(
        stats_sha256=config01_stats['stats_sha256'],
        matches_config01_anchor=config01_stats['stats_sha_match'],
        identical_across_all_six_widths=bool(stats_eq),
        n_stat_samples=config01_stats['n_stat_samples'],
        gate_pass=bool(config01_stats['stats_sha_match'] and stats_eq),
        rationale='the applied feat_mean/feat_std re-hash to the CONFIG-01 '
                  'stats_sha256 (config_authority10 ANCHORS) and are '
                  'byte-equal across every ladder ckpt_config.json '
                  '(R6: verify CONFIG-01 standardization vectors across '
                  'widths); width/blocks per config match the registered '
                  'specs')

    # dev stream + tab3 anchor identity
    zv = np.load(paths['val_npz'])
    labels_full = np.asarray(zv['g_true'], np.float32)
    n_b = min(n_eval, len(labels_full), N_EVAL_ARCH)
    labels = np.ascontiguousarray(labels_full[:n_b])
    Gpanel = panel_matrices(labels)
    gpred_prod_reg = np.asarray(zv['g_pred'], np.float32)[:n_b]
    eq_reg, _ = score_pred_panel(panel_matrices(gpred_prod_reg), Gpanel)
    anchor = float(np.median(eq_reg))
    full_run = (not smoke) and n_b == N_EVAL_ARCH
    anchor_ok = (not full_run) or bool(
        abs(anchor - TAB3_PROD_MED) <= MED_IDENTITY_RTOL * TAB3_PROD_MED)
    with open(paths['c6_capacity_eval']) as f:
        c6 = json.load(f)
    c6_rows = c6['rows']
    c6_pins_ok = bool(
        float(c6_rows['mlp_cap_17M_b8']['tab3_4096']['rel_param_err_median'])
        == TAB3_CAP_MED
        and float(c6_rows['abl_mlp']['tab3_4096']['rel_param_err_median'])
        == TAB3_ABL_MED
        and float(c6_rows['prod_thermal_random']['tab3_4096']
                  ['rel_param_err_median']) == TAB3_PROD_MED
        and float(c6['ogn_over_best_mlp_gap']) == TAB3_R_ARCH)
    gates['A2_tab3_anchor_identity'] = dict(
        anchor_recompute=anchor, anchor_registered=TAB3_PROD_MED,
        anchor_ok=bool(anchor_ok), c6_pins_ok=c6_pins_ok,
        smoke_truncated=bool(not full_run),
        gate_pass=bool(anchor_ok and c6_pins_ok),
        rationale='pure-f64 rescore of the registered val g_pred rows '
                  '0..%d must reproduce the 3.7697e-3 tab3 anchor within '
                  'rtol %g (rh_c3 G1 identity; enforced at full n only); '
                  'c6_capacity_eval extract must equal the in-script pins'
                  % (n_b, MED_IDENTITY_RTOL))

    # engine (NO x64 — rh_c3 convention) + features
    eng, fmb_policy, code_root = load_engine6(want_x64=False)
    import jax
    import jax.numpy as jnp
    print('%s building float32-pipeline features: seed-1007 n=%d'
          % (TAG, n_b), flush=True)
    BX, BE = build_features_float32(eng, labels)
    bx_r, be_r = build_features_float32(eng, labels[:EVAL_CHUNK_F32])
    feat_det = bool(np.array_equal(bx_r, BX[:EVAL_CHUNK_F32])
                    and np.array_equal(be_r, BE[:EVAL_CHUNK_F32]))

    # checkpoints: originals + reproA duplicates (dedup on byte identity)
    r20_rows = {}
    with open(paths['r20_manifest']) as f:
        r20 = json.load(f)
    for r in (r20.get('results', {}) or {}).get('rows', []):
        r20_rows[r.get('name')] = r
    model_list = list(PROD_TRIPLET) + ['abl_mlp'] + list(LADDER)
    digest_rows, states = {}, {}
    g1_ok = True
    for name in model_list:
        spec = CKPTS[name]
        path = os.path.join(ckpt_root, name, 'final_state.msgpack')
        raw, row = verify_and_restore(path, spec['sha'], spec['bytes'])
        mrow = r20_rows.get(name) or {}
        row['r20_manifest_sha256'] = mrow.get('sha256')
        row['r20_status'] = mrow.get('status')
        row['match'] = bool(row['match']
                            and mrow.get('sha256') == spec['sha']
                            and mrow.get('status') == 'present_hashed')
        digest_rows[name] = row
        if raw is None or not row['match']:
            g1_ok = False
            continue
        model, params, bstats, n_par = build_state(eng, name, spec, raw,
                                                   config01_stats)
        states[name] = make_apply(model, params, bstats)
        print('%s   loaded %s (%s...) n_params=%d'
              % (TAG, name, row['sha256'][:16], n_par), flush=True)
    dup_rows = {}
    for name in LADDER:
        dpath = os.path.join(ckpt_root, REPROA_RUN, name,
                             'final_state.msgpack')
        pin = REPROA_DUPS[name]
        if not os.path.exists(dpath):
            dup_rows[name + '@' + REPROA_RUN] = dict(
                path=dpath, error='missing', match=False)
            g1_ok = False
            continue
        with open(dpath, 'rb') as f:
            data = f.read()
        sha = _sha256_bytes(data)
        bit_identical = bool(sha == digest_rows[name]['sha256'])
        drow = dict(path=os.path.abspath(dpath), sha256=sha,
                    bytes=len(data), pin_sha256=pin['sha'],
                    pin_bytes=pin['bytes'],
                    match=bool(sha == pin['sha']
                               and len(data) == int(pin['bytes'])),
                    bit_identical_to_original=bit_identical,
                    reproA_receipt='campaign11/results/reproA/T-%s' % name)
        dup_rows[name + '@' + REPROA_RUN] = drow
        g1_ok = g1_ok and drow['match']
        del data
    digest_rows.update(dup_rows)
    exp_check = expected_published_check(paths, digest_rows)
    gates['G1_checkpoints_loaded_digests'] = dict(
        rows=digest_rows, expected_published_crosscheck=exp_check,
        gate_pass=bool(g1_ok),
        rationale='originals byte-verified against the r20 PRV manifest '
                  'rows (present_hashed) BEFORE deserialization; reproA '
                  'duplicates hashed from local bytes against the '
                  'reproA-20260725 receipt pins, with original-vs-duplicate '
                  'bit-identity recorded per pair; expected_published.json '
                  'comparison recorded, gate decides on the recorded pins')
    if not g1_ok:
        payload = dict(part='archa1', gates=gates, smoke=smoke,
                       fatal='checkpoint verification failed',
                       environment=env_block(nproc), wall_s=time.time() - t0)
        write_payload(out_dir, 'archa1', payload)
        return False

    # evaluation (f32 forward, f64 scoring — rh_c3 protocol); duplicates are
    # bit-identical -> identical predictions by construction; evaluate each
    # DISTINCT byte-set once and copy rows (recorded).
    per_model, err_store, logit_store = {}, {}, {}
    det_ok = True
    for name in model_list:
        apply_f = states[name]
        outs = []
        for s in range(0, n_b, EVAL_CHUNK_F32):
            e = min(s + EVAL_CHUNK_F32, n_b)
            logits = np.asarray(apply_f(jnp.array(BX[s:e]),
                                        jnp.array(BE[s:e])))
            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            outs.append(logits)
        l0 = np.asarray(apply_f(jnp.array(BX[:min(EVAL_CHUNK_F32, n_b)]),
                                jnp.array(BE[:min(EVAL_CHUNK_F32, n_b)])))
        if l0.ndim == 3:
            l0 = l0.reshape(-1, l0.shape[-1])
        det_ok = det_ok and bool(np.array_equal(l0, outs[0]))
        logits_all = np.concatenate(outs, 0)
        preds = np.asarray(eng.g_gen.reconstruct(jnp.array(logits_all)))
        eq, ef = score_pred_panel(np.asarray(preds, np.float64), Gpanel)
        med = float(np.median(eq))
        per_model[name] = dict(
            median_err_q=med, err_q_ci95=boot_median_ci(eq),
            median_err_f=float(np.median(ef)), n=int(eq.size),
            n_nonfinite=int(np.sum(~np.isfinite(eq))),
            n_params=CKPTS[name]['n_params'])
        err_store[name] = (eq, ef)
        logit_store[name] = logits_all.astype(np.float32)
        print('%s   %-18s median err_q=%.6e n=%d'
              % (TAG, name, med, eq.size), flush=True)
    # duplicate rows (bit-identical -> same per-sample arrays; else would
    # have been evaluated separately — enforced by construction above)
    duplicates = {}
    for name in LADDER:
        drow = digest_rows[name + '@' + REPROA_RUN]
        bit_id = bool(drow.get('bit_identical_to_original'))
        med_o = per_model[name]['median_err_q']
        med_d = med_o if bit_id else None
        if not bit_id:
            # distinct bytes: evaluate the duplicate separately
            with open(drow['path'], 'rb') as f:
                data = f.read()
            from flax import serialization as flax_ser
            raw_d = flax_ser.msgpack_restore(data)
            del data
            model, params, bstats, _n = build_state(
                eng, name, CKPTS[name], raw_d, config01_stats)
            apply_d = make_apply(model, params, bstats)
            outs = []
            for s in range(0, n_b, EVAL_CHUNK_F32):
                e = min(s + EVAL_CHUNK_F32, n_b)
                lg = np.asarray(apply_d(jnp.array(BX[s:e]),
                                        jnp.array(BE[s:e])))
                if lg.ndim == 3:
                    lg = lg.reshape(-1, lg.shape[-1])
                outs.append(lg)
            preds_d = np.asarray(eng.g_gen.reconstruct(
                jnp.array(np.concatenate(outs, 0))))
            eq_d, _ef_d = score_pred_panel(np.asarray(preds_d, np.float64),
                                           Gpanel)
            med_d = float(np.median(eq_d))
            err_store[name + '@' + REPROA_RUN] = (eq_d, _ef_d)
        sig = sigma_run_pair(med_o, med_d, bit_id)
        duplicates[name] = dict(
            bit_identical=bit_id, median_original=med_o,
            median_duplicate=med_d,
            abs_delta_median=abs(med_o - med_d),
            sigma_run=sig, sigma_run_rel=sig / med_o if med_o > 0 else None,
            note=('duplicate bytes identical to the original (deterministic '
                  'retrain, reproA-20260725 finding) -> sigma_run == 0 '
                  'exactly; predictions identical by construction'
                  if bit_id else 'duplicate evaluated separately'))

    # capacity exponent alpha-hat per depth track + G-A1
    tracks = {}
    for depth in ('b4', 'b8'):
        names = ['mlp_cap_1p1M_%s' % depth, 'mlp_cap_4p4M_%s' % depth,
                 'mlp_cap_17M_%s' % depth]
        seq = []
        for lo, hi in ((0, 1), (1, 2)):
            n_lo, n_hi = names[lo], names[hi]
            N_lo, N_hi = CKPTS[n_lo]['n_params'], CKPTS[n_hi]['n_params']
            log_ratio = math.log(N_hi / N_lo)
            m_lo = per_model[n_lo]['median_err_q']
            m_hi = per_model[n_hi]['median_err_q']
            a = alpha_hat(m_lo, m_hi, N_lo, N_hi)
            eci = boot_alpha_ci(err_store[n_lo][0], err_store[n_hi][0],
                                log_ratio)
            s_lo = duplicates[n_lo]['sigma_run_rel'] or 0.0
            s_hi = duplicates[n_hi]['sigma_run_rel'] or 0.0
            cci, d_alpha = combine_alpha_ci(a, eci, s_lo, s_hi, log_ratio)
            seq.append(dict(pair='%s->%s' % (n_lo, n_hi),
                            n_params=[N_lo, N_hi],
                            medians=[m_lo, m_hi], alpha_hat=a,
                            eval_ci95=eci, run_noise_d_alpha=d_alpha,
                            combined_ci95=cci))
        tracks[depth] = seq
    deciding = tracks['b8'][1]              # b8 4.4M -> 17.4M (R6 wording;
    # b8 is the registered standardized-MLP depth: mlp_cap_17M_b8 is the
    # printed Table-IV std-MLP row and the rh_c3 subject)
    branch, branch_rat = ga1_branch(deciding['combined_ci95'][0],
                                    deciding['combined_ci95'][1])
    # legacy-convention context (registered c6_capacity_eval medians)
    legacy_ctx = {}
    for m in LADDER:
        try:
            legacy_ctx[m] = float(
                c6_rows[m]['tab3_4096']['rel_param_err_median'])
        except Exception:                                     # noqa: BLE001
            legacy_ctx[m] = None
    legacy_alpha = {}
    for depth in ('b4', 'b8'):
        lo_m = legacy_ctx.get('mlp_cap_4p4M_%s' % depth)
        hi_m = legacy_ctx.get('mlp_cap_17M_%s' % depth)
        if lo_m and hi_m:
            legacy_alpha['%s_4p4_to_17' % depth] = alpha_hat(
                lo_m, hi_m, CKPTS['mlp_cap_4p4M_%s' % depth]['n_params'],
                CKPTS['mlp_cap_17M_%s' % depth]['n_params'])

    # cross-checks vs registered CPU anchors (rh_c3 / rg_h9)
    xchecks, x_ok = {}, True
    for m, ref in RH_C3_SEED1007.items():
        mine = per_model[m]['median_err_q']
        rel = abs(mine - ref) / ref
        ok = (not full_run) or bool(rel <= XHOST_MED_RTOL)
        xchecks[m] = dict(mine=mine, registered_rh_c3=ref, rel_dev=rel,
                          ok=ok)
        x_ok = x_ok and ok
    prod512 = float(np.median(err_store['prod_thermal_random'][0][:512]))
    rel512 = abs(prod512 - RG_H9_D0_MED) / RG_H9_D0_MED
    ok512 = (not full_run) or bool(rel512 <= XHOST_MED_RTOL)
    xchecks['prod_thermal_random_rows512'] = dict(
        mine=prod512, registered_rg_h9_delta0=RG_H9_D0_MED,
        rel_dev=rel512, ok=ok512)
    x_ok = x_ok and ok512
    gates['A3_registered_cpu_anchor_consistency'] = dict(
        checks=xchecks, tol_rel=XHOST_MED_RTOL,
        smoke_truncated=bool(not full_run),
        gate_pass=bool(x_ok),
        rationale='fresh-feature CPU medians must sit within %g relative of '
                  'the registered same-protocol anchors (rh_c3 per_model '
                  'seed1007, rg_h9 delta=0 rows 0..512); the f32 eigh '
                  'pipeline is host-class dependent at the ~1e-5 level, so '
                  'a violation of this band means a convention/model error, '
                  'not arithmetic jitter; enforced at full n only'
                  % XHOST_MED_RTOL)
    finite_ok = all(v['n_nonfinite'] == 0 for v in per_model.values())
    gates['A4_determinism_finiteness'] = dict(
        feature_block_regen_identical=feat_det,
        repeated_forward_chunk_identical=bool(det_ok),
        all_finite=bool(finite_ok),
        gate_pass=bool(feat_det and det_ok and finite_ok),
        rationale='regenerated feature block and repeated jitted forward '
                  'byte-identical; every per-sample error finite')
    alpha_vals = [s['alpha_hat'] for d in tracks.values() for s in d]
    ci_vals = [x for d in tracks.values() for s in d
               for x in s['eval_ci95'] + s['combined_ci95']]
    g4_ok = bool(np.all(np.isfinite(alpha_vals))
                 and np.all(np.isfinite(ci_vals))
                 and all(s['combined_ci95'][0] <= s['alpha_hat']
                         <= s['combined_ci95'][1]
                         for d in tracks.values() for s in d)
                 and isinstance(branch, str))
    gates['G4_alpha_gA1_emitted'] = dict(
        gate_pass=g4_ok,
        deciding=dict(track='b8', pair=deciding['pair'],
                      alpha_hat=deciding['alpha_hat'],
                      eval_ci95=deciding['eval_ci95'],
                      combined_ci95=deciding['combined_ci95'],
                      branch=branch),
        rationale='alpha-hat + eval CI (paired per-H cluster bootstrap, '
                  'rh_c3 pins) + run-noise-combined CI + the G-A1 branch '
                  'emitted and finite; deciding leg = b8 track 4.4M->17.4M '
                  '(R6), sigma_run from the original-vs-reproA duplicate '
                  'pairs (bit-identical pairs contribute exactly 0)')

    npz_path = os.path.join(out_dir, 'expc1_archa1_persample.npz')
    store = dict(labels_dev1007=labels, nodes_n_params=np.asarray(
        [CKPTS[m]['n_params'] for m in model_list], np.int64))
    for name, (eq, ef) in err_store.items():
        key = name.replace('@', '_AT_')
        store['err_q_%s' % key] = eq
        store['err_f_%s' % key] = ef
    for name, lg in logit_store.items():
        store['logits55_%s' % name] = lg
    np.savez_compressed(npz_path, **store)
    print('%s wrote %s' % (TAG, npz_path), flush=True)

    payload = dict(
        part='archa1',
        item='ARCH-A1 / R6 (FINAL_PLAN.md ~:346; power F3)',
        protocol=dict(
            stream='registered seed-1007 val labels rows 0..%d (tab3_4096 '
                   'dev rows, byte-pinned)' % n_b,
            convention='rh_c3 CPU protocol: f32 dataset-pipeline features '
                       '(solve_batch_kernel, p1_core.py:783-855), f32 '
                       'forward, float64 err_pair scoring '
                       '(c10_h0_sensitivity.py:213-219)',
            bootstrap='per-Hamiltonian cluster bootstrap B=%d, '
                      'default_rng(%d), order stats %d/%d; alpha CIs '
                      'draw-paired (one shared index draw per replicate)'
                      % (BOOT_B, BOOT_SEED, BOOT_LO, BOOT_HI),
            run_noise='sigma_run = |median(orig) - median(reproA dup)| / '
                      'sqrt(2) per capacity point; propagated as d_alpha = '
                      'sqrt(srel_lo^2+srel_hi^2)/ln(N_hi/N_lo), added in '
                      'quadrature to the eval-CI half-widths at 1.96 sigma',
            g_a1_boundaries=[GA1_LOW, GA1_HIGH]),
        results=dict(per_model=per_model, duplicates=duplicates,
                     alpha_tracks=tracks,
                     g_a1=dict(deciding_track='b8',
                               deciding_pair=deciding['pair'],
                               alpha_hat=deciding['alpha_hat'],
                               eval_ci95=deciding['eval_ci95'],
                               run_noise_d_alpha=deciding
                               ['run_noise_d_alpha'],
                               combined_ci95=deciding['combined_ci95'],
                               branch=branch, branch_rationale=branch_rat),
                     legacy_convention_context=dict(
                         medians_tab3_tpu=legacy_ctx,
                         alpha_4p4_to_17=legacy_alpha,
                         note='registered c6_capacity_eval (TPU legacy '
                              'convention) — context only, never the '
                              'G-A1 deciding statistic'),
                     persample_npz=os.path.basename(npz_path),
                     persample_npz_sha256=_sha256_file(npz_path)),
        checkpoints=digest_rows,
        fmb_rho_policy=fmb_policy,
        gates=gates, smoke=bool(smoke),
        environment=env_block(nproc),
        script_sha256=_sha256_file(os.path.abspath(__file__)),
        wall_s=time.time() - t0)
    write_payload(out_dir, 'archa1', payload)
    print('%s G-A1: alpha_hat=%.4f combined_ci=[%.4f, %.4f] branch=%s'
          % (TAG, deciding['alpha_hat'], deciding['combined_ci95'][0],
             deciding['combined_ci95'][1], branch), flush=True)
    return all(bool(g.get('gate_pass')) for g in gates.values())


# ==================================================================== merging
def merge_and_emit(out_dir, nproc, smoke_flag):
    """Merge whatever part payloads exist into the stage result JSON.
    all_gates_pass requires BOTH parts present, every executed gate True and
    neither payload smoke."""
    payloads = {}
    for part in ('cal1', 'archa1'):
        p = os.path.join(out_dir, 'expc1_part_%s.json' % part)
        if os.path.exists(p):
            with open(p) as f:
                payloads[part] = json.load(f)
    gates = {}
    # G1 union
    g1_rows, g1_pass, g1_have = {}, True, False
    exp_checks = {}
    for part, pl in payloads.items():
        g1 = (pl.get('gates') or {}).get('G1_checkpoints_loaded_digests')
        if g1:
            g1_have = True
            g1_rows.update(g1.get('rows') or {})
            exp_checks.update(g1.get('expected_published_crosscheck') or {})
            g1_pass = g1_pass and bool(g1.get('gate_pass'))
    gates['G1_checkpoints_loaded_digests'] = dict(
        {'pass': bool(g1_have and g1_pass)},
        rationale=('all requested checkpoints loaded from byte-verified '
                   'bytes with recorded sha256 digests matching the '
                   'registered pins (r20 PRV manifest / reproA receipts); '
                   'expected_published.json cross-check recorded per row '
                   '(gate decides on the recorded pins)'
                   if g1_have else 'no part executed'),
        rows=g1_rows, expected_published_crosscheck=exp_checks)
    for key, part in (('G2_perbudget_se_finite_registered', 'cal1'),
                      ('G3_thresholds_table_emitted', 'cal1'),
                      ('G4_alpha_gA1_emitted', 'archa1')):
        src = (payloads.get(part, {}).get('gates') or {}).get(key)
        if src is None:
            gates[key] = {'pass': None,
                          'rationale': 'part %r not executed in this '
                                       'out_dir yet' % part}
        else:
            g = dict(src)
            g['pass'] = bool(g.pop('gate_pass', False))
            gates[key] = g
    # auxiliary gates carried per part
    aux = {}
    for part, pl in payloads.items():
        for k, v in (pl.get('gates') or {}).items():
            if k.startswith('G'):
                continue
            g = dict(v)
            g['pass'] = bool(g.pop('gate_pass', False))
            aux['%s/%s' % (part, k)] = g
    gates.update(aux)
    smoke_any = bool(smoke_flag or any(pl.get('smoke')
                                       for pl in payloads.values()))
    executed = {k: g for k, g in gates.items() if g.get('pass') is not None}
    both = ('cal1' in payloads) and ('archa1' in payloads)
    all_pass = bool(both and executed
                    and all(g['pass'] for g in executed.values())
                    and not smoke_any)
    result = dict(
        schema='expc1-cal1-archa1-v1',
        stage=STAGE,
        task=('experiment-campaign1 CPU track: CAL-1 noise-grid seed-spread '
              'calibration (dev-1007, 6 budgets x 100 H x 20 gaussian '
              'draws, CPU-f64) + ARCH-A1 capacity-ladder CPU-f64 rescore '
              'with run-noise term and G-A1 decision '
              '(panel/FINAL_PLAN.md CAL-1 ~:559 / R6 ~:346 / §3 / §6)'),
        parts_executed=sorted(payloads),
        conventions=dict(
            cal1_produced='cpu-f64 (STAT-04 cpu64 branch)',
            cal1_legacy_companion='NOT PRODUCED — accelerator convention '
                                  'unavailable on CPU; CPU-f32 proxy ruled '
                                  'not needed (see part payload protocol)',
            archa1='rh_c3 CPU-f64-scoring convention (f32 pipeline features '
                   '+ f32 forward + f64 error arithmetic)'),
        results={part: pl.get('results') for part, pl in payloads.items()},
        gates=gates,
        gate_pass_bits={k: g.get('pass') for k, g in gates.items()},
        all_gates_pass=all_pass,
        smoke=smoke_any,
        environment=env_block(nproc),
        script_sha256=_sha256_file(os.path.abspath(__file__)),
        emitted_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    jp = os.path.join(out_dir, '%s_result.json' % STAGE)
    with open(jp, 'w') as f:
        json.dump(_jsonable(result), f, indent=1)
        f.write('\n')
    print('%s wrote %s (all_gates_pass=%s)' % (TAG, jp, all_pass),
          flush=True)
    if smoke_any and payloads:
        print('%s SMOKE RECORD — all_gates_pass forced False, never '
              'quotable' % TAG, flush=True)
    return all_pass


# =================================================================== selftest
def selftest():
    """Structural checks of every pure code path — numpy + stdlib only; no
    jax/flax/scipy/engine and no registered inputs required."""
    failures = []

    def check(name, ok):
        print('  [%s] %s' % ('ok' if ok else 'FAIL', name))
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(0)
    # scoring: err_shifted == err_pair[0]; gauge properties
    G = rng.standard_normal((MP, MP))
    G = 0.5 * (G + G.T)
    ng = float(np.linalg.norm(G))
    check('err_shifted_zero_on_identity',
          err_shifted(G.copy(), G, ng) == 0.0)
    Gs = G.copy()
    np.fill_diagonal(Gs, np.diag(Gs) + 0.37)
    check('err_shifted_gauges_out_uniform_diag_shift',
          err_shifted(Gs, G, ng) < 1e-12)
    check('err_shifted_equals_err_pair_q',
          abs(err_shifted(Gs + 0.01, G, ng) - err_pair(Gs + 0.01, G)[0])
          < 1e-15)
    # panel roundtrip
    v = rng.standard_normal((8, LABEL55))
    Gm = panel_matrices(v)
    r, c = np.triu_indices(MP)
    check('panel_triu_roundtrip',
          float(np.max(np.abs(Gm[:, r, c] - v))) < 1e-12)
    check('panel_symmetric',
          float(np.max(np.abs(Gm - np.swapaxes(Gm, 1, 2)))) == 0.0)
    # budget grid pins
    sw = sweep50()
    check('sweep50_nodes_pinned',
          bool(np.allclose([float(sw[i]) for i in NODES6], NODES6_NS_PIN,
                           rtol=1e-12)))
    # gaussian drawer: determinism, normalization, clean limit
    p = np.abs(rng.standard_normal(252))
    p /= p.sum()
    r1 = np.random.default_rng(np.random.SeedSequence([0, 2, 0, 5, 7]))
    r2 = np.random.default_rng(np.random.SeedSequence([0, 2, 0, 5, 7]))
    d1 = draw_gaussian(p, 9102.98, r1)
    d2 = draw_gaussian(p, 9102.98, r2)
    check('drawer_deterministic_stream', bool(np.array_equal(d1, d2)))
    check('drawer_normalized', abs(float(np.sum(d1)) - 1.0) < 1e-12)
    check('drawer_nonnegative', bool(np.all(d1 >= 0)))
    d_clean = draw_gaussian(p, 1e18,
                            np.random.default_rng(np.random.SeedSequence(
                                [0, 2, 0, 0, 0])))
    check('drawer_clean_limit',
          float(np.max(np.abs(d_clean - p))) < 1e-7)
    r3 = np.random.default_rng(np.random.SeedSequence([0, 2, 0, 5, 8]))
    check('drawer_streams_distinct',
          not np.array_equal(d1, draw_gaussian(p, 9102.98, r3)))
    # bootstrap pins
    check('boot_indices_25_975', (BOOT_LO, BOOT_HI) == (25, 975))
    check('boot_ci_constant', boot_median_ci(np.ones(64)) == [1.0, 1.0])
    check('boot_ci_deterministic',
          boot_median_ci(np.arange(100.0))
          == boot_median_ci(np.arange(100.0)))
    e_hd = np.abs(rng.standard_normal((20, 5))) + 1.0
    ci_c = boot_median_ci_cluster(e_hd)
    check('boot_cluster_ordered', ci_c[0] <= ci_c[1])
    check('boot_cluster_constant',
          boot_median_ci_cluster(np.ones((16, 4))) == [1.0, 1.0])
    # seed-spread stats + threshold arithmetic
    st = seed_spread_stats([1.0, 1.02, 0.98])
    se_expect = np.std([1.0, 1.02, 0.98], ddof=1) / math.sqrt(3)
    check('se_arithmetic', abs(st['se'] - se_expect) < 1e-15)
    check('thr_floor_10', st['thr_G3_NOISEclaim_pct'] == 10.0)
    check('thr_floor_5', st['thr_GNINJ_pct'] == 5.0)
    st2 = seed_spread_stats([1.0, 1.2, 0.8])
    check('thr_2se_binds_when_large',
          abs(st2['thr_G3_NOISEclaim_pct'] - 2 * st2['se_rel_pct']) < 1e-12
          and st2['thr_G3_NOISEclaim_pct'] > 10.0)
    # alpha machinery: exact on a synthetic power law err = N^-alpha
    n1, n2 = 4.4e6, 17.4e6
    a_true = 0.15
    m1, m2 = n1 ** -a_true, n2 ** -a_true
    check('alpha_hat_exact',
          abs(alpha_hat(m1, m2, n1, n2) - a_true) < 1e-12)
    base = np.abs(rng.standard_normal(512)) + 0.5
    e_lo = base * m1
    e_hi = base * m2                       # per-sample ratio constant
    ci = boot_alpha_ci(e_lo, e_hi, math.log(n2 / n1))
    check('alpha_boot_degenerate_on_constant_ratio',
          abs(ci[0] - a_true) < 1e-9 and abs(ci[1] - a_true) < 1e-9)
    try:
        boot_alpha_ci(e_lo[:100], e_hi, math.log(n2 / n1))
        check('alpha_boot_length_guard', False)
    except ValueError:
        check('alpha_boot_length_guard', True)
    # run-noise combination
    check('sigma_run_zero_when_bit_identical',
          sigma_run_pair(1.0, 2.0, True) == 0.0)
    check('sigma_run_pair_scale',
          abs(sigma_run_pair(1.0, 1.1, False) - 0.1 / math.sqrt(2)) < 1e-15)
    cci, d_a = combine_alpha_ci(0.10, [0.08, 0.12], 0.0, 0.0,
                                math.log(n2 / n1))
    check('combine_ci_noop_at_zero_run_noise',
          d_a == 0.0 and abs(cci[0] - 0.08) < 1e-15
          and abs(cci[1] - 0.12) < 1e-15)
    cci2, d_a2 = combine_alpha_ci(0.10, [0.08, 0.12], 0.05, 0.05,
                                  math.log(n2 / n1))
    check('combine_ci_widens_with_run_noise',
          d_a2 > 0 and cci2[0] < 0.08 and cci2[1] > 0.12
          and cci2[0] < 0.10 < cci2[1])
    # G-A1 branch logic (all three branches)
    check('ga1_authorize', ga1_branch(0.081, 0.2)[0] == 'authorize')
    check('ga1_close', ga1_branch(-0.02, 0.049)[0] == 'close')
    check('ga1_straddle',
          ga1_branch(0.03, 0.10)[0] == 'straddle_no_arch_tpu_this_window')
    check('ga1_inside_band_is_straddle',
          ga1_branch(0.055, 0.075)[0] == 'straddle_no_arch_tpu_this_window')
    # CAL-1 worker vs serial reference on a synthetic system (numpy-only)
    D = 12
    Vq, _ = np.linalg.qr(rng.standard_normal((D, D)))
    E = np.sort(rng.standard_normal(D))
    pt = np.exp(-(E - E.min()))
    pt /= pt.sum()
    ops = rng.standard_normal((MP * MP, D, D))
    _POOL_STATE['ops'] = ops
    si, bx, be = cal1_features_for_H(
        (3, Vq, E, pt, [0, 49], [1e10, 1e2], 2))
    ok_w = (si == 3 and bx.shape == (2, 2, MP, MP)
            and be.shape == (2, 2) and np.isfinite(bx).all())
    # serial reference for one cell
    rngc = np.random.default_rng(np.random.SeedSequence([0, 2, 49, 3, 1]))
    pm = draw_gaussian(pt, 1e2, rngc)
    rho = (Vq * pm) @ Vq.T
    ref = np.einsum('ji,kij->k', rho, ops).reshape(MP, MP)
    ok_w = ok_w and bool(np.array_equal(ref, bx[1, 1]))
    ok_w = ok_w and abs(float(np.sum(pm * E)) - be[1, 1]) < 1e-15
    check('cal1_worker_matches_serial_reference', bool(ok_w))
    # pins self-consistency
    check('reproA_dup_pins_equal_originals',
          all(REPROA_DUPS[m]['sha'] == CKPTS[m]['sha'] for m in LADDER))
    check('tab3_pins_selfconsistent',
          abs(TAB3_CAP_MED / TAB3_PROD_MED - TAB3_R_ARCH) < 1e-12)
    # optional: expected_published reachable -> pin + published prod sha
    try:
        p = _first_existing(
            [EXPECTED_PUBLISHED_CANDS[0],
             os.path.join(_REPO_ROOT, EXPECTED_PUBLISHED_CANDS[1])],
            'expected_published.json')
        with open(p) as f:
            exp = json.load(f)
        check('expected_published_prod_pin',
              exp.get('prod_thermal_random', {}).get('expected_sha256')
              == CKPTS['prod_thermal_random']['sha'])
        check('expected_published_ablmlp_pin',
              exp.get('abl_mlp', {}).get('expected_sha256')
              == CKPTS['abl_mlp']['sha'])
    except FileNotFoundError:
        print('  [skip] expected_published.json not reachable here')
    if failures:
        print('SELFTEST_FAIL: %s' % failures)
        return 1
    print('SELFTEST_OK')
    return 0


# ----------------------------------------------------------------------- main
def main():
    if '--selftest' in sys.argv[1:]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(
        description='expc1 CAL-1 + ARCH-A1 CPU stage')
    ap.add_argument('out_dir')
    ap.add_argument('--part', choices=('cal1', 'archa1', 'all'),
                    default='all')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--ckpt-root', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-merge', action='store_true',
                    help='(internal) child mode: write the part payload '
                         'only; the parent merges')
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    smoke = args.smoke or os.environ.get('EXPC1_SMOKE', '') == '1'

    if args.part == 'all':
        # cal1 needs jax_enable_x64=True from the first jax op; archa1 must
        # run WITHOUT it (rh_c3 f32 convention) -> one subprocess per part.
        if not args.ckpt_root:
            raise SystemExit('%s --ckpt-root is required for evaluation '
                             'runs (local mirror from expc1_fetch_ckpts.sh);'
                             ' use --selftest for the no-input structural '
                             'check.' % TAG)
        rc_all = 0
        for part in ('cal1', 'archa1'):
            cmd = [sys.executable, os.path.abspath(__file__), out_dir,
                   '--part', part, '--nproc', str(args.nproc),
                   '--no-merge']
            if args.ckpt_root:
                cmd += ['--ckpt-root', args.ckpt_root]
            if smoke:
                cmd += ['--smoke']
            print('%s spawning part %s: %s' % (TAG, part, ' '.join(cmd)),
                  flush=True)
            rc = subprocess.call(cmd)
            rc_all |= rc
        ok = merge_and_emit(out_dir, args.nproc, smoke)
        sys.exit(0 if (rc_all == 0 and (ok or smoke)) else 1)

    if not args.ckpt_root:
        raise SystemExit('%s --ckpt-root is required for evaluation runs '
                         '(local mirror from expc1_fetch_ckpts.sh); use '
                         '--selftest for the no-input structural check.'
                         % TAG)
    ckpt_root = os.path.abspath(os.path.expanduser(args.ckpt_root))
    t0 = time.time()
    if args.part == 'cal1':
        ok = run_cal1(out_dir, ckpt_root, max(1, args.nproc), smoke)
    else:
        ok = run_archa1(out_dir, ckpt_root, max(1, args.nproc), smoke)
    if not args.no_merge:
        merge_and_emit(out_dir, args.nproc, smoke)
    print('%s part %s done in %.1fs gates_pass=%s'
          % (TAG, args.part, time.time() - t0, ok), flush=True)
    if not ok and not smoke:
        sys.exit(1)


if __name__ == '__main__':
    main()
