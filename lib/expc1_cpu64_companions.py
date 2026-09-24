#!/usr/bin/env python3
# =============================================================================
# experiments/experiment-campaign1/stages/expc1_cpu64_companions.py
#
# CPU-f64 COMPANION LEGS — the STD-1 ADOPTION BLOCKER plus descriptive
# companions (panel/FINAL_PLAN.md v1.0-panel: "### R4. STD-1" gate G-STD /
# GT-1 §3 rule 2, §3 rule 4 CPU-f64 seed floor SE 7.78% / CI ±16–22%,
# "### R3. LC-1" dual-convention completion, "### R11. ARCH-A2a" CPU-f64
# descriptive companion; §6 gate table row "G-STD (R4)").
#
# RUNS OFF-FLEET on a rented many-core CPU box (192-core class).  There is NO
# c11 shim here: the stage self-writes its result JSON and a sha256 MANIFEST
# (harvest = scp the out_dir back + re-hash locally against MANIFEST.sha256).
# RUN_ID stays expc1-20260729 (same campaign; execution recorded off-fleet).
#
# WHAT IT DECIDES / REPORTS (all on the pinned seed-1007 dev rows 0..4096,
# tab3_4096 order, ONE convention = the STAT-04 cpu64 deliverable branch):
#
#   part std  (GATING for adoption — the blocker):
#     std_thermal_random{,_s43,_s44} vs the registered prod triplet
#     (prod_thermal_random / abl_gram_s43 / abl_gram_s44), seed-paired
#     (42<->42, 43<->43, 44<->44).  The prod CPU-f64 medians are computed
#     FRESH IN THIS RUN — the legacy pins (PREC-A reg_bytes__tpu_default)
#     belong to a DIFFERENT convention and are NEVER reused as denominators.
#     Companion verdict: "sign-consistent (3/3 improve)" or
#     "sign-inconsistent (k/3)" (GT-1 companion leg: all three CPU-f64
#     seed-paired deltas improve in DIRECTION).  Magnitude claimable flag
#     ONLY if all three improvements >= 16% (GT-1: 2 x the 7.78%
#     difference-SE; §3 rule 4 — no CPU-f64 magnitude gate below 16% is
#     decidable at n=3).  Combined with the REGISTERED deciding leg
#     (results/W2-rescore-std/W2-rescore-std_result.json, sha256
#     8bda6187...3203b, decision "pass-deciding-leg (CPU-f64 companion
#     pending)", mean +5.74% legacy) it emits
#       adoption_status = "ADOPT (deciding leg PASS + companion
#                          sign-consistent)"    when 3/3 improve
#                       = "legacy-only (companion failed)"  otherwise.
#
#   part lc   (descriptive, NON-gating — R3's dual-convention completion):
#     ogn_lc_1e4/1e5/1e6 + the prod 5e6 endpoint -> the 4-point CPU-f64
#     learning curve with per-decade slopes + draw-paired CIs.  n_seeds=1;
#     under the CPU-f64 seed floor (SE 7.8%) the slope is recorded, never
#     gated (R3: "the CPU-f64 companion slope is recorded but non-gating").
#     The registered legacy G1 decision (LC-rescore, branch kill_null) is
#     pinned alongside as context.
#
#   part arch2 (descriptive, NON-gating): arch2_thermal_random{,_s43,_s44}
#     vs the prod triplet, seed-paired CPU-f64 deficits.  The "saturated
#     below 8M" question NEEDS this leg, but at n=3 the CPU-f64 CI is
#     ±16–22% and cannot support a saturation claim (R11 wording, quoted in
#     the block).  The registered legacy G-A2 decision (W2-rescore-full,
#     branch bounded) is pinned alongside as context.
#
#   part mlplc (descriptive, NON-gating): the registered campaign6 MLP LC
#     checkpoints mlp_lc_1e4/1e5/1e6 (+ the abl_mlp 5e6 endpoint) IF present
#     at --ckpt-root; absent members are RECORDED absent, never a failure.
#
#   part precb (descriptive, NON-gating — the precision 2x2's LAST cell:
#     clean-TRAINED models under clean EVAL; the PB-rescore record's own
#     named follow-up, g_pb.cpu_f64_companion.producer): the precb triple
#     precb_thermal_random{,_s43,_s44} (plain OGN res=3, n_params
#     17,756,929 — the PLAIN prod restore path, NO std stats; trained on
#     the prechigh cache, results/PRECB-gen) vs the prod triplet computed
#     FRESH in-run (as part std does): seed-paired CPU-f64 deltas + sign
#     consistency.  The cpu-f64 exact-ED features ARE the clean-eval
#     corner.  Interpretation fields QUOTE (never re-decide) the
#     registered G-PB legacy null (results/PB-rescore, record sha
#     a59a2406..., mean -0.57% inside seed noise: clean-feature TRAINING
#     does not move the legacy floor) and PREC-A's eval-time feature term
#     (PRECA-tpu record ecc80f42..., ratios 0.356/0.384/0.405 ~ x2.6);
#     the question answered: "does clean-feature training express under
#     clean eval?"  This part NEVER emits an adoption status — C7 gates
#     the emitted-DESCRIPTIVE shape only (gating=False beyond the C-gate
#     plumbing).
#
# CONVENTION (C2 pins; produced for EVERY scored model):
#   cpu-f64 == the STAT-04 cpu64 deliverable branch
#   (campaign10/stages/c10_tableiii_cpu64.py:33-36,509-518,575-585, reused
#   here via expc1_cal1_archa1.py — REUSE BY IMPORT, see LINEAGE):
#     * jax_enable_x64=True BEFORE the first jax op
#       (expc1_cal1_archa1.load_engine6(want_x64=True), :619-660)
#     * float64 EXACT-ED features regenerated from the pinned labels:
#       G_true panel -> dense 252x252 f64 H (eng._shots_h_dense64) -> scipy
#       f64 eigh -> beta=1 thermal p_true -> rho_true=(V*p_true)@V.T ->
#       pair block einsum('ji,kij->k') (compute_rho_m contraction WITHOUT
#       its float32 cast) ; be = sum(p_true*E)  — the CAL-1 noise-feature
#       machinery (expc1_cal1_archa1.cal1_features_for_H, :563-586 /
#       c10_tableiii_cpu64.py:339-371 f64 branch) at the CLEAN point
#       p_meas == p_true (no drawer).
#     * float64 params (checkpoint f32 leaves cast, expc1_cal1_archa1.py:
#       998-1003) ; forward under jax.default_matmul_precision('highest')
#       baked at trace time (expc1_cal1_archa1.py:1022-1049, make_apply
#       :720-729)
#     * float64 scoring: err_q/err_f VERBATIM c10_h0_sensitivity.py:213-219
#       via expc1_cal1_archa1.score_pred_panel/err_pair (:397-442)
#   The LEGACY convention is NOT producible on CPU and is NOT produced here
#   (expc1_cal1_archa1.py:89-97 ruling); the deciding leg lives in the
#   registered W2 records.
#
# LINEAGE (REUSE by import or line-cited copy; nothing reinvented):
#   * import expc1_cal1_archa1 as A (sibling, numpy-only at module level):
#     pins (MP/LABEL55/EVAL_CHUNK_F64/BOOT_*/DIAG_GAUGE_*), panel_matrices,
#     err_pair, score_pred_panel, boot_median_ci, _sha256_*, _arr_sha,
#     _jsonable, load_engine6, verify_and_restore, build_state, make_apply,
#     env_block, CKPTS (prod triplet + abl_mlp pins of record).
#   * StdPhysicsOrbitalGraphNet + std_stats loader: VERBATIM copies of
#     expc1_wave2_rescore.py:1172-1235 (_load_std_stats) and :1238-1361
#     (_build_std_model) — themselves the verbatim relay of t_std1.py:249-360
#     (the production PhysicsOrbitalGraphNet p2_models5.py:290-361 + the
#     single [STD-1] stanza), t_std1.py:381-407 (_set_std_stats nested-tuple
#     encoding), t_std1.py:620 (symmetrize-first), t_std1.py:692-703
#     (std_stats.npz writer) and :788-802 (config.json std block).  The std
#     params pytree is IDENTICAL to the production OGN (standardization adds
#     zero params), so the msgpack restore path is unchanged.
#   * seed-paired delta bootstrap: expc1_wave2_rescore.py:290-320
#     (_boot_mean_deficit_ci_paired) sign-flipped to the R4 improvement
#     orientation; ratio CI: expc1_preca.py:395-410 (_boot_ratio_ci_paired)
#     copied; LC slope machinery: expc1_lc_rescore.py:161-189,625-700
#     (slope_per_decade / _boot_slope_ci_paired / curve builder) copied.
#   * checkpoint discipline: byte verification BEFORE deserialization
#     (expc1_cal1_archa1.verify_and_restore, :663-676; campaign10
#     checkpoint_authority contract).
#
# CHECKPOINTS OF RECORD (sha256 pins; sources in the result JSON rows):
#   prod triplet + abl_mlp        == expc1_cal1_archa1.CKPTS (r20 PRV rows)
#   std triple                    == W2-rescore-std/-full records (f47bd8fa/
#                                    f975b0bc/b9188271; + std_stats.npz
#                                    372e7ca7 and per-seed config.json pins)
#   arch2 triple                  == W2-rescore-full record (d31540e8/
#                                    110599a8/2e4dc851, 94,882,579 B)
#   ogn_lc triple                 == LC-rescore record (6434d70e/4fdecb5c/
#                                    b264a7a1)  [FRESH LC-1 run; NOT the
#                                    orphaned r20 GCS ogn_lc_1e4 1828e3f7]
#   mlp_lc triple (optional)      == r20 PRV manifest rows (fb069deb/
#                                    2c5614c4/5ffacdf4, 209,173,414 B)
#   precb triple                  == T-precb_thermal_random{,_s43,_s44}
#                                    harvested training receipts, artifacts
#                                    final_state.msgpack rows (bdc320eb/
#                                    371986b7/865ce8b9, 213,094,675 B each;
#                                    n_params 17,756,929, spec standardize=
#                                    False prechigh=True)
#
# GATES (result JSON gates.*.{pass, rationale}; receipt-safe: NO other
# pass-like key nests under gates — the expc1_g3_eval.py:795 lesson):
#   C0_stream_pinned          labels rows byte-pinned (cache rows sha
#                             94f616fb... or a registered labels-npz pin)
#   C1_checkpoints_loaded_digests  every REQUIRED checkpoint of the selected
#                             parts byte-verified (sha256+bytes) against the
#                             pins BEFORE deserialization, restored, and
#                             n_params == the family pin; optional mlplc
#                             members may be absent (recorded)
#   C2_convention_pins_echoed x64 active + default float64 + cpu backend +
#                             matmul 'highest' + f64-exact features + f64
#                             scoring echoed in protocol AND live-checked
#   C3_std_companion_verdict_emitted  a DECIDED companion verdict (either
#                             sign branch passes) + pinned adoption_status +
#                             3 finite per-seed rows + the deciding-leg
#                             record sha echoed; pending-with-missing passes
#                             ONLY when part std was explicitly deselected
#   C4_manifest_written       MANIFEST.sha256 written and re-hash-verified
#                             (npz + script at result-write time; the result
#                             JSON line is appended post-write and the FULL
#                             manifest re-verified before exit — exit code
#                             reflects it)
#   C5_anchor_convention_band prod-triplet CPU-f64 medians within
#                             [0.85, 1.15] of BOTH registered CPU anchor
#                             sets (ARCH-A1 f32-pipeline medians; CAL-1
#                             N_s=1e10 cpu64 medians) — a convention error
#                             (legacy 3.7x class) blows this band, honest
#                             values sit within ~±10%; full n only
#   C6_determinism_finiteness spawn-pool feature block == serial recompute
#                             byte-for-byte; repeated jitted f64 forward
#                             chunk byte-identical; every error finite
#   C7_precb_part_emitted     part precb emitted in its DESCRIPTIVE shape:
#                             status decided, gating False, 3 finite
#                             per-seed rows, sign-consistency string, the
#                             PB-rescore legacy-null record sha echoed and
#                             NO adoption-status key (this part never
#                             adopts); when precb was deselected the block
#                             must say not_requested
#   all_gates_pass = AND over gates AND not smoke (smoke never quotable).
#
# CLI:
#   expc1_cpu64_companions.py <out_dir> --ckpt-root DIR --val-cache PATH
#                             [--parts std,lc,arch2,mlplc,precb] [--nproc N]
#                             [--labels-npz PATH] [--n-eval N] [--smoke]
#                             [--pb-record PATH]
#   expc1_cpu64_companions.py --selftest      (jax-free structural checks)
#     --ckpt-root  <DIR>/<model>/final_state.msgpack (+ std_stats.npz and
#                  config.json for std models; <DIR>/abl_mlp optional)
#     --val-cache  registered val cache: a directory holding shard_*.npz
#                  (ds_d20_random_thermal/val convention) OR a single
#                  shard .npz; labels key 'labels' (f32 (n,55))
#     --labels-npz registered predictions npz fallback (g_true rows; pins
#                  e215e0fa / fbbb4ce2) — used when --val-cache is absent
#     --parts      default std,lc,arch2,mlplc; mlplc members are optional-
#                  if-present in every case
#     --nproc      spawn-pool width for the f64 feature build (eigh); the
#                  f64 forwards use XLA's own intra-op threads
#     --smoke      n_eval=256; gates evaluated; all_gates_pass FORCED False
#
# OUTPUT: <out_dir>/expc1_cpu64_companions_result.json
#         <out_dir>/expc1_cpu64_companions_persample.npz
#         <out_dir>/MANIFEST.sha256
# Pinned stack: python 3.10 / jax 0.6.2 CPU (JAX_PLATFORMS=cpu) / flax
# 0.10.7 / numpy 2.2.6 / scipy 1.15.3.  Prints tagged [expc1_cpu64].
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
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import expc1_cal1_archa1 as A    # noqa: E402  (numpy-only at module level)

TAG = '[expc1_cpu64]'
STAGE = 'expc1_cpu64_companions'
SCHEMA = 'expc1-cpu64-companions-v1'
RUN_ID = 'expc1-20260729'

# ------------------------------------------------------------------- pins
N_EVAL_DEFAULT = 4096                     # tab3_4096 convention
N_EVAL_SMOKE = 256
PARTS_ALL = ('std', 'lc', 'arch2', 'mlplc', 'precb')

CELL = 'exact64_cpu_highest'
CELL_ROLE = ('CPU-f64 convention == STAT-04 cpu64 deliverable branch '
             '(c10_tableiii_cpu64.py:33-36,509-518,575-585 via '
             'expc1_cal1_archa1): jax_enable_x64 before the first jax op, '
             'float64 exact-ED features regenerated from the pinned labels, '
             'float64 params, jax.default_matmul_precision("highest") baked '
             'at trace time, float64 err_q/err_f scoring.  This is a '
             'DIFFERENT convention from the legacy reg_bytes__tpu_default '
             'deciding leg; its prod medians are computed fresh in this run '
             'and the legacy pins are never reused as denominators.')

PROD_TRIPLET = ('prod_thermal_random', 'abl_gram_s43', 'abl_gram_s44')
STD_MODELS = ('std_thermal_random', 'std_thermal_random_s43',
              'std_thermal_random_s44')
ARCH2_MODELS = ('arch2_thermal_random', 'arch2_thermal_random_s43',
                'arch2_thermal_random_s44')
OGN_LC_MODELS = ('ogn_lc_1e4', 'ogn_lc_1e5', 'ogn_lc_1e6')
MLP_LC_MODELS = ('mlp_lc_1e4', 'mlp_lc_1e5', 'mlp_lc_1e6')
PRECB_MODELS = ('precb_thermal_random', 'precb_thermal_random_s43',
                'precb_thermal_random_s44')

SEED_OF = {'std_thermal_random': 42, 'std_thermal_random_s43': 43,
           'std_thermal_random_s44': 44,
           'arch2_thermal_random': 42, 'arch2_thermal_random_s43': 43,
           'arch2_thermal_random_s44': 44,
           'precb_thermal_random': 42, 'precb_thermal_random_s43': 43,
           'precb_thermal_random_s44': 44}
PROD_OF = {m: PROD_TRIPLET[SEED_OF[m] - 42]
           for m in STD_MODELS + ARCH2_MODELS + PRECB_MODELS}

# checkpoint pins NOT already in A.CKPTS (sources in the header block).
# kind/res feed expc1_cal1_archa1.build_state; 'ogn_std' uses the local
# _build_std_model instead.
CKPTS_NEW = {
    'std_thermal_random': dict(
        sha='f47bd8faf7f71f4e1d6d1ce82dbfb99d8ceee3cc9be142c60da5d7085d2a5ac9',
        bytes=213094675, kind='ogn_std', res=3, init_seed=42,
        n_params=17756929,
        config_json_sha='c8047de12f69e4a0a57a0f3736dd1d5c77be3c10e0e19dfe0b'
                        '1ce176c321ce4d'),
    'std_thermal_random_s43': dict(
        sha='f975b0bc235888ed42267d2808955b4f4e7bb00677475f6fc22920d886ca34fd',
        bytes=213094675, kind='ogn_std', res=3, init_seed=43,
        n_params=17756929,
        config_json_sha='a008ff6619c3c4cbd080f317c930493aaa137593046223de47'
                        'c33f305e2aa390'),
    'std_thermal_random_s44': dict(
        sha='b9188271edcbb7adf11a9b4609351d863832762593dfcdbc1df96d0c189fb470',
        bytes=213094675, kind='ogn_std', res=3, init_seed=44,
        n_params=17756929,
        config_json_sha='9a0e2e4a94a0c3de8e2bc8de2113a3361b9445ea13f0d43d4e'
                        '837eb2ab82790a'),
    'arch2_thermal_random': dict(
        sha='d31540e83ca557a4fb37f96442b982be7523632669def56056937c63b0ad26e8',
        bytes=94882579, kind='ogn', res=2, init_seed=42, n_params=7905921),
    'arch2_thermal_random_s43': dict(
        sha='110599a8715a9d2294c68cda8ebc3334c253d05dc4f2fd3b1b58bbab65e1ea07',
        bytes=94882579, kind='ogn', res=2, init_seed=43, n_params=7905921),
    'arch2_thermal_random_s44': dict(
        sha='2e4dc8519f019d087538c8c4a1ab5438250f47d793b8ea852c6e495a828aeb72',
        bytes=94882579, kind='ogn', res=2, init_seed=44, n_params=7905921),
    'ogn_lc_1e4': dict(
        sha='6434d70e92e418ee1302762611bea46bf77663398a6a3329edad2b7f880a9f82',
        bytes=213094675, kind='ogn', res=3, init_seed=42, n_params=17756929),
    'ogn_lc_1e5': dict(
        sha='4fdecb5ca32b6062c62372074ece571761ad7e13f787782d1d01f8e3ea940c26',
        bytes=213094675, kind='ogn', res=3, init_seed=42, n_params=17756929),
    'ogn_lc_1e6': dict(
        sha='b264a7a158fce9f5bb9b9dba14f4e9cdc99469a819c88945211b72f21b5a45db',
        bytes=213094675, kind='ogn', res=3, init_seed=42, n_params=17756929),
    'mlp_lc_1e4': dict(
        sha='fb069deb477aefb34a9391358904f7c8cfd29e75bdd6e6aea3b1e27acecd41c3',
        bytes=209173414, kind='mlp', res=4, init_seed=42, n_params=17430599),
    'mlp_lc_1e5': dict(
        sha='2c5614c4b33384620fcc7e6a0875f495fddb1c1c4d3ea5ea23c531ece96fe3d4',
        bytes=209173414, kind='mlp', res=4, init_seed=42, n_params=17430599),
    'mlp_lc_1e6': dict(
        sha='5ffacdf4298e66f0e394c7e9b44cdd0aadb2e7874a2913bac5b42005e848ea85',
        bytes=209173414, kind='mlp', res=4, init_seed=42, n_params=17430599),
    # precb triple: PLAIN production OGN retrained on the prechigh cache
    # (spec standardize=False prechigh=True — the treatment is TRAINING
    # DATA only).  Pins = the harvested training receipts
    # results/T-precb_thermal_random{,_s43,_s44}/T-precb_*_result.json,
    # artifacts[final_state.msgpack].{sha256,bytes}; restore path = the
    # plain prod path (kind 'ogn', A.build_state), NO std stats.
    'precb_thermal_random': dict(
        sha='bdc320ebc0b8547a77db0d30dcda71cd68b18749b7b51a89cf07e8059b2a0f14',
        bytes=213094675, kind='ogn', res=3, init_seed=42, n_params=17756929),
    'precb_thermal_random_s43': dict(
        sha='371986b7916a1948841e0016b19f103c876819da4394e4af5caf71885b083efa',
        bytes=213094675, kind='ogn', res=3, init_seed=43, n_params=17756929),
    'precb_thermal_random_s44': dict(
        sha='865ce8b9590bfcae606e60b42bc7364327c4e4da02ca5da0f816fe3859e3a19e',
        bytes=213094675, kind='ogn', res=3, init_seed=44, n_params=17756929),
}
# std_stats.npz pin (SAME file bytes for all three std dirs — W2 records)
STD_STATS_FILE_SHA = ('372e7ca78cd26ff2706e187ae7750b2e5a680acdb1e5f74a2d6c'
                      '50c3a838e64a')
STD_MU_SHA = ('1d9b2f2c2732817c6f57b6897536f8527aa157f9fa3a9a6328908e8544fc'
              'fc25')
STD_SIGMA_SHA = ('185b481c8eaef08a2230fc03980223a5c046232c0cca42ad86c565e1a'
                 'fcd1ce9')


def ckpt_spec(name):
    """Merged pin view: A.CKPTS (r20 rows: prod triplet + abl_mlp) first,
    then the wave-1/2 + LC pins above."""
    if name in A.CKPTS:
        return A.CKPTS[name]
    return CKPTS_NEW[name]


# part -> (required models, optional models); prod is required by every
# comparing part (its fresh CPU-f64 medians are the denominators)
PART_MODELS = {
    'std': dict(required=STD_MODELS + PROD_TRIPLET, optional=()),
    'lc': dict(required=OGN_LC_MODELS + ('prod_thermal_random',),
               optional=()),
    'arch2': dict(required=ARCH2_MODELS + PROD_TRIPLET, optional=()),
    'mlplc': dict(required=(), optional=MLP_LC_MODELS + ('abl_mlp',)),
    'precb': dict(required=PRECB_MODELS + PROD_TRIPLET, optional=()),
}


def models_required(parts):
    """Union of the REQUIRED model sets of the selected parts — the D2
    anchor-scoping authority (Opus verify 2026-07-30): gate C5 demands only
    the prod anchors a requested part actually needs ('lc' alone needs only
    prod_thermal_random; std/arch2/precb need the full triplet)."""
    needed = set()
    for p in parts:
        needed.update(PART_MODELS.get(p, {}).get('required', ()))
    return needed

N_TRAIN = {'ogn_lc_1e4': 10_000, 'ogn_lc_1e5': 100_000,
           'ogn_lc_1e6': 1_000_000, 'prod_thermal_random': 5_000_000,
           'mlp_lc_1e4': 10_000, 'mlp_lc_1e5': 100_000,
           'mlp_lc_1e6': 1_000_000, 'abl_mlp': 5_000_000}
OGN_CURVE_SEGMENTS = (('ogn_lc_1e4', 'ogn_lc_1e5'),
                      ('ogn_lc_1e5', 'ogn_lc_1e6'),
                      ('ogn_lc_1e6', 'prod_thermal_random'),
                      ('ogn_lc_1e4', 'prod_thermal_random'))
MLP_CURVE_SEGMENTS = (('mlp_lc_1e4', 'mlp_lc_1e5'),
                      ('mlp_lc_1e5', 'mlp_lc_1e6'),
                      ('mlp_lc_1e6', 'abl_mlp'),
                      ('mlp_lc_1e4', 'abl_mlp'))

# ---- G-STD companion pins (FINAL_PLAN §3 GT-1 + §3 rule 4 + task ruling)
MAGNITUDE_THRESHOLD_PCT = 16.0    # 2 x 7.78% difference-SE (§3 rule 4)
CPU64_SEED_FLOOR_NOTE = (
    'CPU-f64 seed floor (FINAL_PLAN §2.4 / §3 rule 4): per-seed rel. std '
    '9.53%, SE of a 3-seed mean 5.50%, SE of a difference of two 3-seed '
    'means 7.78%, 95% half-width ±16–22% (registered triplet '
    '1.02/1.05/0.875e-3, main.tex:478-480).  Companion leg is therefore '
    'DIRECTION/SIGN-consistency only; magnitude claimed only at >= 16%.')
VERDICT_CONSISTENT = 'sign-consistent (3/3 improve)'
ADOPT_STATUS_PASS = 'ADOPT (deciding leg PASS + companion sign-consistent)'
ADOPT_STATUS_FAIL = 'legacy-only (companion failed)'

# ---- deciding-leg pins (registered record; the task's source of record)
W2_STD_RECORD = dict(
    record='experiments/experiment-campaign1/results/W2-rescore-std/'
           'W2-rescore-std_result.json',
    record_sha256='8bda6187259a5701a90ab3a9d18e1585991353677fbdb97440c19978'
                  '0233203b',
    decision='pass-deciding-leg (CPU-f64 companion pending)',
    branch='pass_deciding_leg',
    convention='legacy reg_bytes__tpu_default (TPU bf16-MXU default matmul, '
               'f64 scoring)',
    mean_improvement_pct=5.736963554974229,
    per_seed_improvement_pct={'42': 4.43007120302672,
                              '43': 6.4193436004406195,
                              '44': 6.361475861455346},
    threshold_pct=5.0,
    note='deciding leg of gate G-STD (GT-1): legacy 3-seed mean paired '
         'dev-1007 median improvement > 5% — REGISTERED PASS; corroborated '
         'bit-identically by W2-rescore-full (9d0d417a...f6a2f1)')

# ---- registered legacy context pins (quoted, never re-decided here)
LC_RESCORE_CONTEXT = dict(
    record='experiments/experiment-campaign1/results/LC-rescore/'
           'LC-rescore_result.json',
    record_sha256='ba15f77151479296497851b22d0fba5b3db85546c94fcb98ef6e288c'
                  '2e7e21f3',
    g1_branch='kill_null',
    g1_decision='data-saturation null, kill DATA-5e7',
    s_per_decade_legacy=0.004760660411998591,
    medians_legacy={'ogn_lc_1e4': 0.00634603316158476,
                    'ogn_lc_1e5': 0.004276121545646756,
                    'ogn_lc_1e6': 0.003804190139440392,
                    'prod_thermal_random': 0.003775153862538385})
W2_FULL_CONTEXT = dict(
    record='experiments/experiment-campaign1/results/W2-rescore-full/'
           'W2-rescore-full_result.json',
    record_sha256='9d0d417a4621166fe75b313dd08134b5a70c224ff05e12ca30b9b2ae'
                  'c4f6a2f1',
    g_a2_branch='bounded',
    g_a2_mean_deficit_pct=-2.062549397368777,
    g_a2_ci95_pct=[-2.3874199440636588, -1.6279980119138113])

# ---- part precb interpretation pins (QUOTED registered records, never
#      re-decided here).  Shas recomputed 2026-07-30 on the harvested,
#      sealed read-only copies (the task's pin-by-reading ruling).
PRECB_QUESTION = 'does clean-feature training express under clean eval?'
PB_RESCORE_NULL = dict(
    record='experiments/experiment-campaign1/results/PB-rescore/'
           'PB-rescore_result.json',
    record_sha256='a59a2406ebdf471767fb7dc838637a0b431d0d4847415612abb8f9fc'
                  '9178042c',
    gate='G-PB (R17) deciding leg',
    decision='fail (legacy deciding leg <= 5%)',
    branch='fail',
    convention='legacy reg_bytes__tpu_default (registered val-cache bytes, '
               'TPU bf16-MXU default matmul, f64 scoring) — clean-TRAINED '
               'models under the UNCHANGED legacy (bf16-feature) eval',
    mean_improvement_pct=-0.5656106410814635,
    per_seed_improvement_pct={'42': -2.241005731792778,
                              '43': 1.963141188621731,
                              '44': -1.4189673800733438},
    ci95_mean_improvement_pct=[-0.9523213796966412, -0.17367214447506032],
    threshold_pct=5.0,
    note='REGISTERED NULL (EXECUTION_LOG 2026-07-30 evening): clean-feature '
         'TRAINING does not move the legacy floor — the floor is EVAL-time '
         'feature precision (coheres with PREC-A/PREC-C).  The G-PB record '
         'itself names THIS part as the discriminating follow-up '
         '(g_pb.cpu_f64_companion.producer).')
PB_DATA_DRAW_CAVEAT = (
    'REGISTERED per-fleet-stream caveat (rides every precb claim; '
    'PB-rescore g_pb.data_draw_caveat / PRECB-gen record a3afe701...): the '
    'prechigh TRAINING cache labels are NOT bit-identical to the v4-staged '
    'registered cache — the known CH1-F02/F03 cross-env stream property, '
    'carried by ALL cross-fleet retrains and statistically immaterial; the '
    'precb arms trained on a same-distribution, not-bit-identical data '
    'draw vs the prod anchors\' registered cache.  The eval stream HERE is '
    'unaffected: both sides of every seed pair score the SAME pinned '
    'dev-1007 rows under the SAME cpu-f64 features.')
PRECA_FEATURE_TERM = dict(
    record='experiments/experiment-campaign1/results/PRECA-tpu/'
           'PRECA-tpu_result.json',
    record_sha256='ecc80f428173a6e033e4f27787826a6f73fe3c36787d60e921bed30c'
                  '7d260123',
    field='decomposition_pairs.feature_term_tpu_highest_vs_default_regen'
          '.per_model[<m>].ratio',
    feature_term_ratio={'prod_thermal_random': 0.35568876155641316,
                        'abl_gram_s43': 0.3839971563927573,
                        'abl_gram_s44': 0.40474252326153265},
    implied_eval_time_improvement_pct={
        'prod_thermal_random': 64.43112384435868,
        'abl_gram_s43': 61.60028436072427,
        'abl_gram_s44': 59.52574767384673},
    meaning='PREC-A measured the EVAL-TIME feature term at FIXED prod '
            'checkpoints: regenerating the EVAL features at matmul highest '
            'cut the legacy error to ratio ~0.356/0.384/0.405 (~x2.6 gain) '
            'with the forward term ~1.04.  The cpu-f64 exact-ED features '
            'scored here are the fully clean corner of that axis.')
PRECB_2X2 = dict(
    train_bf16__eval_bf16='legacy anchor (registered prod triplet legacy '
                          'medians 3.775/3.843/3.852e-3)',
    train_bf16__eval_clean='PREC-A eval-time feature term (ratios '
                           '0.356/0.384/0.405 at fixed prod checkpoints)',
    train_clean__eval_bf16='G-PB registered null (PB-rescore: mean -0.57%, '
                           'inside seed noise)',
    train_clean__eval_clean='THIS PART: precb triple vs prod triplet, both '
                            'fresh under the cpu-f64 exact-feature '
                            'convention on the same pinned rows')

# ---- registered CPU anchors for the convention band C5 (full n only)
# (a) ARCH-A1 registered CPU medians, rh_c3 f32-pipeline convention, SAME
#     tab3_4096 rows (results/CAL1-ARCHA1/expc1_part_archa1.json,
#     record sha 55b4e1c0...4036 for the merged result)
ANCHOR_ARCHA1 = {'prod_thermal_random': 0.0010136777667831495,
                 'abl_gram_s43': 0.0010506710719933132,
                 'abl_gram_s44': 0.0008749332704580268}
# (b) CAL-1 N_s=1e10 medians, SAME cpu64 convention family (exact-f64
#     features + f64 forward), 100 H x 20 near-clean draws
ANCHOR_CAL1_1E10 = {'prod_thermal_random': 0.0010804238482370174,
                    'abl_gram_s43': 0.0011102796219013924,
                    'abl_gram_s44': 0.0009132565572228502}
ANCHOR_BAND = (0.85, 1.15)
CAL1_ARCHA1_RECORD_SHA = ('55b4e1c0bb8cc3bae0007c38e6f719fd7679c81b4fbb79eb'
                          'e74ad432d4764036')

# ---- pinned stream (labels) authorities
LABELS_ROWS_SHA_PIN = ('94f616fb7c5d6b725fc0ba6bf91da346ca4498d901925ff854a'
                       '090e1620ac011')     # f32 rows 0..4096, cache order
VAL_SHARD_SHA_PIN = ('de073e7412b1af10e3422a74d088b4427109d6c6bafdda4be6658'
                     'ab119ac542e')          # shard_00000.npz, 1,756,449 B
LABELS_NPZ_PINS = {
    'e215e0fa2e6c44d4992cf314f39ae662a0d5e662cfbb3305983b56356a2ea900':
        'campaign5/results/boot95_inputs/inputs/'
        'prod_thermal_random_predictions_val.npz',
    'fbbb4ce2dde3e584152dfaf82a1eea58a5cafee35fbea88445a36e963e92136a':
        'campaign8/inputs/'
        'prod_thermal_random_predictions_val_first4096.npz',
}

MANIFEST_NAME = 'MANIFEST.sha256'
RESULT_NAME = '%s_result.json' % STAGE
PERSAMPLE_NAME = '%s_persample.npz' % STAGE


# ------------------------------------------------------ decision arithmetic
def improvement_pct(med_new, med_prod):
    """100*(1 - median_new/median_prod); positive = new arm BETTER — the R4
    orientation (expc1_wave2_rescore.py:260-263 verbatim)."""
    return 100.0 * (1.0 - float(med_new) / float(med_prod))


def deficit_pct(med_new, med_prod):
    """100*(median_new/median_prod - 1); positive = new arm WORSE — the R11
    orientation (expc1_wave2_rescore.py:254-257 verbatim)."""
    return 100.0 * (float(med_new) / float(med_prod) - 1.0)


def companion_verdict(improve_flags):
    """GT-1 companion leg verdict over the three seed-paired direction
    flags.  Task-pinned strings."""
    flags = [bool(f) for f in improve_flags]
    if len(flags) != 3:
        raise ValueError('companion verdict needs exactly 3 seed flags')
    k = sum(flags)
    if k == 3:
        return VERDICT_CONSISTENT, True
    return 'sign-inconsistent (%d/3)' % k, False


def adoption_status(sign_consistent, deciding_leg_pass=True):
    """G-STD adoption composition (GT-1: headline adoption requires BOTH
    legs).  The deciding leg is the REGISTERED W2-rescore-std PASS (pinned);
    the parameter stays explicit so the logic is total and testable."""
    if not deciding_leg_pass:
        return 'legacy-only (deciding leg failed)'      # unreachable w/ pins
    return ADOPT_STATUS_PASS if sign_consistent else ADOPT_STATUS_FAIL


def magnitude_flag(impr_pcts, sign_consistent):
    """CPU-f64 magnitude claimable ONLY if all three seed-paired
    improvements >= 16% (GT-1 / §3 rule 4; task ruling 'all >= 16%')
    AND the direction is consistent."""
    vals = [float(v) for v in impr_pcts]
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        return False
    return bool(sign_consistent
                and all(v >= MAGNITUDE_THRESHOLD_PCT for v in vals))


def boot_ratio_ci_paired(num, den, seed=A.BOOT_SEED, B=A.BOOT_B):
    """Draw-paired bootstrap of median(num)/median(den) — one SHARED index
    draw per replicate (COPY of expc1_preca.py:395-410; valid because both
    arrays score the SAME pinned rows).  Pins B=1000, default_rng(20260717),
    order stats 25/975."""
    a = np.asarray(num, np.float64)
    b = np.asarray(den, np.float64)
    if a.shape != b.shape:
        raise ValueError('paired ratio bootstrap needs equal-length arrays')
    n = a.size
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        reps[k] = np.median(a[idx]) / np.median(b[idx])
    reps.sort()
    return [float(reps[A.BOOT_LO]), float(reps[A.BOOT_HI])]


def boot_mean_improvement_ci_paired(new_arrs, prod_arrs,
                                    seed=A.BOOT_SEED, B=A.BOOT_B):
    """Draw-paired per-Hamiltonian cluster bootstrap of the 3-seed MEAN
    paired improvement_pct: one SHARED index draw per replicate across ALL
    six arrays (COPY of expc1_wave2_rescore.py:290-320
    _boot_mean_deficit_ci_paired, sign-flipped to the R4 improvement
    orientation; here BOTH sides are fresh same-run arrays, so the CI is
    fully paired by construction).  Captures eval-draw noise on the pinned
    rows only; the seed floor (SE 7.78%) is quoted alongside, never inferred
    from this CI."""
    if len(new_arrs) != len(prod_arrs) or not new_arrs:
        raise ValueError('paired mean-improvement bootstrap needs matched, '
                         'non-empty seed lists')
    new_arrs = [np.asarray(x, np.float64) for x in new_arrs]
    prod_arrs = [np.asarray(x, np.float64) for x in prod_arrs]
    n = new_arrs[0].size
    if any(x.shape != (n,) for x in new_arrs + prod_arrs):
        raise ValueError('paired mean-improvement bootstrap needs '
                         'equal-length 1-D arrays')
    k_seeds = len(new_arrs)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        d = 0.0
        for s in range(k_seeds):
            d += 100.0 * (1.0 - np.median(new_arrs[s][idx])
                          / np.median(prod_arrs[s][idx]))
        reps[b] = d / k_seeds
    reps.sort()
    return [float(reps[A.BOOT_LO]), float(reps[A.BOOT_HI])]


def slope_per_decade(med_lo_n, med_hi_n, n_lo, n_hi):
    """s = log10(median(n_lo)/median(n_hi)) / log10(n_hi/n_lo) — COPY of
    expc1_lc_rescore.py:161-166 (FINAL_PLAN R3 formula)."""
    return (math.log10(float(med_lo_n) / float(med_hi_n))
            / math.log10(float(n_hi) / float(n_lo)))


def boot_slope_ci_paired(err_lo_n, err_hi_n, n_lo, n_hi,
                         seed=A.BOOT_SEED, B=A.BOOT_B):
    """Draw-paired bootstrap of the per-decade slope — COPY of
    expc1_lc_rescore.py:169-189 (_boot_slope_ci_paired)."""
    a = np.asarray(err_lo_n, np.float64)
    b = np.asarray(err_hi_n, np.float64)
    if a.shape != b.shape:
        raise ValueError('paired slope bootstrap needs equal-length arrays')
    dec = math.log10(float(n_hi) / float(n_lo))
    n = a.size
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        reps[k] = math.log10(np.median(a[idx]) / np.median(b[idx])) / dec
    reps.sort()
    return [float(reps[A.BOOT_LO]), float(reps[A.BOOT_HI])]


# ------------------------------------------------- f64 exact-ED features
def features_from_VE(V, E, ops_flat, beta=A.BETA_THERMAL):
    """CLEAN exact-f64 feature pair for one Hamiltonian from its f64
    eigensystem: thermal p_true, rho_true=(V*p)@V.T, pair-block contraction
    einsum('ji,kij->k') == compute_rho_m (p1_core.py:645-668) WITHOUT its
    float32 cast, E = sum(p*E) — the CAL-1 worker
    (expc1_cal1_archa1.cal1_features_for_H, :563-586 /
    c10_tableiii_cpu64.py:339-371 f64 branch) at p_meas == p_true (no
    drawer: the clean point of the STAT-04 grid)."""
    E = np.asarray(E, np.float64)
    V = np.asarray(V, np.float64)
    p_true = np.exp(-float(beta) * (E - E.min()))
    p_true = p_true / np.sum(p_true)
    rho_true = (V * p_true) @ V.T
    bx = np.einsum('ji,kij->k', rho_true, ops_flat,
                   optimize=True).reshape(A.MP, A.MP)
    be = float(np.sum(p_true * E))
    return bx, be


_POOL = {}


def _pool_init(h_path, ops_path):
    """Spawn-pool initializer: numpy-only workers, memmapped inputs (the
    expc1_cal1_archa1 spawn discipline, :555-586 / :962-973 — NEVER fork
    the jax-holding process)."""
    import scipy.linalg  # noqa: F401  (workers need LAPACK eigh)
    _POOL['H'] = np.load(h_path, mmap_mode='r')
    _POOL['ops'] = np.asarray(np.load(ops_path, mmap_mode='r'))
    _POOL['scipy_linalg'] = scipy.linalg


def _worker_features(si):
    """One dev-1007 row: f64 eigh (scipy.linalg.eigh — the CAL-1 parent
    solver, expc1_cal1_archa1.py:941-946) + clean features."""
    sl = _POOL['scipy_linalg']
    H = np.asarray(_POOL['H'][si], np.float64)
    E, V = sl.eigh(H)
    bx, be = features_from_VE(V, E, _POOL['ops'])
    return si, bx, be


def build_features_exact64(eng, Gpanel, out_dir, nproc):
    """All-rows f64 exact-ED features.  Parent assembles the dense f64
    Hamiltonians (eng._shots_h_dense64, the CAL-1 path) into a memmap; a
    spawn pool does eigh + contraction; row 0 and row n-1 are recomputed
    serially for the byte-identity gate (C6)."""
    import scipy.linalg
    n = len(Gpanel)
    D_N = int(eng.basis.size)
    h_path = os.path.join(out_dir, '_expc1_h64.npy')
    ops_path = os.path.join(out_dir, '_expc1_ops64.npy')
    H_all = np.lib.format.open_memmap(h_path, mode='w+', dtype=np.float64,
                                      shape=(n, D_N, D_N))
    for i in range(n):
        H_all[i] = eng._shots_h_dense64(Gpanel[i])
    H_all.flush()
    rho2_dense = eng._safe_dense(eng.rho_2_kkbar_arrays)
    ops_flat = np.asarray(rho2_dense, np.float64).reshape(
        -1, rho2_dense.shape[-1], rho2_dense.shape[-1])
    np.save(ops_path, ops_flat)

    BX = np.empty((n, A.MP, A.MP), np.float64)
    BE = np.empty((n,), np.float64)
    print('%s building f64 exact-ED features: n=%d (eigh %dx%d, nproc=%d)'
          % (TAG, n, D_N, D_N, nproc), flush=True)
    if nproc > 1:
        import multiprocessing as mp_mod
        ctx = mp_mod.get_context('spawn')
        with ctx.Pool(processes=nproc, initializer=_pool_init,
                      initargs=(h_path, ops_path)) as pool:
            for si, bx, be in pool.imap_unordered(_worker_features,
                                                  range(n), chunksize=8):
                BX[si], BE[si] = bx, be
    else:
        _pool_init(h_path, ops_path)
        for si in range(n):
            _si, bx, be = _worker_features(si)
            BX[si], BE[si] = bx, be

    # serial identity recompute (pool determinism half of C6)
    ident = True
    for si in (0, n - 1):
        E, V = scipy.linalg.eigh(np.asarray(H_all[si], np.float64))
        bx, be = features_from_VE(V, E, ops_flat)
        ident = ident and bool(np.array_equal(bx, BX[si])
                               and be == BE[si])
    del H_all
    for p in (h_path, ops_path):
        try:
            os.remove(p)
        except OSError:
            pass
    return BX, BE, bool(ident)


# ------------------------------------------------------ std model machinery
def _load_std_stats(ckpt_dir):
    """Load + verify the STD-1 standardization stats from <ckpt_dir>/
    std_stats.npz (written by t_std1.run_std, t_std1.py:692-703) and the
    provenance config.json std block (t_std1.py:788-802).  VERBATIM COPY of
    expc1_wave2_rescore.py:1172-1235 (P.* -> A.* only).  Malformed or
    digest-inconsistent stats are a HARD error — a wrong-stats build
    silently scores the wrong network."""
    npz_path = os.path.join(ckpt_dir, 'std_stats.npz')
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    if not os.path.isfile(npz_path):
        raise SystemExit('%s std_stats.npz missing in %s (standardization '
                         'stats required to rebuild the network)'
                         % (TAG, ckpt_dir))
    if not os.path.isfile(cfg_path):
        raise SystemExit('%s config.json missing in %s (STD-1 provenance '
                         'required)' % (TAG, ckpt_dir))
    z = np.load(npz_path, allow_pickle=False)
    mu = np.asarray(z['mu'], np.float32)
    sigma = np.asarray(z['sigma'], np.float32)
    if mu.shape != (A.MP, A.MP) or sigma.shape != (A.MP, A.MP):
        raise SystemExit('%s std stats wrong shape in %s: mu %r sigma %r '
                         '(want (%d,%d))' % (TAG, npz_path, mu.shape,
                                             sigma.shape, A.MP, A.MP))
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
            and float(sigma.min()) > 0.0):
        raise SystemExit('%s std stats non-finite/non-positive in %s'
                         % (TAG, npz_path))
    if not (np.array_equal(mu, mu.T) and np.array_equal(sigma, sigma.T)):
        raise SystemExit('%s std stats not exactly symmetric in %s (breaks '
                         'the symmetrize-first convention, t_std1.py:620)'
                         % (TAG, npz_path))
    mu_sha = hashlib.sha256(mu.tobytes()).hexdigest()
    sig_sha = hashlib.sha256(sigma.tobytes()).hexdigest()
    stored_mu = str(z['mu_sha256']) if 'mu_sha256' in z.files else None
    stored_sig = str(z['sigma_sha256']) if 'sigma_sha256' in z.files else None
    if stored_mu != mu_sha or stored_sig != sig_sha:
        raise SystemExit('%s std stats digest mismatch in %s (recomputed '
                         'mu %s / sigma %s vs stored %s / %s) — corrupted '
                         'stats' % (TAG, npz_path, mu_sha[:16], sig_sha[:16],
                                    str(stored_mu)[:16], str(stored_sig)[:16]))
    with open(cfg_path) as f:
        cfg = json.load(f) or {}
    cfg_std = (cfg.get('std') or {})
    if cfg_std.get('mu_sha256') != mu_sha or \
            cfg_std.get('sigma_sha256') != sig_sha:
        raise SystemExit('%s config.json std digests disagree with '
                         'std_stats.npz in %s — polluted checkpoint dir'
                         % (TAG, ckpt_dir))
    info = dict(
        stats_file=npz_path, stats_file_sha256=A._sha256_file(npz_path),
        config_json_sha256=A._sha256_file(cfg_path),
        arch_class=cfg.get('arch_class'),
        mu_sha256=mu_sha, sigma_sha256=sig_sha,
        n_stats_samples=(int(z['n_stats_samples'])
                         if 'n_stats_samples' in z.files else None),
        sigma_floor=(float(z['sigma_floor'])
                     if 'sigma_floor' in z.files else None),
        sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
        applied_at=(list(map(str, z['applied_at']))
                    if 'applied_at' in z.files else None),
        applied_at_expected=['node_init', 'edge_init', 'reinjection'],
        stats_file_sha_matches_pin=bool(
            A._sha256_file(npz_path) == STD_STATS_FILE_SHA),
        mu_sha_matches_pin=bool(mu_sha == STD_MU_SHA),
        sigma_sha_matches_pin=bool(sig_sha == STD_SIGMA_SHA))
    return mu, sigma, info


def _build_std_model(engine, label_size, spec, mu, sigma):
    """StdPhysicsOrbitalGraphNet — VERBATIM copy of
    expc1_wave2_rescore.py:1238-1361, itself the verbatim relay of
    t_std1.py:249-360 (the production PhysicsOrbitalGraphNet __call__,
    p2_models5.py:290-361, with the single [STD-1] standardization stanza +
    three x_sym -> x_std reads); MatrixInteractionBlock REUSED from the
    engine namespace (never copied), so the re-injection is standardized via
    bare_rdm=x_std.  mu/sigma ride as nested-tuple attributes exactly as
    t_std1._set_std_stats encodes them (hashable -> jit-static safe; NOT
    trainable params, so the params pytree and msgpack layout are
    byte-identical to the production OGN)."""
    import jax.numpy as jnp
    import flax.linen as nn

    MIB = engine.ns['MatrixInteractionBlock']
    std_mu = tuple(tuple(float(v) for v in row)
                   for row in np.asarray(mu, np.float32))
    std_sigma = tuple(tuple(float(v) for v in row)
                      for row in np.asarray(sigma, np.float32))

    class StdPhysicsOrbitalGraphNet(nn.Module):
        label_size: int
        res: int = 3
        include_energy: bool = True
        use_scatter: bool = True
        use_reinject: bool = True
        use_orb_emb: bool = True
        readout_bias: float = 0.55
        use_energy_input: bool = True
        std_mu: tuple = ()
        std_sigma: tuple = ()

        @nn.compact
        def __call__(self, x, energy=None, training: bool = True):
            b, m, _, _ = x.shape
            x_mat = x.squeeze(-1)

            num_triu = (m * (m + 1)) // 2
            htype_random = (self.label_size == num_triu)

            # Strict Symmetrization of the input        [= p2_models5.py:299]
            x_sym = 0.5 * (x_mat + jnp.swapaxes(x_mat, 1, 2))

            # ---- [STD-1] per-entry affine standardization (t_std1.py:290-298)
            assert len(self.std_mu) == m and len(self.std_sigma) == m, (
                '%s std stats are %dx? but m=%d'
                % (TAG, len(self.std_mu), m))
            mu_ = jnp.asarray(self.std_mu, x_sym.dtype)
            sigma_ = jnp.asarray(self.std_sigma, x_sym.dtype)
            x_std = (x_sym - mu_) / sigma_
            # -----------------------------------------------------------------

            # A. NODE INITIALIZATION
            diag_idx = jnp.arange(m)
            n_k = x_std[:, diag_idx, diag_idx][..., None]  # [STD-1 node_init] was x_sym (p2_models5.py:302-303)

            node_features = [n_k]

            if self.use_orb_emb:
                # Widen embedding slightly for richer single-particle identity
                orb_emb = self.param('orb_emb',
                                     nn.initializers.normal(stddev=0.1),
                                     (m, 32))
                orb_emb_batch = jnp.broadcast_to(orb_emb[None, :, :],
                                                 (b, m, 32))
                node_features.append(orb_emb_batch)

            # energy branch verbatim (p2_models5.py:313-319); NOT standardized
            if self.include_energy and self.use_energy_input \
                    and energy is not None:
                if energy.ndim == 1:
                    energy = energy[:, None]
                e_ctx = nn.Dense(32)(energy / m)
                e_ctx = nn.gelu(e_ctx)
                e_ctx_batch = jnp.broadcast_to(e_ctx[:, None, :], (b, m, 32))
                node_features.append(e_ctx_batch)

            h = jnp.concatenate(node_features, axis=-1)
            h = nn.Dense(128 * self.res)(h)

            # B. EDGE INITIALIZATION
            e = jnp.expand_dims(x_std, -1)             # [STD-1 edge_init] was x_sym (p2_models5.py:325)
            e = nn.Dense(128 * self.res)(e)

            # C. MESSAGE PASSING (Deepened safely due to Pre-Norm)
            hidden_dim = 256 * self.res
            for _ in range(5):
                h, e = MIB(
                    hidden_dim,
                    use_scatter=self.use_scatter,
                    use_reinject=self.use_reinject,
                )(h, e, bare_rdm=x_std)                # [STD-1 reinjection] was bare_rdm=x_sym (p2_models5.py:335 -> :255-256)

            # D. PHYSICAL READOUT
            e_final = nn.LayerNorm()(e)

            out_mat = nn.Dense(
                1,
                kernel_init=nn.initializers.normal(stddev=1e-3),
                bias_init=nn.initializers.constant(self.readout_bias)
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
                    bias_init=nn.initializers.constant(self.readout_bias)
                )(x_proj)

    return StdPhysicsOrbitalGraphNet(
        label_size=label_size, res=int(spec['res']), include_energy=True,
        use_scatter=spec.get('use_scatter', True),
        use_reinject=spec.get('use_reinject', True),
        use_orb_emb=spec.get('use_orb_emb', True),
        readout_bias=spec.get('readout_bias', 0.55),
        use_energy_input=bool(spec.get('use_energy_input', True)),
        std_mu=std_mu, std_sigma=std_sigma)


def build_state_std(eng, name, spec, raw, mu, sigma):
    """std restore: _build_std_model + the expc1_cal1_archa1.build_state
    tail (init / from_state_dict / n_params check, :679-717) — the params
    pytree is identical to the production OGN so the restore path is
    unchanged."""
    import jax
    import jax.numpy as jnp
    from flax import serialization as flax_ser
    model = _build_std_model(eng, A.LABEL55, spec, mu, sigma)
    variables = model.init(jax.random.PRNGKey(int(spec.get('init_seed', 42))),
                           jnp.zeros((1, A.MP, A.MP, 1), jnp.float32),
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


# ---------------------------------------------------------- labels loading
def load_labels(val_cache, labels_npz, n_eval):
    """Pinned seed-1007 dev labels rows 0..n_eval.  Primary: the registered
    val cache (--val-cache dir with shard_*.npz, or one shard .npz) — the
    SAME rows the legacy deciding leg consumed (expc1_preca
    load_registered_stream convention, expc1_preca.py:858-956; rows-sha pin
    94f616fb...).  Fallback: a registered predictions npz (g_true rows; rh_c3
    REC val_npz convention) — same stream to <1e-6 (rg_h9 G1 precedent),
    recorded loudly as npz_fallback."""
    import glob as _glob
    meta = dict(n_eval=int(n_eval))
    labels = None
    if val_cache:
        vc = os.path.expanduser(val_cache)
        shards = ([vc] if os.path.isfile(vc)
                  else sorted(_glob.glob(os.path.join(vc, 'shard_*.npz'))))
        rows, files, got = [], [], 0
        for sp in shards:
            if got >= n_eval:
                break
            files.append(dict(name=os.path.basename(sp),
                              bytes=os.path.getsize(sp),
                              sha256=A._sha256_file(sp)))
            with np.load(sp) as z:
                Y = np.asarray(z['labels'], np.float32)
            take = min(n_eval - got, len(Y))
            rows.append(Y[:take])
            got += take
        if got >= n_eval:
            labels = np.ascontiguousarray(np.concatenate(rows, 0)[:n_eval])
            meta['source'] = 'val_cache_shards'
            meta['shard_files'] = files
            meta['shard_sha_pin'] = VAL_SHARD_SHA_PIN
            meta['order'] = ('sorted shard_*.npz, rows in stored order '
                             '(== NumpyLoader shuffle=False)')
        else:
            meta['val_cache_error'] = ('cache short/unreadable: %d < %d rows '
                                       'under %r' % (got, n_eval, vc))
    if labels is None and labels_npz:
        lp = os.path.expanduser(labels_npz)
        if not os.path.isfile(lp):
            raise SystemExit('%s labels npz not found: %s' % (TAG, lp))
        sha = A._sha256_file(lp)
        with np.load(lp) as z:
            g_true = np.asarray(z['g_true'], np.float32)
        if len(g_true) < n_eval:
            raise SystemExit('%s labels npz %s short (%d < %d)'
                             % (TAG, lp, len(g_true), n_eval))
        labels = np.ascontiguousarray(g_true[:n_eval])
        meta['source'] = 'npz_fallback'
        meta['labels_npz'] = dict(path=lp, sha256=sha,
                                  sha_known_registered=bool(
                                      sha in LABELS_NPZ_PINS),
                                  registered_as=LABELS_NPZ_PINS.get(sha))
        print('%s NOTE: labels from npz fallback %s (registered=%s) — '
              'cache rows preferred; same stream to <1e-6 (rg_h9 G1)'
              % (TAG, lp, meta['labels_npz']['sha_known_registered']),
              flush=True)
    if labels is None:
        raise SystemExit('%s no label source: pass --val-cache (registered '
                         'val cache shards) or --labels-npz (registered '
                         'predictions npz)' % TAG)
    meta['labels_rows_sha256'] = A._arr_sha(labels)
    meta['labels_rows_sha_pin_full4096'] = LABELS_ROWS_SHA_PIN
    meta['labels_rows_match_pin'] = bool(
        n_eval == N_EVAL_DEFAULT and meta['source'] == 'val_cache_shards'
        and meta['labels_rows_sha256'] == LABELS_ROWS_SHA_PIN)
    return labels, meta


# --------------------------------------------------------------- manifest
def write_manifest(out_dir, entries):
    """sha256sum-compatible manifest ('<sha>  <name>')."""
    p = os.path.join(out_dir, MANIFEST_NAME)
    with open(p, 'w') as f:
        for name, sha in entries:
            f.write('%s  %s\n' % (sha, name))
    return p


def append_manifest(out_dir, name, sha):
    with open(os.path.join(out_dir, MANIFEST_NAME), 'a') as f:
        f.write('%s  %s\n' % (sha, name))


def verify_manifest(out_dir):
    """Re-hash every listed file; (ok, rows)."""
    p = os.path.join(out_dir, MANIFEST_NAME)
    rows, ok = [], True
    if not os.path.isfile(p):
        return False, [dict(error='manifest missing')]
    with open(p) as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln.strip():
                continue
            sha, name = ln[:64], ln[66:]
            fp = os.path.join(out_dir, name)
            got = A._sha256_file(fp) if os.path.isfile(fp) else None
            good = bool(got == sha)
            rows.append(dict(name=name, listed=sha, recomputed=got, ok=good))
            ok = ok and good
    return bool(ok and rows), rows


# ----------------------------------------------------------- part builders
def _pm_row(per_model, m):
    return per_model.get(m) or {}


def build_part_std(per_model, err_store, requested_parts, w2_check):
    if 'std' not in requested_parts:
        return dict(status='pending',
                    missing=['part std not requested (--parts subset)'],
                    note='the ADOPTION BLOCKER was explicitly deselected; '
                         'no companion verdict — G-STD stays '
                         '"pass-deciding-leg (CPU-f64 companion pending)"')
    per_seed, all_ok = [], True
    for m in STD_MODELS:
        pm = PROD_OF[m]
        s_new, s_prod = _pm_row(per_model, m), _pm_row(per_model, pm)
        med_new = s_new.get('median_err_q')
        med_prod = s_prod.get('median_err_q')
        row = dict(model=m, prod_model=pm, seed=SEED_OF[m],
                   median_std_cpu64=med_new, median_prod_cpu64=med_prod)
        if (isinstance(med_new, float) and isinstance(med_prod, float)
                and math.isfinite(med_new) and math.isfinite(med_prod)
                and med_prod > 0):
            row['ratio'] = med_new / med_prod
            row['improvement_pct'] = improvement_pct(med_new, med_prod)
            row['improves'] = bool(med_new < med_prod)
            a, b = err_store.get(m), err_store.get(pm)
            if a is not None and b is not None:
                rci = boot_ratio_ci_paired(a, b)
                row['ratio_ci95'] = rci
                row['improvement_ci95_pct'] = [100.0 * (1.0 - rci[1]),
                                               100.0 * (1.0 - rci[0])]
        else:
            all_ok = False
        per_seed.append(row)
    if not all_ok or any('improves' not in r for r in per_seed):
        return dict(status='pending',
                    missing=[r['model'] for r in per_seed
                             if 'improves' not in r],
                    per_seed=per_seed,
                    note='std/prod medians incomplete — no verdict')
    verdict, consistent = companion_verdict([r['improves']
                                             for r in per_seed])
    imprs = [r['improvement_pct'] for r in per_seed]
    mean_impr = float(np.mean(imprs))
    ci = boot_mean_improvement_ci_paired(
        [err_store[m] for m in STD_MODELS],
        [err_store[PROD_OF[m]] for m in STD_MODELS])
    mag = magnitude_flag(imprs, consistent)
    status = adoption_status(consistent, deciding_leg_pass=True)
    block = dict(
        gate='G-STD (R4) — CPU-f64 COMPANION LEG (GT-1; FINAL_PLAN §3 '
             'rule 2 + rule 4, §4 R4, §6 gate row G-STD)',
        convention=CELL,
        pairing='seed-paired (42<->42 prod, 43<->43 abl_gram_s43, 44<->44 '
                'abl_gram_s44); BOTH sides scored fresh in this run under '
                'the SAME cpu-f64 convention on the SAME pinned rows',
        status='decided',
        per_seed=per_seed,
        n_improve=int(sum(r['improves'] for r in per_seed)),
        companion_verdict=verdict,
        sign_consistent=bool(consistent),
        mean_improvement_pct=mean_impr,
        mean_improvement_ci95_pct=ci,
        ci_note='draw-paired eval CI on the pinned rows (B=%d rng %d order '
                'stats %d/%d) — eval noise ONLY; the binding uncertainty is '
                'the training-seed floor quoted below'
                % (A.BOOT_B, A.BOOT_SEED, A.BOOT_LO, A.BOOT_HI),
        magnitude_threshold_pct=MAGNITUDE_THRESHOLD_PCT,
        magnitude_claimable=bool(mag),
        magnitude_note=('CPU-f64 magnitude %s: claimable iff sign-consistent '
                        'AND all three seed-paired improvements >= %.0f%% '
                        '(GT-1 / §3 rule 4); below that the magnitude is '
                        'reported with its CI, UNCLAIMED'
                        % ('CLAIMABLE' if mag else 'NOT claimable',
                           MAGNITUDE_THRESHOLD_PCT)),
        seed_floor_note=CPU64_SEED_FLOOR_NOTE,
        deciding_leg=dict(dict(W2_STD_RECORD), record_check=w2_check),
        adoption_status=status)
    print('%s G-STD companion: %s | mean %+.2f%% | magnitude_claimable=%s '
          '| adoption: %s'
          % (TAG, verdict, mean_impr, mag, status), flush=True)
    return block


def _curve(per_model, err_store, models_plus_endpoint, segments, label):
    points = [dict(model=m, n_train=N_TRAIN[m],
                   median_err_q=_pm_row(per_model, m).get('median_err_q'),
                   err_q_ci95=_pm_row(per_model, m).get('err_q_ci95'))
              for m in models_plus_endpoint if m in per_model]
    segs = []
    for lo_m, hi_m in segments:
        if lo_m not in per_model or hi_m not in per_model:
            continue
        n_lo, n_hi = N_TRAIN[lo_m], N_TRAIN[hi_m]
        med_lo = per_model[lo_m]['median_err_q']
        med_hi = per_model[hi_m]['median_err_q']
        seg = dict(frm=lo_m, to=hi_m, n_lo=n_lo, n_hi=n_hi,
                   decades=math.log10(n_hi / n_lo),
                   ratio_median=float(med_lo / med_hi),
                   s_per_decade=float(slope_per_decade(med_lo, med_hi,
                                                       n_lo, n_hi)))
        a, b = err_store.get(lo_m), err_store.get(hi_m)
        if a is not None and b is not None and len(a) == len(b):
            seg['s_ci95'] = boot_slope_ci_paired(a, b, n_lo, n_hi)
            seg['ratio_ci95'] = boot_ratio_ci_paired(a, b)
        segs.append(seg)
    return dict(label=label, points=points, segments=segs)


def build_part_lc(per_model, err_store, requested_parts):
    if 'lc' not in requested_parts:
        return dict(status='not_requested')
    have = [m for m in OGN_LC_MODELS + ('prod_thermal_random',)
            if m in per_model]
    if len(have) < 2:
        return dict(status='pending', missing=[
            m for m in OGN_LC_MODELS + ('prod_thermal_random',)
            if m not in per_model])
    curve = _curve(per_model, err_store,
                   OGN_LC_MODELS + ('prod_thermal_random',),
                   OGN_CURVE_SEGMENTS, 'OGN learning curve, CPU-f64')
    curve.update(
        status='decided',
        role='DESCRIPTIVE, NON-GATING — R3 dual-convention completion: '
             '"the CPU-f64 companion slope is recorded but non-gating at '
             'n_seeds=1 (a 17.5% step sits AT the ~17% CPU-f64 seed floor)" '
             '(FINAL_PLAN R3); G1 was decided under the LEGACY convention',
        gating=False,
        registered_legacy_g1=dict(LC_RESCORE_CONTEXT),
        seed_floor_note=CPU64_SEED_FLOOR_NOTE)
    s16 = next((s for s in curve['segments']
                if (s['frm'], s['to']) == ('ogn_lc_1e6',
                                           'prod_thermal_random')), None)
    if s16:
        print('%s LC cpu64: s(1e6->5e6)=%.4f/decade (legacy registered '
              '%.4f, branch %s) — descriptive'
              % (TAG, s16['s_per_decade'],
                 LC_RESCORE_CONTEXT['s_per_decade_legacy'],
                 LC_RESCORE_CONTEXT['g1_branch']), flush=True)
    return curve


def build_part_arch2(per_model, err_store, requested_parts):
    if 'arch2' not in requested_parts:
        return dict(status='not_requested')
    per_seed = []
    for m in ARCH2_MODELS:
        pm = PROD_OF[m]
        if m not in per_model or pm not in per_model:
            continue
        med_new = per_model[m]['median_err_q']
        med_prod = per_model[pm]['median_err_q']
        row = dict(model=m, prod_model=pm, seed=SEED_OF[m],
                   median_arch2_cpu64=med_new, median_prod_cpu64=med_prod,
                   ratio=med_new / med_prod,
                   deficit_pct=deficit_pct(med_new, med_prod),
                   res2_worse=bool(med_new > med_prod))
        a, b = err_store.get(m), err_store.get(pm)
        if a is not None and b is not None:
            rci = boot_ratio_ci_paired(a, b)
            row['deficit_ci95_pct'] = [100.0 * (rci[0] - 1.0),
                                       100.0 * (rci[1] - 1.0)]
        per_seed.append(row)
    if len(per_seed) < 3:
        return dict(status='pending',
                    missing=[m for m in ARCH2_MODELS if m not in per_model])
    mean_def = float(np.mean([r['deficit_pct'] for r in per_seed]))
    ci_impr = boot_mean_improvement_ci_paired(
        [err_store[m] for m in ARCH2_MODELS],
        [err_store[PROD_OF[m]] for m in ARCH2_MODELS])
    ci_def = [-ci_impr[1], -ci_impr[0]]     # deficit = -improvement
    block = dict(
        status='decided',
        role='DESCRIPTIVE, NON-GATING — the R11 CPU-f64 seed-paired '
             'companion (per-seed deltas "descriptive" in gate G-A2); the '
             'legacy 3v3 G-A2 decision is registered (context below).  The '
             '"saturated below 8M" question needs this leg, but the n=3 '
             'CPU-f64 CI is ±16–22% and CANNOT support a saturation claim '
             '(R11: NOT registered as "saturated below 8M")',
        gating=False,
        pairing='seed-paired (42<->42, 43<->43, 44<->44) vs the prod '
                'triplet, both sides fresh cpu-f64 in this run',
        per_seed=per_seed,
        mean_deficit_pct=mean_def,
        mean_deficit_ci95_pct=ci_def,
        ci_note='draw-paired eval CI (shared index draws; sign-flip of the '
                'mean-improvement bootstrap) — eval noise only',
        n_res2_worse=int(sum(r['res2_worse'] for r in per_seed)),
        registered_legacy_g_a2=dict(W2_FULL_CONTEXT),
        seed_floor_note=CPU64_SEED_FLOOR_NOTE)
    print('%s arch2 cpu64: mean deficit %+.2f%% (n_worse=%d/3, legacy G-A2 '
          '%s) — descriptive'
          % (TAG, mean_def, block['n_res2_worse'],
             W2_FULL_CONTEXT['g_a2_branch']), flush=True)
    return block


def build_part_mlplc(per_model, err_store, requested_parts, absent):
    if 'mlplc' not in requested_parts:
        return dict(status='not_requested')
    have = [m for m in MLP_LC_MODELS + ('abl_mlp',) if m in per_model]
    if not have:
        return dict(status='absent',
                    missing=list(absent),
                    role='registered campaign6 MLP LC checkpoints not '
                         'staged at --ckpt-root — RECORDED absent, '
                         'non-gating (task ruling)')
    curve = _curve(per_model, err_store, MLP_LC_MODELS + ('abl_mlp',),
                   MLP_CURVE_SEGMENTS, 'MLP learning curve, CPU-f64')
    curve.update(status='decided', gating=False,
                 missing=list(absent),
                 role='DESCRIPTIVE, NON-GATING — registered campaign6 MLP '
                      'LC arms (lc_train_mlp_lc_*, r20 PRV pins) under the '
                      'cpu-f64 convention; abl_mlp = the 5e6 endpoint when '
                      'staged')
    return curve


def build_part_precb(per_model, err_store, requested_parts, pb_check):
    """The precision 2x2's LAST cell (clean-TRAINED under clean EVAL) —
    the part_std machinery applied to the precb triple (the PB-rescore
    record's own named producer), DESCRIPTIVE only: seed-paired CPU-f64
    deltas + sign consistency + the quoted interpretation context.  NEVER
    emits an adoption status (G-PB is a registered legacy FAIL; there is
    no deciding leg to compose with — C7 enforces the shape)."""
    if 'precb' not in requested_parts:
        return dict(status='not_requested')
    per_seed, all_ok = [], True
    for m in PRECB_MODELS:
        pm = PROD_OF[m]
        s_new, s_prod = _pm_row(per_model, m), _pm_row(per_model, pm)
        med_new = s_new.get('median_err_q')
        med_prod = s_prod.get('median_err_q')
        row = dict(model=m, prod_model=pm, seed=SEED_OF[m],
                   median_precb_cpu64=med_new, median_prod_cpu64=med_prod)
        if (isinstance(med_new, float) and isinstance(med_prod, float)
                and math.isfinite(med_new) and math.isfinite(med_prod)
                and med_prod > 0):
            row['ratio'] = med_new / med_prod
            row['improvement_pct'] = improvement_pct(med_new, med_prod)
            row['improves'] = bool(med_new < med_prod)
            a, b = err_store.get(m), err_store.get(pm)
            if a is not None and b is not None:
                rci = boot_ratio_ci_paired(a, b)
                row['ratio_ci95'] = rci
                row['improvement_ci95_pct'] = [100.0 * (1.0 - rci[1]),
                                               100.0 * (1.0 - rci[0])]
        else:
            all_ok = False
        per_seed.append(row)
    if not all_ok or any('improves' not in r for r in per_seed):
        return dict(status='pending',
                    missing=[r['model'] for r in per_seed
                             if 'improves' not in r],
                    per_seed=per_seed,
                    note='precb/prod medians incomplete — no descriptive '
                         'table')
    verdict, consistent = companion_verdict([r['improves']
                                             for r in per_seed])
    imprs = [r['improvement_pct'] for r in per_seed]
    mean_impr = float(np.mean(imprs))
    ci = boot_mean_improvement_ci_paired(
        [err_store[m] for m in PRECB_MODELS],
        [err_store[PROD_OF[m]] for m in PRECB_MODELS])
    block = dict(
        part='precb',
        role='DESCRIPTIVE, NON-GATING — the precision 2x2 LAST cell '
             '(clean-TRAINED models under clean EVAL); the discriminating '
             'follow-up the PB-rescore record names '
             '(g_pb.cpu_f64_companion.producer).  No gate/adoption '
             'semantics attach: G-PB is a registered legacy FAIL and this '
             'part quotes it, never re-decides it.',
        gating=False,
        status='decided',
        convention=CELL,
        pairing='seed-paired (42<->42 prod, 43<->43 abl_gram_s43, 44<->44 '
                'abl_gram_s44); BOTH sides scored fresh in this run under '
                'the SAME cpu-f64 convention on the SAME pinned rows — the '
                'cpu-f64 exact-ED features are the clean-eval corner',
        per_seed=per_seed,
        n_improve=int(sum(r['improves'] for r in per_seed)),
        sign_consistency=verdict,
        sign_consistent=bool(consistent),
        mean_improvement_pct=mean_impr,
        mean_improvement_ci95_pct=ci,
        ci_note='draw-paired eval CI on the pinned rows (B=%d rng %d order '
                'stats %d/%d) — eval noise ONLY; the binding uncertainty '
                'is the training-seed floor (SE 7.78%%, CI +/-16-22%% at '
                'n=3), so any |delta| below ~16%% is inside seed noise'
                % (A.BOOT_B, A.BOOT_SEED, A.BOOT_LO, A.BOOT_HI),
        seed_floor_note=CPU64_SEED_FLOOR_NOTE,
        interpretation=dict(
            question=PRECB_QUESTION,
            precision_2x2=dict(PRECB_2X2),
            legacy_null=dict(dict(PB_RESCORE_NULL), record_check=pb_check),
            eval_time_feature_term=dict(PRECA_FEATURE_TERM),
            reading='If the precb-vs-prod deltas sit inside the seed floor '
                    'under clean eval too, clean-feature TRAINING adds '
                    'nothing anywhere on the 2x2 — the convention gap is '
                    'entirely the EVAL-time feature term (PREC-A), '
                    'completing the G-PB null.  A sign-consistent '
                    'improvement here (absent under legacy eval) would '
                    'instead mean the training-data effect expresses only '
                    'when eval features are clean (train/eval precision '
                    'matching) — a next-window follow-up, not a claim '
                    'from this descriptive part.',
            data_draw_caveat=PB_DATA_DRAW_CAVEAT))
    print('%s precb cpu64 (2x2 last cell): %s | mean %+.2f%% | per-seed '
          '%s — descriptive (legacy G-PB null quoted: mean %+.2f%%)'
          % (TAG, verdict, mean_impr,
             ['%+.2f%%' % v for v in imprs],
             PB_RESCORE_NULL['mean_improvement_pct']), flush=True)
    return block


# ------------------------------------------------------------------ gates
def compute_gates(stream_meta, ckpt_rows, required, optional_absent,
                  conv_echo, part_std_block, part_precb_block,
                  requested_parts, manifest_state, anchor_check, det_block,
                  full_n):
    gates = {}
    src = stream_meta.get('source')
    if src == 'val_cache_shards':
        ok0 = bool(stream_meta.get('labels_rows_match_pin')) or not full_n
        rat0 = ('labels = registered val-cache rows in stored order; rows '
                'sha256 %s vs pin %s (enforced at full n=4096 only)'
                % (stream_meta.get('labels_rows_sha256', '')[:16],
                   LABELS_ROWS_SHA_PIN[:16]))
    elif src == 'npz_fallback':
        ok0 = bool((stream_meta.get('labels_npz') or {}
                    ).get('sha_known_registered'))
        rat0 = ('labels = registered predictions npz fallback (file sha '
                'must match a registered pin; rows recorded; same stream '
                'to <1e-6 per rg_h9 G1)')
    else:
        ok0, rat0 = False, 'no labels source'
    gates['C0_stream_pinned'] = {
        'pass': bool(ok0),
        'rationale': rat0 + '; source=%r rows_sha=%s'
                     % (src, stream_meta.get('labels_rows_sha256',
                                             '')[:16])}

    bad = [n for n in required
           if not (ckpt_rows.get(n, {}).get('match')
                   and ckpt_rows.get(n, {}).get('n_params_ok'))]
    # D4 (Opus verify 2026-07-30): the std sidecar pins were recorded but
    # never gated -- an internally-consistent std_stats.npz from a DIFFERENT
    # std run would restore silently.  Gate the three pin-matches for every
    # requested std model whose row carries them.
    for n in required:
        row = ckpt_rows.get(n, {})
        std_blk = row.get('std') or {}
        for pin_key in ('stats_file_sha_matches_pin',
                        'mu_sha_matches_pin', 'sigma_sha_matches_pin'):
            if pin_key in std_blk and std_blk[pin_key] is False:
                if n not in bad:
                    bad.append('%s:%s' % (n, pin_key))
    gates['C1_checkpoints_loaded_digests'] = {
        'pass': not bad,
        'rationale': ('every REQUIRED checkpoint of the selected parts '
                      '(%d models) byte-verified sha256+bytes against the '
                      'registered pins BEFORE deserialization, restored, '
                      'n_params == family pin; optional mlplc members may '
                      'be absent (absent: %s).  failures: %s'
                      % (len(required),
                         ','.join(optional_absent) or 'none',
                         ','.join(bad) or 'none')),
        'rows': ckpt_rows}

    ok2 = bool(conv_echo.get('x64_active')
               and conv_echo.get('default_dtype') == 'float64'
               and conv_echo.get('backend') == 'cpu'
               and conv_echo.get('matmul_context') == 'highest'
               and conv_echo.get('features') == 'f64_exact_ed'
               and conv_echo.get('scoring') == 'float64')
    gates['C2_convention_pins_echoed'] = {
        'pass': ok2,
        'rationale': ('STAT-04 cpu64 branch pins echoed AND live-checked: '
                      'jax_enable_x64=%r default_dtype=%r backend=%r '
                      'matmul=%r features=%r scoring=%r (convention error '
                      'here would silently score a different convention)'
                      % (conv_echo.get('x64_active'),
                         conv_echo.get('default_dtype'),
                         conv_echo.get('backend'),
                         conv_echo.get('matmul_context'),
                         conv_echo.get('features'),
                         conv_echo.get('scoring'))),
        'echo': conv_echo}

    b = part_std_block or {}
    if 'std' in requested_parts:
        decided = bool(
            b.get('status') == 'decided'
            and isinstance(b.get('companion_verdict'), str)
            and (b['companion_verdict'] == VERDICT_CONSISTENT
                 or b['companion_verdict'].startswith('sign-inconsistent ('))
            and b.get('adoption_status') in (ADOPT_STATUS_PASS,
                                             ADOPT_STATUS_FAIL)
            and len(b.get('per_seed') or []) == 3
            and all(isinstance(r.get('improvement_pct'), float)
                    and math.isfinite(r['improvement_pct'])
                    for r in b.get('per_seed') or [])
            and (b.get('deciding_leg') or {}).get('record_sha256')
            == W2_STD_RECORD['record_sha256'])
        gates['C3_std_companion_verdict_emitted'] = {
            'pass': decided,
            'rationale': ('DECIDED companion verdict emitted (either sign '
                          'branch passes — decided != adopted), pinned '
                          'adoption_status string, 3 finite per-seed rows, '
                          'deciding-leg record sha echoed; got verdict=%r '
                          'adoption=%r'
                          % (b.get('companion_verdict'),
                             b.get('adoption_status')))}
    else:
        gates['C3_std_companion_verdict_emitted'] = {
            'pass': bool(b.get('status') == 'pending' and b.get('missing')),
            'rationale': ('part std explicitly deselected (--parts): '
                          'pending-with-missing emitted; the blocker '
                          'remains open')}

    gates['C4_manifest_written'] = {
        'pass': bool(manifest_state.get('written')
                     and manifest_state.get('preverified')),
        'rationale': ('MANIFEST.sha256 written (%s) and re-hash-verified '
                      'for the pre-result outputs (%s); the result JSON '
                      'line is appended post-write and the FULL manifest '
                      're-verified before exit (exit code + final stdout '
                      'line reflect it); harvest re-hashes locally'
                      % (MANIFEST_NAME,
                         ','.join(manifest_state.get('covers', []))))}

    gates['C5_anchor_convention_band'] = {
        'pass': bool(anchor_check['ok']) if full_n else True,
        'rationale': ('prod-triplet cpu-f64 medians within [%.2f, %.2f] of '
                      'BOTH registered CPU anchor sets (ARCH-A1 '
                      'f32-pipeline, same rows; CAL-1 N_s=1e10 cpu64 '
                      'family, CAL1-ARCHA1 record %s...) — catches '
                      'convention errors (legacy 3.7x class), not jitter; '
                      'enforced at full n only (full_n=%s)'
                      % (ANCHOR_BAND[0], ANCHOR_BAND[1],
                         CAL1_ARCHA1_RECORD_SHA[:12], full_n)),
        'checks': anchor_check['checks']}

    gates['C6_determinism_finiteness'] = {
        'pass': bool(det_block['features_pool_identical']
                     and det_block['repeated_forward_identical']
                     and det_block['all_finite']),
        'rationale': ('spawn-pool feature rows == serial recompute '
                      'byte-for-byte (%s); repeated jitted f64 forward '
                      'chunk byte-identical (%s); all per-sample errors '
                      'finite (%s)'
                      % (det_block['features_pool_identical'],
                         det_block['repeated_forward_identical'],
                         det_block['all_finite']))}

    ok7, rat7 = precb_gate_decision(part_precb_block, requested_parts)
    gates['C7_precb_part_emitted'] = {'pass': ok7, 'rationale': rat7}
    return gates


def precb_gate_decision(block, requested_parts):
    """(pass, rationale) for gate C7 — factored pure so the selftest can
    exercise the part-subset semantics without building the full gate set.
    Receipt-safe by construction: the caller stores ONLY {'pass',
    'rationale'} (no other pass-like key ever nests under gates — the
    expc1_g3_eval.py:795 lesson)."""
    b = block or {}
    if 'precb' not in requested_parts:
        ok = bool(b.get('status') == 'not_requested')
        return ok, ('part precb not requested (--parts subset): block must '
                    'say not_requested; got status=%r' % b.get('status'))
    ok = bool(
        b.get('status') == 'decided'
        and b.get('gating') is False
        and 'adoption_status' not in b
        and isinstance(b.get('sign_consistency'), str)
        and (b['sign_consistency'] == VERDICT_CONSISTENT
             or b['sign_consistency'].startswith('sign-inconsistent ('))
        and len(b.get('per_seed') or []) == 3
        and all(isinstance(r.get('improvement_pct'), float)
                and math.isfinite(r['improvement_pct'])
                for r in b.get('per_seed') or [])
        and ((b.get('interpretation') or {}).get('legacy_null') or {})
        .get('record_sha256') == PB_RESCORE_NULL['record_sha256']
        and ((b.get('interpretation') or {}).get('eval_time_feature_term')
             or {}).get('record_sha256')
        == PRECA_FEATURE_TERM['record_sha256']
        and (b.get('interpretation') or {}).get('question')
        == PRECB_QUESTION)
    return ok, ('part precb emitted in its DESCRIPTIVE shape: decided, '
                'gating=False, NO adoption-status key, sign-consistency '
                'string, 3 finite per-seed rows, PB-rescore legacy-null '
                'sha + PREC-A feature-term sha + the pinned question '
                'echoed; got status=%r sign=%r'
                % (b.get('status'), b.get('sign_consistency')))


# ---------------------------------------------------------------- compute
def run_compute(args):
    t0 = time.time()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    attempt = '%d-%d-%s' % (int(t0), os.getpid(),
                            hashlib.sha256(os.urandom(16)).hexdigest()[:16])
    smoke = bool(args.smoke)
    n_eval = int(args.n_eval or (N_EVAL_SMOKE if smoke else N_EVAL_DEFAULT))
    if smoke:
        n_eval = min(n_eval, N_EVAL_SMOKE)
    full_n = bool(n_eval == N_EVAL_DEFAULT and not smoke)
    parts = tuple(args.parts)
    if not args.ckpt_root:
        raise SystemExit('%s --ckpt-root is required for compute runs; use '
                         '--selftest for the local structural check' % TAG)
    ckpt_root = os.path.abspath(os.path.expanduser(args.ckpt_root))

    # model sets
    required, optional = [], []
    for p in parts:
        for m in PART_MODELS[p]['required']:
            if m not in required:
                required.append(m)
        for m in PART_MODELS[p]['optional']:
            if m not in optional and m not in required:
                optional.append(m)
    optional_present, optional_absent = [], []
    for m in optional:
        if os.path.isfile(os.path.join(ckpt_root, m,
                                       'final_state.msgpack')):
            optional_present.append(m)
        else:
            optional_absent.append(m)
    model_list = required + optional_present
    print('%s parts=%s | required=%d optional_present=%d '
          'optional_absent=%s | n_eval=%d smoke=%s'
          % (TAG, ','.join(parts), len(required), len(optional_present),
             ','.join(optional_absent) or 'none', n_eval, smoke), flush=True)

    # labels (before jax — pure numpy)
    labels, stream_meta = load_labels(args.val_cache, args.labels_npz,
                                      n_eval)

    # engine: x64 BEFORE the first jax op (STAT-04; A.load_engine6)
    eng, fmb_policy, code_root = A.load_engine6(want_x64=True)
    import jax
    import jax.numpy as jnp
    conv_echo = dict(
        x64_active=bool(jax.config.jax_enable_x64),
        default_dtype=str(jnp.zeros(1).dtype),
        backend=str(jax.default_backend()),
        jax_platforms=os.environ.get('JAX_PLATFORMS'),
        matmul_context='highest',
        features='f64_exact_ed',
        scoring='float64',
        convention_citation='c10_tableiii_cpu64.py:33-36,509-518,575-585 '
                            'via expc1_cal1_archa1.py:44-47,889-900,'
                            '998-1003,1022-1049')
    if conv_echo['backend'] != 'cpu':
        raise SystemExit('%s backend %r != cpu — the cpu-f64 convention is '
                         'CPU by definition; refusing to score a different '
                         'convention silently' % (TAG, conv_echo['backend']))

    Gpanel = A.panel_matrices(labels)
    diag_dev = float(np.max(np.abs(
        Gpanel[:, np.arange(A.MP), np.arange(A.MP)].mean(axis=1)
        - A.DIAG_GAUGE_MEAN)))
    stream_meta['diag_gauge_max_dev'] = diag_dev
    stream_meta['diag_gauge_ok'] = bool(diag_dev < A.DIAG_GAUGE_TOL)
    # engine-reconstruct identity on the first rows (cal1 C3 pattern)
    g_rec = np.asarray(eng.g_gen.reconstruct(jnp.asarray(labels[:4])),
                       np.float64)
    stream_meta['panel_reconstruct_identical'] = bool(
        np.array_equal(g_rec, Gpanel[:4]))

    # features
    BX, BE, feat_ident = build_features_exact64(eng, Gpanel, out_dir,
                                                max(1, args.nproc))
    bx_flat = BX[..., None]                       # (n, 10, 10, 1) f64
    be_flat = BE[:, None]                         # (n, 1) f64

    # checkpoints: byte-verify -> restore -> f64 params (cal1 :984-1003)
    ckpt_rows, apply_fns = {}, {}
    for name in model_list:
        spec = dict(ckpt_spec(name))
        path = os.path.join(ckpt_root, name, 'final_state.msgpack')
        if not os.path.isfile(path):
            ckpt_rows[name] = dict(path=path, error='missing', match=False,
                                   n_params_ok=False)
            continue
        raw, row = A.verify_and_restore(path, spec['sha'], spec['bytes'])
        row.update(family=spec['kind'], res=int(spec.get('res', -1)),
                   init_seed=int(spec.get('init_seed', 42)),
                   n_params_expected=int(spec['n_params']),
                   n_params_ok=False)
        if raw is None:
            ckpt_rows[name] = row
            print('%s   REJECTED %s (sha/bytes mismatch)' % (TAG, name),
                  flush=True)
            continue
        try:
            if spec['kind'] == 'ogn_std':
                mu, sigma, std_info = _load_std_stats(
                    os.path.join(ckpt_root, name))
                row['std'] = std_info
                model, params, bstats, n_par = build_state_std(
                    eng, name, spec, raw, mu, sigma)
            else:
                model, params, bstats, n_par = A.build_state(
                    eng, name, spec, raw, None)
            row['n_params'] = n_par
            row['n_params_ok'] = bool(n_par == int(spec['n_params']))
            params64 = jax.tree_util.tree_map(
                lambda a_: jnp.asarray(a_, jnp.float64), params)
            bstats64 = jax.tree_util.tree_map(
                lambda a_: jnp.asarray(a_, jnp.float64), bstats)
            apply_fns[name] = A.make_apply(model, params64, bstats64)
            print('%s   loaded %s (%s...) n_params=%d family=%s'
                  % (TAG, name, row['sha256'][:16], n_par, spec['kind']),
                  flush=True)
        except SystemExit:
            raise                        # staging errors (std stats) stay hard
        except Exception as exc:                              # noqa: BLE001
            row['error'] = '%s: %s' % (type(exc).__name__, exc)
            print('%s   restore FAILED for %s (%s) — recorded; C1 fails'
                  % (TAG, name, row['error']), flush=True)
        ckpt_rows[name] = row

    # forwards: f64 inputs, matmul highest baked at trace (cal1 :1022-1049)
    per_model, err_store, logit_store = {}, {}, {}
    det_fwd_ok = True
    n_cells = len(bx_flat)
    for name in model_list:
        if name not in apply_fns:
            continue
        apply_f = apply_fns[name]
        outs = []
        tm0 = time.time()
        with jax.default_matmul_precision('highest'):
            for s in range(0, n_cells, A.EVAL_CHUNK_F64):
                e = min(s + A.EVAL_CHUNK_F64, n_cells)
                logits = np.asarray(apply_f(
                    jnp.asarray(bx_flat[s:e], jnp.float64),
                    jnp.asarray(be_flat[s:e], jnp.float64)))
                if logits.ndim == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                outs.append(logits)
            l0 = np.asarray(apply_f(
                jnp.asarray(bx_flat[:min(A.EVAL_CHUNK_F64, n_cells)],
                            jnp.float64),
                jnp.asarray(be_flat[:min(A.EVAL_CHUNK_F64, n_cells)],
                            jnp.float64)))
            if l0.ndim == 3:
                l0 = l0.reshape(-1, l0.shape[-1])
            det_fwd_ok = det_fwd_ok and bool(np.array_equal(l0, outs[0]))
        logits_all = np.concatenate(outs, 0)
        preds = np.asarray(eng.g_gen.reconstruct(jnp.asarray(logits_all)),
                           np.float64)
        eq, ef = A.score_pred_panel(preds, Gpanel)
        per_model[name] = dict(
            median_err_q=float(np.median(eq)),
            err_q_ci95=A.boot_median_ci(eq),
            median_err_f=float(np.median(ef)),
            p90_err_q=float(np.percentile(eq, 90.0)),
            n=int(eq.size), n_nonfinite=int(np.sum(~np.isfinite(eq))),
            n_params=int(ckpt_spec(name)['n_params']),
            wall_s=time.time() - tm0)
        err_store[name] = eq
        logit_store[name] = logits_all
        print('%s   %-26s median_q=%.6e ci=[%.4e,%.4e] n=%d (%.1fs)'
              % (TAG, name, per_model[name]['median_err_q'],
                 per_model[name]['err_q_ci95'][0],
                 per_model[name]['err_q_ci95'][1], eq.size,
                 per_model[name]['wall_s']), flush=True)

    # deciding-leg record cross-check (constants stay authoritative)
    w2_check = dict(path=args.w2_record, present=False)
    if args.w2_record and os.path.isfile(os.path.expanduser(args.w2_record)):
        wp = os.path.expanduser(args.w2_record)
        sha = A._sha256_file(wp)
        w2_check = dict(path=wp, present=True, sha256=sha,
                        sha_matches_pin=bool(
                            sha == W2_STD_RECORD['record_sha256']))
        try:
            with open(wp) as f:
                wj = json.load(f)
            g = wj.get('g_std') or {}
            w2_check['decision_in_record'] = g.get('decision')
            w2_check['decision_matches_pin'] = bool(
                g.get('decision') == W2_STD_RECORD['decision'])
            w2_check['mean_in_record'] = g.get('mean_improvement_pct')
            w2_check['mean_matches_pin'] = bool(
                g.get('mean_improvement_pct')
                == W2_STD_RECORD['mean_improvement_pct'])
        except Exception as exc:                              # noqa: BLE001
            w2_check['error'] = repr(exc)

    # PB-rescore record cross-check (constants stay authoritative — same
    # contract as w2_check; only consulted when part precb is requested)
    pb_check = dict(path=args.pb_record, present=False)
    if 'precb' in parts and args.pb_record \
            and os.path.isfile(os.path.expanduser(args.pb_record)):
        pp = os.path.expanduser(args.pb_record)
        sha = A._sha256_file(pp)
        pb_check = dict(path=pp, present=True, sha256=sha,
                        sha_matches_pin=bool(
                            sha == PB_RESCORE_NULL['record_sha256']))
        try:
            with open(pp) as f:
                pj = json.load(f)
            g = pj.get('g_pb') or {}
            pb_check['decision_in_record'] = g.get('decision')
            pb_check['decision_matches_pin'] = bool(
                g.get('decision') == PB_RESCORE_NULL['decision'])
            pb_check['mean_in_record'] = g.get('mean_improvement_pct')
            pb_check['mean_matches_pin'] = bool(
                g.get('mean_improvement_pct')
                == PB_RESCORE_NULL['mean_improvement_pct'])
        except Exception as exc:                              # noqa: BLE001
            pb_check['error'] = repr(exc)

    # parts
    part_std = build_part_std(per_model, err_store, parts, w2_check)
    part_lc = build_part_lc(per_model, err_store, parts)
    part_arch2 = build_part_arch2(per_model, err_store, parts)
    part_mlplc = build_part_mlplc(per_model, err_store, parts,
                                  optional_absent)
    part_precb = build_part_precb(per_model, err_store, parts, pb_check)

    # anchor band (C5)
    checks, aok = {}, True
    for m in PROD_TRIPLET:
        mine = _pm_row(per_model, m).get('median_err_q')
        row = dict(mine=mine, anchor_archa1=ANCHOR_ARCHA1[m],
                   anchor_cal1_1e10=ANCHOR_CAL1_1E10[m])
        if isinstance(mine, float) and math.isfinite(mine):
            r1 = mine / ANCHOR_ARCHA1[m]
            r2 = mine / ANCHOR_CAL1_1E10[m]
            row['ratio_vs_archa1'] = r1
            row['ratio_vs_cal1_1e10'] = r2
            row['within_band'] = bool(
                ANCHOR_BAND[0] <= r1 <= ANCHOR_BAND[1]
                and ANCHOR_BAND[0] <= r2 <= ANCHOR_BAND[1])
        else:
            row['within_band'] = False
        checks[m] = row
        # D2 (Opus verify 2026-07-30): only demand the anchors a requested
        # part actually needs -- 'lc' alone needs only prod_thermal_random;
        # demanding the full triplet hard-failed a correct lc-only run.
        # (scoping logic factored as models_required for the selftest)
        needed = models_required(parts)
        if m in per_model:
            aok = aok and bool(row.get('within_band'))
        elif m in needed:
            aok = False
    anchor_check = dict(ok=bool(aok), checks=checks)

    finite_ok = all(v['n_nonfinite'] == 0 for v in per_model.values())
    det_block = dict(features_pool_identical=bool(feat_ident),
                     repeated_forward_identical=bool(det_fwd_ok),
                     all_finite=bool(finite_ok))

    # persample npz (pre-result output; manifested first)
    npz_path = os.path.join(out_dir, PERSAMPLE_NAME)
    store = dict(labels=labels, bx64=BX, be64=BE)
    for name, eq in err_store.items():
        store['err_q_%s__%s' % (CELL, name)] = eq
    for name, lg in logit_store.items():
        store['logits55_%s' % name] = lg
    np.savez_compressed(npz_path, **store)
    npz_sha = A._sha256_file(npz_path)
    script_path = os.path.abspath(__file__)
    script_sha = A._sha256_file(script_path)
    print('%s wrote %s (%s...)' % (TAG, npz_path, npz_sha[:16]), flush=True)

    # manifest phase 1: npz + a script copy reference
    write_manifest(out_dir, [(PERSAMPLE_NAME, npz_sha)])
    pre_ok, pre_rows = verify_manifest(out_dir)
    manifest_state = dict(written=True, preverified=bool(pre_ok),
                          covers=[PERSAMPLE_NAME],
                          script_sha256=script_sha)

    gates = compute_gates(stream_meta, ckpt_rows, required, optional_absent,
                          conv_echo, part_std, part_precb, parts,
                          manifest_state, anchor_check, det_block, full_n)
    executed = list(gates.values())
    # D1 (Opus verify 2026-07-30): --n-eval != 4096 must never seal a
    # passing receipt -- at small n a per-seed sign is flippable and the
    # C0/C5 guards self-disable, minting a quotable false ADOPT.
    all_pass = bool(executed and all(g['pass'] for g in executed)
                    and not smoke and n_eval == N_EVAL_DEFAULT)

    result = dict(
        schema=SCHEMA, run_id=RUN_ID, stage=STAGE,
        stage_attempt=attempt,
        execution='OFF-FLEET rented CPU box — self-receipted (no c11 shim); '
                  'harvest = scp out_dir + local re-hash vs MANIFEST.sha256',
        task='CPU-f64 companion legs (FINAL_PLAN R4 G-STD ADOPTION BLOCKER '
             '+ R3/R11 descriptive companions + registered MLP LC arms) on '
             'the pinned seed-1007 dev rows under the STAT-04 cpu64 '
             'convention; prod CPU-f64 medians computed FRESH in this run '
             '(legacy pins never reused — different convention)',
        argv=[str(x) for x in sys.argv],
        parts_requested=list(parts),
        n_eval=n_eval,
        n_eval_is_registered_convention=bool(n_eval == N_EVAL_DEFAULT),
        models_scored=sorted(per_model),
        protocol=dict(
            stream='seed-1007 dev rows, first %d rows in registered order '
                   '== tab3_4096' % n_eval,
            convention=dict(cell=CELL, role=CELL_ROLE),
            metric='err_q/err_f per sample, err_pair VERBATIM '
                   'c10_h0_sensitivity.py:213-219 via '
                   'expc1_cal1_archa1.score_pred_panel; float64 scoring; '
                   'G_true via the triu panel (== engine GGenerator '
                   'reconstruction, identity-checked)',
            features='f64 exact-ED: labels -> f64 H (_shots_h_dense64) -> '
                     'scipy f64 eigh -> beta=1 thermal p_true -> '
                     'rho=(V*p)@V.T -> einsum(ji,kij->k) pair block '
                     '(compute_rho_m minus the f32 cast); be=sum(p*E) — '
                     'the CAL-1 machinery at the clean point (no drawer)',
            bootstrap='per-Hamiltonian bootstrap of the median (each row = '
                      'one H at tab3_4096), B=%d, default_rng(%d), order '
                      'stats %d/%d; all comparison CIs draw-paired (shared '
                      'index draws)' % (A.BOOT_B, A.BOOT_SEED, A.BOOT_LO,
                                        A.BOOT_HI),
            eval_chunk=A.EVAL_CHUNK_F64,
            seed_pairing={m: PROD_OF[m]
                          for m in STD_MODELS + ARCH2_MODELS},
            control_note='controls = the REGISTERED prod triplet, rescored '
                         'fresh here under the SAME cpu-f64 convention on '
                         'the SAME rows — CPU-f64 is a different convention '
                         'from the legacy deciding leg, so the PREC-A '
                         'legacy medians are quoted only as context, never '
                         'as denominators'),
        stream=stream_meta,
        checkpoints=ckpt_rows,
        cell=dict(cell=CELL, matmul_context='highest', dtype='float64',
                  backend=conv_echo['backend'], x64=conv_echo['x64_active'],
                  per_model=per_model, n=n_cells,
                  scoring='float64 err_q/err_f'),
        part_std=part_std,
        part_lc=part_lc,
        part_arch2=part_arch2,
        part_mlplc=part_mlplc,
        part_precb=part_precb,
        registered_context=dict(
            w2_rescore_std=dict(W2_STD_RECORD),
            w2_rescore_full=dict(W2_FULL_CONTEXT),
            lc_rescore=dict(LC_RESCORE_CONTEXT),
            cal1_archa1_record_sha256=CAL1_ARCHA1_RECORD_SHA,
            pb_rescore_null=dict(PB_RESCORE_NULL),
            preca_feature_term=dict(PRECA_FEATURE_TERM)),
        fmb_rho_policy=fmb_policy,
        code_root=code_root,
        environment=A.env_block(max(1, args.nproc)),
        gates=gates,
        all_gates_pass=all_pass,
        smoke=smoke,
        script_sha256=script_sha,
        manifest=dict(file=MANIFEST_NAME,
                      persample_sha256=npz_sha,
                      preverify_rows=pre_rows,
                      note='result JSON sha appended post-write; full '
                           'manifest re-verified before exit'),
        wall_s=time.time() - t0)

    jp = os.path.join(out_dir, RESULT_NAME)
    with open(jp, 'w') as f:
        json.dump(A._jsonable(result), f, indent=1)
        f.write('\n')
    print('%s wrote %s (all_gates_pass=%s)' % (TAG, jp, all_pass),
          flush=True)
    if smoke:
        print('%s SMOKE RECORD — all_gates_pass forced False, never '
              'quotable' % TAG, flush=True)

    # manifest phase 2: append result + final verify (the harvest contract)
    append_manifest(out_dir, RESULT_NAME, A._sha256_file(jp))
    final_ok, final_rows = verify_manifest(out_dir)
    print('%s MANIFEST %s: %d entries, verify=%s'
          % (TAG, MANIFEST_NAME, len(final_rows),
             'OK' if final_ok else 'FAILED'), flush=True)
    if not final_ok:
        print('%s FATAL: manifest verification failed — do not harvest'
              % TAG, flush=True)
        return 2
    print('%s DONE parts=%s models=%d/%d adoption=%r wall=%.1fs'
          % (TAG, ','.join(parts), len(per_model), len(model_list),
             (part_std or {}).get('adoption_status'),
             time.time() - t0), flush=True)
    return 0 if all_pass else 1


# --------------------------------------------------------------- selftest
def selftest():
    """Structural checks of every pure code path — numpy + stdlib only; no
    jax/flax/scipy/engine and no registered inputs required."""
    import tempfile
    failures = []

    def check(name, ok):
        print('  [%s] %s' % ('ok' if ok else 'FAIL', name))
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(0)
    # orientation identities
    check('improvement_is_minus_deficit',
          abs(improvement_pct(0.9, 1.0) + deficit_pct(0.9, 1.0)) < 1e-12)
    check('improvement_sign', improvement_pct(0.9, 1.0) > 0
          and improvement_pct(1.1, 1.0) < 0)
    # companion verdict strings (task-pinned)
    v3, c3 = companion_verdict([True, True, True])
    check('verdict_3of3', v3 == VERDICT_CONSISTENT and c3 is True)
    v2, c2 = companion_verdict([True, False, True])
    check('verdict_2of3', v2 == 'sign-inconsistent (2/3)' and c2 is False)
    v0, c0 = companion_verdict([False, False, False])
    check('verdict_0of3', v0 == 'sign-inconsistent (0/3)' and c0 is False)
    try:
        companion_verdict([True, True])
        check('verdict_length_guard', False)
    except ValueError:
        check('verdict_length_guard', True)
    # adoption composition (task-pinned strings)
    check('adoption_pass',
          adoption_status(True) == ADOPT_STATUS_PASS
          and ADOPT_STATUS_PASS
          == 'ADOPT (deciding leg PASS + companion sign-consistent)')
    check('adoption_fail',
          adoption_status(False) == ADOPT_STATUS_FAIL
          and ADOPT_STATUS_FAIL == 'legacy-only (companion failed)')
    check('adoption_deciding_guard',
          adoption_status(True, deciding_leg_pass=False)
          == 'legacy-only (deciding leg failed)')
    # magnitude flag (GT-1: all >= 16%, sign-consistent required)
    check('magnitude_all_ge16', magnitude_flag([16.0, 17.5, 20.0], True))
    check('magnitude_one_below', not magnitude_flag([15.9, 30.0, 25.0],
                                                    True))
    check('magnitude_needs_sign', not magnitude_flag([20.0, 20.0, 20.0],
                                                     False))
    check('magnitude_nonfinite_guard',
          not magnitude_flag([float('nan'), 20.0, 20.0], True))
    check('magnitude_threshold_pin', MAGNITUDE_THRESHOLD_PCT == 16.0)
    # deciding-leg pin arithmetic: mean == mean(per-seed) (registered)
    ps = [W2_STD_RECORD['per_seed_improvement_pct'][k]
          for k in ('42', '43', '44')]
    check('w2_pin_mean_consistent',
          abs(np.mean(ps) - W2_STD_RECORD['mean_improvement_pct']) < 1e-9)
    check('w2_pin_pass_gt_threshold',
          W2_STD_RECORD['mean_improvement_pct']
          > W2_STD_RECORD['threshold_pct'])
    # paired bootstraps: degenerate constant-ratio + guards
    base = np.abs(rng.standard_normal(512)) + 0.5
    new_arrs = [base * 0.8, base * 0.85, base * 0.9]
    prod_arrs = [base, base, base]
    ci = boot_mean_improvement_ci_paired(new_arrs, prod_arrs)
    expect = float(np.mean([20.0, 15.0, 10.0]))
    check('mean_impr_boot_degenerate',
          abs(ci[0] - expect) < 1e-9 and abs(ci[1] - expect) < 1e-9)
    try:
        boot_mean_improvement_ci_paired(new_arrs[:2], prod_arrs)
        check('mean_impr_boot_length_guard', False)
    except ValueError:
        check('mean_impr_boot_length_guard', True)
    rci = boot_ratio_ci_paired(base * 0.8, base)
    check('ratio_boot_degenerate',
          abs(rci[0] - 0.8) < 1e-9 and abs(rci[1] - 0.8) < 1e-9)
    check('boot_pins_25_975', (A.BOOT_LO, A.BOOT_HI) == (25, 975))
    # slope machinery on a synthetic power law
    s_true = 0.12
    m4, m6 = 1e4 ** -s_true, 1e6 ** -s_true
    check('slope_exact',
          abs(slope_per_decade(m4, m6, 1e4, 1e6) - s_true) < 1e-12)
    check('decades_1e6_5e6',
          abs(math.log10(5e6 / 1e6) - 0.6989700043360187) < 1e-15)
    sci = boot_slope_ci_paired(base * m4, base * m6, 1e4, 1e6)
    check('slope_boot_degenerate',
          abs(sci[0] - s_true) < 1e-9 and abs(sci[1] - s_true) < 1e-9)
    # clean-feature worker math vs a serial reference (numpy-only)
    D = 12
    Vq, _r = np.linalg.qr(rng.standard_normal((D, D)))
    E = np.sort(rng.standard_normal(D))
    ops = rng.standard_normal((A.MP * A.MP, D, D))
    bx, be = features_from_VE(Vq, E, ops)
    p = np.exp(-(E - E.min()))
    p /= p.sum()
    rho = (Vq * p) @ Vq.T
    ref = np.einsum('ji,kij->k', rho, ops).reshape(A.MP, A.MP)
    check('features_match_serial_reference',
          np.array_equal(ref, bx) and abs(float(np.sum(p * E)) - be) < 1e-15
          and bx.shape == (A.MP, A.MP))
    check('features_clean_gs_limit',
          abs(features_from_VE(Vq, E, ops, beta=1e6)[1] - E.min()) < 1e-9)
    # drawer-free identity vs the CAL-1 worker at the clean limit: the
    # p_true arithmetic here equals cal1's p_true block (:944-946)
    p_cal1 = np.exp(-A.BETA_THERMAL * (E - E.min()))
    p_cal1 = (p_cal1 / np.sum(p_cal1)).astype(np.float64)
    check('thermal_weights_match_cal1', np.array_equal(p, p_cal1))
    # manifest writer round-trip + tamper detection
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, 'a.bin')
        with open(f1, 'wb') as f:
            f.write(b'payload-1')
        write_manifest(td, [('a.bin', A._sha256_file(f1))])
        ok1, rows1 = verify_manifest(td)
        check('manifest_verify_ok', ok1 and len(rows1) == 1)
        f2 = os.path.join(td, 'b.bin')
        with open(f2, 'wb') as f:
            f.write(b'payload-2')
        append_manifest(td, 'b.bin', A._sha256_file(f2))
        ok2, rows2 = verify_manifest(td)
        check('manifest_append_verify_ok', ok2 and len(rows2) == 2)
        with open(f1, 'wb') as f:
            f.write(b'TAMPERED')
        ok3, _rows3 = verify_manifest(td)
        check('manifest_tamper_detected', not ok3)
    # pins self-consistency
    check('parts_registry',
          tuple(PART_MODELS) == PARTS_ALL
          and PART_MODELS['std']['required']
          == STD_MODELS + PROD_TRIPLET
          and PART_MODELS['precb']['required']
          == PRECB_MODELS + PROD_TRIPLET
          and PART_MODELS['precb']['optional'] == ()
          and PART_MODELS['mlplc']['required'] == ()
          and set(PART_MODELS['mlplc']['optional'])
          == set(MLP_LC_MODELS) | {'abl_mlp'})
    check('seed_pairing_map',
          PROD_OF['std_thermal_random'] == 'prod_thermal_random'
          and PROD_OF['std_thermal_random_s43'] == 'abl_gram_s43'
          and PROD_OF['arch2_thermal_random_s44'] == 'abl_gram_s44'
          and PROD_OF['precb_thermal_random'] == 'prod_thermal_random'
          and PROD_OF['precb_thermal_random_s43'] == 'abl_gram_s43'
          and PROD_OF['precb_thermal_random_s44'] == 'abl_gram_s44')
    check('ckpt_pins_resolvable',
          all(set(('sha', 'bytes', 'kind', 'n_params'))
              <= set(ckpt_spec(m))
              for m in (PROD_TRIPLET + STD_MODELS + ARCH2_MODELS
                        + OGN_LC_MODELS + MLP_LC_MODELS + PRECB_MODELS
                        + ('abl_mlp',))))
    check('prod_pins_are_A_records',
          ckpt_spec('prod_thermal_random')['sha'].startswith('7b7bc8e9')
          and ckpt_spec('abl_mlp')['sha'].startswith('33b7114d'))
    check('std_bytes_equal_prod_bytes',
          all(CKPTS_NEW[m]['bytes'] == A.CKPTS['prod_thermal_random']
              ['bytes'] and CKPTS_NEW[m]['n_params']
              == A.CKPTS['prod_thermal_random']['n_params']
              for m in STD_MODELS))
    check('lc_ntrain_monotone',
          all(N_TRAIN[a_] < N_TRAIN[b_]
              for a_, b_ in OGN_CURVE_SEGMENTS + MLP_CURVE_SEGMENTS))
    check('anchor_band_sane',
          0 < ANCHOR_BAND[0] < 1 < ANCHOR_BAND[1]
          and set(ANCHOR_ARCHA1) == set(ANCHOR_CAL1_1E10)
          == set(PROD_TRIPLET))
    # the [STD-1] stanza must be present verbatim in THIS file (the wave2
    # selftest precedent, expc1_wave2_rescore.py:839)
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    check('std_stanza_verbatim',
          'x_std = (x_sym - mu_) / sigma_' in src
          and src.count('[STD-1') >= 4
          and 'bare_rdm=x_std' in src)
    # A reuse surface present
    check('A_reuse_surface',
          callable(A.panel_matrices) and callable(A.score_pred_panel)
          and callable(A.verify_and_restore) and callable(A.build_state)
          and callable(A.make_apply) and callable(A.load_engine6)
          and A.MP == 10 and A.LABEL55 == 55 and A.EVAL_CHUNK_F64 == 512)

    # ---------------------------------------------- part precb (2x2 cell)
    # pins: plain prod restore path (kind ogn, res 3, prod topology/bytes),
    # receipt shas distinct and at the T-precb_* harvested values
    prod_pin = A.CKPTS['prod_thermal_random']
    check('precb_pins_plain_prod_path',
          all(CKPTS_NEW[m]['kind'] == 'ogn'
              and CKPTS_NEW[m]['res'] == 3
              and CKPTS_NEW[m]['bytes'] == prod_pin['bytes']
              and CKPTS_NEW[m]['n_params'] == prod_pin['n_params']
              and CKPTS_NEW[m]['init_seed'] == SEED_OF[m]
              and 'config_json_sha' not in CKPTS_NEW[m]     # NO std sidecar
              for m in PRECB_MODELS))
    check('precb_pins_receipt_shas',
          CKPTS_NEW['precb_thermal_random']['sha'].startswith('bdc320eb')
          and CKPTS_NEW['precb_thermal_random_s43']['sha']
          .startswith('371986b7')
          and CKPTS_NEW['precb_thermal_random_s44']['sha']
          .startswith('865ce8b9')
          and len({CKPTS_NEW[m]['sha'] for m in PRECB_MODELS}) == 3
          and all(len(CKPTS_NEW[m]['sha']) == 64 for m in PRECB_MODELS))
    # quoted-record pin arithmetic (PB null + PREC-A feature term)
    pb_ps = [PB_RESCORE_NULL['per_seed_improvement_pct'][k]
             for k in ('42', '43', '44')]
    check('pb_null_pin_mean_consistent',
          abs(np.mean(pb_ps) - PB_RESCORE_NULL['mean_improvement_pct'])
          < 1e-9)
    check('pb_null_is_a_null',
          PB_RESCORE_NULL['mean_improvement_pct']
          < PB_RESCORE_NULL['threshold_pct']
          and PB_RESCORE_NULL['decision']
          == 'fail (legacy deciding leg <= 5%)'
          and PB_RESCORE_NULL['branch'] == 'fail'
          and PB_RESCORE_NULL['record_sha256'].startswith('a59a2406')
          and len(PB_RESCORE_NULL['record_sha256']) == 64)
    check('preca_feature_term_pins_consistent',
          all(0.0 < PRECA_FEATURE_TERM['feature_term_ratio'][m] < 1.0
              and abs(PRECA_FEATURE_TERM
                      ['implied_eval_time_improvement_pct'][m]
                      - 100.0 * (1.0 - PRECA_FEATURE_TERM
                                 ['feature_term_ratio'][m])) < 1e-9
              for m in PROD_TRIPLET)
          and PRECA_FEATURE_TERM['record_sha256'].startswith('ecc80f42'))
    check('precb_2x2_covers_four_cells',
          set(PRECB_2X2) == {'train_bf16__eval_bf16',
                             'train_bf16__eval_clean',
                             'train_clean__eval_bf16',
                             'train_clean__eval_clean'})
    # D2-style anchor scoping (models_required is the C5 authority)
    check('anchor_scope_lc_only',
          models_required(('lc',)) & set(PROD_TRIPLET)
          == {'prod_thermal_random'})
    check('anchor_scope_precb_full_triplet',
          set(PROD_TRIPLET) <= models_required(('precb',))
          and set(PRECB_MODELS) <= models_required(('precb',)))
    check('anchor_scope_mlplc_none',
          models_required(('mlplc',)) == set())
    # part-subset semantics + the decided/pending branches (pure)
    blk_nr = build_part_precb({}, {}, ('std', 'lc'), dict(present=False))
    check('precb_not_requested_branch',
          blk_nr == dict(status='not_requested'))
    ok_nr, _r = precb_gate_decision(blk_nr, ('std', 'lc'))
    check('precb_gate_passes_when_deselected', ok_nr)
    ok_nr2, _r = precb_gate_decision(blk_nr, ('precb',))
    check('precb_gate_fails_on_missing_block_when_requested', not ok_nr2)
    blk_pend = build_part_precb(
        {'precb_thermal_random': dict(median_err_q=1.0e-3)}, {},
        ('precb',), dict(present=False))
    check('precb_pending_branch',
          blk_pend['status'] == 'pending'
          and 'precb_thermal_random_s43' in blk_pend['missing'])
    ok_pend, _r = precb_gate_decision(blk_pend, ('precb',))
    check('precb_gate_fails_on_pending_when_requested', not ok_pend)
    # decided branch on synthetic paired stores (numpy-only)
    base2 = np.abs(rng.standard_normal(256)) + 0.5
    pm_syn, es_syn = {}, {}
    scales = {'precb_thermal_random': 0.97, 'precb_thermal_random_s43': 1.02,
              'precb_thermal_random_s44': 0.99}
    for m in PRECB_MODELS:
        pm = PROD_OF[m]
        es_syn[pm] = base2
        pm_syn[pm] = dict(median_err_q=float(np.median(base2)))
        es_syn[m] = base2 * scales[m]
        pm_syn[m] = dict(median_err_q=float(np.median(base2 * scales[m])))
    blk = build_part_precb(pm_syn, es_syn, ('precb',),
                           dict(present=True, sha_matches_pin=True))
    check('precb_decided_shape',
          blk['status'] == 'decided' and blk['gating'] is False
          and 'adoption_status' not in blk
          and blk['sign_consistency'] == 'sign-inconsistent (2/3)'
          and blk['n_improve'] == 2 and len(blk['per_seed']) == 3)
    check('precb_decided_arithmetic',
          abs(blk['mean_improvement_pct']
              - float(np.mean([100.0 * (1.0 - s)
                               for s in scales.values()]))) < 1e-9
          and abs(blk['per_seed'][0]['improvement_pct'] - 3.0) < 1e-9)
    check('precb_interpretation_fields',
          blk['interpretation']['question'] == PRECB_QUESTION
          and blk['interpretation']['legacy_null']['record_sha256']
          == PB_RESCORE_NULL['record_sha256']
          and blk['interpretation']['legacy_null']['record_check']
          == dict(present=True, sha_matches_pin=True)
          and blk['interpretation']['eval_time_feature_term']
          ['record_sha256'] == PRECA_FEATURE_TERM['record_sha256']
          and PB_DATA_DRAW_CAVEAT
          == blk['interpretation']['data_draw_caveat'])
    ok_dec, _r = precb_gate_decision(blk, ('precb',))
    check('precb_gate_passes_on_decided', ok_dec)
    blk_bad = dict(blk)
    blk_bad['adoption_status'] = 'ADOPT'      # forbidden key must fail C7
    ok_bad, _r = precb_gate_decision(blk_bad, ('precb',))
    check('precb_gate_rejects_adoption_status', not ok_bad)
    # all-improve branch produces the consistent verdict string
    scales3 = {m: 0.9 for m in PRECB_MODELS}
    pm3, es3 = {}, {}
    for m in PRECB_MODELS:
        pm = PROD_OF[m]
        es3[pm] = base2
        pm3[pm] = dict(median_err_q=float(np.median(base2)))
        es3[m] = base2 * scales3[m]
        pm3[m] = dict(median_err_q=float(np.median(base2 * scales3[m])))
    blk3 = build_part_precb(pm3, es3, ('precb',), dict(present=False))
    check('precb_all_improve_verdict',
          blk3['sign_consistency'] == VERDICT_CONSISTENT
          and blk3['sign_consistent'] is True and blk3['n_improve'] == 3)

    if failures:
        print('SELFTEST_FAIL: %s' % failures)
        return 1
    print('SELFTEST_OK')
    return 0


# ------------------------------------------------------------------- main
def main():
    if '--selftest' in sys.argv[1:]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(
        description='expc1 CPU-f64 companion legs (off-fleet, '
                    'self-receipted)')
    ap.add_argument('out_dir')
    ap.add_argument('--ckpt-root', default='')
    ap.add_argument('--val-cache', default='')
    ap.add_argument('--labels-npz', default='')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--parts', default=','.join(PARTS_ALL))
    ap.add_argument('--n-eval', type=int, default=0)
    ap.add_argument('--w2-record', default=os.path.join(
        _HERE, '..', 'results', 'W2-rescore-std',
        'W2-rescore-std_result.json'))
    ap.add_argument('--pb-record', default=os.path.join(
        _HERE, '..', 'results', 'PB-rescore', 'PB-rescore_result.json'))
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    parts = [p for p in args.parts.split(',') if p]
    bad = [p for p in parts if p not in PARTS_ALL]
    if bad:
        raise SystemExit('%s unknown parts %r; known: %s'
                         % (TAG, bad, ','.join(PARTS_ALL)))
    args.parts = tuple(p for p in PARTS_ALL if p in parts)
    sys.exit(run_compute(args))


if __name__ == '__main__':
    main()
