"""Campaign configuration: model registry, task registry, worker queues.

Single source of truth for the referee-response computation campaign.
Every number here is either (a) fixed by the manuscript / round-3 git
archaeology (notebook commit 7b87e0f) or (b) a new, documented decision
recorded in campaign/report/initial_report.tex.
"""
import os

# ---------------------------------------------------------------- paths
HOME = os.path.expanduser("~")
SMOKE = os.environ.get("CAMPAIGN_SMOKE", "0") == "1"      # tiny local CPU test mode
# smoke runs live in a separate scratch so their done-markers/checkpoints never
# shadow production state
WORKDIR = os.path.join(HOME, "workspace",
                       "campaign_smoke" if SMOKE else "campaign")
DATA_DIR = os.path.join(WORKDIR, "datasets")
CKPT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR = os.path.join(WORKDIR, "results")
STATUS_DIR = os.path.join(WORKDIR, "status")
GCS_BUCKET = "gs://iflp-486215-fermionicml-campaign"

VM_NAME = os.environ.get("CAMPAIGN_VM", "localvm")
WORKER_ID = int(os.environ.get("CAMPAIGN_WORKER", "0"))

# ------------------------------------------------------- physics: d=20 era
D20 = dict(D_SP=20, N_ELEC=10, PAIRS=True, SCALE_FACTOR=10.0, M_PAIRS=10,
           DS0=252, GPU_BATCH_SIZE=256)
# ------------------------------------------------------- physics: d=12 era
D12 = dict(D_SP=12, N_ELEC=6, PAIRS=False, SCALE_FACTOR=1.0, M_PAIRS=6,
           DN=924, GPU_BATCH_SIZE=64)

G_INIT, G_STOP = 0.1, 1.0          # G_ij ~ U(0.1, 1.0); diag-mean gauge 0.55
BETA_THERMAL = 1.0
BETA_GS = 100.0                     # ground-state projector (validated by task beta100_*)

# dataset seeds: two halves replicate the original 2-host structure (42, 43);
# validation seed 1007 is NEW (round-3 used 43, which collides with train half 2 -> leakage; documented)
TRAIN_SEEDS = (42, 43)
VAL_SEED = 1007
INIT_SEED = 42

# GEVP diagnostics
LAMBDA_RIDGE = 1e-9
SNORM_CUTOFF = 1e-7
LAMBDA_POS_THRESHOLD = 1e-12        # positive-lambda classification (round-3 corrected value)
RIDGE_SWEEP = (1e-11, 1e-10, 1e-9, 1e-8, 1e-7)
CUTOFF_SWEEP = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5)
GEVP_NSAMPLES = 4096                # per trained panel (const panel published with 4096)
GEVP_NSAMPLES_UNTRAINED = 2048

# finite-shot protocol (Sec V.C): 100 Hamiltonians x 100 draws x 50 shot points
SHOTS_N_STATES = 100
SHOTS_N_NOISE = 100
SHOTS_SWEEP_LO, SHOTS_SWEEP_HI, SHOTS_SWEEP_NPTS = 1e2, 1e10, 50
WLS_SPECTRAL_CUTOFF = 1e-9
WLS_DROP_P = 1e-12

# ablation-eval finite-shot rows (identical seeded draws across all models)
ABL_SHOT_BUDGETS = (1e4, 1e5, 1e6, 1e7, 1e8)
ABL_SHOTS_N_STATES = 20
ABL_SHOTS_N_NOISE = 20

# beta=100 projector validation
BETA100_NSAMPLES = 512

# kernel ridge baseline (declared protocol; full-dataset KRR is O(N^3)-infeasible)
KRR_N_TRAIN = 20000
KRR_N_VAL = 5000
KRR_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2)
KRR_GAMMAS = ("scale", 0.1, 1.0, 10.0)

# Richardson-Gaudin overlay
RG_N_TARGETS = 1024

# ---------------------------------------------------------------- model registry
# arch: ogn | mlp ; toggles only meaningful for ogn
# ens:  (h_type, state_type, beta_label)
_PROD_OPT = dict(batch_size=256, peak_lr=3e-4, weight_decay=1e-4, clip=1.0)

MODELS = {
    # --- production panels ---
    "prod_thermal_random": dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "prod_gs_random":      dict(era="d20", h_type="random", state_type="gs",
                                num_samples=5_000_000, epochs=50, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "prod_gs_const":       dict(era="d20", h_type="const", state_type="gs",
                                num_samples=2_000_000, epochs=10, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "nearinit_random":     dict(era="d20", h_type="random", state_type="gs",
                                num_samples=10_000, epochs=1, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    # --- d=12 models for the BCS panels (OGN; the lost original was CNN-era — documented decision) ---
    "d12_const":           dict(era="d12", h_type="const", state_type="gs",
                                num_samples=1_000_000, epochs=10, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "d12_vect":            dict(era="d12", h_type="vect", state_type="gs",
                                num_samples=1_000_000, epochs=10, loss="gram",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    # --- [ABL-REF] suite: identical thermal ensemble, identical budget (25 ep), matched params ---
    "abl_gram_s43":        dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=43, **_PROD_OPT),
    "abl_gram_s44":        dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=44, **_PROD_OPT),
    "abl_rdm_s42":         dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="rdm",
                                arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "abl_rdm_s43":         dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="rdm",
                                arch="ogn", res=3, init_seed=43, **_PROD_OPT),
    "abl_rdm_s44":         dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="rdm",
                                arch="ogn", res=3, init_seed=44, **_PROD_OPT),
    # DeepResMLP res=4 (hidden 1024, 8 blocks) ~= 1.74e7 params vs OGN 1.776e7 (within 2%)
    "abl_mlp":             dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="mlp", res=4, init_seed=42, **_PROD_OPT),
    "abl_noscatter":       dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=42, use_scatter=False, **_PROD_OPT),
    "abl_noreinject":      dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=42, use_reinject=False, **_PROD_OPT),
    "abl_noembed":         dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=42, use_orb_emb=False, **_PROD_OPT),
    "abl_neutralbias":     dict(era="d20", h_type="random", state_type="thermal",
                                num_samples=5_000_000, epochs=25, loss="gram",
                                arch="ogn", res=3, init_seed=42, readout_bias=0.0, **_PROD_OPT),
    # --- June-2026 follow-up: rdm-convergence + covariance-under-rdm + representability ---
    # converged (50-epoch) rdm vs matched gram, on both the thermal (V.C) and the
    # GS (V.B covariance) ensembles. gram-GS-50 already exists as prod_gs_random.
    "rdm_thermal_random_50":      dict(era="d20", h_type="random", state_type="thermal",
                                       num_samples=5_000_000, epochs=50, loss="rdm",
                                       arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "rdm_thermal_random_50_s43":  dict(era="d20", h_type="random", state_type="thermal",
                                       num_samples=5_000_000, epochs=50, loss="rdm",
                                       arch="ogn", res=3, init_seed=43, **_PROD_OPT),
    "gram_thermal_random_50":     dict(era="d20", h_type="random", state_type="thermal",
                                       num_samples=5_000_000, epochs=50, loss="gram",
                                       arch="ogn", res=3, init_seed=42, **_PROD_OPT),
    "gram_thermal_random_50_s43": dict(era="d20", h_type="random", state_type="thermal",
                                       num_samples=5_000_000, epochs=50, loss="gram",
                                       arch="ogn", res=3, init_seed=43, **_PROD_OPT),
    "rdm_gs_random_50":           dict(era="d20", h_type="random", state_type="gs",
                                       num_samples=5_000_000, epochs=50, loss="rdm",
                                       arch="ogn", res=3, init_seed=42, **_PROD_OPT),
}

# ---------------------------------------------------------------- [V2] production-run v2 lanes
# Additive, env-gated (CAMPAIGN_V2=1): the v2 registry (ognrepro.v2.registry)
# is mirrored as MODELS["v2::<lane>"] = dict(v2=True, lane=<lane>, ...) and
# tasks train_v2::<lane>; tasks/t_train.run dispatches spec['v2'] lanes to
# tasks/t_train_v2.run.  With the flag unset nothing below changes.
V2 = os.environ.get("CAMPAIGN_V2", "0") == "1"
if V2:
    import sys as _sys
    _LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebook_release", "src", "lib")
    if _LIB not in _sys.path:
        _sys.path.insert(0, _LIB)
    from ognrepro.v2 import registry as _V2R
    for _lane, _vs in _V2R.V2_LANES.items():
        MODELS["v2::" + _lane] = dict(v2=True, lane=_lane, era=_vs["era"], h_type=_vs["h_type"],
                                      state_type=_vs["state_type"], num_samples=_vs["num_samples"],
                                      epochs=_vs["epochs"], loss=_vs["cfg"].get("loss", "metric"),
                                      arch=_vs["arch"], res=_vs["res"], init_seed=_vs["init_seed"],
                                      tier=_vs["tier"], **_PROD_OPT)

if SMOKE:  # tiny overrides for local CPU validation
    for _m in MODELS.values():
        _m["num_samples"] = 512
        _m["epochs"] = 1
        _m["res"] = 1
        _m["batch_size"] = 64

# ---------------------------------------------------------------- task registry
# kind: train | analysis ; requires: task names that must be DONE first
TASKS = {
    # training tasks (one per model; [V2] lanes appear as train_v2::<lane> when CAMPAIGN_V2=1)
    **{f"train_{m}": dict(kind="train", model=m, requires=[]) for m in MODELS},

    # analysis tasks
    "shots_thermal":  dict(kind="analysis", entry="tasks.t_shots:run",
                           requires=["train_prod_thermal_random"]),
    "gevp_random":    dict(kind="analysis", entry="tasks.t_gevp_diag:run_random",
                           requires=["train_prod_gs_random", "train_nearinit_random"]),
    "gevp_const":     dict(kind="analysis", entry="tasks.t_gevp_diag:run_const",
                           requires=["train_prod_gs_const"]),
    "nullmode_const": dict(kind="analysis", entry="tasks.t_nullmode:run",
                           requires=["train_prod_gs_const"]),
    "beta100_random": dict(kind="analysis", entry="tasks.t_beta100:run_random", requires=[]),
    "beta100_const":  dict(kind="analysis", entry="tasks.t_beta100:run_const", requires=[]),
    "d12_figures":    dict(kind="analysis", entry="tasks.t_rg_overlay:run",
                           requires=["train_d12_const", "train_d12_vect"]),
    "gbase_probe":    dict(kind="analysis", entry="tasks.t_ablation_eval:run_gbase_probe",
                           requires=[]),
    "kernel_ridge":   dict(kind="analysis", entry="tasks.t_kernel_ridge:run", requires=[]),
    "ablation_eval":  dict(kind="analysis", entry="tasks.t_ablation_eval:run",
                           requires=["train_prod_thermal_random"] +
                                    [f"train_{m}" for m in MODELS if m.startswith("abl_")] +
                                    ["kernel_ridge"]),
    "values_assemble": dict(kind="analysis", entry="tasks.t_values:run", requires=[],
                            rerunnable=True),

    # --- June-2026 follow-up analyses ---
    "gevp_rdm_gs":     dict(kind="analysis", entry="tasks.t_gevp_diag:run_rdm_gs",
                            requires=["train_rdm_gs_random_50"]),
    "shots_rdm_thermal": dict(kind="analysis", entry="tasks.t_shots:run_rdm",
                            requires=["train_rdm_thermal_random_50"]),
    "shots_gram_thermal50": dict(kind="analysis", entry="tasks.t_shots:run_gram50",
                            requires=["train_gram_thermal_random_50"]),
    "repres_stress":   dict(kind="analysis", entry="tasks.t_repres:run",
                            requires=["train_gram_thermal_random_50",
                                      "train_rdm_thermal_random_50"]),
}

# ---------------------------------------------------------------- worker queues
# June-2026 follow-up campaign. These are MULTI-HOST slices: v4-16 (2 hosts),
# v5litepod-16 / v6e-16 (4 hosts), 4 chips/host => 10 independent single-host
# workers total. Queues are keyed by per-host worker id (0..3) and are IDENTICAL
# on all three VMs (redundancy by duplication: each VM's per-VM GCS prefix gets a
# complete replica). Within a VM the 4 worker-queues shard the work and run in
# parallel; on v4 (only workers 0,1) the two hosts drain queues 0,1 then ADOPT
# the orphaned queue-2/3 tasks (no worker 2/3 heartbeat), so v4 -- on-demand,
# preemption-proof -- remains the guaranteed full backstop.
# repres_stress (worker 2) depends on train_rdm_thermal (worker 0) AND
# train_gram_thermal (worker 2); within a VM all workers share the results prefix
# so the cross-worker done-marker is visible and deps_met gates it correctly.
QUEUES = {
    0: ["train_rdm_thermal_random_50", "shots_rdm_thermal"],
    1: ["train_rdm_gs_random_50", "gevp_rdm_gs"],
    2: ["train_gram_thermal_random_50", "shots_gram_thermal50", "repres_stress"],
    3: ["train_rdm_thermal_random_50_s43", "train_gram_thermal_random_50_s43"],
}
# After exhausting its own queue a worker adopts any unfinished task whose owner
# has been silent for ADOPT_AFTER_S (self-healing without hard locking; the 2-VM
# redundancy makes occasional duplicate work harmless).
ADOPT_AFTER_S = 45 * 60
HEARTBEAT_S = 300


def gcs_sync_dir(src, dst, timeout=3600):
    """Mirror directory src -> dst (either side may be a gs:// URL).

    Prefers `gcloud storage rsync -r`; the v5e image ships a gcloud too old to
    have that subcommand, so on failure falls back to `cp -r src <parent(dst)>/`
    (requires basename(src) == basename(dst), true at every call site).
    Best-effort: returns bool, never raises. No-op in smoke mode.
    """
    if SMOKE:
        return False
    import subprocess
    src = src.rstrip("/")
    dst = dst.rstrip("/")
    try:
        r = subprocess.run(["gcloud", "storage", "rsync", "-r", src, dst],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return True
        parent = dst.rsplit("/", 1)[0]
        r = subprocess.run(["gcloud", "storage", "cp", "-r", src, parent + "/"],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
