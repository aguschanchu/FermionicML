# Referee-response computation campaign

Standalone, unattended campaign answering the `[AWAITING-RUN]` items of the PRA
referee report (`../referee_report.tex`). Runs redundantly on two TPU pods
(`tpu-v5e-spot-eu-b`, `tpu-v6e-spot-eu-a`), 4 independent single-host workers
each, no SSH tunnel required after launch.

## Layout
- `config.py` — model registry, task registry, per-worker queues, all protocol constants.
- `engine.py` + `engine_parts/p1..p6` — notebook-faithful physics/ML engine
  (one shared exec namespace, see engine.py docstring).
- `tasks/` — thin task drivers (`t_train`, `t_gevp_diag`, `t_shots`, ...).
- `runner.py` — per-worker queue executor: done-markers, dependency checks,
  heartbeats, orphan adoption, best-effort GCS sync.
- `run_campaign.sh` — on-VM launcher (sets single-host TPU env vars, nohup).
- `deploy.sh` — local: scp package to all workers, provision, launch.
- `report/initial_report.tex` — the campaign plan document.
- `reference/` — previous campaign baselines (values_prev.json, RESULTS_prev.md).

## Artifacts
- Results: `gs://iflp-486215-fermionicml-campaign/results/<vm>/<task>/`
- Checkpoints: `gs://iflp-486215-fermionicml-campaign/checkpoints/<vm>/<model>/`
- Heartbeats: `gs://iflp-486215-fermionicml-campaign/status/<vm>/workerN.json`

## Operations
- Local CPU smoke test: `CAMPAIGN_SMOKE=1 CAMPAIGN_WORKER=2 python runner.py`
  (GCS disabled in smoke mode).
- Status check: `gcloud storage cat gs://iflp-486215-fermionicml-campaign/status/<vm>/worker0.json`
- After spot preemption: restart the VM, then re-run
  `bash deploy.sh --launch` — runners resume from GCS done-markers and
  per-epoch checkpoints.
