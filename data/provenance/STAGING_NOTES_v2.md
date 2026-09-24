# FermionicML staged tree -- v2 engine surface

Staged 2026-09-11 15:45:00 -0300 from the LIVE campaign engine (the tree carrying the
`[V2]` keyword additions in `engine_parts/p1_core.py`,
`engine_parts/p3_training.py` and `config.py`).  The v1 acceptance
staging area `release_bundle_v2/acceptance-public/code/
FermionicML-staged/` is SEALED and was not modified; this is a
second, independent staged tree for the v2 distribution.

## Sources

- `FermionicML/FermionicML_thermal_2body.py`  <-  `release_bundle_v2/acceptance-public/code/FermionicML-staged/FermionicML/FermionicML_thermal_2body.py`
- `FermionicML/LICENSE`  <-  `release_bundle_v2/acceptance-public/code/FermionicML-staged/FermionicML/LICENSE`
- `FermionicML/README.md`  <-  `campaign/README.md`
- `FermionicML/config.py`  <-  `campaign/config.py`
- `FermionicML/engine.py`  <-  `campaign/engine.py`
- `FermionicML/engine_parts/p1_core.py`  <-  `campaign/engine_parts/p1_core.py`
- `FermionicML/engine_parts/p2_models.py`  <-  `campaign/engine_parts/p2_models.py`
- `FermionicML/engine_parts/p3_training.py`  <-  `campaign/engine_parts/p3_training.py`
- `FermionicML/engine_parts/p4_gevp.py`  <-  `campaign/engine_parts/p4_gevp.py`
- `FermionicML/engine_parts/p5_shots.py`  <-  `campaign/engine_parts/p5_shots.py`
- `FermionicML/engine_parts/p6_bcs_rg.py`  <-  `campaign/engine_parts/p6_bcs_rg.py`
- `FermionicML/generator.py`  <-  `release_bundle_v2/acceptance-public/code/FermionicML-staged/FermionicML/generator.py`
- `FermionicML/requirements.txt`  <-  `release_bundle_v2/acceptance-public/code/FermionicML-staged/FermionicML/requirements.txt`
- `FermionicML/runner.py`  <-  `campaign/runner.py`
- `FermionicML/tasks/__init__.py`  <-  `campaign/tasks/__init__.py`
- `FermionicML/tasks/t_ablation_eval.py`  <-  `campaign/tasks/t_ablation_eval.py`
- `FermionicML/tasks/t_beta100.py`  <-  `campaign/tasks/t_beta100.py`
- `FermionicML/tasks/t_gevp_diag.py`  <-  `campaign/tasks/t_gevp_diag.py`
- `FermionicML/tasks/t_kernel_ridge.py`  <-  `campaign/tasks/t_kernel_ridge.py`
- `FermionicML/tasks/t_nullmode.py`  <-  `campaign/tasks/t_nullmode.py`
- `FermionicML/tasks/t_repres.py`  <-  `campaign/tasks/t_repres.py`
- `FermionicML/tasks/t_rg_overlay.py`  <-  `campaign/tasks/t_rg_overlay.py`
- `FermionicML/tasks/t_shots.py`  <-  `campaign/tasks/t_shots.py`
- `FermionicML/tasks/t_train.py`  <-  `campaign/tasks/t_train.py`
- `FermionicML/tasks/t_train_v2.py`  <-  `campaign/tasks/t_train_v2.py`
- `FermionicML/tasks/t_values.py`  <-  `campaign/tasks/t_values.py`

## Byte-fidelity gates (verified at staging)

- `FermionicML/FermionicML_thermal_2body.py` sha256 `969a6361a763879593916c04094b381dd48ec938cbf7a31f00c2b750821abfc4` == the SM-printed reference hash (PASS,
  copied byte-for-byte out of the sealed v1 staging area).
- `FermionicML/generator.py` sha256 `d2cbf9803a4e63d27f0bd24250d6764b99db6b3995a68c9b6804bd513622bfd4` == the SM-printed reference hash (PASS,
  copied byte-for-byte out of the sealed v1 staging area).

- Full tree: `STAGING_MANIFEST.sha256` (26 files,
  deposit-manifest-v2 entry grammar).
