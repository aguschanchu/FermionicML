# fermionicml-notebooks-v2

Reproduction bundle for the **v2 production run** of the observable-to-generator
network (OGN) inverse-problem study.  It is a *second, versioned* distribution
alongside `fermionicml-notebooks-v1`: the v1 bundle is sealed and unchanged, and
nothing in this tree overwrites, edits or supersedes it.  Where the two overlap
(the vendored reference module, the d16 reference target sets, the deposit
manifest tooling) the bytes are identical by construction.

Everything here runs under one convention, `exact64_cpu_highest`: CPU backend,
`JAX_ENABLE_X64=1` armed **before** the first jax import, matmul precision
`highest` baked at trace time, f64 parameters, and f64 exact-diagonalisation
features rebuilt on the host — never a served f32 cache row.

## The v2 configuration

The v2 run changes three things relative to the published (v1) pipeline, all
carried behind one versioned object, `ognrepro.v2.config.V2Config`
(`cfg_version = 2`), whose resolved value is recorded in every lane's
`config.json` and reproduced in `tools/copy_lists/V2-CHECKPOINTS.snapshot.json`:

1. **HIGHEST-precision data.**  Both the Hamiltonian build and the pair-block
   contraction of the dataset generator run at `jax.lax.Precision.HIGHEST`
   (`data_precision="highest"`), and the cache carries a covariance
   (thermal) or ground-state (GS) sidecar used by the training loss.  The
   v1-era caches were generated at the default TPU matmul precision, i.e. from
   bf16-rounded Hamiltonians; the resulting label inconsistency sits *above*
   the v2 clean medians, so the data had to be regenerated before the v2
   numbers could carry any meaning.
2. **The exact affine readout on the dense-random family**
   (`readout="affine"`, `ognrepro.readout.project_affine`).  Two facts that
   are linear in the couplings — the generator's diagonal gauge
   `mean_k G_kk = 0.55` and the energy-shell identity
   `<G, rho>_F = sum_k pair_energies[k] rho_kk - E` — are imposed in closed
   form by a parameter-free 2x2 solve applied to the network's symmetric
   output.  The flax parameter tree is unchanged, so `n_params` is comparable
   across the v1/v2 pair.
3. **The covariance-based ("state-aware metric") loss with a log-linear
   lambda anneal** (`loss="metric"`, `ognrepro.v2.losses`):
   `L = mean(q_M + lam(step) * q_Sinf) + w_ridge * ||dw||^2`, with
   `q_M = Var_rho_true(dH)`, `q_Sinf = dw^T (S - u u^T) dw`, and `lam`
   annealed log-linearly from `lam0` to `lam1` over `anneal_frac` of the run.

With every v2 flag off (`V2Config.published()`) the code paths *are* the
published ones.  That control is shipped: `v1chk_std_thermal_random_s42` is a
v1-configuration lane trained on the v2 data, scored on exactly the rows every
v2 lane is scored on.

## Registry snapshot

This distribution was built against `results_v2/registry/V2-CHECKPOINTS.json`
**as of 2026-09-11T18:12:16Z** (40 lanes registered at that instant, the last of
them `v2abl_rdm_s42`); the bundle was assembled on 2026-09-11, and the exact
freeze instant is stamped in the header of
`tools/copy_lists/checkpoints_v2.tsv` and as `SNAPSHOT_UTC` in
`tools/expected_hashes_v2.py`.  The frozen copy of the registry travels with
the bundle as `tools/copy_lists/V2-CHECKPOINTS.snapshot.json`, and
`tools/expected_hashes_v2.py` carries the same lane -> `final_state` digest map
as importable data.

**This snapshot is the final registry.**  Every v2 lane has landed and is
registered: the Tier C ablation lanes `v2abl_noreinject_s42`, `v2ogn_mse_s42`,
`v2abl_neutralbias_s42` and `v2gram_thermal_random_50_s42`, and the `rdm` arm
`v2abl_rdm_s42`, which were still training when the previous snapshot
(2026-09-10T17:53:35Z, 35 lanes) was taken, are now bundled alongside the
Tier C lanes that had already landed then (`v2abl_noscatter_s42`,
`v2ogn_noE_s42`, `v2abl_noembed_s42`).  This build supersedes the
2026-09-10 snapshot build in full; it followed the recipe in the "Rebuilding
against the final registry" section (roster `v2-dist-2026-09-11-final`), and
nothing else about the build changed.  One record note: the registry entry
for `v2ogn_noE_s42` cites `V2-TRAIN-v2ogn_noE_s42`, whose `DONE.json` records
`gates_pass: false`; the training record of record for that lane is
`V2-TRAIN-v2ogn_noE_s42-r2` (`gates_pass: true`).  Both are indexed in
`data/provenance/v2_records_index.json`.

## Checkpoints in this distribution

Checkpoints are ~203 MiB each, so the bundle carries a **flagship-class
roster** rather than every registered lane: every lane a notebook restores,
every lane `w6_score.py --v2` scores, and one representative of each v2
configuration family.  Seed replicas whose only role is a band are represented
by their sealed medians in `data/values/`, not by weights.

### Included (27 lanes)

| tier | lane | data_precision | readout | loss | aux | n_params | final_state sha256 (16) |
|---|---|---|---|---|---|---:|---|
| d16prog | `v2d16progorig_const_gs_s42` | highest | published | metric | psi0 | 17771458 | `d22ae7652a7550e7` |
| d16prog | `v2d16progorig_vect_gs_s42` | highest | published | metric | psi0 | 17772228 | `d1d3d27ef49055fd` |
| d16prog | `v2d16progstd_const_gs_s42` | highest | published | metric | psi0 | 17771458 | `54d00e19a966c7df` |
| d16prog | `v2d16progstd_vect_gs_s42` | highest | published | metric | psi0 | 17772228 | `fab7a27d67fd4f2b` |
| d20 | `v1chk_std_thermal_random_s42` | default | published | gram | None | 17756929 | `9c72bf5d12038b82` |
| d20 | `v2abl_mlp_s42` | highest | published | metric | M | 17430599 | `1ba6645cdb1e3629` |
| d20 | `v2abl_neutralbias_s42` | highest | affine | metric | M | 17756929 | `805b4b7672deb44a` |
| d20 | `v2abl_noembed_s42` | highest | affine | metric | M | 17744321 | `1ee6b58035dc7a95` |
| d20 | `v2abl_noreinject_s42` | highest | affine | metric | M | 17753089 | `3e529480d3230620` |
| d20 | `v2abl_noscatter_s42` | highest | affine | metric | M | 14803969 | `a3b66b9625d984c1` |
| d20 | `v2abl_rdm_s42` | highest | affine | rdm | None | 17756929 | `2cf68c44dcf55776` |
| d20 | `v2gram_thermal_random_50_s42` | highest | affine | gram | None | 17756929 | `16be3ef09470e427` |
| d20 | `v2gs_const_s42` | highest | published | metric | psi0 | 17778818 | `90bdb57439342c68` |
| d20 | `v2lc_thermal_1e6_s42` | highest | affine | metric | M | 17756929 | `9c5d5964f25fa8a2` |
| d20 | `v2lc_thermal_2e6_s42` | highest | affine | metric | M | 17756929 | `650ea581753534bb` |
| d20 | `v2lc_thermal_2p5e5_s42` | highest | affine | metric | M | 17756929 | `1b956c1f74e40c8c` |
| d20 | `v2lc_thermal_5e5_s42` | highest | affine | metric | M | 17756929 | `0731dea3d167fbc7` |
| d20 | `v2mlp_cap_s42` | highest | published | metric | M | 17430599 | `e347e4a03208da37` |
| d20 | `v2mlp_gs_random_s42` | highest | published | metric | psi0 | 17430599 | `9fcf5967c5e91e2a` |
| d20 | `v2ogn_mse_s42` | highest | affine | mse | None | 17756929 | `2d8e62c2fa06e268` |
| d20 | `v2ogn_noE_s42` | highest | affine | metric | M | 17744577 | `367f245b1b6e7ec2` |
| d20 | `v2std_thermal_random_s42` | highest | affine | metric | M | 17756929 | `f5b7e4cc6d2f0bff` |
| d20 | `v2std_thermal_random_s43` | highest | affine | metric | M | 17756929 | `64dbdb91bf87a445` |
| d20 | `v2std_thermal_random_s44` | highest | affine | metric | M | 17756929 | `75a4e760b1b0eafe` |
| d20 | `v2stdgs_gram_s42` | highest | affine | gram | None | 17756929 | `1ab83f4a8ba803e7` |
| d20 | `v2stdgs_random_s42` | highest | affine | metric | psi0 | 17756929 | `f5784b584d6f2aae` |
| d20 | `v2stdgs_twin_s42` | highest | affine | metric | psi0 | 17756929 | `80323aab2e7936fd` |

### Excluded, and why

- `v2d16prog*_s43` — seed replica; sealed V2-D16SCORE medians carried in data/values/v2_d16score_values.json
- `v2d16prog*_s44` — seed replica; sealed V2-D16SCORE medians carried in data/values/v2_d16score_values.json
- `v2std_thermal_random_s43_xhost` — byte-different cross-host replica of s43 with an identical sealed median; the cross-host claim is carried by the sealed record V2-XFLEET-01, not by 203 MiB of weights
- `v2stdgs_random_s43` — seed replica; sealed medians carried in data/values/v2_clean_medians.json
- `v2stdgs_random_s44` — seed replica; sealed medians carried in data/values/v2_clean_medians.json
- `v2stdgs_twin_s43` — seed replica; sealed medians carried in data/values/v2_clean_medians.json
- `v2stdgs_twin_s44` — seed replica; sealed medians carried in data/values/v2_clean_medians.json

Every excluded lane's sealed numbers are still present: `data/values/`
carries the full `V2-CLEAN` / `V2-D16SCORE` median tables for **all** sealed
lanes, each with its record@sha citation, and
`data/provenance/v2_records_index.json` indexes every sealed v2 record with
its `RESULT_MANIFEST` digest.

## Layout

```
notebooks/     A-D, the v2 notebooks (see below)
lib/ognrepro/  the reproduction library, including the v2/ package
lib/FermionicML/
               the staged v2 engine surface (engine.py, engine_parts/p1..p6,
               config.py, tasks/) carrying the [V2] keyword additions, plus
               the two SHA-printed reference modules copied byte-for-byte out
               of the sealed v1 staging area
lib/expc1_*.py the sealed campaign-1 scoring primitives (panel_matrices,
               score_pred_panel, boot_median_ci, features_from_VE), vendored
               so the honest score path needs nothing outside the bundle
checkpoints/d20/<lane>/{config.json,final_state.msgpack[,std_stats.npz]}
checkpoints/d16prog/<lane>/...
               epoch_state.msgpack is deliberately absent
data/values/   the replay layer: v2_fig*_values.json shipped VERBATIM from the
               sealed V2-FIG* records, plus authored sidecars (clean medians,
               d16 scores, shots, CMP-01, geometry, T0 pair, KRR, cross-eval,
               robustness, cross-host) each carrying a record@sha provenance
               block
data/streams/  v2_val1007_labels_first4096.npy -- the pinned seed-1007
               validation rows 0..4095 (rows sha256 94f616fb..) every honest
               v2 clean number is computed on
data/percells/ the sealed D16PROG-P1-REF reference target sets
data/provenance/
               v2_records_index.json, the registry snapshot, and the v2
               staging manifest and notes
tools/         manifest_tools.py (deposit-manifest-v2), expected_hashes_v2.py,
               the frozen copy lists and the per-file dist digest
scripts/, env/ the training drivers and the pinned environment
```

## Notebooks

| notebook | what it does |
|---|---|
| `A_v2_model_and_data.ipynb` | the v2 configuration table, the registry-vs-bytes checkpoint check, the pinned stream and its gauge identity, HIGHEST-precision f64 exact-ED features with a determinism gate, the d16 reference surfaces |
| `B_v2_ogn_restore_and_scores.ipynb` | `restore_v2` on the bundled lanes, the honest f64 forward, and the sealed clean medians reproduced on the same rows (v2 vs the v1-configuration control); the d16prog forward-equivalence leg |
| `C_v2_geometry_and_comparators.ipynb` | the sealed geometry, comparator, finite-shot and robustness records, with each record's *decision* re-derived from its inputs |
| `D_v2_results_reproduction.ipynb` | every v2 figure re-drawn from its values sidecar, bit-identity asserted against a fresh disk read and against the declared `sha256_float64`, and every record@sha citation checked against the record's own manifest |

**Runtime knob.**  The notebooks read `V2NB_SMOKE` (default `1`) and score a
reduced number of rows so the whole set executes in about a quarter of an hour
on a four-core workstation.  `V2NB_SMOKE=0` restores the full pre-registered
surfaces (4096 stream rows, 1024 d16 target rows); `V2NB_N_D20`,
`V2NB_N_D16`, `V2NB_N_FEAT` and `V2NB_LANES` override individually.  A reduced
run never compares against the 4096-row number: it compares against the
**sealed median over the same rows**, transcribed from each record's own
per-sample array (`prefix_median_err_q` in `data/values/v2_clean_medians.json`).

## Verifying this distribution

```
python3 notebook_release/build/verify_battery.py --bundle v2      # 7 checks
python3 notebook_release/build/w6_score.py --v2 \
        --bundle-root notebook_release/dist/fermionicml-notebooks-v2
```

`w6_score.py --v2` restores every bundled d20 lane **from the bundle**
(`OGNREPRO_ROOT` and `OGNREPRO_ENGINE_DIR` point into this tree), rebuilds the
f64 exact-ED features for the pinned rows, and gates the ratio of medians
against the sealed `V2-CLEAN` values at the **pre-registered band 1.05**.  This
is a different surface from the v1 scorer's d16 retrain band (1.0248) and the
two are not interchangeable; see the module docstring.

## Rebuilding against the final registry

This build is the product of that rebuild (roster `v2-dist-2026-09-11-final`,
27 checkpoints: 23 d20 + 4 d16prog).  To rebuild against a later
registry, extend `production_v2/dist/v2_roster.json` and re-run:

```
python3 production_v2/dist/pull_v2_checkpoints.py --stage <STAGE>
python3 notebook_release/build/make_v2_values.py
python3 notebook_release/build/make_v2_notebooks.py
python3 production_v2/dist/stage_v2_engine.py
python3 notebook_release/build/freeze_v2_lists.py --pass 1 --staging <STAGE>
python3 notebook_release/build/build_bundle.py --bundle v2 --staging <STAGE> \
        --phase layout,code,data,checkpoints,src
python3 notebook_release/build/freeze_v2_lists.py --pass 2
python3 notebook_release/build/build_bundle.py --bundle v2 --staging <STAGE> --phase code
python3 notebook_release/build/build_bundle.py --bundle v2 --staging <STAGE> --phase manifest,tar
python3 notebook_release/build/verify_battery.py --bundle v2
python3 notebook_release/build/w6_score.py --v2
```

The v1 distribution is not touched by any of these: `--bundle v1` remains the
default of every tool and reproduces the v1 behaviour byte for byte.

## License

MIT (`LICENSE`), inherited from the FermionicML reference module.
