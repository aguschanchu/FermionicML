# RETRAIN.md — full-scale retraining of the bundled checkpoints

This bundle ships every published checkpoint; you never need to retrain to
run the notebooks. This document is for reproducing the *training* itself:
the 12 d16prog lanes and the 14 d20/d12-era models, via
`scripts/train_lane.py` (specs transcribed from the campaign registries;
`--list` prints the roster).

> **WARNING — never point `--out`, `--cache-dir`, or `--cache-root` at the
> research repository's `results/` trees (or at this bundle's own
> `checkpoints/`).** Those are sealed records. Retraining writes into a
> fresh directory of your own (`./retrain_out` by default); `build_cache.py`
> refuses any cache root that resolves under a `**/results/` path.

## 1. TPU VM setup (full-scale d16prog / d20 lanes)

The d16prog lanes were trained on TPU v4 with **4 local chips (ndev=4)** —
a `v4-8` VM, or ONE worker of a `v4-16`. The registered d16 label stream
splits over exactly 4 devices, so use a 4-chip host.

```bash
gcloud compute tpus tpu-vm create retrain-v4 \
    --zone <zone> --accelerator-type v4-8 \
    --version tpu-ubuntu2204-base
gcloud compute tpus tpu-vm ssh retrain-v4 --zone <zone>
```

On the VM (Ubuntu 22.04 ships python3.10, matching the pinned stack):

```bash
sudo apt-get update && sudo apt-get install -y python3.10-venv
python3.10 -m venv ~/ognrepro-pinned
source ~/ognrepro-pinned/bin/activate
cd <bundle>/env
pip install -r requirements-pinned.txt     # includes the vendored
                                           # fermionic_mbody wheel
pip install libtpu==0.0.17                 # TPU runtime (TPU VMs only)
python -c "import jax; print(jax.devices())"   # expect 4 TPU v4 chips
python <bundle>/scripts/train_lane.py --device-report
```

## 2. d16 caches (build once, ~1–2 h on a many-core host)

The d16prog lanes train on pre-built blocked-solver caches. Build them on
any many-core CPU host (the TPU VM host CPUs work; so does a workstation):

```bash
# vect sector: labels are CPU-minted from the pinned replay chain
python <bundle>/scripts/build_cache.py vect_gs \
    --cache-root ~/d16_caches --nproc $(nproc)

# const sector: REUSES the digest-gated d14-extract label arrays
# (labels_{half0,half1,val}.npy) — a CPU replay can never reproduce that
# TPU-minted stream, so the arrays themselves are required:
python <bundle>/scripts/build_cache.py const_gs \
    --labels-dir <dir-with-labels_*.npy> \
    --cache-root ~/d16_caches --nproc $(nproc)
```

Each build ends with a **certificate digest check**: the per-split
labels/features/energy sha256 roll-ups are compared against the embedded
build-certificate constants (the same constants `train_lane.py` gates on).
A full build must print `MATCH` on every row — the rebuilt cache is then
bit-identical to the certified one. **Do not train on a cache that failed
the comparison.** `--selftest` (2 shards/split, ~1 min) sanity-checks the
solver + digest plumbing without a certificate claim.

## 3. Training a lane

```bash
# d16prog lane (TPU v4, ndev=4; ~850 s/lane — measured 842.6 s):
python <bundle>/scripts/train_lane.py d16progorig_const_gs \
    --cache-dir ~/d16_caches --out ~/retrain_out

# std arms additionally load the lane's standardization sidecar
# (default: <bundle>/checkpoints/d16prog/<lane>/std_stats.npz):
python <bundle>/scripts/train_lane.py d16progstd_const_gs \
    --cache-dir ~/d16_caches --out ~/retrain_out

# d20/d12-era model (dataset generated live by the engine on the TPU):
python <bundle>/scripts/train_lane.py prod_thermal_random \
    --out ~/retrain_out
```

`train_lane.py` verifies the d16 cache digests against the build
certificate BEFORE training (fail-closed), trains with the engine's own
`train_model` (identical loss/optimizer/seed conventions), then writes
`final_state.msgpack`, `hist.npy`/`hist.json`, and `config.json` under
`<out>/<lane>/`, printing sha256 + wall-time receipts.

Expected walls: **~850 s per d16prog lane** on v4 ndev=4 (measured
842.6 s). The d20 production lanes are larger jobs (5e6 samples × 25–50
epochs; hours on v4). `--seed-override N` retrains a lane at a different
init seed.

## 4. What "reproduced" means (read this)

**Retrained checkpoint SHAs WILL differ** from the published ones across
hardware, jax/libtpu versions, and even repeated runs on other hosts —
float accumulation order in pmap training is not bit-stable across
stacks. Equivalence is defined at the **forward-pass / score level**: score
the retrained checkpoint on the sealed reference targets (the P4 protocol
implemented in `verify_checkpoints.py --full` for the published weights)
and compare score medians. Two measured bands apply, at different depths:

- **Evaluating a GIVEN checkpoint is deterministic-class**: restoring a
  published checkpoint and forwarding the sealed targets under the CPU-f64
  convention reproduces the sealed per-target predictions at max|Δ| ≤ 1e-6
  (measured bit-exact for the const-std lane). `verify_checkpoints.py
  --full` asserts this.
- **A fresh RETRAIN is a new SGD trajectory**: expect its score median to
  land in the same band the campaign's own seeds span — the sealed 3-seed
  bands are ×2.7-wide for these lanes, and the campaign's predeclared
  seed-band resolution floor is **~3–5×** (nothing finer is readable).
  Our verification retrains on TPU v4 (one worker of a v4-16, pinned
  stack, bit-identical caches) landed **inside** the sealed 3-seed band
  (const std) and within 7% of its edge (vect orig), both on the
  *better* side. The tighter d20-era cross-fleet twin envelope
  (ratio ≤ 1.0248, record XFLEET-01) was measured for the long
  d20 productions and does NOT transfer to these short d16 lanes —
  do not expect it.

(The *caches* ARE byte-reproducible; the certificate check above proves
yours bit-identical. The *published checkpoints* are verified byte-level
by `verify_checkpoints.py --quick` and forward-level by `--full`.)

## 5. CPU smoke path (no TPU needed, minutes)

Any lane runs end-to-end on a laptop CPU at toy scale — 512 samples,
1 epoch, res 1, batch 64:

```bash
python <bundle>/scripts/train_lane.py prod_thermal_random --smoke
python <bundle>/scripts/train_lane.py d16progorig_const_gs --smoke
```

d16 smoke lanes generate their 512-row blocked cache on the fly (CPU-
minted labels — for the const sector that is the disclosed cpu stream
class, fine for smoke; no certificate claim). Smoke runs validate the
plumbing, not the physics: losses are finite but untrained-quality.
