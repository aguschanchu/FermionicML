"""Pure-numpy checks of the Tier-B v2 stage helpers (production_v2/stages/
v2_robustness.py, v2_crosseval.py, v2_timers.py): the ported bootstrap /
plateau / noise-stream / H0-arm / slope primitives, the crosseval row
arithmetic and the timer extractors on a synthetic lane dir.  No engine, no
checkpoint, no jax.

    JAX_PLATFORMS=cpu <pinned python> -m ognrepro.v2.tests.test_tierb_stages
"""
import json
import os
import sys
import tempfile

import numpy as np

from .. import testenv

REPO_ROOT = testenv.setup_env()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from production_v2.stages import v2_robustness as RB      # noqa: E402
from production_v2.stages import v2_crosseval as XE       # noqa: E402
from production_v2.stages import v2_timers as TM          # noqa: E402


def test_order_stats_and_plateau(ctx=None):
    reps = np.arange(1000, dtype=float)
    s = RB.order_stats(reps)
    assert s["p2.5"] == 25.0 and s["p97.5"] == 975.0 and s["p16"] == 160.0 and s["p84"] == 840.0
    sweep = np.geomspace(1e10, 1e2, 20)
    med = np.linspace(1.0, 2.0, 20)
    plat = RB.plateau_stat(sweep, med)
    assert plat == float(np.nanmedian(med[sweep >= 1e9]))


def test_paired_plateau_factors_identity(ctx=None):
    rng = np.random.default_rng(0)
    sweep = np.geomspace(1e10, 1e2, 6)[[0, 1, 2, 5]]
    base = np.abs(rng.standard_normal((4, 5, 3))) + 1.0
    errs = {"baseline": base, "arm": 3.0 * base}
    b, res = RB.paired_plateau_factors(errs, sweep, np.random.default_rng(1), 200, arms=["arm"])
    assert abs(res["arm"]["plateau_factor"] - 3.0) < 1e-12
    assert abs(res["arm"]["ci95"][0] - 3.0) < 1e-12 and abs(res["arm"]["ci95"][1] - 3.0) < 1e-12


def test_symmetrize_and_psd(ctx=None):
    x = np.random.default_rng(2).standard_normal((3, 10, 10, 1))
    s = RB.symmetrize_block(x)
    assert np.allclose(s[..., 0], np.swapaxes(s[..., 0], 1, 2))
    eye = np.broadcast_to(np.eye(10), (2, 10, 10)).copy()[..., None]
    assert RB.psd_violation_frac(eye) == 0.0
    assert RB.psd_violation_frac(-eye) == 1.0


def test_offman_noise_stream_pinned(ctx=None):
    a = RB.offman_noise(3, (4, 10, 10, 1), np.float64)
    b = RB.offman_noise(3, (4, 10, 10, 1), np.float64)
    c = RB.offman_noise(4, (4, 10, 10, 1), np.float64)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.dtype == np.float64
    ref = np.random.default_rng(np.random.SeedSequence([0, 7, 3])).standard_normal((4, 10, 10, 1))
    assert np.array_equal(a, ref)


def test_h0_arms_construction(ctx=None):
    e_pair = np.linspace(-0.9, 0.9, 10)
    arms, rms0 = RB.h0_build_arms(RB.H0_DELTAS, e_pair)
    assert set(arms) == set(RB.H0_ARMS_ORDER)
    assert abs(rms0 - np.sqrt(np.mean(e_pair ** 2))) < 1e-15
    for name, (ep, z, dl) in arms.items():
        if name == "nominal":
            assert dl == 0.0 and np.array_equal(ep, e_pair)
            continue
        assert abs(np.mean(z)) <= 1e-12
        assert abs(np.sqrt(np.mean((ep - e_pair) ** 2)) / rms0 - dl) <= 1e-12


def test_zero_inputs_and_slope(ctx=None):
    zin = RB.zero_G_inputs()
    assert tuple(gv for gv, _G in zin) == RB.ZERO_GRANGE
    assert np.all(zin[1][1] == 0.0) and zin[0][1][0, 0] == 0.0 and zin[2][1][0, 0] == 0.55 and zin[2][1][0, 1] == 0.02
    deltas = np.array(RB.H0_DELTAS)
    assert abs(RB.loglog_slope(deltas, 2.5 * deltas ** 1.0) - 1.0) < 1e-12
    disp = np.outer(deltas, np.ones(50)) * (1.0 + 0.01 * np.random.default_rng(5).standard_normal(50))
    lo, hi = RB.h0_slope_ci(disp, deltas, 20260825, B=50)
    assert abs(lo - 1.0) < 1e-6 and abs(hi - 1.0) < 1e-6


def test_shift_stream_expectation(ctx=None):
    meta = {"baseline": dict(era=dict(g_range=[0.1, 1.0]), g_batch_rows_sha256="g1", p_true_rows_sha256="p1"),
            "beta_0.5": dict(era=dict(g_range=[0.1, 1.0]), g_batch_rows_sha256="g1", p_true_rows_sha256="p2"),
            "grange_0.1_0.5": dict(era=dict(g_range=[0.1, 0.5]), g_batch_rows_sha256="g2", p_true_rows_sha256="p3")}
    ok, d = RB.shift_stream_expectation(meta)
    assert ok and d["n_boxes"] == 2 and d["n_distinct_g_shas"] == 2 and d["p_true_pairwise_distinct"]
    meta["beta_0.5"]["g_batch_rows_sha256"] = "gX"
    assert not RB.shift_stream_expectation(meta)[0]


def test_h_from_ops_identity(ctx=None):
    rng = np.random.default_rng(3)
    ops = rng.standard_normal((9, 4, 4))
    ops = 0.5 * (ops + np.swapaxes(ops, 1, 2))
    G = rng.standard_normal((3, 3))
    ep = rng.standard_normal(3)
    mocc = np.abs(rng.standard_normal((4, 3)))
    H = RB.h_from_ops(G, ep, ops, mocc)
    ref = -np.tensordot(G.ravel(), ops, axes=1)
    ref[np.diag_indices(4)] += mocc @ ep
    assert np.allclose(H, 0.5 * (ref + ref.T)) and np.allclose(H, H.T)


def test_crosseval_rows(ctx=None):
    rng = np.random.default_rng(4)
    n = 64
    bundle = {k: np.abs(rng.standard_normal(n)) + 0.5 for k in XE.PERSAMPLE_KEYS}
    bundle["lam_bar"] = np.full(n, 2.0)
    bundle["lam_eff"] = np.full(n, 0.5)
    assert XE.rom_of(bundle) == 4.0
    G = dict(TRUE_S=np.ones(n))
    BI = np.stack([rng.integers(0, n, n) for _ in range(64)])
    row = XE.arm_row("arm", bundle, G, BI, 1, 62)
    assert row["rom"]["value"] == 4.0 and row["rom_ci95"][0]["value"] == 4.0 and row["rom_ci95"][1]["value"] == 4.0
    assert row["rom"]["hex"] == float.hex(4.0)
    rows = {"a": dict(rom=XE._fp(1.0)), "b": dict(rom=XE._fp(3.0))}
    s = XE.xeval_summary(rows, {"a": 42, "b": 43})
    assert s["rom_mean"]["value"] == 2.0 and s["rom_spread_max_over_min"]["value"] == 3.0 and s["n_seeds"] == 2
    assert XE._rel(1.0, 0.0) == float("inf") and XE._rel(2.0, 1.0) == 1.0


def test_timers_extractors(ctx=None):
    es = [100.0, 50.0, 52.0, 51.0]
    st = TM._series_stats(es)
    assert st["C1_first"] == 100.0 and st["C2_median_excl_first"] == 51.0 and st["C3_sum"] == 253.0
    assert not st["C2_degraded_to_compile_inclusive"] and TM._series_stats([7.0])["C2_degraded_to_compile_inclusive"]
    th = TM.samples_per_s(256, 100, es)
    assert th["samples_per_epoch"] == 25600 and abs(th["per_epoch"][0] - 256.0) < 1e-12 and abs(th["median_excl_first"] - 25600 / 51.0) < 1e-9
    d = tempfile.mkdtemp(prefix="v2_timers_test_")
    lane = os.path.join(d, "v2std_thermal_random_s42")
    os.makedirs(lane)
    cfg = dict(lane="v2std_thermal_random_s42", spec=dict(era="d20", h_type="random", state_type="thermal", arch="ogn_std", init_seed=42,
                                                          epochs=4, batch_size=256), epoch_seconds=es, wall_s=300.0, total_steps=400,
               steps_per_epoch=100, epochs_run=4, n_params=17, final_state_sha256="ab", hist=[1, 2, 3, 4],
               env=dict(hostname="h", device_kind="TPU v4", backend="tpu", n_local_devices=4, jax="0.6.2"),
               dataset=dict(root="/x", schemas=dict(half0=dict(wall_s=10.0), val=dict(wall_s=1.0))))
    json.dump(cfg, open(os.path.join(lane, "config.json"), "w"))
    json.dump(dict(epoch=3, hist=dict(loss=[1, 2, 3, 4], epoch_seconds=es)), open(os.path.join(lane, "meta.json"), "w"))
    open(os.path.join(lane, "DONE"), "w").write("2026-09-10T00:00:00\n")
    row, consumed = TM.read_lane_dir(lane, "FAM-TEST")
    assert row["epoch_series_config_vs_meta_identical"] and row["epoch_series_len_matches_epochs_run"]
    assert row["V_C4_driver_wall_s"] == 300.0 and row["V_C6_s_per_epoch_incl"] == 75.0 and row["gaps"]["C1sum_vs_C4"]["delta_s"] == 47.0
    assert row["V_CG_cache_generation_wall_s"] == {"half0": 10.0, "val": 1.0} and sorted(row["clocks_present"]) == ["V-C1", "V-C2", "V-C3", "V-C4", "V-C6", "V-C7", "V-CG"]
    assert len(consumed) == 3
    tab = TM.audit_table([row])[0]
    assert tab["samples_per_s_V_C2"] == 25600 / 51.0 and tab["host"] == "h" and tab["device"] == "TPU v4"
    D = TM.demonstrations([row], [])
    assert set(D) >= {"D-1", "D-3", "D-4"} and D["D-1"]["delta_s"]["value"] == 47.0
    assert TM._receipt_hazard_keys(dict(all_gates_pass=True, x=dict(pass_=1))) == ["/x/pass_"]
    assert TM.FALSE_FRIENDS[0] == "N_s"


def main():
    n_ok = n_fail = 0
    for name in sorted(k for k in globals() if k.startswith("test_")):
        try:
            globals()[name](None)
            n_ok += 1
            print("PASS %s" % name)
        except Exception:  # noqa: BLE001
            import traceback
            n_fail += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    print("%d passed, %d failed" % (n_ok, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
