"""Pure-numpy checks of the second Tier-B pair (production_v2/stages/
v2_kernel_ridge.py, v2_beta1_panel.py): the KRR feature/kernel/solve
primitives, the config-pin reader and the cache-split reader, and the beta=1
panel's band/estimand helpers plus the binding row gate.  No engine, no
checkpoint, no jax op, no campaign geometry import.

    JAX_PLATFORMS=cpu <pinned python> -m ognrepro.v2.tests.test_tierb2_stages
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

from production_v2.stages import v2_beta1_panel as B1     # noqa: E402
from production_v2.stages import v2_kernel_ridge as KR    # noqa: E402


# --------------------------------------------------------------- v2_kernel_ridge
def test_krr_feature_extraction(ctx=None):
    rng = np.random.default_rng(0)
    R = rng.standard_normal((7, 10, 10))
    en = rng.standard_normal(7)
    Xf = KR.collect_features(R[..., None], en)
    assert Xf.shape == (7, 56), Xf.shape
    Rs = 0.5 * (R + np.transpose(R, (0, 2, 1)))
    iu = np.triu_indices(10, k=0)
    assert np.array_equal(Xf[:, :55], Rs[:, iu[0], iu[1]])
    assert np.array_equal(Xf[:, 55], np.asarray(en, np.float64))
    # the symmetrization is what the protocol declares: a symmetric block is a fixed point
    assert np.array_equal(KR.collect_features(Rs[..., None], en), Xf)


def test_krr_sqdist_matches_definition(ctx=None):
    rng = np.random.default_rng(1)
    A, Bm = rng.standard_normal((5, 4)), rng.standard_normal((3, 4))
    D = KR._sqdist(A, Bm)
    ref = np.array([[np.sum((a - b) ** 2) for b in Bm] for a in A])
    assert D.shape == (5, 3) and np.allclose(D, ref, rtol=0, atol=1e-10)
    assert np.all(D >= 0.0)
    assert np.allclose(np.diag(KR._sqdist(A, A)), 0.0, atol=1e-10)


def test_krr_fit_solves_the_ridge_system(ctx=None):
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((40, 6))
    Y = rng.standard_normal((40, 3))
    K = np.exp(-0.3 * KR._sqdist(Z, Z))
    alpha = 1e-4
    dual = KR._krr_fit(K, Y, alpha)
    assert dual.shape == (40, 3)
    resid = (K + alpha * np.eye(40)) @ dual - Y
    assert np.max(np.abs(resid)) < 1e-8, np.max(np.abs(resid))
    # bit-exact reproducibility of the same cell (the determinism gate's primitive)
    assert np.array_equal(dual, KR._krr_fit(np.exp(-0.3 * KR._sqdist(Z, Z)), Y, alpha))


def test_krr_gamma_scale_convention(ctx=None):
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((50, 56)) * 2.0
    n_feat = Z.shape[1]
    gamma_scale = 1.0 / (n_feat * max(Z.var(), 1e-12))
    assert abs(gamma_scale - 1.0 / (56 * Z.var())) < 1e-15
    gammas = [(g, gamma_scale if g == "scale" else float(g)) for g in KR.KRR_GAMMAS]
    assert [g for g, _ in gammas] == ["scale", 0.1, 1.0, 10.0]
    assert gammas[0][1] == gamma_scale and gammas[3][1] == 10.0
    assert len(gammas) * len(KR.KRR_ALPHAS) == 16


def test_krr_config_pins_match_campaign_config(ctx=None):
    p = os.path.join(REPO_ROOT, "campaign", "config.py")
    if not os.path.isfile(p):
        return
    cfg = KR.read_config_pins(p)
    assert cfg["KRR_N_TRAIN"] == KR.KRR_N_TRAIN == 20000
    assert cfg["KRR_N_VAL"] == KR.KRR_N_VAL == 5000
    assert cfg["KRR_ALPHAS"] == KR.KRR_ALPHAS
    assert cfg["KRR_GAMMAS"] == KR.KRR_GAMMAS
    assert cfg["TRAIN_SEEDS"] == KR.TRAIN_SEEDS and cfg["VAL_SEED"] == KR.VAL_SEED
    assert len(cfg["_sha256"]) == 64


def test_krr_split_reader_order_and_accounting(ctx=None):
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "half0")
        os.makedirs(d)
        rng = np.random.default_rng(4)
        rows = []
        for i in range(3):
            lab = np.asarray(rng.uniform(0.1, 1.0, (8, 55)), np.float32)
            rows.append(lab)
            np.savez(os.path.join(d, "shard_%05d.npz" % i), labels=lab,
                     energy=np.asarray(rng.standard_normal((8, 1)), np.float32),
                     features=np.asarray(rng.standard_normal((8, 10, 10, 1)), np.float32))
        json.dump(dict(seed=42, h_type="random", state_type="thermal", beta=1.0,
                       matmul_precision="highest", num_samples=24, complete=True),
                  open(os.path.join(d, "schema.json"), "w"))
        L, E, F, acct = KR.load_split_rows(d, 20, "train")
        assert L.shape == (20, 55) and E.shape == (20,) and F.shape == (20, 10, 10)
        assert np.array_equal(L, np.concatenate(rows)[:20])          # file order, no shuffle
        assert acct["n_rows_taken"] == 20 and acct["n_rows_read"] == 24
        assert [s["file"] for s in acct["shards"]] == ["shard_%05d.npz" % i for i in range(3)]
        assert acct["schema"]["seed"] == 42 and acct["schema"]["matmul_precision"] == "highest"
        try:
            KR.load_split_rows(d, 25, "train")
        except SystemExit:
            pass
        else:
            raise AssertionError("a short split must be a hard error, never a silent truncation")


# ---------------------------------------------------------------- v2_beta1_panel
def test_b1_sub_bands_and_band_indices(ctx=None):
    lam = np.array([1.0, 1.02, 1.04, 2.0, 2.01, 9.0])       # 5 % relative-gap blocks
    blocks = B1.sub_bands_local(lam)
    assert [list(b) for b in blocks] == [[0, 1, 2], [3, 4], [5]]
    # the nine-smallest band expands over complete blocks (PHY-06 block rule)
    lam55 = np.concatenate([[0.0], np.linspace(1.0, 1.4, 12), np.linspace(10.0, 20.0, 42)])
    idx = B1.band_indices_local(lam55)
    assert idx.size >= B1.NBAND and np.all(lam55[idx] > B1.TAU_POS)
    blocks, bmn, bmask = B1.blocks_for(lam55)
    assert bmask.sum() == idx.size
    assert sum(len(b) for b in blocks) == int((lam55 > B1.TAU_POS).sum())


def test_b1_arm_quantities_closures(ctx=None):
    rng = np.random.default_rng(5)
    lam = np.concatenate([[1e-18], np.sort(rng.uniform(0.01, 3.0, 54))])
    a2 = rng.uniform(0.0, 1.0, 55)
    dwS = float(a2.sum())                                    # completeness: sum_k a2 = ||dw||_S^2
    bc = B1.blocks_for(lam)
    grid = np.logspace(-10, 1, 111)
    q = B1.arm_quantities(a2, dwS, lam, bc, grid)
    assert q["closure1"] <= B1.CLOSURE_TOL and q["closure2"] <= B1.CLOSURE_TOL
    assert q["p_c7_max_dev"] <= 1e-12
    assert abs(q["f_null"] - a2[0] / dwS) < 1e-15
    lam_pos = lam[lam > B1.TAU_POS]
    assert lam_pos.min() - 1e-12 <= q["lam_eff"] <= lam_pos.max() + 1e-12
    assert abs(q["lam_bar"] - lam_pos.mean()) < 1e-15
    assert q["F"][-1] <= 1.0 + 1e-12 and np.all(np.diff(q["F"]) >= -1e-15)


def test_b1_arm_quantities_single_mode_limit(ctx=None):
    lam = np.concatenate([[0.0], np.linspace(0.5, 5.0, 54)])
    a2 = np.zeros(55)
    a2[20] = 2.0                                             # all residual weight on one mode
    q = B1.arm_quantities(a2, 2.0, lam, B1.blocks_for(lam), np.logspace(-3, 1, 41))
    assert q["f_null"] == 0.0 and abs(q["pos_share"] - 1.0) < 1e-15
    assert abs(q["lam_eff"] - lam[20]) < 1e-15
    assert abs(q["p_soft"] + q["p_bulk"] - 1.0) < 1e-15


def test_b1_row_binding_gate_1ulp_amendment(ctx=None):
    """ID-0 amendment 2026-09-10: PASS iff the 45 off-diagonal triu55 columns are
    bit-identical, every gauge-diagonal difference is <= 2^-23, and the rows line up."""
    rng = np.random.default_rng(6)
    lab = np.asarray(rng.uniform(0.1, 1.0, (32, 55)), np.float32).astype(np.float64)
    r, c = np.triu_indices(10)
    diag_cols = np.flatnonzero(r == c)

    same = B1.row_binding(lab, lab, 32)
    assert same["strict_bit_identical"] and same["pass_"] and same["max_abs_dev"] == 0.0
    assert same["offdiagonal_bit_identical"] and same["diagonal_bit_identical"]
    assert "tolerances.json" in same["note"] and "ID-0 amendment" in same["note"]

    # inside the band: gauge-diagonal only, exactly 1 ULP -> PASS, strict identity recorded False
    inband = lab.copy()
    inband[:, diag_cols] += B1.ULP_F32
    g = B1.row_binding(lab, inband, 32)
    assert g["pass_"] and not g["strict_bit_identical"]
    assert g["offdiagonal_bit_identical"] and not g["diagonal_bit_identical"]
    assert g["columns_differing"] == diag_cols.tolist() == g["triu55_diagonal_columns"]
    assert abs(g["diagonal_max_abs_dev_in_ulp"] - 1.0) < 1e-9
    assert g["criterion"]["diagonal_within_1ulp"] and g["criterion"]["row_count_and_order"]

    # outside the band: 4 ULP on the diagonal -> FAIL
    wide = lab.copy()
    wide[:, diag_cols] += 4.0 * B1.ULP_F32
    w = B1.row_binding(lab, wide, 32)
    assert not w["pass_"] and not w["criterion"]["diagonal_within_1ulp"]

    # any OFF-diagonal difference, however tiny, must FAIL: the amendment covers the gauge only
    off = lab.copy()
    off[:, 1] += B1.ULP_F32 / 8.0
    o = B1.row_binding(lab, off, 32)
    assert not o["pass_"] and not o["criterion"]["offdiagonal_bit_identical"]

    # a genuine row permutation must be O(1), never mistaken for a rounding artefact
    perm = B1.row_binding(lab, lab[::-1], 32)
    assert not perm["pass_"] and not perm["offdiagonal_bit_identical"]
    assert perm["max_abs_dev"] > 1e-3


def test_b1_lam_v2form_criterion(ctx=None):
    """The ratified anchor criterion: abs dev <= 1e-12 AND rel on positive modes <= 1e-4,
    with the null modes excluded from the relative test."""
    assert B1.LAM_ABS_TOL == 1e-12 and B1.LAM_REL_POS_TOL == 1e-4
    assert B1.ULP_F32 == 2.0 ** -23
    lam_reg = np.concatenate([[8.1e-16], np.linspace(0.02, 3.0, 54)])
    lam = lam_reg.copy()
    lam[0] += 2.0e-15                       # a huge RELATIVE move on the null mode
    lam[7] += 1e-14                         # a tiny absolute move on a positive mode
    rel_ = np.abs(lam - lam_reg) / np.maximum(np.abs(lam_reg), 1e-300)
    pos = lam_reg > B1.TAU_POS
    abs_dev = float(np.max(np.abs(lam - lam_reg)))
    rel_pos = float(rel_[pos].max())
    assert rel_.max() > 1.0                                   # verbatim form blows up on the null
    assert abs_dev <= B1.LAM_ABS_TOL and rel_pos <= B1.LAM_REL_POS_TOL   # v2form accepts
    # and it still rejects a real drift on a positive mode
    lam2 = lam_reg.copy()
    lam2[7] += 1e-3
    assert float(np.max(np.abs(lam2 - lam_reg))) > B1.LAM_ABS_TOL


def test_b1_pins_are_the_parent_pins(ctx=None):
    assert B1.BETA == 1.0 and B1.TAU_POS == 1e-12
    assert B1.RIDGE_SWEEP == [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    assert B1.CUTOFF_SWEEP == [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    assert B1.POS_THR_SWEEP == [1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    assert (B1.BOOT_SEED, B1.BOOT_B) == (20260703, 1000)
    assert (B1.CI_LO_IDX, B1.CI_HI_IDX) == (25, 975)
    assert B1.ROT_SEED == 20260724 and (B1.GAP_TOL, B1.NBAND) == (0.05, 9)
    assert B1.N_PANEL_FULL == 4096 and B1.REPRO_TOL == 1e-6
    assert B1.SENS_BINS == [1e-4, 1e-1]
    # the registered anchors are the numbers the parent carries
    assert B1.ANCHORS["lam_min_pos_median"] == 0.024765957404805618
    assert B1.ANCHORS["kappa_MS_median"] == 54.2341915228308
    assert B1.ANCHORS_RKC1["lam_eff"] == 0.39884795734405565


def test_b1_bootstrap_block_is_pinned(ctx=None):
    def block(n):
        rng = np.random.default_rng(B1.BOOT_SEED)
        BI = np.empty((B1.BOOT_B, n), np.int64)
        for b in range(B1.BOOT_B):
            BI[b] = rng.integers(0, n, n)
        return BI
    a, b = block(64), block(64)
    assert np.array_equal(a, b)                              # one shared, reproducible row block
    assert a.min() >= 0 and a.max() < 64


def test_b1_jsonable_drops_nonfinite(ctx=None):
    out = B1._js(dict(a=np.float64(np.nan), b=np.float64(np.inf), c=np.int64(3),
                      d=np.array([1.0, 2.0]), e=np.bool_(True), f=float("nan")))
    assert out == dict(a=None, b=None, c=3, d=[1.0, 2.0], e=True, f=None)
    assert KR._jsonable(dict(x=np.float32(1.5), y=(1, 2), z=float("inf"))) == dict(x=1.5, y=[1, 2], z=None)


def _main():
    ok = fail = 0
    for name in sorted(k for k in globals() if k.startswith("test_")):
        try:
            globals()[name](None)
            ok += 1
            print("PASS %s" % name)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print("FAIL %s: %r" % (name, exc))
            import traceback
            traceback.print_exc()
    print("\n%d passed, %d failed" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(_main())
