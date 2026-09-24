"""production_v2.stages.v2_d16prog_figaux -- structural tests (no engine, no
checkpoints, no jax op): the lane -> published name map, the OGN key
grammar and census mapping against the SEALED D16PROG-FIGAUX-M1/M2 key
lists, and the timestamp-free npz writer (byte-reproducible, np.load
compatible).

Standalone (not registered in run_tests.MODULES, which binds the engine):

    JAX_PLATFORMS=cpu <pinned python> -m ognrepro.v2.tests.test_figaux
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.abspath(__file__)
_REPO = _HERE
for _ in range(7):
    _REPO = os.path.dirname(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from production_v2.stages import v2_d16prog_figaux as FG  # noqa: E402

_SEALED = os.path.join(_REPO, "experiments", "experiment-campaign2-r2", "results",
                       "D16PROG-FIGAUX-%s", "expc2r2_d16prog_figaux_percell.npz")


def test_lane_name_map(ctx=None):
    assert len(FG.V2_LANES) == 12 and len(set(FG.V2_LANES)) == 12
    assert FG.published_name("v2d16progstd_const_gs_s42") == "d16progstd_const_gs"
    assert FG.published_name("v2d16progorig_vect_gs_s43") == "d16progorig_vect_gs_s43"
    assert FG.published_name("v2d16progstd_vect_gs_s44") == "d16progstd_vect_gs_s44"
    assert FG.lane_sector("v2d16progstd_const_gs_s42") == "const_gs"
    assert FG.lane_sector("d16progorig_vect_gs") == "vect_gs"
    pubs = sorted(FG.published_name(ln) for ln in FG.V2_LANES)
    assert len(set(pubs)) == 12
    # the parent M1 roster order: const lanes first, then vect
    assert [FG.lane_sector(ln) for ln in FG.V2_LANES] == ["const_gs"] * 6 + ["vect_gs"] * 6


def test_ogn_key_grammar(ctx=None):
    assert FG.parse_ogn_key("M2", "figaux96_OGN_d16progstd_const_gs_rho2_pred") == \
        ("figaux96_OGN_", "d16progstd_const_gs", "_rho2_pred")
    assert FG.parse_ogn_key("M2", "figaux96_OGN_d16progstd_const_gs_s43_pred") == \
        ("figaux96_OGN_", "d16progstd_const_gs_s43", "_pred")
    assert FG.parse_ogn_key("M1", "figaux_Vref_OGN_d16progorig_vect_gs_s44_lambda") == \
        ("figaux_Vref_OGN_", "d16progorig_vect_gs_s44", "_lambda")
    assert FG.parse_ogn_key("M1", "figaux_U_RG-CURVE_ghat") is None
    assert FG.parse_ogn_key("M2", "figaux96_REF-ED_psi") is None


def test_census_against_sealed_percells(ctx=None):
    """The mapped full census has exactly the sealed key count per leg, the
    classical/OGN split is 119/60 (M1) and 57/36 (M2), and a smoke roster
    yields classical + 5 (resp. 6) keys per lane."""
    for leg, n_all, n_cl, per_lane, n_lanes in (("M1", 179, 119, 5, 12), ("M2", 93, 57, 6, 6)):
        p = _SEALED % leg
        if not os.path.isfile(p):
            print("  (sealed %s percell not present; census test skipped)" % leg)
            continue
        with np.load(p) as z:
            keys = list(z.files)
        assert len(keys) == n_all
        run, full, classical, templates = FG.expected_census(leg, keys, FG.V2_LANES, [])
        assert len(classical) == n_cl and len(templates) == n_lanes
        assert len(full) == n_all and run == full
        assert all(("_OGN_v2d16prog" in k) or (k in classical) for k in full)
        smoke_lanes = ["v2d16progstd_const_gs_s42", "v2d16progorig_vect_gs_s42"]
        run_s, _f, _c, _t = FG.expected_census(leg, keys, smoke_lanes, [])
        n_ogn_lanes = 2 if leg == "M1" else 1
        assert len(run_s) == n_cl + per_lane * n_ogn_lanes
        run_p, _f, _c, _t = FG.expected_census(leg, keys, smoke_lanes, ["d16progstd_const_gs"])
        assert len(run_p) == len(run_s) + per_lane


def test_savez_deterministic_roundtrip(ctx=None):
    store = {"a_f64": np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0,
             "b_bool": np.array([True, False, True]),
             "c_i64": np.arange(5, dtype=np.int64),
             "d_nan": np.array([1.0, np.nan, 3.0])}
    d = tempfile.mkdtemp(prefix="v2figaux_")
    p1, p2 = os.path.join(d, "one.npz"), os.path.join(d, "two.npz")
    FG.savez_deterministic(p1, store)
    FG.savez_deterministic(p2, store)
    assert open(p1, "rb").read() == open(p2, "rb").read()
    with np.load(p1) as z:
        assert list(z.files) == list(store)
        for k in store:
            assert FG.bytes_equal(z[k], store[k])
            assert FG._np_equal(z[k], store[k])
    assert not FG.bytes_equal(np.zeros(3), np.zeros(3, np.float32))


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(None)
                print("PASS %s" % name)
            except Exception:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                fails += 1
                print("FAIL %s" % name)
                traceback.print_exc()
    print("%d failed" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
