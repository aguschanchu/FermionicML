"""ognrepro.v2.registry -- the v2 production checkpoint registry.

Every OGN checkpoint the manuscript uses, re-specified for the v2 configuration
(plan 'production run v2', author decisions 2026-09-09).  Training budgets and
optimizer block are the published ones (campaign/config.py MODELS / _PROD_OPT;
notebook_release/src/scripts/train_lane.py LANES); the v2 delta per lane is
the `cfg` block (V2Config overrides) + the architecture class.

Spec fields
    era, h_type, state_type, num_samples (trainer budget), epochs, arch, res,
    init_seed, batch_size, peak_lr, weight_decay, clip, gen_bs,
    cfg              V2Config overrides (resolved per lane at run time)
    cache_samples    corpus size of the FAMILY cache the lane trains on
                     (twins train 1e4 rows of the 5e6 GS corpus; learning-curve
                     lanes read a prefix of the thermal corpus)
    rows_per_half    learning curve: rows read from each production half
    toggles          ogn ablation toggles (use_scatter/use_reinject/use_orb_emb/readout_bias)
    use_energy_input False for the noE arm (gauge-only projection)
    mlp_width/mlp_blocks  capacity-matched MLP (published 1024 / 8 at res 4)
    tier, surfaces   manuscript relevance (A > B > C) and the surfaces served
    sector           d16 lanes: const_gs | vect_gs (certified caches)
"""
import copy

_OPT = dict(batch_size=256, peak_lr=3e-4, weight_decay=1e-4, clip=1.0)
GEN_BS = {"d20": 4096, "d12": 512, "d16n8": 64}
BETA_THERMAL = 1.0
G_INIT, G_STOP = 0.1, 1.0
D20_OGN_PARAMS = 17_756_929          # res 3, 55 outputs (projection adds none)
D20_MLP_PARAMS = 17_430_599          # DeepResMLP res 4 / mlp_cap 1024x8
D16_PARAMS = {"const_gs": 17_771_458, "vect_gs": 17_772_228}

V2 = dict()                                   # production v2 config = V2Config() defaults
V1 = dict(data_precision="default", aux=None, readout="published", shell="off",
          loss="gram", w_gauge=1.0, loss_precision="default")


def _d20(h_type, state_type, n, ep, arch, seed, cfg=None, tier="A", surfaces=(), **kw):
    s = dict(era="d20", h_type=h_type, state_type=state_type, num_samples=int(n), epochs=int(ep),
             arch=arch, res=3, init_seed=int(seed), gen_bs=GEN_BS["d20"],
             cfg=dict(cfg or {}), cache_samples=int(kw.pop("cache_samples", n)),
             rows_per_half=kw.pop("rows_per_half", None), toggles=dict(kw.pop("toggles", {})),
             use_energy_input=kw.pop("use_energy_input", True), tier=tier,
             surfaces=list(surfaces), **_OPT)
    s.update(kw)
    return s


V2_LANES = {}

# ---- Tier A: thermal flagship (Table I / Fig 6 / SXVII / SVIII / SX / SXI / SXIII / SXX / SXXI)
for _sd in (42, 43, 44):
    V2_LANES["v2std_thermal_random_s%d" % _sd] = _d20(
        "random", "thermal", 5_000_000, 25, "ogn_std", _sd, tier="A",
        surfaces=["TableI", "Fig6", "SXVII", "SVIII", "SX", "SXI", "SXIII", "SXX", "SXXI"])
# ---- Tier A: ground state, metric arm (Fig 4 / SIII / SIV / Fig S3a-b) + diagnostic control + twins
for _sd in (42, 43, 44):
    V2_LANES["v2stdgs_random_s%d" % _sd] = _d20(
        "random", "gs", 5_000_000, 50, "ogn_std", _sd, tier="A",
        surfaces=["Fig4", "SIII", "SIV", "FigS3ab"])
    V2_LANES["v2stdgs_twin_s%d" % _sd] = _d20(
        "random", "gs", 10_000, 1, "ogn_std", _sd, tier="A", cache_samples=5_000_000,
        surfaces=["Fig4", "SIII"], note="near-init twin: 1e4 rows x 1 epoch of the 5e6 GS corpus")
V2_LANES["v2stdgs_gram_s42"] = _d20(
    "random", "gs", 5_000_000, 50, "ogn_std", 42, cfg=dict(loss="gram"), tier="A",
    surfaces=["D-1 gate"], note="T=0 diagnostic control: identical lane with the gram loss")
# ---- Tier A: d16 progression (Figs 2-3 / S1 OGN points) -- certified caches + psi0 sidecar
for _sector, _ht, _ls in (("const_gs", "const", 1), ("vect_gs", "vect", 3)):
    for _kind in ("std", "orig"):
        for _sd in (42, 43, 44):
            V2_LANES["v2d16prog%s_%s_s%d" % (_kind, _sector, _sd)] = dict(
                era="d16n8", h_type=_ht, state_type="gs", num_samples=1_000_000, epochs=10,
                arch=("ogn_std" if _kind == "std" else "ogn"), res=3, init_seed=_sd,
                gen_bs=GEN_BS["d16n8"], cfg=dict(), cache_samples=1_000_000, rows_per_half=None,
                toggles={}, use_energy_input=True, label_size=_ls, sector=_sector, tier="A",
                surfaces=["Fig2", "Fig3", "FigS1"], **_OPT)
# ---- Tier B
V2_LANES["v2gs_const_s42"] = _d20("const", "gs", 2_000_000, 10, "ogn", 42, tier="B",
                                  surfaces=["FigS3c", "SIII-const"])
for _tag, _n in (("2p5e5", 250_000), ("5e5", 500_000), ("1e6", 1_000_000), ("2e6", 2_000_000)):
    V2_LANES["v2lc_thermal_%s_s42" % _tag] = _d20(
        "random", "thermal", _n, 25, "ogn_std", 42, tier="B", cache_samples=5_000_000,
        rows_per_half=_n // 2, surfaces=["LC (new SM figure)"],
        note="learning curve: first N/2 rows of each production half, fixed 25 epochs")
V2_LANES["v2mlp_cap_s42"] = _d20("random", "thermal", 5_000_000, 25, "mlp_std", 42, tier="B",
                                 res=4, mlp_width=1024, mlp_blocks=8, surfaces=["SXV-replacement"],
                                 note="standardized capacity-matched MLP (campaign6 mlp_cap_17M_b8)")
# ---- Tier C (single seed, last)
for _name, _tog in (("noscatter", dict(use_scatter=False)), ("noreinject", dict(use_reinject=False)),
                    ("noembed", dict(use_orb_emb=False)), ("neutralbias", dict(readout_bias=0.0))):
    V2_LANES["v2abl_%s_s42" % _name] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 42,
                                             toggles=_tog, tier="C", surfaces=["ABL-grid", "SXIV-replacement"])
V2_LANES["v2ogn_mse_s42"] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 42, cfg=dict(loss="mse"),
                                 tier="C", surfaces=["ABL-grid"])
V2_LANES["v2ogn_noE_s42"] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 42, use_energy_input=False,
                                 tier="C", surfaces=["ABL-grid", "FigS5-control"],
                                 note="noE arm: use_energy_input=False, gauge-only projection")
V2_LANES["v2abl_mlp_s42"] = _d20("random", "thermal", 5_000_000, 25, "mlp", 42, res=4, tier="C",
                                 surfaces=["ABL-grid"])
V2_LANES["v2mlp_gs_random_s42"] = _d20("random", "gs", 5_000_000, 25, "mlp", 42, res=4, tier="C",
                                       surfaces=["SIII-mlp"], note="MLP on the GS ensemble (25 epochs)")
V2_LANES["v2gram_thermal_random_50_s42"] = _d20("random", "thermal", 5_000_000, 50, "ogn_std", 42,
                                                cfg=dict(loss="gram"), tier="C", surfaces=["SXII"])
V2_LANES["v2abl_rdm_s42"] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 42, cfg=dict(loss="rdm"),
                                 tier="C", surfaces=["ABL-grid", "SXII"],
                                 note="rdm loss = the published engine step (30-62 h); last")
# ---- cross-host twin (XFLEET-01): the s43 flagship retrained on a second host from the identical cache bytes
V2_LANES["v2std_thermal_random_s43_xhost"] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 43, tier="A",
                                                  surfaces=["XFLEET-01"], twin_of="v2std_thermal_random_s43",
                                                  note="cross-host twin: identical spec/seed/cache on another host; ratio <= 1.0248 pass, 1.10 hold")
# ---- pipeline control (never cited): the v2 trainer with every v2 flag off
V2_LANES["v1chk_std_thermal_random_s42"] = _d20("random", "thermal", 5_000_000, 25, "ogn_std", 42,
                                                cfg=dict(V1), tier="ID", surfaces=["ID-7"])

SMOKE_OVERRIDES = dict(num_samples=512, epochs=1, res=1, batch_size=64, gen_bs=64,
                       cache_samples=512, rows_per_half=None, mlp_width=32, mlp_blocks=2)


def spec(lane, smoke=False):
    if lane not in V2_LANES:
        raise KeyError("unknown v2 lane %r" % (lane,))
    s = copy.deepcopy(V2_LANES[lane])
    s["lane"] = lane
    if smoke:
        for k, v in SMOKE_OVERRIDES.items():
            if k in s or k in ("num_samples", "epochs", "res", "batch_size", "gen_bs", "cache_samples"):
                s[k] = v
    return s


def lanes(tier=None):
    return [n for n, s in V2_LANES.items() if tier is None or s["tier"] == tier]


def family_aux(state_type):
    """The sidecar the FAMILY cache carries (every lane of a family shares one
    cache; gram/mse/rdm lanes simply do not load it)."""
    return "psi0" if state_type == "gs" else "M"


def cache_cfg(cfg_resolved, state_type):
    """The V2Config the family cache is keyed by: lane config with the family aux."""
    import dataclasses  # noqa: PLC0415
    return dataclasses.replace(cfg_resolved, aux=family_aux(state_type))


def cache_plan(s, cfg_resolved, m=10, D=252):
    """Cache key + bytes for the disk gate (d20 only; d16 caches are certified inputs)."""
    from .gen import dataset_key, aux_width  # noqa: PLC0415
    ccfg = cache_cfg(cfg_resolved, s["state_type"])
    A = m * m
    aux = ccfg.aux
    side = aux_width(aux, A, D) * (2 if (aux == "M" and ccfg.m_dtype == "float16") else 4)
    n = int(s["cache_samples"])
    n_val = max(int(s["gen_bs"]), int(round(0.05 * n)))
    per_row = 55 * 4 + m * m * 4 + 4 + side       # labels f32 + features f32 + energy + aux (approx)
    return dict(key=dataset_key(s["era"], s["h_type"], s["state_type"], ccfg,
                                tag=("smoke512" if n == 512 else None)),
                aux=aux, n_train=n, n_val=n_val, bytes_per_row=per_row,
                gb_total=(n + n_val) * per_row / 1e9)


def disk_need_gb(s, cfg_resolved, cache_root, m=10, D=252):
    """GB still to write for a lane (caches not yet complete + ~1 GB outputs)."""
    import os, json  # noqa: PLC0415
    cp = cache_plan(s, cfg_resolved, m, D)
    need = 1.0
    for split, n in (("half0", cp["n_train"] // 2), ("half1", cp["n_train"] - cp["n_train"] // 2),
                     ("val", cp["n_val"])):
        p = os.path.join(cache_root, cp["key"], split, "schema.json")
        try:
            done = bool(json.load(open(p)).get("complete"))
        except Exception:  # noqa: BLE001
            done = False
        if not done:
            need += n * cp["bytes_per_row"] / 1e9
    return need


def param_pin(s, m=10):
    if s["era"] == "d16n8":
        return D16_PARAMS[s["sector"]]
    if s["arch"] in ("mlp", "mlp_std"):
        return D20_MLP_PARAMS
    if s["h_type"] == "random" and s["res"] == 3:
        return D20_OGN_PARAMS
    return None
