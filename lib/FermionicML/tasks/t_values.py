"""values.json assembler + LaTeX snippet renderer: run(out_dir).

Idempotent, rerunnable. Splices every completed task's <task>.json under the
matching section of values.json (schema modeled on
campaign/reference/values_prev.json), marks missing sections "PENDING", and
renders two LaTeX snippets:
  * tab_gevp_diag.tex     -- referee's GEVP diagnostics table
                             (panels: random-untrained / random-trained /
                              const-trained)
  * tab_model_configs.tex -- App-B per-model configuration table from
                             C.MODELS + the checkpoints' config.json

Never fails on missing pieces: renders whatever exists. Does NOT import the
heavy engine.
"""
import importlib.metadata
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C

PROVENANCE = "referee-response campaign 2026-06; satelite2/campaign"


# ----------------------------------------------------------------- IO utils
def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _is_done(task):
    return os.path.exists(os.path.join(C.RESULTS_DIR, task, "DONE.json"))


def _task_json(task):
    """The completed task's main results (<task>.json, else the largest other
    .json in the task dir — tasks name their main output differently, e.g.
    shots_thermal/shots.json, d12_figures/rg_overlay.json — else DONE payload)."""
    if not _is_done(task):
        return None
    data = _read_json(os.path.join(C.RESULTS_DIR, task, f"{task}.json"))
    if data is not None:
        return data
    import glob as _glob
    cands = [p for p in _glob.glob(os.path.join(C.RESULTS_DIR, task, "*.json"))
             if os.path.basename(p) not in ("DONE.json",)
             and not os.path.basename(p).startswith("FAIL")]
    for p in sorted(cands, key=os.path.getsize, reverse=True):
        data = _read_json(p)
        if isinstance(data, dict) and data:
            return data
    done = _read_json(os.path.join(C.RESULTS_DIR, task, "DONE.json"))
    if isinstance(done, dict):
        return {k: v for k, v in done.items()
                if k not in ("task", "vm", "worker", "finished_utc")} or done
    return done


def _completed_tasks():
    try:
        return sorted(t for t in os.listdir(C.RESULTS_DIR)
                      if os.path.isdir(os.path.join(C.RESULTS_DIR, t))
                      and _is_done(t))
    except Exception:
        return []


# ----------------------------------------------------------------- meta
def _meta():
    versions = {}
    for pkg in ("jax", "jaxlib", "flax", "optax", "numpy", "scipy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
    devices = None
    try:
        import jax
        devices = [str(d) for d in jax.devices()]
    except Exception:
        pass
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vm": C.VM_NAME,
        "worker": C.WORKER_ID,
        "smoke": C.SMOKE,
        "driver": "campaign tasks/t_values.py",
        "provenance": PROVENANCE,
        "init_seed": C.INIT_SEED,
        "versions": versions,
        "devices": devices,
    }


# ----------------------------------------------------------------- LaTeX fmt
def _tex_escape(s):
    return str(s).replace("_", r"\_")


def _fmt_num(v):
    """LaTeX-friendly number; '--' for missing."""
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return _tex_escape(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(f):
        return "--"
    if f == 0:
        return "$0$"
    if float(f).is_integer() and abs(f) < 1e6:
        return f"${int(f)}$"
    a = abs(f)
    if 1e-2 <= a < 1e4:
        return f"${f:.4g}$"
    mant, expo = f"{f:.2e}".split("e")
    return f"${mant}\\times10^{{{int(expo)}}}$"


# ----------------------------------------------------------------- GEVP table
_METRIC_KEYS = {"f_err", "lam_min_pos", "kappa", "eps_I", "f_par"}


def _find_panels(obj, path=""):
    """Yield (path, dict) for nested dicts carrying GEVP diagnostic keys."""
    if not isinstance(obj, dict):
        return
    if _METRIC_KEYS & set(obj.keys()):
        yield (path, obj)
    for k, v in obj.items():
        if isinstance(v, dict):
            sub = f"{path}.{k}" if path else str(k)
            yield from _find_panels(v, sub)


def _pick_panel(data, include, exclude=()):
    if not isinstance(data, dict):
        return None
    cands = list(_find_panels(data))
    for inc in include:
        for path, d in cands:
            pl = path.lower()
            if inc in pl and not any(e in pl for e in exclude):
                return d
    for path, d in cands:  # fallback: first non-excluded metric dict
        if not any(e in path.lower() for e in exclude):
            return d
    return None


def _get(panel, keys):
    if not isinstance(panel, dict):
        return None
    for k in keys:
        if k in panel and not isinstance(panel[k], (dict, list)):
            return panel[k]
    return None


def _stab_delta(panel, which):
    """Spread of f_err over the ridge/cutoff stability sweep, if recorded."""
    if not isinstance(panel, dict):
        return None
    direct = _get(panel, (f"{which}_stability_delta", f"stability_delta_{which}",
                          f"delta_{which}"))
    if direct is not None:
        return direct
    for key in (f"stability_{which}", f"{which}_stability", f"{which}_sweep",
                "stability_corrected", "stability"):
        s = panel.get(key)
        if isinstance(s, dict):
            fe = s.get("f_err")
            if (isinstance(fe, (list, tuple)) and len(fe) == 2
                    and all(isinstance(x, (int, float)) for x in fe)):
                return abs(fe[1] - fe[0])
    return None


_GEVP_ROWS = [
    (r"$f_{\rm err}$", lambda p: _get(p, ("f_err",))),
    (r"$\epsilon_I$", lambda p: _get(p, ("eps_I",))),
    (r"$f_{\rm par}$", lambda p: _get(p, ("f_par",))),
    (r"$R$", lambda p: _get(p, ("R",))),
    (r"$\|\Delta w\|_S^2$", lambda p: _get(p, ("dw_S2", "dw_norm_S2",
                                               "norm_dw_S2", "delta_w_S2",
                                               "dwS2", "dw_S_sq"))),
    (r"ratio$_S$ (median)", lambda p: _get(p, ("ratio_S_median", "ratio_S",
                                               "ratio_S_mean"))),
    (r"$\lambda_{\min}^{+}$", lambda p: _get(p, ("lam_min_pos",))),
    (r"$\lambda_{\max}$", lambda p: _get(p, ("lam_max",))),
    (r"$\kappa$", lambda p: _get(p, ("kappa",))),
    (r"$r_{\max}$", lambda p: _get(p, ("r_max",))),
    (r"$r_{\rm med}$", lambda p: _get(p, ("r_med",))),
    (r"$n_{\rm discarded}$", lambda p: _get(p, ("n_discarded",))),
    (r"null modes", lambda p: _get(p, ("null_mode_count", "null_count"))),
    (r"cond$(S)$", lambda p: _get(p, ("cond_S",))),
    (r"ridge stab.\ $\Delta f_{\rm err}$", lambda p: _stab_delta(p, "ridge")),
    (r"cutoff stab.\ $\Delta f_{\rm err}$", lambda p: _stab_delta(p, "cutoff")),
    (r"disc.\ err.\ weight", lambda p: _get(p, ("disc_err_weight",
                                                "discarded_err_weight",
                                                "disc_weight",
                                                "discarded_error_weight"))),
]


def _render_gevp_table(gevp_random, gevp_const):
    panels = [
        ("random--untrained",
         _pick_panel(gevp_random, ("untrained", "nearinit", "near_init"))),
        ("random--trained",
         _pick_panel(gevp_random, ("trained",), exclude=("untrained", "nearinit"))),
        ("const--trained",
         _pick_panel(gevp_const, ("trained", "const"), exclude=("untrained",))),
    ]
    lines = [
        "% Auto-generated by tasks/t_values.py -- " + PROVENANCE,
        "% GEVP diagnostics; missing entries rendered as --, pending panels left blank.",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & " + " & ".join(name for name, _ in panels) + r" \\",
        r"\midrule",
    ]
    for label, getter in _GEVP_ROWS:
        cells = []
        for name, panel in panels:
            if label == r"$\epsilon_I$" and name.startswith("const"):
                cells.append("N/A")
                continue
            try:
                cells.append(_fmt_num(getter(panel)))
            except Exception:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------- model table
_ROLES = {
    "prod_thermal_random": "Production thermal (Secs.~V--VI; finite-shot protocol)",
    "prod_gs_random": "GEVP random panel (Sec.~IV)",
    "prod_gs_const": "GEVP const panel (Sec.~IV)",
    "nearinit_random": "Untrained baseline for GEVP diagnostics",
    "d12_const": "BCS uniform panels ($d=12$)",
    "d12_vect": "BCS vectorial panels ($d=12$)",
    "abl_gram_s43": "Ablation: gram loss, seed 43",
    "abl_gram_s44": "Ablation: gram loss, seed 44",
    "abl_rdm_s42": "Ablation: RDM loss, seed 42",
    "abl_rdm_s43": "Ablation: RDM loss, seed 43",
    "abl_rdm_s44": "Ablation: RDM loss, seed 44",
    "abl_mlp": "Ablation: DeepResMLP baseline (param-matched)",
    "abl_noscatter": "Ablation: no scatter contraction",
    "abl_noreinject": "Ablation: no RDM reinjection",
    "abl_noembed": "Ablation: no orbital embedding",
    "abl_neutralbias": "Ablation: neutral readout bias",
}
_FAMILY = {"ogn": "OGN", "mlp": "DeepResMLP", "cnn": "CoordResMLP"}


def _fmt_samples(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "--"
    if n >= 1_000_000 and n % 100_000 == 0:
        return f"${n / 1e6:g}\\times10^{{6}}$"
    if n >= 1_000 and n % 1_000 == 0:
        return f"${n // 1000}$k"
    return str(n)


def _fmt_params(n):
    if n is None:
        return "--"
    try:
        return f"{int(n) / 1e6:.2f}M"
    except (TypeError, ValueError):
        return "--"


def _render_model_table():
    split = r"$2\times50\%$ (s42/s43); val $5\%$ (s1007)"
    lines = [
        "% Auto-generated by tasks/t_values.py -- " + PROVENANCE,
        "% App-B model configuration table (one row per registered model).",
        r"\begin{tabular}{lllllllllllll}",
        r"\toprule",
        ("Model & Ensemble & Family & $(d,N)$ & Samples & Split & Epochs & "
         r"Optimizer & Params & Bias & Seed & Hardware & Role \\"),
        r"\midrule",
    ]
    for name, spec in C.MODELS.items():
        cfg = _read_json(os.path.join(C.CKPT_DIR, name, "config.json")) or {}
        era = spec.get("era", "?")
        dn = "$(20,10)$" if era == "d20" else "$(12,6)$" if era == "d12" else "--"
        if spec.get("state_type") == "thermal":
            ens = f"{spec.get('h_type', '?')}/thermal, $\\beta=1$"
        else:
            ens = f"{spec.get('h_type', '?')}/GS"
        family = _FAMILY.get(spec.get("arch"), str(spec.get("arch")))
        family += f" (res={spec.get('res', '?')})"
        opt = (f"AdamW ({_fmt_num(spec.get('peak_lr'))}, "
               f"wd {_fmt_num(spec.get('weight_decay'))}, "
               f"clip {_fmt_num(spec.get('clip'))})")
        params = _fmt_params(cfg.get("n_params"))
        hardware = cfg.get("hardware") or "--"
        row = " & ".join([
            _tex_escape(name), ens, family, dn,
            _fmt_samples(spec.get("num_samples")), split,
            str(spec.get("epochs", "--")), opt, params,
            _fmt_num(spec.get("readout_bias", 0.55)),
            str(spec.get("init_seed", "--")),
            _tex_escape(hardware),
            _ROLES.get(name, "--"),
        ])
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------- assembly
def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    values = {"meta": _meta()}
    done, pending, consumed = [], [], set()

    def section(path, task, transform=None):
        """Splice the completed task's JSON at `path`; PENDING otherwise."""
        data = _task_json(task)
        target = values
        for k in path[:-1]:
            target = target.setdefault(k, {})
        dotted = ".".join(path)
        if data is None:
            target[path[-1]] = "PENDING"
            pending.append(dotted)
        else:
            if transform is not None:
                try:
                    data = transform(data)
                except Exception:
                    pass
            target[path[-1]] = data
            done.append(dotted)
            consumed.add(task)

    # covariance.*: merge each GEVP task's panel dicts under covariance,
    # keeping whatever panel names the diagnostics part emitted.
    cov = {}
    for task, fallback_key in (("gevp_random", "random"), ("gevp_const", "const")):
        data = _task_json(task)
        if data is None:
            cov[fallback_key] = "PENDING"
            pending.append(f"covariance.{fallback_key}")
            continue
        if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
            for k, v in data.items():
                cov[k if k not in cov else f"{task}.{k}"] = v
        else:
            cov[fallback_key] = data
        done.append(f"covariance.{fallback_key}")
        consumed.add(task)
    values["covariance"] = cov
    section(("covariance", "null_audit_const"), "nullmode_const")

    section(("shots",), "shots_thermal")
    section(("rg",), "d12_figures",
            transform=lambda d: d.get("rg", d) if isinstance(d, dict) else d)
    section(("beta100", "random"), "beta100_random")
    section(("beta100", "const"), "beta100_const")
    section(("ablation", "eval"), "ablation_eval")
    section(("ablation", "kernel_ridge"), "kernel_ridge")
    section(("ablation", "gbase_probe"), "gbase_probe")

    # training summaries (DONE payload + checkpoint config.json)
    training = {}
    for m in C.MODELS:
        task = f"train_{m}"
        data = _task_json(task)
        cfg = _read_json(os.path.join(C.CKPT_DIR, m, "config.json")) or {}
        if data is None and not cfg:
            training[m] = "PENDING"
            pending.append(f"training.{m}")
            continue
        entry = {}
        if isinstance(data, dict):
            entry.update({k: data[k] for k in
                          ("n_params", "final_loss", "epochs_run", "val_metrics")
                          if k in data})
        for k in ("n_params", "epochs_run", "hardware"):
            entry.setdefault(k, cfg.get(k))
        training[m] = entry
        done.append(f"training.{m}")
        consumed.add(task)
    values["training"] = training

    # anything else completed and unmapped -> "other"
    leftovers = [t for t in _completed_tasks()
                 if t not in consumed and t != "values_assemble"]
    if leftovers:
        other = {}
        for t in leftovers:
            data = _task_json(t)
            if data is not None:
                other[t] = data
                done.append(f"other.{t}")
        if other:
            values["other"] = other

    with open(os.path.join(out_dir, "values.json"), "w") as f:
        json.dump(values, f, indent=1, default=str)

    # ---- LaTeX snippets (best effort; never fail the task)
    for fname, render in (
            ("tab_gevp_diag.tex",
             lambda: _render_gevp_table(_task_json("gevp_random"),
                                        _task_json("gevp_const"))),
            ("tab_model_configs.tex", _render_model_table)):
        try:
            text = render()
        except Exception as exc:
            text = f"% rendering failed: {exc}\n"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(text)

    payload = {"sections_done": sorted(set(done)),
               "sections_pending": sorted(set(pending))}

    task_name = os.path.basename(os.path.normpath(out_dir))
    with open(os.path.join(out_dir, f"{task_name}.json"), "w") as f:
        json.dump(payload, f, indent=1)
    return payload
