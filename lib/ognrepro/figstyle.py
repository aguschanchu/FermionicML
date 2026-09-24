"""ognrepro.figstyle -- shared matplotlib rc for the release notebooks.

PROVENANCE
    Source: NEW FILE (no extracted body).  Extraction date: 2026-08-23.
    Tag: ADAPTED (bundle-original).  Headless-safe (Agg), NO LaTeX
    dependency (usetex stays False), mathtext only.
"""
import os


def apply(backend_agg=None):
    """Apply the bundle rc.  backend_agg=None auto-selects Agg when no
    display is available; True/False forces.  Returns matplotlib."""
    import matplotlib
    if backend_agg is None:
        backend_agg = not os.environ.get("DISPLAY")
    if backend_agg:
        try:
            matplotlib.use("Agg", force=False)
        except Exception:                                    # noqa: BLE001
            pass
    matplotlib.rcParams.update({
        "text.usetex": False,              # no LaTeX toolchain required
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "errorbar.capsize": 2.0,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (3.4, 2.55),     # single-column PRA-ish
        "axes.prop_cycle": matplotlib.cycler(color=[
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]),
    })
    return matplotlib


FIGSIZE_1COL = (3.4, 2.55)
FIGSIZE_2COL = (7.0, 2.8)
