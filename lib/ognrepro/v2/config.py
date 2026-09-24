"""ognrepro.v2.config -- the versioned v2 configuration (one frozen object).

`V2Config` is threaded through generation (data precision, sidecar kind and
dtype), model construction (readout, gauge value, shell row, standardization
convention), the loss (metric / gram / mse, lambda schedule, penalty weights,
quadratic-form precision) and the loader (aux packing).  `resolve()` fills in
the family-dependent fields from (state_type, label_size, m, arch,
use_energy_input) with pure rules and marks the object resolved; the
resolved object is what every config.json / schema.json / record stores.

Rules encoded here (plan "Production run v2", author decisions 2026-09-09):
  * aux:      'psi0' for ground states, 'M' (packed 100x100 covariance) for
              thermal states, only when loss == 'metric'; None otherwise.
  * readout:  'affine' only for the dense-random family (label_size ==
              m(m+1)/2) on the OGN classes; const/vect heads and the MLP
              arms keep the published readout.
  * shell:    'off' when the energy input branch is disabled (the noE arm
              stays energy-free: gauge row only).
  * w_gauge:  0 whenever the affine readout is on (the gauge row makes the
              trace penalty redundant -- asserted), 1 (engine default) else.
  * std_convention: 'std1_entry' for arch 'ogn_std' (the manuscript's
              flagship STD-1 per-entry standardization, t_std2.py),
              'triu55' for 'mlp_std' (campaign10 55-feature transform,
              MLP capacity arm only), None for unstandardized arms.
"""
import dataclasses
import json
from typing import Optional

CFG_VERSION = 2

AUX_KINDS = ("M", "psi0", None)
READOUTS = ("affine", "published")
SHELLS = ("euclid", "off")
LOSSES = ("metric", "gram", "mse", "rdm")
PRECISIONS = ("highest", "default")
STD_CONVENTIONS = ("std1_entry", "triu55", None)
M_DTYPES = ("float16", "float32")


@dataclasses.dataclass(frozen=True)
class V2Config:
    cfg_version: int = CFG_VERSION
    # --- data generation
    data_precision: str = "highest"      # matmul precision of BOTH the H build and the feature contraction
    aux: Optional[str] = None            # 'M' | 'psi0' | None (resolved from state_type when loss == 'metric')
    m_dtype: str = "float16"             # on-disk dtype of the packed covariance sidecar
    m_chunk: int = 32                    # device chunk of the covariance kernel
    # --- readout
    readout: str = "affine"              # 'affine' (gauge + shell projection) | 'published'
    gauge_value: float = 0.55            # the data generator's diagonal-mean convention
    shell: str = "euclid"                # 'euclid' | 'off' (gauge row only)
    # --- loss
    loss: str = "metric"                 # 'metric' | 'gram' | 'mse'
    lam0: float = 10.0                   # lambda anneal start (S_inf anchor weight)
    lam1: float = 1e-2                   # lambda anneal end
    anneal_frac: float = 0.8             # fraction of total steps over which lambda anneals (log-linear)
    w_gauge: float = 0.0                 # trace-penalty weight (0 iff the affine readout is on)
    w_ridge: float = 1e-2                # engine ridge weight (gram_w_ridge)
    loss_precision: str = "highest"      # precision of the loss quadratic forms
    net_precision: Optional[str] = None  # network matmul precision (None = published default)
    # --- standardization
    std_convention: Optional[str] = None # 'std1_entry' | 'triu55' | None
    std_n_max: int = 300_000             # rows used for the statistics (first rows, file order)
    sigma_floor: float = 1e-8            # STD-1 sigma floor (t_std2 SIGMA_FLOOR)
    # --- bookkeeping
    resolved: bool = False

    def __post_init__(self):
        if self.data_precision not in PRECISIONS:
            raise ValueError("data_precision %r" % (self.data_precision,))
        if self.aux not in AUX_KINDS:
            raise ValueError("aux %r" % (self.aux,))
        if self.m_dtype not in M_DTYPES:
            raise ValueError("m_dtype %r" % (self.m_dtype,))
        if self.readout not in READOUTS:
            raise ValueError("readout %r" % (self.readout,))
        if self.shell not in SHELLS:
            raise ValueError("shell %r" % (self.shell,))
        if self.loss not in LOSSES:
            raise ValueError("loss %r" % (self.loss,))
        if self.loss_precision not in PRECISIONS:
            raise ValueError("loss_precision %r" % (self.loss_precision,))
        if self.net_precision not in (None, "highest", "default"):
            raise ValueError("net_precision %r" % (self.net_precision,))
        if self.std_convention not in STD_CONVENTIONS:
            raise ValueError("std_convention %r" % (self.std_convention,))
        if self.readout == "affine" and self.w_gauge != 0.0:
            raise ValueError("the affine readout makes the trace penalty redundant: w_gauge must be 0")
        if not (0.0 <= self.anneal_frac <= 1.0):
            raise ValueError("anneal_frac must be in [0, 1]")

    # ------------------------------------------------------------ helpers
    def published(self):
        """True iff every v2 change is off (the published numerics)."""
        return (self.data_precision == "default" and self.aux is None
                and self.readout == "published" and self.loss == "gram"
                and self.loss_precision == "default")

    def projection_on(self):
        return self.readout == "affine"

    def cache_tag(self):
        """Tag that makes precision + sidecar part of the cache key."""
        p = "phi" if self.data_precision == "highest" else "pdef"
        if self.aux == "M":
            a = "aM_f16" if self.m_dtype == "float16" else "aM_f32"
        elif self.aux == "psi0":
            a = "apsi0"
        else:
            a = "anone"
        return "%s_%s" % (p, a)

    def to_json(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("cfg_version", None)
        return cls(cfg_version=CFG_VERSION, **d)

    def dumps(self):
        return json.dumps(self.to_json(), sort_keys=True)


V2_PRODUCTION = V2Config()
V1_PUBLISHED = V2Config(data_precision="default", aux=None, readout="published",
                        shell="off", loss="gram", w_gauge=1.0,
                        loss_precision="default")


def resolve(cfg, *, state_type, label_size, m, arch="ogn_std",
            use_energy_input=True):
    """Fill the family-dependent fields with the plan's rules; pure."""
    n_triu = (int(m) * (int(m) + 1)) // 2
    dense = int(label_size) == n_triu
    if cfg.loss == "metric":
        aux = "psi0" if state_type == "gs" else "M"
    else:
        aux = None
    if cfg.readout == "affine" and dense and arch in ("ogn", "ogn_std"):
        readout = "affine"
    else:
        readout = "published"
    shell = cfg.shell
    if not use_energy_input or readout != "affine":
        shell = "off"                      # the shell row exists only inside the affine readout
    if readout == "affine":
        w_gauge = 0.0
    else:
        w_gauge = 1.0 if cfg.loss not in ("mse", "rdm") else 0.0
    if arch == "ogn_std":
        std = "std1_entry"
    elif arch == "mlp_std":
        std = "triu55"
    else:
        std = None
    return dataclasses.replace(cfg, aux=aux, readout=readout, shell=shell,
                               w_gauge=w_gauge, std_convention=std,
                               resolved=True)
