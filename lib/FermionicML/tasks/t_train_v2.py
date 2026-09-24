"""[V2] Training task driver for the production-run v2 lanes: run(model_name, out_dir).

Thin wrapper: the whole v2 protocol (HIGHEST caches with val split, operator
tables, standardization statistics, V2Config resolution, engine install,
train_model(v2=cfg), config.json / predictions_val.npz / quick_val.json)
lives in notebook_release/src/scripts/train_lane_v2.run_lane_v2 so the
campaign runner and the release trainer can never disagree.  Campaign
conventions kept: caches under C.DATA_DIR (datasets_v2/<key>), checkpoints
under C.CKPT_DIR/<lane>, best-effort GCS push, a <task>.json summary.
Enabled by CAMPAIGN_V2=1 (config.py registers the v2 lanes then).
"""
import json
import os
import sys
import time
import types

_CAMPAIGN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_CAMPAIGN)
_SCRIPTS = os.path.join(_REPO, "notebook_release", "src", "scripts")
_LIB = os.path.join(_REPO, "notebook_release", "src", "lib")
for _p in (_CAMPAIGN, _SCRIPTS, _LIB):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OGNREPRO_ENGINE_DIR", _CAMPAIGN)

import config as C  # noqa: E402


def run(model_name, out_dir):
    import train_lane_v2 as TLV2  # noqa: PLC0415
    lane = model_name[len("v2::"):] if model_name.startswith("v2::") else model_name
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    gcs = None if C.SMOKE else "%s/results_v2/checkpoints/%s" % (C.GCS_BUCKET, C.VM_NAME)
    args = types.SimpleNamespace(smoke=bool(C.SMOKE), cache_dir=C.DATA_DIR, out=C.CKPT_DIR,
                                 resume=True, gcs=gcs, force=False)
    rc = TLV2.run_lane_v2(lane, args)
    ckpt_dir = os.path.join(C.CKPT_DIR, lane)
    with open(os.path.join(ckpt_dir, "config.json")) as fh:
        cfg = json.load(fh)
    payload = dict(lane=lane, n_params=cfg.get("n_params"), final_loss=cfg.get("final_loss"),
                   epochs_run=cfg.get("epochs_run"), quick_val=cfg.get("quick_val"),
                   final_state_sha256=cfg.get("final_state_sha256"), rc=rc,
                   ckpt_dir=ckpt_dir, wall_s=time.time() - t0)
    task_name = os.path.basename(os.path.normpath(out_dir))
    with open(os.path.join(out_dir, "%s.json" % task_name), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return payload
