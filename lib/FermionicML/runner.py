"""Unattended campaign runner: one process per TPU-VM worker.

Executes the worker's queue from config.QUEUES in order, with done-markers,
dependency checks, heartbeats, and best-effort GCS sync. Designed for spot
VMs: every step is idempotent and resumable; on relaunch, completed tasks are
skipped (local DONE marker or marker previously synced to GCS by this VM).

Run via run_campaign.sh (sets TPU single-host env vars, venv, nohup).
"""
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [w{C.WORKER_ID}] {msg}", flush=True)


def _gcs(args, timeout=600):
    """Best-effort gcloud storage call; never raises. Disabled in smoke mode."""
    if C.SMOKE:
        return False
    try:
        r = subprocess.run(["gcloud", "storage"] + args, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def gcs_prefix():
    return f"{C.GCS_BUCKET}/results/{C.VM_NAME}"


def done_path(task):
    return os.path.join(C.RESULTS_DIR, task, "DONE.json")


def is_done(task):
    if os.path.exists(done_path(task)):
        return True
    # after a preemption-relaunch the local disk is fresh: consult the bucket
    if _gcs(["ls", f"{gcs_prefix()}/{task}/DONE.json"], timeout=60):
        _gcs(["cp", "-r", f"{gcs_prefix()}/{task}", C.RESULTS_DIR + "/"], timeout=1800)
        return os.path.exists(done_path(task))
    return False


def mark_done(task, payload=None):
    os.makedirs(os.path.dirname(done_path(task)), exist_ok=True)
    with open(done_path(task), "w") as f:
        json.dump({"task": task, "vm": C.VM_NAME, "worker": C.WORKER_ID,
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   **(payload or {})}, f, indent=1)
    sync_task(task)


def sync_task(task):
    src = os.path.join(C.RESULTS_DIR, task)
    if os.path.isdir(src):
        C.gcs_sync_dir(src, f"{gcs_prefix()}/{task}")


def deps_met(task):
    return all(is_done(d) for d in C.TASKS[task].get("requires", []))


# ------------------------------------------------------------------ heartbeat
_current = {"task": None, "since": None}


def _heartbeat_loop():
    hb_local = os.path.join(C.STATUS_DIR, f"worker{C.WORKER_ID}.json")
    os.makedirs(C.STATUS_DIR, exist_ok=True)
    while True:
        try:
            with open(hb_local, "w") as f:
                json.dump({"vm": C.VM_NAME, "worker": C.WORKER_ID,
                           "task": _current["task"], "since": _current["since"],
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
            _gcs(["cp", hb_local,
                  f"{C.GCS_BUCKET}/status/{C.VM_NAME}/worker{C.WORKER_ID}.json"], timeout=60)
        except Exception:
            pass
        time.sleep(C.HEARTBEAT_S)


def owner_recently_alive(task):
    """Adoption guard: is the task's static owner heartbeating on this VM?"""
    owner = next((w for w, q in C.QUEUES.items() if task in q), None)
    if owner is None:
        return False
    hb = os.path.join(C.STATUS_DIR, f"worker{owner}.json")
    try:
        # workers have separate disks; consult the bucket copy
        tmp = "/tmp/hb_probe.json"
        ok = _gcs(["cp", f"{C.GCS_BUCKET}/status/{C.VM_NAME}/worker{owner}.json", tmp],
                  timeout=60)
        src = tmp if ok else hb
        with open(src) as f:
            d = json.load(f)
        ts = time.mktime(time.strptime(d["utc"], "%Y-%m-%dT%H:%M:%SZ"))
        return (time.time() - ts) < C.ADOPT_AFTER_S
    except Exception:
        return False


# ------------------------------------------------------------------ execution
def run_task(task):
    spec = C.TASKS[task]
    _current.update(task=task, since=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    t0 = time.time()
    log(f"START {task}")
    out_dir = os.path.join(C.RESULTS_DIR, task)
    os.makedirs(out_dir, exist_ok=True)
    try:
        if spec["kind"] == "train":
            mod = importlib.import_module("tasks.t_train")
            payload = mod.run(spec["model"], out_dir)
        else:
            mod_name, fn_name = spec["entry"].split(":")
            mod = importlib.import_module(mod_name)
            payload = getattr(mod, fn_name)(out_dir)
        mark_done(task, {"wall_s": time.time() - t0, **(payload or {})})
        log(f"DONE  {task} ({time.time() - t0:.0f}s)")
        return True
    except Exception:
        err = traceback.format_exc()
        log(f"FAIL  {task}\n{err}")
        with open(os.path.join(out_dir, f"FAIL_{int(time.time())}.txt"), "w") as f:
            f.write(err)
        sync_task(task)
        return False
    finally:
        _current.update(task=None, since=None)


def main():
    for d in (C.WORKDIR, C.DATA_DIR, C.CKPT_DIR, C.RESULTS_DIR, C.STATUS_DIR):
        os.makedirs(d, exist_ok=True)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    log(f"runner up on {C.VM_NAME} worker {C.WORKER_ID}; queue={C.QUEUES[C.WORKER_ID]}")

    # backfill: re-sync any locally-completed task whose upload may have failed
    # (e.g. the old-gcloud rsync gap) — cheap and idempotent
    for t in {t for q in C.QUEUES.values() for t in q}:
        if os.path.exists(done_path(t)):
            sync_task(t)

    my_queue = list(C.QUEUES[C.WORKER_ID])
    fail_counts = {}
    while True:
        progressed = False
        pending = [t for t in my_queue if not is_done(t)]

        # adoption: anything left anywhere once my own queue is drained
        if not pending:
            all_tasks = [t for q in C.QUEUES.values() for t in q]
            pending = [t for t in all_tasks if not is_done(t)
                       and not owner_recently_alive(t)]
            if pending:
                log(f"adopting orphaned tasks: {pending[:3]}...")

        for task in pending:
            if not deps_met(task):
                continue
            if fail_counts.get(task, 0) >= 3:
                continue  # give up after 3 attempts; humans read FAIL_*.txt
            ok = run_task(task)
            progressed = progressed or ok
            if not ok:
                fail_counts[task] = fail_counts.get(task, 0) + 1

        # idempotent re-assembly of values.json picks up new results each pass
        if progressed and is_done("values_assemble"):
            try:
                run_task("values_assemble")
            except Exception:
                pass

        remaining = [t for q in C.QUEUES.values() for t in q if not is_done(t)]
        if not remaining:
            log("ALL TASKS COMPLETE")
            _gcs(["rsync", "-r", C.RESULTS_DIR, f"{gcs_prefix()}"], timeout=7200)
            break
        if not progressed:
            log(f"waiting; {len(remaining)} tasks remain (blocked/foreign): {remaining[:6]}")
            time.sleep(600 + 60 * C.WORKER_ID)


if __name__ == "__main__":
    main()
