"""Small helpers for the experiment scripts: parallel map with incremental CSV output, reference discovery."""
from __future__ import annotations

import glob
import json
import os
import traceback

import numpy as np
import pandas as pd


class _Safe:
    """Picklable wrapper: never lose the batch for one failing task."""

    def __init__(self, func):
        self.func = func

    def __call__(self, t):
        try:
            return self.func(t)
        except Exception as e:
            key = str(t.get("task_key", "")) if isinstance(t, dict) else ""
            return dict(task_key=key, error=repr(e), trace=traceback.format_exc()[-1200:])


def pool_map(func, tasks, workers=8, out_csv=None, desc="", keep_order=False):
    """Run func(task) -> dict (or list of dicts) in a forkserver pool; rows are appended to out_csv as they arrive."""
    import multiprocessing as mp
    rows = []
    done_keys = set()
    if out_csv and os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            rows = prev.to_dict("records")
            if "task_key" in prev.columns:
                okp = prev[prev["error"].isna()] if "error" in prev.columns else prev
                done_keys = set(okp["task_key"].astype(str))
                rows = okp.to_dict("records")                     # failed tasks are retried
        except Exception:
            rows = []
    if not tasks:
        return pd.DataFrame(rows)
    todo = [t for t in tasks if str(t.get("task_key", id(t))) not in done_keys] if isinstance(tasks[0], dict) else list(tasks)
    print(f"[{desc}] {len(todo)} tasks ({len(rows)} already done)", flush=True)
    if not todo:
        return pd.DataFrame(rows)

    _safe = _Safe(func)
    if workers <= 1:
        it = (_safe(t) for t in todo)
    else:
        ctx = mp.get_context("forkserver")
        pool = ctx.Pool(workers)
        it = pool.imap(_safe, todo) if keep_order else pool.imap_unordered(_safe, todo)
    n = 0
    for out in it:
        for row in (out if isinstance(out, list) else [out]):
            rows.append(row)
        n += 1
        if out_csv:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        if n % 10 == 0 or n == len(todo):
            print(f"[{desc}] {n}/{len(todo)}", flush=True)
    if workers > 1:
        pool.close(); pool.join()
    return pd.DataFrame(rows)


def find_refs(grid_dir, T, m, phi, tag="ref"):
    """Reference solution files for the given lists of T, m and phase (phase matched to 3 decimals)."""
    out = []
    for T_ in np.atleast_1d(T):
        for m_ in np.atleast_1d(m):
            for ph in np.atleast_1d(phi):
                fs = glob.glob(os.path.join(grid_dir, f"sol_{tag}_T{T_:g}_m{m_:g}_phi{ph:.3f}.pkl"))
                out += fs
    return sorted(out)


def save_json(obj, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
