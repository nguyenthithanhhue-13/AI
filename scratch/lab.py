"""Công cụ thử giả thuyết chi phí cho từng robot, chạy song song trên nhiều nhân CPU."""
import sys, itertools, os
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import *
from strategy import *

_D = {}


def _init(split="train"):
    rows, labels, scenes = load_split(split)
    _D["W"] = [world_from_scene(s) for s in scenes]
    _D["M"] = [mission_from_scene(s) for s in scenes]
    _D["Y"] = [labels[i * 10:(i + 1) * 10] for i in range(len(scenes))]


def cond_ok(w, m, cond):
    if cond is None: return True
    k, v = cond
    if k == "rain": return w["rain"] == v
    return m[k] == v


def _eval(job):
    r, cond, P, order = job
    ok = n = 0
    for w, m, ys in zip(_D["W"], _D["M"], _D["Y"]):
        if not cond_ok(w, m, cond): continue
        q = path_q(w, legs_for(w, m), P, r == 4)
        n += 1
        ok += pick(q, w["heading"], order) == ys[r]
    return ok / max(n, 1), n


def run(jobs, split="train", procs=11):
    with Pool(procs, initializer=_init, initargs=(split,)) as p:
        res = p.map(_eval, jobs, chunksize=1)
    return res


def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, vals)) for vals in itertools.product(*kw.values())]


def report(jobs, res, top=6):
    by = {}
    for j, (acc, n) in zip(jobs, res):
        by.setdefault((j[0], j[1]), []).append((acc, n, j[2], j[3]))
    for k, v in by.items():
        v.sort(key=lambda t: -t[0])
        print(f"robot {k[0]} cond {k[1]} (n={v[0][1]})")
        for acc, n, P, order in v[:top]:
            print(f"   {acc:.4f}  {P} {order}")
