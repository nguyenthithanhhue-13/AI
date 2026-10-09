"""R0 (luật từ điển): thử mọi thứ tự phá hòa (24 tương đối + 24 tuyệt đối) cho từng nhóm (mưa / khô), thông tin đúng, train / val."""
import sys, json, itertools, collections, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_lib import *
from strat_ml import SG, lm_excl, group_key
cfg = json.load(open("outputs/strategy_hybrid.json", encoding="utf-8"))["0"]
res = {}
for split in ("train", "validation"):
    for x in load(split):
        w = x["w"]; legs = x["legs"]
        g = group_key(cfg["key"], x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], len(legs) == 2)
        p = cfg["params"][g]
        q = SG(w, False, lm_excl(w, legs)).q(np.array(p["theta"]), legs)
        b = min(q); a = [d for d in range(4) if q[d] <= b + 1e-3]
        res.setdefault((split, g), []).append((a, w["heading"], x["y"][0]))
orders = [("rel", "".join(o)) for o in itertools.permutations("SRLB")] + [("abs", o) for o in itertools.permutations(range(4))]
for g in sorted({g for _, g in res}):
    T = res[("train", g)]; V = res.get(("validation", g), [])
    def acc(L, k, o):
        return sum((min(a, key=lambda d: o.index(REL[h][d])) if k == "rel" else min(a, key=lambda d: list(o).index(d))) == y for a, h, y in L)
    sc = sorted(((acc(T, k, o), k, o) for k, o in orders), key=lambda z: -z[0])[:4]
    cur = cfg["params"][g]["order"]
    print(g, "hiện tại", cur, acc(T, *cur), "/", len(T), "| tốt nhất:", [(k, "".join(map(str, o)), a, acc(V, k, o)) for a, k, o in sc], "val n", len(V))
