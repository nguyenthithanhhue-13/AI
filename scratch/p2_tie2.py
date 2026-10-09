"""Thứ tự PHÁ HÒA riêng từng robot (đề: "mỗi robot phá hòa theo quy tắc riêng") cho cấu hình lai hiện tại: chọn trên train
trong 24 thứ tự tương đối + 24 tuyệt đối, đo validation. Ghi thứ tự chọn được vào cache/p2_tie2.json."""
import sys, json, itertools
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key, StrategyML

H = json.load(open("outputs/strategy_hybrid.json", encoding="utf-8"))
ORD = [("rel", "".join(o)) for o in itertools.permutations("SRLB")] + [("abs", list(o)) for o in itertools.permutations(range(4))]


def argset(x, r):
    cfg = H[str(r)]
    w = x["w"]; legs = x["legs"]
    goal = next((t for t, v in w["landmarks"].items() if tuple(legs[-1][0]) in v), None)
    p = cfg["params"].get(group_key(cfg["key"], x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], len(legs) == 2, goal))
    if p is None:
        return []
    return argmins(SG(w, r == 4, lm_excl(x)).q(theta_of(**p), legs), 1e-6)


def pick(a, h, o):
    return min(a, key=lambda d: o[1].index(REL[h][d])) if o[0] == "rel" else min(a, key=lambda d: o[1].index(d))


TR = load("train"); VA = load("validation"); out = {}
for r in (1, 2, 3, 4, 5, 6, 7, 8):
    if H[str(r)]["method"] != "cost":
        continue
    At = [(argset(x, r), x["w"]["heading"], x["y"][r]) for x in TR]
    Av = [(argset(x, r), x["w"]["heading"], x["y"][r]) for x in VA]
    sc = lambda A, o: sum(bool(a) and pick(a, h, o) == y for a, h, y in A)
    base_t, base_v = sc(At, ("rel", "SRLB")), sc(Av, ("rel", "SRLB"))
    best = max(ORD, key=lambda o: sc(At, o))
    ties = sum(len(a) > 1 for a, h, y in At)
    out[r] = best
    print(f"R{r}: hòa {ties}/{len(At)} | SRLB train {base_t} val {base_v} -> {best} train {sc(At, best)} val {sc(Av, best)}", flush=True)
json.dump({str(k): v for k, v in out.items()}, open("cache/p2_tie2.json", "w"))
