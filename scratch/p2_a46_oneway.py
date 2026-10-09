"""Giả thuyết: robot lập kế hoạch BỎ QUA chiều một chiều (coi là hai chiều); bước đầu vẫn phải hợp lệ.
Dùng tham số chi phí theo điều kiện đã dò (cache/p2_cd3_r<r>_<key>.json), so độ đúng có / không bỏ qua một chiều."""
import sys, json
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key

r = int(sys.argv[1]); key = sys.argv[2]
P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))


def acc(D, relax):
    ok = 0
    for x in D:
        w = x["w"]
        p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
        if p is None:
            continue
        w2 = dict(w, adj={n: {d: (i[0], i[1], True) for d, i in e.items()} for n, e in w["adj"].items()}) if relax else w
        q = SG(w2, r == 4, lm_excl(x)).q(theta_of(**p), x["legs"])
        s = w["robot"]
        q = [v if (d in w["adj"].get(s, {}) and edge_ok(w["adj"][s][d], r == 4)) else INF for d, v in enumerate(q)]
        a = argmins(q, 1e-6)
        ok += bool(a) and min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) == x["y"][r]
    return ok / len(D)


TR = load("train")[:800]; VA = load("validation")
for relax in (False, True):
    print(f"R{r} {key} bỏ qua một chiều={relax}: train {acc(TR, relax):.3f}  validation {acc(VA, relax):.3f}", flush=True)
