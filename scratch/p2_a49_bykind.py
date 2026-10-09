import sys, json, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key
r = int(sys.argv[1]); key = sys.argv[2]
P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))
c = collections.Counter()
for x in load("train"):
    w = x["w"]; p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
    a = argmins(SG(w, r == 4, lm_excl(x)).q(theta_of(**p), x["legs"]), 1e-6)
    pred = min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) if a else None
    g = x["m"]["goal_ref"]; k = (g["kind"] if g else "tên") + ("+ghé" if x["m"]["via"] else "")
    c[(k, "n")] += 1; c[(k, "ok")] += pred == x["y"][r]
for k in sorted({k for k, _ in c}):
    print(f"R{r} {k:22s} {c[(k,'ok')]}/{c[(k,'n')]} = {c[(k,'ok')]/c[(k,'n')]:.3f}")
