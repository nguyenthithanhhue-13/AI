import sys, json, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key
from p2_a20_dagperc import bfs_dist
c = collections.Counter()
for r, key in ((1, "nd_u"), (2, "nd"), (3, "w"), (7, "nd_f"), (8, "v"), (5, "nd")):
    P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))
    for x in load("train")[:1200]:
        w = x["w"]; p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
        a = argmins(SG(w, r == 4, lm_excl(x)).q(theta_of(**p), x["legs"]), 1e-6)
        pred = min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) if a else None
        dd = bfs_dist(w, x["legs"][0], r == 4).get(w["robot"], 99)
        b = "1-3" if dd <= 3 else "4-6" if dd <= 6 else "7-9" if dd <= 9 else "10+"
        c[(r, b, "n")] += 1; c[(r, b, "ok")] += pred == x["y"][r]
for r in (1, 2, 3, 5, 7, 8):
    print(f"R{r}: " + "  ".join(f"{b}: {c[(r,b,'ok')]/max(1,c[(r,b,'n')]):.3f} (n={c[(r,b,'n')]})" for b in ("1-3", "4-6", "7-9", "10+")))
