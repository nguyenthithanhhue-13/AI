import sys, json
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key
from p2_a20_dagperc import bfs_dist
from p2_draw import draw
P = json.load(open("cache/p2_cd3_r3_w.json", encoding="utf-8")); k = 0
for i, x in enumerate(load("train")):
    w = x["w"]; p = P[x["s"]["weather"]]
    q = SG(w, False, lm_excl(x)).q(theta_of(**p), x["legs"]); a = argmins(q, 1e-6)
    pred = min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) if a else None
    dd = bfs_dist(w, x["legs"][0], False).get(w["robot"], 99)
    if dd > 3 or pred == x["y"][3]: continue
    k += 1
    if k > 6: break
    print(f"#{i} R3={ACTIONS[x['y'][3]]} đoán {ACTIONS[pred]} q={[round(v,2) if v < INF else None for v in q]} cách {dd} đoạn, mũi {ACTIONS[w['heading']]} {x['s']['weather']} đêm={x['s']['style']=='night'} legs={x['legs']}  10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
    print(draw(x))
