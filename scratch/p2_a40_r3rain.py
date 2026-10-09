import sys
sys.path.insert(0, "scratch")
from p2_fast import *
from p2_draw import draw
p = dict(crowd=1, lm=0.0, cover=-0.9, tB=0.5); th = theta_of(**p)
D = [x for x in load("train") if x["w"]["rain"] and x["s"]["style"] != "night" and len(x["legs"]) == 1]
k = 0
for x in D:
    g = SG(x["w"], False, lm_excl(x)); q = g.q(th, x["legs"]); a = argmins(q, 1e-6)
    y = x["y"][3]
    if y in a: continue
    q0 = SG(x["w"], False, lm_excl(x)).q(theta_of(), x["legs"])
    k += 1
    if k > 4: break
    w = x["w"]
    print(f"R3={ACTIONS[y]} q={[round(v,2) if v < INF else None for v in q]} hops={[int(v) if v < INF else None for v in q0]} mũi {ACTIONS[w['heading']]} gấp={x['m']['urgent']} dễ vỡ={x['m']['fragile']}  10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
    print(draw(x))
