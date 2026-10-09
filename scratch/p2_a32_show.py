import sys, json
sys.path.insert(0, "scratch")
from p2_fast import *
from p2_draw import draw
r = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
p = json.load(open(f"cache/p2_cd_r{r}_all.txt"))["p"]; th = theta_of(**p)
D = load("train"); k = 0
for i, x in enumerate(D):
    if len(x["legs"]) != 1 or len(x["legs"][0]) != 1: continue
    g = SG(x["w"], r == 4, lm_excl(x))
    q = g.q(th, x["legs"]); a = argmins(q, 1e-6)
    if x["y"][r] in a: continue
    k += 1
    if k > K: break
    w = x["w"]
    print(f"#{i} R{r} nhãn {ACTIONS[x['y'][r]]}  q={[round(v,2) if v < INF else None for v in q]}  hướng mũi {ACTIONS[w['heading']]} mưa={w['rain']} gấp={x['m']['urgent']} dễ vỡ={x['m']['fragile']}  10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
    print(draw(x))
print(p)
