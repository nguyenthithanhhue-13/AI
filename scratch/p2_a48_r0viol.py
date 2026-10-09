import sys
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a16_chain import build
from p2_draw import draw
D = load("train"); k = 0
for i, x in enumerate(D):
    ec, tc = build(x, ["hops", "lm", "crowd"])
    q = qvals(x["w"], x["legs"], ec, tc, False); a = argmins(q, 1e-7); y = x["y"][0]
    if y in a: continue
    ec2, tc2 = build(x, ["hops", "lm"]); q2 = qvals(x["w"], x["legs"], ec2, tc2, False)
    k += 1
    if k > 5: continue
    w = x["w"]
    print(f"#{i} R0={ACTIONS[y]} mũi {ACTIONS[w['heading']]} legs={x['legs']} q(hops,lm,crowd)={[round(v%10000) if v < INF else None for v in q]} hòa(hops,lm)={argmins(q2,1e-7)} mưa={w['rain']} đêm={x['s']['style']=='night'} dễ vỡ={x['m']['fragile']} gấp={x['m']['urgent']}  10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
    print(draw(x))
print("tổng vi phạm", k)
