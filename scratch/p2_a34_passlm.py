import sys, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_budget import first_q
from p2_draw import draw
D = load("train"); r = int(sys.argv[1]); k = 0
th_lm = np.array([0, 0, 0, 1, 0, 0, 0], float)
for i, x in enumerate(D):
    if len(x["legs"]) != 1: continue
    q0 = first_q(x, th_lm, 0, r == 4); q2 = first_q(x, th_lm, 2, r == 4)
    if not (min(q0) >= 1 and min(q2) == 0): continue      # đường ngắn nhất buộc xuyên địa điểm, đường +2 tránh được
    y = x["y"][r]
    if q2[y] == 0: continue                                # robot đã tránh
    k += 1
    if k > 4: break
    w = x["w"]
    print(f"#{i} R{r}={ACTIONS[y]} q(địa điểm, ngân sách 0)={q0} q(+2)={q2} hướng mũi {ACTIONS[w['heading']]} mưa={w['rain']} gấp={x['m']['urgent']} dễ vỡ={x['m']['fragile']} 10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
    print("   ", x["m"]["text"][:160])
    print(draw(x))
