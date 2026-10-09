import sys, time, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *
D = load("train")[:300]
th = theta_of(crowd=2, cover=-0.5, tR=0.5, tL=1.0, tB=3, lm=0.3)
tc = mk_tcost(0.5, 1.0, 3)
bad = 0; t = time.time()
G = [SG(x["w"], False, lm_excl(x)) for x in D]
print("dựng", time.time() - t)
t = time.time()
for x, g in zip(D, G):
    q1 = g.q(th, x["legs"])
    L = lm_excl(x); e0 = mk_ecost(crowd=2, cover=-0.5); ec = lambda i, n, d: e0(i, n, d) + 0.3 * ((n[0] + DRC[d][0], n[1] + DRC[d][1]) in L)
    q2 = qvals(x["w"], x["legs"], ec, tc, False)
    if any(abs(a - b) > 1e-6 for a, b in zip(q1, q2) if a < INF or b < INF) or [a == INF for a in q1] != [b == INF for b in q2]:
        bad += 1
        if bad < 3: print(q1, q2)
print("lệch", bad, "thời gian", time.time() - t)
t = time.time()
for x, g in zip(D, G):
    g.q(th, x["legs"])
print("nhanh", time.time() - t)
