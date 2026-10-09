import sys, itertools, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *
D = load("train")[:600]
G0 = [SG(x["w"], False, lm_excl(x)) for x in D]; G1 = [SG(x["w"], True, lm_excl(x)) for x in D]
Y = np.array([x["y"] for x in D])
res = []
for lm, cr, cv, tu in itertools.product([3, 5, 10, 100], [0, 0.5, 1, 3], [-0.5, 0, 0.5], [0, 0.5]):
    th = theta_of(crowd=cr, cover=cv, lm=lm, tR=tu, tL=tu, tB=2 * tu)
    ins = np.zeros(10); uni = np.zeros(10)
    for i, x in enumerate(D):
        for legged, G in ((False, G0), (True, G1)):
            a = argmins(G[i].q(th, x["legs"]), 1e-6)
            for r in ([4] if legged else [0, 1, 2, 3, 5, 6, 7, 8, 9]):
                ins[r] += Y[i, r] in a; uni[r] += a == [Y[i, r]]
    res.append(((lm, cr, cv, tu), ins / len(D), uni / len(D)))
for r in range(10):
    b = sorted(res, key=lambda z: (-z[1][r], -z[2][r]))[:2]
    print(f"R{r}", " | ".join(f"{p} ins={i[r]:.3f} uni={u[r]:.3f}" for p, i, u in b))
