"""Quét lưới trọng số (một chiều số đoạn = 1) cho cả 10 robot; lưu số cảnh nhãn ∈ argmin và số cảnh argmin duy nhất đúng,
theo từng nhóm điều kiện. -> cache/p2_grid1.pkl"""
import sys, time, itertools, numpy as np, pickle
sys.path.insert(0, "scratch")
from p2_fast import *
D = load("train")
G0 = [SG(x["w"], False, lm_excl(x)) for x in D]
G1 = [SG(x["w"], True, lm_excl(x)) for x in D]
Y = np.array([x["y"] for x in D])
SPL = {"all": lambda x: True, "rain": lambda x: x["w"]["rain"], "dry": lambda x: not x["w"]["rain"], "urg": lambda x: x["m"]["urgent"],
       "nourg": lambda x: not x["m"]["urgent"], "frag": lambda x: x["m"]["fragile"], "nofrag": lambda x: not x["m"]["fragile"],
       "via": lambda x: bool(x["m"]["via"]), "novia": lambda x: not x["m"]["via"]}
MASK = np.array([[f(x) for x in D] for f in SPL.values()])
grid = list(itertools.product([0, 0.25, 0.5, 1, 2, 3, 5], [-0.5, -0.25, 0, 0.25, 0.5, 1], [0, 0.1, 0.5, 1, 2], [0, 0.25, 0.5, 1, 2]))
res = np.zeros((len(grid), 10, len(SPL), 2), dtype=np.int32)
t = time.time()
for gi, (cr, cv, lm, tu) in enumerate(grid):
    th = theta_of(crowd=cr, cover=cv, lm=lm, tR=tu, tL=tu, tB=2 * tu)
    INS = np.zeros((len(D), 10), bool); UNI = np.zeros((len(D), 10), bool)
    for i, x in enumerate(D):
        for legged, G in ((False, G0), (True, G1)):
            q = G[i].q(th, x["legs"])
            a = argmins(q, 1e-6)
            for r in ([4] if legged else [0, 1, 2, 3, 5, 6, 7, 8, 9]):
                INS[i, r] = Y[i, r] in a; UNI[i, r] = a == [Y[i, r]]
    for si in range(len(SPL)):
        m = MASK[si]
        res[gi, :, si, 0] = INS[m].sum(0); res[gi, :, si, 1] = UNI[m].sum(0)
    if gi % 50 == 0:
        print(gi, len(grid), f"{time.time()-t:.0f}s", flush=True)
        pickle.dump((grid, list(SPL), MASK.sum(1), res[:gi + 1]), open("cache/p2_grid1.pkl", "wb"))
pickle.dump((grid, list(SPL), MASK.sum(1), res), open("cache/p2_grid1.pkl", "wb"))
print("xong", f"{time.time()-t:.0f}s")
