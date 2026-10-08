"""Dò tham số (phạt địa điểm theo LOẠI) cho một robot bằng tối ưu từng tọa độ; đo thêm trên validation.
    python scratch/p2_cd2.py <robot> [n_train] [cond]"""
import sys, json, time, numpy as np
sys.path.insert(0, "scratch")
from p2_fast2 import *

r = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 700; cond = sys.argv[3] if len(sys.argv) > 3 else "all"
C = {"all": lambda x: True, "rain": lambda x: x["w"]["rain"], "dry": lambda x: not x["w"]["rain"],
     "urg": lambda x: x["m"]["urgent"], "nourg": lambda x: not x["m"]["urgent"],
     "frag": lambda x: x["m"]["fragile"], "nofrag": lambda x: not x["m"]["fragile"]}[cond]
D = [x for x in load("train") if C(x)][:N]
V = [x for x in load("validation") if C(x)]
G = [SG2(x, r == 4) for x in D]; GV = [SG2(x, r == 4) for x in V]
VALS = {"crowd": [0, 0.5, 1, 2, 3, 5, 8, 12], "cover": [-0.9, -0.6, -0.3, 0, 0.5, 1, 2, 4], "stairs": [-0.5, 0, 1, 3, 10],
        "tR": [0, 0.25, 0.5, 1, 2, 4], "tL": [0, 0.25, 0.5, 1, 2, 4], "tB": [0, 0.5, 1, 2, 4, 8]}
LMV = [0, 0.5, 1, 2, 3, 5, 8, 12]
KEYS = ["crowd", "cover"] + ["lm_" + t for t in TYPES] + ["tR", "tL", "tB"] + (["stairs"] if r == 4 else [])


def evals(theta, GG, XX):
    u = 0; ins = 0
    for g, x in zip(GG, XX):
        a = argmins(g.q(theta), 1e-6)
        u += a == [x["y"][r]]; ins += x["y"][r] in a
    return u, ins


theta = np.zeros(len(FN2)); theta[0] = 1
for t in TYPES:
    theta[FN2.index("lm_" + t)] = 2
u, ins = evals(theta, G, D); best = u + 0.5 * (ins - u)
t0 = time.time()
for rnd in range(4):
    ch = False
    for k in KEYS:
        i = FN2.index(k)
        for v in (LMV if k.startswith("lm_") else VALS[k]):
            if v == theta[i]:
                continue
            th = theta.copy(); th[i] = v
            u2, ins2 = evals(th, G, D); s = u2 + 0.5 * (ins2 - u2)
            if s > best:
                best, theta, u, ins, ch = s, th, u2, ins2, True
    uv, iv = evals(theta, GV, V)
    print(f"R{r} {cond} vòng {rnd}: train duy nhất đúng {u}/{len(D)} ({u/len(D):.3f}) trong argmin {ins/len(D):.3f} | validation {uv/len(V):.3f} / {iv/len(V):.3f}  ({time.time()-t0:.0f}s)\n   " +
          " ".join(f"{n}={v:g}" for n, v in zip(FN2, theta) if v), flush=True)
    if not ch:
        break
json.dump({"theta": theta.tolist(), "train_uni": u / len(D), "val_uni": uv / len(V)}, open(f"cache/p2_cd2_r{r}_{cond}.json", "w"))
