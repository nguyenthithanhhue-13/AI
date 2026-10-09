"""Mô hình chi phí CỘNG ĐIỀU KIỆN: trọng số đặc trưng f của cảnh = gốc_f + sum_c delta_f^c * [điều kiện c],
c ∈ {đêm, mưa, dễ vỡ, gấp, có điểm ghé}. Dò từng tọa độ trên train, đo validation (hòa: thẳng > phải > trái > quay đầu).
    python scratch/p2_cd4.py <robot> [n_train]"""
import sys, json, time, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 800
FEATS = ["crowd", "cover", "lm", "tR", "tL", "tB"] + (["stairs"] if r == 4 else [])
CONDS = ["night", "rain", "frag", "urg", "via"]
IDX = {"crowd": 1, "cover": 2, "stairs": 4, "lm": 5, "tR": 6, "tL": 7, "tB": 8}
BASEV = {"crowd": [0, 0.5, 1, 2, 3, 5, 8], "cover": [-0.9, -0.6, -0.3, 0, 0.5, 1, 2], "lm": [0, 0.5, 1, 2, 3, 5, 8],
         "tR": [0, 0.25, 0.5, 1, 2], "tL": [0, 0.25, 0.5, 1, 2], "tB": [0, 0.5, 1, 2, 4, 8], "stairs": [-0.5, 0, 1, 3, 10]}
DELV = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]


def conds(x):
    return np.array([x["s"]["style"] == "night", x["w"]["rain"], x["m"]["fragile"], x["m"]["urgent"], bool(x["m"]["via"])], float)


def prep(D):
    return [(SG(x["w"], r == 4, lm_excl(x)), x["legs"], conds(x), x["y"][r], x["w"]["heading"]) for x in D]


TR = prep(load("train")[:N]); VA = prep(load("validation"))
P = {("b", f): 0.0 for f in FEATS}
P.update({(c, f): 0.0 for c in CONDS for f in FEATS})


def theta(P, cv):
    th = theta_of()
    for f in FEATS:
        th[IDX[f]] = P[("b", f)] + sum(P[(c, f)] * cv[i] for i, c in enumerate(CONDS))
    return th


def score(P, data, exact=False):
    u = ins = ok = 0
    for g, legs, cv, y, h in data:
        q = g.q(theta(P, cv), legs)
        a = argmins(q, 1e-6)
        u += a == [y]; ins += y in a
        if exact:
            ok += bool(a) and min(a, key=lambda d: "SRLB".index(REL[h][d])) == y
    return (u + 0.5 * (ins - u)) if not exact else ok


best = score(P, TR); t0 = time.time()
for rnd in range(3):
    ch = False
    for key in list(P):
        vals = BASEV[key[1]] if key[0] == "b" else DELV
        for v in vals:
            if v == P[key]:
                continue
            Q = dict(P); Q[key] = v
            s = score(Q, TR)
            if s > best:
                best, P, ch = s, Q, True
    tr = score(P, TR, True); va = score(P, VA, True)
    print(f"R{r} vòng {rnd}: train {tr}/{len(TR)} = {tr/len(TR):.4f}  validation {va}/{len(VA)} = {va/len(VA):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print("   ", {f"{c}.{f}": v for (c, f), v in P.items() if v}, flush=True)
    if not ch:
        break
json.dump({f"{c}|{f}": v for (c, f), v in P.items()}, open(f"cache/p2_cd4_r{r}.json", "w"))
