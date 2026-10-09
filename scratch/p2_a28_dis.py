"""Giả thuyết: robot tránh đi XUYÊN giao lộ có địa điểm thuộc loại được nhắc như GÂY NHIỄU trong câu.
Loại gây nhiễu lấy từ bộ đọc câu học train (r["excluded"]). So tỉ lệ nhãn ∈ argmin với phạt địa điểm chung.
    python scratch/p2_a28_dis.py"""
import sys, pickle, itertools, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
import nlp3
from common import CACHE

D = load("train")[:800]
cp = CACHE / "p2_dis_train800.pkl"
if cp.exists():
    DIS = pickle.load(open(cp, "rb"))
else:
    p = nlp3.MissionParser3.load(CACHE / "p2_nlp3_trainonly.pkl")
    DIS = []
    for x in D:
        L = x["w"]["landmarks"]
        r = p.parse(x["m"]["text"], {t for t, v in L.items() if v}, L)
        DIS.append(set(r.get("excluded") or ()) | set(r.get("excluded_via") or ()))
    pickle.dump(DIS, open(cp, "wb"))
Y = np.array([x["y"] for x in D])
tg = [{p for Lg in x["legs"] for p in Lg} for x in D]
sets = {
    "lm tất cả": [frozenset({p for v in x["w"]["landmarks"].values() for p in v} - t) for x, t in zip(D, tg)],
    "lm gây nhiễu": [frozenset({p for k, v in x["w"]["landmarks"].items() if k in ds for p in v} - t) for x, t, ds in zip(D, tg, DIS)],
    "lm KHÔNG nhiễu": [frozenset({p for k, v in x["w"]["landmarks"].items() if k not in ds for p in v} - t) for x, t, ds in zip(D, tg, DIS)],
}
for name, LS in sets.items():
    G0 = [SG(x["w"], False, L) for x, L in zip(D, LS)]; G1 = [SG(x["w"], True, L) for x, L in zip(D, LS)]
    res = []
    for lm, cr in itertools.product([0.5, 1, 3, 100], [0, 1]):
        th = theta_of(lm=lm, crowd=cr)
        ins = np.zeros(10)
        for i, x in enumerate(D):
            for legged, G in ((False, G0), (True, G1)):
                a = argmins(G[i].q(th, x["legs"]), 1e-6)
                for r in ([4] if legged else [0, 1, 2, 3, 5, 6, 7, 8, 9]):
                    ins[r] += Y[i, r] in a
        res.append(((lm, cr), ins / len(D)))
    print(name, "| tốt nhất từng robot:", " ".join(f"R{r}={max(z[1][r] for z in res):.3f}" for r in range(10)), flush=True)
