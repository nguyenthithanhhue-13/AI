"""Trên VALIDATION: tránh địa điểm gây nhiễu (theo câu) so với tránh mọi địa điểm."""
import sys, pickle, itertools, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
import nlp3
from common import CACHE
split = sys.argv[1] if len(sys.argv) > 1 else "validation"
D = load(split)[:600]
cp = CACHE / f"p2_dis_{split}600.pkl"
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
for name, LS in {"mọi địa điểm": [frozenset({p for v in x["w"]["landmarks"].values() for p in v} - t) for x, t in zip(D, tg)],
                 "chỉ gây nhiễu": [frozenset({p for k, v in x["w"]["landmarks"].items() if k in ds for p in v} - t) for x, t, ds in zip(D, tg, DIS)]}.items():
    G0 = [SG(x["w"], False, L) for x, L in zip(D, LS)]; G1 = [SG(x["w"], True, L) for x, L in zip(D, LS)]
    best = np.zeros(10); bp = [None] * 10
    for lm, cr, cv in itertools.product([0, 1, 3, 8], [0, 1, 3], [-0.5, 0, 0.5]):
        th = theta_of(lm=lm, crowd=cr, cover=cv, tB=1)
        ins = np.zeros(10)
        for i, x in enumerate(D):
            for legged, G in ((False, G0), (True, G1)):
                a = argmins(G[i].q(th, x["legs"]), 1e-6)
                for r in ([4] if legged else [0, 1, 2, 3, 5, 6, 7, 8, 9]):
                    ins[r] += a == [Y[i, r]]
        for r in range(10):
            if ins[r] > best[r]: best[r] = ins[r]; bp[r] = (lm, cr, cv)
    print(split, name, "| duy nhất đúng tốt nhất:", " ".join(f"R{r}={best[r]/len(D):.3f}{bp[r]}" for r in range(10)), flush=True)
