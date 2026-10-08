"""Giả thuyết: chi phí KẾT THÚC phụ thuộc HƯỚNG ĐI VÀO đích (lên / xuống / trái / phải). Dò trên các cảnh KHÔNG điểm ghé.
    python scratch/p2_a56_arrive.py <robot> <key cũ>"""
import sys, json, itertools, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

r = int(sys.argv[1]); key = sys.argv[2]
P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))


def q_arrive(g, th, legs, arr):
    S = g.N * 4
    wts = np.maximum(g.F @ th[:9], 1e-6)
    src, dst, ww = [], [], []
    for t in legs[-1]:
        if t in g.idx:
            for h in range(4):
                src.append(S); dst.append(g.idx[t] * 4 + h); ww.append(arr[h] + 1.0)
    M = csr_matrix((np.concatenate([wts, ww]), (np.concatenate([g.fr, src]), np.concatenate([g.to, dst]))), shape=(S + 1, S + 1))
    V = dijkstra(M, indices=S) - 1.0
    q = [INF] * 4
    for d, si, base, t in g.first:
        if np.isfinite(V[si]):
            q[d] = float(base @ th[:6] + t @ th[9:12] + V[si])
    return q


def acc(D, arr):
    ok = 0
    for x, g, th in D:
        a = argmins(q_arrive(g, th, x["legs"], arr), 1e-6)
        ok += bool(a) and min(a, key=lambda d: "SRLB".index(REL[x["w"]["heading"]][d])) == x["y"][r]
    return ok


def prep(split):
    out = []
    for x in load(split):
        if x["m"]["via"]:
            continue
        p = P.get(group_key(key, x["s"]["style"] == "night", x["w"]["rain"], x["m"]["urgent"], x["m"]["fragile"], False))
        out.append((x, SG(x["w"], r == 4, lm_excl(x)), theta_of(**p)))
    return out


TR = prep("train")[:600]; VA = prep("validation")
base = acc(TR, [0, 0, 0, 0])
best = (base, [0, 0, 0, 0])
for arr in itertools.product([0, 0.5, 1, 2], repeat=4):
    s = acc(TR, list(arr))
    if s > best[0]:
        best = (s, list(arr))
print(f"R{r}: không phạt hướng tới train {base}/{len(TR)} validation {acc(VA, [0,0,0,0])}/{len(VA)} | tốt nhất {best[1]} (lên,xuống,trái,phải): train {best[0]}/{len(TR)} validation {acc(VA, best[1])}/{len(VA)}")
