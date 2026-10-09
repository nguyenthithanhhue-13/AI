"""Bộ tính Q nhanh với phạt đi ngang địa điểm RIÊNG THEO LOẠI. Đặc trưng một bước chuyển:
[1, đông, mái che, thường, bậc thang, lm_<10 loại>, rẽ phải, rẽ trái, quay đầu] (18)."""
import sys, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, "scratch")
from p2_lib import *

TYPES = PLACES
FN2 = ["one", "crowd", "cover", "normal", "stairs"] + ["lm_" + t for t in TYPES] + ["tR", "tL", "tB"]
NE = 5 + len(TYPES)
RELI = {"S": 0, "R": 1, "L": 2, "B": 3}


class SG2:
    def __init__(self, x, legged):
        w = x["w"]
        nodes = sorted(set(w["adj"]) | {w["robot"]} | {p for v in w["landmarks"].values() for p in v})
        self.idx = {n: i for i, n in enumerate(nodes)}
        self.N = len(nodes)
        tgt = {p for L in x["legs"] for p in L}
        lt = {p: TYPES.index(t) for t, v in w["landmarks"].items() for p in v if p not in tgt}
        fr, to, F = [], [], []

        def base(info, n2):
            b = np.zeros(NE)
            st = info[0]
            b[0] = 1; b[1] = st == "crowded"; b[2] = st == "covered"; b[3] = st == "normal"; b[4] = bool(info[1])
            if n2 in lt:
                b[5 + lt[n2]] = 1
            return b
        for n, dd in w["adj"].items():
            for d, info in dd.items():
                if not edge_ok(info, legged):
                    continue
                n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                if n2 not in self.idx:
                    continue
                b = base(info, n2)
                for h in range(4):
                    t = np.zeros(3)
                    k = RELI[REL[h][d]]
                    if k:
                        t[k - 1] = 1
                    fr.append(self.idx[n2] * 4 + d); to.append(self.idx[n] * 4 + h); F.append(np.concatenate([b, t]))
        self.fr = np.array(fr, dtype=np.int32); self.to = np.array(to, dtype=np.int32)
        self.F = np.array(F).reshape(-1, NE + 3)
        s, h0 = w["robot"], w["heading"]
        self.first = []
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, legged):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if n2 not in self.idx:
                continue
            t = np.zeros(3)
            k = RELI[REL[h0][d]]
            if k:
                t[k - 1] = 1
            self.first.append((d, self.idx[n2] * 4 + d, np.concatenate([base(info, n2), t])))
        self.legs = x["legs"]

    def _ctg(self, wts, targets, init):
        S = self.N * 4
        src, dst, ww = [], [], []
        for t in targets:
            if t not in self.idx:
                continue
            for h in range(4):
                v = init(t, h)
                if v is not None and np.isfinite(v):
                    src.append(S); dst.append(self.idx[t] * 4 + h); ww.append(v + 1.0)
        if not src:
            return None
        M = csr_matrix((np.concatenate([wts, ww]), (np.concatenate([self.fr, src]), np.concatenate([self.to, dst]))), shape=(S + 1, S + 1))
        return dijkstra(M, indices=S) - 1.0

    def q(self, theta):
        wts = np.maximum(self.F @ theta, 1e-6)
        legs = self.legs
        V = self._ctg(wts, legs[-1], lambda t, h: 0.0)
        if V is None:
            return [INF] * 4
        for cands in reversed(legs[:-1]):
            Vp = V
            V = self._ctg(wts, cands, lambda t, h: Vp[self.idx[t] * 4 + h])
            if V is None:
                return [INF] * 4
        q = [INF] * 4
        for d, si, f in self.first:
            if np.isfinite(V[si]):
                q[d] = float(max(f @ theta, 1e-6) + V[si])
        return q
