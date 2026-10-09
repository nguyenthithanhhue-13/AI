"""Bộ tính Q nhanh: Dijkstra của scipy trên đồ thị trạng thái (giao lộ, hướng vừa đi), trọng số = đặc trưng · tham số.
Đặc trưng mỗi bước chuyển: [1, đông, mái che, thường, bậc thang, rẽ phải, rẽ trái, quay đầu]."""
import sys, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, "scratch")
from p2_lib import *

FN = ["one", "crowd", "cover", "normal", "stairs", "lm", "tR", "tL", "tB"]
RELI = {"S": 0, "R": 1, "L": 2, "B": 3}


class SG:
    """Đồ thị trạng thái của một cảnh cho một kiểu robot (có chân hay không)."""

    def __init__(self, w, legged, lmset=frozenset()):
        nodes = sorted(set(w["adj"]) | {w["robot"]} | {p for v in w["landmarks"].values() for p in v})
        self.idx = {n: i for i, n in enumerate(nodes)}
        self.N = len(nodes); self.w = w
        fr, to, F = [], [], []
        for n, dd in w["adj"].items():
            for d, info in dd.items():
                if not edge_ok(info, legged):
                    continue
                n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                if n2 not in self.idx:
                    continue
                st = info[0]
                base = [1.0, st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmset]
                for h in range(4):
                    t = [0.0, 0.0, 0.0]
                    k = RELI[REL[h][d]]
                    if k:
                        t[k - 1] = 1.0
                    # đồ thị NGƯỢC: từ (n2, d) về (n, h)
                    fr.append(self.idx[n2] * 4 + d); to.append(self.idx[n] * 4 + h); F.append(base + t)
        self.fr = np.array(fr, dtype=np.int32); self.to = np.array(to, dtype=np.int32)
        self.F = np.array(F, dtype=np.float64).reshape(-1, len(FN))
        # các bước đầu từ vị trí robot
        s = w["robot"]; h0 = w["heading"]; self.first = []
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, legged):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if n2 not in self.idx:
                continue
            st = info[0]
            t = [0.0, 0.0, 0.0]
            k = RELI[REL[h0][d]]
            if k:
                t[k - 1] = 1.0
            self.first.append((d, self.idx[n2] * 4 + d, np.array([1.0, st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmset], dtype=float), np.array(t)))

    def ctg(self, theta, targets, init=None):
        """Chi phí tới đích cho mọi trạng thái. targets: list giao lộ; init: chi phí ban đầu tại các trạng thái đích (dict) hoặc 0."""
        S = self.N * 4
        wts = self.F @ theta[:9]
        if (wts <= 0).any():
            wts = np.maximum(wts, 1e-6)
        src, dst, ww = [], [], []
        for t in targets:
            if t not in self.idx:
                continue
            for h in range(4):
                v = 0.0 if init is None else init.get((t, h), INF)
                if v < INF:
                    src.append(S); dst.append(self.idx[t] * 4 + h); ww.append(v + 1.0)
        if not src:
            return None
        M = csr_matrix((np.concatenate([wts, ww]), (np.concatenate([self.fr, src]), np.concatenate([self.to, dst]))), shape=(S + 1, S + 1))
        dist = dijkstra(M, indices=S)
        return dist - 1.0

    def q(self, theta, legs, first_turn=1.0):
        V = self.ctg(theta, legs[-1])
        if V is None:
            return [INF] * 4
        for cands in reversed(legs[:-1]):
            init = {}
            for t in cands:
                if t in self.idx:
                    for h in range(4):
                        v = V[self.idx[t] * 4 + h]
                        if np.isfinite(v):
                            init[(t, h)] = v
            V = self.ctg(theta, cands, init)
            if V is None:
                return [INF] * 4
        q = [INF] * 4
        for d, si, base, t in self.first:
            v = V[si]
            if np.isfinite(v):
                q[d] = float(base @ theta[:6] + first_turn * (t @ theta[9:12]) + v)
        return q


def theta_of(crowd=0.0, cover=0.0, normal=0.0, stairs=0.0, lm=0.0, tR=0.0, tL=0.0, tB=0.0, turn=None, one=1.0, tR0=None, tL0=None, tB0=None):
    if turn is not None:
        tR = tL = turn
        tB = tB or turn
    # tR0 / tL0 / tB0: phạt rẽ RIÊNG cho bước đầu (so với hướng mũi robot); mặc định = phạt rẽ dọc đường
    return np.array([one, crowd, cover, normal, stairs, lm, tR, tL, tB,
                     tR if tR0 is None else tR0, tL if tL0 is None else tL0, tB if tB0 is None else tB0])


def lm_excl(x):
    """Giao lộ bị phạt khi đi vào. LMMODE (thí nghiệm): pass = giao lộ có địa điểm (mặc định); adj = giao lộ KỀ địa điểm;
    both = cả hai."""
    import os
    w = x["w"]
    tg = {p for L in x["legs"] for p in L}
    lm = {p for v in w["landmarks"].values() for p in v} - tg
    mode = os.environ.get("LMMODE", "pass")
    if mode == "pass":
        return frozenset(lm)
    adj = {(p[0] + a, p[1] + b) for p in lm for a, b in DRC} - lm - tg - {w["robot"]}
    return frozenset(adj if mode == "adj" else adj | lm)
