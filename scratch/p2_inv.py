"""Học NGƯỢC hàm chi phí của một robot (structured perceptron trên đường đi tối ưu).

Đặc trưng mỗi bước chuyển (giao lộ, hướng) -> (giao lộ kế, hướng mới):
  one (=1, trọng số cố định 1), crowd, cover, stairs, lm_<loại> (đi VÀO giao lộ có địa điểm loại đó, trừ đích / điểm ghé
  của nhiệm vụ), tR, tL, tB (rẽ phải / trái / quay đầu so với hướng đang đi; bước đầu tính theo hướng mũi robot).
Với mỗi cảnh: đường tốt nhất bắt đầu bằng bước của nhãn phải rẻ hơn (chặt, lề MARGIN) đường tốt nhất của mọi bước khác.
Vi phạm -> theta += lr * (f(đường nhãn) - f(đường khác tốt nhất)) với dấu sao cho chi phí đường nhãn giảm.
    python scratch/p2_inv.py <robot> [điều kiện: all|rain|dry|urg|nourg|frag|nofrag] [epochs]"""
import sys, time, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, "scratch")
from p2_lib import *

TYPES = PLACES
FN = ["one", "crowd", "cover", "stairs"] + ["lm_" + t for t in TYPES] + ["tR", "tL", "tB"]
NF = len(FN)
RELI = {"S": 0, "R": 1, "L": 2, "B": 3}


class G:
    def __init__(self, x, legged):
        w = x["w"]
        self.w = w; self.legs = x["legs"]
        nodes = sorted(set(w["adj"]) | {w["robot"]} | {p for v in w["landmarks"].values() for p in v})
        self.idx = {n: i for i, n in enumerate(nodes)}
        self.N = len(nodes)
        tgt = {p for L in x["legs"] for p in L}
        lmtype = {p: t for t, v in w["landmarks"].items() for p in v if p not in tgt}
        fr, to, F = [], [], []
        self.edge_of = {}
        for n, dd in w["adj"].items():
            for d, info in dd.items():
                if not edge_ok(info, legged):
                    continue
                n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                if n2 not in self.idx:
                    continue
                st = info[0]
                base = np.zeros(NF)
                base[0] = 1; base[1] = st == "crowded"; base[2] = st == "covered"; base[3] = bool(info[1])
                if n2 in lmtype:
                    base[4 + TYPES.index(lmtype[n2])] = 1
                for h in range(4):
                    f = base.copy()
                    k = RELI[REL[h][d]]
                    if k:
                        f[4 + len(TYPES) + k - 1] = 1
                    u, v = self.idx[n2] * 4 + d, self.idx[n] * 4 + h          # đồ thị NGƯỢC: (n2,d) -> (n,h)
                    self.edge_of[(u, v)] = len(F)
                    fr.append(u); to.append(v); F.append(f)
        self.fr = np.array(fr, dtype=np.int32); self.to = np.array(to, dtype=np.int32)
        self.F = np.array(F).reshape(-1, NF)
        s, h0 = w["robot"], w["heading"]
        self.first = {}
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, legged):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if n2 not in self.idx:
                continue
            st = info[0]
            f = np.zeros(NF)
            f[0] = 1; f[1] = st == "crowded"; f[2] = st == "covered"; f[3] = bool(info[1])
            if n2 in lmtype:
                f[4 + TYPES.index(lmtype[n2])] = 1
            k = RELI[REL[h0][d]]
            if k:
                f[4 + len(TYPES) + k - 1] = 1
            self.first[d] = (self.idx[n2] * 4 + d, f)

    def _leg(self, wts, targets, init):
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
            return None, None
        M = csr_matrix((np.concatenate([wts, ww]), (np.concatenate([self.fr, src]), np.concatenate([self.to, dst]))), shape=(S + 1, S + 1))
        dist, pred = dijkstra(M, indices=S, return_predecessors=True)
        return dist - 1.0, pred

    def solve(self, theta):
        """Trả về dict d -> (chi phí, vector đặc trưng của đường tốt nhất bắt đầu bằng d)."""
        wts = np.maximum(self.F @ theta, 1e-3)
        legs = self.legs
        S = self.N * 4
        V2, P2 = self._leg(wts, legs[-1], lambda t, h: 0.0)
        if V2 is None:
            return {}
        if len(legs) == 2:
            V1, P1 = self._leg(wts, legs[0], lambda t, h: V2[self.idx[t] * 4 + h])
            if V1 is None:
                return {}
            chain = [(V1, P1), (V2, P2)]
        else:
            chain = [(V2, P2)]
        out = {}
        for d, (si, f0) in self.first.items():
            V, P = chain[0]
            if not np.isfinite(V[si]):
                continue
            feats = f0.copy()
            k = 0; u = si
            while True:
                p = P[u]
                if p == S or p < 0:
                    if k + 1 < len(chain):
                        k += 1
                        V, P = chain[k]
                        if P[u] < 0:
                            break
                        continue
                    break
                feats += self.F[self.edge_of[(p, u)]] if (p, u) in self.edge_of else 0
                u = p
            out[d] = (float(max(f0 @ theta, 1e-3) + V[si] - 0.0), feats)
        return out


def run(r, cond="all", epochs=30, lr=0.05, margin=0.05, theta0=None, D=None, verbose=True):
    D = D if D is not None else load("train")
    C = {"all": lambda x: True, "rain": lambda x: x["w"]["rain"], "dry": lambda x: not x["w"]["rain"],
         "urg": lambda x: x["m"]["urgent"], "nourg": lambda x: not x["m"]["urgent"],
         "frag": lambda x: x["m"]["fragile"], "nofrag": lambda x: not x["m"]["fragile"]}[cond]
    X = [x for x in D if C(x)]
    Gs = [G(x, r == 4) for x in X]
    theta = np.zeros(NF) if theta0 is None else theta0.copy()
    theta[0] = 1.0
    best = (-1, theta.copy())
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(len(X))
        ok = 0
        for i in order:
            res = Gs[i].solve(theta)
            y = X[i]["y"][r]
            if y not in res:
                continue
            cy, fy = res[y]
            others = [(c, f) for d, (c, f) in res.items() if d != y]
            if not others:
                ok += 1
                continue
            co, fo = min(others, key=lambda z: z[0])
            if cy < co - 1e-9:
                ok += 1
            if cy > co - margin:
                theta -= lr * (fy - fo)
                theta[0] = 1.0
                theta[1:] = np.clip(theta[1:], -0.9, 50)
        acc = ok / len(X)
        if acc > best[0]:
            best = (acc, theta.copy())
        if verbose:
            print(f"R{r} {cond} ep {ep} đúng {acc:.4f}  " + " ".join(f"{n}={v:.2f}" for n, v in zip(FN, theta) if abs(v) > 0.02), flush=True)
    return best


if __name__ == "__main__":
    r = int(sys.argv[1]); cond = sys.argv[2] if len(sys.argv) > 2 else "all"; ep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    t = time.time()
    acc, th = run(r, cond, ep, lr=float(sys.argv[4]) if len(sys.argv) > 4 else 0.05)
    print("TỐT NHẤT", f"{acc:.4f}", " ".join(f"{n}={v:.2f}" for n, v in zip(FN, th)), f"{time.time()-t:.0f}s")
