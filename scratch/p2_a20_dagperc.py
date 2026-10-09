"""Học tiêu chí PHỤ trên DAG các đường tối ưu theo chi phí chính (mặc định: số đoạn) bằng perceptron có cấu trúc.
Chỉ dùng các cảnh có hòa và nhãn nằm trong tập tối ưu. Đặc trưng một bước chuyển:
  crowd, cover, normal, stairs, oneway, lm_<loại> (vào giao lộ có địa điểm, trừ đích / ghé), tS0/tR0/tL0/tB0 (bước ĐẦU so
  với hướng mũi), tR, tL, tB (các bước sau), mv_U/D/L/R (hướng tuyệt đối của bước), dU... (bước đầu theo hướng tuyệt đối).
    python scratch/p2_a20_dagperc.py <robot> [epochs]"""
import sys, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *

TYPES = PLACES
FN = ["crowd", "cover", "normal", "stairs", "oneway"] + ["lm_" + t for t in TYPES] + ["tS0", "tR0", "tL0", "tB0", "tR", "tL", "tB"] + \
     ["mvU", "mvD", "mvL", "mvR", "d0U", "d0D", "d0L", "d0R"]
NF = len(FN); IX = {n: i for i, n in enumerate(FN)}


def bfs_dist(w, targets, legged, init=None):
    """Số đoạn ít nhất tới tập đích (init: chi phí ban đầu tại đích) — Dijkstra trọng số 1."""
    import heapq
    dist = {}
    h = []
    for t in targets:
        v = 0 if init is None else init.get(t)
        if v is None:
            continue
        if v < dist.get(t, INF):
            dist[t] = v
            heapq.heappush(h, (v, t))
    while h:
        v, n2 = heapq.heappop(h)
        if v > dist.get(n2, INF):
            continue
        for d in range(4):
            n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, legged):
                continue
            if v + 1 < dist.get(n, INF):
                dist[n] = v + 1
                heapq.heappush(h, (v + 1, n))
    return dist


class Dag:
    def __init__(self, x, legged):
        self.x = x; w = x["w"]; self.w = w; self.legged = legged
        legs = x["legs"]
        self.V = [None] * len(legs)
        self.V[-1] = bfs_dist(w, legs[-1], legged)
        if len(legs) == 2:
            self.V[0] = bfs_dist(w, legs[0], legged, {v: self.V[1].get(v) for v in legs[0]})
        tgt = {p for L in legs for p in L}
        self.lmtype = {p: t for t, v in w["landmarks"].items() for p in v if p not in tgt}
        self.memo = {}

    def feat(self, n, d, h, first):
        info = self.w["adj"][n][d]
        n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
        f = np.zeros(NF)
        st = info[0]
        f[IX["crowd"]] = st == "crowded"; f[IX["cover"]] = st == "covered"; f[IX["normal"]] = st == "normal"
        f[IX["stairs"]] = bool(info[1])
        back = self.w["adj"].get(n2, {}).get(OPP[d])
        f[IX["oneway"]] = back is not None and not back[2]
        if n2 in self.lmtype:
            f[IX["lm_" + self.lmtype[n2]]] = 1
        rel = REL[h][d]
        if first:
            f[IX["t" + rel + "0"]] = 1
            f[IX["d0" + "UDLR"[d]]] = 1
        elif rel != "S":
            f[IX["t" + rel]] = 1
        f[IX["mv" + "UDLR"[d]]] = 1
        return f

    def best(self, k, n, h, theta):
        """(chi phí phụ nhỏ nhất, đặc trưng) từ trạng thái (chặng k, giao lộ n, hướng h) tới hết hành trình, đi trên DAG."""
        key = (k, n, h)
        if key in self.memo:
            return self.memo[key]
        legs = self.x["legs"]
        V = self.V[k]
        cands = []
        if n in legs[k] and V.get(n) == (0 if k == len(legs) - 1 else self.V[k + 1].get(n)):
            cands.append((0.0, np.zeros(NF)) if k == len(legs) - 1 else self.best(k + 1, n, h, theta))
        for d, info in self.w["adj"].get(n, {}).items():
            if not edge_ok(info, self.legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if V.get(n2, INF) != V.get(n, INF) - 1:
                continue
            f = self.feat(n, d, h, False)
            c, g = self.best(k, n2, d, theta)
            cands.append((c + f @ theta, f + g))
        res = min(cands, key=lambda z: z[0]) if cands else (INF, np.zeros(NF))
        self.memo[key] = res
        return res

    def first_moves(self, theta):
        self.memo = {}
        w = self.w; s, h0 = w["robot"], w["heading"]
        V = self.V[0]
        out = {}
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, self.legged):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if V.get(n2, INF) != V.get(s, INF) - 1:
                continue
            f = self.feat(s, d, h0, True)
            c, g = self.best(0, n2, d, theta)
            out[d] = (c + f @ theta, f + g)
        return out


def train(r, epochs=15, lr=0.1, margin=0.01, D=None, verbose=True):
    D = D if D is not None else load("train")
    dags = []
    for x in D:
        g = Dag(x, r == 4)
        fm = g.first_moves(np.zeros(NF))
        if len(fm) > 1 and x["y"][r] in fm:
            dags.append(g)
    theta = np.zeros(NF)
    rng = np.random.default_rng(0)
    best = (-1, theta.copy())
    for ep in range(epochs):
        ok = 0
        for i in rng.permutation(len(dags)):
            g = dags[i]; y = g.x["y"][r]
            fm = g.first_moves(theta)
            cy, fy = fm[y]
            co, fo = min(((c, f) for d, (c, f) in fm.items() if d != y), key=lambda z: z[0])
            ok += cy < co - 1e-9
            if cy > co - margin:
                theta -= lr * (fy - fo)
        acc = ok / len(dags)
        if acc > best[0]:
            best = (acc, theta.copy())
        if verbose:
            print(f"R{r} ep {ep}: {ok}/{len(dags)} = {acc:.4f}", flush=True)
    return best, dags


if __name__ == "__main__":
    r = int(sys.argv[1]); ep = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    (acc, th), dags = train(r, ep)
    print("TỐT NHẤT", f"{acc:.4f}")
    for n, v in sorted(zip(FN, th), key=lambda z: -abs(z[1])):
        if abs(v) > 1e-9:
            print(f"   {n:12s} {v:+.3f}")
    np.save(f"cache/p2_dagperc_r{r}.npy", th)
