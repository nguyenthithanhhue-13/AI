"""Công cụ: với chi phí chính cho trước (theo cảnh), tìm các bước đầu hòa và tính min / max đặc trưng trên DAG các đường tối ưu
(không gian trạng thái (giao lộ, hướng vừa đi)), hai chặng (điểm ghé) được xử lý bằng cách nối DAG chặng 1 với DAG chặng 2.
    python scratch/p2_dag.py <robot> <tên chi phí chính>"""
import sys, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *

BIG = 1000.0


def lm_excl(x):
    w = x["w"]
    return {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}


PRIM = {
    "hops_lm": lambda x: ((lambda L: (lambda info, n, d: BIG + ((n[0] + DRC[d][0], n[1] + DRC[d][1]) in L)))(lm_excl(x)), mk_tcost()),
}

FEATS = ["crowd", "cover", "normal", "turns", "tR", "tL", "tB", "oneway", "lm_all", "stairs"]


def step_feats(x, n, d, h, info):
    n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
    rel = REL[h][d]
    allp = {p for v in x["w"]["landmarks"].values() for p in v}
    ow = x["w"]["adj"].get(n2, {}).get(OPP[d])
    return [info[0] == "crowded", info[0] == "covered", info[0] == "normal", rel != "S", rel == "R", rel == "L", rel == "B",
            ow is not None and not ow[2], n2 in allp, bool(info[1])]


def dag_feats(x, ec, tc, legged, eps=1e-6):
    """Trả về (q, argmin, F) với F[d] = list (min, max) đặc trưng trên các đường tối ưu bắt đầu bằng d."""
    w = x["w"]; legs = x["legs"]
    # chi phí tới đích theo từng chặng
    Vs = []
    term = {(n, h): 0.0 for n in legs[-1] for h in range(4)}
    V = cost_to_go(w, term, ec, tc, legged); Vs.append(V)
    for cands in reversed(legs[:-1]):
        term = {(n, h): V[(n, h)] for n in cands for h in range(4) if (n, h) in V}
        V = cost_to_go(w, term, ec, tc, legged); Vs.append(V)
    Vs.reverse()     # Vs[0] = chặng đầu (có tính phần còn lại)
    memo = {}

    def best(k, n, h):
        """Từ trạng thái (n, h) ở chặng k: (min, max) đặc trưng tới cuối hành trình."""
        key = (k, n, h)
        if key in memo:
            return memo[key]
        V = Vs[k]
        acc = None
        # kết thúc chặng k tại n?
        if n in legs[k] and (k == len(legs) - 1 and abs(V[(n, h)]) < eps or k < len(legs) - 1 and abs(V[(n, h)] - Vs[k + 1].get((n, h), INF)) < eps):
            acc = [(0, 0)] * len(FEATS) if k == len(legs) - 1 else best(k + 1, n, h)
        for d, info in w["adj"].get(n, {}).items():
            if not edge_ok(info, legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            v2 = V.get((n2, d), INF)
            if abs(ec(info, n, d) + tc(h, d) + v2 - V[(n, h)]) > eps:
                continue
            sub = best(k, n2, d)
            f = step_feats(x, n, d, h, info)
            v = [(a + s[0], a + s[1]) for a, s in zip(f, sub)]
            acc = v if acc is None else [(min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(acc, v)]
        memo[key] = acc
        return acc

    s, h0 = w["robot"], w["heading"]
    q = qvals(w, legs, ec, tc, legged)
    a = argmins(q, eps)
    F = {}
    for d in a:
        info = w["adj"][s][d]
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        sub = best(0, n2, d)
        f = step_feats(x, s, d, h0, info)
        F[d] = [(fa + t[0], fa + t[1]) for fa, t in zip(f, sub)] if sub else None
    return q, a, F


if __name__ == "__main__":
    r = int(sys.argv[1]); pn = sys.argv[2]
    D = load("train")
    import os
    if os.environ.get("NOSWAP"): D = [x for x in D if all(k == v for k, v in x["s"]["road_look"].items())]
    if os.environ.get("SWAP"): D = [x for x in D if not all(k == v for k, v in x["s"]["road_look"].items())]
    stat = collections.Counter(); n = 0; inset = 0
    for x in D:
        ec, tc = PRIM[pn](x)
        q, a, F = dag_feats(x, ec, tc, r == 4)
        y = x["y"][r]
        inset += y in a
        if len(a) < 2 or y not in a or any(F[d] is None for d in a):
            continue
        n += 1
        for i, nm in enumerate(FEATS):
            for j, mm in enumerate(("min", "max")):
                vals = [F[d][i][j] for d in a]
                if len(set(vals)) > 1:
                    stat[f"{nm}.{mm} khác"] += 1
                    stat[f"{nm}.{mm} nhỏ"] += F[y][i][j] == min(vals)
                    stat[f"{nm}.{mm} lớn"] += F[y][i][j] == max(vals)
    print(f"R{r} {pn}: trong argmin {inset}/{len(D)}, số ca hòa {n}")
    for k in sorted(stat):
        if k.endswith("khác"):
            b = k[:-5]; m = stat[k]
            print(f"  {b:12s} khác {m:4d}  chọn nhỏ nhất {stat[b + ' nhỏ'] / m:.3f}  lớn nhất {stat[b + ' lớn'] / m:.3f}")
