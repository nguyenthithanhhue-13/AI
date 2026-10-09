"""Robot có đi theo đường NGẮN NHẤT THEO ĐỘ DÀI PIXEL (tọa độ giao lộ thật trong scenes.json) không? So với ngắn nhất theo số đoạn.
Với mỗi robot, trên train: tỉ lệ nhãn = bước đầu của đường ngắn nhất pixel; và riêng các cảnh mà số đoạn HÒA giữa >= 2 bước.
    python scratch/p2_pxlen.py"""
import sys, heapq, math, collections
sys.path.insert(0, "scratch")
from p2_lib import *


def dij(w, xy, targets, legged, init=None):
    """chi phí tới đích (độ dài pixel) cho mọi giao lộ; đồ thị ngược."""
    radj = collections.defaultdict(list)
    for n, dd in w["adj"].items():
        for d, info in dd.items():
            if edge_ok(info, legged) and info[0] != "missing":
                m = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                radj[m].append(n)
    D = {}
    h = [((init or {}).get(t, 0.0), t) for t in targets]
    heapq.heapify(h)
    while h:
        c, n = heapq.heappop(h)
        if n in D:
            continue
        D[n] = c
        for m in radj.get(n, []):
            if m not in D and n in xy and m in xy:
                heapq.heappush(h, (c + math.dist(xy[m], xy[n]), m))
    return D


def first_moves(w, xy, legs, legged, metric):
    s = w["robot"]
    if metric == "hop":
        xy1 = {n: (n[1] * 100.0, n[0] * 100.0) for n in xy}
    else:
        xy1 = xy
    V = dij(w, xy1, legs[-1], legged)
    if len(legs) == 2:
        V = dij(w, xy1, legs[0], legged, {v: V.get(v, 1e9) for v in legs[0]})
    sc = {}
    for d, info in w["adj"].get(s, {}).items():
        if edge_ok(info, legged):
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if n2 in V and n2 in xy1:
                sc[d] = math.dist(xy1[s], xy1[n2]) + V[n2]
    return sc


D = load("train") + load("validation")
c = collections.Counter()
for x in D:
    xy = {tuple(n["rc"]): tuple(n["xy"]) for n in x["s"]["nodes"]}
    for r in range(10):
        if r == 4:
            continue
        hp = first_moves(x["w"], xy, x["legs"], False, "hop"); px = first_moves(x["w"], xy, x["legs"], False, "px")
        if not hp:
            continue
        mh = min(hp.values()); tie = [d for d in hp if hp[d] <= mh + 1e-6]
        bpx = min(px, key=px.get); y = x["y"][r]
        c[r, "n"] += 1; c[r, "px"] += bpx == y; c[r, "hop_in"] += y in tie
        if len(tie) >= 2:
            c[r, "tie"] += 1; c[r, "tie_px"] += bpx == y
for r in range(10):
    if r == 4:
        continue
    print(f"R{r}: nhãn = bước ngắn nhất pixel {c[r, 'px'] / c[r, 'n']:.3f} | nhãn thuộc tập ngắn nhất số đoạn {c[r, 'hop_in'] / c[r, 'n']:.3f} | "
          f"khi số đoạn hòa ({c[r, 'tie']} cảnh): nhãn = ngắn nhất pixel {c[r, 'tie_px'] / max(1, c[r, 'tie']):.3f}")
