"""Giả thuyết: Dijkstra XUÔI từ robot, heap phá hòa theo khóa của giao lộ; cha = nút đầu tiên cải thiện (chặt) chi phí.
Truy ngược từ đích lấy ra đầu tiên -> bước đầu."""
import sys, heapq, itertools, collections
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1]); D = load("train"); leg = r == 4
def fwd(w, start, targets, keyf, order, strict=True):
    dist = {start: 0}; par = {start: None}; h = [(0, keyf(start), start)]; done = set()
    while h:
        c, _, n = heapq.heappop(h)
        if n in done: continue
        done.add(n)
        if n in targets:
            path = [n]
            while par[path[-1]] is not None: path.append(par[path[-1]])
            path.reverse()
            if len(path) < 2: return None
            a, b = path[0], path[1]
            return DRC.index((b[0] - a[0], b[1] - a[1]))
        for d in order:
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, leg): continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            nc = c + 1
            if nc < dist.get(n2, INF) or (not strict and nc == dist.get(n2) and n2 not in done):
                dist[n2] = nc; par[n2] = n
                heapq.heappush(h, (nc, keyf(n2), n2))
    return None
KEYS = {"rc": lambda n: (n[0], n[1]), "-r-c": lambda n: (-n[0], -n[1]), "cr": lambda n: (n[1], n[0]), "-c-r": lambda n: (-n[1], -n[0]),
        "r-c": lambda n: (n[0], -n[1]), "-rc": lambda n: (-n[0], n[1]), "c-r": lambda n: (n[1], -n[0]), "-cr": lambda n: (-n[1], n[0])}
res = []
for kn, kf in KEYS.items():
    for order in itertools.permutations(range(4)):
        for strict in (True, False):
            ok = 0
            for x in D:
                legs = x["legs"]
                d = fwd(x["w"], x["w"]["robot"], set(legs[0]), kf, order, strict)
                ok += d == x["y"][r]
            res.append((ok / len(D), kn, "".join("UDLR"[i] for i in order), strict))
res.sort(reverse=True)
for t in res[:8]: print(t)
