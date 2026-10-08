"""Giả thuyết: bộ sinh chạy Dijkstra chuẩn với chi phí (số đoạn, số địa điểm đi ngang) và phá hòa theo THỨ TỰ DUYỆT.
Thử nhiều biến thể trên các cảnh KHÔNG có điểm ghé của R0:
  - xuôi (từ robot) / ngược (từ đích); trạng thái = giao lộ hoặc (giao lộ, hướng);
  - khóa phá hòa trong heap: thứ tự chèn (FIFO / LIFO), (hàng, cột), (cột, hàng), có dấu;
  - thứ tự duyệt láng giềng: 24 hoán vị hướng tuyệt đối.
    python scratch/p2_a19_heap2.py [robot]"""
import sys, heapq, itertools, collections
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = [x for x in load("train") if len(x["legs"]) == 1]
leg = r == 4


def cost(x, lmset, n2):
    return 1000 + (n2 in lmset)


def fwd(x, order, keyf, lmset):
    w = x["w"]; s = w["robot"]; T = set(x["legs"][0])
    dist = {s: 0}; par = {s: None}; cnt = itertools.count()
    h = [(0, keyf(s, next(cnt)), s)]; done = set()
    while h:
        c, _, n = heapq.heappop(h)
        if n in done:
            continue
        done.add(n)
        if n in T:
            p = n
            while par[p] != s and par[p] is not None:
                p = par[p]
            return DRC.index((p[0] - s[0], p[1] - s[1])) if par[p] is not None else None
        for d in order:
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, leg):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            nc = c + cost(x, lmset, n2)
            if nc < dist.get(n2, 1 << 60):
                dist[n2] = nc; par[n2] = n
                heapq.heappush(h, (nc, keyf(n2, next(cnt)), n2))
    return None


def bwd(x, order, keyf, lmset):
    w = x["w"]; s = w["robot"]; T = sorted(x["legs"][0])
    dist = {t: 0 for t in T}; nxt = {t: None for t in T}; cnt = itertools.count()
    h = [(0, keyf(t, next(cnt)), t) for t in T]; heapq.heapify(h); done = set()
    while h:
        c, _, n2 = heapq.heappop(h)
        if n2 in done:
            continue
        done.add(n2)
        if n2 == s:
            return nxt[s]
        for d in order:                         # cạnh n -> n2 theo hướng d
            n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, leg):
                continue
            nc = c + cost(x, lmset, n2)
            if nc < dist.get(n, 1 << 60):
                dist[n] = nc; nxt[n] = d
                heapq.heappush(h, (nc, keyf(n, next(cnt)), n))
    return None


KEYS = {"fifo": lambda n, k: k, "lifo": lambda n, k: -k, "rc": lambda n, k: (n[0], n[1]), "-r-c": lambda n, k: (-n[0], -n[1]),
        "cr": lambda n, k: (n[1], n[0]), "-c-r": lambda n, k: (-n[1], -n[0]), "r-c": lambda n, k: (n[0], -n[1]),
        "-rc": lambda n, k: (-n[0], n[1])}
res = []
LM = [{p for v in x["w"]["landmarks"].values() for p in v} - set(x["legs"][0]) for x in D]
for meth, f in (("xuôi", fwd), ("ngược", bwd)):
    for kn, kf in KEYS.items():
        for order in itertools.permutations(range(4)):
            ok = sum(f(x, order, kf, L) == x["y"][r] for x, L in zip(D, LM))
            res.append((ok, meth, kn, "".join("UDLR"[i] for i in order)))
res.sort(reverse=True)
print(f"R{r}: {len(D)} cảnh không điểm ghé")
for t in res[:10]:
    print("  ", t)
