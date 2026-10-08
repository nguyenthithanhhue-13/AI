"""Giả thuyết: bộ sinh dữ liệu tìm đường bằng BFS / Dijkstra chuẩn, hòa được phá theo THỨ TỰ DUYỆT (thứ tự hướng của
láng giềng, thứ tự lấy ra khỏi hàng đợi) -> bước đầu phụ thuộc cả cây tìm kiếm, không chỉ bước đầu.
Thử: BFS xuôi từ robot (cha = nút lấy ra trước), BFS ngược từ đích (bước kế = nút lấy ra trước), heap theo (chi phí, hàng, cột).
    python scratch/p2_a6_bfsorder.py <robot> [split]"""
import sys, collections, itertools, heapq
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]); split = sys.argv[2] if len(sys.argv) > 2 else "train"
D = load(split)
leg = r == 4


def fwd_bfs_first(w, start, targets, order):
    """BFS xuôi; trả về (khoảng cách, bước đầu) tới đích gặp trước tiên."""
    par = {start: None}; q = collections.deque([start]); first = {start: None}
    while q:
        n = q.popleft()
        if n in targets:
            return first[n], n
        for d in order:
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, leg):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if n2 in par:
                continue
            par[n2] = n; first[n2] = d if n == start else first[n]
            q.append(n2)
    return None, None


def bwd_bfs_next(w, start, targets, order):
    """BFS ngược từ tập đích; bước kế của mỗi nút = hướng tới nút đã phát hiện ra nó."""
    nxt = {t: None for t in targets}; q = collections.deque(sorted(targets))
    while q:
        n2 = q.popleft()
        if n2 == start:
            return nxt[start]
        for d in order:       # d = hướng đi từ n tới n2
            n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
            info = w["adj"].get(n, {}).get(d)
            if info is None or not edge_ok(info, leg) or n in nxt:
                continue
            nxt[n] = d
            q.append(n)
    return nxt.get(start)


def run(method, order):
    ok = 0; tot = 0
    for x in D:
        w = x["w"]; legs = x["legs"]; y = x["y"][r]
        s = w["robot"]
        if method == "fwd":
            if len(legs) == 1:
                d, _ = fwd_bfs_first(w, s, set(legs[0]), order)
            else:     # hai chặng: chọn tổ hợp ít đoạn nhất (gần đúng: via gặp trước)
                d, v = fwd_bfs_first(w, s, set(legs[0]), order)
        else:
            if len(legs) == 1:
                d = bwd_bfs_next(w, s, set(legs[0]), order)
            else:
                d = bwd_bfs_next(w, s, set(legs[0]), order)
        tot += 1; ok += d == y
    return ok / tot


ORD = list(itertools.permutations(range(4)))
for method in ("fwd", "bwd"):
    res = sorted(((run(method, o), o) for o in ORD), reverse=True)[:4]
    print(method, [(round(a, 4), "".join("UDLR"[i] for i in o)) for a, o in res], flush=True)
