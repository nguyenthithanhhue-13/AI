"""Ca hòa cùng đích: với từng bước đầu, min/max các đặc trưng trên MỌI đường ngắn nhất (theo số đoạn) bắt đầu bằng bước đó.
    python scratch/p2_a5_pathfeat.py <robot>"""
import sys, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1])
single, multi = pickle.load(open(f"cache/p2_tiesplit_r{r}.pkl", "rb"))
leg = r == 4


def hopdist(w, targets):
    """BFS ngược: số đoạn ít nhất từ mỗi giao lộ tới tập đích (theo đoạn đi được)."""
    dist = {t: 0 for t in targets}
    frontier = list(targets)
    while frontier:
        nxt = []
        for n2 in frontier:
            for d in range(4):
                n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
                info = w["adj"].get(n, {}).get(d)
                if info is None or not edge_ok(info, leg) or n in dist:
                    continue
                dist[n] = dist[n2] + 1
                nxt.append(n)
        frontier = nxt
    return dist


def leg_feats(w, start, h0, targets):
    """DP trên DAG đường ngắn nhất: với mỗi bước đầu d -> dict đặc trưng (min, max)."""
    dist = hopdist(w, targets)
    memo = {}

    def best(n, h):   # trả về tuple các (min, max) của (crowd, cover, normal, turns, turnsR, turnsL) từ n (vừa đi hướng h) tới đích
        if n in targets and dist[n] == 0:
            return [(0, 0)] * 6
        key = (n, h)
        if key in memo:
            return memo[key]
        acc = None
        for d, info in w["adj"].get(n, {}).items():
            if not edge_ok(info, leg):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if dist.get(n2, INF) != dist[n] - 1:
                continue
            sub = best(n2, d)
            rel = REL[h][d]
            add = [info[0] == "crowded", info[0] == "covered", info[0] == "normal", rel != "S", rel == "R", rel == "L"]
            v = [(a + s[0], a + s[1]) for a, s in zip(add, sub)]
            acc = v if acc is None else [(min(x[0], y[0]), max(x[1], y[1])) for x, y in zip(acc, v)]
        memo[key] = acc
        return acc

    out = {}
    for d, info in w["adj"].get(start, {}).items():
        if not edge_ok(info, leg):
            continue
        n2 = (start[0] + DRC[d][0], start[1] + DRC[d][1])
        if dist.get(n2, INF) != dist.get(start, INF) - 1:
            continue
        sub = best(n2, d)
        rel = REL[h0][d]
        add = [info[0] == "crowded", info[0] == "covered", info[0] == "normal", rel != "S", rel == "R", rel == "L"]
        out[d] = [(a + s[0], a + s[1]) for a, s in zip(add, sub)]
    return out


NAMES = ["crowd", "cover", "normal", "turns", "turnsR", "turnsL"]
stat = collections.Counter(); n = 0
for x, a, y, reach in single:
    common = set.intersection(*[reach[d] for d in a])
    cb = sorted(common)[0]
    if len(cb) > 1:      # có điểm ghé: bỏ qua cho đơn giản
        continue
    w = x["w"]
    F = leg_feats(w, w["robot"], w["heading"], {cb[0]})
    if y not in F or any(d not in F for d in a):
        stat["thiếu"] += 1
        continue
    n += 1
    for i, nm in enumerate(NAMES):
        for j, mm in enumerate(("min", "max")):
            vals = [F[d][i][j] for d in a]
            if len(set(vals)) > 1:
                stat[f"{nm}.{mm} khác nhau"] += 1
                stat[f"{nm}.{mm} chọn nhỏ nhất"] += F[y][i][j] == min(vals)
                stat[f"{nm}.{mm} chọn lớn nhất"] += F[y][i][j] == max(vals)
print("số ca (không điểm ghé)", n, stat.get("thiếu", 0))
for k in sorted(stat):
    if "khác" in k:
        base = k.replace(" khác nhau", "")
        m = stat[k]
        print(f"  {base:14s} khác nhau {m:4d}   chọn nhỏ nhất {stat[base + ' chọn nhỏ nhất']/m:.3f}   chọn lớn nhất {stat[base + ' chọn lớn nhất']/m:.3f}")
