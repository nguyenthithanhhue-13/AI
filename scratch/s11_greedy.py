"""Robot 9 (Tham Lam): chỉ nhìn một bước. Thử các giả thuyết điểm số = chi phí đoạn + khoảng cách còn lại."""
import sys, itertools, math
sys.path.insert(0, "src")
from collections import Counter
from common import *
from strategy import *

rows, labels, scenes = load_split("train")
W = [world_from_scene(s) for s in scenes]
M = [mission_from_scene(s) for s in scenes]
Y = [labels[i * 10 + 9] for i in range(len(scenes))]


def greedy_q(w, m, dist, P, lam=1.0):
    legs = legs_for(w, m)
    targets = legs[0]
    s = w["robot"]
    q = [INF] * 4
    for d, info in w["adj"].get(s, {}).items():
        c = edge_cost(info, P, False)
        if c == INF: continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        if dist == "man":
            h = min(abs(n2[0] - t[0]) + abs(n2[1] - t[1]) for t in targets)
        elif dist == "euc":
            h = min(math.hypot(n2[0] - t[0], n2[1] - t[1]) for t in targets)
        elif dist == "cheb":
            h = min(max(abs(n2[0] - t[0]), abs(n2[1] - t[1])) for t in targets)
        q[d] = c + lam * h + turn_cost(w["heading"], d, P)
    return q


res = []
for dist in ["man", "euc", "cheb"]:
    for crowd, cover, lam, turn in itertools.product([0, 0.5, 1, 2], [0, -0.5], [1, 2, 100], [0, 0.5]):
        P = {"crowd": crowd, "cover": cover, "turn": turn}
        ok = 0; ins = 0
        for i, (w, m) in enumerate(zip(W, M)):
            q = greedy_q(w, m, dist, P, lam)
            ok += pick(q, w["heading"]) == Y[i]
            ins += q[Y[i]] <= min(q) + 1e-6
        res.append((ok / len(Y), ins / len(Y), dist, P, lam))
res.sort(key=lambda t: -t[0])
for r in res[:12]: print(r)
