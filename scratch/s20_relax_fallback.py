"""Thử cách xử lý "không tìm được đường": làm hỏng bản đồ ĐÚNG (train + validation) giống lỗi CV
(xóa đoạn / đóng nhầm / đảo một chiều) đến khi robot 0 hết đường, rồi so cách cũ (RELAX=None) với cách mới."""
import sys, random
sys.path.insert(0, "src")
from common import *
import strategy
from strategy import predict, q_values, INF

rng = random.Random(0)
res = {}
for split in ("train", "validation"):
    _, labels, scenes = load_split(split)
    for si, s in enumerate(scenes):
        w = world_from_scene(s); m = mission_from_scene(s); y = labels[si * 10:(si + 1) * 10]
        edges = [(a, d) for a in w["adj"] for d in w["adj"][a] if a < (a[0] + DRC[d][0], a[1] + DRC[d][1])]
        adj = {n: dict(e) for n, e in w["adj"].items()}
        bad = dict(w, adj=adj)
        for _ in range(8):
            a, d = rng.choice(edges)
            b = (a[0] + DRC[d][0], a[1] + DRC[d][1])
            kind = rng.choice(["missing", "closed", "oneway"])
            if kind == "missing":
                adj[a].pop(d, None); adj[b].pop(OPP[d], None)
            elif kind == "closed" and d in adj[a]:
                adj[a][d] = ("closed",) + adj[a][d][1:]; adj[b][OPP[d]] = ("closed",) + adj[b][OPP[d]][1:]
            elif kind == "oneway" and d in adj[a]:
                adj[a][d] = adj[a][d][:2] + (False,); adj[b][OPP[d]] = adj[b][OPP[d]][:2] + (True,)
            if min(q_values(bad, m, 0)) == INF:
                break
        else:
            continue
        for name, R in (("cũ", None), ("mới R=5", 5.0), ("mới R=20", 20.0), ("mới R=100", 100.0)):
            strategy.RELAX = R
            p = [predict(bad, m, r) for r in range(10)]
            c = res.setdefault(name, [0, 0])
            c[0] += sum(a == b for a, b in zip(p, y)); c[1] += 10
for name, (ok, n) in res.items():
    print(f"   {name:10s} đúng {ok}/{n} = {ok / n:.3f}  ({n // 10} cảnh bị làm hỏng đến mức hết đường)")
