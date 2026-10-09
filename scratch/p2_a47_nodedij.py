"""Giả thuyết CÀI ĐẶT: Dijkstra trên GIAO LỘ (không có trạng thái hướng), phạt rẽ tính theo hướng đi vào nút từ CHA tốt nhất
(hướng mũi robot ở nút xuất phát). Xuôi từ robot tới điểm đến; với điểm ghé: chặng 1 tới điểm ghé gần nhất theo chi phí, rồi
chặng 2 từ đó (hướng vào = hướng của bước cuối chặng 1). So với Dijkstra trên (giao lộ, hướng) cùng tham số.
    python scratch/p2_a47_nodedij.py <robot> <key>"""
import sys, json, heapq, itertools
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import group_key

r = int(sys.argv[1]); key = sys.argv[2]
P = json.load(open(f"cache/p2_cd3_r{r}_{key}.json", encoding="utf-8"))
leg = r == 4


def node_dij(x, start, h0, targets, p, L):
    """Trả về (chi phí, bước đầu, nút đích đạt được, hướng cuối)."""
    w = x["w"]
    tc = {"S": 0.0, "R": p.get("tR", 0), "L": p.get("tL", 0), "B": p.get("tB", 0)}
    dist = {start: 0.0}; inc = {start: h0}; first = {start: None}
    cnt = itertools.count(); hp = [(0.0, next(cnt), start)]; done = set()
    while hp:
        c, _, n = heapq.heappop(hp)
        if n in done:
            continue
        done.add(n)
        if n in targets and n != start:
            return c, first[n], n, inc[n]
        for d, info in w["adj"].get(n, {}).items():
            if not edge_ok(info, leg):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            st = info[0]
            e = 1 + p.get("crowd", 0) * (st == "crowded") + p.get("cover", 0) * (st == "covered") + p.get("lm", 0) * (n2 in L) \
                + p.get("stairs", 0) * bool(info[1]) + tc[REL[inc[n]][d]]
            nc = c + max(e, 1e-6)
            if nc < dist.get(n2, INF) - 1e-12:
                dist[n2] = nc; inc[n2] = d; first[n2] = d if n == start else first[n]
                heapq.heappush(hp, (nc, next(cnt), n2))
    return INF, None, None, None


def predict(x):
    w = x["w"]
    p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
    L = lm_excl(x)
    legs = x["legs"]
    if len(legs) == 1:
        return node_dij(x, w["robot"], w["heading"], set(legs[0]), p, L)[1]
    best = (INF, None)
    for v in legs[0]:
        c1, d1, _, hv = node_dij(x, w["robot"], w["heading"], {v}, p, L)
        c2 = node_dij(x, v, hv, set(legs[1]), p, L)[0] if c1 < INF else INF
        if c1 + c2 < best[0]:
            best = (c1 + c2, d1)
    return best[1]


def state_pred(x):
    w = x["w"]
    p = P.get(group_key(key, x["s"]["style"] == "night", w["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
    q = SG(w, leg, lm_excl(x)).q(theta_of(**p), x["legs"])
    a = argmins(q, 1e-6)
    return min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d])) if a else None


for split in ("train", "validation"):
    D = load(split)[:800]
    a = sum(predict(x) == x["y"][r] for x in D) / len(D)
    b = sum(state_pred(x) == x["y"][r] for x in D) / len(D)
    print(f"R{r} {key} {split}: Dijkstra trên giao lộ {a:.3f} | trên (giao lộ, hướng) {b:.3f}", flush=True)
