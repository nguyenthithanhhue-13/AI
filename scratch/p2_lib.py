"""(vòng private) Thư viện phân tích chiến thuật trên thông tin đúng (scenes.json của train / validation)."""
import sys, collections, heapq, math, pickle
sys.path.insert(0, "src")
from common import *

INF = float("inf")
REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]


def legs_true(s, w):
    m = s["mission"]
    legs = []
    if m["via"]:
        legs.append([tuple(m["via_ref"]["rc"])] if m["via_ref"] else list(w["landmarks"][m["via"]]))
    legs.append([tuple(m["goal_ref"]["rc"])] if m["goal_ref"] else list(w["landmarks"][m["goal"]]))
    return legs


def load(split):
    rows, labels, scenes = load_split(split)
    out = []
    import os
    look = os.environ.get("LOOK") == "1"
    for i, s in enumerate(scenes):
        perm = os.environ.get("PERM")     # thí nghiệm: hoán vị trạng thái, vd "crowded:covered,covered:crowded"
        if perm:
            mp = dict(kv.split(":") for kv in perm.split(","))
            s = dict(s, edges=[dict(e, status=mp.get(e["status"], e["status"])) for e in s["edges"]])
        if look:      # thí nghiệm: trạng thái đoạn đường = KIỂU VẼ (road_look) thay vì trạng thái thật
            s = dict(s, edges=[dict(e, status=s["road_look"].get(e["status"], e["status"])) for e in s["edges"]])
        w = world_from_scene(s)
        out.append(dict(s=s, w=w, legs=legs_true(s, w), y=labels[i * 10:(i + 1) * 10], m=s["mission"]))
    return out


def edge_ok(info, legged):
    status, stairs, allowed = info
    return not (status == "closed" or not allowed or (stairs and not legged))


def cost_to_go(w, terminal, ecost, tcost, legged):
    """Dijkstra ngược trên (giao lộ, hướng vừa đi). ecost(info, n, d) -> chi phí đoạn; tcost(h, d) -> phạt rẽ."""
    adj = w["adj"]
    V = dict(terminal)
    heap = [(v, n, h) for (n, h), v in terminal.items()]
    heapq.heapify(heap)
    while heap:
        v, n2, d = heapq.heappop(heap)
        if v > V.get((n2, d), INF):
            continue
        n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
        info = adj.get(n, {}).get(d)
        if info is None or not edge_ok(info, legged):
            continue
        c = ecost(info, n, d)
        for h in range(4):
            nv = v + c + tcost(h, d)
            if nv < V.get((n, h), INF) - 1e-12:
                V[(n, h)] = nv
                heapq.heappush(heap, (nv, n, h))
    return V


def qvals(w, legs, ecost, tcost, legged, first_turn=1.0):
    term = {(n, h): 0.0 for n in legs[-1] for h in range(4)}
    V = cost_to_go(w, term, ecost, tcost, legged)
    for cands in reversed(legs[:-1]):
        term = {(n, h): V[(n, h)] for n in cands for h in range(4) if (n, h) in V}
        V = cost_to_go(w, term, ecost, tcost, legged)
    s, h0 = w["robot"], w["heading"]
    q = [INF] * 4
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, legged):
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        v = V.get((n2, d), INF)
        q[d] = ecost(info, s, d) + tcost(h0, d) * first_turn + v
    return q


def mk_ecost(normal=0.0, crowd=0.0, cover=0.0, stairs=0.0):
    def f(info, n, d):
        st = info[0]
        c = 1.0 + (crowd if st == "crowded" else cover if st == "covered" else normal)
        if info[1]:
            c += stairs
        return c
    return f


def mk_tcost(R=0.0, L=0.0, B=0.0):
    tab = {"S": 0.0, "R": R, "L": L, "B": B}
    return lambda h, d: tab[REL[h][d]]


def argmins(q, eps=1e-6):
    b = min(q)
    if b == INF:
        return []
    return [d for d in range(4) if q[d] <= b + eps]
