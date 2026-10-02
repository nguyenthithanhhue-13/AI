"""Chiến thuật của 10 robot: world + mission + robot_id -> hướng đi (0..3).

Bộ máy chung: tìm đường rẻ nhất trên không gian trạng thái (giao lộ, hướng đang quay)
với chi phí = tổng chi phí đoạn đường + phạt đổi hướng. Mỗi robot chỉ khác nhau ở bộ tham số.
"""
import heapq
import math

from common import DRC, OPP, rel_turn

INF = float("inf")
REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]


def resolve_targets(world, typ, ref):
    """Danh sách giao lộ ứng viên cho một loại địa điểm, đã áp dụng tham chiếu không gian (nếu có)."""
    cands = list(world["landmarks"].get(typ, []))
    if len(cands) <= 1 or ref is None:
        return cands
    kind, anchor = ref
    if kind == "north":
        key = lambda p: p[0]
    elif kind == "south":
        key = lambda p: -p[0]
    elif kind == "west":
        key = lambda p: p[1]
    elif kind == "east":
        key = lambda p: -p[1]
    else:
        anc = world["landmarks"].get(anchor, [])
        if not anc:
            return cands
        a = anc[0]
        sign = 1 if kind == "near" else -1
        key = lambda p: sign * math.hypot(p[0] - a[0], p[1] - a[1])
    best = min(key(p) for p in cands)
    return [p for p in cands if key(p) <= best + 1e-9]


def edge_cost(info, P, legged):
    status, stairs, allowed = info
    if status == "closed" or not allowed:
        return INF
    if stairs and not legged:
        return INF
    c = 1.0
    if status == "crowded":
        c += P.get("crowd", 0.0)
    elif status == "covered":
        c += P.get("cover", 0.0)
    else:
        c += P.get("normal", 0.0)
    if stairs:
        c += P.get("stairs", 0.0)
    return c


def turn_cost(h, d, P):
    r = REL[h][d]
    if r == "S":
        return 0.0
    if r == "R":
        return P.get("turnR", P.get("turn", 0.0))
    if r == "L":
        return P.get("turnL", P.get("turn", 0.0))
    return P.get("turnB", P.get("turn", 0.0))


def cost_to_go(world, terminal, P, legged):
    """Dijkstra ngược. terminal: {(node, h): giá trị}. Trả về V[(node, h)] = chi phí nhỏ nhất tới đích."""
    adj = world["adj"]
    V = dict(terminal)
    heap = [(v, n, h) for (n, h), v in terminal.items()]
    heapq.heapify(heap)
    while heap:
        v, n2, d = heapq.heappop(heap)
        if v > V.get((n2, d), INF):
            continue
        # trạng thái (n2, d) nghĩa là vừa tới n2 theo hướng d -> giao lộ trước đó là n
        n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
        info = adj.get(n, {}).get(d)
        if info is None:
            continue
        w = edge_cost(info, P, legged)
        if w == INF:
            continue
        for h in range(4):
            nv = v + w + turn_cost(h, d, P)
            if nv < V.get((n, h), INF) - 1e-12:
                V[(n, h)] = nv
                heapq.heappush(heap, (nv, n, h))
    return V


def path_q(world, legs, P, legged):
    """Q[d] = chi phí tốt nhất nếu bước đầu đi hướng d. legs = [ứng viên via], [ứng viên goal]."""
    term = {(n, h): 0.0 for n in legs[-1] for h in range(4)}
    V = cost_to_go(world, term, P, legged)
    for cands in reversed(legs[:-1]):
        term = {(n, h): V[(n, h)] for n in cands for h in range(4) if (n, h) in V}
        V = cost_to_go(world, term, P, legged)
    s, h0 = world["robot"], world["heading"]
    q = [INF] * 4
    for d, info in world["adj"].get(s, {}).items():
        w = edge_cost(info, P, legged)
        if w == INF:
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        v = V.get((n2, d), INF)
        q[d] = w + turn_cost(h0, d, P) * P.get("first_turn", 1.0) + v
    return q


def pick(q, heading, order="SRLB", eps=1e-6):
    """Chọn hướng có Q nhỏ nhất; hòa thì theo thứ tự ưu tiên so với hướng mũi robot."""
    best = min(q)
    if best == INF:
        # không có đường: chọn theo thứ tự ưu tiên
        return min(range(4), key=lambda d: order.index(REL[heading][d]))
    ties = [d for d in range(4) if q[d] <= best + eps]
    return min(ties, key=lambda d: order.index(REL[heading][d]))


# Tham số suy ra từ nhãn train (xem GIAI_THICH.md). Chi phí một đoạn = 1 + phụ phí.
#   crowd/cover/normal: phụ phí theo trạng thái; stairs: phụ phí bậc thang (chỉ robot 4 đi được)
#   turn (turnR/turnL): phạt rẽ 90 độ; turnB: phạt quay đầu
HOPS = {}
PARAMS = {
    0: lambda w, m: HOPS,
    1: lambda w, m: {"crowd": 5},
    2: lambda w, m: {"cover": -0.5},
    3: lambda w, m: {"cover": -0.8} if w["rain"] else {"crowd": 2},
    4: lambda w, m: {"crowd": 1, "stairs": 4} if w["rain"] else {"crowd": 1},
    5: lambda w, m: {"turn": 3, "turnB": 30},
    6: lambda w, m: HOPS if m["urgent"] else {"crowd": 4, "cover": -0.3},
    7: lambda w, m: {"crowd": 6, "turn": 1.5, "turnB": 15} if m["fragile"] else HOPS,
    8: lambda w, m: {"turnR": 0.5, "turnL": 4.5, "turnB": 9},
}
GREEDY = {"crowd": 0.5}
TIE_ORDER = "SRLB"   # hòa: đi thẳng > rẽ phải > rẽ trái > quay đầu (so với hướng mũi robot)


def _man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_q(world, legs):
    """Robot 9: chọn một đích gần nhất theo Manhattan (hòa -> địa điểm đứng trước theo hàng, cột),
    rồi chọn bước làm khoảng cách Manhattan còn lại nhỏ nhất; hòa thì tránh đường đông."""
    s = world["robot"]
    targets = sorted(legs[0])
    q = [INF] * 4
    if not targets:
        return q
    t = min(targets, key=lambda t: _man(s, t))
    for d, info in world["adj"].get(s, {}).items():
        c = edge_cost(info, GREEDY, False)
        if c == INF:
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        q[d] = _man(n2, t) + c - 1
    return q


def q_values(world, mission, robot_id):
    legs = legs_for(world, mission)
    if not legs[-1]:
        return [INF] * 4
    if robot_id == 9:
        return greedy_q(world, legs)
    return path_q(world, legs, PARAMS[robot_id](world, mission), robot_id == 4)


def legal_moves(world, robot_id):
    s = world["robot"]
    return [d for d, info in world["adj"].get(s, {}).items() if edge_cost(info, HOPS, robot_id == 4) < INF]


def predict(world, mission, robot_id):
    """Hướng đi (0=UP,1=DOWN,2=LEFT,3=RIGHT) của robot_id."""
    q = q_values(world, mission, robot_id)
    h = world["heading"]
    if min(q) == INF:
        # không tìm được đường (đồ thị đọc sai...): chọn một bước hợp lệ theo thứ tự ưu tiên
        legal = legal_moves(world, robot_id) or list(world["adj"].get(world["robot"], {})) or [0, 1, 2, 3]
        return min(legal, key=lambda d: TIE_ORDER.index(REL[h][d]))
    return pick(q, h, TIE_ORDER)


def legs_for(world, mission):
    legs = []
    if mission.get("via"):
        c = resolve_targets(world, mission["via"], mission.get("via_ref"))
        if c:
            legs.append(c)
    legs.append(resolve_targets(world, mission["goal"], mission.get("goal_ref")))
    return legs
