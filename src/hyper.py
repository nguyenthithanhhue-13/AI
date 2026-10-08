"""(vòng private) Mô hình SIÊU TUYẾN TÍNH — phần DỰ ĐOÁN bằng numpy.

Chi phí một đường = w · [số đoạn, đông, mái che, xuyên địa điểm, bậc thang, rẽ phải, rẽ trái, quay đầu, đoạn không mái che,
xuyên từng loại địa điểm (10)] + phạt rẽ ở bước đầu (R, L, B); w = softplus(MLP(điều kiện cảnh)). Bước đi = argmin trên MẶT PARETO
các đường (số đoạn <= ngắn nhất + 5), hòa -> thứ tự kiểu rẽ bước đầu của robot. Huấn luyện: scratch/p2_hyper.py (torch), mặt Pareto:
scratch/p2_pareto.py (TYPED=1 UNC=1), xuất trọng số: scratch/p2_hyper_export.py -> outputs/hyper_<mode>.npz.
"""
import collections, heapq
import numpy as np

from common import DRC, PLACES, rel_turn

REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]
SLACK = 5
KINDS_MAP = ("anchor_near", "north_most", "south_most", "west_most", "east_most")


def ok_edge(info, legged):
    status, stairs, allowed = info
    return not (status in ("closed", "missing") or not allowed or (stairs and not legged))


def _dom(v, front):
    for u in front:
        if all(a <= b for a, b in zip(u, v)):
            return True
    return False


_CACHE = {}


def fronts(w, legs, legged):
    """{bước đầu d: (kiểu rẽ so với mũi robot, mảng [k, 19] đặc trưng các đường Pareto)} — có bộ nhớ đệm theo (bản đồ, chặng,
    có chân) vì mọi robot không chân / mọi bộ trọng số dùng chung một mặt Pareto."""
    key = (id(w), tuple(tuple(L) for L in legs), bool(legged))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is w:
        return hit[1]
    if len(_CACHE) > 64:
        _CACHE.clear()
    out = _fronts(w, legs, legged)
    _CACHE[key] = (w, out)
    return out


def _fronts(w, legs, legged):
    tg = {p for L in legs for p in L}
    lm = {p for v in w["landmarks"].values() for p in v} - tg
    lmt = {p: PLACES.index(t) for t, v in w["landmarks"].items() for p in v if p not in tg}

    def lv(n):
        return tuple(int(lmt.get(n) == k) for k in range(10))
    goal = set(legs[-1]); via = set(legs[0]) if len(legs) == 2 else None
    adj = w["adj"]
    radj = collections.defaultdict(list)
    for n, dd in adj.items():
        for d, info in dd.items():
            if ok_edge(info, legged):
                radj[(n[0] + DRC[d][0], n[1] + DRC[d][1])].append(n)

    def rbfs(tgts):
        D = {t: 0 for t in tgts}; q = collections.deque(tgts)
        while q:
            n = q.popleft()
            for m in radj.get(n, []):
                if m not in D:
                    D[m] = D[n] + 1; q.append(m)
        return D
    BIGD = 10 ** 6
    dg = rbfs(list(goal))
    if via:
        rem0 = {}
        for v in via:
            for n, k in rbfs([v]).items():
                rem0[n] = min(rem0.get(n, BIGD), k + dg.get(v, BIGD))
    else:
        rem0 = dg
    s0 = w["robot"]; h0 = w["heading"]
    if rem0.get(s0, BIGD) >= BIGD:
        return {}
    H = rem0[s0] + SLACK
    out = {}
    for d, info in adj.get(s0, {}).items():
        if not ok_edge(info, legged):
            continue
        n2 = (s0[0] + DRC[d][0], s0[1] + DRC[d][1])
        st = info[0]
        f0 = (1, st == "crowded", st == "covered", n2 in lm, bool(info[1]), 0, 0, 0, st != "covered") + lv(n2)
        ph = 1 if (via and n2 in via) else (0 if via else 1)
        labels = collections.defaultdict(list); res = []
        start = (n2, d, ph)
        heap = [(f0[0], f0, start)]; labels[start].append(f0)
        while heap:
            hop, f, (n, h, p) = heapq.heappop(heap)
            if f not in labels[(n, h, p)]:
                continue
            if p == 1 and n in goal:
                if not _dom(f, res):
                    res = [u for u in res if not all(a <= b for a, b in zip(f, u))] + [f]
                continue
            if hop + (dg if p == 1 else rem0).get(n, BIGD) > H or _dom(f, res):
                continue
            for d2, info2 in adj.get(n, {}).items():
                if not ok_edge(info2, legged):
                    continue
                m = (n[0] + DRC[d2][0], n[1] + DRC[d2][1])
                rel = REL[h][d2]; st2 = info2[0]
                p2 = 1 if (p == 1 or (via and m in via)) else 0
                g = (f[0] + 1, f[1] + (st2 == "crowded"), f[2] + (st2 == "covered"), f[3] + (m in lm), f[4] + bool(info2[1]),
                     f[5] + (rel == "R"), f[6] + (rel == "L"), f[7] + (rel == "B"), f[8] + (st2 != "covered")) + \
                    tuple(a + b for a, b in zip(f[9:], lv(m)))
                if g[0] + (dg if p2 == 1 else rem0).get(m, BIGD) > H:
                    continue
                key = (m, d2, p2); L = labels[key]
                if _dom(g, L):
                    continue
                labels[key] = [u for u in L if not all(a <= b for a, b in zip(g, u))] + [g]
                heapq.heappush(heap, (g[0], g, key))
        if res:
            out[d] = (REL[h0][d], np.array(res, np.float64))
    return out


def conditions(mode, w, legs, night, urgent, fragile, mapgoal):
    def typ(p):
        return next((t for t, v in w["landmarks"].items() if tuple(p) in [tuple(q) for q in v]), None)
    goal = typ(legs[-1][0]) if legs[-1] else None
    via = typ(legs[0][0]) if len(legs) == 2 and legs[0] else None
    base = [w["rain"], night, urgent, fragile, len(legs) == 2, mapgoal]
    TOK = ["rain", "night", "urg", "frag", "via", "mapg", "goal", "viat"]
    if "+" in mode or mode.startswith("g:") or (mode in TOK and mode != "goal"):          # danh sách điều kiện riêng của robot, vd "night+urg+goal"
        c = []
        for t in mode.split("+"):
            if t.startswith("g:"):        # đích thuộc nhóm loại
                c += [goal in t[2:].split(",")]
            else:
                c += [base[TOK.index(t)]] if TOK.index(t) < 6 else [(goal if t == "goal" else via) == q for q in PLACES]
        return np.array(c, np.float64)
    if mode == "4":
        c = base[:4]
    elif mode == "basic":
        c = base
    elif mode == "goal":
        c = base + [goal == t for t in PLACES]
    else:
        c = base + [goal == t for t in PLACES] + [via == t for t in PLACES]
    return np.array(c, np.float64)


def _weights(W, c):
    h = np.maximum(W["w0"] @ c + W["b0"], 0)
    h = np.maximum(W["w1"] @ h + W["b1"], 0)
    return np.logaddexp(0, W["w2"] @ h + W["b2"])


def logp(models, r, w, legs, night, urgent, fragile, mapgoal):
    """log xác suất 4 hướng (trung bình các hạt giống, -inf = không đi được) hoặc None."""
    M = models.get(r)
    if not M or not legs or not legs[-1]:
        return None
    fr = fronts(w, legs, r == 4)
    if not fr:
        return None
    c = conditions(M["cond"], w, legs, night, urgent, fragile, mapgoal)
    out = np.zeros(4)
    for W in M["nets"]:
        wt = _weights(W, c)
        z = np.full(4, -np.inf)
        for d, (rel, F) in fr.items():
            first = {"S": 0.0, "R": wt[19], "L": wt[20], "B": wt[21]}[rel]
            z[d] = -(float((F @ wt[:19]).min()) + first) / (0.2 * np.exp(float(W.get("t", 0.0))))
        out += z - np.logaddexp.reduce(z[np.isfinite(z)])
    return out / len(M["nets"])


def predict(models, r, w, legs, night, urgent, fragile, mapgoal):
    """models: {robot: dict(cond=..., order=..., nets=[trọng số])} -> hướng đi hoặc None."""
    M = models.get(r)
    if not M or not legs or not legs[-1]:
        return None
    fr = fronts(w, legs, r == 4)
    if not fr:
        return None
    c = conditions(M["cond"], w, legs, night, urgent, fragile, mapgoal)
    q = {}
    for d, (rel, F) in fr.items():
        v = 0.0
        for W in M["nets"]:
            wt = _weights(W, c)
            first = {"S": 0.0, "R": wt[19], "L": wt[20], "B": wt[21]}[rel]
            v += float((F @ wt[:19]).min()) + first
        q[d] = v / len(M["nets"]) + M["order"].index(rel) * 1e-5
    return int(min(q, key=q.get))


def load(path):
    z = np.load(path, allow_pickle=True)
    meta = z["meta"].item()
    out = {}
    for r, (cond, order, nseed) in meta.items():
        out[int(r)] = dict(cond=cond, order=order, nets=[{k: z[f"r{r}_s{s}_{k}"] for k in ("w0", "b0", "w1", "b1", "w2", "b2", "t")
                                                          if f"r{r}_s{s}_{k}" in z.files} for s in range(nseed)])
    return out
