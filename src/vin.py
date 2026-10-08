"""(vòng private) BỘ TÌM ĐƯỜNG CÓ HÀM CHI PHÍ HỌC ĐƯỢC — phần DỰ ĐOÁN bằng numpy (không cần torch).

Huấn luyện ở scratch/p2_vin2.py (torch): chi phí một bước chuyển (giao lộ, hướng vừa đi) -> giao lộ kế = MLP(đặc trưng bước,
điều kiện cảnh, cờ bước xuất phát); chi phí tới đích = lặp giá trị trên đồ thị trạng thái (2 chặng nếu có điểm ghé); bước đi =
argmin. Đặc trưng bước: [đông, mái che, thường, bậc thang, vào giao lộ có địa điểm (trừ đích / ghé), loại địa điểm đó (10),
rẽ S/R/L/B]. Điều kiện cảnh: [mưa, đêm, gấp, dễ vỡ, có điểm ghé, nơi giao mô tả qua bản đồ, loại nơi giao (10), loại điểm ghé (10)].
Mỗi robot một nhóm mạng (gộp nhiều hạt giống); trọng số ở outputs/vin_<mode>.npz.
"""
import numpy as np

from common import DRC, PLACES, rel_turn

REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]
NF = 19
K = 48
DEAD = 324


def edge_ok(info, legged):
    status, stairs, allowed = info
    return not (status in ("closed", "missing") or not allowed or (stairs and not legged))


def encode(w, legs, legged):
    tg = {p for L in legs for p in L}
    lt = {p: PLACES.index(t) for t, v in w["landmarks"].items() for p in v if p not in tg}
    nxt = np.full((325, 4), DEAD, np.int64)
    E = np.zeros((325, 4, NF), np.float32)
    for n, dd in w["adj"].items():
        if not (0 <= n[0] < 9 and 0 <= n[1] < 9):
            continue
        for d, info in dd.items():
            if not edge_ok(info, legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if not (0 <= n2[0] < 9 and 0 <= n2[1] < 9):
                continue
            st = info[0]
            base = [st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lt] + [lt.get(n2) == k for k in range(10)]
            for h in range(4):
                s = (n[0] * 9 + n[1]) * 4 + h
                nxt[s, d] = (n2[0] * 9 + n2[1]) * 4 + d
                rel = REL[h][d]
                E[s, d] = base + [rel == "S", rel == "R", rel == "L", rel == "B"]
    return nxt, E


def conditions(w, legs, night, urgent, fragile, mapgoal):
    def typ(p):
        return next((t for t, v in w["landmarks"].items() if tuple(p) in [tuple(q) for q in v]), None)
    goal = typ(legs[-1][0]) if legs[-1] else None
    via = typ(legs[0][0]) if len(legs) == 2 and legs[0] else None
    return np.array([w["rain"], night, urgent, fragile, len(legs) == 2, mapgoal] + [goal == t for t in PLACES] +
                    [via == t for t in PLACES], np.float32)


def _mlp(W, x):
    h = np.maximum(x @ W["w0"].T + W["b0"], 0)
    h = np.maximum(h @ W["w1"].T + W["b1"], 0)
    z = (h @ W["w2"].T + W["b2"])[..., 0]
    return np.logaddexp(0, z) + 0.05


def _values(c, nxt, term):
    V = term.copy()
    for _ in range(K):
        Vn = np.minimum(term, (c + V[nxt]).min(1))
        Vn[DEAD] = 1e3
        if np.array_equal(Vn, V):
            break
        V = Vn
    return V


def logprobs(W, w, legs, legged, night, urgent, fragile, mapgoal):
    """log xác suất 4 bước đầu của một mạng."""
    nxt, E = encode(w, legs, legged)
    C = conditions(w, legs, night, urgent, fragile, mapgoal)
    x = np.concatenate([E, np.broadcast_to(C, (325, 4, len(C))), np.zeros((325, 4, 1), np.float32)], -1)
    c = _mlp(W, x)
    term = np.full(325, 60.0); term[DEAD] = 1e3
    for p in legs[-1]:
        term[(p[0] * 9 + p[1]) * 4:(p[0] * 9 + p[1]) * 4 + 4] = 0.0
    V = _values(c, nxt, term)
    if len(legs) == 2:
        t1 = np.full(325, 60.0); t1[DEAD] = 1e3
        for p in legs[0]:
            s0 = (p[0] * 9 + p[1]) * 4
            t1[s0:s0 + 4] = V[s0:s0 + 4]
        V = _values(c, nxt, t1)
    st = (w["robot"][0] * 9 + w["robot"][1]) * 4 + w["heading"]
    xs = np.concatenate([E[st], np.broadcast_to(C, (4, len(C))), np.ones((4, 1), np.float32)], -1)
    q = _mlp(W, xs) + V[nxt[st]]
    q = np.where(nxt[st] == DEAD, 1e3, q)
    z = -q / (0.3 * np.exp(W["out_t"]))
    return z - np.logaddexp.reduce(z)


def predict(weights, r, w, legs, night, urgent, fragile, mapgoal):
    """weights: {robot: [dict trọng số của từng hạt giống]} -> hướng đi hoặc None."""
    nets = weights.get(r)
    if not nets or not legs or not legs[-1]:
        return None
    lp = sum(logprobs(W, w, legs, r == 4, night, urgent, fragile, mapgoal) for W in nets)
    s = w["robot"]
    ok = [d for d, info in w["adj"].get(s, {}).items() if edge_ok(info, r == 4)]
    if not ok:
        return None
    return int(max(ok, key=lambda d: lp[d]))


def load(path):
    z = np.load(path)
    out = {}
    for k in z.files:          # tên: r{robot}_s{seed}_{tham số}
        rr, ss, name = k.split("_", 2)
        out.setdefault(int(rr[1:]), {}).setdefault(int(ss[1:]), {})[name] = z[k]
    return {r: [d[s] for s in sorted(d)] for r, d in out.items()}
