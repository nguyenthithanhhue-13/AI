"""Đặc trưng cho từng BƯỚC ĐI khả dĩ (4 hướng) của một cảnh, dùng cho bộ học chiến thuật (vòng private).
Nhóm đặc trưng:
  - Q dưới một dải giả thuyết chi phí (số đoạn = 1 + phụ phí đông / mái che / địa điểm đi ngang / rẽ / bậc thang): hiệu Q(d) - min Q
    và cờ "tối ưu" (hiệu < 1e-6). Robot thường và robot có chân (đi được bậc thang) tính riêng.
  - tham lam (robot 9): khoảng cách Manhattan / Euclid tới điểm đến gần nhất sau bước đi.
  - bước đầu: hướng tuyệt đối, quan hệ với hướng mũi (thẳng / phải / trái / quay đầu), trạng thái đoạn đầu, bậc thang,
    giao lộ kế có địa điểm (trừ đích / ghé) không, bậc của giao lộ kế.
  - cảnh: mưa, gấp, dễ vỡ, có điểm ghé, số địa điểm.
    feats(x) -> mảng (4, F) và mặt nạ hợp lệ (4,) cho robot thường và robot có chân."""
import sys, itertools, math, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import SG, theta_of, lm_excl, argmins, INF, DRC, REL, ACTIONS, edge_ok

HYP = []
for cr, cv, lm, tu in itertools.product([0, 1, 3], [-0.5, 0, 0.5], [0, 0.5, 3], [0, 0.5, 2]):
    HYP.append(dict(crowd=cr, cover=cv, lm=lm, tR=tu, tL=tu, tB=2 * tu))
HYP += [dict(tR=0.5, tL=3, tB=6), dict(tR=3, tL=0.5, tB=6), dict(tR=0.5, tL=4.5, tB=9), dict(tR=4.5, tL=0.5, tB=9),
        dict(crowd=10), dict(crowd=10, lm=3), dict(cover=-0.8), dict(cover=-0.8, lm=3), dict(stairs=4), dict(stairs=-0.5),
        dict(crowd=1, stairs=4), dict(turn=3, tB=30), dict(turn=1.5, tB=15, crowd=6)]
THETAS = [theta_of(**h) for h in HYP]
NH = len(HYP)


def _cap(v):
    return 30.0 if not np.isfinite(v) else min(v, 30.0)


def feats(x):
    w = x["w"]; legs = x["legs"]; m = x["m"]
    s, h0 = w["robot"], w["heading"]
    L = lm_excl(x)
    out = {}
    for legged in (False, True):
        g = SG(w, legged, L)
        Q = np.array([[_cap(v) for v in g.q(th, legs)] for th in THETAS])           # (NH, 4)
        mn = Q.min(1, keepdims=True)
        dQ = np.minimum(Q - mn, 30.0)
        opt = (dQ < 1e-6).astype(float)
        F = []
        valid = np.zeros(4, bool)
        tg = legs[0]
        for d in range(4):
            info = w["adj"].get(s, {}).get(d)
            ok = info is not None and edge_ok(info, legged)
            valid[d] = ok
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            man = min(abs(n2[0] - t[0]) + abs(n2[1] - t[1]) for t in tg)
            man0 = min(abs(s[0] - t[0]) + abs(s[1] - t[1]) for t in tg)
            euc = min(math.hypot(n2[0] - t[0], n2[1] - t[1]) for t in tg)
            rel = REL[h0][d]
            st = info[0] if info else "none"
            f = list(dQ[:, d]) + list(opt[:, d]) + [
                man - man0, euc, float(ok),
                d == 0, d == 1, d == 2, d == 3,
                rel == "S", rel == "R", rel == "L", rel == "B",
                st == "normal", st == "crowded", st == "covered", bool(info and info[1]),
                n2 in L, len(w["adj"].get(n2, {})),
                w["rain"], m["urgent"], m["fragile"], bool(m["via"]), len([p for v in w["landmarks"].values() for p in v]),
                h0]
            F.append(f)
        out[legged] = (np.array(F, dtype=float), valid)
    return out


FEAT_NAMES = [f"dQ{i}" for i in range(NH)] + [f"opt{i}" for i in range(NH)] + [
    "dman", "euc", "legal", "U", "D", "L", "R", "relS", "relR", "relL", "relB", "e_normal", "e_crowd", "e_cover", "e_stairs",
    "next_lm", "next_deg", "rain", "urgent", "fragile", "via", "n_lm", "heading"]
