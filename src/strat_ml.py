"""(vòng private test) Chiến thuật 10 robot HỌC TỪ NHÃN train mới.

Đề vòng private đổi chiến thuật ngầm (trọng số, điều kiện, phá hòa riêng từng robot). Phân tích trên train (scratch/p2_*):
  - robot 0 luôn đi trên đường ít đoạn nhất (2000/2000 cảnh), hòa thì tránh đi XUYÊN giao lộ có địa điểm khác
    (231/231 ca; trừ đích / điểm ghé của nhiệm vụ), rồi ít đường đông...;
  - robot 9 chấm từng bước theo Manhattan tới điểm đến + phạt đông / phạt vào địa điểm;
  - các robot còn lại phụ thuộc nhiều yếu tố (đông, mái che, địa điểm đi ngang, rẽ, mưa, gấp, dễ vỡ).
Vì vậy mỗi robot dùng một bộ phân loại HistGradientBoosting trên các BƯỚC ĐI khả dĩ. Đặc trưng của một bước:
  - chi phí tới đích (Dijkstra trên (giao lộ, hướng)) dưới một dải giả thuyết chi phí: hiệu Q(d) - min Q và cờ tối ưu;
  - tham lam: thay đổi khoảng cách Manhattan / Euclid tới điểm đến kế tiếp;
  - bước đầu: hướng tuyệt đối, quan hệ với hướng mũi, trạng thái đoạn đầu, bậc thang, có vào giao lộ có địa điểm không;
  - cảnh: mưa, gấp, dễ vỡ, có điểm ghé, số địa điểm, hướng mũi.
Học trên train (+ validation cho bản cuối) bằng bản đồ + nhiệm vụ ĐÚNG từ scenes.json; khi dự đoán dùng bản đồ CV + nhiệm vụ
đọc từ câu.
"""
import itertools
import math
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from common import DRC, OPP, rel_turn

INF = float("inf")
REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]
RELI = {"S": 0, "R": 1, "L": 2, "B": 3}


def edge_ok(info, legged):
    status, stairs, allowed = info
    return not (status in ("closed", "missing") or not allowed or (stairs and not legged))


def lm_excl(world, legs):
    """Giao lộ có địa điểm, trừ các điểm đến của nhiệm vụ (đích / điểm ghé)."""
    return frozenset({p for v in world["landmarks"].values() for p in v} - {p for L in legs for p in L})


class SG:
    """Đồ thị trạng thái (giao lộ, hướng vừa đi) của một cảnh; trọng số = đặc trưng · theta.
    Đặc trưng một bước chuyển: [1, đông, mái che, thường, bậc thang, vào địa điểm, rẽ phải, rẽ trái, quay đầu]."""

    def __init__(self, w, legged, lmset=frozenset()):
        nodes = sorted(set(w["adj"]) | {w["robot"]} | {p for v in w["landmarks"].values() for p in v})
        self.idx = {n: i for i, n in enumerate(nodes)}
        self.N = len(nodes)
        fr, to, F = [], [], []
        for n, dd in w["adj"].items():
            for d, info in dd.items():
                if not edge_ok(info, legged):
                    continue
                n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                if n2 not in self.idx:
                    continue
                st = info[0]
                base = [1.0, st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmset]
                for h in range(4):
                    t = [0.0, 0.0, 0.0]
                    k = RELI[REL[h][d]]
                    if k:
                        t[k - 1] = 1.0
                    fr.append(self.idx[n2] * 4 + d); to.append(self.idx[n] * 4 + h); F.append(base + t)
        self.fr = np.array(fr, dtype=np.int32); self.to = np.array(to, dtype=np.int32)
        self.F = np.array(F, dtype=np.float64).reshape(-1, 9)
        s = w["robot"]; h0 = w["heading"]; self.first = []
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, legged):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if n2 not in self.idx:
                continue
            st = info[0]
            t = [0.0, 0.0, 0.0]
            k = RELI[REL[h0][d]]
            if k:
                t[k - 1] = 1.0
            self.first.append((d, self.idx[n2] * 4 + d,
                               np.array([1.0, st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmset], dtype=float),
                               np.array(t)))

    def ctg(self, theta, targets, init=None):
        S = self.N * 4
        wts = np.maximum(self.F @ theta, 1e-6)
        src, dst, ww = [], [], []
        for t in targets:
            if t not in self.idx:
                continue
            for h in range(4):
                v = 0.0 if init is None else init.get((t, h), INF)
                if v < INF:
                    src.append(S); dst.append(self.idx[t] * 4 + h); ww.append(v + 1.0)
        if not src:
            return None
        M = csr_matrix((np.concatenate([wts, ww]), (np.concatenate([self.fr, src]), np.concatenate([self.to, dst]))), shape=(S + 1, S + 1))
        return dijkstra(M, indices=S) - 1.0

    def q(self, theta, legs):
        V = self.ctg(theta, legs[-1])
        if V is None:
            return [INF] * 4
        for cands in reversed(legs[:-1]):
            init = {}
            for t in cands:
                if t in self.idx:
                    for h in range(4):
                        v = V[self.idx[t] * 4 + h]
                        if np.isfinite(v):
                            init[(t, h)] = v
            V = self.ctg(theta, cands, init)
            if V is None:
                return [INF] * 4
        q = [INF] * 4
        for d, si, base, t in self.first:
            v = V[si]
            if np.isfinite(v):
                q[d] = float(base @ theta[:6] + t @ theta[6:] + v)
        return q


def theta_of(crowd=0.0, cover=0.0, normal=0.0, stairs=0.0, lm=0.0, tR=0.0, tL=0.0, tB=0.0, turn=None):
    if turn is not None:
        tR = tL = turn
        tB = tB or turn
    return np.array([1.0, crowd, cover, normal, stairs, lm, tR, tL, tB])


HYP = [dict(crowd=cr, cover=cv, lm=lm, tR=tu, tL=tu, tB=2 * tu)
       for cr, cv, lm, tu in itertools.product([0, 1, 3], [-0.5, 0, 0.5], [0, 0.5, 3], [0, 0.5, 2])]
HYP += [dict(tR=0.5, tL=3, tB=6), dict(tR=3, tL=0.5, tB=6), dict(tR=0.5, tL=4.5, tB=9), dict(tR=4.5, tL=0.5, tB=9),
        dict(crowd=10), dict(crowd=10, lm=3), dict(cover=-0.8), dict(cover=-0.8, lm=3), dict(stairs=4), dict(stairs=-0.5),
        dict(crowd=1, stairs=4), dict(turn=3, tB=30), dict(turn=1.5, tB=15, crowd=6)]
THETAS = [theta_of(**h) for h in HYP]


def _cap(v):
    return 30.0 if not np.isfinite(v) else min(v, 30.0)


def move_feats(w, legs, urgent, fragile, has_via):
    """{legged: (mảng (4, F), mặt nạ bước hợp lệ (4,))}."""
    s, h0 = w["robot"], w["heading"]
    L = lm_excl(w, legs)
    out = {}
    nlm = len([p for v in w["landmarks"].values() for p in v])
    for legged in (False, True):
        g = SG(w, legged, L)
        Q = np.array([[_cap(v) for v in g.q(th, legs)] for th in THETAS])
        dQ = np.minimum(Q - Q.min(1, keepdims=True), 30.0)
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
            F.append(list(dQ[:, d]) + list(opt[:, d]) + [
                man - man0, euc, float(ok), d == 0, d == 1, d == 2, d == 3,
                rel == "S", rel == "R", rel == "L", rel == "B",
                st == "normal", st == "crowded", st == "covered", bool(info and info[1]),
                n2 in L, len(w["adj"].get(n2, {})),
                w["rain"], urgent, fragile, has_via, nlm, h0])
        out[legged] = (np.array(F, dtype=float), valid)
    return out


def is_night(img_path):
    """Kiểu vẽ "night" = BAN ĐÊM (robot đổi cách đi). Độ sáng trung bình ảnh: đêm <= 49, ngày >= 221 trên toàn bộ
    train + validation (sai 0 / 700 ảnh với ngưỡng 128)."""
    from PIL import Image
    return float(np.asarray(Image.open(img_path).convert("L").resize((64, 64))).mean()) < 128


def group_key(key, night, rain, urgent, fragile, via=False, goal=None):
    """Tên nhóm điều kiện, cùng cách viết với scratch/p2_cd3.py (nơi tham số được dò). goal = LOẠI nơi giao: điều kiện ẩn của
    robot 2, 3, 7 (scratch/p2_cond.py tự dò: chia theo loại nơi giao tăng validation R2 0,78 -> 0,90, R3 0,66 -> 0,86)."""
    nd = "đêm" if night else "ngày"
    w = "rain" if rain else "dry"
    u = "/gấp=" + str(bool(urgent)); f = "/dễ vỡ=" + str(bool(fragile)); v = "ghé=" + str(bool(via))
    if key.startswith("gt"):
        g = str(goal)
        return {"gt": g, "gt_w": g + "/" + w, "gt_nd": g + "/" + nd, "gt_f": g + f, "gt_u": g + u, "gt_v": g + "/" + v}[key]
    return {"nd": nd, "nd_w": nd + "/" + w, "nd_u": nd + u, "nd_f": nd + f, "none": "all", "w": w, "v": v,
            "nd_v": nd + "/" + v, "nd_v_u": nd + "/" + v + u, "nd_v_f": nd + "/" + v + f, "nd_v_w": nd + "/" + v + "/" + w,
            "nd_w_u": nd + "/" + w + u, "nd_w_f": nd + "/" + w + f, "nd_u_f": nd + u + f}[key]


PLACE_ORDER = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]


def ml4_row(w, legs, r, qcfg, f, night, urgent, fragile):
    """Đặc trưng "ml4" của 4 bước đi: đặc trưng bước (f = move_feats) + cờ đêm + chi phí mô hình lai của robot r (hiệu so với
    min, cờ tối ưu) + loại nơi giao one-hot. Trả về (list 4 vector, mặt nạ hợp lệ)."""
    F, valid = f[r == 4]
    goal = next((t for t, v in w["landmarks"].items() if legs[-1] and tuple(legs[-1][0]) in [tuple(q) for q in v]), None)
    q = np.zeros(4)
    p = qcfg["params"].get(group_key(qcfg["key"], night, w["rain"], urgent, fragile, len(legs) == 2, goal)) if qcfg else None
    if p is not None:
        qq = np.array(SG(w, r == 4, lm_excl(w, legs)).q(theta_of(**p), legs))
        qq = np.where(np.isfinite(qq), qq, 99.0)
        q = np.minimum(qq - qq.min(), 30)
    g = np.array([goal == t for t in PLACE_ORDER], float)
    return [np.concatenate([F[d], [night, q[d], float(q[d] < 1e-6)], g]) for d in range(4)], valid


class StrategyML:
    """Chiến thuật LAI (vòng private): robot có hàm chi phí tìm được theo điều kiện (đêm / ngày × mưa / gấp / dễ vỡ, scratch/p2_cd3.py)
    dùng Dijkstra với tham số đó (hòa: thẳng > phải > trái > quay đầu); robot còn lại dùng bộ phân loại học từ nhãn."""

    def __init__(self, models, hybrid=None):
        self.models = models          # robot -> bộ phân loại (xác suất "đây là bước của robot")
        self.hybrid = hybrid or {}    # robot (str) -> {"method": "cost", "key": ..., "params": {nhóm: tham số}}

    def _cost_move(self, w, legs, r, cfg, night, urgent, fragile):
        goal = next((t for t, v in w["landmarks"].items() if legs[-1] and tuple(legs[-1][0]) in [tuple(q) for q in v]), None)
        p = cfg["params"].get(group_key(cfg["key"], night, w["rain"], urgent, fragile, len(legs) == 2, goal))
        if p is None:
            return None
        g = SG(w, r == 4, lm_excl(w, legs))
        q = g.q(theta_of(**p), legs)
        b = min(q)
        if not np.isfinite(b):
            return None
        a = [d for d in range(4) if q[d] <= b + 1e-6]
        kind, o = cfg.get("order", ["rel", "SRLB"])     # thứ tự phá hòa riêng của robot (scratch/p2_tie2.py)
        return min(a, key=lambda d: o.index(REL[w["heading"]][d])) if kind == "rel" else min(a, key=lambda d: list(o).index(d))

    def _tree_move(self, w, legs, r, cfg, night, urgent, fragile, mapgoal=False):
        """Cây điều kiện (scratch/p2_tree.py): đi xuống theo các điều kiện của cảnh tới lá, rồi tìm đường với tham số của lá."""
        goal = next((t for t, v in w["landmarks"].items() if legs[-1] and tuple(legs[-1][0]) in [tuple(q) for q in v]), None)
        rows = max(n[0] for n in w["adj"]) + 1 if w["adj"] else 0
        feat = {"mưa": w["rain"], "đêm": night, "gấp": urgent, "dễ vỡ": fragile, "ghé": len(legs) == 2, "đích bản đồ": mapgoal,
                "lưới>=7 hàng": rows >= 7, "mũi dọc": w["heading"] in (0, 1)}
        node = cfg["tree"]
        while "split" in node:
            n = node["split"]
            if n == "loại đích":
                k = str(goal)
            elif n == "nhóm đích3":
                k = {"library": "a", "lecture": "a", "office": "a", "canteen": "b", "dorm": "b"}.get(goal, "c")
            else:
                k = str(bool(feat[n]))
            node = node["kids"].get(k) or next(iter(node["kids"].values()))
        p = node["leaf"]
        q = SG(w, r == 4, lm_excl(w, legs)).q(theta_of(**p), legs)
        b = min(q)
        if not np.isfinite(b):
            return None
        a = [d for d in range(4) if q[d] <= b + 1e-6]
        return min(a, key=lambda d: "SRLB".index(REL[w["heading"]][d]))

    @staticmethod
    def _lex_move(w, legs, r, cfg, night, urgent, fragile):
        """Chuỗi tiêu chí theo thứ tự TỪ ĐIỂN (scratch/p2_lex.py), mã hóa thành một vector trọng số bậc thang (1e6, 1e4, 1e2, 1).
        Robot 0: mưa -> (số đoạn + địa điểm đi ngang, ít đoạn không mái che, ít đông), khô -> (số đoạn + địa điểm, ít đông,
        ít quay đầu); hòa cuối theo thứ tự tương đối. Validation (thông tin đúng) 0,987."""
        p = cfg["params"].get(group_key(cfg["key"], night, w["rain"], urgent, fragile, len(legs) == 2))
        if p is None:
            return None
        q = SG(w, r == 4, lm_excl(w, legs)).q(np.array(p["theta"]), legs)
        b = min(q)
        if not np.isfinite(b):
            return None
        a = [d for d in range(4) if q[d] <= b + 1e-3]
        kind, o = p["order"]
        return min(a, key=lambda d: o.index(REL[w["heading"]][d])) if kind == "rel" else min(a, key=lambda d: list(o).index(d))

    @staticmethod
    def _greedy_move(w, legs, cfg, night, urgent, fragile):
        """Robot 9 (tham lam, scratch/p2_a37_r9dist.py): điểm bước = khoảng cách (0 Manhattan / 2 Euclid trên lưới) từ giao lộ
        kế tới điểm đến gần nhất + a*đông + b*vào địa điểm + c*mái che + t*rẽ; hòa theo thứ tự tương đối o.
        Nhóm: đêm/ngày + dễ vỡ + gấp."""
        g = ("đêm" if night else "ngày") + str(bool(fragile)) + str(bool(urgent))
        p = cfg["params"].get(g)
        if p is None:
            return None
        di, a, b, c, t, o = p
        s = w["robot"]; tg = legs[0]; h = w["heading"]
        L = lm_excl(w, legs)
        sc = {}
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, False):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            dist = min((abs(n2[0] - q[0]) + abs(n2[1] - q[1])) if di == 0 else math.hypot(n2[0] - q[0], n2[1] - q[1]) for q in tg)
            sc[d] = dist + a * (info[0] == "crowded") + b * (n2 in L) + c * (info[0] == "covered") + (0 if REL[h][d] == "S" else t)
        if not sc:
            return None
        mn = min(sc.values())
        A = [d for d, v in sc.items() if v <= mn + 1e-9]
        return min(A, key=lambda d: o.index(REL[h][d]))

    @staticmethod
    def _greedy2_move(w, legs, cfg, night, urgent, fragile):
        """Robot 9 (scratch/p2_greedy_exact_fix.py): điểm bước = khoảng cách từ giao lộ kế tới điểm đến gần nhất (Manhattan khi
        gấp, Euclid lưới khi không gấp — theo tham số nhóm) + a*đông + c*mái che + b*vào địa điểm + bt[loại] + phạt rẽ R/L/B;
        HÒA -> thứ tự hướng TUYỆT ĐỐI (A0312 = LÊN, PHẢI, XUỐNG, TRÁI). Nhóm: (đêm, gấp, dễ vỡ)."""
        p = cfg["params"].get(str((int(bool(night)), int(bool(urgent)), int(bool(fragile)))))
        if p is None or not legs or not legs[0]:
            return None
        s = w["robot"]; tg = legs[0]; h = w["heading"]
        tgs = {q for L in legs for q in L}
        lmt = {q: PLACE_ORDER.index(t) for t, v in w["landmarks"].items() for q in v if q not in tgs}
        sc = {}
        for d, info in w["adj"].get(s, {}).items():
            if not edge_ok(info, False):
                continue
            n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
            if p["dist"] == 0:
                dist = min(abs(n2[0] - q[0]) + abs(n2[1] - q[1]) for q in tg)
            else:
                dist = min(math.hypot(n2[0] - q[0], n2[1] - q[1]) for q in tg)
            rel = REL[h][d]
            sc[d] = dist + p["a"] * (info[0] == "crowded") + p["c"] * (info[0] == "covered") + \
                (p["b"] + p["bt"][lmt[n2]] if n2 in lmt else 0.0) + {"S": 0.0, "R": p["R"], "L": p["L"], "B": p["B"]}[rel]
        if not sc:
            return None
        o = p["order"]
        if o.startswith("A"):
            ab = [int(c) for c in o[1:]]
            return min(sc, key=lambda d: (round(sc[d], 9), ab.index(d)))
        return min(sc, key=lambda d: (round(sc[d], 9), o.index(REL[h][d])))

    def predict_scene(self, w, legs, urgent, fragile, has_via, night=False, mapgoal=False):
        """10 hướng đi cho một cảnh."""
        f = move_feats(w, legs, urgent, fragile, has_via)
        out = []
        for r in range(10):
            cfg = self.hybrid.get(str(r))
            if cfg and cfg.get("method") == "greedy2":
                d = self._greedy2_move(w, legs, cfg, night, urgent, fragile)
                if d is not None:
                    out.append(int(d))
                    continue
                cfg = cfg.get("fallback") or cfg
            if cfg and cfg.get("method") in ("cost", "greedy", "lex", "tree"):
                fn = {"cost": self._cost_move, "lex": self._lex_move, "tree": self._tree_move,
                      "greedy": lambda *a: self._greedy_move(*a[:2], *a[3:])}[cfg["method"]]
                d = fn(w, legs, r, cfg, night, urgent, fragile) if cfg["method"] != "tree" else                     self._tree_move(w, legs, r, cfg, night, urgent, fragile, mapgoal)
                if d is not None:
                    out.append(int(d))
                    continue
            if cfg and cfg.get("method") == "hv":
                # GỘP: trung bình log-xác suất các bộ siêu tuyến tính (cfg["sets"]) và bộ tìm đường vin, trọng số cfg["a"]
                import hyper as _hy, vin as _vin
                hs = getattr(self, "hypers", {})
                L = [_hy.logp(hs[t], r, w, legs, night, urgent, fragile, mapgoal) for t in cfg.get("sets", [""]) if t in hs]
                L = [x for x in L if x is not None]
                lv = None
                if getattr(self, "vin", None) and r in self.vin and legs and legs[-1]:
                    lv = sum(_vin.logprobs(W, w, legs, r == 4, night, urgent, fragile, mapgoal) for W in self.vin[r]) / len(self.vin[r])
                if L or lv is not None:
                    a = cfg.get("a", 0.67) if (L and lv is not None) else (1.0 if L else 0.0)
                    sc = a * (sum(L) / len(L) if L else 0) + (1 - a) * (lv if lv is not None else 0)
                    ok = [d for d, info in w["adj"].get(w["robot"], {}).items() if _hy.ok_edge(info, r == 4)]
                    if ok:
                        out.append(int(max(ok, key=lambda d: sc[d] if np.isfinite(sc[d]) else -1e18)))
                        continue
                cfg = cfg.get("fallback") or cfg
            if cfg and cfg.get("method") == "hyper" and getattr(self, "hyper", None) and r in self.hyper:
                # mô hình siêu tuyến tính trên mặt Pareto (src/hyper.py, scratch/p2_hyper.py)
                import hyper as _hy
                d = _hy.predict(self.hyper, r, w, legs, night, urgent, fragile, mapgoal)
                if d is not None:
                    out.append(int(d))
                    continue
                cfg = cfg.get("fallback") or cfg
            if cfg and cfg.get("method") == "vin" and getattr(self, "vin", None) and r in self.vin:
                # bộ tìm đường có hàm chi phí học được (src/vin.py, scratch/p2_vin2.py)
                import vin as _vin
                d = _vin.predict(self.vin, r, w, legs, night, urgent, fragile, mapgoal)
                if d is not None:
                    out.append(int(d))
                    continue
            if cfg and cfg.get("method") == "ml4" and getattr(self, "ml4", None) and r in self.ml4:
                # bộ phân loại học trên đặc trưng bước + đêm + chi phí mô hình lai + loại nơi giao (scratch/p2_ml4_train.py)
                rows, valid = ml4_row(w, legs, r, cfg.get("qcfg"), f, night, urgent, fragile)
                idx = [d for d in range(4) if valid[d]]
                if idx:
                    p = self.ml4[r].predict_proba(np.array([rows[d] for d in idx]))[:, 1]
                    out.append(int(idx[int(np.argmax(p))]))
                    continue
            F, valid = f[r == 4]
            idx = [d for d in range(4) if valid[d]]
            if not idx:
                idx = [d for d in range(4) if d in w["adj"].get(w["robot"], {})] or [0]
                out.append(idx[0])
                continue
            p = self.models[r].predict_proba(F[idx])[:, 1]
            out.append(int(idx[int(np.argmax(p))]))
        return out

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))
