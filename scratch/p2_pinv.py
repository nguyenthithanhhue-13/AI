"""DÒ NGƯỢC trọng số chi phí trên MẶT PARETO (scratch/p2_pareto.py): chi phí đường = w · [số đoạn, đông, mái che, xuyên địa điểm,
bậc thang, rẽ phải, rẽ trái, quay đầu] + phạt rẽ ở bước đầu (wR0, wL0, wB0); bước đi = argmin, hòa -> thứ tự S/R/L/B riêng.
Chia cảnh theo nhóm điều kiện (KEY); mỗi nhóm dò ngẫu nhiên + tinh chỉnh tọa độ trên train, đo validation (thông tin đúng).
    python scratch/p2_pinv.py <robot> <key> [n_random]
key: tổ hợp các chữ: n(đêm) w(mưa) u(gấp) f(dễ vỡ) v(có ghé) g(loại đích) m(đích mô tả bản đồ) -, ví dụ "nu", "gw", "-" """
import sys, pickle, itertools, collections, time, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]); KEY = sys.argv[2]; NR = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
leg = int(r == 4)
RELS = "SRLB"


def prep(split):
    D = load(split)
    P = pickle.load(open(f"cache/p2_pareto_{split}_{leg}.pkl", "rb"))
    rows = []      # mỗi cảnh: list (d, rel0, F)
    for x, fr in zip(D, P):
        rows.append([(d, rel, F.astype(np.float64)) for d, (rel, F) in sorted(fr.items())])
    return D, rows


def gkey(x):
    s = x["s"]; m = x["m"]; k = []
    for ch in KEY:
        if ch == "n": k.append("đêm" if s["style"] == "night" else "ngày")
        if ch == "w": k.append("mưa" if x["w"]["rain"] else "khô")
        if ch == "u": k.append("gấp" if m["urgent"] else "-")
        if ch == "f": k.append("vỡ" if m["fragile"] else "-")
        if ch == "v": k.append("ghé" if m["via"] else "-")
        if ch == "g": k.append(str(next((t for t, v in x["w"]["landmarks"].items() if tuple(x["legs"][-1][0]) in v), None)))
        if ch == "m": k.append("bđ" if (m["goal_ref"] or {}).get("kind") in ("anchor_near", "north_most", "south_most", "west_most", "east_most") else "-")
    return "/".join(k) or "-"


class Batch:
    """Gói các cảnh để tính nhanh: ma trận F gộp, chỉ số đoạn theo (cảnh, bước)."""
    def __init__(self, rows, ys):
        F, seg, mv, rel, sc = [], [], [], [], []
        for i, rr in enumerate(rows):
            for d, rl, Fm in rr:
                seg.append(len(F) and sum(len(f) for f in F)); F.append(Fm); mv.append(d); rel.append(RELS.index(rl)); sc.append(i)
        self.F = np.concatenate(F) if F else np.zeros((0, 8)); self.seg = np.array(seg, int)
        self.mv = np.array(mv); self.rel = np.array(rel); self.sc = np.array(sc); self.y = np.array(ys); self.n = len(rows)
        self.first = np.zeros((len(mv), 3)); self.first[self.rel == 1, 0] = 1; self.first[self.rel == 2, 1] = 1; self.first[self.rel == 3, 2] = 1

    def predict(self, w, order):
        """w: 11 số. order: thứ tự ưu tiên kiểu rẽ bước đầu khi hòa (chuỗi hoán vị SRLB)."""
        if not len(self.mv):
            return np.zeros(0, int)
        c = np.minimum.reduceat(self.F @ w[:8], self.seg) + self.first @ w[8:11]
        pr = np.array([order.index(RELS[k]) for k in self.rel]) * 1e-7
        c = c + pr
        out = np.full(self.n, -1)
        best = np.full(self.n, np.inf)
        for j in range(len(c)):
            i = self.sc[j]
            if c[j] < best[i] - 1e-9:
                best[i] = c[j]; out[i] = self.mv[j]
        return out

    def acc(self, w, order):
        return (self.predict(w, order) == self.y).sum()


ORDERS = ["".join(p) for p in itertools.permutations("SRLB")]
NAMES = ["hop", "crowd", "cover", "lm", "stairs", "tR", "tL", "tB", "tR0", "tL0", "tB0"]


def rand_w(rng):
    w = np.zeros(11)
    w[0] = 1.0
    for k in range(1, 11):
        if rng.random() < 0.5:
            w[k] = rng.choice([0.25, 0.5, 1, 2, 3, 5, 8, 20, 100, 1000]) * (1 if k != 2 or rng.random() < 0.6 else -0.3)
    if r != 4:
        w[4] = 0
    if rng.random() < 0.3:
        w[0] = rng.choice([0, 1, 100, 1e4])
    return w


def fit(B, rng, nr):
    best = (-1, None, None)
    for _ in range(nr):
        w = rand_w(rng); o = ORDERS[rng.integers(24)]
        a = B.acc(w, o)
        if a > best[0]:
            best = (a, w, o)
    a, w, o = best
    vals = [0, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 20, 50, 100, 1000, 1e4]
    for it in range(3):
        ch = False
        for k in range(11):
            if r != 4 and k == 4:
                continue
            for v in vals + [-0.3, -0.6, -1] * (k == 2):
                w2 = w.copy(); w2[k] = v; a2 = B.acc(w2, o)
                if a2 > a:
                    a, w, ch = a2, w2, True
        for o2 in ORDERS:
            a2 = B.acc(w, o2)
            if a2 > a:
                a, o, ch = a2, o2, True
        if not ch:
            break
    return a, w, o


if __name__ == "__main__":
    t0 = time.time()
    DT, RT = prep("train"); DV, RV = prep("validation")
    gt = collections.defaultdict(list); gv = collections.defaultdict(list)
    for x, rr in zip(DT, RT):
        gt[gkey(x)].append((rr, x["y"][r]))
    for x, rr in zip(DV, RV):
        gv[gkey(x)].append((rr, x["y"][r]))
    rng = np.random.default_rng(0)
    tot_t = tot_v = n_t = n_v = 0
    for g in sorted(gt):
        Bt = Batch([a for a, _ in gt[g]], [b for _, b in gt[g]])
        a, w, o = fit(Bt, rng, NR)
        Bv = Batch([a for a, _ in gv.get(g, [])], [b for _, b in gv.get(g, [])])
        av = Bv.acc(w, o) if gv.get(g) else 0
        tot_t += a; n_t += Bt.n; tot_v += av; n_v += Bv.n
        ws = " ".join(f"{NAMES[k]}={w[k]:g}" for k in range(11) if w[k])
        print(f"  [{g}] train {a}/{Bt.n} = {a / Bt.n:.3f} | val {av}/{Bv.n} | {o} | {ws}", flush=True)
    print(f"R{r} key={KEY}: train {tot_t / n_t:.3f} validation {tot_v / max(1, n_v):.3f} ({time.time() - t0:.0f}s)")
