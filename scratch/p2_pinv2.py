"""DÒ NGƯỢC trọng số chi phí trên MẶT PARETO CÓ LOẠI ĐỊA ĐIỂM (TYPED=1 scratch/p2_pareto.py).
Chi phí đường = w · [số đoạn, đông, mái che, xuyên địa điểm (mọi loại), bậc thang, rẽ R, rẽ L, quay đầu, xuyên loại 1..10]
+ phạt rẽ ở bước đầu (R0, L0, B0); bước = argmin, hòa -> thứ tự kiểu rẽ bước đầu. Nhóm điều kiện theo KEY như p2_pinv.
    python scratch/p2_pinv2.py <robot> <key> [n_random] [seed]"""
import sys, pickle, itertools, collections, time, json, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]); KEY = sys.argv[2]; NR = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
leg = int(r == 4)
RELS = "SRLB"
PL = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
NAMES = ["hop", "crowd", "cover", "lm", "stairs", "tR", "tL", "tB"] + PL + ["tR0", "tL0", "tB0"]
NW = len(NAMES)


def prep(split):
    D = load(split)
    P = pickle.load(open(f"cache/p2_paretot_{split}_{leg}.pkl", "rb"))
    return D, [[(d, rel, F.astype(np.float64)) for d, (rel, F) in sorted(fr.items())] for fr in P]


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
    def __init__(self, rows, ys):
        F, seg, mv, rel, sc = [], [], [], [], []
        pos = 0
        for i, rr in enumerate(rows):
            for d, rl, Fm in rr:
                seg.append(pos); pos += len(Fm); F.append(Fm); mv.append(d); rel.append(RELS.index(rl)); sc.append(i)
        self.F = np.concatenate(F) if F else np.zeros((0, NW - 3)); self.seg = np.array(seg, int)
        self.mv = np.array(mv); self.rel = np.array(rel); self.sc = np.array(sc); self.y = np.array(ys); self.n = len(rows)
        self.first = np.zeros((len(mv), 3))
        for k in (1, 2, 3):
            self.first[self.rel == k, k - 1] = 1
        self.ymask = self.mv == self.y[self.sc] if len(mv) else np.zeros(0, bool)

    def acc(self, w, order):
        if not len(self.mv):
            return 0
        c = np.minimum.reduceat(self.F @ w[:-3], self.seg) + self.first @ w[-3:]
        c = c + np.array([order.index(RELS[k]) for k in range(4)])[self.rel] * 1e-7
        best = np.full(self.n, np.inf); np.minimum.at(best, self.sc, c)
        win = c <= best[self.sc] + 1e-9
        return int((win & self.ymask).sum())


ORDERS = ["".join(p) for p in itertools.permutations("SRLB")]
VALS = [0, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 20, 50, 100, 1000, 1e4]


def rand_w(rng):
    w = np.zeros(NW); w[0] = rng.choice([1, 1, 1, 100, 1e4])
    for k in [1, 2, 3, 5, 6, 7, NW - 3, NW - 2, NW - 1] + ([4] if r == 4 else []):
        if rng.random() < 0.5:
            w[k] = rng.choice([0.25, 0.5, 1, 2, 3, 5, 8, 20, 100]) * (-0.3 if k == 2 and rng.random() < 0.4 else 1)
    S = rng.random(10) < rng.choice([0.1, 0.2, 0.3, 0.5])      # nhóm loại bị phạt thêm, cùng mức
    w[8:18] = S * rng.choice([0.5, 1, 2, 3, 5, 8, 20, 100])
    return w


def fit(B, rng, nr):
    best = (-1, None, None)
    for _ in range(nr):
        w = rand_w(rng); o = ORDERS[rng.integers(24)]
        a = B.acc(w, o)
        if a > best[0]:
            best = (a, w, o)
    a, w, o = best
    for it in range(4):
        ch = False
        for k in range(NW):
            if r != 4 and k == 4:
                continue
            for v in VALS + ([-0.3, -0.6, -1, -2] if k == 2 else []):
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
    rng = np.random.default_rng(SEED)
    tot_t = tot_v = n_t = n_v = 0; out = {}
    for g in sorted(gt):
        Bt = Batch([a for a, _ in gt[g]], [b for _, b in gt[g]])
        a, w, o = fit(Bt, rng, NR)
        Bv = Batch([a for a, _ in gv.get(g, [])], [b for _, b in gv.get(g, [])])
        av = Bv.acc(w, o) if gv.get(g) else 0
        tot_t += a; n_t += Bt.n; tot_v += av; n_v += Bv.n
        out[g] = dict(w=w.tolist(), order=o)
        ws = " ".join(f"{NAMES[k]}={w[k]:g}" for k in range(NW) if w[k])
        print(f"  [{g}] train {a}/{Bt.n} = {a / Bt.n:.3f} | val {av}/{Bv.n} | {o} | {ws}", flush=True)
    print(f"R{r} key={KEY} (có loại địa điểm): train {tot_t / n_t:.3f} validation {tot_v / max(1, n_v):.3f} ({time.time() - t0:.0f}s)")
    json.dump(out, open(f"cache/p2_pinv2_r{r}_{KEY}.json", "w", encoding="utf-8"), ensure_ascii=False)
