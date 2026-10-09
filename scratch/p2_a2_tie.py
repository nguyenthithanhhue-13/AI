"""Dò quy tắc phá hòa cho một robot: chi phí chính + (tùy chọn) chi phí phụ theo thứ tự từ điển + thứ tự hướng.
    python scratch/p2_a2_tie.py <robot> [split]"""
import sys, itertools, collections
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]); split = sys.argv[2] if len(sys.argv) > 2 else "train"
D = load(split)
BIG = 1000.0
PRIMARY = {0: dict(e={}, t={})}
pe = PRIMARY.get(r, dict(e={}, t={}))


def comp(sec):
    """sec: tên chi phí phụ -> (ecost, tcost) đã nhân thang BIG cho chi phí chính."""
    e0 = mk_ecost(**pe["e"]); t0 = mk_tcost(**pe["t"])
    se = {"crowd+": lambda i: 1.0 if i[0] == "crowded" else 0.0, "crowd-": lambda i: -1.0 if i[0] == "crowded" else 0.0,
          "cover+": lambda i: 1.0 if i[0] == "covered" else 0.0, "cover-": lambda i: -1.0 if i[0] == "covered" else 0.0,
          "normal-": lambda i: -1.0 if i[0] == "normal" else 0.0}
    st = {"turns": (1, 1, 1), "turnsRL2": (1, 1, 2), "turnL": (0, 1, 1), "turnR": (1, 0, 1)}
    if sec is None:
        return e0, t0, 1.0
    if sec in se:
        f = se[sec]
        return (lambda i, n, d: BIG * e0(i, n, d) + f(i)), (lambda h, d: BIG * t0(h, d)), 1.0
    if sec in st:
        R, L, B = st[sec]
        t1 = mk_tcost(R, L, B)
        return (lambda i, n, d: BIG * e0(i, n, d)), (lambda h, d: BIG * t0(h, d) + t1(h, d)), 1.0
    if sec in ("turns_noinit",):
        t1 = mk_tcost(1, 1, 1)
        return (lambda i, n, d: BIG * e0(i, n, d)), (lambda h, d: BIG * t0(h, d) + t1(h, d)), 0.0


SECS = [None, "turns", "turns_noinit", "turnsRL2", "turnL", "turnR", "crowd+", "crowd-", "cover+", "cover-"]
ABS = list(itertools.permutations(range(4)))
RELO = list(itertools.permutations("SRLB"))
for sec in SECS:
    ec, tc, ft = comp(sec)
    res_abs = collections.Counter(); res_rel = collections.Counter(); amb = 0; inset = 0
    for x in D:
        q = qvals(x["w"], x["legs"], ec, tc, r == 4, first_turn=ft if ft is not None else 1.0)
        a = argmins(q, eps=1e-6)
        y = x["y"][r]
        inset += y in a
        if len(a) > 1:
            amb += 1
        for o in ABS:
            res_abs[o] += bool(a) and min(a, key=lambda d: o.index(d)) == y
        h = x["w"]["heading"]
        for o in RELO:
            res_rel[o] += bool(a) and min(a, key=lambda d: o.index(REL[h][d])) == y
    ba = res_abs.most_common(2); br = res_rel.most_common(2)
    n = len(D)
    print(f"R{r} phụ={str(sec):13s} trong argmin {inset/n:.4f}  còn hòa {amb:4d}  | tốt nhất thứ tự tuyệt đối "
          f"{[(''.join('UDLR'[i] for i in o), round(v/n, 4)) for o, v in ba]}  | tương đối {[(''.join(o), round(v/n, 4)) for o, v in br]}", flush=True)
