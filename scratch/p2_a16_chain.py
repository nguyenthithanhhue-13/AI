"""Thử CHUỖI tiêu chí theo thứ tự từ điển cho một robot (mỗi tiêu chí nhân một bậc thang 100 lần).
Tiêu chí: hops, lm (địa điểm đi ngang, trừ đích/ghé), crowd, -cover, normal, turns, tR, tL, tB, stairs.
    python scratch/p2_a16_chain.py <robot> "hops,lm,crowd,-cover" ["hops,lm,..."]"""
import sys, itertools, collections
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]) if __name__ == "__main__" else 0; chains = sys.argv[2:]
D = load("train") if __name__ == "__main__" else None
import os
if D is not None and os.environ.get("COND"):
    _c = os.environ["COND"]
    D = [x for x in D if {"rain": x["w"]["rain"], "dry": not x["w"]["rain"], "night": x["s"]["style"] == "night", "day": x["s"]["style"] != "night"}[_c]]
leg = r == 4
ABS = list(itertools.permutations(range(4))); RELO = list(itertools.permutations("SRLB"))


def build(x, chain):
    n = len(chain)
    sc = {c.lstrip("-"): (100.0 ** (n - 1 - i)) * (-1 if c.startswith("-") else 1) for i, c in enumerate(chain)}
    L = {p for v in x["w"]["landmarks"].values() for p in v} - {p for Lg in x["legs"] for p in Lg}
    base = sc.get("hops", 0.0)
    off = sum(abs(v) for v in sc.values()) + 1.0          # giữ chi phí dương

    def ec(info, nd, d):
        n2 = (nd[0] + DRC[d][0], nd[1] + DRC[d][1])
        st = info[0]
        c = off + base
        c += sc.get("lm", 0) * (n2 in L) + sc.get("crowd", 0) * (st == "crowded") + sc.get("cover", 0) * (st == "covered")
        c += sc.get("normal", 0) * (st == "normal") + sc.get("stairs", 0) * bool(info[1])
        return c
    tab = {"S": 0.0, "R": sc.get("turns", 0) + sc.get("tR", 0), "L": sc.get("turns", 0) + sc.get("tL", 0),
           "B": sc.get("turns", 0) + sc.get("tB", 0)}
    tc = lambda h, d: tab[REL[h][d]]
    return ec, tc


if __name__ != "__main__":
    chains = []
for ch in chains:
    chain = ch.split(",")
    ins = 0; uni = 0; ties = []
    for x in D:
        ec, tc = build(x, chain)
        q = qvals(x["w"], x["legs"], ec, tc, leg)
        a = argmins(q, 1e-7)
        y = x["y"][r]
        ins += y in a; uni += a == [y]
        if len(a) > 1 and y in a:
            ties.append((x, a, y))
    ra = collections.Counter(); rr = collections.Counter()
    for x, a, y in ties:
        h = x["w"]["heading"]
        for o in ABS:
            ra[o] += min(a, key=lambda d: o.index(d)) == y
        for o in RELO:
            rr[o] += min(a, key=lambda d: o.index(REL[h][d])) == y
    (oa, va), = ra.most_common(1) if ra else [((0, 1, 2, 3), 0)]
    (orr, vr), = rr.most_common(1) if rr else [(tuple("SRLB"), 0)]
    print(f"R{r} {ch:40s} trong argmin {ins}  duy nhất đúng {uni}  còn hòa {len(ties)}  -> +thứ tự tuyệt đối "
          f"{''.join('UDLR'[i] for i in oa)} = {uni + va}  | tương đối {''.join(orr)} = {uni + vr}", flush=True)
