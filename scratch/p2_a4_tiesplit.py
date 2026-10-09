"""Tách ca hòa: (i) cùng một đích (một bản) vs (ii) các bước hòa dẫn tới các bản khác nhau.
    python scratch/p2_a4_tiesplit.py <robot>"""
import sys, collections, itertools, math
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1])
D = load("train")
ec, tc = mk_ecost(), mk_tcost()
leg = r == 4
ABS = list(itertools.permutations(range(4)))
RELO = list(itertools.permutations("SRLB"))
single = []; multi = []
for x in D:
    w = x["w"]; y = x["y"][r]; legs = x["legs"]
    q = qvals(w, legs, ec, tc, leg)
    a = argmins(q)
    if len(a) < 2:
        continue
    # với từng tổ hợp (via, goal) cụ thể: q và tập argmin
    combos = list(itertools.product(*legs))
    best = min(q)
    reach = collections.defaultdict(set)    # bước -> các tổ hợp đạt chi phí tốt nhất
    for cb in combos:
        qc = qvals(w, [[c] for c in cb], ec, tc, leg)
        for d in a:
            if abs(qc[d] - best) < 1e-6:
                reach[d].add(cb)
    common = set.intersection(*[reach[d] for d in a])
    (single if common else multi).append((x, a, y, reach))
print("hòa cùng một đích:", len(single), " hòa giữa các bản khác nhau:", len(multi))
for name, L in (("cùng đích", single), ("khác bản", multi)):
    ra = collections.Counter(); rr = collections.Counter()
    for x, a, y, _ in L:
        h = x["w"]["heading"]
        for o in ABS:
            ra[o] += min(a, key=lambda d: o.index(d)) == y
        for o in RELO:
            rr[o] += min(a, key=lambda d: o.index(REL[h][d])) == y
    n = len(L)
    print(name, n, "tuyệt đối", [("".join("UDLR"[i] for i in o), round(v / n, 3)) for o, v in ra.most_common(3)],
          "tương đối", [("".join(o), round(v / n, 3)) for o, v in rr.most_common(3)])
pickle.dump((single, multi), open(f"cache/p2_tiesplit_r{r}.pkl", "wb"))
