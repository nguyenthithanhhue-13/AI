"""R0: chi phí = (số đoạn, số địa điểm đi ngang) theo thứ tự từ điển; xem phần hòa còn lại + thứ tự hướng / đặc trưng phụ."""
import sys, collections, itertools
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1]); D = load(sys.argv[2] if len(sys.argv) > 2 else "train"); leg = r == 4
BIG = 1000.0
def mk(x, wl=1.0, sec=None):
    lms = {p for v in x["w"]["landmarks"].values() for p in v}
    def ec(info, n, d):
        n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
        c = BIG + wl * (n2 in lms)
        if sec: c += sec(info)
        return c
    return ec
ABS = list(itertools.permutations(range(4))); RELO = list(itertools.permutations("SRLB"))
SECS = {None: None, "crowd+": lambda i: 0.01 * (i[0] == "crowded"), "crowd-": lambda i: -0.01 * (i[0] == "crowded"),
        "cover+": lambda i: 0.01 * (i[0] == "covered"), "cover-": lambda i: -0.01 * (i[0] == "covered")}
for sn, sf in SECS.items():
    inset = 0; ties = []
    for x in D:
        q = qvals(x["w"], x["legs"], mk(x, 1.0, sf), mk_tcost(), leg)
        a = argmins(q, 1e-6); y = x["y"][r]
        inset += y in a
        if len(a) > 1 and y in a: ties.append((x, a, y))
    ra = collections.Counter(); rr = collections.Counter()
    for x, a, y in ties:
        h = x["w"]["heading"]
        for o in ABS: ra[o] += min(a, key=lambda d: o.index(d)) == y
        for o in RELO: rr[o] += min(a, key=lambda d: o.index(REL[h][d])) == y
    n = len(ties) or 1
    print(f"phụ={sn}: trong argmin {inset}/{len(D)}  còn hòa {len(ties)}  tuyệt đối {[(''.join('UDLR'[i] for i in o), round(v/n,3)) for o, v in ra.most_common(2)]}  tương đối {[(''.join(o), round(v/n,3)) for o, v in rr.most_common(2)]}", flush=True)
    if sn is None: pickle.dump(ties, open(f"cache/p2_ties_lm_r{r}.pkl", "wb"))
