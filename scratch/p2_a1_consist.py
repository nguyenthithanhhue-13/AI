"""Tỉ lệ nhãn nằm trong tập argmin của từng họ chi phí đơn giản (chưa xét phá hòa) + số bước hợp lệ trung bình."""
import sys, itertools, collections, time
sys.path.insert(0, "scratch")
from p2_lib import *
D = load("train")
t = time.time()
fams = {
  "hops": (mk_ecost(), mk_tcost()),
  "hops+turn.5": (mk_ecost(), mk_tcost(.5, .5, .5)),
  "crowd5": (mk_ecost(crowd=5), mk_tcost()),
  "cover-.5": (mk_ecost(cover=-.5), mk_tcost()),
  "crowd2": (mk_ecost(crowd=2), mk_tcost()),
  "turn3B30": (mk_ecost(), mk_tcost(3, 3, 30)),
  "R.5L4.5B9": (mk_ecost(), mk_tcost(.5, 4.5, 9)),
}
for name, (ec, tc) in fams.items():
    cons = collections.Counter(); uniq = collections.Counter()
    for x in D:
        for r in range(10):
            q = qvals(x["w"], x["legs"], ec, tc, r == 4)
            a = argmins(q)
            cons[r] += x["y"][r] in a
            uniq[r] += (len(a) == 1 and x["y"][r] in a)
    print(f"{name:12s} trong argmin: " + " ".join(f"{cons[r]/len(D):.3f}" for r in range(10)) + "  | argmin duy nhất & đúng: " + " ".join(f"{uniq[r]/len(D):.2f}" for r in range(10)), flush=True)
print(f"{time.time()-t:.0f}s")
