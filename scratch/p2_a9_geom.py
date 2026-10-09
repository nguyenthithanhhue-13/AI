"""Giả thuyết: hòa số đoạn được phá bằng ĐỘ DÀI HÌNH HỌC (tọa độ pixel xy của giao lộ, vì giao lộ bị lệch)."""
import sys, math, collections
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1]) if len(sys.argv) > 1 else 0
D = load("train")
BIG = 1e6
for mode in ("len", "-len"):
    ok = 0; amb = 0; tie_ok = 0; nt = 0
    for x in D:
        xy = {tuple(n["rc"]): n["xy"] for n in x["s"]["nodes"]}
        sg = 1 if mode == "len" else -1
        def ec(info, n, d, xy=xy, sg=sg):
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            return BIG + sg * math.hypot(xy[n][0] - xy[n2][0], xy[n][1] - xy[n2][1])
        q = qvals(x["w"], x["legs"], ec, mk_tcost(), r == 4)
        a = argmins(q, 1e-6)
        y = x["y"][r]
        ok += a == [y]; amb += len(a) > 1
        q0 = qvals(x["w"], x["legs"], mk_ecost(), mk_tcost(), r == 4)
        if len(argmins(q0)) > 1:
            nt += 1; tie_ok += a == [y]
    print(f"R{r} {mode}: đúng {ok}/{len(D)}  còn hòa {amb}  | trên các ca hòa số đoạn: {tie_ok}/{nt}")
