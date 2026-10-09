"""Giả thuyết: R3 (thời tiết) tránh đoạn đường SÁT HỒ (giao lộ không tồn tại trong lưới) khi mưa.
Chi phí = 1 + crowd*đông + cover*mái + lm*địa điểm + lake*[một đầu mút kề ô hồ]; dò lưới nhỏ theo mưa / khô."""
import sys, itertools
sys.path.insert(0, "scratch")
from p2_lib import *

r = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TR = load("train"); VA = load("validation")


def lakeset(x):
    s = x["s"]; R, C = s["grid"]["rows"], s["grid"]["cols"]
    nodes = {tuple(n["rc"]) for n in s["nodes"]}
    holes = {(i, j) for i in range(R) for j in range(C) if (i, j) not in nodes}
    return {n for n in nodes if any((n[0] + a, n[1] + b) in holes for a, b in DRC)}, len(holes)


def run(D, p):
    ok = 0
    for x in D:
        L = {q for v in x["w"]["landmarks"].values() for q in v} - {q for Lg in x["legs"] for q in Lg}
        LK, _ = lakeset(x)
        cr, cv, lm, lk, tb = p

        def ec(info, n, d):
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            return 1 + cr * (info[0] == "crowded") + cv * (info[0] == "covered") + lm * (n2 in L) + lk * ((n in LK) or (n2 in LK))
        q = qvals(x["w"], x["legs"], ec, mk_tcost(0, 0, tb), r == 4)
        a = argmins(q, 1e-6)
        h = x["w"]["heading"]
        ok += bool(a) and min(a, key=lambda d: "SRLB".index(REL[h][d])) == x["y"][r]
    return ok


nh = sum(lakeset(x)[1] > 0 for x in TR)
print("số cảnh train có hồ:", nh, "/", len(TR))
for wcond in (True, False):
    T = [x for x in TR if x["w"]["rain"] == wcond][:400]; V = [x for x in VA if x["w"]["rain"] == wcond]
    res = sorted(((run(T, p), p) for p in itertools.product([0.5, 1, 2], [-0.6, 0, 0.5], [0, 2, 5], [0, 0.5, 1, 2, 5], [0.5])), reverse=True)
    for sc, p in res[:3]:
        print("mưa" if wcond else "khô", f"train {sc}/{len(T)}  validation {run(V, p)}/{len(V)}", dict(zip(["crowd", "cover", "lm", "lake", "tB"], p)), flush=True)
