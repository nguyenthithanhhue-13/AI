"""Tìm CHUỖI TIÊU CHÍ THEO THỨ TỰ TỪ ĐIỂN cho một robot, riêng từng nhóm điều kiện (beam search).
Tiêu chí chính = số đoạn + a*đông + b*địa điểm + c*mái che (có trọng số); sau đó tối đa 3 tiêu chí phụ (từ điển); hòa cuối
cùng phá theo thứ tự hướng (tương đối / tuyệt đối) tốt nhất. Phát hiện ở R0: mưa -> (số đoạn, địa điểm, NHIỀU mái che, ít đông),
khô -> (số đoạn, địa điểm, ít đông): 99-99,6%.
    python scratch/p2_lex.py <robot> <khóa nhóm> [n_train]"""
import sys, json, time, itertools, numpy as np
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); key = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 600
NM = lambda x: x["s"]["style"] == "night"
KF = {"all": lambda x: "all", "w": lambda x: x["s"]["weather"], "nd": lambda x: "đêm" if NM(x) else "ngày",
      "nd_w": lambda x: ("đêm" if NM(x) else "ngày") + "/" + x["s"]["weather"],
      "f": lambda x: "dv=" + str(x["m"]["fragile"]), "u": lambda x: "gấp=" + str(x["m"]["urgent"]),
      "v": lambda x: "ghé=" + str(bool(x["m"]["via"])),
      "nd_f": lambda x: ("đêm" if NM(x) else "ngày") + "/dv=" + str(x["m"]["fragile"]),
      "nd_u": lambda x: ("đêm" if NM(x) else "ngày") + "/gấp=" + str(x["m"]["urgent"]),
      "w_f": lambda x: x["s"]["weather"] + "/dv=" + str(x["m"]["fragile"]),
      "w_u": lambda x: x["s"]["weather"] + "/gấp=" + str(x["m"]["urgent"]),
      "w_v": lambda x: x["s"]["weather"] + "/ghé=" + str(bool(x["m"]["via"])),
      "nd_v": lambda x: ("đêm" if NM(x) else "ngày") + "/ghé=" + str(bool(x["m"]["via"]))}[key]
# vector trên [one, crowd, cover, normal, stairs, lm, tR, tL, tB] (bước đầu dùng cùng phạt rẽ)
E = lambda **k: np.array([k.get(n, 0.0) for n in ["one", "crowd", "cover", "normal", "stairs", "lm", "tR", "tL", "tB"]])
CRIT = {"lm": E(lm=1), "crowd": E(crowd=1), "-cover": E(cover=-1), "cover": E(cover=1), "uncov": E(crowd=1, normal=1),
        "normal": E(normal=1), "turns": E(tR=1, tL=1, tB=1), "tR": E(tR=1), "tL": E(tL=1), "tB": E(tB=1),
        "stairs": E(stairs=1), "-stairs": E(stairs=-1), "hops": E(one=1)}
if r != 4:
    CRIT.pop("stairs"); CRIT.pop("-stairs")
PRIM = {}
for a, b, c, t in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [-0.5, 0, 1], [0, 1]):
    PRIM[f"P(đông{a},đđ{b},mái{c},rẽ{t})"] = E(one=1, crowd=a, lm=b, cover=c, tR=t, tL=t, tB=2 * t)
ORD = [("rel", "".join(o)) for o in itertools.permutations("SRLB")] + [("abs", o) for o in itertools.permutations(range(4))]


def full_theta(prim, chain):
    th = PRIM[prim] * 1e6
    for i, c in enumerate(chain):
        th = th + CRIT[c] * (100.0 ** (2 - i))
    return np.concatenate([th, th[6:9]])


def prep(D):
    return [(SG(x["w"], r == 4, lm_excl(x)), x["legs"], x["y"][r], x["w"]["heading"]) for x in D]


def argsets(G, th):
    return [argmins(g.q(th, legs), 1e-3) for g, legs, y, h in G]


def score(G, A):
    """(số khớp với thứ tự phá hòa tốt nhất, thứ tự đó, số nhãn trong tập tối ưu)."""
    ins = sum(y in a for (g, l, y, h), a in zip(G, A))
    best = (-1, None)
    for o in ORD:
        ok = 0
        for (g, l, y, h), a in zip(G, A):
            if not a:
                continue
            pred = min(a, key=lambda d: o[1].index(REL[h][d])) if o[0] == "rel" else min(a, key=lambda d: o[1].index(d))
            ok += pred == y
        if ok > best[0]:
            best = (ok, o)
    return best[0], best[1], ins


def search(G):
    # bước 1: tiêu chí chính
    res = []
    for p in PRIM:
        A = argsets(G, full_theta(p, []))
        s, o, ins = score(G, A)
        res.append((s, ins, p, (), o))
    res.sort(key=lambda z: (-z[0], -z[1]))
    beam = res[:6]
    best = beam[0]
    for depth in range(3):
        cand = []
        for s0, i0, p, ch, o0 in beam:
            for c in CRIT:
                if c in ch:
                    continue
                A = argsets(G, full_theta(p, list(ch) + [c]))
                s, o, ins = score(G, A)
                cand.append((s, ins, p, ch + (c,), o))
        cand.sort(key=lambda z: (-z[0], -z[1]))
        beam = cand[:6]
        if beam[0][0] > best[0]:
            best = beam[0]
    return best


def evaluate(G, p, ch, o):
    A = argsets(G, full_theta(p, list(ch)))
    ok = 0
    for (g, l, y, h), a in zip(G, A):
        if a:
            pred = min(a, key=lambda d: o[1].index(REL[h][d])) if o[0] == "rel" else min(a, key=lambda d: o[1].index(d))
            ok += pred == y
    return ok


import collections
TR = load("train"); VA = load("validation")
gtr = collections.defaultdict(list); gva = collections.defaultdict(list)
for x in TR: gtr[KF(x)].append(x)
for x in VA: gva[KF(x)].append(x)
t0 = time.time(); tot_t = tot_v = 0; OUTP = {}
for k in sorted(gtr):
    G = prep(gtr[k][:N]); s, ins, p, ch, o = search(G)
    GV = prep(gva.get(k, [])); v = evaluate(GV, p, ch, o) if GV else 0
    GT = prep(gtr[k]); tt = evaluate(GT, p, ch, o)
    tot_t += tt; tot_v += v
    OUTP[k] = {"prim": p, "chain": list(ch), "order": [o[0], list(o[1]) if o[0] == "abs" else o[1]]}
    print(f"R{r} {key}={k}: train {tt}/{len(gtr[k])}  validation {v}/{len(gva.get(k, []))}  {p} {ch} {o}  ({time.time()-t0:.0f}s)", flush=True)
print(f"R{r} {key}: TỔNG train {tot_t/len(TR):.4f}  validation {tot_v/len(VA):.4f}", flush=True)
json.dump(OUTP, open(f"cache/p2_lex_r{r}_{key}.json", "w"), ensure_ascii=False)
