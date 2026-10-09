"""R0: in các ca sai với (hops, lm, crowd) + thứ tự tương đối SRLB."""
import sys
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a16_chain import build
SYM = {"normal": "-", "crowded": "=", "covered": "~"}
D = load("train"); r = 0
chain = ["hops", "lm", "crowd"]
n = 0
for i, x in enumerate(D):
    ec, tc = build(x, chain)
    q = qvals(x["w"], x["legs"], ec, tc, False)
    a = argmins(q, 1e-7)
    h = x["w"]["heading"]
    pred = min(a, key=lambda d: "SRLB".index(REL[h][d])) if a else None
    y = x["y"][r]
    if pred == y: continue
    n += 1
    if n > 7: continue
    w = x["w"]
    lmset = {p: k[:2].upper() for k, v in w["landmarks"].items() for p in v}
    tg = {p for L in x["legs"] for p in L}
    print(f"#{i} heading {ACTIONS[h]} nhãn {ACTIONS[y]} đoán {ACTIONS[pred] if pred is not None else None}  q={[round(v,4) if v < INF else None for v in q]} legs={x['legs']}  10 robot: {''.join(ACTIONS[v][0] for v in x['y'])}")
    R = max(nn["rc"][0] for nn in x["s"]["nodes"]) + 1; C = max(nn["rc"][1] for nn in x["s"]["nodes"]) + 1
    for rr in range(R):
        a1 = ""; b1 = ""
        for cc in range(C):
            p = (rr, cc)
            mk = "RB" if p == w["robot"] else (lmset[p] if p in lmset else ("++" if p in w["adj"] else "  "))
            if p in tg and p in lmset: mk = mk.lower()
            a1 += mk
            e = w["adj"].get(p, {}).get(3)
            if e and e[0] != "closed":
                s = SYM[e[0]] if not e[1] else "s"
                if not e[2]: s = "<"
                elif w["adj"].get((rr, cc + 1), {}).get(2) and not w["adj"][(rr, cc + 1)][2][2]: s = ">"
                a1 += s * 2
            else: a1 += "  "
            e2 = w["adj"].get(p, {}).get(1)
            if e2 and e2[0] != "closed":
                s = {"normal": "|", "crowded": "H", "covered": ":"}[e2[0]] if not e2[1] else "s"
                if not e2[2]: s = "^"
                elif w["adj"].get((rr + 1, cc), {}).get(0) and not w["adj"][(rr + 1, cc)][0][2]: s = "v"
                b1 += s + "   "
            else: b1 += "    "
        print("   " + a1); print("   " + b1)
print("tổng sai", n)
