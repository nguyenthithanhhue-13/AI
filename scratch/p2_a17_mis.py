"""In các cảnh mà nhãn của robot KHÔNG nằm trong argmin của một giả thuyết chi phí (để soi bằng mắt).
    python scratch/p2_a17_mis.py <robot> <crowd> <cover> <lm> <turn> [K]"""
import sys
sys.path.insert(0, "scratch")
from p2_fast import *

r = int(sys.argv[1]); cr, cv, lm, tu = map(float, sys.argv[2:6]); K = int(sys.argv[6]) if len(sys.argv) > 6 else 6
D = load("train")
th = theta_of(crowd=cr, cover=cv, lm=lm, tR=tu, tL=tu, tB=2 * tu)
SYM = {"normal": "-", "crowded": "=", "covered": "~"}
shown = 0
for i, x in enumerate(D):
    w = x["w"]
    g = SG(w, r == 4, lm_excl(x))
    q = g.q(th, x["legs"])
    a = argmins(q, 1e-6)
    y = x["y"][r]
    if y in a:
        continue
    q0 = SG(w, r == 4, lm_excl(x)).q(theta_of(), x["legs"])
    lmset = {p: k[:2].upper() for k, v in w["landmarks"].items() for p in v}
    tg = {p for L in x["legs"] for p in L}
    print(f"#{i} heading {ACTIONS[w['heading']]} rain={w['rain']} urgent={x['m']['urgent']} fragile={x['m']['fragile']}  nhãn R{r}={ACTIONS[y]}  "
          f"q={[round(v, 2) if v < INF else None for v in q]}  hops={[int(v) if v < INF else None for v in q0]}  legs={x['legs']}")
    print("   nhãn 10 robot:", "".join(ACTIONS[v][0] for v in x["y"]))
    R = max(n["rc"][0] for n in x["s"]["nodes"]) + 1; C = max(n["rc"][1] for n in x["s"]["nodes"]) + 1
    for rr in range(R):
        a1 = ""; b1 = ""
        for cc in range(C):
            p = (rr, cc)
            mk = "RB" if p == w["robot"] else (lmset[p] if p in lmset else ("++" if p in w["adj"] else "  "))
            if p in tg and p in lmset:
                mk = mk.lower()
            a1 += mk
            e = w["adj"].get(p, {}).get(3)
            if e and e[0] != "closed":
                s = SYM[e[0]] if not e[1] else "s"
                if not e[2]:
                    s = "<"
                elif w["adj"].get((rr, cc + 1), {}).get(2) and not w["adj"][(rr, cc + 1)][2][2]:
                    s = ">"
                a1 += s * 2
            else:
                a1 += "  "
            e2 = w["adj"].get(p, {}).get(1)
            if e2 and e2[0] != "closed":
                s = {"normal": "|", "crowded": "H", "covered": ":"}[e2[0]] if not e2[1] else "s"
                if not e2[2]:
                    s = "^"
                elif w["adj"].get((rr + 1, cc), {}).get(0) and not w["adj"][(rr + 1, cc)][0][2]:
                    s = "v"
                b1 += s + "   "
            else:
                b1 += "    "
        print("   " + a1); print("   " + b1)
    shown += 1
    if shown >= K:
        break
