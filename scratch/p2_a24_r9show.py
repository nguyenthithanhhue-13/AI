import sys
sys.path.insert(0, "scratch")
from p2_lib import *
D = load("train"); r = 9
man = lambda p, t: abs(p[0] - t[0]) + abs(p[1] - t[1])
SYM = {"normal": "-", "crowded": "=", "covered": "~"}
k = 0
for i, x in enumerate(D):
    w = x["w"]; s = w["robot"]; tg = x["legs"][0]; h = w["heading"]
    lm = {p for v in w["landmarks"].values() for p in v} - {p for L in x["legs"] for p in L}
    mv = {}
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, False): continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        mv[d] = (min(man(n2, t) for t in tg), info[0], n2 in lm, REL[h][d])
    sc = {d: m + 2 * (st == "crowded") for d, (m, st, l, rl) in mv.items()}
    mn = min(sc.values())
    if sc[x["y"][r]] <= mn: continue
    k += 1
    if k > 10: break
    print(f"#{i} robot {s} heading {ACTIONS[h]} đích {tg} rain={w['rain']} urg={x['m']['urgent']} frag={x['m']['fragile']} nhãn {ACTIONS[x['y'][r]]} | " +
          "  ".join(f"{ACTIONS[d]}:man{m},{st[:5]},{'LM' if l else ''},{rl}" for d, (m, st, l, rl) in mv.items()) + f" | 10 robot {''.join(ACTIONS[v][0] for v in x['y'])}")
