import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
from p2_a7_show import draw
r = 0; D = load("train"); BIG = 1000.0
for x in D:
    w = x["w"]; lms = {p: k for k, v in w["landmarks"].items() for p in v}
    ec = lambda info, n, d: BIG + ((n[0] + DRC[d][0], n[1] + DRC[d][1]) in lms)
    q = qvals(w, x["legs"], ec, mk_tcost(), False)
    a = argmins(q, 1e-6); y = x["y"][r]
    if y in a: continue
    q0 = qvals(w, x["legs"], mk_ecost(), mk_tcost(), False)
    marks = {p: k[:2].upper() for p, k in lms.items()}; marks[w["robot"]] = "RB"
    print(f"heading {ACTIONS[w['heading']]} nhãn {ACTIONS[y]}  q(hops,lm)={[round(v,1) if v < INF else None for v in q]}  goal={x['m']['goal']} ref={x['m']['goal_ref']} via={x['m']['via']} vref={x['m']['via_ref']}")
    print("  legs", x["legs"], " nhãn 10 robot", [ACTIONS[v][0] for v in x["y"]])
    R = max(n["rc"][0] for n in x["s"]["nodes"]) + 1; C = max(n["rc"][1] for n in x["s"]["nodes"]) + 1
    for i in range(R):
        print("   " + " ".join(f"{marks.get((i, j), '..' if (i, j) in w['adj'] else '  '):2s}" for j in range(C)))
