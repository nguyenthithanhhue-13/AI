import sys, json, collections
sys.path.insert(0, "scratch")
from p2_fast import *
P = json.load(open("cache/p2_cd3_r3_w.json", encoding="utf-8"))
D = load("train"); c = collections.Counter(); n = 0
for x in D:
    p = P[x["s"]["weather"]]; th = theta_of(**p)
    g = SG(x["w"], False, lm_excl(x)); q = g.q(th, x["legs"]); a = argmins(q, 1e-6)
    h = x["w"]["heading"]; y = x["y"][3]
    pred = min(a, key=lambda d: "SRLB".index(REL[h][d]))
    if pred == y: continue
    n += 1
    c["nhãn " + REL[h][y]] += 1; c["đoán " + REL[h][pred]] += 1
    c[("cặp", REL[h][pred], REL[h][y])] += 1
    c["nhãn hợp lệ số bước=" + str(sum(1 for d, i in x["w"]["adj"].get(x["w"]["robot"], {}).items() if edge_ok(i, False)))] += 1
print("số lỗi", n)
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:20]: print("  ", k, v)
