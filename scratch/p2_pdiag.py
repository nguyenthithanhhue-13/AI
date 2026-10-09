"""Soi cảnh SAI của một nhóm điều kiện sau khi dò trọng số (cache/p2_pinv2_r<r>_<key>.json): với mỗi bước đi, đường rẻ nhất
theo trọng số đã dò (đặc trưng) và chi phí; nhãn thật; các đường Pareto của bước thật.
    python scratch/p2_pdiag.py <robot> <key> <nhóm> [số cảnh]"""
import sys, json, numpy as np
sys.argv, (r, KEY, G) = sys.argv[:1] + [sys.argv[1], sys.argv[2]], sys.argv[1:4]
NSHOW = 8
exec(open("scratch/p2_pinv2.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0])
cfg = json.load(open(f"cache/p2_pinv2_r{r}_{KEY}.json", encoding="utf-8"))[G]
w = np.array(cfg["w"]); o = cfg["order"]
DT, RT = prep("train")
short = ["hop", "đông", "mái", "lm", "bậc", "R", "L", "B"] + [p[:3] for p in PL]
shown = 0; stat = __import__("collections").Counter()
for x, rr in zip(DT, RT):
    if gkey(x) != G:
        continue
    y = x["y"][r]; res = []
    for d, rl, F in rr:
        c = F @ w[:-3]; j = int(np.argmin(c))
        res.append((c[j] + np.eye(4)[RELS.index(rl)][1:] @ w[-3:] + [o.index(RELS[k]) for k in range(4)][RELS.index(rl)] * 1e-7, d, rl, F[j], F))
    pd = min(res, key=lambda z: z[0])[1]
    if pd == y:
        continue
    stat["sai"] += 1
    if shown >= NSHOW:
        continue
    shown += 1
    w_ = x["w"]
    print(f"\n== {x['s']['scene_id']} robot {w_['robot']} mũi {'UDLR'[w_['heading']]} đích {x['legs'][-1]} nhãn {'UDLR'[y]} đoán {'UDLR'[pd]}")
    for c, d, rl, f, F in res:
        tag = "NHÃN" if d == y else ("đoán" if d == pd else "")
        nz = " ".join(f"{short[k]}={int(f[k])}" for k in range(len(f)) if f[k])
        print(f"   {'UDLR'[d]} ({rl}) chi phí {c:7.2f} {tag:5s} | {nz}")
    yF = next(F for c, d, rl, f, F in res if d == y)
    print("   các đường Pareto của bước thật:", "; ".join(" ".join(f"{short[k]}={int(v[k])}" for k in range(len(v)) if v[k]) for v in yF[:6]))
print(stat)
