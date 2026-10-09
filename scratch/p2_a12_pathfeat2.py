"""Ca hòa số đoạn (không điểm ghé): min/max của thêm đặc trưng trên các đường ngắn nhất theo từng bước đầu."""
import sys, collections
sys.path.insert(0, "scratch")
from p2_lib import *
r = int(sys.argv[1])
single, multi = pickle.load(open(f"cache/p2_tiesplit_r{r}.pkl", "rb"))
leg = r == 4
def hopdist(w, targets):
    dist = {t: 0 for t in targets}; fr = list(targets)
    while fr:
        nx = []
        for n2 in fr:
            for d in range(4):
                n = (n2[0] - DRC[d][0], n2[1] - DRC[d][1])
                info = w["adj"].get(n, {}).get(d)
                if info is None or not edge_ok(info, leg) or n in dist: continue
                dist[n] = dist[n2] + 1; nx.append(n)
        fr = nx
    return dist
NAMES = ["lm", "oneway", "deg3+", "npaths", "border", "lm_type_same"]
stat = collections.Counter(); n = 0
for x, a, y, reach in single:
    cb = sorted(set.intersection(*[reach[d] for d in a]))[0]
    if len(cb) > 1: continue
    w = x["w"]; tgt = cb[0]
    lmn = {p for v in w["landmarks"].values() for p in v} - {tgt}
    gtype = [k for k, v in w["landmarks"].items() if tgt in v][0]
    R = max(p[0] for p in w["adj"]); C = max(p[1] for p in w["adj"])
    dist = hopdist(w, {tgt}); memo = {}
    def feat(n, d, info):
        n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
        return [n2 in lmn, info[2] and any(True for _ in [0]) and (w["adj"].get(n2, {}).get(OPP[d]) or (0, 0, True))[2] is False,
                len(w["adj"].get(n2, {})) >= 3, 0, n2[0] in (0, R) or n2[1] in (0, C), n2 in set(w["landmarks"].get(gtype, [])) - {tgt}]
    def best(n):
        if n == tgt: return [(0, 0)] * 3 + [(1, 1)] + [(0, 0)] * 2
        if n in memo: return memo[n]
        acc = None
        for d, info in w["adj"].get(n, {}).items():
            if not edge_ok(info, leg): continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if dist.get(n2, INF) != dist[n] - 1: continue
            sub = best(n2); f = feat(n, d, info)
            v = [(fa + s[0], fa + s[1]) for fa, s in zip(f, sub)]
            v[3] = sub[3]
            if acc is None: acc = v
            else:
                acc = [(min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(acc, v)]
                acc[3] = (acc[3][0] if False else 0, 0)
        memo[n] = acc; return acc
    # số đường ngắn nhất riêng
    cnt = {}
    def npaths(n):
        if n == tgt: return 1
        if n in cnt: return cnt[n]
        s = 0
        for d, info in w["adj"].get(n, {}).items():
            if not edge_ok(info, leg): continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if dist.get(n2, INF) == dist[n] - 1: s += npaths(n2)
        cnt[n] = s; return s
    s0 = w["robot"]; F = {}
    for d in a:
        info = w["adj"][s0][d]; n2 = (s0[0] + DRC[d][0], s0[1] + DRC[d][1])
        sub = best(n2); f = feat(s0, d, info)
        v = [(fa + q[0], fa + q[1]) for fa, q in zip(f, sub)]
        v[3] = (npaths(n2), npaths(n2))
        F[d] = v
    n += 1
    for i, nm in enumerate(NAMES):
        for j, mm in enumerate(("min", "max")):
            vals = [F[d][i][j] for d in a]
            if len(set(vals)) > 1:
                stat[f"{nm}.{mm} khác"] += 1
                stat[f"{nm}.{mm} nhỏ"] += F[y][i][j] == min(vals)
                stat[f"{nm}.{mm} lớn"] += F[y][i][j] == max(vals)
print("số ca", n)
for k in sorted(stat):
    if k.endswith("khác"):
        b = k[:-5]; m = stat[k]
        print(f"  {b:16s} khác {m:4d}  chọn nhỏ nhất {stat[b+' nhỏ']/m:.3f}  lớn nhất {stat[b+' lớn']/m:.3f}")
