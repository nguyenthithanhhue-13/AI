"""R9 DÒ CHÍNH XÁC luật tham lam gọn: điểm bước = khoảng cách (Manhattan hoặc Euclid lưới, chọn theo nhóm) + a*đông + c*mái che
+ b*vào địa điểm + b_k*vào loại k + rẽ R/L/B; hòa (|chênh| < 1e-9) -> thứ tự kiểu rẽ. Nhóm: KEY (n đêm, u gấp, f dễ vỡ, v ghé, w mưa).
Dò tọa độ nhiều vòng từ nhiều điểm xuất phát trên train, đo validation (thông tin đúng). Đặc trưng lấy từ scratch/p2_greedy.py.
    python scratch/p2_greedy_exact.py [key=nuf] [restarts]"""
import sys, os, itertools, collections, json, numpy as np
KEY = sys.argv[1] if len(sys.argv) > 1 else "nuf"; NRS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
sys.argv = ["x", "9"]
g = {"__name__": "x"}
exec(open("scratch/p2_greedy.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0], g)
load = g["load"]; feats = g["feats"]; PL = g["PL"]
RELS = "SRLB"; ORDERS = ["".join(p) for p in itertools.permutations(RELS)] + ["A" + "".join(p) for p in itertools.permutations("0123")]
# cột đặc trưng (p2_greedy.feats): 0 man 1 euc 2 bfs 3 px | 4..7 hiệu | 8 crowd 9 cover 10 normal 11 stairs 12 lm 13..22 loại | 23..26 S R L B


def prep(split):
    D = load(split); X = np.full((len(D), 4, 27), np.nan); keys = []; Y = []
    for i, x in enumerate(D):
        xy = {tuple(n["rc"]): tuple(n["xy"]) for n in x["s"]["nodes"]}
        for d, v in feats(x["w"], x["legs"], xy).items():
            X[i, d] = v
        m = x["m"]; k = []
        for ch in KEY:
            k.append({"n": x["s"]["style"] == "night", "u": m["urgent"], "f": m["fragile"], "v": bool(m["via"]), "w": x["w"]["rain"]}[ch])
        keys.append(tuple(int(b) for b in k)); Y.append(x["y"][9])
    return X, keys, np.array(Y)


def score(X, p):
    """p: dict(dist=0|1, a, c, b, bt[10], R, L, B, order) -> dự đoán [n]."""
    s = X[:, :, p["dist"]] + p["a"] * X[:, :, 8] + p["c"] * X[:, :, 9] + p["b"] * X[:, :, 12] + X[:, :, 13:23] @ np.array(p["bt"]) + \
        X[:, :, 24] * p["R"] + X[:, :, 25] * p["L"] + X[:, :, 26] * p["B"]
    s = np.where(np.isnan(s), np.inf, s)
    if p["order"].startswith("A"):       # thứ tự hướng TUYỆT ĐỐI, vd "A0312" = LÊN, PHẢI, XUỐNG, TRÁI (0 U, 1 D, 2 L, 3 R)
        ab = [int(c) for c in p["order"][1:]]
        s = s + np.array([ab.index(d) for d in range(4)])[None, :] * 1e-7
        return s.argmin(1)
    tie = np.array([[p["order"].index(c) for c in RELS]]) * 1e-7      # rẽ của bước d: cột 23..26 one-hot
    rel = np.nan_to_num(X[:, :, 23:27]).argmax(-1)
    s = s + np.take(tie[0], rel)
    return s.argmin(1)


VALS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]


def fit(X, y, rng):
    best = None
    for rs in range(NRS):
        p = dict(dist=int(rng.integers(2)), a=float(rng.choice(VALS)), c=float(rng.choice([0, -0.25, -0.5, 0.25])), b=float(rng.choice(VALS)),
                 bt=[0.0] * 10, R=0.0, L=0.0, B=float(rng.choice([0, 0.25, 0.5])), order=ORDERS[rng.integers(48)])
        acc = (score(X, p) == y).sum()
        for it in range(5):
            ch = False
            for k in ["dist", "a", "c", "b", "R", "L", "B", "order"] + [("bt", j) for j in range(10)]:
                if k == "dist":
                    cand = [0, 1]
                elif k == "order":
                    cand = ORDERS
                elif k == "c":
                    cand = [-1, -0.5, -0.25, -0.1] + VALS
                else:
                    cand = VALS + ([-0.25, -0.5] if k in ("R", "L") else [])
                for v in cand:
                    q = dict(p, bt=list(p["bt"]))
                    if isinstance(k, tuple):
                        q["bt"][k[1]] = v
                    else:
                        q[k] = v
                    a2 = (score(X, q) == y).sum()
                    if a2 > acc:
                        acc, p, ch = a2, q, True
            if not ch:
                break
        if best is None or acc > best[0]:
            best = (acc, p)
    return best


if __name__ == "__main__":
    XT, KT, YT = prep("train"); XV, KV, YV = prep("validation")
    rng = np.random.default_rng(0)
    tt = tv = 0; out = {}
    for k in sorted(set(KT)):
        it = [i for i, kk in enumerate(KT) if kk == k]; iv = [i for i, kk in enumerate(KV) if kk == k]
        acc, p = fit(XT[it], YT[it], rng)
        av = (score(XV[iv], p) == YV[iv]).sum() if iv else 0
        tt += acc; tv += av; out[str(k)] = p
        bt = " ".join(f"{PL[j][:4]}={v:g}" for j, v in enumerate(p["bt"]) if v)
        print(f"  {KEY}={k}: train {acc}/{len(it)} val {av}/{len(iv)} | {'euc' if p['dist'] else 'man'} a={p['a']} c={p['c']} b={p['b']} R={p['R']} L={p['L']} B={p['B']} {p['order']} | {bt}", flush=True)
    print(f"R9 luật gọn key={KEY}: train {tt / len(YT):.3f} validation {tv / len(YV):.3f}")
    json.dump(out, open(f"cache/p2_greedy_exact_{KEY}.json", "w"), ensure_ascii=False)
