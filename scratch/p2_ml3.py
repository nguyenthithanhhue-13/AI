"""Học máy chiến thuật v3: đặc trưng cũ (cache/p2_mlfeats_*.pkl) + cờ ĐÊM + Q của mô hình chi phí theo điều kiện
(outputs/strategy_hybrid.json, tham số chỉ học train) cho chính robot đó. Học train -> đo validation.
    python scratch/p2_ml3.py [robot ...]"""
import sys, json, pickle, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import SG, theta_of, lm_excl, INF
from p2_lib import load
from strat_ml import group_key
from common import CACHE, OUT
from sklearn.ensemble import HistGradientBoostingClassifier

H = json.load(open(OUT / "strategy_hybrid.json", encoding="utf-8"))
ROB = [int(a) for a in sys.argv[1:]] or list(range(10))


def extra(split):
    cp = CACHE / f"p2_ml3extra_{split}.pkl"
    if cp.exists():
        return pickle.load(open(cp, "rb"))
    D = load(split); out = []
    for x in D:
        night = x["s"]["style"] == "night"
        row = {}
        for r in range(1, 9):
            cfg = H[str(r)]
            p = cfg["params"].get(group_key(cfg["key"], night, x["w"]["rain"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])))
            if p is None:
                row[r] = np.zeros(4); continue
            q = np.array(SG(x["w"], r == 4, lm_excl(x)).q(theta_of(**p), x["legs"]))
            q = np.where(np.isfinite(q), q, 99.0)
            row[r] = np.minimum(q - q.min(), 30)
        out.append((night, row))
    pickle.dump(out, open(cp, "wb"))
    return out


data = {s: pickle.load(open(CACHE / f"p2_mlfeats_{s}.pkl", "rb")) for s in ("train", "validation")}
ex = {s: extra(s) for s in ("train", "validation")}


def rows(split, r):
    FS, Y = data[split]; E = ex[split]
    X, y, g = [], [], []
    for i, f in enumerate(FS):
        F, valid = f[r == 4]
        night, row = E[i]
        for d in range(4):
            if valid[d]:
                q = row.get(r, np.zeros(4))
                X.append(np.concatenate([F[d], [night, q[d], float(q[d] < 1e-6)]])); y.append(int(Y[i, r] == d)); g.append((i, d))
    return np.array(X), np.array(y), g


for r in ROB:
    Xt, yt, _ = rows("train", r)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(Xt, yt)
    X, y, g = rows("validation", r)
    p = clf.predict_proba(X)[:, 1]; best = {}
    for (i, d), pv in zip(g, p):
        if i not in best or pv > best[i][1]:
            best[i] = (d, pv)
    Y = data["validation"][1]
    print(f"R{r}: validation {np.mean([best[i][0] == Y[i, r] for i in range(len(Y))]):.3f}", flush=True)
