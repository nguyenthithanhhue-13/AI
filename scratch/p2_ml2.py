"""So các cấu hình bộ học chiến thuật trên đặc trưng đã lưu (cache/p2_mlfeats_*.pkl).
  A: chỉ train -> validation;  B: train + 4/5 validation -> 1/5 còn lại (5 phần, ước lượng sát test vì bản cuối học cả validation)."""
import sys, pickle, numpy as np
sys.path.insert(0, "src")
from common import CACHE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

tr = pickle.load(open(CACHE / "p2_mlfeats_train.pkl", "rb"))
va = pickle.load(open(CACHE / "p2_mlfeats_validation.pkl", "rb"))


def rows(FS, Y, r, idx):
    X, y, g = [], [], []
    for i in idx:
        F, valid = FS[i][r == 4]
        for d in range(4):
            if valid[d]:
                X.append(F[d]); y.append(int(Y[i, r] == d)); g.append((i, d))
    return np.array(X), np.array(y), g


def acc(clf, FS, Y, r, idx):
    X, y, g = rows(FS, Y, r, idx)
    p = clf.predict_proba(X)[:, 1]
    best = {}
    for (i, d), pv in zip(g, p):
        if i not in best or pv > best[i][1]:
            best[i] = (d, pv)
    return np.mean([best[i][0] == Y[i, r] for i in idx])


CFG = {
    "hgb_reg": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=60, l2_regularization=1.0, random_state=0),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0),
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
}
which = sys.argv[1:] or list(CFG)
for name in which:
    A, B = [], []
    for r in range(10):
        Xt, yt, _ = rows(*tr, r, range(len(tr[0])))
        clf = CFG[name]().fit(Xt, yt)
        A.append(acc(clf, *va, r, range(len(va[0]))))
        folds = []
        nv = len(va[0])
        for k in range(5):
            te = [i for i in range(nv) if i % 5 == k]; trv = [i for i in range(nv) if i % 5 != k]
            Xv, yv, _ = rows(*va, r, trv)
            c2 = CFG[name]().fit(np.vstack([Xt, Xv, Xv]), np.concatenate([yt, yv, yv]))   # validation nhân đôi trọng số (giống test hơn)
            folds.append(acc(c2, *va, r, te) * len(te))
        B.append(sum(folds) / nv)
        print(f"{name} R{r}: chỉ train {A[-1]:.3f} | train+val 5 phần {B[-1]:.3f}", flush=True)
    print(f"{name}: macro chỉ train {np.mean(A):.4f} | train+val 5 phần {np.mean(B):.4f}", flush=True)
