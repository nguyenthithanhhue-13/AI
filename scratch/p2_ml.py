"""Bản nền HỌC MÁY cho chiến thuật (vòng private): mỗi robot một bộ HistGradientBoosting trên các bước đi khả dĩ,
học trên train (thông tin đúng), đo trên validation (thông tin đúng).  python scratch/p2_ml.py"""
import sys, time, pickle, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import load, macro_accuracy
from p2_ml_feats import feats, FEAT_NAMES
from sklearn.ensemble import HistGradientBoostingClassifier
from common import CACHE

t = time.time()
data = {}
for split in ("train", "validation"):
    path = CACHE / f"p2_mlfeats_{split}.pkl"
    if path.exists():
        data[split] = pickle.load(open(path, "rb"))
    else:
        D = load(split)
        data[split] = ([feats(x) for x in D], np.array([x["y"] for x in D]))
        pickle.dump(data[split], open(path, "wb"))
    print(split, f"{time.time()-t:.0f}s", flush=True)


def rows(FS, Y, r):
    X, y, grp = [], [], []
    for i, f in enumerate(FS):
        F, valid = f[r == 4]
        for d in range(4):
            if valid[d]:
                X.append(F[d]); y.append(int(Y[i, r] == d)); grp.append((i, d))
    return np.array(X), np.array(y), grp


models = {}
res = {}
for r in range(10):
    Xtr, ytr, _ = rows(*data["train"], r)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(Xtr, ytr)
    models[r] = clf
    for split in ("train", "validation"):
        FS, Y = data[split]
        X, y, grp = rows(FS, Y, r)
        p = clf.predict_proba(X)[:, 1]
        best = {}
        for (i, d), pv in zip(grp, p):
            if i not in best or pv > best[i][1]:
                best[i] = (d, pv)
        acc = np.mean([best.get(i, (0, 0))[0] == Y[i, r] for i in range(len(FS))])
        res[(r, split)] = acc
    print(f"R{r}: train {res[(r, 'train')]:.4f}  validation {res[(r, 'validation')]:.4f}", flush=True)
print("macro train", np.mean([res[(r, 'train')] for r in range(10)]), "validation", np.mean([res[(r, 'validation')] for r in range(10)]))
pickle.dump(models, open(CACHE / "p2_ml_models.pkl", "wb"))
