"""Học bộ phân loại "ml4" cho các robot chỉ định: đặc trưng bước đi (strat_ml.move_feats) + cờ đêm + chi phí của mô hình lai
(chính robot đó, theo nhóm điều kiện có loại nơi giao) + loại nơi giao (one-hot).
    python scratch/p2_ml4_train.py trainonly|final 5 7     -> outputs/strategy_ml4_<mode>.pkl"""
import sys, json, pickle, numpy as np
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_lib import load
from common import CACHE, OUT
from strat_ml import ml4_row
from sklearn.ensemble import HistGradientBoostingClassifier

mode = sys.argv[1]; ROB = [int(a) for a in sys.argv[2:]]
H = json.load(open(OUT / ("strategy_hybrid.json" if mode == "trainonly" else "strategy_hybrid_final.json"), encoding="utf-8"))
splits = ["train"] if mode == "trainonly" else ["train", "validation"]
models = {}
for r in ROB:
    X, y = [], []
    for sp in splits:
        FS, Y = pickle.load(open(CACHE / f"p2_mlfeats_{sp}.pkl", "rb"))
        for x, f in zip(load(sp), FS):
            rows, valid = ml4_row(x["w"], x["legs"], r, H[str(r)].get("qcfg", H[str(r)]), f, x["s"]["style"] == "night", x["m"]["urgent"], x["m"]["fragile"])
            for d in range(4):
                if valid[d]:
                    X.append(rows[d]); y.append(int(x["y"][r] == d))
    models[r] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=0).fit(np.array(X), np.array(y))
    print("đã học R", r, len(y), flush=True)
pickle.dump(models, open(OUT / f"strategy_ml4_{mode}.pkl", "wb"))
