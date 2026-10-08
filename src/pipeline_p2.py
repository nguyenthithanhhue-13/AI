"""(vòng private test) Ghép CV + NLP (nlp3) + chiến thuật học từ nhãn (strat_ml).

    python src/pipeline_p2.py fit  [trainonly|final]      # học bộ đọc câu + chiến thuật (trainonly: chỉ train; final: train + validation)
    python src/pipeline_p2.py eval [cv_mode=v1final]       # validation end-to-end: CV thật, NLP + chiến thuật chỉ học train
    python src/pipeline_p2.py test [cv_mode=v1final]       # test -> outputs/private_result/predictions.json

CV dùng mô hình đã học (outputs/models_<cv_mode>/); trên validation mới: 98,3% bản đồ đúng hoàn toàn (giao lộ / đoạn đường /
địa điểm / robot / hướng / thời tiết / chú giải), nên không học lại ở vòng này.
Khi dự đoán chỉ dùng ảnh, câu yêu cầu và robot_id (không dùng scenes.json).
"""
import json
import pickle
import sys
import time
from collections import Counter

import numpy as np

from common import *
import nlp3
from strategy import resolve_targets
from strat_ml import StrategyML, move_feats, is_night

OUTP = OUT / "private_result"
OUTP.mkdir(exist_ok=True)


def legs_from_scene(s):
    w = world_from_scene(s); m = s["mission"]
    legs = []
    if m["via"]:
        legs.append([tuple(m["via_ref"]["rc"])] if m["via_ref"] else list(w["landmarks"][m["via"]]))
    legs.append([tuple(m["goal_ref"]["rc"])] if m["goal_ref"] else list(w["landmarks"][m["goal"]]))
    return w, legs


def legs_from_mission(w, m):
    legs = []
    if m.get("via"):
        c = resolve_targets(w, m["via"], m.get("via_ref"))
        if c:
            legs.append(c)
    legs.append(resolve_targets(w, m["goal"], m.get("goal_ref")))
    return legs


def train_rows(splits):
    """Đặc trưng bước đi trên thông tin ĐÚNG của các split có nhãn -> {robot: (X, y)}."""
    X = {r: [] for r in range(10)}; Y = {r: [] for r in range(10)}
    for split in splits:
        cp = CACHE / f"p2_stratfeats_{split}.pkl"
        if cp.exists():
            feats, labels = pickle.load(open(cp, "rb"))
        else:
            rows, labels, scenes = load_split(split)
            feats = []
            for s in scenes:
                w, legs = legs_from_scene(s)
                m = s["mission"]
                feats.append(move_feats(w, legs, m["urgent"], m["fragile"], bool(m["via"])))
            pickle.dump((feats, labels), open(cp, "wb"))
        for i, f in enumerate(feats):
            for r in range(10):
                F, valid = f[r == 4]
                y = labels[i * 10 + r]
                for d in range(4):
                    if valid[d]:
                        X[r].append(F[d]); Y[r].append(int(d == y))
    return {r: (np.array(X[r]), np.array(Y[r])) for r in range(10)}


def fit(mode):
    from sklearn.ensemble import HistGradientBoostingClassifier
    splits = ["train"] if mode == "trainonly" else ["train", "validation"]
    t = time.time()
    missions = [s["mission"] for sp in splits for s in load_split(sp)[2]]
    p = nlp3.MissionParser3().fit(missions, verbose=True)
    p.save(OUT / f"nlp3_{mode}.pkl")
    from nlp2 import save_embed_cache
    save_embed_cache()
    print(f"bộ đọc câu: {time.time() - t:.0f}s", flush=True)
    data = train_rows(splits)
    models = {}
    for r in range(10):
        X, y = data[r]
        models[r] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=60,
                                                   l2_regularization=1.0, random_state=0).fit(X, y)
    StrategyML(models).save(OUT / f"strategy_ml_{mode}.pkl")
    print(f"chiến thuật: {time.time() - t:.0f}s", flush=True)


def cv_worlds(split, cv_mode):
    path = CACHE / f"world_{split}_{cv_mode}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    from run_cv import run
    return run(split, cv_mode)


def predict_split(split, cv_mode, mode):
    rows = load_json(DATA / split / "observations.json")
    worlds = cv_worlds(split, cv_mode)
    p = nlp3.MissionParser3.load(OUT / f"nlp3_{mode}.pkl")
    st = StrategyML.load(OUT / f"strategy_ml_{mode}.pkl")
    hp = OUT / ("strategy_hybrid_final.json" if mode == "final" and (OUT / "strategy_hybrid_final.json").exists() else "strategy_hybrid.json")
    st.hybrid = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
    preds = []; stat = Counter()
    for si in range(len(rows) // 10):
        img = rows[si * 10]["image"]; text = rows[si * 10]["mission"]
        w = worlds[img]["world"]
        if w is None:
            preds += [0] * 10; stat["cv lỗi"] += 1
            continue
        L = w["landmarks"]
        try:
            r = p.parse(text, {t for t, v in L.items() if v}, L)
            m = nlp3.resolve3(r, L, w)
            legs = legs_from_mission(w, m)
            if not legs[-1]:
                raise ValueError("không có đích")
            night = is_night(DATA / split / img)
            stat["đêm"] += night
            mapgoal = bool(m.get("goal_ref")) and m["goal_ref"][0] == "pos"
            preds += st.predict_scene(w, legs, m["urgent"], m["fragile"], len(legs) == 2, night, mapgoal)
            stat["mô tả qua bản đồ"] += "mapref" in r and r["goal_ref"] is not None and r["goal_ref"][0] == "pos"
        except Exception as e:
            preds += [0] * 10; stat["lỗi: " + type(e).__name__] += 1
    from nlp2 import save_embed_cache
    save_embed_cache()
    return rows, preds, stat


def evaluate(cv_mode):
    rows, labels, _ = load_split("validation")
    rows, preds, stat = predict_split("validation", cv_mode, "trainonly")
    score, per = macro_accuracy(rows, labels, preds)
    print(f"validation (CV {cv_mode} + NLP và chiến thuật chỉ học train): {score:.4f}  " +
          " ".join(f"R{r}={v:.3f}" for r, v in per.items()), dict(stat))


def make_submission(cv_mode):
    rows, preds, stat = predict_split("test", cv_mode, "final")
    sample = load_json(DATA / "test" / "sample_submission.json")
    assert len(preds) == len(rows) == len(sample) == 12000, (len(preds), len(rows), len(sample))
    assert all(isinstance(x, int) and 0 <= x <= 3 for x in preds)
    (OUTP / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    print("đã ghi", OUTP / "predictions.json", "số dòng:", len(preds), "phân bố:", sorted(Counter(preds).items()), dict(stat))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2] if len(sys.argv) > 2 else "final")
    elif cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "v1final")
    elif cmd == "test":
        make_submission(sys.argv[2] if len(sys.argv) > 2 else "v1final")
