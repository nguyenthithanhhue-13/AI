"""Ghép CV + NLP + chiến thuật.

    python src/pipeline.py eval  [cv_mode=dev]        # đo trên validation, không dùng scenes.json khi dự đoán
    python src/pipeline.py test  [cv_mode=final]      # sinh outputs/predictions.json

Mỗi cảnh (1 ảnh + 1 mission) chỉ xử lý MỘT lần rồi dùng lại cho cả 10 robot; kết quả CV và NLP lưu cache.
NLP dùng nlp2.py (đọc được cả cách nói lạ); loại địa điểm lạ được chốt bằng những gì CV thấy trên bản đồ.
"""
import json
import pickle
import sys

from common import *
from nlp2 import MissionParser2, resolve_with_map
from strategy import predict


# ---------------- NLP ----------------
def train_missions():
    tr = [s["mission"] for s in load_split("train")[2]]
    va = [s["mission"] for s in load_split("validation")[2]]
    return tr, va


def parse_split(split, nlp_mode):
    """nlp_mode: 'trainonly' (mô phỏng gặp cách nói hoàn toàn mới)
                 | 'cv5' (chỉ cho validation: mỗi câu được đọc bởi mô hình không học câu đó)
                 | 'final' (học train + validation; dùng cho test)."""
    path = CACHE / f"nlp2_{split}_{nlp_mode}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    rows = load_json(DATA / split / "observations.json")
    texts = [rows[i]["mission"] for i in range(0, len(rows), 10)]
    tr, va = train_missions()
    if nlp_mode == "trainonly":
        p = MissionParser2().fit(tr)
        out = [p.parse(t) for t in texts]
    elif nlp_mode == "final":
        mp = OUT / "nlp2_final.pkl"
        if mp.exists():
            p = MissionParser2.load(mp)
        else:
            p = MissionParser2().fit(tr + va)
            p.save(mp)
        out = [p.parse(t) for t in texts]
    elif nlp_mode == "cv5":
        assert split == "validation"
        out = [None] * len(texts)
        for k in range(5):
            rest = [m for i, m in enumerate(va) if i % 5 != k]
            p = MissionParser2().fit(tr + rest)
            for i in range(k, len(texts), 5):
                out[i] = p.parse(texts[i])
    pickle.dump(out, open(path, "wb"))
    from nlp2 import save_embed_cache
    save_embed_cache()
    return out


# ---------------- CV ----------------
def cv_split(split, cv_mode):
    path = CACHE / f"world_{split}_{cv_mode}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    from run_cv import run
    return run(split, cv_mode)


# ---------------- ghép ----------------
def predict_scene(world, mission, resolve=True):
    """mission: kết quả MissionParser2.parse (resolve=True) hoặc mission đúng từ scenes.json (resolve=False)."""
    if world is None:
        return [0] * 10
    m = resolve_with_map(mission, world["landmarks"]) if resolve else mission
    out = []
    for r in range(10):
        try:
            out.append(int(predict(world, m, r)))
        except Exception:
            out.append(0)
    return out


def run_split(split, cv_mode, nlp_mode):
    rows = load_json(DATA / split / "observations.json")
    worlds = cv_split(split, cv_mode)
    missions = parse_split(split, nlp_mode)
    preds = []
    for si in range(len(rows) // 10):
        img = rows[si * 10]["image"]
        preds += predict_scene(worlds[img]["world"], missions[si])
    return rows, preds


def evaluate(cv_mode="dev"):
    rows, labels, scenes = load_split("validation")
    worlds = cv_split("validation", cv_mode)
    print("=== validation: macro accuracy theo từng cấu hình (để biết khâu nào kéo điểm xuống) ===")
    for name, use_cv, nlp_mode in [("bản đồ ĐÚNG + mission ĐÚNG (trần)", False, None),
                                   ("bản đồ ĐÚNG + NLP 5-fold", False, "cv5"),
                                   ("bản đồ ĐÚNG + NLP chỉ học train (cách nói lạ)", False, "trainonly"),
                                   ("CV + mission ĐÚNG", True, None),
                                   ("CV + NLP 5-fold (cách nói đã quen)", True, "cv5"),
                                   ("CV + NLP chỉ học train (cách nói lạ)", True, "trainonly")]:
        missions = parse_split("validation", nlp_mode) if nlp_mode else [mission_from_scene(s) for s in scenes]
        preds = []
        for si, s in enumerate(scenes):
            w = worlds[s["image"]]["world"] if use_cv else world_from_scene(s)
            preds += predict_scene(w, missions[si], resolve=nlp_mode is not None)
        score, per = macro_accuracy(rows, labels, preds)
        print(f"{name:48s} {score:.4f}  " + " ".join(f"R{r}={v:.3f}" for r, v in per.items()))


def make_submission(cv_mode="final"):
    rows, preds = run_split("test", cv_mode, "final")
    sample = load_json(DATA / "test" / "sample_submission.json")
    assert len(preds) == len(rows) == len(sample), (len(preds), len(rows), len(sample))
    assert all(isinstance(p, int) and 0 <= p <= 3 for p in preds)
    (OUT / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    from collections import Counter
    print("đã ghi", OUT / "predictions.json", "số dòng:", len(preds), "phân bố:", sorted(Counter(preds).items()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "dev")
    elif cmd == "test":
        make_submission(sys.argv[2] if len(sys.argv) > 2 else "final")
