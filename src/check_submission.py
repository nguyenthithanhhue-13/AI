"""Tự kiểm tra outputs/predictions.json trước khi nộp, đếm tham số mô hình, và in vài bộ đếm "sức khỏe" của lượt chạy.

    python src/check_submission.py [cv_mode=final]

Các bộ đếm chỉ là SỐ ĐẾM tổng hợp do chính pipeline sinh ra (không in câu hay ảnh test, không dùng để huấn luyện).
So sánh test với validation cho biết test có "lạ" hơn dữ liệu đã học hay không.
"""
import json
import pickle
import sys
from collections import Counter

from common import *
from nlp2 import resolve_with_map
from strategy import INF, q_values

mode = sys.argv[1] if len(sys.argv) > 1 else "final"
pred_path = OUT / "predictions.json"
preds = json.loads(pred_path.read_text(encoding="utf-8"))
rows = load_json(DATA / "test" / "observations.json")
sample = load_json(DATA / "test" / "sample_submission.json")

ok = True


def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  LỖI  ") + msg)
    ok &= bool(cond)


print("Kiểm tra", pred_path)
check(isinstance(preds, list), "là một danh sách JSON")
check(len(preds) == len(rows) == len(sample), f"độ dài {len(preds)} = số dòng test {len(rows)} = sample {len(sample)}")
check(all(type(p) is int for p in preds), "mọi phần tử là số nguyên (không phải chuỗi / số thực)")
check(all(0 <= p <= 3 for p in preds), "mọi giá trị nằm trong 0..3")
check(type(sample[0]) is type(preds[0]), "cùng kiểu phần tử với sample_submission.json")
check(all(rows[i]["robot_id"] == i % 10 for i in range(len(rows))), "thứ tự dòng test: mỗi cảnh 10 dòng, robot 0..9")
print("  phân bố:", sorted(Counter(preds).items()))


def health(split):
    from pipeline import cv_split, parse_split      # dùng lại cache; thiếu thì tự chạy
    rws = load_json(DATA / split / "observations.json")
    worlds = cv_split(split, mode)
    from pipeline import presents_of
    missions = parse_split(split, "final", presents_of(split, worlds), "_" + mode)
    n = len(rws) // 10
    st = Counter()
    for si in range(n):
        w = worlds[rws[si * 10]["image"]]["world"]
        m = missions[si]
        if w is None:
            st["CV lỗi (đoán mặc định)"] += 1
            continue
        st["đích là tên gọi ĐÃ BIẾT"] += bool(m["goal_known"])
        st["có điểm ghé"] += m["via"] is not None
        st["  trong đó điểm ghé là tên gọi đã biết"] += bool(m["via"] is not None and m["via_known"])
        st["có tham chiếu không gian cho đích"] += m["goal_ref"] is not None
        st["gấp"] += bool(m["urgent"]); st["dễ vỡ"] += bool(m["fragile"])
        r = resolve_with_map(m, w["landmarks"], w)
        st["robot 0 không tìm được đường"] += min(q_values(w, r, 0)) == INF
        st["chú giải đọc bằng chữ + bố cục"] += worlds[rws[si * 10]["image"]]["info"]["legend_how"] == "chữ + bố cục"
    return n, st


print("\nBộ đếm sức khỏe (tỉ lệ trên số cảnh):")
res = {}
for split in ("validation", "test"):
    try:
        res[split] = health(split)
    except FileNotFoundError as e:
        print("  (thiếu cache cho", split, "->", e.filename, ")")
keys = list(next(iter(res.values()))[1]) if res else []
for k in keys:
    print(f"  {k:42s} " + "   ".join(f"{sp}: {st[k] / n:6.1%}" for sp, (n, st) in res.items()))

print("\nTham số mô hình dùng khi dự đoán:")
from mlp import MLP
total = 0
for f in sorted((OUT / f"models_{mode}").glob("*.npz")):
    n = MLP.load(f).n_params
    total += n
    print(f"   {f.name:28s} {n:>12,}")
cnn_txt = OUT / f"models_{mode}" / "node_cnn_params.txt"
if cnn_txt.exists() and (OUT / f"models_{mode}" / "node_cnn.onnx").exists():
    n = int(cnn_txt.read_text())
    total += n
    print(f"   {'node_cnn.onnx':28s} {n:>12,}")
nlp = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
clfs = [nlp.role_clf, nlp.kind_clf, nlp.phrase_clf, nlp.goal_clf, nlp.type_clf, nlp.tagger.clf]
clfs += [c for c in (getattr(nlp, "e5_type", None), getattr(nlp, "e5_phrase", None)) if c is not None]
n_nlp = sum(c.coef_.size + c.intercept_.size for c in clfs)
print(f"   {'NLP (' + str(len(clfs)) + ' hồi quy logistic)':28s} {n_nlp:>12,}")
total += n_nlp
if getattr(nlp, "use_e5", False):
    # multilingual-e5-small (pretrained công khai): 117.653.760 tham số; kiểm tra chéo bằng cỡ file float32
    e5_file = OUT / "e5_small" / "onnx" / "model.onnx"
    n_e5 = 117_653_760
    assert abs(e5_file.stat().st_size / 4 - n_e5) / n_e5 < 0.02, "cỡ file e5 không khớp số tham số"
    print(f"   {'multilingual-e5-small':28s} {n_e5:>12,}")
    total += n_e5
check(total <= 200_000_000, f"tổng tham số {total:,} <= 200.000.000")
print("\nKẾT LUẬN:", "file hợp lệ" if ok else "CÓ LỖI, chưa nộp được")
