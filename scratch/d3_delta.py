"""Tạo file nộp khi máy này thiếu các mô hình CNN của bản 0,9778:
lấy file gốc (đọc ảnh bằng CNN) làm nền, chỉ THAY đáp án ở những cảnh mà thay đổi NLP làm đổi kết quả đọc câu.

  A = NLP cấu hình nền (các cờ trong OFF đặt về giá trị cũ), B = NLP cấu hình mới; cả hai học lại trên train + validation
  trên CHÍNH máy này (để khác biệt A/B chỉ do thay đổi đang thử). Bản đồ: cache/world_test_final.pkl của máy này (MLP cũ).
  Một cảnh được thay khi: (1) kết quả đọc câu sau khi chốt theo bản đồ của A khác B, và
                          (2) bản đồ cũ + A cho ra ĐÚNG 10 đáp án của file gốc (hai cách đọc ảnh đồng ý ở cảnh đó).

    python scratch/d3_delta.py <file_gốc> <file_ra> TÊN_CỜ=giá_trị_cũ [TÊN_CỜ=giá_trị_cũ ...]
"""
import sys, json, pickle
sys.path.insert(0, "src")
from collections import Counter
from common import *
import nlp2
from nlp2 import MissionParser2, resolve_with_map
from strategy import predict

base_path, out_path = sys.argv[1], sys.argv[2]
OFF = dict(a.split("=", 1) for a in sys.argv[3:])
rows = load_json(DATA / "test" / "observations.json")
worlds = pickle.load(open("cache/world_test_final.pkl", "rb"))
base = json.load(open(base_path))
missions = [s["mission"] for sp in ("train", "validation") for s in load_split(sp)[2]]
n = len(rows) // 10
KEYS = ("goal", "via", "goal_ref", "via_ref", "urgent", "fragile")


def run(flags):
    new = {k: getattr(nlp2, k) for k in flags}
    for k, v in flags.items():
        setattr(nlp2, k, eval(v))
    p = MissionParser2().fit(missions)
    res = []
    for si in range(n):
        w = worlds[rows[si * 10]["image"]]["world"]
        if w is None:
            res.append(None); continue
        m = p.parse(rows[si * 10]["mission"], {t for t, v in w["landmarks"].items() if v})
        r = resolve_with_map(m, w["landmarks"], w)
        res.append(({k: r[k] for k in KEYS}, [predict(w, r, k) for k in range(10)]))
    for k, v in new.items():
        setattr(nlp2, k, v)
    return res


A = run(OFF)
B = run({})
out = list(base)
c = Counter()
per_robot = [0] * 10
for si in range(n):
    if A[si] is None:
        continue
    agree = A[si][1] == base[si * 10:si * 10 + 10]
    c["bản đồ cũ + NLP nền khớp file gốc"] += agree
    if A[si][0] == B[si][0]:
        continue
    c["cảnh mà thay đổi NLP làm đổi kết quả đọc câu"] += 1
    for k in KEYS:
        if A[si][0][k] != B[si][0][k]:
            c[f"   đổi {k}"] += 1
    if not agree:
        c["   ...bỏ qua vì bản đồ cũ không khớp file gốc"] += 1
        continue
    if A[si][1] != B[si][1]:
        c["cảnh được THAY đáp án"] += 1
        for r in range(10):
            per_robot[r] += A[si][1][r] != B[si][1][r]
        out[si * 10:si * 10 + 10] = B[si][1]
json.dump(out, open(out_path, "w"))
print(f"{n} cảnh;", "; ".join(f"{k.strip()}: {v}" for k, v in c.items()))
print(f"file ra khác file gốc {sum(a != b for a, b in zip(out, base))} đáp án, theo robot 0..9: {per_robot} -> {out_path}")
nlp2.save_embed_cache()
