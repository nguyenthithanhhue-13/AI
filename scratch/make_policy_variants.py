"""Sinh các bản nộp chỉ khác nhau ở MIXED_POLICY (cách xử lý câu phụ trộn 'vế gấp + vế không gấp').
Dùng lại world test đã cache và mô hình NLP cuối; không huấn luyện lại gì.
    python scratch/make_policy_variants.py calm last first urgent"""
import sys, json, pickle
sys.path.insert(0, "src")
from common import *
import nlp2
from nlp2 import resolve_with_map
from strategy import predict

rows = load_json(DATA / "test" / "observations.json")
worlds = pickle.load(open("cache/world_test_final.pkl", "rb"))
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
n = len(rows) // 10
base = json.load(open(OUT / "cac_ban_nop_cu" / "predictions_v11_lb0.9467.json"))
for pol in sys.argv[1:]:
    nlp2.MIXED_POLICY = pol
    preds, urg = [], 0
    for si in range(n):
        w = worlds[rows[si * 10]["image"]]["world"]
        m = p.parse(rows[si * 10]["mission"], None if w is None else {t for t, v in w["landmarks"].items() if v})
        urg += bool(m["urgent"])
        if w is None:
            preds += [0] * 10
            continue
        r = resolve_with_map(m, w["landmarks"], w)
        preds += [predict(w, r, k) for k in range(10)]
    r = [0] * 10
    for i, (a, b) in enumerate(zip(preds, base)):
        r[i % 10] += a != b
    out = OUT / "cac_ban_nop_cu" / f"predictions_v15_disrule_mixed_{pol}.json"
    json.dump(preds, open(out, "w"))
    print(f"{pol:7s}: gấp {urg / n:.1%} | khác bản 0.9467 theo robot 0..9: {r} -> {out.name}", flush=True)
nlp2.save_embed_cache()
