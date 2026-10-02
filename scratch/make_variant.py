"""Sinh một biến thể bài nộp: giống bản chính nhưng TẮT luật "từ phủ định chung" cho gấp/dễ vỡ.
Dùng để so trên bảng xếp hạng xem luật đó có lợi hay hại."""
import sys, json, pickle
sys.path.insert(0, "src")
from common import *
from nlp2 import MissionParser2, save_embed_cache
from pipeline import predict_scene, cv_split

p = MissionParser2.load(OUT / "nlp2_final.pkl")
p.no_gen_neg = True
rows = load_json(DATA / "test" / "observations.json")
worlds = cv_split("test", "final")
preds = []
for si in range(len(rows) // 10):
    preds += predict_scene(worlds[rows[si * 10]["image"]]["world"], p.parse(rows[si * 10]["mission"]))
save_embed_cache()
assert len(preds) == 12000 and all(type(x) is int and 0 <= x <= 3 for x in preds)
(OUT / "predictions_v3b_no_generic_negation.json").write_text(json.dumps(preds), encoding="utf-8")
a = json.loads((OUT / "predictions.json").read_text())
print("đã ghi; khác bản chính", sum(x != y for x, y in zip(a, preds)), "dòng")
