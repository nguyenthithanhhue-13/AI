"""Giá trị thật của mô hình nghĩa e5 với cách nói lạ: học train, đo validation, TẮT các luật từ khóa viết tay
(vì luật được viết sau khi xem validation nên không thể dùng để đo trung thực)."""
import sys
sys.path.insert(0, "src")
from common import *
from nlp2 import MissionParser2
from eval_nlp2 import evaluate

_, trl, trs = load_split("train"); _, val, vas = load_split("validation")
Yv = [val[i * 10:(i + 1) * 10] for i in range(len(vas))]
tr = [s["mission"] for s in trs]
p = MissionParser2(); p.use_e5 = False; p.fit(tr); p.ablate_rules = True
print("không e5, không luật"); evaluate(p, vas, Yv, 0)
p = MissionParser2(); p.use_e5 = True; p.use_knowledge = False; p.fit(tr); p.ablate_rules = True
for w in (0.0, 0.5, 1.0):
    p.ctx_weight = w
    print(f"e5 chỉ học dữ liệu train, không luật, trọng số ngữ cảnh = {w}"); evaluate(p, vas, Yv, 0)
p = MissionParser2(); p.use_e5 = True; p.use_knowledge = True; p.fit(tr); p.ablate_rules = True
for w in (0.0, 0.5):
    p.ctx_weight = w
    print(f"e5 + ví dụ viết tay, không luật, trọng số ngữ cảnh = {w}"); evaluate(p, vas, Yv, 0)
