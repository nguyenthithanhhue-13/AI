"""Thí nghiệm: nếu bộ phân loại KHÔNG nhận ra khung câu 'điểm ghé' nào (giả lập khung câu lạ ở test),
luật cấu trúc + từ chỉ thứ tự có tự tìm lại được điểm ghé không?"""
import sys
sys.path.insert(0, "src")
from common import *
from nlp2 import MissionParser2
from eval_nlp2 import evaluate

_, trl, trs = load_split("train"); _, val, vas = load_split("validation")
p = MissionParser2().fit([s["mission"] for s in trs])
p.ablate_via = True
print("train (ablate via LR):"); evaluate(p, trs, [trl[i * 10:(i + 1) * 10] for i in range(len(trs))], 0)
print("validation (ablate via LR):"); evaluate(p, vas, [val[i * 10:(i + 1) * 10] for i in range(len(vas))], 8)
