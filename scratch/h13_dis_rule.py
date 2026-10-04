"""Luật DIS_RULE có bắt nhầm đích / điểm ghé THẬT không? Đếm trên train + validation (có đáp án):
với mỗi lần nhắc địa điểm đã biết, luật đánh dấu 'gây nhiễu' hay không, so với vai trò thật.
NEG_SENT (luật viết riêng cho 6 kiểu câu gây nhiễu của train) được TẮT để xem riêng sức của luật tổng quát."""
import sys, re, pickle
sys.path.insert(0, "src")
from collections import Counter
from common import *
import nlp2
from nlp2 import *
p = pickle.load(open("outputs/nlp2_final.pkl", "rb"))
c = Counter()
for sp in ("train", "validation"):
    for s in load_split(sp)[2]:
        m = s["mission"]
        anchors = {m[k]["anchor"] for k in ("goal_ref", "via_ref") if m[k] and m[k]["anchor"]}
        for me in p._mentions(m["text"], known_only=True)[0]:
            t = me["type"]
            role = "đích" if t == m["goal"] else ("điểm ghé" if t == m["via"] else ("mốc" if t in anchors else "gây nhiễu"))
            st = me["sent"]
            mark = len(me["ms"]) == 1 and not re.search(ORD_CUE, st) and not re.search(VIA_VERB, st) and not re.search(DOUBLE_NEG, st) \
                and bool(re.search(GEN_NEG, st) or re.search(PAST_CUE, st))
            c[f"{role:9s} -> luật đánh dấu gây nhiễu: {mark}"] += 1
for k, v in sorted(c.items()): print(f"   {v:5d}  {k}")
