"""Phân tích lỗi NLP khi gặp cách nói lạ: học train, đo validation. Lỗi do đâu (thiếu tên gọi / khung câu / cụm từ)?"""
import sys
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from nlp import *

trs = load_split("train")[2]; vas = load_split("validation")[2]
tr = [s["mission"] for s in trs]; va = [s["mission"] for s in vas]
p = MissionParser().fit(tr)
full = Lexicon(mine_lexicon(tr + va), {})
c = Counter()
for s in vas:
    m = s["mission"]; res = p.parse(m["text"])
    present = {l["type"] for l in s["landmarks"]}
    ments, plain = p._mentions(m["text"])
    fm = [x for sent in sentences(m["text"]) for x in full.find(tokens(sent))]
    known_goal_alias = any(t == m["goal"] for (_, _, _, _, t, _) in ments)
    c["goal ok"] += res["goal"] == m["goal"]
    c["goal alias known to train lexicon"] += known_goal_alias
    c["goal ok | alias known"] += known_goal_alias and res["goal"] == m["goal"]
    c["goal ok | alias unknown"] += (not known_goal_alias) and res["goal"] == m["goal"]
    # bộ phân loại cả câu
    gp = res["goal_probs"]
    c["textclf argmax ok"] += max(gp, key=gp.get) == m["goal"]
    c["textclf argmax ok (restricted to map types)"] += max(present, key=lambda t: gp.get(t, 0)) == m["goal"]
    taken = {t for (_, _, _, _, t, _) in ments if t != m["goal"]}
    cand = [t for t in present if t not in taken] or list(present)
    c["textclf ok (map types minus other known mentions) | alias unknown"] += (not known_goal_alias) and max(cand, key=lambda t: gp.get(t, 0)) == m["goal"]
    c["via ok"] += res["via"] == m["via"]
    c["urgent ok"] += res["urgent"] == m["urgent"]; c["fragile ok"] += res["fragile"] == m["fragile"]
    c["mentions found (train lex)"] += len(ments); c["mentions found (full lex)"] += len(fm)
n = len(vas)
for k, v in c.items(): print(f"{k:70s} {v:5d}  ({v / n:.3f})")
