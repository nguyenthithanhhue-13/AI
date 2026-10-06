"""Yêu cầu được gắn gấp / dễ vỡ CHỈ nhờ mô hình nghĩa (không có từ khóa, không có câu đã học). Chỉ đếm."""
import sys, pickle, re
sys.path.insert(0, "src")
from collections import Counter
import numpy as np
from common import *
import nlp2
from nlp2 import *
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
for sp in ("validation", "test"):
    rows = load_json(DATA / sp / "observations.json")
    c = Counter(); n = len(rows) // 10
    for si in range(n):
        t = rows[si * 10]["mission"]
        r = p.parse(t); ev = list(p._ev)
        ments, plain, plain_acc = p._mentions(t)
        sent_u = any(re.search(URGENT_POS, disambiguate(me["sent"], " ".join(me["acc"]))) and not re.search(URGENT_NEG, me["sent"]) for me in ments)
        sent_f = any(re.search(FRAGILE_POS, disambiguate(me["sent"], " ".join(me["acc"]))) and not re.search(FRAGILE_NEG, me["sent"]) for me in ments)
        for name, key, i, sent in (("GẤP", "urgent", 0, sent_u), ("DỄ VỠ", "fragile", 1, sent_f)):
            if not r[key]: continue
            strong = any(e[i] for e in ev) or sent
            c[f"{name}: có từ khóa / câu đã học" if strong else f"{name}: CHỈ mô hình nghĩa"] += 1
            if not strong:
                # độ chắc của mô hình nghĩa ở câu quyết định
                best = 0
                for s, acc in zip(plain, plain_acc):
                    if s in p.phrase: continue
                    pe = p.e5_phrase.predict_proba(embedder().encode([acc]))[0]
                    cl = list(p.e5_phrase.classes_); best = max(best, pe[cl.index(1 if i == 0 else 2)])
                c[f"    {name} chỉ mô hình nghĩa, độ chắc {'>0.9' if best > 0.9 else ('0.75-0.9' if best > 0.75 else '<0.75')}"] += 1
    print(f"\n== {sp}: {n}")
    for k in sorted(c): print(f"   {c[k]:4d} ({c[k] / n:5.1%})  {k}")
nlp2.save_embed_cache()
