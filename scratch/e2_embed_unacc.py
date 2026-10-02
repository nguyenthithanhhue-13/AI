"""Như e1 nhưng huấn luyện bằng cả dạng có dấu lẫn không dấu -> đo trên dạng không dấu của tên gọi chỉ có ở validation."""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scratch")
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
from nlp import mine_lexicon, norm, PLACE_TYPES
from embed import Embedder
import importlib

tr = [s["mission"] for s in load_split("train")[2]]; va = [s["mission"] for s in load_split("validation")[2]]
E = Embedder()
import re
from collections import Counter, defaultdict


def accented_forms(missions, phrases):
    out = defaultdict(Counter); pset = set(phrases); ks = {len(p.split()) for p in phrases}
    for m in missions:
        acc = re.findall(r"\w+", m["text"].lower()); un = [norm(t) for t in acc]
        for k in ks:
            for i in range(len(un) - k + 1):
                key = " ".join(un[i:i + k])
                if key in pset: out[key][" ".join(acc[i:i + k])] += 1
    return {p: (max(out[p], key=lambda s: (s != p, out[p][s])) if out[p] else p) for p in phrases}


lex_tr = mine_lexicon(tr); lex_all = mine_lexicon(tr + va)
new = {a: t for a, t in lex_all.items() if a not in lex_tr}
acc_tr = accented_forms(tr, list(lex_tr)); acc_new = accented_forms(va, list(new))
canon = {"thư viện": "library", "ký túc xá": "dorm", "nhà thể thao": "sports", "trạm y tế": "clinic", "căn tin": "canteen",
         "bãi xe": "parking", "giảng đường": "lecture", "phòng thí nghiệm": "lab", "phòng hành chính": "office", "cổng trường": "gate"}
X = [acc_tr[a] for a in lex_tr] + list(lex_tr) + list(canon) + [norm(c) for c in canon]
y = [PLACE_TYPES.index(t) for t in lex_tr.values()] * 2 + [PLACE_TYPES.index(t) for t in canon.values()] * 2
names = list(new)
for C in (5, 50):
    clf = LogisticRegression(C=C, max_iter=3000).fit(E.encode(X), y)
    for variant, texts in (("có dấu", [acc_new[a] for a in names]), ("không dấu", names)):
        P = clf.predict_proba(E.encode(texts))
        ok = sum(PLACE_TYPES[clf.classes_[int(p.argmax())]] == new[a] for a, p in zip(names, P))
        top3 = sum(PLACE_TYPES.index(new[a]) in clf.classes_[np.argsort(-p)[:3]] for a, p in zip(names, P))
        print(f"C={C} [{variant}] đúng {ok}/{len(names)}, trong top-3: {top3}/{len(names)}")
E.save()
