"""Đoán loại của một địa điểm chỉ từ NGỮ CẢNH quanh nó (món hàng, người nhận...), không nhìn tên địa điểm.
Học trên train, đo trên validation ở những lần nhắc đích / điểm ghé mà tên gọi KHÔNG có trong từ điển train."""
import sys
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from common import *
from nlp import PLACE_TYPES
from nlp2 import MissionParser2, embedder

trs = load_split("train")[2]; vas = load_split("validation")[2]
tr = [s["mission"] for s in trs]; va = [s["mission"] for s in vas]
p_tr = MissionParser2(); p_tr.use_e5 = False; p_tr.fit(tr)
p_all = MissionParser2(); p_all.use_e5 = False; p_all.fit(tr + va)
E = embedder()


def ctx_text(me, before=7, after=5):
    acc, i, j, ms = me["acc"], me["i"], me["j"], me["ms"]
    lo = max([m[1] for m in ms if m[1] <= i] + [i - before, 0])
    hi = min([m[0] for m in ms if m[0] >= j] + [j + after, len(acc)])
    return " ".join(acc[lo:i]) + " ___ " + " ".join(acc[j:hi])


def collect(parser, missions, only_unknown_to=None):
    X, y = [], []
    for m in missions:
        ments, _, _ = parser._mentions(m["text"], known_only=True)
        for me in ments:
            if me["type"] not in (m["goal"], m["via"]): continue
            if only_unknown_to is not None and me["text"] in only_unknown_to.lex.alias2type: continue
            X.append(ctx_text(me)); y.append(PLACE_TYPES.index(me["type"]))
    return X, y


Xtr, ytr = collect(p_tr, tr)
Xva, yva = collect(p_all, va, only_unknown_to=p_tr)
print("train contexts", len(Xtr), "| validation contexts of unknown aliases", len(Xva))
print("ví dụ:", Xva[:4])
clf = LogisticRegression(C=20, max_iter=3000).fit(E.encode(Xtr), ytr)
P = clf.predict_proba(E.encode(Xva))
top1 = np.mean(clf.classes_[P.argmax(1)] == np.array(yva))
top3 = np.mean([y in clf.classes_[np.argsort(-p)[:3]] for p, y in zip(P, yva)])
print(f"chỉ dùng ngữ cảnh: đúng {top1:.3f}, trong top-3 {top3:.3f}  (đoán bừa = 0.10)")
E.save()
