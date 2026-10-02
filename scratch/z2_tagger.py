"""Bộ gán nhãn cụm địa điểm: học trên train, đo trên validation (đặc biệt với tên gọi chỉ có ở validation)."""
import sys
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from nlp import *
from nlp_tagger import *

tr = [s["mission"] for s in load_split("train")[2]]; va = [s["mission"] for s in load_split("validation")[2]]
lex_tr = Lexicon(mine_lexicon(tr), ngram_counts(tr))
lex_full = Lexicon(mine_lexicon(tr + va), ngram_counts(tr + va))


def spans_in(sent, lex):
    """Khoảng (start, end) theo chỉ số tag_tokens của các tên địa điểm khớp từ điển."""
    tt = tag_tokens(sent)
    idx = [i for i, t in enumerate(tt) if t not in (",", ":")]
    words = [tt[i] for i in idx]
    return tt, [(idx[a], idx[b - 1] + 1) for (a, b, t, al) in lex.find(words)]


S, SP = [], []
for m in tr:
    for sent in sentences(m["text"]):
        tt, sp = spans_in(sent, lex_tr)
        S.append(tt); SP.append(sp)
tg = SpanTagger().fit(S, SP)
print("train sentences", len(S), "vocab", len(tg.vocab))
c = Counter(); ex = []
for m in va:
    for sent in sentences(m["text"]):
        tt, known = spans_in(sent, lex_tr)
        _, full = spans_in(sent, lex_full)
        new = [s for s in full if s not in known]
        pred = tg.spans(tt, known)
        for (a, b) in new:
            c["new alias spans"] += 1
            hit = [p for p in pred if p[0] < b and p[1] > a]
            c["new alias: detected (overlap)"] += bool(hit)
            c["new alias: exact boundaries"] += (a, b) in pred
            if not hit and len(ex) < 12: ex.append(("MISS", " ".join(tt), " ".join(tt[a:b])))
        for (a, b) in pred:
            c["predicted spans"] += 1
            if not any(a < fb and b > fa for fa, fb in full):
                c["predicted span is not a place"] += 1
                if len(ex) < 30: ex.append(("FALSE", " ".join(tt), " ".join(tt[a:b])))
for k, v in c.items(): print(f"{k:40s} {v}")
for e in ex: print("  ", e)
