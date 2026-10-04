"""Đo khả năng nhận câu gấp / không gấp / dễ vỡ / không dễ vỡ / trung tính CHƯA TỪNG GẶP, dùng các ví dụ viết tay mới
(PHRASES_MORE) làm câu lạ. Quyết định cuối cùng lấy từ parser._phrase_flags (luật từ khóa + mô hình nghĩa), đúng như khi dự đoán.
  (A) cấu hình cũ: chỉ học PHRASES cũ, chấm trên TOÀN BỘ PHRASES_MORE
  (B) cấu hình mới: học PHRASES cũ + 4/5 PHRASES_MORE, chấm trên 1/5 còn lại (5 lượt)
Mỗi câu chấm ở 2 dạng: có dấu và không dấu. Không dùng test."""
import sys
sys.path.insert(0, "src")
import numpy as np
from common import *
import nlp_knowledge, nlp2
from nlp2 import MissionParser2, _unaccent

missions = [s["mission"] for sp in ("train", "validation") for s in load_split(sp)[2]]
OLD = {c: list(v) for c, v in nlp_knowledge.PHRASES.items()}
MORE = nlp_knowledge.PHRASES_MORE
items = [(a, c) for c, lst in MORE.items() for a in lst if a not in OLD[c]]
fold = np.random.default_rng(0).permutation(len(items)) % 5


def score(p, held, ths):
    """-> dict th -> [đúng gấp, tổng gấp, nhầm gấp, tổng không gấp, đúng dễ vỡ, tổng dễ vỡ, nhầm dễ vỡ, tổng không dễ vỡ] cho 2 dạng"""
    out = {}
    for th in ths:
        nlp2.E5_TH = th
        r = {"có dấu": [0] * 8, "không dấu": [0] * 8}
        for a, c in held:
            for form, s, acc in (("có dấu", _unaccent(a), a), ("không dấu", _unaccent(a), _unaccent(a))):
                s = " ".join(nlp2.tokens(nlp2.sentences(s)[0])) if nlp2.sentences(s) else s
                u, f = p._phrase_flags(nlp2.sentences(acc)[0] if nlp2.sentences(acc) else s, acc.lower())
                x = r[form]
                if c == 1: x[0] += u; x[1] += 1
                else: x[2] += u; x[3] += 1
                if c == 2: x[4] += f; x[5] += 1
                else: x[6] += f; x[7] += 1
        out[th] = r
    return out


def show(name, tot):
    print("\n==", name)
    for th, r in tot.items():
        for form, x in r.items():
            print(f"   ngưỡng {th} {form:9s}: GẤP nhận đúng {x[0]}/{x[1]} ({x[0] / x[1]:.0%}), nhận nhầm {x[2]}/{x[3]} ({x[2] / x[3]:.1%})"
                  f" | DỄ VỠ nhận đúng {x[4]}/{x[5]} ({x[4] / x[5]:.0%}), nhận nhầm {x[6]}/{x[7]} ({x[6] / x[7]:.1%})")


nlp2.USE_PHRASES_MORE = False   # phép đo này tự chia PHRASES_MORE thành phần học / phần chấm
THS = [(0.6, 0.8), (0.5, 0.6), (0.4, 0.5), (0.0, 0.0)]
nlp_knowledge.PHRASES = OLD
p = MissionParser2().fit(missions)
show("(A) cấu hình cũ, chấm trên toàn bộ ví dụ mới", score(p, items, THS))

tot = None
for k in range(5):
    nlp_knowledge.PHRASES = {c: OLD[c] + [a for (a, cc), f in zip(items, fold) if cc == c and f != k] for c in OLD}
    p = MissionParser2().fit(missions)
    r = score(p, [it for it, f in zip(items, fold) if f == k], THS)
    if tot is None: tot = r
    else:
        for th in r:
            for form in r[th]:
                tot[th][form] = [a + b for a, b in zip(tot[th][form], r[th][form])]
show("(B) học thêm 4/5 ví dụ mới, chấm trên 1/5 còn lại", tot)
nlp2.save_embed_cache()
