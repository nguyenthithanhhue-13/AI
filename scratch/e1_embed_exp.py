"""Mô hình nghĩa (e5) có giúp với thứ CHƯA GẶP không? Học trên train, đo trên phần chỉ có ở validation.
 (a) đoán loại của tên gọi địa điểm chỉ có ở validation
 (b) phân loại câu con gấp / dễ vỡ / trung tính chỉ có ở validation
"""
import sys, re, unicodedata
sys.path.insert(0, "src")
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from common import *
from nlp import mine_lexicon, sentences, tokens, norm, PLACE_TYPES
from nlp2 import keyword_scores, URGENT_POS, URGENT_NEG, FRAGILE_POS, FRAGILE_NEG, CANONICAL
from embed import Embedder

tr = [s["mission"] for s in load_split("train")[2]]; va = [s["mission"] for s in load_split("validation")[2]]
E = Embedder()


def accented_forms(missions, phrases):
    """phrase không dấu -> dạng có dấu phổ biến nhất tìm thấy trong văn bản."""
    out = defaultdict(Counter)
    want = defaultdict(list)
    for p in phrases: want[len(p.split())].append(p)
    pset = set(phrases)
    for m in missions:
        acc = re.findall(r"\w+", m["text"].lower())
        un = [norm(t) for t in acc]
        for k in want:
            for i in range(len(un) - k + 1):
                key = " ".join(un[i:i + k])
                if key in pset: out[key][" ".join(acc[i:i + k])] += 1
    res = {}
    for p in phrases:
        if out[p]:
            # ưu tiên dạng có dấu (khác với dạng không dấu)
            c = out[p]
            best = max(c, key=lambda s: (s != p, c[s]))
            res[p] = best
        else:
            res[p] = p
    return res


lex_tr = mine_lexicon(tr); lex_all = mine_lexicon(tr + va)
new = {a: t for a, t in lex_all.items() if a not in lex_tr}
acc_tr = accented_forms(tr, list(lex_tr)); acc_new = accented_forms(va, list(new))
print("train aliases", len(lex_tr), "| validation-only aliases", len(new))
Xtr_txt = [acc_tr[a] for a in lex_tr] + ["thư viện", "ký túc xá", "nhà thể thao", "trạm y tế", "căn tin", "bãi xe", "giảng đường", "phòng thí nghiệm", "phòng hành chính", "cổng trường"]
ytr = [PLACE_TYPES.index(t) for t in lex_tr.values()] + [PLACE_TYPES.index(t) for t in ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]]
clf = LogisticRegression(C=50, max_iter=3000).fit(E.encode(Xtr_txt), ytr)
names = list(new)
for variant, texts in (("có dấu", [acc_new[a] for a in names]), ("không dấu", names)):
    P = clf.predict_proba(E.encode(texts))
    full = np.zeros((len(names), 10)); full[:, clf.classes_] = P
    ok_e = sum(PLACE_TYPES[int(p.argmax())] == new[a] for a, p in zip(names, full))
    kw = np.array([keyword_scores(a) for a in names])
    ok_k = sum(k.max() > 0 and PLACE_TYPES[int(k.argmax())] == new[a] for a, k in zip(names, kw))
    comb = np.log(full + 0.02) + 1.0 * kw
    ok_c = sum(PLACE_TYPES[int(c.argmax())] == new[a] for a, c in zip(names, comb))
    print(f"(a) [{variant}] e5: {ok_e}/{len(names)} | từ khóa viết tay: {ok_k}/{len(names)} | kết hợp: {ok_c}/{len(names)}")
    if variant == "có dấu":
        for a, p in zip(names, full):
            if PLACE_TYPES[int(p.argmax())] != new[a]: print("     e5 sai:", acc_new[a], "->", PLACE_TYPES[int(p.argmax())], "(đúng:", new[a] + ")")

# ---------- (b) câu gấp / dễ vỡ ----------
PREFIX = r"^(robot ơi|yêu cầu mới|nhờ bạn nhé|xin chào|chào robot|nhắn robot)\b[,: ]*"


def plain_table(missions, lex):
    """câu con (có dấu, chữ thường) không chứa tên địa điểm -> [n, n_urgent, n_fragile]"""
    pat = re.compile("|".join(sorted(map(re.escape, lex), key=len, reverse=True)))
    tab = defaultdict(lambda: [0, 0, 0])
    for m in missions:
        t = re.sub(r"\[đơn #\d+\]", " ", m["text"].lower())
        for p in re.split(r"[.!?]+", t):
            p = re.sub(PREFIX, "", re.sub(r"\s+", " ", p).strip(" ,")).strip(" ,")
            if not p or pat.search(norm(p)): continue
            if norm(p) == p and re.search(r"[a-z]", p) and len(p) > 0 and any(ch in p for ch in "aeiouy") and p != unicodedata.normalize("NFC", p): pass
            x = tab[p]; x[0] += 1; x[1] += bool(m["urgent"]); x[2] += bool(m["fragile"])
    return tab


def lab(x): return 1 if x[1] >= 0.9 * x[0] else (2 if x[2] >= 0.9 * x[0] else 0)


ttab = plain_table(tr, lex_all); vtab = plain_table(va, lex_all)
S = [s for s, x in ttab.items() if x[0] >= 3]; y = [lab(ttab[s]) for s in S]
print("train phrases:", len(S), Counter(y))
pc = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(E.encode(S), y)
known_norm = {norm(s) for s in ttab}
V = [s for s, x in vtab.items() if x[0] >= 3 and norm(s) not in known_norm]; yv = [lab(vtab[s]) for s in V]
pred = pc.predict(E.encode(V))
rule = [1 if (re.search(URGENT_POS, norm(s)) and not re.search(URGENT_NEG, norm(s))) else (2 if (re.search(FRAGILE_POS, norm(s)) and not re.search(FRAGILE_NEG, norm(s))) else 0) for s in V]
print(f"(b) câu con chỉ có ở validation: {len(V)} | e5 đúng {sum(a == b for a, b in zip(pred, yv))} | luật từ khóa đúng {sum(a == b for a, b in zip(rule, yv))}")
for s, a, b, n in zip(V, pred, yv, [vtab[s][0] for s in V]):
    print(f"     {'OK ' if a == b else 'SAI'} n={n:3d} thật={b} e5={a} | {s}")
E.save()
