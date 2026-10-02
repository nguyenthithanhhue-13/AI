"""So sánh các mô hình nghĩa pretrained nhỏ (<= ~135 triệu tham số) cho hai việc với cách nói CHƯA GẶP:
 (a) đoán loại của tên gọi địa điểm chỉ có ở validation; (b) phân loại câu gấp / dễ vỡ chỉ có ở validation.
Học trên train, không dùng luật từ khóa hay ví dụ viết tay -> phép đo trung thực."""
import sys, re
sys.path.insert(0, "src"); sys.path.insert(0, "scratch")
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from common import *
from nlp import mine_lexicon, norm, PLACE_TYPES

MODELS = sys.argv[1:] or ["intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                          "keepitreal/vietnamese-sbert", "bkai-foundation-models/vietnamese-bi-encoder"]
tr = [s["mission"] for s in load_split("train")[2]]; va = [s["mission"] for s in load_split("validation")[2]]


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
Xa = [acc_tr[a] for a in lex_tr] + list(lex_tr) + list(canon) + [norm(c) for c in canon]
ya = [PLACE_TYPES.index(t) for t in lex_tr.values()] * 2 + [PLACE_TYPES.index(t) for t in canon.values()] * 2
names = list(new)

# câu con gấp / dễ vỡ (có dấu)
PREFIX = r"^(robot ơi|yêu cầu mới|nhờ bạn nhé|xin chào|chào robot|nhắn robot)\b[,: ]*"


def plain_table(missions):
    pat = re.compile("|".join(sorted(map(re.escape, lex_all), key=len, reverse=True)))
    tab = defaultdict(lambda: [0, 0, 0])
    for m in missions:
        t = re.sub(r"\[đơn #\d+\]", " ", m["text"].lower())
        for p in re.split(r"[.!?]+", t):
            p = re.sub(PREFIX, "", re.sub(r"\s+", " ", p).strip(" ,")).strip(" ,")
            if not p or pat.search(norm(p)): continue
            x = tab[p]; x[0] += 1; x[1] += bool(m["urgent"]); x[2] += bool(m["fragile"])
    return tab


lab = lambda x: 1 if x[1] >= 0.9 * x[0] else (2 if x[2] >= 0.9 * x[0] else 0)
ttab = plain_table(tr); vtab = plain_table(va)
S = [s for s, x in ttab.items() if x[0] >= 3]; yS = [lab(ttab[s]) for s in S]
known = {norm(s) for s in ttab}
V = [s for s, x in vtab.items() if x[0] >= 3 and norm(s) not in known]; yV = [lab(vtab[s]) for s in V]
Vacc = [i for i, s in enumerate(V) if norm(s) != s]


def encoder(name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cpu")
    n = sum(p.numel() for p in m.parameters())
    pre = "query: " if "e5" in name else ""
    return (lambda texts: m.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False)), n


for name in MODELS:
    try:
        enc, n = encoder(name)
    except Exception as e:
        print(name, "-> lỗi tải:", str(e)[:150]); continue
    clf = LogisticRegression(C=50, max_iter=3000).fit(enc(Xa), ya)
    res = []
    for texts in ([acc_new[a] for a in names], names):
        P = clf.predict_proba(enc(texts))
        res.append(sum(PLACE_TYPES[clf.classes_[int(p.argmax())]] == new[a] for a, p in zip(names, P)))
        res.append(sum(PLACE_TYPES.index(new[a]) in clf.classes_[np.argsort(-p)[:3]] for a, p in zip(names, P)))
    pc = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(enc(S), yS)
    pred = pc.predict(enc(V))
    okp = sum(a == b for a, b in zip(pred, yV)); okp_acc = sum(pred[i] == yV[i] for i in Vacc)
    print(f"{name}  ({n / 1e6:.0f}M tham số)")
    print(f"    tên gọi lạ: có dấu {res[0]}/{len(names)} (top-3 {res[1]}), không dấu {res[2]}/{len(names)} (top-3 {res[3]})"
          f" | câu gấp/dễ vỡ lạ: {okp}/{len(V)} (riêng câu có dấu {okp_acc}/{len(Vacc)})", flush=True)
