"""Bộ đếm TỔNG HỢP: tỉ lệ câu có cụm khớp MỜ (lệch chính tả) với từ điển tên gọi của bộ đọc final, theo split; tách theo cụm lệch
1 từ ngắn / dài, và theo loại khớp."""
import sys, collections
sys.path.insert(0, "src")
from common import *
import nlp3, nlp
from nlp import norm, tokens, sentences
p = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl"); lex = p.p2.lex
for split in ("train", "validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    c = collections.Counter(); n = len(rows) // 10
    for i in range(n):
        t = rows[i * 10]["mission"]; anyf = False
        for s in sentences(t):
            toks = tokens(s)
            for a, b, ty, al in lex.find(toks):
                if toks[a:b] != al.split():
                    anyf = True
                    c["cụm mờ"] += 1
                    diff = [w for w, x in zip(toks[a:b], al.split()) if w != x]
                    c["mờ: từ lệch dài >= 5"] += any(len(w) >= 5 for w in diff)
                    c["mờ: từ lệch là từ thông dụng"] += any(lex.ngram_freq.get(w, 0) >= 100 for w in diff)
        c["câu có khớp mờ"] += anyf
    print(split, n, {k: round(v / n, 3) for k, v in sorted(c.items())})
