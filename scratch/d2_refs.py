"""Chỉ dẫn vị trí có thể bị sót ở test: chỉ đếm theo các mẫu định sẵn (không in câu)."""
import sys, pickle, re
sys.path.insert(0, "src")
from collections import Counter
from common import *
import nlp2
from nlp2 import *
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
D = r"(bac|nam|dong|tay|tren|duoi|trai|phai)"
REL = r"(gan|xa|sat|canh|ke|giap|cach|lien)"
rows = load_json(DATA / "test" / "observations.json"); worlds = pickle.load(open("cache/world_test_final.pkl", "rb"))
c = Counter()
for si in range(len(rows) // 10):
    w = worlds[rows[si * 10]["image"]]["world"]
    if w is None: continue
    present = {x for x, v in w["landmarks"].items() if v}
    t = rows[si * 10]["mission"]
    m = p.parse(t, present); r = resolve_with_map(m, w["landmarks"], w)
    ments = p._mentions(t)[0]
    for key in ("goal", "via"):
        if r[key] is None or r[key + "_ref"] is not None or len(w["landmarks"].get(r[key], [])) < 2 or not m[key + "_known"]: continue
        c["(tổng: tên ĐÃ BIẾT, 2 bản, không có hướng)"] += 1
        for me in [x for x in ments if x["type"] == r[key]]:
            toks = me["toks"]; j = me["j"]
            after = " ".join(toks[j:j + 7]); before = " ".join(toks[max(0, me["i"] - 5):me["i"]])
            nxt = [x for x in me["ms"] if x[0] >= j]
            gap = nxt[0][0] - j if nxt else None
            mrel = re.search(r"\b" + REL + r"\b", after)
            mdir = re.search(r"\b" + D + r"\b", after)
            if mrel and nxt and gap <= 7:
                relpos = len(after[:mrel.start()].split())
                neg = bool(re.search(r"\b(khong|chang|cha|dau)\b", " ".join(toks[j:nxt[0][0]])))
                c[f"GẦN/XA: mốc cách {gap} từ | từ quan hệ '{mrel.group()}' ở vị trí {relpos} | mốc {'đã biết' if nxt[0][2] else 'lạ'}, 1 bản: {len(w['landmarks'].get(nxt[0][2], [])) == 1 if nxt[0][2] else '?'} | phủ định ở giữa: {neg} | 'hơn/nhất' sau mốc: {bool(re.match(r'(hon|nhat)', ' '.join(toks[nxt[0][1]:nxt[0][1] + 1])))}"] += 1
            elif mrel:
                c[f"GẦN/XA: có '{mrel.group()}' nhưng không thấy mốc trong 7 từ"] += 1
            if mdir:
                pos = len(after[:mdir.start()].split()); prev = toks[j + pos - 1] if pos > 0 else "(ngay sau tên)"
                prev = prev if prev in ("phia", "ben", "man", "huong", "goc", "dau", "mien", "canh", "mep", "ria", "khu", "nua", "o", "nam", "me", "dang", "cuoi", "tan", "ve", "(ngay sau tên)") else "(từ khác)"
                nx = toks[j + pos + 1] if j + pos + 1 < len(toks) else "(hết câu)"
                nx = nx if nx in ("ban", "cung", "nhat", "hon", "cua", "(hết câu)") else "(từ khác)"
                c[f"HƯỚNG: '{mdir.group()}' ở vị trí {pos}, từ đứng trước: {prev}, từ đứng sau: {nx}"] += 1
            if re.search(r"\b" + D + r"\b", before): c["HƯỚNG đứng TRƯỚC tên (trong 5 từ)"] += 1
            break
for k, v in c.most_common(): print(f"   {v:3d}  {k}")
