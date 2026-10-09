"""Điểm ghé có thể bị sót: yêu cầu có từ chỉ thứ tự / ghé mà kết quả không có điểm ghé. Chỉ đếm theo mẫu định sẵn."""
import sys, pickle, re
sys.path.insert(0, "src")
from collections import Counter
from common import *
import nlp2
from nlp2 import *
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
CUES = {"ghé / tạt / rẽ / tiện đường": r"\b(ghe|tat (qua|vao|ngang)|re (qua|vao)|tien duong|tien the|ngang qua)\b",
        "trước khi / trước tiên / trước đã / đầu tiên": r"\b(truoc khi|truoc tien|truoc da|truoc het|dau tien|viec dau)\b",
        "sau đó / xong thì / rồi mới / rồi hãy / tiếp theo": r"\b(sau do|xong (thi|roi|la)|roi (moi|hay)|tiep (theo|do)|ke (do|tiep))\b",
        "lấy / nhận / đón (hàng) ở ...": r"\b(lay|nhan|don|linh|lanh) [a-z ]{0,25}\b(o|tai|tu)\b"}
for sp in ("validation", "test"):
    rows = load_json(DATA / sp / "observations.json"); worlds = pickle.load(open(f"cache/world_{sp}_final.pkl", "rb"))
    c = Counter(); n = 0
    for si in range(len(rows) // 10):
        w = worlds[rows[si * 10]["image"]]["world"]
        if w is None: continue
        n += 1
        present = {x for x, v in w["landmarks"].items() if v}
        t = rows[si * 10]["mission"]
        m = p.parse(t, present); r = resolve_with_map(m, w["landmarks"], w)
        has_via = r["via"] is not None
        sents = [s for s in nlp2.sentences(t) if not re.search(NEG_SENT, s)]
        txt = " . ".join(sents)
        hits = [k for k, rx in CUES.items() if re.search(rx, txt)]
        ments = p._mentions(t)[0]
        nplace_sents = len({me["si"] for me in ments})
        c["có điểm ghé" if has_via else "KHÔNG có điểm ghé"] += 1
        if not has_via:
            for k in hits: c[f"   KHÔNG điểm ghé nhưng có: {k}"] += 1
            c["   KHÔNG điểm ghé nhưng có ít nhất một dấu hiệu"] += bool(hits)
            c["   KHÔNG điểm ghé, có dấu hiệu, và chỉ thấy 1 địa điểm không bị phủ định"] += bool(hits) and sum(1 for me in ments if not re.search(NEG_SENT, me["sent"])) <= 1
        else:
            c["   có điểm ghé và có ít nhất một dấu hiệu"] += bool(hits)
    print(f"\n== {sp}: {n}")
    for k, v in c.items(): print(f"   {v:4d} ({v / n:5.1%})  {k}")
