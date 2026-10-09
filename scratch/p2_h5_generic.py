"""Bộ đếm tổng hợp trên test (không in câu): các cảnh chỉ lượt TỔNG QUÁT (mới) nhận ra mô tả qua bản đồ -> đầu ngữ ngay trước
cụm là đầu ngữ CHUNG (địa điểm / nơi / chỗ ...) hay là TÊN một loại địa điểm (nguy cơ: tham chiếu chọn bản "thư viện phía bắc nhất")."""
import sys, pickle, collections, re
sys.path.insert(0, "src")
from common import *
import nlp3, mapref
p = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl"); mapref.VOCAB = p.vocab
GEN = {"dia diem", "diem", "cho", "noi", "vi tri", "khu vuc", "toa nha", "cong trinh", "dia chi", "diem den", "dich den", "phong", "day", "khu", "toa"}
for split in ("validation", "test"):
    rows = load_json(DATA / split / "observations.json")
    c = collections.Counter()
    for i in range(len(rows) // 10):
        t = rows[i * 10]["mission"]
        f = mapref.find(t, p.vocab)
        if not f or not f.get("generic"):
            continue
        a, b = f["span"]
        pre = mapref.unaccent(t[:a]).split()[-3:]
        head = mapref.unaccent(t[a:b]).split()[:2]
        g = " ".join(head) in GEN or head[0] in GEN
        c["tổng"] += 1; c[f["kind"]] += 1
        c["đầu ngữ chung" if g else "đầu ngữ KHÁC (có thể là tên loại)"] += 1
        r = p.p2.parse(t[:a] + "XYZ" + t[b:], None)
        c["trước cụm là tên địa điểm đã biết"] += bool(r.get("goal_known")) and False
    print(split, dict(c))
