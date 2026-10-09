"""Bộ đếm TỔNG HỢP trên test: nhóm "cách nói mới" (khung mô tả bản đồ chỉ lượt TỔNG QUÁT mới nhận ra = gần chắc là cảnh MỒI theo
p16) có trùng với nhóm "tên lạ" và nhóm "hai bộ đọc bất đồng" không."""
import sys, pickle, collections
sys.path.insert(0, "src")
from common import *
import nlp3, mapref
from pipeline_p2 import legs_from_mission
pa = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl"); pb = nlp3.MissionParser3.load(OUT / "nlp3_final.pkl")
rows = load_json(DATA / "test" / "observations.json")
W = pickle.load(open(CACHE / "world_test_v1final.pkl", "rb"))
c = collections.Counter()
for i in range(len(rows) // 10):
    w = W[rows[i * 10]["image"]]["world"]
    if w is None:
        continue
    L = w["landmarks"]; t = rows[i * 10]["mission"]
    mapref.VOCAB = pb.vocab
    f = mapref.find(t, pb.vocab)
    novel = bool(f and f.get("generic"))
    b = pb.parse(t, {k for k, v in L.items() if v}, L); a = pa.parse(t, {k for k, v in L.items() if v}, L)
    isref = bool(b.get("goal_ref")) and b["goal_ref"][0] == "pos"
    unk = (not b.get("goal_known") and not isref) or (b.get("via") and not b.get("via_known"))
    dis = a["goal"] != b["goal"] or a.get("via") != b.get("via")
    grp = "MỚI (khung bản đồ mới)" if novel else "còn lại"
    c[grp, "n"] += 1; c[grp, "tên lạ"] += bool(unk); c[grp, "bất đồng"] += dis
    c["tên lạ" if unk else "tên quen", "n"] += 1; c["tên lạ" if unk else "tên quen", "bất đồng"] += dis
for g in ("MỚI (khung bản đồ mới)", "còn lại"):
    n = c[g, "n"]; print(f"{g:24s} n={n:4d} | có tên lạ {c[g, 'tên lạ'] / n:.2f} | hai bộ đọc bất đồng {c[g, 'bất đồng'] / n:.2f}")
for g in ("tên lạ", "tên quen"):
    n = c[g, "n"]; print(f"{g:24s} n={n:4d} | hai bộ đọc bất đồng {c[g, 'bất đồng'] / n:.2f}")
