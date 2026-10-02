"""Bộ đếm tổng hợp trên test so với validation. CHỈ in số đếm, không in nội dung câu hay ảnh test."""
import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from common import *
from nlp import MissionParser, sentences, tokens

p = MissionParser.load(OUT / "nlp_final.pkl")
for split in ["validation", "test"]:
    rows = load_json(DATA / split / "observations.json")
    worlds = pickle.load(open(CACHE / f"world_{split}_final.pkl", "rb"))
    st = Counter(); nlm = []; nment = []; nplain_unknown = 0; nplain = 0
    n = len(rows) // 10
    for si in range(n):
        text = rows[si * 10]["mission"]
        w = worlds[rows[si * 10]["image"]]["world"]
        res = p.parse(text)
        ments, plain = p._mentions(text)
        st["không tìm thấy tên địa điểm nào"] += len(ments) == 0
        st["không lần nhắc nào được coi là đích (conf<0.5)"] += res["conf"] < 0.5
        st["đích không có trên bản đồ"] += res["goal"] not in w["landmarks"]
        nment.append(len(ments))
        nlm.append(sum(len(v) for v in w["landmarks"].values()))
        for s in plain:
            nplain += 1
            nplain_unknown += s not in p.phrase
        st["urgent"] += res["urgent"]; st["fragile"] += res["fragile"]; st["via"] += res["via"] is not None
        st["goal_ref"] += res["goal_ref"] is not None
    print(split, n, {k: f"{v} ({v / n:.1%})" for k, v in st.items()})
    print("   số địa điểm CV thấy / cảnh: %.2f | số lần nhắc địa điểm / câu: %.2f | câu con không có địa điểm mà KHÔNG nằm trong bảng cụm từ: %d/%d (%.1f%%)"
          % (np.mean(nlm), np.mean(nment), nplain_unknown, nplain, 100 * nplain_unknown / nplain))
