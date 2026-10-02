"""Thống kê trên train/validation: mỗi yêu cầu có câu nói về gấp / dễ vỡ không, khẳng định hay phủ định?
Và trên validation: thời tiết đọc từ biểu tượng so với đọc từ chữ trong chú giải."""
import sys, re, pickle
sys.path.insert(0, "src")
from collections import Counter
from common import *
from nlp import sentences
from nlp2 import URGENT_NEG, URGENT_POS, FRAGILE_NEG, FRAGILE_POS

for sp in ("train", "validation"):
    sc = load_split(sp)[2]
    c = Counter()
    for s in sc:
        m = s["mission"]; ss = sentences(m["text"])
        un = any(re.search(URGENT_NEG, x) for x in ss); fn = any(re.search(FRAGILE_NEG, x) for x in ss)
        c[("urgent", "khẳng định" if m["urgent"] else ("phủ định rõ" if un else "không nhắc"))] += 1
        c[("fragile", "khẳng định" if m["fragile"] else ("phủ định rõ" if fn else "không nhắc"))] += 1
        c[("số câu con", len(ss))] += 0
    n = len(sc)
    print(sp, {k: f"{v / n:.1%}" for k, v in sorted(c.items()) if v})
sc = load_split("validation")[2]
w = pickle.load(open(CACHE / "world_validation_dev.pkl", "rb"))
c = Counter()
for s in sc:
    info = w[s["image"]]["info"]; true = s["weather"] == "rain"
    icon = info["p_rain_icon"] > 0.5
    c[("icon đúng", icon == true)] += 1
    if info["rain_text"] is not None:
        c[("chữ có thông tin; chữ đúng / icon đúng", info["rain_text"] == true, icon == true)] += 1
print(dict(c))
