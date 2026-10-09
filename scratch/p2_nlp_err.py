"""Lỗi ĐỌC YÊU CẦU trên validation (bộ đọc chỉ học train): chạy parser trên BẢN ĐỒ THẬT (tách khỏi lỗi CV), so điểm đến (đích, điểm
ghé, cờ gấp / dễ vỡ) với scenes.json. In từng cảnh sai (validation được phép xem) để phân loại lỗi.
    python scratch/p2_nlp_err.py"""
import sys, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_lib import *
import nlp3
from pipeline_p2 import legs_from_mission
p = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl")
D = load(sys.argv[1] if len(sys.argv) > 1 else "validation")
c = collections.Counter()
for x in D:
    w = x["w"]; L = w["landmarks"]; m = x["m"]
    r = p.parse(m["text"], {t for t, v in L.items() if v}, L)
    mm = nlp3.resolve3(r, L, w)
    legs = legs_from_mission(w, mm)
    tl = [sorted(map(tuple, l)) for l in x["legs"]]; pl = [sorted(map(tuple, l)) for l in legs]
    errs = []
    if pl[-1] != tl[-1]:
        errs.append("ĐÍCH")
    if (len(pl) == 2) != (len(tl) == 2) or (len(pl) == 2 and pl[0] != tl[0]):
        errs.append("GHÉ")
    if bool(mm["urgent"]) != bool(m["urgent"]):
        errs.append("gấp")
    if bool(mm["fragile"]) != bool(m["fragile"]):
        errs.append("vỡ")
    for e in errs:
        c[e] += 1
    if errs:
        c["cảnh sai"] += 1
        print(f"[{'/'.join(errs)}] {x['s']['scene_id']} | thật: goal={m['goal']} ref={(m['goal_ref'] or {}).get('kind')} via={m['via']} "
              f"gấp={m['urgent']} vỡ={m['fragile']} | đọc: goal={mm['goal']} via={mm.get('via')} gấp={mm['urgent']} vỡ={mm['fragile']}")
        print("     ", m["text"])
print(dict(c), "/", len(D))
