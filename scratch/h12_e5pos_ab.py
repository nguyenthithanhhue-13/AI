"""A/B: cho / không cho mô hình nghĩa tự nhận câu là GẤP (khi không có từ khóa), trên bộ câu thử vòng 2 + 3."""
import sys, re
sys.path.insert(0, "src")
from common import *
import nlp2
from nlp2 import MissionParser2, _unaccent
T = []
for f in ("scratch/h10_round2.py", "scratch/h11_round3.py"):
    src = open(f, encoding="utf-8").read()
    T += eval("[" + src.split("T = [", 1)[1].split("]\nmissions = ", 1)[0] + "]")
p = MissionParser2().fit([s["mission"] for sp in ("train", "validation") for s in load_split(sp)[2]])
for flag in (True, False):
    nlp2.E5_POS_URGENT = flag
    tp = fp = fn = tn = 0; bad = []
    for text, u, f in T:
        for t in (text, _unaccent(text)):
            r = bool(p.parse(t)["urgent"])
            tp += r and u; fp += r and not u; fn += (not r) and u; tn += (not r) and not u
            if r != bool(u): bad.append(t)
    print(f"mô hình nghĩa tự nhận gấp = {flag}: đúng {tp + tn}/{tp + fp + fn + tn} | sót {fn} | nhầm {fp}")
    for b in bad: print("     ", b)
nlp2.save_embed_cache()
