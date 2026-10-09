"""Độ đi vòng TỐI THIỂU của bước thật (số đoạn ngắn nhất qua bước thật - số đoạn ngắn nhất chung) theo khoảng cách ngắn nhất,
từng robot (train + val, thông tin đúng). Tìm "ngân sách đi vòng" phụ thuộc khoảng cách.
    python scratch/p2_detour.py [robots]"""
import sys, pickle, collections, numpy as np
sys.path.insert(0, "scratch")
from p2_lib import *
RB = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 5, 6, 7, 8]
D = load("train") + load("validation")
P = pickle.load(open("cache/p2_paretotu_train_0.pkl", "rb")) + pickle.load(open("cache/p2_paretotu_validation_0.pkl", "rb"))
for r in RB:
    tab = collections.defaultdict(collections.Counter)
    for x, fr in zip(D, P):
        if not fr or x["y"][r] not in fr:
            continue
        s = min(F[:, 0].min() for _, F in fr.values())
        e = fr[x["y"][r]][1][:, 0].min() - s
        tab[min(int(s), 14)][int(e)] += 1
    print(f"R{r}: số đoạn ngắn nhất -> phân bố đi vòng tối thiểu của bước thật (0/1/2/3/4+)")
    for s in sorted(tab):
        c = tab[s]; n = sum(c.values())
        print(f"   {s:2d} (n={n:3d}): " + " ".join(f"{c[k] / n:5.2f}" for k in (0, 1, 2, 3)) + f" {sum(v for k, v in c.items() if k >= 4) / n:5.2f}")
