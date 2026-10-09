"""Giả thuyết: có điểm ghé -> robot chỉ tìm đường tới điểm ghé (TUẦN TỰ) chứ không tối ưu cả hành trình (KHỚP).
Với chi phí = số đoạn (theta one=1), đếm trên train: cảnh mà tập bước tối ưu khớp / tuần tự KHÁC nhau, nhãn mỗi robot nằm ở đâu."""
import sys, collections
sys.path.insert(0, "scratch")
from p2_fast import *
TR = load("train") + load("validation")
th = theta_of()          # one = 1: số đoạn
c = collections.Counter()
for x in TR:
    if len(x["legs"]) != 2:
        continue
    for r in range(10):
        g = SG(x["w"], r == 4, frozenset())
        qj = g.q(th, x["legs"]); qs = g.q(th, [x["legs"][0]])
        Aj = set(argmins(qj, 1e-6)); As = set(argmins(qs, 1e-6))
        if not Aj or not As or Aj == As:
            continue
        y = x["y"][r]
        c[r, "khác"] += 1
        c[r, "chỉ khớp"] += y in Aj and y not in As
        c[r, "chỉ tuần tự"] += y in As and y not in Aj
        c[r, "cả hai"] += y in As and y in Aj
        c[r, "không"] += y not in As and y not in Aj
for r in range(10):
    print(f"R{r}: cảnh khác nhau {c[r, 'khác']:4d} | nhãn chỉ ở KHỚP {c[r, 'chỉ khớp']:4d} | chỉ ở TUẦN TỰ {c[r, 'chỉ tuần tự']:4d} | cả hai {c[r, 'cả hai']:3d} | không {c[r, 'không']:3d}")
