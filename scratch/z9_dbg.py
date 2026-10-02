import sys, pickle
sys.path.insert(0, "src")
import numpy as np
from common import *
import nlp2
from nlp2 import *

trs = load_split("train")[2]
p = MissionParser2().fit([s["mission"] for s in trs])
p.ablate_via = True
t = "Chào robot, hàng dễ vỡ, đi cẩn thận. Trước tiên qua phòng lab mạn trái nhận bộ dụng cụ thí nghiệm, sau đó vận chuyển khay cơm đến nhà ăn giúp mình. Bỏ qua dãy phòng nội trú, không phải ở đó. Mai giao cũng kịp."
orig = MissionParser2._structure


def traced(ments, role, negs, P):
    print("before:", [(m["text"], int(r), bool(n)) for m, r, n in zip(ments, role, negs)])
    orig(ments, role, negs, P)
    print("after: ", [(m["text"], int(r)) for m, r in zip(ments, role)])


p._structure = traced
r = p.parse(t)
print({k: r[k] for k in ("goal", "via", "goal_ref", "via_ref")})
for me in p._mentions(t)[0]:
    print(me["text"], "| seg:", me["seg"], "| prev tok:", me["toks"][me["i"] - 1] if me["i"] else None)
