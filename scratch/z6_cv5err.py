"""Các cảnh validation mà NLP 5-fold (bản 2) làm sai bước đi, với bản đồ đúng."""
import sys
sys.path.insert(0, "src")
from common import *
from pipeline import parse_split, predict_scene
from nlp2 import resolve_with_map, PLACE_TYPES
from eval_nlp2 import gold

rows, labels, scenes = load_split("validation")
mode = sys.argv[1] if len(sys.argv) > 1 else "cv5"
ms = parse_split("validation", mode)
n = 0
for si, s in enumerate(scenes):
    w = world_from_scene(s)
    p = predict_scene(w, ms[si])
    if p != labels[si * 10:(si + 1) * 10]:
        n += 1
        r = resolve_with_map(ms[si], w["landmarks"])
        print(si, s["mission"]["text"])
        print("    gold", gold(s["mission"]))
        print("    pred", r, "| goal_known", ms[si]["goal_known"], "via_known", ms[si]["via_known"], "excluded", ms[si]["excluded"])
print("scenes wrong:", n)
