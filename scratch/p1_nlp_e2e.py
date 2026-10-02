"""world ĐÚNG + NLP dự đoán -> điểm; in các cảnh NLP làm sai bước đi."""
import sys
sys.path.insert(0, "src")
from common import *
from pipeline import parse_split, predict_scene

rows, labels, scenes = load_split("validation")
for mode in ["trainonly", "cv5"]:
    ms = parse_split("validation", mode)
    preds = []
    bad = []
    for si, s in enumerate(scenes):
        p = predict_scene(world_from_scene(s), ms[si])
        preds += p
        if p != labels[si * 10:(si + 1) * 10]:
            bad.append(si)
    score, per = macro_accuracy(rows, labels, preds)
    print(mode, f"{score:.4f}", [round(v, 3) for v in per.values()], "scenes with errors:", len(bad))
    if mode == "cv5":
        for si in bad:
            g = mission_from_scene(scenes[si]); p = ms[si]
            print("  ", scenes[si]["mission"]["text"])
            print("      gold", g)
            print("      pred", {k: p[k] for k in g})
