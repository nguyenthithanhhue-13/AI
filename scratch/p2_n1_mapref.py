"""Đo bộ nhận diện mô tả qua bản đồ (src/mapref.py) trên train + validation."""
import sys, collections
sys.path.insert(0, "src")
from common import *
import mapref
mapref.VOCAB = mapref.build_vocab([s["mission"]["text"] for s in load_split("train")[2]])
for split in ("train", "validation"):
    rows, labels, scenes = load_split(split)
    c = collections.Counter(); bad = []
    for s in scenes:
        m = s["mission"]; g = m["goal_ref"]
        k = g["kind"] if g and (g["kind"].endswith("_most") or g["kind"] == "anchor_near") else None
        d = mapref.detect(m["text"])
        dk = d[0] if d else None
        c[(k, dk)] += 1
        if k != dk: bad.append((k, dk, d[1] if d else None, m["text"]))
    print(split, sorted(((str(a), str(b), n) for (a, b), n in c.items()), key=lambda x: (x[0], x[1])))
    for b in bad[:12]: print("   SAI", b)
