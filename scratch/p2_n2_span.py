import sys, collections, random
sys.path.insert(0, "src")
from common import *
import mapref
mapref.VOCAB = mapref.build_vocab([s["mission"]["text"] for s in load_split("train")[2]])
random.seed(3)
for split in ("train", "validation"):
    sc = load_split(split)[2]
    for s in random.sample(sc, min(400, len(sc))):
        t = s["mission"]["text"]; f = mapref.find(t)
        if f and random.random() < 0.04:
            a, b = f["span"]
            print(f"[{f['kind']}] ...{t[max(0,a-25):a]}<<{t[a:b]}>>{t[b:b+25]}...   mốc: {t[f['anchor_span'][0]:f['anchor_span'][1]] if f['anchor_span'] else None}")
