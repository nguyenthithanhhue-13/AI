"""In các ví dụ viết tay mới bị _phrase_flags đọc sai (học PHRASES cũ + 4/5 PHRASES_MORE, chấm 1/5), dạng có dấu."""
import sys
sys.path.insert(0, "src")
import numpy as np
from common import *
import nlp_knowledge, nlp2
from nlp2 import MissionParser2, _unaccent
missions = [s["mission"] for sp in ("train", "validation") for s in load_split(sp)[2]]
OLD = {c: list(v) for c, v in nlp_knowledge.PHRASES.items()}
items = [(a, c) for c, lst in nlp_knowledge.PHRASES_MORE.items() for a in lst if a not in OLD[c]]
fold = np.random.default_rng(0).permutation(len(items)) % 5
nlp2.USE_PHRASES_MORE = False
err = {"gấp bị sót": [], "nhầm thành gấp": [], "dễ vỡ bị sót": [], "nhầm thành dễ vỡ": []}
for k in range(5):
    nlp_knowledge.PHRASES = {c: OLD[c] + [a for (a, cc), f in zip(items, fold) if cc == c and f != k] for c in OLD}
    p = MissionParser2().fit(missions)
    for (a, c), f in zip(items, fold):
        if f != k: continue
        for acc in ((a, _unaccent(a)) if "--both" in sys.argv else (a,)):
            ss = nlp2.sentences(acc)
            if not ss: continue
            u, fr = p._phrase_flags(ss[0], acc.lower())
            if c == 1 and not u: err["gấp bị sót"].append(acc)
            if c != 1 and u: err["nhầm thành gấp"].append(f"{acc} [{c}]")
            if c == 2 and not fr: err["dễ vỡ bị sót"].append(acc)
            if c != 2 and fr: err["nhầm thành dễ vỡ"].append(f"{acc} [{c}]")
for k, v in err.items():
    print(f"\n{k} ({len(v)}):", "; ".join(v))
