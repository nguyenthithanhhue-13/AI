import sys
sys.path.insert(0, "src")
from common import *
from nlp import *

tr = [s["mission"] for s in load_split("train")[2]]
va = [s["mission"] for s in load_split("validation")[2]]
lex = mine_lexicon(tr + va)
L = Lexicon(lex)
places = {a: t for a, t in lex.items()}
print(len(lex))
by = defaultdict(list)
for a, t in lex.items(): by[t].append(a)
for t, v in by.items(): print(t, len(v), v)
for s in ["ben trong la do thuy tinh", "do gom ben trong, tranh va cham", "hang ky va dap", "kien hang nay rat mong manh", "uu tien so mot, di ngay"]:
    print(s, "->", L.find(tokens(s)))
# mọi câu "thuần" (không địa điểm) bị khớp mờ nhầm?
from collections import Counter
c = Counter()
for m in tr + va:
    for s in sentences(m["text"]):
        toks = tokens(s)
        for (i, j, t, a) in L.find(toks):
            w = " ".join(toks[i:j])
            if w != a: c[(w, a)] += 1
print("fuzzy matches:", len(c))
for k, v in c.most_common(400): print(v, k)
