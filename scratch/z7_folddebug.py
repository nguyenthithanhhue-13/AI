import sys
sys.path.insert(0, "src")
from common import *
from nlp2 import *
from nlp_tagger import tag_tokens

trs = load_split("train")[2]; vas = load_split("validation")[2]
tr = [s["mission"] for s in trs]; va = [s["mission"] for s in vas]
for si in map(int, sys.argv[1:]):
    k = si % 5
    p = MissionParser2().fit(tr + [m for i, m in enumerate(va) if i % 5 != k])
    t = va[si]["text"]
    ments, plain = p._mentions(t)
    print(si, t)
    print("  mentions:", [(m["text"], m["type"]) for m in ments])
    for sent in p._sents(t):
        tt = tag_tokens(sent); pr = p.tagger.predict(tt)
        print("   ", " ".join(f"{a}[{b:.2f}]" for a, b in zip(tt, pr)))
    r = p.parse(t)
    print("  parse:", {k2: r[k2] for k2 in ("goal", "goal_ref", "via", "via_ref", "goal_known", "excluded")})
    print("  right_bound has dong/man/trai/tren:", [w in p.right_bound for w in ("dong", "man", "trai", "tren")], "| in vocab:", [w in p.tagger.vocab for w in ("dong", "man", "trai", "tren")])
