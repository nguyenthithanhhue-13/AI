import sys, re
sys.path.insert(0, "src")
from collections import Counter
from common import *
from nlp2 import *

trs = load_split("train")[2]
p = MissionParser2().fit([s["mission"] for s in trs])
# 1) vì sao fragile trên train bị sai?
c = Counter()
for s in trs:
    m = s["mission"]; r = p.parse(m["text"])
    if r["fragile"] != m["fragile"] or r["urgent"] != m["urgent"]:
        ments, plain = p._mentions(m["text"])
        for sent in {me["sent"] for me in ments}:
            mo = re.search(FRAGILE_POS, sent); mu = re.search(URGENT_POS, sent)
            if mo and not re.search(FRAGILE_NEG, sent): c[("F", mo.group(0))] += 1
            if mu and not re.search(URGENT_NEG, sent): c[("U", mu.group(0))] += 1
        if sum(c.values()) < 4: print(m["text"], "| pred F", r["fragile"], "gold", m["fragile"], [(me["text"], me["type"]) for me in ments])
print(c.most_common(20))
# 2) một ca cụ thể
t = "[đơn #6297] có đơn giao bưu kiện ở nơi phục vụ bữa trưa. nơi giữ xe không phải đimể nhận. cang nhanh càng tốt. cảm ơn!"
r = p.parse(t)
print({k: (v if not hasattr(v, "round") else dict(zip(PLACE_TYPES, v.round(2)))) for k, v in r.items()})
print([(me["text"], me["type"]) for me in p._mentions(t)[0]])
print("span probs:", dict(zip(PLACE_TYPES, p.span_probs("noi phuc vu bua trua").round(2))))
