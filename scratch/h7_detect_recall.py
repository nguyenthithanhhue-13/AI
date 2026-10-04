"""Tỉ lệ PHÁT HIỆN tên gọi bị giấu: với mỗi lần một tên gọi bị giấu xuất hiện trong câu chấm, _mentions có trả về
một cụm chồng lên nó không? In tỉ lệ theo từng tên gọi và vài câu bị sót. (train + validation, 5 nhóm như h1)"""
import sys
sys.path.insert(0, "src")
from collections import Counter, defaultdict
from common import *
exec(open("scratch/h1_alias_holdout.py", encoding="utf-8").read().split("tot_n = tot_s = 0")[0])
from nlp import sentences, tokens
hit, tot, exact = Counter(), Counter(), Counter()
miss = defaultdict(list)
for k in range(5):
    held = {a for a in aliases if group[a] == k}
    p = MissionParser2(); p.use_e5 = "--e5" in sys.argv
    p.fit([missions[i] for i in range(len(scenes)) if not has[i] & held])
    for i in range(len(scenes)):
        if not has[i] & held: continue
        ments = p._mentions(missions[i]["text"])[0]
        for a in has[i] & held:
            at = a.split()
            for me in ments[:1] if not ments else [None]:
                pass
            # mọi vị trí alias xuất hiện trong từng câu con
            seen = set()
            for me in ments:
                seen.add((me["si"], me["i"], me["j"]))
            for si, sent in enumerate(p._sents(missions[i]["text"])):
                words = p._tt(sent)[1]
                for s0 in range(len(words) - len(at) + 1):
                    if words[s0:s0 + len(at)] == at:
                        tot[a] += 1
                        ov = [(x, y) for (s_, x, y) in seen if s_ == si and x < s0 + len(at) and y > s0]
                        if ov:
                            hit[a] += 1
                            exact[a] += (s0, s0 + len(at)) in ov
                        elif len(miss[a]) < 2:
                            miss[a].append(" ".join(words))
T, H, X = sum(tot.values()), sum(hit.values()), sum(exact.values())
print(f"PHÁT HIỆN: {H}/{T} = {H / T:.3f} | đúng ranh giới {X}/{T} = {X / T:.3f}")
for a in sorted(tot, key=lambda a: hit[a] / tot[a])[:25]:
    print(f"   {a:28s} {hit[a]:3d}/{tot[a]:3d}  đúng ranh giới {exact[a]:3d}   {miss[a][:1]}")
