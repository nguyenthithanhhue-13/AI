"""Câu phụ nhiều vế trộn 'dễ vỡ' + 'không dễ vỡ' ở test: đếm theo thứ tự vế và các nhóm từ định sẵn (không in câu)."""
import sys, pickle, re
sys.path.insert(0, "src")
from collections import Counter
from common import *
import nlp2
from nlp2 import *
p = pickle.load(open(OUT / "nlp2_final.pkl", "rb"))
CONTRA = r"\b(nhung|chu|ma|tuy|du|that ra|thuc ra|tuong|nghe|co ve|hoi|kha|trong)\b"
EMPH = r"\b(rat|lam|qua|dang|can|that|cuc ky|het suc)\b"
for sp in ("validation", "test"):
    rows = load_json(DATA / sp / "observations.json")
    c = Counter(); n = len(rows) // 10
    for si in range(n):
        t = rows[si * 10]["mission"]
        ments, plain, plain_acc = p._mentions(t)
        r = p.parse(t)
        c["dễ vỡ (kết quả hiện tại)"] += bool(r["fragile"])
        for s, acc in zip(plain, plain_acc):
            if s in p.phrase: continue
            s2 = disambiguate(s, acc)
            parts = [x.strip() for x in re.split(r"[:,] ", s2) if x.strip()]
            pos = [bool(re.search(FRAGILE_POS, x)) and not re.search(FRAGILE_NEG, x) for x in parts]
            neg = [bool(re.search(FRAGILE_NEG, x)) for x in parts]
            if len(parts) == 1:
                if re.search(FRAGILE_POS, s2) and re.search(FRAGILE_NEG, s2): c["1 vế: vừa có từ khóa dễ vỡ vừa khớp luật không-dễ-vỡ"] += 1
                continue
            if not (any(pos) and any(neg)): continue
            seq = [("V" if a else "K") for a, b in zip(pos, neg) if a or b]
            order = "dễ vỡ TRƯỚC, không-dễ-vỡ SAU" if seq[0] == "V" and seq[-1] == "K" else ("không-dễ-vỡ TRƯỚC, dễ vỡ SAU" if seq[0] == "K" and seq[-1] == "V" else "khác")
            P = parts[pos.index(True)]; N = parts[neg.index(True)]
            c[f"{len(parts)} vế trộn | {order} | hiện đang coi là dễ vỡ: {bool(r['fragile'])}"] += 1
            c[f"    [{order}] vế dễ vỡ có từ nhượng bộ/nghi ngờ"] += bool(re.search(CONTRA, P))
            c[f"    [{order}] vế dễ vỡ có từ nhấn mạnh"] += bool(re.search(EMPH, P))
            c[f"    [{order}] vế không-dễ-vỡ có từ nhượng bộ/nghi ngờ"] += bool(re.search(CONTRA, N))
            c[f"    [{order}] số từ (vế dễ vỡ, vế không-dễ-vỡ) = ({len(P.split())}, {len(N.split())})"] += 1
    print(f"\n== {sp}: {n}")
    for k in sorted(c): print(f"   {c[k]:4d} ({c[k] / n:5.1%})  {k}")
