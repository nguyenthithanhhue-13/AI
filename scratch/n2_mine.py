"""Khai thác từ điển tên gọi địa điểm (alias -> loại) từ dữ liệu có nhãn.

Bước 1: lấy chuỗi alias từ các khung câu gây nhiễu (rất sạch: 'khong can ghe X', ...).
Bước 2: gán loại cho alias bằng thống kê: trong những câu KHÔNG phải câu gây nhiễu, alias xuất hiện
        thì loại của nó phải thuộc {goal, via, anchor} của cảnh đó.
"""
import json, re, sys, unicodedata
from collections import Counter, defaultdict
D = 'delivery_public/'
splits = sys.argv[1:] or ['train']


def norm(t):
    t = t.lower().replace('đ', 'd')
    t = unicodedata.normalize('NFD', t)
    return ''.join(c for c in t if unicodedata.category(c) != 'Mn')


def sents(t):
    t = re.sub(r'\[don #\d+\]', ' ', norm(t))
    return [re.sub(r'\s+', ' ', p).strip(' ,') for p in re.split(r'[.!?]+', t) if p.strip(' ,')]


DIS = [r'^khong can ghe (.+)$', r'^dung nham voi (.+) nhe$', r'^hom qua da giao o (.+) roi$',
       r'^nguoi nhan da roi (.+) roi$', r'^(.+) khong phai diem nhan$', r'^bo qua (.+), khong phai o do$']
PREFIX = r'^(robot oi|yeu cau moi|nho ban nhe|xin chao|chao robot|nhan robot)[,:]? '

scenes = []
for sp in splits:
    scenes += json.load(open(D + sp + '/scenes.json', encoding='utf-8'))

alias_cnt = Counter()
rest = []   # (các câu không phải gây nhiễu, tập loại hợp lệ)
for s in scenes:
    m = s['mission']
    types = {m['goal']}
    if m['via']: types.add(m['via'])
    for k in ('goal_ref', 'via_ref'):
        if m[k] and m[k]['anchor']: types.add(m[k]['anchor'])
    keep = []
    for x in sents(m['text']):
        x = re.sub(PREFIX, '', x)
        for pat in DIS:
            mm = re.match(pat, x)
            if mm:
                alias_cnt[mm.group(1)] += 1
                break
        else:
            keep.append(x)
    rest.append((' . '.join(keep), types))

print('alias strings from distractor frames:', len(alias_cnt))
lex = {}
unk = []
for a, n in alias_cnt.items():
    tc = Counter(); tot = 0
    pat = re.compile(r'(?<![a-z])' + re.escape(a) + r'(?![a-z])')
    for text, types in rest:
        if pat.search(text):
            tot += 1
            for t in types: tc[t] += 1
    if tot == 0:
        unk.append((a, n)); continue
    t, c = tc.most_common(1)[0]
    second = tc.most_common(2)[1][1] if len(tc) > 1 else 0
    lex[a] = (t, c, tot, second)
by = defaultdict(list)
for a, (t, c, tot, second) in lex.items():
    by[t].append(f'{a}({c}/{tot};{second})')
for t, v in by.items():
    print(t, len(v), ' | '.join(sorted(v)))
print('no type evidence:', unk)
