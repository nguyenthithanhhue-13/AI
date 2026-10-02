"""Thống kê các câu con (đã bỏ dấu) để tìm khung câu và cụm gấp/dễ vỡ."""
import json, re, sys, unicodedata
from collections import Counter, defaultdict
D = 'delivery_public/'


def norm(t):
    t = t.lower().replace('đ', 'd')
    t = unicodedata.normalize('NFD', t)
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return t


def sents(t):
    t = norm(t)
    t = re.sub(r'\[don #\d+\]', ' ', t)
    parts = re.split(r'[.!?]+', t)
    return [re.sub(r'\s+', ' ', p).strip(' ,') for p in parts if p.strip(' ,')]


for sp in ['train', 'validation']:
    sc = json.load(open(D + sp + '/scenes.json', encoding='utf-8'))
    cnt = Counter(); U = Counter(); F = Counter()
    for s in sc:
        for x in set(sents(s['mission']['text'])):
            cnt[x] += 1; U[x] += s['mission']['urgent']; F[x] += s['mission']['fragile']
    print('=====', sp, 'unique sentences', len(cnt))
    for x, n in cnt.most_common(150 if sp == 'train' else 110):
        print(f'{n:5d} U={U[x] / n:.2f} F={F[x] / n:.2f} | {x}')
