import json, random, sys
D = 'delivery_public/'
random.seed(1)
for sp, k in [('train', 45), ('validation', 45)]:
    sc = json.load(open(D + sp + '/scenes.json', encoding='utf-8'))
    print('=====', sp, 'avg len', sum(len(s['mission']['text']) for s in sc) / len(sc))
    for s in random.sample(sc, k):
        m = s['mission']
        gr = m['goal_ref'] and (m['goal_ref']['kind'], m['goal_ref']['anchor'])
        vr = m['via_ref'] and (m['via_ref']['kind'], m['via_ref']['anchor'])
        print(f"[g={m['goal']} {gr or ''} v={m['via']} {vr or ''} U={int(m['urgent'])} F={int(m['fragile'])}] {m['text']}")
