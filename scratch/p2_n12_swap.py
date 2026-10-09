"""Phép thử TÊN MỚI trên CÂU THẬT của validation: thay cụm tên đích / điểm ghé (tên đã biết, tìm bằng từ điển) bằng TÊN MỚI TỰ SOẠN cùng
loại (không dấu), giữ nhãn; đọc bằng bộ đọc chỉ học train trên bản đồ thật; đo đúng loại đích / điểm ghé và phân loại lỗi.
    python scratch/p2_n12_swap.py [số lượt=2] [seed]"""
import sys, random, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_lib import *
import nlp3
from nlp import norm, tokens
from pipeline_p2 import legs_from_mission
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 2; random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
# tên mới tự soạn (kiến thức chung), không dấu
NOVEL = {
    "library": ["trung tam hoc lieu", "phong luu tru tai lieu", "kho sach truong", "phong muon tra sach", "nha sach thu vien so",
                "phong tu lieu khoa", "khu doc tai lieu"],
    "dorm": ["khu luu tru sinh vien", "nha nghi sinh vien", "day nha noi tru b", "khu nha o tap the", "toa ky tuc c",
             "lang sinh vien", "phong ngu sinh vien"],
    "sports": ["nha tap the duc", "san van dong nho", "khu ren luyen the chat", "phong gym truong", "san bong ro",
               "nha thi dau da nang", "khu tap luyen"],
    "clinic": ["phong cham soc suc khoe", "diem so cuu", "phong y te hoc duong", "tram cuu thuong", "phong kham da khoa truong",
               "khu dieu tri", "phong bac si truc"],
    "canteen": ["khu am thuc sinh vien", "quan com truong", "phong an chung", "nha an tap the", "khu phuc vu suat an",
                "bep an sinh vien", "quay giai khat"],
    "parking": ["diem do phuong tien", "khu de xe may", "bai giu xe dap", "nha de xe sinh vien", "khu dau o to", "ham gui xe",
                "san de xe"],
    "lecture": ["toa nha hoc chung", "khu len lop", "giang duong lon a", "phong hoc ly thuyet", "hoi truong hoc", "day phong hoc b",
                "nha hoc chinh"],
    "lab": ["phong nghien cuu", "phong thuc hanh hoa", "xuong thi nghiem", "khu thuc nghiem", "phong lab vat ly",
            "phong may thuc hanh", "trung tam thi nghiem"],
    "office": ["phong cong tac sinh vien", "khu hanh chinh tong hop", "van phong nha truong", "phong quan ly dao tao",
               "phong tiep nhan ho so", "ban giam hieu", "phong tai vu"],
    "gate": ["cong chinh truong", "loi ra vao", "chot bao ve so 2", "cong phu", "tram kiem soat ra vao", "cong sau truong",
             "cua vao chinh"],
}
p = nlp3.MissionParser3.load(OUT / "nlp3_trainonly.pkl")
lex = p.p2.lex
D = load("validation")
c = collections.Counter(); ex = collections.defaultdict(list)
for rep in range(NREP):
    for x in D:
        m = x["m"]; w = x["w"]; L = w["landmarks"]
        toks = tokens(norm(m["text"]))
        hits = lex.find(toks)
        targets = {m["goal"]} | ({m["via"]} if m["via"] else set())
        repl = {}
        for i, j, t, a in hits:
            if t in targets:
                repl[(i, j)] = random.choice(NOVEL[t])
        if not repl:
            continue
        import re
        text = norm(m["text"])
        for (i, j), nm in sorted(repl.items(), key=lambda z: -z[0][0]):
            old = " ".join(toks[i:j])
            text = re.sub(r"(?<![a-z0-9])" + re.escape(old) + r"(?![a-z0-9])", nm, text, count=1)
        r = p.parse(text, {t for t, v in L.items() if v}, L)
        mm = nlp3.resolve3(r, L, w); legs = legs_from_mission(w, mm)
        tl = [sorted(map(tuple, l)) for l in x["legs"]]; pl = [sorted(map(tuple, l)) for l in legs]
        okg = pl[-1] == tl[-1]
        okv = (len(pl) == 2) == (len(tl) == 2) and (len(pl) < 2 or pl[0] == tl[0])
        c["n"] += 1; c["đích đúng"] += okg; c["ghé đúng"] += okv; c["cả hai đúng"] += okg and okv
        if not okg:
            k2 = "đích: lấy nhầm điểm ghé" if m["via"] and mm["goal"] == m["via"] else "đích: sai loại khác"
            c[k2] += 1; ex[k2].append(text)
        if not okv:
            k2 = "ghé: mất" if m["via"] and len(pl) < 2 else "ghé: thừa" if not m["via"] else "ghé: sai loại"
            c[k2] += 1; ex[k2].append(text)
print({k: (v if k == "n" else round(v / c["n"], 3)) for k, v in sorted(c.items())})
for k, v in ex.items():
    print("==", k)
    for t in v[:3]:
        print("   ", t)
