"""(vòng private test) Nơi giao CHỈ được mô tả qua bản đồ.

Hai dạng (scenes.json: goal_ref.kind):
  - north_most / south_most / west_most / east_most: địa điểm có hàng / cột cực trị trong MỌI địa điểm trên bản đồ
    ("địa điểm xa nhất về phía bắc", "chỗ ở trên cùng bản đồ", "địa điểm nằm cao nhất trên bản đồ", "địa điểm ngoài cùng
    bên trái", "chỗ nằm sát mép phải nhất", "địa điểm ở rìa trái nhất"...).
  - anchor_near: địa điểm (khác mốc) gần mốc nhất theo khoảng cách Manhattan ("nơi cạnh nhà ăn nhất", "chỗ nằm sát X nhất",
    "địa điểm kề X nhất", "địa điểm gần X nhất").
Đo trên train + validation (scratch/p2_n1_mapref.py): vị trí đúng = cực trị duy nhất 100%, gần mốc nhất theo Manhattan 100%
(đường chim bay sai 5 / 723 cảnh train); nhận diện đúng 300/300 cảnh validation, 2042/2043 train.

Nhận diện dựa vào CẤU TRÚC chứ không dựa vào câu mẫu: đầu ngữ CHUNG (địa điểm / chỗ / nơi / điểm / vị trí ...) + từ chỉ hướng
hoặc từ chỉ khoảng cách gần + "nhất" / "cùng". Tham chiếu chọn bản của loại cũ ("thư viện phía bắc", "X gần Y hơn") có đầu ngữ
là TÊN địa điểm cụ thể và không có "nhất", nên không bị bắt.
Chữ được chuẩn hóa: bỏ dấu, chữ thường, sửa lỗi gõ một chữ (đảo / rơi / thừa / thay) cho các từ khóa dài, chỉ khi từ đó hiếm
trong dữ liệu huấn luyện (xem build_vocab).
"""
import re
import unicodedata

KINDS_MOST = ("north_most", "south_most", "west_most", "east_most")


def unaccent(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def _unaccent_keep(s):
    """Bỏ dấu GIỮ NGUYÊN độ dài chuỗi (mỗi ký tự -> một ký tự) để ánh xạ vị trí về câu gốc."""
    out = []
    for c in s:
        c = c.lower() if len(c.lower()) == 1 else c
        if c == "đ":
            out.append("d")
        else:
            d = unicodedata.normalize("NFD", c)
            out.append(d[0] if d else c)
    return "".join(out)


# chỉ sửa lỗi gõ cho từ khóa đủ dài (>= 4 chữ cái) để không nuốt từ thật
LONG = ["nhat", "cung", "phia", "huong", "trai", "phai", "tren", "duoi", "ngoai", "diem", "canh", "thap", "dong"]
REAL = {"nhac", "nhan", "nhanh", "nhau", "nham", "phai", "phia", "trai", "trong", "tren", "tran", "duoc", "dung", "dong",
        "dang", "diem", "dien", "dieu", "canh", "cach", "thap", "thuc", "thay", "thoi", "ngoai", "ngay", "nguoi", "huong",
        "hong", "hang", "duoi", "doi", "doan", "hai", "phong", "chang", "nhung", "trung", "truong", "chung", "nhieu",
        "phut", "thang", "trang", "trao", "cung", "rang", "tang", "dinh", "nhap", "nhat", "chat", "that", "khat",
        "tri", "tan", "cuc", "goc", "khu", "lan", "can", "vach", "ben", "toa", "tai", "phan", "hon"}
VOCAB = {}      # tần suất từ (không dấu) trong dữ liệu huấn luyện


def build_vocab(texts):
    """Tần suất từng từ (không dấu). Một từ chỉ bị "sửa" thành từ khóa cách nó một lỗi gõ khi nó hiếm (< 3 lần) hoặc từ khóa
    phổ biến hơn nó >= 30 lần (lỗi gõ của dữ liệu là đảo / rơi chữ lẻ tẻ; từ thật như "tập", "xưởng" không bao giờ bị sửa)."""
    from collections import Counter
    return dict(Counter(t for x in texts for t in re.findall(r"[a-z0-9]+", unaccent(x))))


def _dl1(a, b):
    """True nếu a, b cách nhau đúng một phép đảo hai chữ liền nhau / xóa / chèn / thay."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def norm_tokens(text, vocab=None):
    vocab = VOCAB if vocab is None else vocab
    toks = re.findall(r"[a-z0-9]+", unaccent(text))
    out = []
    for t in toks:
        if t in REAL or len(t) < 3:
            out.append(t)
            continue
        c = [k for k in LONG if _dl1(t, k)]
        n = vocab.get(t, 0)
        out.append(c[0] if len(c) == 1 and (n < 3 or vocab.get(c[0], 0) >= 30 * n) else t)
    return out


HEAD = r"(?:dia diem|diem|cho|noi|vi tri|khu vuc|khu|toa nha|toa|dia chi|diem den|dich den|cong trinh|o)"
FILL = r"(?:\w+ ){0,3}?"          # vài từ đệm ("nằm", "ở", "mà", "đang"...)
NOT_NHAT = r"(?! (?:dinh|quyet|tri|loat|thiet|thoi))"
DIR_MOST = [
    # "xa nhất về phía bắc", "xa nhất phía bắc", "phía bắc nhất", "cực bắc" (cho phép một từ gõ hỏng thay "phía")
    r"\bxa nhat (?:ve )?(?:phia |huong |mien |ben |\w{1,4} )?(bac|nam|tay|dong)\b",
    r"\bxa nhat (?:ve )?(?:phia |huong |ben )(trai|phai|tren|duoi)\b",
    r"\b(?:phia|huong|mien|ben) (bac|nam|tay|dong) nhat\b" + NOT_NHAT,
    r"\bcuc (bac|nam|tay|dong)\b",
    # "tận cùng phía bắc", "tận cùng bên trái"
    r"\btan cung (?:ve )?(?:phia |huong |ben |mien )?(bac|nam|tay|dong|trai|phai|tren|duoi)\b",
    # "rìa trái cùng", "mép dưới cùng"
    r"\b(?:mep|ria|bia|goc|canh|ben|phia|phan) (trai|phai) cung\b",
    # "sát mép trái bản đồ nhất"
    r"\b(?:mep|ria|bia|canh|ben|phia|phan) (trai|phai|tren|duoi) (?:cua )?ban do nhat\b",
    # "tít phía bắc", "ở đầu phía bắc", "cuối phía nam", "đỉnh bản đồ" (= trên cùng), "phía bắc hơn tất cả / hơn mọi nơi"
    r"\btit (?:phia |ve phia |huong |ben )?(bac|nam|tay|dong|tren|duoi|trai|phai)\b",
    r"\b(?:dau|cuoi) (?:phia|huong|mien) (bac|nam|tay|dong)\b",
    r"\b(?:phia|huong|ben|mep|ria) (bac|nam|tay|dong|tren|duoi|trai|phai) (?:hon tat ca|hon moi|bac nhat|hon het)\b",
    r"\b(?:xa|cao|thap) (?:hon tat ca|hon moi noi|bac nhat) (?:ve )?(?:phia |huong )?(bac|nam|tay|dong|tren|duoi|trai|phai)\b",
    # "trên cùng bản đồ", "dưới cùng", "ngoài cùng bên trái", "bên trái cùng"
    r"\b(tren|duoi|ten|ren|dui) cung\b",
    r"\bngoai cung (?:ben |phia |phan |\w{1,4} )?(trai|phai|tren|duoi|tari|rai|hai)\b",
    r"\b(?:ben|phia) (trai|phai) cung\b",
    # "cao nhất / thấp nhất (trên bản đồ)"
    r"\b(cao|thap) nhat\b" + NOT_NHAT,
    # "sát mép trái nhất", "ở rìa phải nhất", "mép trên nhất", "góc trên nhất", "bên trái nhất", "phía trên nhất"
    r"\b(?:mep|ria|bia|goc|canh|ben|phia|phan|bien|man) (trai|phai|tren|duoi|rai|hai|ren) nhat\b" + NOT_NHAT,
    r"\b(trai|phai) nhat\b" + NOT_NHAT,
]
DIRMAP = {"bac": "north_most", "tren": "north_most", "ren": "north_most", "ten": "north_most", "cao": "north_most",
          "nam": "south_most", "duoi": "south_most", "dui": "south_most", "thap": "south_most",
          "tay": "west_most", "trai": "west_most", "tari": "west_most", "rai": "west_most",
          "dong": "east_most", "phai": "east_most", "hai": "east_most"}
# dài trước ngắn (luân phiên regex lấy nhánh đầu tiên khớp)
PROX = r"(?:sat ben|ke ben|ngay canh|ngay sat|sat vach|lien ke|sat canh|gan ben|lan can|gan voi|canh voi|sat voi|gan|canh|sat|ke)"
ANC_BAD = r"(?:hon|nua|qua|mep|ria|bia|goc|ben|phia|phan|huong|bien|man|gan|sat|nhat)\b"


def find(text, vocab=None):
    """Tìm mô tả nơi giao qua bản đồ. Trả về dict(kind, anchor, anchor_span, span) với span = (đầu, cuối) theo KÝ TỰ trong
    câu gốc (từ đầu ngữ chung tới hết cụm, kèm "trên bản đồ" nếu có), hoặc None.
    Dạng cực trị được xét trước (để "sát mép trái nhất" không bị hiểu là "gần mốc 'mép trái'")."""
    vocab = VOCAB if vocab is None else vocab
    keep = _unaccent_keep(text)
    raw = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+", keep)]
    toks = norm_tokens(" ".join(t for t, _, _ in raw), vocab)
    if len(toks) != len(raw):        # không khớp (ký tự lạ): bỏ qua vị trí, chỉ trả về loại
        raw = None
    s = " ".join(toks)
    starts = []
    pos = 0
    for t in toks:
        starts.append(pos)
        pos += len(t) + 1

    def tok_at(ch):
        k = 0
        while k + 1 < len(starts) and starts[k + 1] <= ch:
            k += 1
        return k

    def orig(i, j):
        return (raw[i][1], raw[j][2]) if raw else None

    def tail(j):
        """Kéo dài qua "(trên|ở) bản đồ" ngay sau cụm."""
        if toks[j + 1:j + 4] in (["tren", "ban", "do"], ["o", "ban", "do"]):
            return j + 3
        if toks[j + 1:j + 3] == ["ban", "do"]:
            return j + 2
        return j

    for pat in DIR_MOST:
        for m in re.finditer(pat, s):
            i0, j0 = tok_at(m.start()), tok_at(m.end() - 1)
            lo = max(0, i0 - 6)
            heads = [k for k in range(lo, i0) if re.match(HEAD + r"\b", " ".join(toks[k:i0]))]
            if not heads:
                continue
            h = heads[-1]
            # "địa điểm": lấy cả chữ "địa"
            if h > 0 and toks[h] == "diem" and toks[h - 1] == "dia":
                h -= 1
            return {"kind": DIRMAP[m.group(1)], "anchor": None, "anchor_span": None, "span": orig(h, tail(j0))}
    ANCH = [r"\b" + HEAD + r" " + FILL + PROX + r" ((?:\w+ ){1,7}?)(?:nhat|hon tat ca|hon moi noi khac|hon moi noi)\b" + NOT_NHAT,
            r"\b" + HEAD + r" " + FILL + r"cach ((?:\w+ ){1,7}?)gan nhat\b",               # "địa điểm cách X gần nhất"
            r"\b" + HEAD + r" " + FILL + PROX + r" ((?:\w+ ){1,7}?)hon ca\b"]               # "nơi gần X hơn cả"
    for pat in ANCH:
        for m in re.finditer(pat, s):
            anc = m.group(1).strip()
            if re.match(ANC_BAD, anc):
                continue
            i0, j0 = tok_at(m.start()), tok_at(m.end() - 1)
            a0, a1 = tok_at(m.start(1)), tok_at(m.start(1) + len(anc) - 1)
            return {"kind": "anchor_near", "anchor": anc, "anchor_span": orig(a0, a1), "span": orig(i0, j0)}
    if not RELAX:
        return None
    # Lượt 2 (NỚI): không cần đầu ngữ chung. Trên train + validation, tham chiếu chọn bản ("X phía bắc", "X gần Y hơn") không bao
    # giờ dùng "nhất" / "cùng" -> cụm so sánh nhất đứng một mình vẫn là mô tả nơi giao qua bản đồ (0 bắt nhầm trên 2300 câu).
    for pat in DIR_MOST:
        for m in re.finditer(pat, s):
            i0, j0 = tok_at(m.start()), tok_at(m.end() - 1)
            if m.group(1) in ("cao", "thap") and not ((i0 > 0 and toks[i0 - 1] in ("nam", "o", "tri")) or toks[j0 + 1:j0 + 4] in
                                                       (["tren", "ban", "do"], ["o", "ban", "do"]) or toks[j0 + 1:j0 + 3] == ["ban", "do"]):
                continue      # "ưu tiên cao nhất", "mức thấp nhất" không phải vị trí
            h = i0
            while h > 0 and toks[h - 1] in ("nam", "o", "tai", "sat", "ngay", "phia", "ben", "ve", "xa") and h > i0 - 3:
                h -= 1
            return {"kind": DIRMAP[m.group(1)], "anchor": None, "anchor_span": None, "span": orig(h, tail(j0)), "relaxed": True}
    for pat in (PROX + r" ((?:\w+ ){1,7}?)(?:nhat|hon ca|hon het)\b" + NOT_NHAT, r"\bcach ((?:\w+ ){1,7}?)gan nhat\b"):
        for m in re.finditer(r"\b" + pat, s):
            anc = m.group(1).strip()
            if re.match(ANC_BAD, anc):
                continue
            i0, j0 = tok_at(m.start()), tok_at(m.end() - 1)
            a0, a1 = tok_at(m.start(1)), tok_at(m.start(1) + len(anc) - 1)
            h = i0
            while h > 0 and toks[h - 1] in ("nam", "o", "tai", "ngay") and h > i0 - 2:
                h -= 1
            return {"kind": "anchor_near", "anchor": anc, "anchor_span": orig(a0, a1), "span": orig(h, j0), "relaxed": True}
    if GENERIC:
        g = _generic(toks, raw, keep)
        if g:
            i, j, kind, a = g
            return {"kind": kind, "anchor": " ".join(toks[a[0]:a[1] + 1]) if a else None,
                    "anchor_span": orig(*a) if a else None, "span": orig(i, tail(j)), "generic": True}
    return None


# ---- Lượt 3 (TỔNG QUÁT, chỉ chạy khi hai lượt trên không bắt được gì): theo cấu trúc chứ không theo mẫu câu.
# Trong MỘT vế câu: dấu so sánh nhất (nhất / cùng sau từ hướng / hơn cả / hơn tất cả / cực / tận) + từ chỉ hướng (có từ dẫn
# "phía / hướng / mạn / mép / rìa / biên ...") -> cực trị; + từ chỉ độ gần (gần / sát / kề / cạnh / cách ... ngắn / láng giềng)
# -> gần mốc nhất, mốc = cụm danh từ trước dấu so sánh ("gần X nhất") hoặc sau từ nối ("gần nhất với / so với / tính từ X").
GENERIC = True
DIRW = {"bac": "north_most", "nam": "south_most", "tay": "west_most", "dong": "east_most", "tren": "north_most",
        "duoi": "south_most", "trai": "west_most", "phai": "east_most"}
DIRPRE = {"phia", "huong", "mien", "man", "cuc", "ve", "theo", "tan", "ria", "mep", "bo", "dau", "cuoi", "ben", "phan", "bien",
          "canh", "goc", "tit", "vung", "khu", "day", "hang", "cot", "le"}
PROXW = {"gan", "sat", "ke", "canh", "lien", "lang", "cach", "ngan", "giap", "ngay"}
NOT_ANC = {"dia", "diem", "noi", "cho", "vi", "tri", "khu", "vuc", "o", "nam", "gan", "sat", "ke", "canh", "ben", "lien",
           "lang", "gieng", "cach", "khoang", "toi", "den", "voi", "so", "tinh", "tu", "cua", "hon", "ca", "tat", "moi",
           "khac", "it", "buoc", "ngan", "quang", "mot", "ngoai", "tren", "ban", "do", "co", "la", "ma",
           "dang", "suon", "nach", "vach", "ngay", "xom", "giap", "nhu", "toa", "cong", "trinh", "nha", "can", "doan", "duong",
           "di", "ve", "phia", "nhat", "ke", "nhung", "va", "thi", "cho"}
ANC_STOP = {"giup", "nhe", "dang", "truoc", "roi", "xong", "de", "hang", "nho", "lay", "nhan", "cho", "va",
            "ma", "cam", "giao", "mang", "chuyen", "dua", "nhe", "luon", "ngay", "gap", "nhanh", "di", "nhung"}
ROUTE = {"duong", "loi", "tuyen", "nhanh", "som"}


def _generic(toks, raw, keep):
    n = len(toks)
    # ranh giới vế câu theo dấu câu trong câu gốc
    cut = [False] * n
    if raw:
        for k in range(n - 1):
            if re.search(r"[.,;:!?()\[\]\"]", keep[raw[k][2]:raw[k + 1][1]]):
                cut[k] = True
    clauses, a = [], 0
    for k in range(n):
        if cut[k] or k == n - 1:
            clauses.append((a, k)); a = k + 1
    for c0, c1 in clauses:
        T = toks[c0:c1 + 1]
        for k, t in enumerate(T):
            sup = (t == "nhat" and not (k > 0 and T[k - 1] in ("duy", "thong", "hop", "van", "chac", "sinh"))
                   and not (k > 1 and T[k - 1] == "dong" and T[k - 2] not in DIRPRE | {"nam", "tri", "o"})
                   and not (k + 1 < len(T) and T[k + 1] in ("dinh", "quyet", "tri", "loat", "thiet", "thoi", "la"))) or \
                  (t == "ca" and k > 0 and T[k - 1] in ("hon", "tat")) or (t == "cung" and k > 0 and T[k - 1] in set(DIRW) | {"ngoai", "tan"}) \
                  or (t in ("tan", "tit") and any(x in DIRW for x in T[k + 1:k + 4]))
            if not sup:
                continue
            lo, hi = max(0, k - 8), min(len(T), k + 7)
            # ---- hướng
            best = None
            for q in range(lo, hi):
                w = T[q]
                if w not in DIRW:
                    continue
                pre = T[q - 1] if q > 0 else ""
                nx = T[q + 1] if q + 1 < len(T) else ""
                if pre in ("ben", "phia") and w == "tay" and nx in ("trai", "phai"):
                    continue          # "bên tay phải"
                ok = pre in DIRPRE or (nx in ("nhat", "cung", "xa", "hon") and w in ("bac", "trai", "duoi", "tay", "tren")) \
                    or (pre == "cuc")
                if w in ("nam", "dong", "phai", "tren", "tay") and pre not in DIRPRE:
                    ok = (ok and w in ("tren", "tay") and nx in ("nhat", "cung")) or \
                        (nx in ("nhat", "cung") and pre in ("nam", "tri", "o", "la", "diem", "cho", "noi"))
                if ok and (best is None or abs(q - k) < abs(best - k)):
                    best = q
            prox = [q for q in range(lo, k) if T[q] in PROXW] + [q for q in range(k + 1, hi) if T[q] in PROXW and q <= k + 1]
            route = any(T[q] in ROUTE or (T[q] == "trinh" and T[q - 1] == "lo") for q in range(max(0, k - 4), k))
            if best is not None and not route:
                q0 = min(best, k)
                h = _head(T, q0)
                return c0 + h, c0 + max(best, k), DIRW[T[best]], None
            itxa = [q for q in range(lo, k) if T[q] == "xa" and q > 0 and T[q - 1] == "it"]       # "ít xa X nhất"
            farw = any(T[q] == "xa" and not (q > 0 and T[q - 1] in ("it", "tuc")) for q in range(lo, k))   # "xa X nhất": không phải dạng gần
            if not route and not farw and (prox or itxa):
                p0 = min(prox + itxa)
                # mốc trước "nhất": giữa từ gần và "nhất"
                seg = [q for q in range(p0 + 1, k) if T[q] not in NOT_ANC]
                if seg:
                    a0 = seg[0]; a1 = seg[-1]
                    # kéo về trái nếu tên mốc mở đầu bằng từ chung ("nhà ăn", "khu giảng đường", "phòng y tế", "cổng trường")
                    while a0 - 1 > p0 and T[a0 - 1] in ("nha", "khu", "toa", "cong", "noi", "phong", "san", "bai", "tram", "day", "can"):
                        a0 -= 1
                    h = _head(T, p0)
                    return c0 + h, c0 + k, "anchor_near", (c0 + a0, c0 + a1)
                # mốc sau "nhất": "gần nhất với X", "gần nhất so với X", "gần nhất tính từ X", "gần nhất của X", "gần nhất cạnh X"
                q = k + 1
                while q < len(T) and T[q] in ("voi", "so", "tinh", "tu", "cua", "canh", "ben", "toi", "den", "ke", "sat", "o", "la"):
                    q += 1
                e = q
                while e < len(T) and e < q + 6 and T[e] not in ANC_STOP:
                    e += 1
                if e > q and q > k + 1:
                    h = _head(T, p0)
                    return c0 + h, c0 + e - 1, "anchor_near", (c0 + q, c0 + e - 1)
    # Lượt 4 (KHÔNG CÓ DẤU SO SÁNH NHẤT, "nhất" rơi / gõ hỏng / bị bỏ): đầu ngữ CHUNG đứng liền (qua "nằm / ở / ngay") từ chỉ độ
    # gần hoặc từ dẫn hướng. Train + validation: 9 / 9 cụm như vậy là "gần mốc nhất" (chữ "nhất" gõ hỏng), 0 cụm nghĩa khác.
    for c0, c1 in clauses:
        T = toks[c0:c1 + 1]
        for h in range(len(T)):
            hl = 2 if T[h:h + 2] in (["dia", "diem"], ["vi", "tri"], ["khu", "vuc"], ["toa", "nha"], ["cong", "trinh"]) else \
                1 if T[h] in ("cho", "noi") else 0
            if not hl or (h > 0 and T[h - 1] in ("dia", "vi", "khu", "toa", "cong")):
                continue
            q = h + hl
            while q < len(T) and T[q] in ("nam", "o", "ngay", "ma", "tai") and q < h + hl + 3:
                q += 1
            if q >= len(T):
                continue
            if T[q] in ("phia", "huong", "mep", "ria", "bien", "man", "tan", "tit", "cuc") or T[q:q + 2] == ["ve", "phia"]:
                r = q + (2 if T[q] == "ve" else 1)
                if r < len(T) and T[r] == "phia":
                    r += 1
                if r < len(T) and T[r] in DIRW and not (T[r] == "nam" and r + 1 < len(T) and T[r + 1] in DIRW):
                    return c0 + h, c0 + r, DIRW[T[r]], None
            if T[q] in ("gan", "sat", "ke", "canh", "lien", "giap"):
                a = q + 1
                while a < len(T) and T[a] in ("ke", "ben", "canh", "voi", "sat", "vach", "nach", "suon"):
                    a += 1
                e = a
                while e < len(T) and e < a + 6 and T[e] not in ANC_STOP and T[e] not in ("hon", "nhat"):
                    e += 1
                if e > a and T[a] not in ("mep", "ria", "bien", "phia"):
                    return c0 + h, c0 + e - 1, "anchor_near", (c0 + a, c0 + e - 1)
    return None


def _head(T, q0):
    """Đầu ngữ chung gần nhất trước q0 (trong 6 từ), hoặc q0."""
    for h in range(q0 - 1, max(-1, q0 - 7), -1):
        if T[h] in ("noi", "cho", "toa", "khu", "phong", "day", "cong") or (T[h] == "diem") or (T[h] == "vi" and h + 1 < len(T) and T[h + 1] == "tri"):
            if T[h] == "diem" and h > 0 and T[h - 1] == "dia":
                return h - 1
            return h
        if T[h] in ("den", "toi", "giao", "o", "cho", "tai", "la", "ve") and h < q0 - 1:
            return h + 1
    return q0


RELAX = True


def detect(text, vocab=None):
    """Dạng gọn của find: (kind, anchor, span) hoặc None."""
    f = find(text, vocab)
    return (f["kind"], f["anchor"], f["span"]) if f else None


def resolve(kind, landmarks, anchor_type=None):
    """Vị trí đích (hàng, cột) trên bản đồ, hoặc None. landmarks: {loại: [(r, c), ...]}."""
    allp = [(t, p) for t, v in landmarks.items() for p in v]
    if not allp:
        return None
    if kind in KINDS_MOST:
        key = {"north_most": lambda p: p[0], "south_most": lambda p: -p[0],
               "west_most": lambda p: p[1], "east_most": lambda p: -p[1]}[kind]
        best = min(key(p) for _, p in allp)
        return sorted(p for _, p in allp if key(p) == best)[0]
    if kind == "anchor_near":
        anc = landmarks.get(anchor_type) or []
        if len(anc) != 1:
            return None
        a = anc[0]
        others = [p for _, p in allp if p != a]
        if not others:
            return None
        d = lambda p: abs(p[0] - a[0]) + abs(p[1] - a[1])
        best = min(d(p) for p in others)
        return sorted(p for p in others if d(p) == best)[0]
    return None
