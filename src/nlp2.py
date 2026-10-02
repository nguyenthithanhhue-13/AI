"""NLP bản 2: đọc yêu cầu khi gặp CÁCH NÓI LẠ (tên gọi, khung câu, cụm từ chưa có trong dữ liệu huấn luyện).

So với nlp.py (chỉ nhận ra tên gọi đã có trong từ điển), bản này thêm:
  1. Bộ gán nhãn cụm địa điểm theo ngữ cảnh (nlp_tagger.py) -> tìm được cả tên gọi chưa từng gặp.
  2. Đoán LOẠI của tên gọi lạ: so n-gram ký tự với các tên đã biết + bảng từ khóa tiếng Việt viết tay
     (kiến thức ngôn ngữ chung, không lấy từ test). Kết quả là một phân bố xác suất; bước ghép sẽ chốt loại
     bằng cách chỉ xét những loại THẬT SỰ CÓ trên bản đồ.
  3. Ngữ cảnh của bộ phân loại vai trò được "khái quát hóa": từ hiếm (món hàng, người nhận...) thay bằng X.
  4. Luật từ khóa có xử lý phủ định cho gấp / dễ vỡ / phương hướng khi gặp cụm từ lạ.

Mọi thành phần học được chỉ học từ train (và validation ở bản cuối). Không dùng nội dung test.
"""
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nlp import (KINDS, PLACE_TYPES, ROLES, Lexicon, _DIS_TOK, dl_distance, mine_lexicon, ngram_counts, sentences, tokens)
from nlp_tagger import HEADS, SpanTagger, tag_tokens

# ---------------- kiến thức tiếng Việt viết tay (không dấu) ----------------
# từ khóa gợi ý loại địa điểm; số = trọng số
KEYWORDS = {
    "library": [(r"thu vien", 3), (r"\bsach\b", 2), (r"giao trinh", 2), (r"tai lieu", 1.5), (r"\bdoc\b", 1.5), (r"\bmuon\b", 1.5),
                (r"thu thu", 3), (r"tra cuu", 2), (r"tap chi", 2), (r"tu lieu", 2), (r"luu tru sach", 2), (r"hoc lieu", 3),
                (r"doc gia", 3), (r"sach bao", 2), (r"\bbao chi\b", 2), (r"an pham", 2)],
    "dorm": [(r"ky tuc", 3), (r"\bktx\b", 3), (r"noi tru", 3), (r"phong o\b", 2), (r"khu o\b", 2), (r"cho o\b", 2), (r"nha o\b", 2),
             (r"\bnoi o\b", 2), (r"luu tru", 1.5), (r"cu xa", 3), (r"phong ngu", 2), (r"o cua sinh vien", 2), (r"sinh vien o", 2),
             (r"\bngu\b", 1), (r"\bnghi\b", 1.5), (r"lang sinh vien", 3), (r"nha sinh vien", 3), (r"\btro\b", 2), (r"giuong", 2),
             (r"tap the", 1), (r"can ho", 2)],
    "sports": [(r"the thao", 3), (r"the chat", 3), (r"thi dau", 3), (r"\bsan\b", 1), (r"van dong", 2.5), (r"luyen tap", 2.5),
               (r"tap luyen", 2.5), (r"the duc", 3), (r"\bgym\b", 3), (r"\bbong\b", 2), (r"huan luyen", 2.5), (r"cau long", 3),
               (r"\bboi\b", 2), (r"doi bong", 3), (r"san tap", 3), (r"nha tap", 3), (r"phong tap", 3), (r"da nang", 2), (r"the hinh", 3),
               (r"the luc", 3), (r"ren luyen", 2.5), (r"bong (ro|chuyen|da|ban)", 3), (r"\bvo\b thuat", 3), (r"dien kinh", 3)],
    "clinic": [(r"y te", 3), (r"tram xa", 3), (r"\bkham\b", 2.5), (r"so cuu", 3), (r"bac si", 3), (r"y ta", 3), (r"suc khoe", 3),
               (r"cap cuu", 3), (r"\bthuoc\b", 2), (r"dieu duong", 3), (r"benh", 2), (r"y si", 3), (r"cham soc", 1.5), (r"y khoa", 3),
               (r"dieu tri", 3), (r"\btiem\b", 2), (r"\bduoc\b si", 3), (r"y vu", 3)],
    "canteen": [(r"can tin", 3), (r"cang tin", 3), (r"canteen", 3), (r"nha an", 3), (r"\bbep\b", 2.5), (r"an uong", 3), (r"\ban\b", 2),
                (r"bua", 2), (r"\bcom\b", 2), (r"quan an", 3), (r"am thuc", 3), (r"giai khat", 2.5), (r"cap duong", 3), (r"do an", 2),
                (r"suat an", 3), (r"do uong", 2), (r"ca phe", 2), (r"nau", 2), (r"an (trua|sang|toi)", 3), (r"food", 3)],
    "parking": [(r"\bxe\b", 2.5), (r"bai do", 3), (r"bai dau", 3), (r"gui xe", 3), (r"giu xe", 3), (r"trong xe", 3), (r"de xe", 3),
                (r"dau xe", 3), (r"do xe", 3), (r"ga ?ra", 3), (r"o to", 2), (r"xe (may|dap)", 3)],
    "lecture": [(r"giang duong", 3), (r"lop hoc", 3), (r"phong hoc", 3), (r"hoi truong", 3), (r"len lop", 3), (r"\blop\b", 2),
                (r"giang vien", 2.5), (r"\bgiang\b", 2), (r"tiet hoc", 3), (r"\bday\b hoc", 2), (r"bai giang", 3), (r"\bhoc\b", 1),
                (r"nha hoc", 3), (r"toa hoc", 3), (r"khu hoc", 3), (r"nghe giang", 3), (r"hoc tap", 2), (r"hoc ly thuyet", 3)],
    "lab": [(r"thi nghiem", 3), (r"\blab\b", 3), (r"thuc hanh", 3), (r"thuc nghiem", 3), (r"\bxuong\b", 2.5), (r"nghien cuu", 2.5),
            (r"ky thuat vien", 2.5), (r"thi nghiem vien", 3), (r"thuc tap", 2.5), (r"phong may", 2), (r"mau vat", 2), (r"hoa chat", 2)],
    "office": [(r"hanh chinh", 3), (r"van phong", 3), (r"dao tao", 3), (r"mot cua", 3), (r"hieu bo", 3), (r"giao vu", 3), (r"ho so", 2.5),
               (r"thu tuc", 3), (r"thu ky", 3), (r"giay to", 2), (r"tiep nhan", 2), (r"ke toan", 3), (r"tai vu", 3), (r"cong tac sinh vien", 3),
               (r"\bkhoa\b", 1.5), (r"\bnop\b", 1.5), (r"chuyen vien", 2.5), (r"van thu", 3), (r"hieu truong", 3), (r"giam hieu", 3),
               (r"tai chinh", 3), (r"nhan su", 3), (r"tuyen sinh", 3), (r"hoc vu", 3), (r"khao thi", 3), (r"phong ban", 3),
               (r"bo mon", 2), (r"dang ky", 2), (r"to chuc", 2), (r"tong hop", 2), (r"quan ly dao tao", 3)],
    "gate": [(r"\bcong\b(?! van| tac| nghe| viec)", 3), (r"loi vao", 3), (r"bao ve", 2.5), (r"\bchot\b", 2.5), (r"khach den", 3),
             (r"don khach", 3), (r"cua chinh", 3), (r"ra vao", 3), (r"loi ra", 2.5), (r"tiep don", 2), (r"\bkhach\b", 1.5),
             (r"cua vao", 3), (r"tram gac", 3), (r"\bgac\b", 2.5), (r"barie", 3), (r"le tan", 2), (r"tiep tan", 2)],
}
CANONICAL = {"library": ["thu vien", "tv"], "dorm": ["ky tuc xa", "ktx"], "sports": ["nha the thao", "tt"], "clinic": ["tram y te", "yt"],
             "canteen": ["can tin", "ca"], "parking": ["bai xe", "xe"], "lecture": ["giang duong", "gd"], "lab": ["phong thi nghiem", "tn"],
             "office": ["phong hanh chinh", "hc"], "gate": ["cong truong", "ct"]}
# từ chỉ phương hướng / quan hệ không gian (không phải tên địa điểm)
REF_WORDS = {"man", "phia", "huong", "ben", "goc", "dau", "tren", "duoi", "trai", "phai", "bac", "dong", "tay", "gan", "xa", "sat",
             "canh", "cach", "ke", "ban", "do", "hon", "nhat", "o", "nam", "ngay", "mien", "ve", "khu"}
# từ khung không bao giờ nằm ở đầu/cuối một tên địa điểm (từ chỉ hướng bac/nam/dong/tay... xử lý riêng theo cặp "phia dong")
EDGE_STRIP = {"man", "phia", "huong", "ben", "goc", "gan", "xa", "sat",
              "canh", "cach", "ke", "hon", "nhat", "o", "ngay", "ve", "truoc", "roi", "nhe", "giup", "minh", "lay", "nhan",
              "xong", "thi", "da", "khong", "dang", "can", "cho", "toi", "den", "qua", "ghe", "voi", "la", "va", "tai"}

URGENT_NEG = r"khong (can )?(gap|voi|khan|nhanh)|chua (can )?(gap|voi)|cu tu tu|thong tha|tu tu|luc nao (cung|giao)|bao gio cung|" \
             r"cung (duoc|kip)|khong (co )?gi (gap|voi)|chang (gap|voi)|khoi (gap|voi)|ranh (thi|luc)|khi nao (ranh|tien)|khong han chot|" \
             r"de sau|khong phai (gap|voi)|binh thuong thoi|khong uu tien|tien (thi|luc nao)"
# (bỏ dấu thì "ngay" = ngay/ngày nên chỉ nhận trong cụm rõ nghĩa "đi ngay", "ngay lập tức"...)
URGENT_POS = r"\bgap\b|\bkhan\b|hoa toc|\blap tuc\b|\btuc thi\b|cang (nhanh|som) cang tot|\buu tien\b|cham tre|" \
             r"trong \d+ phut|\b(di|can|giao|lam|chay|mang|den|toi|gui|chuyen) ngay\b|\bngay (lap tuc|bay gio|va luon|nhe|di)\b|\bngay$|" \
             r"nhanh (len|nhe|chan|nhat)|\bcap toc\b|khong (duoc|the) (tre|cham|lau|cho)|\bkip\b|som nhat|sat gio|" \
             r"\bcap bach\b|\btuc toc\b|keo (tre|muon|lo)|dung (de )?(tre|cham|lau)\b|sap (tre|het gio)|het gio|\bcho lau\b|" \
             r"dang (rat )?(sot ruot|nong ruot|voi)|\bvoi lam\b|\brat voi\b|\bthat nhanh\b|\bnhanh nhat\b|\bkhan truong\b"
FRAGILE_NEG = r"khong (de )?(vo|be|hong|nut)\b|khong lo (vo|be|hong)|chac chan|\bben\b|roi cung|va dap cung|khong so (vo|va|roi)|" \
              r"chang (so|lo)|kho (vo|hong)\b|\bcung cap\b|khong mong manh|khong can nhe"
FRAGILE_POS = r"\bde (vo|be|hong|nut|me)\b|thuy tinh|\bgom\b|\b(do|bang|gom) su\b|pha le|mong manh|\bky va\b|tranh va|nhe tay|can than|" \
              r"tranh rung|\bhang de\b|nang niu|khong duoc (roi|lac|va)\b|dung (lam roi|lac|quang)|cam (nem|quang)"
# từ mở đầu một MÓN HÀNG (hộp, khay, thẻ, túi...): "hộp thuốc", "thẻ thư viện" là hàng chứ không phải địa điểm
ITEM_HEADS = {"hop", "khay", "the", "tui", "thung", "tap", "bo", "goi", "chong", "ket", "mau", "chia", "so", "bien", "don",
              "binh", "lo", "cuon", "quyen", "to", "giay", "bia", "micro", "mu", "chan", "buu", "kien", "may", "bom", "linh",
              "nguyen", "hoa", "sach", "chai", "ly", "coc", "bang", "day", "cap", "vali", "phieu", "thu"}
# danh từ chỉ nơi chốn thường mở đầu tên địa điểm
PLACE_HEADS = {"phong", "khu", "noi", "nha", "san", "bai", "toa", "cho", "tram", "cong", "loi", "ham", "xuong", "quay", "kho",
               "sanh", "bep", "chot", "goc", "vuon", "trung", "tang", "vien", "quan", "lop", "hoi", "day"}
GEN_NEG = r"\b(khong|chang|cha|khoi|chua|dau can|dau co|dau phai|gi dau|lam gi|sao duoc)\b|\bdau$"
DOUBLE_NEG = r"khong (duoc|the|nen) (cham|tre|lau|cho|de|muon|roi|va|lac|nem|quang|xoc|lam roi)|khong the (cho|doi)|khong kip|" \
             r"khong chiu (duoc )?(va|rung|xoc)|khong (duoc|nen) (de )?(roi|va)"
DIRS = {"bac", "nam", "dong", "tay", "tren", "duoi", "trai", "phai"}
ORI_WORDS = {"phia", "ben", "man", "huong", "goc", "dau", "mien", "canh", "mep", "ria", "nua"}


from nlp import norm as _unaccent

CANONICAL_ACC = {"library": "thư viện", "dorm": "ký túc xá", "sports": "nhà thể thao", "clinic": "trạm y tế", "canteen": "căn tin",
                 "parking": "bãi xe", "lecture": "giảng đường", "lab": "phòng thí nghiệm", "office": "phòng hành chính", "gate": "cổng trường"}
_EMB = None


def embedder():
    """Mô hình nghĩa pretrained (e5-small, chạy offline). Dùng chung một bản cho cả tiến trình; không nằm trong file pickle."""
    global _EMB
    if _EMB is None:
        from embed import Embedder
        _EMB = Embedder()
    return _EMB


def save_embed_cache():
    if _EMB is not None:
        _EMB.save()


def keyword_scores(text):
    return np.array([sum(w for pat, w in KEYWORDS[t] if re.search(pat, text)) for t in PLACE_TYPES])


# câu "gây nhiễu": phủ định / chuyện đã qua -> địa điểm trong câu không phải đích cũng không phải điểm ghé
# (chú ý: bỏ dấu thì "đừng" và "dụng (cụ)" trùng nhau -> chỉ nhận "dung" khi theo sau là động từ)
NEG_SENT = r"\bkhong (can|phai|nen|ghe|den|toi|qua)\b|\bdung (nham|ghe|den|toi|qua|di|mang|giao|dua|re|vao|lai|tat|dung)\b|\bbo qua\b|\bkhoi (can|phai|ghe)\b|\bhom (qua|truoc|kia)\b|" \
           r"\bda (giao|roi|chuyen|nhan|xong|di)\b|\bnham\b|\bngoai tru\b|\bchang can\b|\bkhong lien quan\b|\b(lan|tuan) truoc\b"
_ORI = r"(phia|ben|man|huong|goc|dau|mien|canh|mep|ria|khu|nua)"


def rule_direction(ctx):
    """ctx: vài từ ngay sau tên địa điểm (đã che địa điểm kế tiếp thành PLC). -> kind hoặc None.
    Chỉ nhận khi từ chỉ hướng đi kèm từ định hướng ("phía", "bên", "mạn"...) hoặc "bản đồ"."""
    head = ctx.split(" PLC")[0] if "PLC" in ctx else ctx
    if "PLC" in ctx and len(head.split()) <= 4:
        neg = bool(re.search(r"\b(khong|chang)\b", head))        # "không xa X" = gần X
        if re.search(r"\b(cach xa|xa)\b", head):
            return "near" if neg else "far"
        if re.search(r"\b(gan|sat|canh|ke|ke ben|ben canh|lien ke|giap)\b", head):
            return "far" if neg else "near"
    for kind, w in (("north", "bac|tren"), ("south", "nam|duoi"), ("west", "tay|trai"), ("east", "dong|phai")):
        if re.search(_ORI + r" (" + w + r")\b", head) or re.search(r"\b(" + w + r") (cung )?(cua )?ban do", head) \
                or re.search(r"^(o |nam )?(" + w.replace("nam|", "") + r") cung\b", head):
            return kind
    return None


class MissionParser2:
    # ================= huấn luyện =================
    def fit(self, missions, verbose=False):
        self.lex = Lexicon(mine_lexicon(missions), ngram_counts(missions))
        cnt = Counter(t for m in missions for s in sentences(m["text"]) for t in tokens(s))
        self.freq = dict(cnt)
        # ---- bộ gán nhãn cụm địa điểm ----
        S, SP = [], []
        lb, rb = Counter(), Counter()
        for m in missions:
            for sent in self._sents(m["text"]):
                tt, words, idx = self._tt(sent)
                S.append(tt)
                found = self.lex.find(words)
                SP.append([(idx[a], idx[b - 1] + 1) for (a, b, t, al) in found])
                for (a, b, t, al) in found:
                    if a > 0:
                        lb[words[a - 1]] += 1
                    if b < len(words):
                        rb[words[b]] += 1
        # các từ hay đứng ngay trước / ngay sau một tên địa điểm = ranh giới của cụm
        self.left_bound = {w for w, c in lb.items() if c >= 5}
        self.right_bound = {w for w, c in rb.items() if c >= 5}
        self.tagger = SpanTagger().fit(S, SP)
        # ---- loại của một cụm tên gọi ----
        names, ytype = [], []
        for a, t in self.lex.alias2type.items():
            names.append(a); ytype.append(PLACE_TYPES.index(t))
        for t, lst in CANONICAL.items():
            for a in lst:
                names.append(a); ytype.append(PLACE_TYPES.index(t))
        self.v_t = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.type_clf = LogisticRegression(C=20, max_iter=3000).fit(self.v_t.fit_transform(names), ytype)
        # ---- vai trò + tham chiếu + bảng cụm từ + bộ phân loại cả câu ----
        rng = np.random.default_rng(0)
        Xl, Xr, Xs, yrole, Rr, ykind = [], [], [], [], [], []
        table = defaultdict(lambda: [0, 0, 0])
        G, yG = [], []
        alias_acc = defaultdict(set)      # tên gọi (không dấu) -> các dạng có dấu gặp trong dữ liệu
        phrase_acc = defaultdict(set)     # câu con (không dấu) -> các dạng có dấu
        main_acc = set()                  # các câu có nhắc địa điểm (dạng có dấu)
        ctx_rows = []                     # (ngữ cảnh quanh một lần nhắc đích/điểm ghé, loại)
        for m in missions:
            ments, plain, plain_acc = self._mentions(m["text"], known_only=True)
            for me in ments:
                alias_acc[me["text"]].add(" ".join(me["acc"][me["i"]:me["j"]]))
                main_acc.add(" ".join(me["acc"]))
            for s, sa in zip(plain, plain_acc):
                phrase_acc[s].add(sa)
            for s in plain:
                t = table[s]
                t[0] += 1; t[1] += bool(m["urgent"]); t[2] += bool(m["fragile"])
            G.append(self._main_text(m["text"])); yG.append(PLACE_TYPES.index(m["goal"]))
            roles = []
            for me in ments:
                toks, ms, i, t = me["toks"], me["ms"], me["i"], me["type"]
                role = None
                idx = [x[0] for x in ms].index(i)
                if any(re.match(p, " ".join(toks)) for p in _DIS_TOK):
                    role = "dis"
                elif idx > 0:
                    p = ms[idx - 1]
                    for key in ("goal", "via"):
                        ref = m[key + "_ref"]
                        if ref and ref["anchor"] == t and p[2] == m[key] and i - p[1] <= 3 \
                                and not re.search(r"\b(ghe|qua|lay|toi|den|truoc|roi)\b", " ".join(toks[p[1]:i])):
                            role = "anc"
                if role is None:
                    role = "goal" if t == m["goal"] else ("via" if t == m["via"] else "dis")
                roles.append(role)
            for k, (me, role) in enumerate(zip(ments, roles)):
                if role in ("goal", "via"):
                    ctx_rows.append((self._ctx_text(me), PLACE_TYPES.index(me["type"])))
                for copy in range(2):     # bản 2: che các từ hiếm thành X (giả lập món hàng / người nhận lạ)
                    left, right, sent, refc = self._ctx(me, drop=(copy == 1), rng=rng)
                    Xl.append(left); Xr.append(right); Xs.append(sent); yrole.append(ROLES.index(role))
                    if role in ("goal", "via"):
                        ref = m[role + "_ref"]
                        kind = 0
                        if ref:
                            if ref["anchor"]:
                                a = self._anchor_of(me)
                                if a is not None and a[2] == ref["anchor"]:
                                    kind = KINDS.index(ref["kind"])
                            else:
                                same = [q for q, x in enumerate(ments) if x["type"] == me["type"] and roles[q] == role]
                                if same[-1] == k:
                                    kind = KINDS.index(ref["kind"])
                        Rr.append(refc); ykind.append(kind)
        self.v_l = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=False)
        self.v_r = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=False)
        self.v_s = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"\S+", lowercase=False)
        X = hstack([self.v_l.fit_transform(["L " + x for x in Xl]), self.v_r.fit_transform(["R " + x for x in Xr]),
                    self.v_s.fit_transform(Xs)]).tocsr()
        self.role_clf = LogisticRegression(C=10, max_iter=3000).fit(X, yrole)
        self.v_k = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, lowercase=False)
        self.kind_clf = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(self.v_k.fit_transform(Rr), ykind)
        self.phrase = {}
        Sx, yS = [], []
        for s, (n, u, f) in table.items():
            if n >= 2:
                lab = 1 if u >= 0.9 * n and n >= 3 else (2 if f >= 0.9 * n and n >= 3 else 0)
                self.phrase[s] = lab
                Sx.append(s); yS.append(lab)
        self.v_p = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.phrase_clf = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(self.v_p.fit_transform(Sx), yS)
        self.v_g = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.goal_clf = LogisticRegression(C=10, max_iter=2000).fit(self.v_g.fit_transform(G), yG)
        # ---- hai bộ phân loại trên vector nghĩa (e5): loại của tên gọi, và gấp / dễ vỡ / trung tính của câu con ----
        # học cả dạng có dấu lẫn không dấu của mỗi tên gọi / câu con đã biết
        self.use_e5 = getattr(self, "use_e5", True)
        if self.use_e5:
            E = embedder()
            tx, ty = [], []
            for a, t in self.lex.alias2type.items():
                for v in {a} | alias_acc.get(a, set()):
                    tx.append(v); ty.append(PLACE_TYPES.index(t))
            from nlp_knowledge import PHRASES, PLACES
            if not getattr(self, "use_knowledge", True):      # thí nghiệm: không dùng ví dụ viết tay
                PHRASES, PLACES = {}, {t: [] for t in PLACE_TYPES}
            for t, v in CANONICAL_ACC.items():
                for s in {v, _unaccent(v)} | {x for a in PLACES[t] for x in (a, _unaccent(a))}:
                    tx.append(s); ty.append(PLACE_TYPES.index(t))
            self.e5_type = LogisticRegression(C=50, max_iter=3000).fit(E.encode(tx), ty)
            # 5 lớp: 0 trung tính, 1 gấp, 2 dễ vỡ, 3 không gấp, 4 không dễ vỡ
            px, py = [], []
            for s, lab in self.phrase.items():
                if lab == 0:
                    lab = 3 if re.search(URGENT_NEG, s) else (4 if re.search(FRAGILE_NEG, s) else 0)
                for v in {s} | phrase_acc.get(s, set()):
                    px.append(v); py.append(lab)
            for lab, lst in PHRASES.items():
                for a in lst:
                    for v in {a, _unaccent(a)}:
                        px.append(v); py.append(lab)
            # câu giao hàng / câu gây nhiễu (có nhắc địa điểm) cũng là "trung tính": lấy mẫu từ dữ liệu
            mrng = np.random.default_rng(1)
            pool = sorted(main_acc)
            for v in [pool[i] for i in mrng.choice(len(pool), min(400, len(pool)), replace=False)]:
                px.append(v); py.append(0)
            self.e5_phrase = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(E.encode(px), py)
            # loại địa điểm đoán từ NGỮ CẢNH quanh nó (món hàng, người nhận...), không nhìn tên: chứng cứ bổ sung cho tên lạ
            self.e5_ctx = LogisticRegression(C=20, max_iter=3000).fit(E.encode([c for c, _ in ctx_rows]), [t for _, t in ctx_rows])
            E.save()
        if verbose:
            print("lexicon", len(self.lex.alias2type), "| roles", Counter(ROLES[y] for y in yrole), "| kinds", Counter(KINDS[y] for y in ykind))
        return self

    # ================= tiện ích =================
    def _sents(self, text):
        # (đã thử tự sửa lỗi gõ cho mọi từ lạ: làm hỏng các từ MỚI hợp lệ như "xong" -> "cong", nên bỏ)
        return sentences(text)

    @staticmethod
    def _tt(sent):
        tt = tag_tokens(sent)
        idx = [i for i, t in enumerate(tt) if t not in (",", ":")]
        return tt, [tt[i] for i in idx], idx

    def _main_text(self, text):
        keep = [s for s in self._sents(text) if not re.search(NEG_SENT, s)]
        return " . ".join(keep)

    def span_probs(self, text, acc=None):
        """Phân bố loại địa điểm của một cụm tên gọi (lạ). Ba nguồn: n-gram ký tự so với tên đã biết,
        từ khóa viết tay, và vector nghĩa e5 (tin hơn khi cụm có dấu)."""
        p = self.type_clf.predict_proba(self.v_t.transform([text]))[0]
        full = np.full(10, 1e-3)
        full[self.type_clf.classes_] = p
        score = np.log(full + 0.03) + (0.0 if getattr(self, "ablate_rules", False) else 2.0) * keyword_scores(text)
        if getattr(self, "use_e5", False):
            s = acc if acc else text
            pe = np.full(10, 1e-3)
            pe[self.e5_type.classes_] = self.e5_type.predict_proba(embedder().encode([s]))[0]
            score += (1.0 if s != text else 0.5) * np.log(pe + 0.02)
        e = np.exp(score - score.max())
        return e / e.sum()

    @staticmethod
    def _ctx_text(me, before=7, after=5):
        """Các từ (có dấu) quanh một lần nhắc địa điểm, tên địa điểm thay bằng '___', cắt ở địa điểm kế bên."""
        acc, i, j, ms = me["acc"], me["i"], me["j"], me["ms"]
        lo = max([m[1] for m in ms if m[1] <= i] + [i - before, 0])
        hi = min([m[0] for m in ms if m[0] >= j] + [j + after, len(acc)])
        return " ".join(acc[lo:i]) + " ___ " + " ".join(acc[j:hi])

    @staticmethod
    def _accented(text, sents_words):
        """Với mỗi câu con (danh sách từ không dấu), trả về danh sách từ CÓ DẤU tương ứng lấy từ văn bản gốc."""
        raw = re.sub(r"\[(đơn|don) #\d+\]", " ", text.lower())
        acc = re.findall(r"[^\W_]+", raw)
        un = [_unaccent(t) for t in acc]
        out, p = [], 0
        for words in sents_words:
            row = []
            for w in words:
                q = next((k for k in range(p, min(p + 15, len(un))) if un[k] == w), None)
                if q is None:
                    row.append(w)
                else:
                    row.append(acc[q]); p = q + 1
            out.append(row)
        return out

    def _mentions(self, text, known_only=False):
        """-> (list dict(si, toks, acc, i, j, type|None, text, ms), các câu con không chứa địa điểm, dạng có dấu của chúng)."""
        out, plain, plain_acc = [], [], []
        sents = self._sents(text)
        acc_rows = self._accented(text, [self._tt(s)[1] for s in sents])
        for si, sent in enumerate(sents):
            tt, words, idx = self._tt(sent)
            acc_w = acc_rows[si]
            ms = list(self.lex.find(words))
            if not known_only:
                back = {ti: wi for wi, ti in enumerate(idx)}
                probs = self.tagger.predict(tt)
                pw = [probs[idx[k]] for k in range(len(words))]       # xác suất "thuộc tên địa điểm" theo từng từ
                # khớp mờ (sửa lỗi gõ) chỉ được nhận khi nó không phải một mẩu của tên địa điểm dài hơn
                # (vd "an tap" trong "phòng ăn tập thể" không phải "sân tập" gõ sai)
                ms = [m for m in ms if " ".join(words[m[0]:m[1]]) == m[3]
                      or not ((m[0] > 0 and pw[m[0] - 1] > 0.5) or (m[1] < len(words) and pw[m[1]] > 0.5))]
                known_tt = [(idx[a], idx[b - 1] + 1) for (a, b, t, al) in ms]
                used = set(k for m in ms for k in range(m[0], m[1]))
                # một từ "có thể thuộc tên địa điểm": từ lạ, hoặc từ quen mà bộ gán nhãn không chắc chắn là không phải
                maybe = lambda k: words[k] not in self.tagger.vocab or pw[k] >= 0.03
                for (a, b) in self.tagger.spans(tt, known_tt):
                    wi = [back[k] for k in range(a, b) if k in back]
                    if not wi:
                        continue
                    i, j = wi[0], wi[-1] + 1
                    if any(k in used for k in range(i, j)):
                        continue
                    # nới cụm sang các từ liền kề cho tới "từ ranh giới" (từ hay đứng ngay trước / sau tên địa điểm
                    # trong dữ liệu huấn luyện) hoặc dấu câu: từ cuối của tên lạ thường bị bỏ sót
                    steps = 0
                    while j < len(words) and steps < 4 and j not in used and idx[j] == idx[j - 1] + 1 \
                            and words[j] not in self.right_bound and maybe(j):
                        j += 1; steps += 1
                    steps = 0
                    while i > 0 and steps < 3 and (i - 1) not in used and idx[i] == idx[i - 1] + 1 \
                            and words[i - 1] not in self.left_bound and maybe(i - 1):
                        i -= 1; steps += 1
                    # gọt các từ khung ở hai đầu và các cụm chỉ hướng ở cuối ("... mạn trên", "... nằm phía đông", "... bên phải bản đồ")
                    changed = True
                    while changed and i < j:
                        changed = False
                        if words[j - 1] in EDGE_STRIP:
                            j -= 1; changed = True
                        elif j - i >= 2 and words[j - 2:j] == ["ban", "do"]:
                            j -= 2; changed = True
                        elif j - i >= 2 and words[j - 1] in DIRS and words[j - 2] in ORI_WORDS:
                            j -= 2; changed = True
                        elif j - i >= 2 and words[j - 1] in DIRS and words[j - 2] in ("o", "nam"):
                            j -= 1; changed = True
                        elif words[j - 1] == "nam" and j - i >= 2 and j < len(words) and words[j] in ORI_WORDS | {"gan", "xa", "sat", "o", "canh"}:
                            j -= 1; changed = True          # "nằm" (động từ) đứng trước cụm chỉ hướng
                    while i < j and words[i] in EDGE_STRIP:
                        i += 1
                    if i >= j or all(w in REF_WORDS for w in words[i:j]):
                        continue
                    if any(i < m[1] and j > m[0] for m in ms):
                        continue
                    if any(dl_distance(w, "robot", 1) <= 1 for w in words[i:j]):      # lời chào gõ sai ("orbot oi")
                        continue
                    span_text = " ".join(words[i:j])
                    kw = keyword_scores(span_text).max()
                    # cụm lạ chỉ được coi là địa điểm khi có dấu hiệu: mở đầu bằng "phòng / khu / nơi / nhà..." hoặc chứa từ khóa
                    # (nếu không, các cụm gấp/dễ vỡ lạ, món hàng lạ, động từ lạ sẽ bị nhận nhầm là địa điểm)
                    if kw == 0 and words[i] not in HEADS:
                        continue
                    if words[i] in ITEM_HEADS and words[i] not in PLACE_HEADS and kw < 3:
                        continue                              # "hộp thuốc", "khay cơm"...: món hàng
                    if j - i == 1 and kw == 0 and self.span_probs(span_text, " ".join(acc_w[i:j])).max() < 0.4:
                        continue
                    ms.append((i, j, None, " ".join(words[i:j])))
                    used.update(range(i, j))
                ms.sort()
            if not ms:
                plain.append(sent)
                plain_acc.append(" ".join(acc_w))
            punct = [k for k, t in enumerate(tt) if t in (",", ":")]
            for (i, j, t, a) in ms:
                # đoạn giữa hai dấu phẩy chứa lần nhắc này (để xét phủ định trong phạm vi hẹp)
                lo = max([k for k in punct if k < idx[i]], default=-1)
                hi = min([k for k in punct if k > idx[j - 1]], default=len(tt))
                out.append({"si": si, "toks": words, "acc": acc_w, "i": i, "j": j, "type": t, "text": a, "ms": ms,
                            "seg": " ".join(tt[lo + 1:hi]), "sent": " ".join(words)})
        return out, plain, plain_acc

    def _g(self, tok, drop, rng):
        f = self.freq.get(tok, 0)
        if f < 8 or (drop and f < 150 and rng.random() < 0.6):
            return "X"
        return tok

    def _ctx(self, me, drop=False, rng=None):
        toks, ms, i = me["toks"], me["ms"], me["i"]
        masked, k = [], 0
        while k < len(toks):
            hit = next((m for m in ms if m[0] == k), None)
            if hit:
                masked.append("TGT" if hit[0] == i else "PLC")
                k = hit[1]
            else:
                g = self._g(toks[k], drop, rng)
                if not (g == "X" and masked and masked[-1] == "X"):     # gộp các từ hiếm liền nhau thành một X
                    masked.append(g)
                k += 1
        pos = masked.index("TGT")
        left = " ".join(masked[max(0, pos - 5):pos])
        after = masked[pos + 1:pos + 6]
        right = " ".join(after)
        ref = []
        for t_i, t in enumerate(after):
            ref.append(t)
            if t == "PLC":
                if t_i + 1 < len(after) and after[t_i + 1] == "hon":
                    ref.append("hon")
                break
        order = f"n{min(len(ms), 3)}i{min([m[0] for m in ms].index(i), 2)}"
        return left, right, " ".join(masked) + " " + order, " ".join(ref[:5])

    @staticmethod
    def _anchor_of(me):
        nxt = [x for x in me["ms"] if x[0] >= me["j"]]
        if nxt and nxt[0][0] - me["j"] <= 3:
            return nxt[0]
        return None

    def _phrase_flags(self, s, acc=None):
        """-> (gấp?, dễ vỡ?) của một câu con không chứa địa điểm:
        bảng cụm đã học -> khớp mờ -> luật từ khóa -> vector nghĩa e5 -> phân loại ký tự."""
        lab = self.phrase.get(s)
        if lab is None:
            best, bl = 3, None
            for p, l in self.phrase.items():
                if abs(len(p) - len(s)) <= 2:
                    d = dl_distance(s, p, 2)
                    if d < best:
                        best, bl = d, l
            if bl is not None and best <= 2:
                lab = bl
        if lab is None:
            for part in re.split(r"[:,] ", s):
                if self.phrase.get(part):
                    lab = self.phrase[part]
        if lab is not None:
            return lab == 1, lab == 2
        # chữ "không" gõ sai ("khng", "khogn"...) -> "khong", để các luật phủ định vẫn bắt được
        s = " ".join("khong" if (len(t) >= 4 and t != "khong" and dl_distance(t, "khong", 1) <= 1) else t for t in s.split())
        u = bool(re.search(URGENT_POS, s)) and not re.search(URGENT_NEG, s)
        f = bool(re.search(FRAGILE_POS, s)) and not re.search(FRAGILE_NEG, s)
        silent = not u and not f and not re.search(URGENT_NEG + "|" + FRAGILE_NEG, s)
        if getattr(self, "ablate_rules", False):              # thí nghiệm: coi như không có luật từ khóa
            u, f, silent = False, False, True
        delivery = re.search(r"\b(can nhan|dang cho|giao|mang|gui|dua|chuyen|ghe|toi|den|hang:)", s)
        if getattr(self, "use_e5", False) and len(tokens(s)) <= 12:
            # Kết hợp luật từ khóa với mô hình nghĩa (5 lớp: trung tính / gấp / dễ vỡ / không gấp / không dễ vỡ).
            # Mô hình nghĩa chỉ được tin khi câu có dấu (ngưỡng 0.6) hoặc khi nó rất chắc (0.8).
            txt = acc if acc else s
            pe = self.e5_phrase.predict_proba(embedder().encode([txt]))[0]
            k = int(self.e5_phrase.classes_[int(np.argmax(pe))])
            sure = pe.max() > (0.6 if txt != s else 0.8)
            # từ phủ định chung ("chẳng", "đâu", "khỏi"...) mà không thuộc mẫu phủ định kép ("không được chậm trễ")
            gen_neg = bool(re.search(GEN_NEG, s)) and not re.search(DOUBLE_NEG, s) and not getattr(self, "no_gen_neg", False)
            if u:
                if (gen_neg and not (sure and k == 1)) or (sure and k == 3 and pe.max() > 0.75):
                    u = False                                 # có từ "gấp" nhưng cả câu mang nghĩa phủ định
            elif sure and k == 1 and not re.search(URGENT_NEG, s) and not gen_neg:
                u = True
            if f:
                if (gen_neg and not (sure and k == 2)) or (sure and k == 4 and pe.max() > 0.75):
                    f = False
            elif sure and k == 2 and not re.search(FRAGILE_NEG, s) and not gen_neg:
                f = True
            return u, f
        if silent and len(tokens(s)) <= 7 and not delivery:
            pr = self.phrase_clf.predict_proba(self.v_p.transform([s]))[0]
            k = int(np.argmax(pr))
            if pr[k] > 0.6:
                return k == 1, k == 2
        return u, f

    @staticmethod
    def _structure(ments, role, negs, P):
        """Luật cấu trúc cho khung câu lạ. Một yêu cầu chỉ có: 1 đích, tối đa 1 điểm ghé, các nơi gây nhiễu (luôn nằm trong
        câu phủ định / chuyện đã qua) và mốc gần/xa. Vậy trong các địa điểm KHÔNG bị phủ định và không phải mốc:
          - chỉ có một  -> nó là đích;
          - có hai      -> một là đích, một là điểm ghé; thứ tự quyết định bằng các từ "trước khi / trước / xong / rồi"
                           và động từ đi kèm (ghé, lấy... so với giao, mang, tới...).
        Sửa trực tiếp mảng `role` (0 đích, 1 điểm ghé, 2 gây nhiễu, 3 mốc)."""
        ent = []
        for k, me in enumerate(ments):
            if negs[k] or role[k] == 3:
                continue
            # chỉ xét những lần nhắc "chắc là địa điểm": không đứng sau từ chỉ món hàng ("thẻ thư viện"),
            # và nếu là tên lạ thì phải mở đầu bằng danh từ chỉ nơi chốn hoặc có từ khóa mạnh
            if me["i"] > 0 and me["toks"][me["i"] - 1] in ITEM_HEADS and me["toks"][me["i"] - 1] not in PLACE_HEADS:
                continue
            if me["type"] is None and me["toks"][me["i"]] not in PLACE_HEADS and keyword_scores(me["text"]).max() < 3:
                continue
            prev = [q for q in range(k) if ments[q]["si"] == me["si"] and ments[q]["j"] <= me["i"]]
            if prev:
                q = prev[-1]
                gap = me["toks"][ments[q]["j"]:me["i"]]
                if len(gap) <= 3 and any(g in ("gan", "xa", "sat", "canh", "ke", "cach") for g in gap) and not negs[q]:
                    role[k] = 3                                # "A gần B", "A cách xa B" -> B là mốc
                    continue
                if gap in (["o"], ["tai"], ["cua"], ["o", "tai"]) and ent and ent[-1][-1] == q:
                    ent[-1].append(k)                          # "thủ thư ở thư viện": người nhận + nơi làm việc là một
                    continue
            ent.append([k])
        if len(ent) == 1:
            if all(role[k] == 2 for k in ent[0]):
                for k in ent[0]:
                    role[k] = 0
            return
        if len(ent) < 2:
            return
        has_via = [any(role[k] == 1 for k in e) for e in ent]
        has_goal = [any(role[k] == 0 for k in e) for e in ent]
        if sum(has_via) == 1 and sum(g and not v for g, v in zip(has_goal, has_via)) >= 1:
            return                                             # bộ phân loại đã cho một kết quả hợp lệ
        top = sorted(ent, key=lambda e: -max(P[k, 0] + P[k, 1] for k in e))[:2]
        top.sort(key=lambda e: e[0])
        A, B = top
        VIA_V = r"\b(ghe|tat|re|lay|nhan|lanh|dung|vong|qua)\b"
        GOAL_V = r"\b(giao|mang|dua|gui|chuyen|toi|den|cho|tra)\b"

        def left_of(e):
            me = ments[e[0]]
            return " ".join(me["toks"][max(0, me["i"] - 5):me["i"]])

        def right_of(e):
            me = ments[e[-1]]
            return " ".join(me["toks"][me["j"]:me["j"] + 6])

        def before_all(e):
            me = ments[e[0]]
            return " ".join(me["toks"][:me["i"]])

        s_a_via = float(np.log(max(P[k, 1] for k in A) + 1e-3) + np.log(max(P[k, 0] for k in B) + 1e-3))
        s_b_via = float(np.log(max(P[k, 1] for k in B) + 1e-3) + np.log(max(P[k, 0] for k in A) + 1e-3))
        same_sent = ments[A[0]]["si"] == ments[B[0]]["si"]
        if re.search(r"\btruoc khi\b", before_all(A)):
            s_b_via += 4                                       # "trước khi <việc chính tới A>, ... B": B đi trước
        elif re.search(r"\b(truoc|xong|roi|sau do|tiep do|ke do|tiep theo)\b",
                       " ".join(ments[A[-1]]["toks"][ments[A[-1]]["j"]:ments[B[0]]["i"] if same_sent else None])):
            s_a_via += 3                                       # "A ... trước / xong / rồi / sau đó ... B": A đi trước
        if re.search(r"\b(truoc|truoc da|da)\b", right_of(B)) and not re.search(r"\btruoc khi\b", right_of(B)):
            s_b_via += 3                                       # "... B trước (đã)": B đi trước
        for e, sign in ((A, 1), (B, -1)):
            l = left_of(e)
            # động từ gần tên địa điểm nhất quyết định
            mv = [m.end() for m in re.finditer(VIA_V, l)]
            mg = [m.end() for m in re.finditer(GOAL_V, l)]
            if mv and (not mg or mv[-1] > mg[-1]):
                s_a_via += 1.5 * sign
            elif mg:
                s_a_via -= 1.5 * sign
        via_e, goal_e = (A, B) if s_a_via >= s_b_via else (B, A)
        for e in ent:
            for k in e:
                role[k] = 1 if e is via_e else (0 if e is goal_e else 2)

    # ================= dự đoán =================
    def parse(self, text):
        ments, plain, plain_acc = self._mentions(text)
        flags = [self._phrase_flags(s, a) for s, a in zip(plain, plain_acc)]
        # câu có nhắc địa điểm cũng có thể kèm ý gấp / dễ vỡ (hoặc là câu gấp/dễ vỡ bị nhận nhầm có địa điểm) -> xét bằng luật từ khóa
        for s in {me["sent"] for me in ments}:
            flags.append((bool(re.search(URGENT_POS, s)) and not re.search(URGENT_NEG, s),
                          bool(re.search(FRAGILE_POS, s)) and not re.search(FRAGILE_NEG, s)))
        gp = self.goal_clf.predict_proba(self.v_g.transform([self._main_text(text)]))[0]
        text_probs = np.full(10, 1e-3)
        text_probs[self.goal_clf.classes_] = gp
        res = {"goal": None, "goal_ref": None, "via": None, "via_ref": None,
               "urgent": any(f[0] for f in flags), "fragile": any(f[1] for f in flags), "conf": 0.0,
               "goal_dist": text_probs / text_probs.sum(), "via_dist": None, "goal_known": False, "via_known": False,
               "excluded": set(), "excluded_via": set(), "goal_anchor_dist": None, "via_anchor_dist": None}
        if not ments:
            res["goal"] = PLACE_TYPES[int(np.argmax(text_probs))]
            return res
        ctx = [self._ctx(me) for me in ments]
        X = hstack([self.v_l.transform(["L " + c[0] for c in ctx]), self.v_r.transform(["R " + c[1] for c in ctx]),
                    self.v_s.transform([c[2] for c in ctx])]).tocsr()
        P = self.role_clf.predict_proba(X)
        kindP = self.kind_clf.predict_proba(self.v_k.transform([c[3] for c in ctx]))
        role = P.argmax(1)
        ablate = getattr(self, "ablate_via", False)       # thí nghiệm: giả vờ bộ phân loại không biết khung câu "điểm ghé" nào
        if ablate:
            P = P.copy(); P[:, 1] = 0.25; P[:, 0] = 0.25
            role[role == 1] = 0
        # ---- luật tổng quát (kiến thức tiếng Việt) bổ sung cho khung câu lạ ----
        fz = lambda s: " ".join("khong" if (len(t) >= 4 and t != "khong" and dl_distance(t, "khong", 1) <= 1) else t for t in s.split())
        negs = [False] * len(ments)
        for k, me in enumerate(ments):
            w, i, j = me["toks"], me["i"], me["j"]
            seg, sent = fz(me["seg"]), fz(me["sent"])         # "khng", "khogn"... -> "khong"
            neg = re.search(NEG_SENT, seg) or (re.search(NEG_SENT, sent) and "," not in me["seg"] and len(me["ms"]) == 1)
            negs[k] = bool(neg)
            if neg and role[k] != 3:
                role[k] = 2                                   # câu phủ định / chuyện đã qua -> gây nhiễu
                continue
            if role[k] in (0, 2) and not neg and not ablate:
                left = " ".join(w[max(0, i - 4):i]); far_left = " ".join(w[max(0, i - 7):i])
                # phần ngay sau tên địa điểm, bỏ qua cụm tham chiếu không gian ("sát PLC", "ở phía dưới bản đồ", "mạn trên"...)
                after, q = [], j
                while q < len(w) and len(after) < 9:
                    hit = next((x for x in me["ms"] if x[0] == q), None)
                    if hit:
                        after.append("PLC"); q = hit[1]
                    else:
                        after.append(w[q]); q += 1
                right = re.sub(r"^((o|nam) )?((cach xa|gan|sat|xa|canh|ke) PLC( hon)?|" + _ORI + r" \w+( ban do)?|(tren|duoi|trai|phai|bac|dong|tay) ban do) ?",
                               "", " ".join(after))
                if re.search(r"\b(ghe|ghe qua|tat qua|tat vao|re qua|tat ngang|dung (o|tai))$", left) \
                        or (re.search(r"\b(lay|nhan|lanh)\b", far_left) and re.match(r"(xong|roi thi|truoc)\b", right)) \
                        or re.match(r"truoc\b(?! khi)", right):
                    role[k] = 1                               # "ghé X", "lấy ... ở X xong", "X trước" -> điểm ghé
        self._structure(ments, role, negs, P)
        dist = [None] * len(ments)
        for k, me in enumerate(ments):
            if me["type"] is not None:
                d = np.full(10, 1e-4); d[PLACE_TYPES.index(me["type"])] = 1.0
            else:
                d = self.span_probs(me["text"], " ".join(me["acc"][me["i"]:me["j"]]))
                if getattr(self, "use_e5", False) and hasattr(self, "e5_ctx") and getattr(self, "ctx_weight", 0.0) > 0:
                    pc = np.full(10, 1e-3)
                    pc[self.e5_ctx.classes_] = self.e5_ctx.predict_proba(embedder().encode([self._ctx_text(me)]))[0]
                    d = d * (pc + 0.02) ** getattr(self, "ctx_weight", 0.0)
            dist[k] = d / d.sum()
        # địa điểm gây nhiễu đã biết loại -> loại đó không thể là đích hay điểm ghé;
        # mốc (gần/xa) đã biết loại -> không thể là đích (nhưng CÓ THỂ trùng loại với điểm ghé)
        res["excluded"] = {me["type"] for k, me in enumerate(ments) if me["type"] is not None and role[k] in (2, 3)}
        res["excluded_via"] = {me["type"] for k, me in enumerate(ments) if me["type"] is not None and role[k] == 2}

        def pick(role_i, avoid=None):
            cands = [k for k in range(len(ments)) if role[k] == role_i and k != avoid]
            if not cands:
                return None
            # ưu tiên lần nhắc đã biết loại; trong cùng nhóm thì theo xác suất vai trò
            return max(cands, key=lambda k: (ments[k]["type"] is not None, P[k, role_i]))

        g = pick(0)
        if g is None:
            gbest = int(np.argmax(P[:, 0]))
            res["conf"] = float(P[gbest, 0])
            if ments[gbest]["type"] is None and P[gbest, 0] > 0.25:
                g = gbest
        if g is not None:
            res["conf"] = max(res["conf"], float(P[g, 0]))
            # gộp chứng cứ của mọi lần nhắc cùng vai trò đích (vd: người nhận + địa điểm)
            d = np.ones(10)
            for k in range(len(ments)):
                if role[k] == 0 or k == g:
                    d *= dist[k] ** (1.0 if ments[k]["type"] is not None else 0.7)
            d = d * (res["goal_dist"] ** 0.3)
            res["goal_dist"] = d / d.sum()
            res["goal_known"] = any(ments[k]["type"] is not None and (role[k] == 0 or k == g) for k in range(len(ments)))
        else:
            d = res["goal_dist"].copy()
            for t in res["excluded"]:
                d[PLACE_TYPES.index(t)] *= 1e-3
            res["goal_dist"] = d / d.sum()
        res["goal"] = PLACE_TYPES[int(np.argmax(res["goal_dist"]))]
        # ---- điểm ghé ----
        via_c = [k for k in range(len(ments)) if role[k] == 1 and not (ments[k]["type"] is not None and ments[k]["type"] == res["goal"])]
        v = max(via_c, key=lambda k: (ments[k]["type"] is not None, P[k, 1])) if via_c else None
        if v is not None:
            d = dist[v].copy()
            if ments[v]["type"] is None:
                d[PLACE_TYPES.index(res["goal"])] *= 0.05 if res["goal_known"] else 0.5
            res["via_dist"] = d / d.sum()
            res["via_known"] = ments[v]["type"] is not None
            res["via"] = PLACE_TYPES[int(np.argmax(res["via_dist"]))]
        # ---- tham chiếu không gian ----
        for key, role_i, main in (("goal", 0, g), ("via", 1, v)):
            if main is None:
                continue
            best = None
            for k in range(len(ments)):
                # mọi lần nhắc cùng vai trò (vd "trưởng phòng nội trú ở dãy phòng nội trú nằm phía tây": tham chiếu nằm ở lần nhắc thứ hai)
                if not (k == main or (role[k] == role_i and (ments[k]["type"] is None or ments[k]["type"] == res[key]))):
                    continue
                kk = int(np.argmax(kindP[k]))
                kind, pr = (KINDS[kk], float(kindP[k, kk])) if kk else (None, 0.0)
                if kind is not None and (pr < 0.7 or "X" in ctx[k][3].split()):
                    kind, pr = None, 0.0        # bộ phân loại không chắc, hoặc ngữ cảnh có từ lạ -> để luật từ khóa quyết định
                # luật từ khóa trên các từ GỐC ngay sau tên địa điểm (bộ phân loại không hiểu từ chỉ hướng lạ như "mạn dưới")
                me = ments[k]
                raw, q = [], me["j"]
                while q < len(me["toks"]) and len(raw) < 5:
                    hit = next((x for x in me["ms"] if x[0] == q), None)
                    if hit:
                        raw.append("PLC")
                        if hit[1] < len(me["toks"]) and me["toks"][hit[1]] == "hon":
                            raw.append("hon")
                        break
                    raw.append(me["toks"][q]); q += 1
                rk = rule_direction(" ".join(raw))
                if rk is not None and (kind is None or pr < 0.9):
                    kind, pr = rk, max(pr, 0.6)
                if kind is None:
                    continue
                anchor = None
                if kind in ("near", "far"):
                    a = self._anchor_of(ments[k])
                    if a is None:
                        continue
                    ak = next(q for q in range(len(ments)) if ments[q]["si"] == ments[k]["si"] and ments[q]["i"] == a[0])
                    anchor = dist[ak]
                if best is None or pr > best[0]:
                    best = (pr, kind, anchor)
            if best:
                a = best[2]
                res[key + "_ref"] = (best[1], PLACE_TYPES[int(np.argmax(a))] if a is not None else None)
                res[key + "_anchor_dist"] = a
        return res

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def resolve_with_map(m, landmarks):
    """Chốt loại đích / điểm ghé / mốc bằng cách chỉ xét các loại CÓ trên bản đồ (landmarks: {loại: [vị trí]}).
    Trả về mission dạng mà strategy.py dùng."""
    present = [t for t in PLACE_TYPES if landmarks.get(t)]
    out = {"goal": m["goal"], "goal_ref": m["goal_ref"], "via": m["via"], "via_ref": m["via_ref"],
           "urgent": m["urgent"], "fragile": m["fragile"]}
    if not present:
        return out
    two = {t for t in present if len(landmarks[t]) >= 2}
    one = {t for t in present if len(landmarks[t]) == 1}

    def choose(dist, allowed, need_two=False):
        cand = [t for t in present if t in allowed]
        if need_two and any(t in two for t in cand):
            cand = [t for t in cand if t in two]
        if not cand:
            cand = present
        return max(cand, key=lambda t: dist[PLACE_TYPES.index(t)])

    excluded = set(m.get("excluded") or ())
    via_known_type = m["via"] if m.get("via_known") else None
    # ---- đích ----
    if not (m.get("goal_known") and m["goal"] in present):
        allowed = {t for t in present if t not in excluded and t != via_known_type} or set(present)
        # có tham chiếu phương hướng -> loại đích phải có 2 bản trên bản đồ
        out["goal"] = choose(m["goal_dist"], allowed, need_two=m["goal_ref"] is not None)
    # ---- điểm ghé ----
    if m["via"] is not None:
        if not (m.get("via_known") and m["via"] in present):
            if m.get("via_dist") is None:
                out["via"], out["via_ref"] = None, None
            else:
                ex_via = set(m.get("excluded_via") or ())
                allowed = {t for t in present if t != out["goal"] and t not in ex_via} or {t for t in present if t != out["goal"]}
                out["via"] = choose(m["via_dist"], allowed, need_two=m["via_ref"] is not None) if allowed else None
        if out["via"] == out["goal"]:
            out["via"], out["via_ref"] = None, None
    # ---- mốc của gần/xa: phải là loại chỉ có MỘT bản trên bản đồ ----
    for key in ("goal", "via"):
        ref = out[key + "_ref"]
        if ref and ref[0] in ("near", "far"):
            a = m.get(key + "_anchor_dist")
            if ref[1] not in one and a is not None:
                cand = [t for t in one if t != out[key]]
                out[key + "_ref"] = (ref[0], max(cand, key=lambda t: a[PLACE_TYPES.index(t)])) if cand else None
            elif ref[1] not in present:
                out[key + "_ref"] = None
    return out
