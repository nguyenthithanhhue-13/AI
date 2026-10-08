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
import math
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import role_lm
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
              "xong", "thi", "da", "khong", "dang", "can", "cho", "toi", "den", "qua", "ghe", "voi", "la", "va", "tai",
              "ngang", "giao", "dich", "keo", "tre",    # "ghé ngang X", "nơi cần giao", "X là đích", "kẻo trễ" (kiến thức chung, h27*)
              "tan",                                    # "đưa đến TẬN X" (hư từ; h27 "Đưa đến tận {X}")
              "vach"}                                   # đợt 20: "X SÁT VÁCH Y" (vách không thuộc tên)
# danh từ THỜI GIAN mở đầu cụm -> không phải địa điểm ("kẻo trễ giờ học", "buổi học", "giờ ăn trưa")
TIME_HEADS = {"gio", "buoi", "tiet", "lich", "luc", "trua", "chieu", "dem", "hom", "tuan", "thoi"}
# --- khung câu mới tự soạn (scratch/h27_paraphrase.py, KHÔNG nhìn test): sửa tổng quát 2026-10-06 ---
NEW_FRAME_FIX = True
# từ mở đầu chung chung không thể tự nó là một tên địa điểm ("Nơi nhận: X" -> "nơi" không phải địa điểm)
LONE_BAD = {"noi", "khu", "phong", "nha", "cho", "toa", "day", "diem", "tang", "trung", "quay", "ban", "van", "loi", "hoi",
            "vien", "hanh", "thu", "can", "ky", "giang"}
# khung NHÃN: "Nơi nhận: X", "Địa chỉ giao: X", "Điểm đến (cuối cùng): X" -> đích; "Điểm ghé: X", "Nơi lấy hàng: X" -> điểm ghé
GOAL_LABEL = r"\b(?:(?:noi|diem|dia chi|dia diem|cho|ben) (?:nhan|giao|den|toi|can giao|can den|giao hang|nhan hang)(?: hang)?" \
             r"(?: cuoi(?: cung)?)?|dich(?: den)?(?: cuoi(?: cung)?)?|diem (?:den|cuoi)(?: cuoi(?: cung)?)?|nguoi nhan (?:dang )?(?:o|tai))" \
             r"(?: la| o| tai)?$"
VIA_LABEL = r"\b(?:(?:noi|diem|cho) (?:ghe|lay|dung|dung chan|trung chuyen|ghe qua|lay hang|lay do)(?: hang| do)?" \
            r"(?: truoc| giua duong)?|ghe truoc|ghe giua duong)(?: la| o| tai)?$"
# khung THỨ TỰ (đợt 7, scratch/h27c_order.py --v7): "Chặng đầu A, chặng cuối B", "Ghé A, sau đó B", "lấy đồ cho người ở B"
GOAL_LABEL2 = r"\b(chang (cuoi|hai|2|thu hai)|sau (do|cung)|cho (nguoi|ban|anh|chi|co|thay|em|chu|bac)( dang)? (o|tai)|cuoi cung( la)?)$"
VIA_LABEL2 = r"\b(chang (dau|mot|1|thu nhat)|diem dau|dau tien( la)?)$"
ANCHOR_REL = r"\b(gan|xa|sat|canh|ke|cach|voi|giap|doi dien|ben|lien ke)\b"   # mốc gần / xa luôn đi sau một từ quan hệ
# động từ ghé ngay trước tên địa điểm -> điểm ghé (bản mới thêm "ghé ngang", "đi ngang qua", "rẽ vào", "dừng chân / dừng lại ở")
VIA_LEFT = r"\b(ghe|ghe qua|tat qua|tat vao|re qua|tat ngang|dung (o|tai))$"
VIA_LEFT2 = r"\b(ghe|ghe qua|ghe vao|ghe ngang|ghe ngang qua|tat qua|tat vao|tat ngang|re qua|re vao|di ngang qua|ngang qua|" \
            r"dung (o|tai)|dung (chan|lai) (o|tai)|dung chan|" \
            r"(truoc tien|dau tien|truoc het)( thi| hay| phai| can)? (qua|toi|den|vao|ra|sang|di qua|di toi|di den))$"

FLAG_MORE3 = True   # đợt 17: thêm cách nói dễ vỡ / gấp (kiến thức chung; bộ h52 B; đo lại trên bộ C viết trước khi sửa)
FLAG_MORE2 = True   # đợt 16: thêm cách nói dễ vỡ / không gấp (kiến thức chung; bộ câu mới scratch/h52_flags_fresh.py)
E5_ITEM_GUARD2 = True  # đợt 16: chỉ cấm e5 với câu MÔ TẢ món hàng ("hàng: X", "hàng cần giao là X", "đơn #..."), không cấm câu nói
                       # về tính chất của hàng ("Hàng nhạy với va đập", "Món này mỏng lắm": e5 0,98 / 0,86 mà trước đây bị bỏ)
ITEM_DESC = r"(hang|mon hang|kien hang|mat hang|don hang|mon|kien)( (can|se) (giao|gui|chuyen|dua|mang))?( (gom|la|co)\b|\s*:|$)|don\b"
E5_CROSS_GUARD = True  # đợt 16: trong MỘT vế mà luật đã thấy cờ này (vd dễ vỡ: "nâng niu giúp"), e5 không được tự thêm cờ kia
# bổ sung 2026-10-04 từ phép đo trên ví dụ viết tay (scratch/h8, h9): phủ định của gấp, cách nói gấp / dễ vỡ khác
URGENT_NEG_MORE = r"|linh hoat|ca ngay|(chieu|toi|mai|sang mai|tuan sau|hom sau|luc sau) (nay )?moi (can|dung|bat dau|lam|hop|thi|hoc)|han (nop )?con (xa|rong|dai)|(khong|chua|chang|khoi|dung) (can |phai |viec gi phai )?(\w+ )?(gap|voi|khan|hoa toc|nhanh)\b|uu tien (binh thuong|thuong|trung binh)|hen gap|gap (lai|sau|nhau|mat)\b|dung (voi|gap|hap tap|cuong)|het (gap|voi)|khong con (gap|voi)|uu tien thap|chua can|" \
                  r"khong can (toi |den |giao |di |mang |lam )?(ngay|lien|hoa toc|uu tien|khan)|" \
                  r"con (nhieu|du|dai|thua) (thoi gian|thi gio|gio)|thoi gian con (dai|nhieu|du)|con du|cham ma chac|" \
                  r"khong (qua|den muc|den noi|he) (gap|voi)|no rush|khong rush|(gap|voi) (gi|lam gi|chi)\b|khong ai (cho|doi|hoi|giuc)|" \
                  r"(tre|muon|cham) (mot )?(chut|ti|xiu) (cung )?(khong sao|duoc|chang sao)|(nguoi|ben) nhan (khong|chua) (gap|voi|can)" + (
                  r"|(?<!dung )(?<!khong duoc )\b(di|chay) (that |hoi |that su )?cham\b(?! tre)" if FLAG_MORE2 else "")   # "đi chậm chỗ gồ ghề"
URGENT_POS_MORE = r"|\ble (nha|nhe|len|le|gium|giup|di)\b|sap (chay|roi di|di roi|xuat phat|khoi hanh|ket thuc)|khong duoc phep (tre|cham|muon)|(thoi )?han (rat |qua )?(sat|gap|can ke)|nhanh chong|(?<!\bda )\b(chay|di|giao|mang|dua|gui|chuyen|toi|den) (that |cho )?nhanh\b|\bhoi (lam|qua|suot)|(dang|bi) hoi|dung de (\w+ ){0,2}(doi|cho)\b|khong (duoc |the |nen )?(tri hoan|chan chu|la ca|cham tre)|(tri hoan|chan chu) (la )?khong duoc|cap cuu|sap (di|roi di|dong cua|het)\b|(muoi|nam|vai|it|may|hai|ba) phut nua|luon va ngay|\b(can|giao|lam|di|mang) lien\b|\blien (bay gio|nhe|nha|giup|tay)|khong co thoi gian|" \
                  r"(tre|cham|muon) la (hong|khong kip|het|lo)|khong (the|duoc) (doi|muon|den tre|den muon)|" \
                  r"(doi|cho) lau khong (duoc|noi)|khong (cho|doi) (lau )?(duoc|noi)|\burgent\b|" \
                  r"(lam|giao|xu ly) truoc cac don|len truoc|\bmau (len|giup|gium|nhe)\b|\bdi mau\b|nuoc den chan|sat nut|" \
                  r"(tre|muon) (lam )?roi|tang toc|" \
                  r"(dung|khong duoc|cam) (di|chay|lam|giao|mang|dua|o) (cham|tre|lau|muon)\b|" \
                  r"cang (nhanh|som) cang (hay|tien)\b"   # "đừng đi chậm nhé" (h27); "càng sớm càng hay" (h32)
FRAGILE_NEG_MORE = r"|(?<!khong )chiu (duoc )?(va|rung|xoc|luc)|thoai mai|manh tay|khoi can|khong can (phai |qua )?(nhe|can than|nang niu|giu gin)|" \
                   r"khong (ngai|so|ky) (va|roi|rung|xoc)|(xoc|lac|roi|va dap|va cham) (cung )?(khong sao|duoc|chang sao)|khong co (do )?(thuy tinh|gi de)"
FRAGILE_POS_MORE = r"|ky (roi|rung|xoc|lac)|tranh (lam )?(roi|do|xoc|lac)|dung (xoc|rung|nghieng|de roi)|(vo|be) la (hong|mat|het|bo|toi|xong)|keo (vo|be|hong|roi|do)\b|\bde (hu|gay|do|mop|tray|dap)\b|" \
                   r"\b(chen|bat|dia|binh|ly|tach|am|lo) (su|gom)\b|thang bang|thang dung|nghieng|nhe nhang(?! (go|mo|dong|bao|noi|goi|nhac|hoi|nhan tin))|\b(di|chay) (that )?em\b|" \
                   r"\bem (thoi|giup|nhe)\b|\bso (roi|vo|va|be)\b|nhay cam|\bfragile\b|coi chung (vo|be|roi)|khong duoc (nghieng|do|xoc|rung)" + (
                   # đợt 16 (h52, kiến thức chung): "coi chừng làm bể", "rơi là hỏng", "va là vỡ", "có thể bị vỡ", "dằn xóc"
                   r"|coi chung (lam )?(vo|be|roi|do|hong)|(roi|va|dap|xoc|nga) (la|thi|se) (hong|vo|be|hu|nut|mop|toi)\b|"
                   r"(?<!khong )(?<!khong the )(?<!chang )\bbi (vo|be|nut|gay|mop)\b|\bdan xoc\b|\bxoc nay\b|\bsanh su\b|"
                   r"(?<!khong )\bco the (bi )?(vo|be|nut|gay|mop)\b" if FLAG_MORE2 else "") + (
                   # đợt 17 (kiến thức chung; bộ h52 B): "nhạy với va đập", "coi chừng va vào", "mà vỡ là...", "chỗ xóc",
                   # "giòn lắm", "dễ sứt mẻ", "không chịu được va đập", "kẻo nứt", "trứng gà"
                   r"|\bnhay (voi |cam voi )?(va|rung|xoc|soc|luc)|coi chung (va|dung|dap|cham)\b|\b(ma|neu|lo) (vo|be|nut|gay|mop|hong) (la|thi)\b|"
                   r"(cho|duong|doan) (xoc|go ghe)\b|\bgion (lam|qua|tan|de)\b|\bde (sut|me|bep|vun|sut me)\b|keo (nut|gay|mop|me|sut|do vo)\b|"
                   r"(?<!khong phai )\bkhong chiu (duoc |noi )?(va|rung|xoc|luc|soc)|\btrung (ga|vit|cut)\b" if FLAG_MORE3 else "")
URGENT_NEG = r"khong (can )?(gap|voi|khan|nhanh)|chua (can )?(gap|voi)|cu tu tu|thong tha|\btu tu\b|luc nao (cung|giao)|bao gio cung|" \
             r"cung (duoc|kip)|khong (co )?gi (gap|voi)|chang (gap|voi)|khoi (gap|voi)|ranh (thi|luc)|khi nao (ranh|tien)|khong han chot|" \
             r"de sau|khong phai (gap|voi)|binh thuong thoi|khong uu tien|tien (thi|luc nao)" + URGENT_NEG_MORE
# bổ sung (kiến thức tiếng Việt chung): hạn giờ, thúc giục, tốc độ
URGENT_MORE = r"|trong (vong )?(\d+|vai|it|may|nam|muoi|hai|ba) (phut|giay)|\d+ phut nua|con (\d+|vai|it|may) phut|" \
              r"truoc \d+ ?(h|gio)\b|\bdeadline\b|\basap\b|\bexpress\b|toc hanh|than toc|sieu toc|het toc luc|" \
              r"nuoc soi lua bong|\btuc khac\b|\bcap ky\b|\bkhan thiet\b|\bbuc thiet\b|\bcan kip\b|" \
              r"(dang|bi) (hoi|giuc|thuc)\b|dung la ca|khong la ca|(tre|muon|kip) gio|" \
              r"sap (bat dau|vao (gio|lop|hoc|hop)|den gio|muon|het han)|khong (cho|doi) duoc|tung (phut|giay)|" \
              r"\bmau (len|mau)\b|\ble (len|le|nhe|gium|giup)\b|nhanh (chan|tay|nhanh|gium|giup)|\bdang (doi|cho) (san|gap)\b" + (
              # đợt 17: "trong nửa tiếng phải tới", "không để chờ lâu được", "trễ hạn là bị phạt"
              r"|trong (vong )?(nua|mot|hai|ba|\d+) (tieng|gio)\b|khong (the |duoc )?de (\w+ ){0,2}(cho|doi) lau|(tre|muon) (han|gio) (la|thi|se) (bi|mat)\b"
              if FLAG_MORE3 else "")
# (bỏ dấu thì "ngay" = ngay/ngày nên chỉ nhận trong cụm rõ nghĩa "đi ngay", "ngay lập tức"...)
URGENT_POS = r"\bgap\b|\bkhan\b|hoa toc|\blap tuc\b|\btuc thi\b|cang (nhanh|som) cang tot|\buu tien\b(?! (ghe|tat|re|qua|di qua|lay|nhan|toi|den)\b)|cham tre|" \
             r"trong \d+ phut|\b(di|can|giao|lam|chay|mang|den|toi|gui|chuyen) ngay\b|\bngay (lap tuc|bay gio|va luon|nhe|di)\b|\bngay$|" \
             r"nhanh (len|nhe|chan|nhat)|\bcap toc\b|khong (duoc|the) (tre|cham|lau|cho)|\bkip\b|som nhat|sat gio|" \
             r"\bcap bach\b|\btuc toc\b|keo (tre|muon|lo)|dung (de )?(tre|cham|lau)\b|sap (tre|het gio)|" \
             r"het gio(?! (lam viec|hanh chinh|lam|mo cua|phuc vu|tiep))|\bcho lau\b|" \
             r"dang (rat )?(sot ruot|nong ruot|voi)|\bvoi lam\b|\brat voi\b|\bthat nhanh\b|\bnhanh nhat\b|\bkhan truong\b" + URGENT_MORE + URGENT_POS_MORE
FRAGILE_NEG = r"khong (de )?(vo|be|hong|nut)\b|khong lo (vo|be|hong)|chac chan|\bben\b(?! (trong|ngoai|canh|trai|phai|kia|nay|duoi|tren|do|hong|nhan|giao|gui))|roi cung|va dap cung|khong so (vo|va|roi)|" \
              r"chang (so|lo)|kho (vo|hong)\b|\bcung cap\b|khong mong manh|khong can nhe" + FRAGILE_NEG_MORE + (
              r"|khong (the )?bi (vo|be|nut|gay|mop|hong)|khong the (vo|be|nut|gay|mop|hong)" if FLAG_MORE2 else "")
FRAGILE_POS = r"\bde (vo|be|hong|nut|me)\b|thuy tinh|\bgom\b(?! (mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi|\d|co|cac|nhung|ca)\b)|\b(do|bang|gom) su\b|pha le|mong manh|\bky va\b|tranh va\b|nhe tay|can than(?! (keo |ke |khong )?(nham|lac|sai|tre|muon|cham|quen|lo)\b)|" \
              r"tranh rung|\bhang de\b|nang niu|khong duoc (roi|lac|va)\b|dung (lam roi|lac(?! (sang|vao|toi|den|qua)\b)|quang)|cam (nem|quang)" + FRAGILE_POS_MORE
# từ mở đầu một MÓN HÀNG (hộp, khay, thẻ, túi...): "hộp thuốc", "thẻ thư viện" là hàng chứ không phải địa điểm
ITEM_HEADS = {"hop", "khay", "the", "tui", "thung", "tap", "bo", "goi", "chong", "ket", "mau", "chia", "so", "bien", "don",
              "binh", "lo", "cuon", "quyen", "to", "giay", "bia", "micro", "mu", "chan", "buu", "kien", "may", "bom", "linh",
              "nguyen", "hoa", "sach", "chai", "ly", "coc", "bang", "day", "cap", "vali", "phieu", "thu"}
# danh từ chỉ nơi chốn thường mở đầu tên địa điểm
PLACE_HEADS = {"phong", "khu", "noi", "nha", "san", "bai", "toa", "cho", "tram", "cong", "loi", "ham", "xuong", "quay", "kho",
               "sanh", "bep", "chot", "goc", "vuon", "trung", "tang", "vien", "quan", "lop", "hoi", "day"}
GEN_NEG = r"\b(khong|chang|cha|khoi|chua|dau can|dau co|dau phai|gi dau|lam gi|sao duoc)\b|(?<!bat )(?<!ban )(?<!hang )(?<!luc )(?<!tran )\bdau$"
DOUBLE_NEG = r"khong (duoc|the|nen) (cham|tre|lau|cho|de|muon|roi|va|lac|nem|quang|xoc|lam roi)|khong the (cho|doi)|khong kip|" \
             r"khong chiu (duoc )?(va|rung|xoc)|khong (duoc|nen) (de )?(roi|va|nghieng|do|xoc|rung|lac)|khong (cho|doi) (lau )?(duoc|noi)|lau khong (duoc|noi)|khong co thoi gian|khong duoc phep (tre|cham|muon)|khong (duoc |the |nen )?(tri hoan|chan chu|la ca)|dung de (\w+ ){0,2}(doi|cho)\b" + (r"|khong (duoc|nen|the) (di|chay|lam|giao|mang|dua|chuyen) (cham|tre|lau|muon)" if FLAG_MORE2 else "") + (r"|khong (the |duoc )?de (\w+ ){0,2}(cho|doi) lau" if FLAG_MORE3 else "")
# đợt 20: mọi nhánh của các mẫu gấp / dễ vỡ phải khớp TRỌN TỪ: "roi cung" (rơi cũng không sao) không được khớp trong "tRỜI CŨNG
# mưa", "con du" (còn dư) trong "con DƯỜNG", "han gap" trong "nHẬN GẤP". Quét train + validation: 0 lần khớp giữa chữ -> không đổi
# gì trên dữ liệu; chỉ chặn báo nhầm ở câu mới. (NO_FLAG_BOUND=1 để tắt, đo số dòng test đổi.)
import os as _os2
FLAG_BOUND = _os2.environ.get("NO_FLAG_BOUND") != "1"


def _bound(rx):
    alts, depth, cur, esc = [], 0, "", False
    for ch in rx:
        if esc:
            cur += ch; esc = False; continue
        if ch == "\\":
            cur += ch; esc = True; continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            alts.append(cur); cur = ""
        else:
            cur += ch
    alts.append(cur)
    out = []
    for a in alts:
        if not a:
            continue
        pre = "" if a.startswith((r"\b", "(?<", "^")) else r"\b"
        suf = "" if a.endswith((r"\b", "$")) else r"\b"
        out.append(pre + a + suf)
    return "|".join(out)


if FLAG_BOUND:
    URGENT_NEG, URGENT_POS, FRAGILE_NEG, FRAGILE_POS, DOUBLE_NEG = (_bound(x) for x in (URGENT_NEG, URGENT_POS, FRAGILE_NEG,
                                                                                        FRAGILE_POS, DOUBLE_NEG))
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


# từ khóa bổ sung đợt 10 (TYPE_FIX, kiến thức chung; scratch/h31_style_aliases.py): "nơi đóng học phí" không phải giảng đường,
# "phòng quản lý sinh viên" không phải ký túc xá, "phòng đo đạc" là phòng thí nghiệm, "bạn cùng phòng" ở ký túc xá
# (bỏ "thi" và "con dau": không dấu trùng hư từ "thì" / "còn đâu" -> cụm "dưới thì mới" thành tên địa điểm, h27)
KEYWORDS_MORE = {"office": [(r"hoc phi", 3), (r"le phi", 3), (r"quan ly sinh vien", 3), (r"dong dau", 3),
                            (r"cong tac (sinh vien|hoc sinh|hssv|nguoi hoc)", 3)],   # đợt 17: "phòng công tác học sinh"
                 "lab": [(r"do dac", 2.5), (r"do luong", 2.5), (r"kinh hien vi", 3), (r"phan tich mau", 3), (r"vi sinh", 3),
                         (r"hoa sinh", 3)],
                 "dorm": [(r"cung phong", 2.5)],
                 "clinic": [(r"huyet ap", 3), (r"bang bo", 3), (r"vet thuong", 3)],
                 "lecture": [(r"(giua|cuoi) ky", 2.5)]}
# đợt 17 (kiến thức chung; các nhóm tên h31 hay nhầm): "phòng hỗ trợ sinh viên" / "nơi xin xác nhận" là văn phòng (không phải
# ký túc xá vì chữ "sinh viên"); "nơi nghỉ khi ốm" là y tế (không phải chỗ ngủ); "lưu trữ luận văn" là thư viện ("lưu trú" ~
# "lưu trữ" khi bỏ dấu); "thầy giáo bộ môn" ở giảng đường; "đá cầu" là thể thao
TYPE_FIX3 = True
if TYPE_FIX3:
    for _t, _kw in {"office": [(r"ho tro (sinh vien|nguoi hoc|hoc sinh)", 3), (r"xac nhan", 2.5), (r"giay (gioi thieu|xac nhan)", 3),
                               (r"\btiep (sinh vien|nhan ho so|dan|cong dan)\b", 2.5)],
                    "clinic": [(r"\b(khi|nguoi|bi|dang|luc) (om|met|benh|sot)\b", 3), (r"nhiet do", 2.5)],
                    "library": [(r"luu tru (tai lieu|luan van|sach|ho so|ban|giao trinh|tu lieu|luan an|bao)", 3), (r"luan van", 2.5)],
                    "lecture": [(r"\b(thay|co) giao\b", 3), (r"giao vien|giang vien", 3), (r"\b(dang day|day hoc|giang day)\b", 2.5),
                                (r"chu nhiem", 2.5), (r"bo mon", 2)],
                    "sports": [(r"\bda (cau|bong|banh)\b", 3)]}.items():
        KEYWORDS_MORE[_t] = KEYWORDS_MORE.get(_t, []) + _kw
# đợt 20 (07/10 tối, kiến thức chung về tên phòng ban / hoạt động trong trường, không lấy từ test): từ khóa nhiều chữ, ít nhập
# nhằng khi bỏ dấu. Sửa lỗi từ khóa "an" (= "ăn") khớp cả "AN ninh / AN toàn / AN tâm" -> "nhân viên an ninh" thành căn tin.
TYPE_FIX4 = True
REF_NEG_FIX = True       # đợt 20: "KHÔNG ở gần Y" = xa Y (luật hiểu phủ định thắng bộ phân loại kind)
# phủ định ĐỨNG SÁT từ quan hệ: "không xa Y", "không ở gần Y", "chẳng gần Y lắm"; "X cách Y không xa" (phủ định sau mốc).
# Train + validation: 0 câu có phủ định gần / xa -> bộ phân loại n-gram chưa từng thấy, chỉ nhìn chữ "xa" / "gần".
NEG_REL = r"\b(khong|chang|cha)( (o|nam|phai|he|qua|that|su|lam|duoc|may|con|la|cai|ban|cho|noi)){0,3} (xa|gan|sat|canh|ke|cach)\b"
NEG_POST = r"(la |co |chi |cung |thi )?(khong|chang|cha) (rat |kha |hoi |qua |that |tuong doi |lam )?(xa|gan)\b"
# hướng BỊ PHỦ ĐỊNH ("X không nằm ở phía bắc", "không phải cái phía tây"): hai bản luôn lệch >= 2 hàng / cột (tham chiếu rõ
# ràng) -> "không phải bản phía bắc" = bản phía nam. Train + validation: 0 câu.
REF_NEG_DIR = True
NEG_DIR_LEFT = r"\b(khong|chang|cha)( (o|nam|phai|thuoc|he|la|cai|ban|cho|noi)){0,4}\s*$"
NEG_DIR_PHRASE = r"\b(khong|chang|cha)( (o|nam|phai|thuoc|he|la|cai|ban|cho|noi)){0,4} " \
                 r"(phia|ben|man|huong|goc|mien|mep|ria|nua) (bac|nam|dong|tay|tren|duoi|trai|phai)\b"
_FLIP_DIR = {"north": "south", "south": "north", "east": "west", "west": "east"}
NEG_SPAN_STOP = True     # đợt 20: tên lạ dừng ở "không / chẳng / chưa" ("nơi cấp thuốc | không ở gần Y")
VACH_FIX = True          # đợt 20: "X sát VÁCH Y": "vách" không thuộc tên mốc
KEYWORDS_MORE4 = {}
# chữ "an" (không dấu) theo nghĩa "ĂN": không phải "AN ninh / AN toàn / AN tâm", không phải "dự ÁN / luận ÁN / phương ÁN / đề ÁN"
_AN_EAT = r"(?<!\bdu )(?<!\bluan )(?<!\bphuong )(?<!\bde )(?<!\bho )\ban\b(?! (ninh|toan|tam|khang|nhien|vi|bai|phan|cu|dinh|hoc|ui|lanh|duong)\b)"
if True:
    for _t, _kw in {
            "gate": [(r"an ninh", 3), (r"thuong truc", 2.5), (r"kiem soat (ra vao|cong|xe|the|nguoi)", 3), (r"soat (ve|the)", 3),
                     (r"(tiep|dang ky|gap) khach", 2.5), (r"khach (den|tham|vang lai)", 3), (r"phu huynh", 2), (r"cong chao", 3)],
            "office": [(r"phap (che|ly)", 3), (r"thi dua|khen thuong", 3), (r"thu (quy|ngan)", 3), (r"quan tri (thiet bi|co so|mang)?", 2.5),
                       (r"thanh tra", 3), (r"hop tac", 2.5), (r"doanh nghiep", 2.5), (r"dam bao chat luong", 3), (r"cong van", 3),
                       (r"the sinh vien", 3), (r"bang (tot nghiep|diem)", 3), (r"chung chi", 2.5), (r"\bdoan (truong|thanh nien)\b", 3),
                       (r"hoi sinh vien", 3), (r"truyen thong", 2.5), (r"ke hoach", 2.5), (r"ban giam (hieu|doc)", 3),
                       (r"pho hieu truong|hieu pho", 3), (r"\btro ly\b", 2), (r"ctsv|cthssv", 3), (r"quan he (doanh nghiep|quoc te|cong chung)", 3)],
            "lab": [(r"mo phong", 2.5), (r"\bphan tich\b", 2), (r"kiem nghiem", 3), (r"thu nghiem", 2.5), (r"co khi", 3), (r"(phong|xuong|lab|thuc hanh|ky thuat|linh kien|mach|vien thong) dien tu", 2.5),
                    (r"vat lieu", 2.5), (r"quang hoc", 3), (r"ong nghiem", 3), (r"che tao", 3), (r"in 3d", 3), (r"\b(hoa|sinh|vat ly) hoc\b", 2)],
            "dorm": [(r"\b(nha|phong|day|khu|cho|noi) ngu\b", 3), (r"o ghep", 2.5), (r"luu xa", 3), (r"nha khach", 2), (r"giuong tang", 3),
                     (r"tap the sinh vien", 3), (r"sinh hoat noi tru", 3)],
            "canteen": [(r"quan (bun|pho|com|che|nuoc|ca phe|tra sua|banh|an)", 3), (r"quay (che|banh|nuoc|com|do an|tra sua)", 3),
                        (r"banh (mi|bao|ngot)", 2.5), (r"com (hop|binh dan|van phong|trua|tam)", 3), (r"tra sua", 3),
                        (r"nuoc (uong|ep|giai khat|ngot|mia)", 2.5), (r"an (vat|nhe|nhanh)", 3), (r"phu bep|dau bep|nau (an|bep|com)", 3),
                        (r"thuc an|mon an", 3)],
            "parking": [(r"cat xe", 3), (r"trong (giu )?xe", 3), (r"\bxe (buyt|khach|dien|hoi)\b", 2.5), (r"ve xe", 2.5),
                        (r"bai (do|dau|gui|giu)", 3)],
            "sports": [(r"vo thuat|day vo|tap vo|phong vo|lop vo", 3), (r"yoga|boxing|aerobic|pilates|zumba|karate|taekwondo|judo|vovinam", 3),
                       (r"cau long|tennis|quan vot|bong ban|bi a|co vua", 3), (r"chay bo", 3), (r"hlv\b", 3)],
            "clinic": [(r"cach ly", 2.5), (r"chua (tri|benh)", 3), (r"tiem (chung|phong|vac)", 3), (r"rang ham mat|nha khoa", 3),
                       (r"(cap|phat|xin|lay) thuoc", 3), (r"om dau|nguoi om|nghi om", 3)],
            "lecture": [(r"giang (bai|day)", 3), (r"dai cuong", 2.5), (r"ly thuyet", 2.5), (r"may chieu", 2), (r"tiet (hoc|sang|chieu)", 3),
                        (r"ngoai ngu|tieng anh", 2), (r"cao hoc", 2.5), (r"lop (hoc|chat luong|tieng)", 3)],
            "library": [(r"tri thuc", 2.5), (r"muon truyen|doc truyen|truyen tranh", 3), (r"tu dien", 2.5), (r"ngoai van", 2.5),
                        (r"sach (cu|quy|giao khoa|tham khao|ngoai van)", 3), (r"luan an|khoa luan", 2.5), (r"gia han", 3),
                        (r"ke sach|gia sach|tu sach", 3), (r"phong doc", 3), (r"im lang|yen tinh", 1.5)]}.items():
        KEYWORDS_MORE4[_t] = _kw


def keyword_scores(text):
    s = np.array([sum(w for pat, w in KEYWORDS[t] if re.search(pat, text)) for t in PLACE_TYPES])
    if TYPE_FIX:
        s = s + np.array([sum(w for pat, w in KEYWORDS_MORE.get(t, []) if re.search(pat, text)) for t in PLACE_TYPES])
    if TYPE_FIX4:
        s = s + np.array([sum(w for pat, w in KEYWORDS_MORE4.get(t, []) if re.search(pat, text)) for t in PLACE_TYPES])
        if re.search(r"\ban\b", text) and not re.search(_AN_EAT, text):
            s[PLACE_TYPES.index("canteen")] -= dict(KEYWORDS["canteen"]).get(r"\ban\b", 0)     # "an ninh", "luận án" không phải "ăn"
    return s


# câu "gây nhiễu": phủ định / chuyện đã qua -> địa điểm trong câu không phải đích cũng không phải điểm ghé
# (chú ý: bỏ dấu thì "đừng" và "dụng (cụ)" trùng nhau -> chỉ nhận "dung" khi theo sau là động từ)
NEG_SENT = r"\bkhong (can|phai|nen|ghe|den|toi|qua(?! (xa|gan|lau|nhieu|kho|muon|tre|som|cham|nhanh|gap|nang|nhe|lon|nho|dong)\b))\b|(?<!phai )(?<!nho )(?<!can )(?<!hay )(?<!se )\bdung (nham|ghe|den|toi|qua|di|mang|giao|dua|re|vao|tat|dung)\b|\bbo qua\b|\bkhoi (can|phai|ghe)\b|\bhom (qua|truoc|kia)\b|" \
           r"\bda (giao|roi|chuyen|nhan|xong|di)\b|\bnham\b|\bngoai tru\b|\bchang can\b|\bkhong lien quan\b|\b(lan|tuan) truoc\b"
# "Không giao ở X", "X hôm nay không nhận hàng", "không được đưa tới X", "khỏi qua X": mọi động từ chở / ghé sau "không"
NEG_SENT2 = NEG_SENT + r"|\bkhong (duoc |the )?(giao|mang|dua|chuyen|gui|nhan|vao|lay|tat|re|van chuyen)\b|\bkhoi (giao|qua|toi|den|di|ghe)\b" \
                       r"|\bdong cua\b|\bkhong (con )?mo cua\b" \
                       r"|\bkhong con (o|lam|nhan|can|tai)\b|\bkhong co ai\b|\bthi khoi\b" \
                       r"|\bkhong (duoc |the )?di (qua|vao|toi|den|ngang|sang)\b|\bda (co hang|co do|nhan du|nhan hang|nhan do)\b" \
                       r"|\bkhong danh cho\b|\bdung lac (sang|vao|toi|den|qua)\b|\bthay vi\b"
# "X đóng cửa rồi", "không còn ở X", "X thì khỏi", "Không đi qua X", "X đã có hàng rồi", "Hàng không dành cho X", "Đừng lạc sang X"
AVOID_LEFT = r"\b(tranh|ne|ne tranh)( xa| khoi| ra khoi)?$"            # "Nhớ né X ra", "Tránh xa X"
# đợt 14 (VIA_FIX, scratch/h37_distractor.py): "X hôm nay nghỉ", "Hủy điểm X", "Tránh đường qua X"
NEG_SENT3 = r"|\b(hom nay|bay gio|dang|da|tam|cuoi tuan) nghi\b|\bhuy (diem|don|chang|ghe|lich|bo)\b|\b(da|bi) huy\b"
AVOID_LEFT2 = r"\b(tranh|ne|ne tranh)( xa| khoi| ra khoi| duong qua| di qua| di vao| qua| vao)?$"
# chữ then chốt của các luật trên (để sửa lỗi gõ đảo / rơi chữ: "khong hge X", "Đừng mnag tới X", "Canteen thì bỏ qau")
KEY_CUES = ["khong", "can", "phai", "nen", "ghe", "den", "toi", "qua", "dung", "nham", "giao", "mang", "dua", "vao", "tat",
            "khoi", "hom", "truoc", "kia", "roi", "chuyen", "xong", "ngoai", "chang", "lien", "quan", "lan", "tuan", "gui",
            "nhan", "lay", "dong", "cua", "con", "thi", "tranh", "noi", "diem", "dia", "dich", "cuoi", "cung", "nguoi", "dang",
            "ngang", "chan", "lai", "tien", "dau", "het", "sang", "nhung", "nho", "duong", "duoc", "van"]
KEY_CUE_SET = set(KEY_CUES)
KEY_CUES2 = ["da", "bo", "vua"]   # đợt 18: chữ then chốt của câu gây nhiễu chỉ dùng để SỬA lỗi gõ ("ã rời" -> "đã rời", "ỏ qua" -> "bỏ qua")
BIGRAM_FIX = True     # sửa âm tiết hợp lệ gõ sai theo từ đứng sau (MissionParser2._fix_seq)


# âm tiết tiếng Việt (không dấu) hợp lệ: phụ âm đầu + vần + phụ âm cuối. Lỗi gõ của dữ liệu thường tạo âm tiết SAI cấu trúc
# ("qau", "acn", "hge", "mnag", "nhna", "iao"), còn từ thật hiếm ("giờ", "hỏng", "tòa") vẫn hợp lệ -> không bị "sửa"
_VOWELS = ["a", "ai", "ao", "au", "ay", "e", "eo", "eu", "i", "ia", "ie", "ieu", "iu", "o", "oa", "oai", "oao", "oay", "oe",
           "oeo", "oi", "oo", "u", "ua", "uay", "ue", "ui", "uo", "uoi", "uou", "uu", "uy", "uya", "uye", "uyu", "y", "ya",
           "ye", "yeu"]
_SYL = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnrstvx])?(" +
                  "|".join(sorted(_VOWELS, key=len, reverse=True)) + r")(ng|nh|ch|[cmnpt])?$")


def _valid_syllable(t):
    return bool(_SYL.match(t))


# đoán loại tên lạ: thêm "láng giềng gần nhất" (tên đã biết giống nhất của từng loại) và "câu mô tả loại" (e5, không cần học)
# 0 = tắt. Đo bằng scratch/h28_knowledge_holdout.py (tên hoàn toàn mới) + h1_alias_holdout.py --fair:
#   KNN 1 + DESC 1: h28 trong câu 0,942 -> 0,956 (có dấu), giấu tên công bằng giữ 0,9983, validation 1,0000.
#   (CHAR_W 0,5 cho h28 tới 0,967 nhưng giấu tên công bằng 0,9983 -> 0,9979: không dùng)
KNN_W, KNN_T = 1.0, 0.03
DESC_W, DESC_T = 1.0, 0.02


def _per_type_max(sims, labels):
    out = np.full(10, -1.0)
    for k in range(10):
        m = labels == k
        if m.any():
            out[k] = sims[m].max()
    return out


def _soft(x, T):
    e = np.exp((x - x.max()) / T)
    return e / e.sum()


def _typo1(t, q):
    """t là q bị gõ sai đúng kiểu của dữ liệu: đảo 2 chữ kề nhau, hoặc rơi / thừa 1 chữ (không tính THAY chữ:
    "qau" là "qua" đảo chữ, không phải "đầu")."""
    if len(t) == len(q):
        d = [k for k in range(len(t)) if t[k] != q[k]]
        return len(d) == 2 and d[1] == d[0] + 1 and t[d[0]] == q[d[1]] and t[d[1]] == q[d[0]]
    if abs(len(t) - len(q)) == 1:
        a, b = (t, q) if len(t) < len(q) else (q, t)
        return any(b[:k] + b[k + 1:] == a for k in range(len(b)))
    return False


# đợt 20: "Đừng dừng ở X" (câu có dấu: "dừng" -> "dungx"), "Chưa cần tới X" (h94_dis_noneg.py)
NEG_SENT4 = r"|\bdung dungx\b|\bchua (can|phai) (ghe|den|toi|qua|vao)\b|\bthi khong( nhe| nha| a| dau)?$"


# đợt 20 (h107): "không cần / không phải / không nên / chẳng cần / khỏi cần" chỉ PHỦ ĐỊNH ĐỊA ĐIỂM khi theo sau là động từ đi
# lại / giao nhận, chính địa điểm ("x" = tên quen đã che, từ mở đầu tên), "là", "ở", "đâu / nữa" hoặc hết vế. Trước đây mọi
# "không cần ..." đều phủ định: "Giao tới thư viện, KHÔNG CẦN báo lại / ký nhận / đi đường tắt" -> thư viện thành gây nhiễu,
# câu mất đích (đoán từ cả câu). Train + validation: không có vế phụ kiểu này trong câu có đích (kiểm tra bằng h95).
NEG_SCOPE_FIX = True
_PLACE_CONT = r"(ghe|den|toi|qua|giao|mang|dua|gui|chuyen|vao|tat|re|x|o|tai|lay|nhan|dung|la|dau|nua|dem|ve|sang|ra|" \
              r"diem|dich|dia|cai|ben|phia|huong|chinh|nguoi|ai|" + "|".join(sorted(PLACE_HEADS - {"cho"})) + r")"
# ("đi" chỉ tính khi theo sau là hướng tới nơi nào đó: "không cần đi qua X" -- "không cần đi đường tắt" thì không;
#  "cho" không dấu = chờ / chỗ / cho: "không cần chờ" không phải phủ định địa điểm)
_SCOPED = r"( thiet| phai| lai| di| can){0,2} " + _PLACE_CONT + r"\b"
# "X (thì) không cần (đâu / nữa)" hết vế: phủ định CHÍNH X. (Không nhận "không cần phải" còn sót sau khi bộ lọc cờ xóa cụm
#  "nâng niu" phía sau: "Giao tới X không cần phải nâng niu".)
_NEG_END = r"|\b(x|thi) (khong|chang|khoi) (can|phai)( thiet)?( dau| nua| a| nhe)?\s*$"
# "đừng + động từ": chỉ phủ định địa điểm khi động từ nhận nơi chốn ("đừng ghé / nhầm với X") hoặc theo sau là hướng tới nơi
# đó ("đừng mang tới X", "đừng đi qua X"); "đừng đi đường vòng", "đừng giao muộn", "đừng đến trễ" thì không (h108).
_DUNG_OK = r"(nham|ghe|tat|re|vao|qua(?! (muon|tre|cham|lau|som|nhanh))|den(?! (muon|tre|cham|som))|toi(?! (muon|tre|cham|som))" \
           r"|(di|mang|giao|dua|dung|chuyen|gui)( \w+){0,2} (qua|toi|den|vao|ve|sang|ra|ngang|o|tai|x)\b)"
NEG_SENT_SCOPED = NEG_SENT.replace(r"\bkhong (can|phai|nen|ghe|den|toi|qua(",
                                   r"\bkhong (can|phai|nen)" + _SCOPED + r"|\bkhong (ghe|den|toi|qua(") \
                          .replace(r"\bdung (nham|ghe|den|toi|qua|di|mang|giao|dua|re|vao|tat|dung)\b", r"\bdung " + _DUNG_OK + r"\b") \
                          .replace(r"\bkhoi (can|phai|ghe)\b", r"\bkhoi ghe\b|\bkhoi (can|phai)" + _SCOPED) \
                          .replace(r"\bchang can\b", r"\bchang can" + _SCOPED) + _NEG_END
NEG_SENT2_SCOPED = NEG_SENT_SCOPED + NEG_SENT2[len(NEG_SENT):]
assert NEG_SENT_SCOPED != NEG_SENT and NEG_SENT2.startswith(NEG_SENT)


def _neg():
    if NEG_SCOPE_FIX and NEW_FRAME_FIX and VIA_FIX:
        return NEG_SENT2_SCOPED + NEG_SENT3 + (NEG_SENT4 if DIS_UNAVAIL else "")
    return (NEG_SENT2 + NEG_SENT3 + (NEG_SENT4 if DIS_UNAVAIL else "") if VIA_FIX else NEG_SENT2) if NEW_FRAME_FIX else NEG_SENT


# chữ chỉ hướng dùng theo nghĩa KHÁC (không dấu trùng nhau): "phải được ghé" (bắt buộc), "nhẹ tay", "trên đường",
# "đông người", "đóng cửa", "bác bảo vệ", "nằm gần", "trái cây" -> không được tính là manh mối hướng (h27)
NOT_DIR = [r"\bphai (?=(duoc|co|la|di|ghe|giao|mang|dua|tat|re|qua|toi|den|lay|nhan|chuyen|van|nho|can|dem|cho|xong|ve|vao)\b)",
           r"\b(khong|chang|cha|dau|chu khong) phai\b",       # "không phải X" (= không là X), không phải "bên phải"
           r"\b(nhe|tan|bang|cam|ra|trong) tay\b", r"\btren (duong|xe|tay|vai|lung)\b", r"\bduoi (mai|troi|mua|nang)\b",
           r"\bdong (cua|nguoi|goi|hop)\b", r"\bbac (si|bao|giu|tai|trong|lao|cong)\b",
           r"\bnam (?=(gan|xa|o|sat|canh|ke|ngay|giua|cach)\b)", r"\btrai cay\b"]


def _all_ref(ws):
    """Cụm chỉ gồm từ chỉ vị trí ("nửa trên bản đồ", "phần phía trên", "phía trên cùng") -> không phải tên địa điểm."""
    ref = REF_WORDS | ORI_WORDS | DIRS | {"phan", "cung", "tit", "han", "vung"} if NEW_FRAME_FIX else REF_WORDS
    return all(w in ref for w in ws)


def _dir_text(s):
    if NEW_FRAME_FIX:
        for p in NOT_DIR:
            s = re.sub(p, " ", s)
    return s
# cụm gấp / dễ vỡ ("không cần vội", "không quá gấp", "không lo vỡ", "không được chậm"...) nói về HÀNG, không phải về địa điểm:
# bỏ chúng đi trước khi xét câu có phủ định một địa điểm hay không ("Giao hộp tới X, không cần vội" -> X vẫn là đích)
FLAG_STRIP = re.compile("|".join(f"(?:{x})" for x in (URGENT_NEG, FRAGILE_NEG, DOUBLE_NEG, URGENT_POS, FRAGILE_POS,
                                                       r"\b(dung|khong duoc|cam) (di |chay |lam |giao |de |mang |dua )?(cham|tre|lau|muon)\b")))
# không dấu "vội" = "với": "đừng nhầm với X" khớp mẫu "đừng ... vội" -> cụm có "nhầm" luôn là phủ định địa điểm, giữ lại
FLAG_KEEP = re.compile(r"\b(nham|lan)\b")


ROAD_CTX = re.compile(r"\b(duong (hoi |rat |kha )?(tron|xau|ngap|uot|dong|toi)|tron truot|troi (dang |sap )?mua|mua (to|lon|phun)|"
                      r"suong mu|toi troi|dong nguoi|di duong)\b")   # "Cảm ơn, đi đường cẩn thận nhé" (h32): dặn đi đường


CAREFUL_OTHER = r"\bcan than (keo|ke|de khong)( bi)? (nham|lac|sai|tre|muon|cham|quen|lo|qua gio)\b"


def _road_free(s):
    """Câu dặn đi đường ("Đường hơi trơn, đi cẩn thận nhé", "trời mưa, chạy cẩn thận"): "cẩn thận" nói về đường đi,
    không phải về hàng -> bỏ cụm đó trước khi xét dấu hiệu DỄ VỠ (kiến thức chung; tỉ lệ dễ vỡ báo trên test cao bất thường)."""
    if NEW_FRAME_FIX and ROAD_CTX.search(s):
        s = re.sub(r"\b(di|chay|lai xe|lai)( duong| xe)?( (that|that su|rat))? can than\b", " ", s)
    return s


DNEG_FIX = True
_DNEG = r"\b(khong|chang) (phai|he) (la )?(khong|chang)\b(?= (gap|voi|khan|can gap|de vo|de be|de hong|mong manh|chac|ben)\b)"


def _strip_flags(s):
    if not NEW_FRAME_FIX:
        return s
    s = FLAG_STRIP.sub(lambda m: m.group() if FLAG_KEEP.search(m.group()) else " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ORI = r"(phia|ben|man|huong|goc|dau|mien|canh|mep|ria|khu|nua)"


MERGE_RULES = True   # gộp "tòa/khu + tên đã biết", kéo cụm tên qua "của", "gần/xa" lấy từ sát mốc nhất


# Bỏ dấu làm nhiều chữ khác nghĩa trùng nhau ("gấp"/"gặp", "bền"/"bên", "vội"/"với"). Nếu câu gốc CÓ dấu thì một từ khóa
# gấp / dễ vỡ chỉ được tính khi dạng có dấu của nó nằm trong danh sách dưới đây; không thì che nó đi trước khi áp luật.
ACC_FORMS = {"gap": {"gấp"}, "khan": {"khẩn"}, "voi": {"vội"}, "ben": {"bền"}, "gom": {"gốm"}, "su": {"sứ"}, "vo": {"vỡ"},
             "be": {"bể"}, "kip": {"kịp"}, "tre": {"trễ"}, "le": {"lẹ"}, "lau": {"lâu"}, "nut": {"nứt", "nút"}, "hong": {"hỏng"},
             "cham": {"chậm", "chạm"}, "som": {"sớm"}, "roi": {"rơi", "rồi"}, "lac": {"lắc"}, "va": {"va", "và"},
             "lien": {"liền"}, "mau": {"mau"}, "em": {"êm"}, "su": {"sứ"}, "hu": {"hư"}, "do": {"đổ", "đồ"}, "muon": {"muộn"},
             "doi": {"đợi"}, "ngay": {"ngay"}}
ACC_CHECK = True
PRESENT_RULE = True    # tên đã biết mà loại không có trên bản đồ -> chắc chắn là địa điểm gây nhiễu (đúng 3297/3297 trên train + validation)
DUNG_CHECK = True      # câu có dấu: chỉ "đừng" mới là phủ định ("dừng lại ở X" là điểm ghé)
DIS_RULE = True        # địa điểm đứng riêng trong câu phủ định / quá khứ, không có động từ ghé hay từ thứ tự -> gây nhiễu
ORD_CUE = r"\b(truoc|xong|sau (do|khi|day)|dau tien|roi (moi|hay|thi)|tiep (theo|do)|ke do|tren duong|tien duong|da roi)\b|\bda$"
VIA_VERB = r"\b(ghe|tat|qua|re|dung chan|ngang|giao|mang|dua|gui|chuyen|toi|den|diem giao|dang cho|can nhan)\b"
PAST_CUE = r"\b(da|vua|hom (qua|truoc|kia|no)|lan truoc|truoc day|tung|truoc kia|luc nay|khi nay)\b"
# đợt 20 (scratch/h94_dis_noneg.py, kiến thức chung): câu đứng riêng nói một nơi KHÔNG HOẠT ĐỘNG ("X đang sửa chữa", "Hôm nay X
# nghỉ", "X đã khóa cửa", "X ngừng hoạt động từ tuần trước") -> gây nhiễu, dù không có chữ phủ định. "trước" trong "tuần / hôm /
# lần trước" là thời gian, không phải dấu hiệu thứ tự "ghé X trước".
DIS_UNAVAIL = True
UNAVAIL_CUE = r"\b(nghi|dong cua|khoa cua|sua chua|bao tri|ngung (hoat dong|nhan|phuc vu|lam viec)|tam (dong|nghi|ngung)|" \
              r"het gio (lam viec|hanh chinh|mo cua|phuc vu)|mat dien|chuyen di|don di|vang (nguoi|mat)|huy|bi huy|da huy|" \
              r"(de|cho) (lan|dip|hom|khi|bua) (sau|khac)|lan sau|de sau)\b"
UNAVAIL_SPAN = r"(dong|khoa|mo|sua|then) cua( \w+)?|ngung hoat dong|hoat dong"     # "khóa cửa" không phải tên địa điểm
TIME_TRUOC = r"\b(tuan|hom|lan|thang|nam|ngay|buoi|ky|dot|tiet|ca|chuyen|dip) truoc\b|\btruoc (day|kia)\b"
PART_OR = True         # câu phụ nhiều vế: xét từng vế rồi gộp
MIXED_POLICY = "last"  # câu trộn (vế gấp + vế không gấp): "calm" = như bản 0.9467; "urgent" = bản v13 (0.9461); "last" / "first"
CALM_WINS = True       # câu nói rõ "không gấp / không dễ vỡ" thắng câu khác chỉ được mô hình nghĩa đoán là gấp / dễ vỡ
E5_POS_URGENT = True  # mô hình nghĩa được tự nhận câu là gấp khi không có từ khóa nào?
E5_POS_FRAGILE = True   # ... và tự nhận câu là DỄ VỠ? BẬT LẠI (06/10 tối): v31c tắt cờ này thì bảng xếp hạng 0,9778 -> 0,9733 (44 dòng robot 7);
                        # nghĩa là test có câu dặn dễ vỡ kiểu mới mà chỉ e5 bắt được. Lập luận cũ (dưới đây) đã sai.
                        # (cũ) trên validation (bộ đọc chỉ học train) e5 tự bật dễ vỡ 0 lần
                        # (bảng cụm + luật bắt đủ 110 / 110); trên câu tự soạn (scratch/h32_e5_flags.py) e5 tự bật nhầm ~3% câu
                        # không dễ vỡ (chào / kết, "đồ nhựa dẻo") và bắt đúng ~10% câu dễ vỡ mới, mà câu không dễ vỡ nhiều hơn
                        # ~5 lần; xác suất e5 không tách được hai nhóm. (Gấp: hai bên ~ cân bằng -> giữ.)
USE_PLACES_MORE = True    # học thêm các cách gọi địa điểm viết tay bổ sung 2026-10-06 (nlp_knowledge.PLACES_MORE)
USE_PHRASES_MORE = True   # học thêm các ví dụ viết tay bổ sung 2026-10-04 (nlp_knowledge.PHRASES_MORE)
KHONG_STRICT = True
REAL_WORDS_NEAR_KHONG = {"hong", "thong", "phong", "chong", "khoang", "khung", "khang", "nhong", "khon", "khoong"} - {"khoong"}
# Danh sách trên mới là lớp bảo vệ chính. Tần suất chỉ là lớp dự phòng: trên train + validation, từ THẬT nằm cạnh "không"
# xuất hiện >= 54 lần (phong 960, hong 199, chong 54) còn LỖI GÕ chỉ <= 5 lần (khogn 5, khng 4, khnog 2...).
# Để ngưỡng 5 làm "khôgn cần ghé X" bị đọc thành điểm ghé -> cả 10 robot sai.
KHONG_FREQ = 20
VETO_NEEDS_NEG = True   # mô hình nghĩa (hiểu phủ định kém) chỉ được bác từ khóa gấp / dễ vỡ khi câu có từ phủ định
E5_RESTORE = False  # (TẮT: hòa trên bộ thử, 0 dòng test) đợt 17: câu không dấu -> thêm dấu (bảng học từ dữ liệu) trước khi hỏi e5 về cờ (scratch/h62_e5_nearmiss.py)
E5_TH_R = 0.8
E5_TH = (0.6, 0.8)   # ngưỡng tin mô hình nghĩa cho câu gấp/dễ vỡ lạ: (câu có dấu, câu không dấu). Đã thử (0.5, 0.6): bảng xếp hạng GIẢM 0.9444 -> 0.9422
DETECT_RULES = True  # giữ "xá" cuối cụm, nối cụm qua "ở", nhận cụm không từ khóa khi ngữ cảnh rất rõ (scratch/h7)
LOC_PREPS = {"o", "toi", "den", "qua", "ghe", "tai", "ve", "sang", "vao"}
CHO_HEAD = True      # giữ "chỗ" ở đầu cụm tên lạ (trước đây bị gọt như từ "cho")
CHAR_W = 1.0         # trọng số n-gram ký tự khi đoán loại tên lạ. (0.5 từng cho 0,9984 nhưng đo lại 4 hạt giống: 0.5 -> 0,9971,
                     # 1.0 -> 0,9978; chênh lệch < 0,001 giữa các lượt là nhiễu số học của e5 theo lô, không dùng để chọn)
RESTORE_ACC = True   # tên lạ không dấu: khôi phục dấu trước khi đưa vào mô hình nghĩa e5
E5_BOTH_FORMS = False  # đợt 19: tên lạ không dấu -> trung bình chứng cứ e5 của dạng khôi phục dấu và dạng không dấu gốc
E5_RAW_W = 0.5         # trọng số của dạng không dấu gốc khi E5_BOTH_FORMS
TYPE_FIX = True      # đợt 10 (scratch/h31_style_aliases.py, tên mới kiểu validation thêm vào so với train): "nơi / chỗ + việc làm"
                     # ("chỗ lấy thuốc"), "ô tô", "ở" cuối tên ("dãy phòng ở"), trợ từ "nha" cuối câu, tra thẳng kho tên tự soạn,
                     # khôi phục dấu nhìn cả từ đứng TRƯỚC lẫn đứng SAU và học thêm từ kho tên tự soạn ("tra cuu" -> "tra cứu")
KB_BOOST = 3.0       # điểm cộng (thang log) cho loại của cụm khớp ĐÚNG một tên trong kho tên tự soạn (nlp_knowledge)
VIA_FIX = True       # đợt 14 (scratch/h36_via.py, khung điểm ghé mới tự soạn: 158 / 200): "Nơi lấy hàng là A", "Đồ đang để ở A,
                     # mang sang B", "Chở thêm đồ từ A sang B", "Đơn gồm hai chặng: A và B", "Điểm lấy hàng: nhà thi đấu"
VIA_SEMI = VIA_MASK = VIA_KHOI = VIA_CHO = VIA_ROLE = True   # công tắc con của VIA_FIX (để quy thay đổi trên test)
VIA_TYPE_FIX = True  # đợt 15: không tìm thấy tên đích + có điểm ghé đã biết -> đích không được trùng loại điểm ghé
ITEM_NOT_BEFORE = {"lay", "nhan", "giao", "gui", "dua", "mang", "chuyen", "noi", "diem", "cho", "de", "cat", "dia"}
STORED_AT = r"\b(do|hang|mon do|goi hang|kien hang|tui do|thung hang|hop)( \w+){0,2} (dang |duoc |da |van )?(de|cat|gui|nam)( san)? (o|tai)$"
DESC_MORE = False    # đợt 13: thêm nlp_knowledge.TYPE_DESC2 (8 mô tả / loại) vào nguồn "mô tả loại" khi đoán loại tên lạ
SPATIAL_FIX2 = True  # đợt 12 (scratch/h35_spatial.py): "ở phía bên tay phải" sai 12 / 12, "ở mé trái" sai 12 / 12
ITEM_FIX = True      # đợt 11 (scratch/h33_items.py): MÓN HÀNG không phải địa điểm và không mang cờ gấp / dễ vỡ. Train + validation:
                     # món hàng chỉ ở dòng "Hàng: X" (221 lần, đều trung tính) hoặc ở chỗ "giao X tới"; món tương quan với loại
                     # đích ("thẻ thư viện", "túi sơ cứu") nhưng độc lập với cờ (mọi món ~ tỉ lệ dễ vỡ chung 0.40); không câu giao
                     # hàng nào chứa lời dặn gấp / dễ vỡ (0 / 1145). Món MỚI của test ("sách thư viện", "Hàng: bình hoa") không
                     # được cướp vai đích / điểm ghé hay bật cờ.
ITEM_LINE = re.compile(r"^(hang|mon hang|kien hang|mat hang|hang hoa|vat pham)\s*:\s*\S")
DELIVER_VERBS = {"giao", "gui", "dua", "mang", "chuyen"}
SLOT_END = {"toi", "den", "qua", "sang", "ve", "vao", "ra", "cho", "tai", "o", "tu", "ghe", "len", "xuong", "nhe", "giup", "gium",
            "la",     # "Nơi cần giao LÀ cổng chính ở phía trên": "là" không mở đầu món hàng (h27: bản đầu làm mất đích)
            "nho", "roi", "xong", "nhung", "thi", "truoc", "sau", "va"}   # "Giao thư viện; NHỚ qua X trước" (h27c --v7)
DEST_PREPS = {"toi", "den", "qua", "sang", "ve", "vao", "cho"}
DELIVER_END = DEST_PREPS | {"o", "tai"}    # "Có đơn giao <món> ở <nơi>"
MODALS = {"phai", "can", "se", "nen", "duoc", "da", "dang", "vua"}       # "<món> phải được đưa tới", "<món> cần chuyển tới"
ITEM_IS = re.compile(r"\b(hang|mon hang|kien hang|mon|do)( (can|se) (giao|gui|chuyen|dua|mang))?$")   # "hàng cần giao là <món>"
PICK_VERBS = {"lay", "nhan"}           # "lấy / nhận <món> ở / tại / từ A": món lấy ở điểm ghé
PICK_END = {"o", "tai", "tu"}
PICK_NOUN = {"nguoi", "diem", "noi", "ben", "khach", "cho", "phong", "ky"}   # "người nhận", "điểm nhận", "nơi nhận": danh từ
ITEM_FIX2 = True    # đợt 16: "ghé X lấy <món>" hết vế: món hàng có tên địa điểm ("thẻ thư viện") không phải điểm ghé / đích
ITEM_TAIL_END = {"truoc", "roi", "xong", "nhe", "nha", "giup", "gium", "dum", "da", "sau", "thi", "va", "luon", "ngay"}
# lời dặn RÕ RÀNG (vị ngữ) vẫn được tính nếu lỡ nằm trong dòng "Hàng: ..." ("Hàng: đồ dễ vỡ"); danh từ (thủy tinh, cấp cứu) thì không
ITEM_FRAGILE_PRED = r"\bde (vo|be|hong|nut|me|gay|mop)\b|va (dap|cham)|mong manh|nhe tay|can than|nang niu|\bky va\b|tranh (va|rung|xoc)"
ITEM_URGENT_PRED = r"\bgap\b|\bkhan\b|hoa toc|lap tuc|cang (nhanh|som)|cham tre|\bcap toc\b|\btuc toc\b"
NAME_TAIL = {"nha", "nhen", "nghen"}          # trợ từ cuối câu dễ bị dính vào tên lạ ("phòng quản lý sinh viên nha")
NHA_KEEP = {"toa", "khu", "day", "can", "ngoi", "mai", "cuoi", "dau", "o"}   # "tòa nhà", "dãy nhà", "nhà ở"...: "nhà" thuộc tên
FRAME_RULE = True    # "ghé / tạt qua X lấy / nhận ... trước": X là địa điểm
FRAME_END = {"lay", "nhan", "lanh", "truoc", "xong"}
FRAME_CUT = {"nam", "o", "gan", "xa", "sat", "phia", "ben", "man", "huong", "canh", "cach", "ke", "mien", "tai"}
SAME_ENT_HEADS = True   # "<người> ở khu / tòa <nơi>": vẫn là một (người nhận + nơi làm việc), cùng vai trò
O_IN_NAME = {"khu", "nha", "phong", "cho", "noi", "toa", "day", "can"}   # "khu ở", "nhà ở", "chỗ ở": "ở" nằm trong tên
# "chỗ ở NỘI TRÚ", "khu ở TẬP THỂ", "nhà ở DÀNH CHO...": "ở" + từ định danh nơi ở -> vẫn là một tên (đợt 16, kiến thức chung)
O_QUAL = {("noi", "tru"), ("tap", "the"), ("ky", "tuc"), ("danh", "cho"), ("cho", "sinh"), ("cho", "tan"), ("cua", "hoc"),
          ("sinh", "vien"), ("cho", "nguoi"), ("lau", "dai")}
DIR_RULE_WINS = True   # hướng đọc từ từ khóa ngay sau tên địa điểm thắng bộ phân loại khi hai bên ra hai hướng khác nhau
PHRASE_NEEDS_CUE = True # bảng cụm từ chỉ gán gấp / dễ vỡ khi cụm có từ ngữ gấp / dễ vỡ (0 cụm bị bỏ trên train + validation đầy đủ)
DIR_GUARD = True     # bộ phân loại chỉ được trả về hướng khi ngữ cảnh có từ chỉ hướng / vị trí
DIR_HINTS = ["bac", "nam", "dong", "tay", "tren", "duoi", "trai", "phai", "man", "phia", "ben", "huong", "goc", "mien",
             "canh", "mep", "ria", "nua", "dau", "cuoi", "top", "left", "right", "north", "south", "east", "west"]
DIR_GUARD2 = True    # đợt 17: ngữ cảnh "phía / mặt / bên trước | sau" mà không có TỪ HƯỚNG (bắc/nam/.../trái/phải) -> không phải hướng
                     # (bản đầu đòi từ hướng ở mọi nơi: làm mất "bên tái (trái) bản đồ" gõ sai, validation)
                     # ("cổng phía trước": "phía trước" không phải hướng trên bản đồ; h48 train)
DIR_CORE = ["bac", "nam", "dong", "tay", "tren", "duoi", "trai", "phai", "top", "left", "right", "north", "south", "east", "west"]
GOAL_FRAME_RULE = True  # "mang ... tới <X>", "đích đến là <X>": X là địa điểm
# (không dùng các chữ không dấu trùng với chữ trong tên địa điểm: nha~nhà, cho~chỗ, de~để, thi~thí, can~căn, nhan~nhân)
GOAL_FRAME_END = {"giup", "nhe", "va", "roi", "truoc", "xong", "lay", "hon", "nhung", "duoc", "luon", "dum", "gium", "voi", "di",
                  "minh", "nhanh", "ngay", "dang"}
ANCHOR_RULE = True   # "<địa điểm> gần / xa <Y> (hơn)": Y là mốc
REF_SLOT = True      # đợt 16: "tới / ở / là <X> gần Y hơn | phía bắc": X là tên địa điểm dù không mở đầu bằng phòng / khu / nơi
                     # (trước đây X lạ bị bỏ -> mốc Y bị nhận làm đích; h48_alias_swap.py "cần ... ở góc tài liệu tham khảo cách xa căng tin")
# "X TRƯỚC 10 giờ / trước giờ học / trước buổi trưa": hạn giờ, không phải "ghé X trước" (h27e: "Giao tới căng tin trước 10 giờ")
TRUOC_TIME = r"truoc (\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)( \w+)? ?(h|g|gio|phut|tieng)\b|truoc (gio|buoi|luc|ngay|toi|trua|" \
             r"chieu|sang|mai|thu|han|deadline|ca|tiet|cuoi|dau|giua|khi|lich)\b"
HEAD_ACC_FIX = True  # đợt 16: dạng CÓ DẤU hợp lệ của từ mở đầu tên địa điểm (HEADS không dấu nhập nhằng: kỵ / ký, cần / căn...)
HEAD_ACC = {"bai": ("bãi",), "ban": ("ban", "bàn"), "bep": ("bếp",), "can": ("căn",), "cho": ("chỗ", "chợ"), "chot": ("chốt",),
            "cong": ("cổng",), "day": ("dãy",), "diem": ("điểm",), "giang": ("giảng",), "ham": ("hầm",), "hanh": ("hành",),
            "hoi": ("hội",), "ky": ("ký", "kí"), "loi": ("lối",), "lop": ("lớp",), "nha": ("nhà",), "noi": ("nơi", "nội"),
            "phong": ("phòng",), "quan": ("quán", "quản"), "quay": ("quầy",), "san": ("sân", "sảnh"), "thu": ("thư", "thủ"),
            "toa": ("tòa", "toà"), "tram": ("trạm",), "van": ("văn",), "vien": ("viện",), "xuong": ("xưởng",), "tang": ("tầng",),
            "sanh": ("sảnh",), "trung": ("trung",)}   # đợt 17: "trứng gà" không phải "trung tâm"
ANCHOR_NEEDS_REL = True  # đợt 16: vai "mốc" cần từ quan hệ (gần / xa / sát / cạnh / kế / cách) trong 4 từ đứng trước
SUBJ_TRIM = True   # đợt 17 (h48 train): chủ ngữ "<người> <nơi> cần nhận": lấy phần nơi; "khu phòng ở cần nhận": "ở" thuộc tên
SUBJ_PLACE_HEADS = {"noi", "phong", "khu", "toa", "nha", "day", "goc", "san", "bai"}
NAME_QUAL_FIX = True
_NUM = r"(mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi|\d+)"
NAME_QUAL = (r"(tang (tret|ham|tren cung|" + _NUM + r")|lau " + _NUM + r"|(khu|day|toa|day nha|toa nha) ([a-h]|\d+|moi|cu|chinh|phu)"
             r"|co so " + _NUM + r"|so " + _NUM + r"|toa nha chinh|toa chinh"
             r"|danh (rieng )?cho (sinh vien|can bo|giao vien|giang vien|hoc vien|khach|nhan vien|tan sinh vien|nu sinh|nam sinh)"
             r"( nam (nhat|hai|ba|tu|cuoi))?|moi xay( dung)?|cua (truong|khoa( [a-z]+){0,2}|vien|ky tuc( xa)?))")
FIT_SEED = 0         # hạt giống ngẫu nhiên khi học bộ đọc (0 = như mọi bản trước)
PICKUP_RULE = True   # "đến X lấy / nhận ..." -> X là điểm ghé (luật viết từ phép thử tự soạn; bật 0 lần trên train + validation)
DIR_MORE = True      # đợt 18: "phía đầu / cuối / đáy bản đồ", "hướng lên trên"
TYPO_FIX2 = True     # đợt 18: chữ then chốt gõ sai có dạng dấu lạ / theo ngữ cảnh hai phía
RECIP_FRAME = True   # đợt 18: khung người nhận "giao ... cho <người có nghề / nơi>", "Người nhận là / : X"
CUA_DOOR = {"chinh", "sau", "truoc", "phu", "ra", "vao", "hong", "lon", "nho", "kinh", "tiep", "so"}   # "cửa chính / sau / phụ..."
TAI_NOUN = {"lieu", "chinh", "vu", "san", "xe", "khoan", "nang", "tro"}   # "tài liệu / chính / vụ / sản / xế..." (không dấu)
PERSON_LEADS = (("ban", "quan", "ly"), ("can", "bo"), ("nhan", "vien"), ("co",), ("chu",), ("anh",), ("chi",), ("thay",), ("bac",))
GOAL_SLOT_STRONG = True   # đợt 17: "Điểm giao: X." (X trọn vế) chắc chắn là địa điểm dù không mở đầu bằng từ quen ("góc tra cứu")
EDGE_ACC_FIX = True   # đợt 17: câu CÓ DẤU: chữ khung ở mép tên chỉ bị gọt khi đúng là dạng khung ("tới", không phải "tối":
                      # "nơi ăn tối"; "về" không phải "vẽ": "phòng vẽ"; "bên" không phải "bến"; "dịch" vụ; "tân" sinh viên)
EDGE_ACC = {"toi": ("tới", "tôi"), "da": ("đã",), "can": ("cần",), "ve": ("về",), "nhan": ("nhận",), "lay": ("lấy",),
            "qua": ("qua", "quá"), "dang": ("đang",), "tre": ("trễ",), "den": ("đến",), "ghe": ("ghé",), "roi": ("rồi",),
            "voi": ("với", "vội"), "keo": ("kẻo",), "tan": ("tận",), "thi": ("thì",), "minh": ("mình",), "dich": ("đích",),
            "la": ("là",), "va": ("và",), "ben": ("bên",), "canh": ("cạnh",), "cach": ("cách",), "man": ("mạn",),
            "phia": ("phía",), "huong": ("hướng",), "nhat": ("nhất",), "hon": ("hơn",), "truoc": ("trước",), "giup": ("giúp",),
            "khong": ("không",), "gan": ("gần",), "sat": ("sát",), "ke": ("kế", "kề"), "tai": ("tại",), "ngang": ("ngang",)}
LONE_BAD = LONE_BAD | {"tram"}   # đợt 16: "Trạm dừng đầu tiên là X": "trạm" đứng một mình không phải tên (train + val: luôn "trạm xá / y tế")
PERSON_TYPO_FIX = True  # đợt 16: khung "<người> cần nhận / đang chờ" tìm trên chuỗi đã sửa lỗi gõ chữ khung ("cần nhn")
NHAN_VIEN_FIX = True    # đợt 16: "đến X nhân viên..." không phải "đến X NHẬN"
FRAME_STRONG = True  # đợt 16: khung câu CHẮC CHẮN có tên địa điểm ("Hàng cho X:", "ghé X lấy / trước", "gần / xa Y hơn",
                     # "nơi cần giao là X"): nhận cụm X / Y lạ dù không mở đầu bằng từ quen (h48_alias_swap.py)
# dạng CÓ DẤU của các từ khung (FRAME_CUT) không dấu: chữ có dấu khác là chữ của tên ("kệ sách", "tài liệu", "cửa ngõ", "nam")
CUT_ACC = {"tai": ("tại",), "ke": ("kế", "kề"), "o": ("ở",), "nam": ("nằm",), "gan": ("gần",), "xa": ("xa",), "sat": ("sát",),
           "canh": ("cạnh",), "cach": ("cách",), "phia": ("phía",), "ben": ("bên",), "man": ("mạn",), "huong": ("hướng",),
           "mien": ("miền",)}
ANCHOR_END = {"hon", "truoc", "lay", "lanh", "dang", "giup", "nhe", "roi", "xong", "va", "nhung", "nhat", "mot", "chut", "lam",
              "khong", "minh"}
CUE_OVERRIDE = True  # chỉ có 2 địa điểm: dấu hiệu trong câu rất mạnh (|điểm| >= CUE_MARGIN) thắng bộ phân loại vai trò
CUE_MARGIN = 4.5
FUZZY_REF_OK = True  # khớp mờ tên đã biết vẫn được nhận khi từ kế bên là từ chỉ hướng ("căng tin mạn dưới")
VIA_EXCLUDES_GOAL = True  # điểm ghé đã biết loại -> đích (tên lạ) không thể cùng loại
VIA_NOT_ANCHOR = True  # địa điểm đứng ngay sau "ghé / tạt qua" không thể là mốc gần / xa
PERSON_RULE = True   # chủ ngữ của "<người> cần nhận / đang chờ ..." là đích (người nhận lạ)
PERSON_STOP = {"minh", "toi", "em", "ban", "nguoi", "ho", "anh", "chi", "co", "thay", "ai", "hang", "don", "viec", "no", "cau",
               "chung", "moi", "ca", "nay", "do", "kia", "robot", "yeu", "cau", "nhan", "nho", "xin", "chao", "hom", "sang", "chieu",
               "luc", "bay", "gio", "va", "thi", "ma", "neu", "khi", "con", "cung", "da", "se", "vua", "rat"}
PERSON_STOP0 = {"minh", "toi", "em", "ban", "nguoi", "ai", "hang", "don", "viec", "robot", "yeu", "nho", "xin", "chao", "nay", "do",
                "va", "thi", "ma", "neu", "khi", "con", "cung", "da", "se", "vua", "rat", "luc", "bay", "hom"}
SEG_FLAGS = False    # (đã thử bật cùng lúc với hạ ngưỡng: bảng xếp hạng giảm) xét gấp / dễ vỡ cả ở vế câu (ngăn bởi dấu phẩy) không chứa địa điểm của câu giao hàng


def disambiguate(s, acc):
    """s: câu không dấu; acc: dạng có dấu (chuỗi) hoặc None. Che các từ khóa mà dạng có dấu cho thấy là chữ khác."""
    if not ACC_CHECK or not acc or acc == s or not re.search(r"[^\x00-\x7f]", acc):
        return s
    spans = list(re.finditer(r"[a-z0-9]+", s))
    aw = re.findall(r"[^\W_]+", acc.lower())
    if len(spans) != len(aw):
        return s
    out, last = [], 0
    for m, a in zip(spans, aw):
        w = m.group()
        if w in ACC_FORMS and a not in ACC_FORMS[w] and _unaccent(a) == w:
            out.append(s[last:m.start()] + "_" * len(w)); last = m.end()
    return "".join(out) + s[last:]


CACH_RULE = True   # "X cách Y xa / không xa / một đoạn ngắn": từ gần/xa đứng SAU địa điểm mốc


def rule_direction(ctx, post=""):
    """ctx: vài từ ngay sau tên địa điểm (đã che địa điểm kế tiếp thành PLC). -> kind hoặc None.
    Chỉ nhận khi từ chỉ hướng đi kèm từ định hướng ("phía", "bên", "mạn"...) hoặc "bản đồ"."""
    head = ctx.split(" PLC")[0] if "PLC" in ctx else ctx
    if "PLC" in ctx and len(head.split()) <= (5 if CACH_RULE else 4):
        neg = bool(re.search(r"\b(khong|chang)\b", head))        # "không xa X" = gần X
        if REF_NEG_FIX:
            neg = bool(re.search(NEG_REL, head))   # đợt 20: phủ định phải đứng SÁT từ quan hệ ("X, không gấp, gần Y" vẫn là gần)
        if MERGE_RULES:
            # từ quan hệ đứng SÁT mốc nhất quyết định ("trạm xá nằm gần X": chữ "xá" không phải "xa")
            rel = [w for w in head.split() if w in ("xa", "gan", "sat", "canh", "ke", "giap")]
            if rel:
                return ("near" if neg else "far") if rel[-1] == "xa" else ("far" if neg else "near")
        if CACH_RULE and re.search(r"\bcach\b", head) and not re.search(r"\b(xa|gan)\b", head):
            # post: vài từ ngay sau địa điểm mốc
            m = re.match(r"(la |co |chi |cung )?(khong |chang |cha |chua )?(rat |kha |hoi |qua |that |tuong doi |lam "
                         + (r"|bao |may |may lam |" if CACH_MORE else "") + r")?(xa|gan)\b", post)
            if m:
                far = m.group(4) == "xa"
                return ("far" if far else "near") if not m.group(2) else ("near" if far else "far")
            if CACH_MORE:
                # đợt 20 (kiến thức chung): "chỉ cách Y một chút / một xíu / chút xíu / vài bước chân" = gần;
                # "cách Y cả một quãng / một đoạn khá xa" = xa
                if re.match(r"(chi |co |khoang |cung )?((mot|vai|may|it) )?(chut|xiu|ti|ti xiu|chut xiu|chut it|buoc chan|buoc)( xiu| thoi| nua)?\b", post):
                    return "near"
                if re.match(r"(ca mot |ca |mot |hang )?(doan|quang|khoang)( duong)?( (rat|kha|hoi|qua|that|tuong doi|kha la))? (dai|xa)\b", post) \
                        or re.match(r"ca mot (quang|doan|khoang)( duong)?\b", post):
                    return "far"
            if re.match(r"(chi |co |khoang )?((mot|vai|may|it) )?(doan|quang|khoang|buoc|met|phut)( chan| duong)? ?(ngan|nho|thoi)?\b", post) \
                    and not re.search(r"\b(dai|xa)\b", post):
                return "near"
            if re.match(r"(ca |mot |hang )?(doan|quang|khoang)( duong)? (dai|xa)\b", post):
                return "far"
        if REF_TOWARD:
            # đợt 20 (kiến thức chung): "X (nằm) về phía / bên phía / hướng về / mạn Y" = phía Y = GẦN Y; "cùng phía với Y" = gần;
            # "phía bên kia / khác phía so với Y" = XA Y; "so với / tính từ Y thì gần / xa hơn" (từ quan hệ đứng SAU mốc)
            if re.search(r"\b(khac phia|khac ben|phia ben kia|ben kia|dau ben kia|dau kia|phia doi dien)( (voi|so voi|cua))?$", head):
                return "near" if neg else "far"
            if re.search(r"\b(huong|lech|nghieng|chech|dich|nam|o) ve$", head):
                return "far" if neg else "near"           # "X hướng về / lệch về Y" = về phía Y
            if re.search(r"\b(cung phia|cung ben|cung huong|cung mien)( voi)?$", head):
                return "far" if neg else "near"
            if re.search(r"\b(so voi|tinh tu|neu tinh tu|xet tu)$", head):
                m2 = re.match(r"(thi |la |co ve )?(khong |chang )?(gan|xa|sat)( hon)?\b", post)
                if m2:
                    far = m2.group(3) == "xa"
                    return ("near" if far else "far") if m2.group(2) else ("far" if far else "near")
            if re.search(r"(^|\b(ve|huong ve|lech ve|nghieng ve|nam|nam ve|o|o ve|ben|chech ve|dich ve) )(phia|ben phia|huong|man|mien)$", head) \
                    and not re.search(r"\b(bac|nam|dong|tay|tren|duoi|trai|phai|truoc|sau)$", head):
                return "far" if neg else "near"
        if re.search(r"\b(cach xa|xa)\b", head):
            return "near" if neg else "far"
        if re.search(r"\b(gan|sat|canh|ke|ke ben|ben canh|lien ke|giap)\b", head):
            return "far" if neg else "near"
    if DIR_MORE:
        # (đợt 18, kiến thức chung; scratch/h35_spatial.py): "phía ĐẦU bản đồ" = trên (trước đây bị đọc thành NAM),
        # "phía CUỐI / ĐÁY bản đồ" = dưới, "hướng lên trên", "hàng trên / dưới cùng"
        if re.search(r"\b(phia|o|nam|ve|huong|mep|ria) (dau|tren dau)( cua)? (ban do|so do|anh)\b|\bhuong len( phia)? tren\b|"
                     r"\b(hang|dai|day|tang|phan) tren cung\b", head):
            return "north"
        if re.search(r"\b(phia|o|nam|ve|huong|mep|ria) (cuoi|day)( cua)? (ban do|so do|anh)\b|\bhuong xuong( phia)? duoi\b|"
                     r"\b(hang|dai|day|tang|phan) duoi cung\b", head):
            return "south"
    for kind, w in (("north", "bac|tren"), ("south", "nam|duoi"), ("west", "tay|trai"), ("east", "dong|phai")):
        m = re.search(_ORI + r" (" + w + r")\b", head) or re.search(r"\b(" + w + r") (cung )?(cua )?ban do", head) \
                or re.search(r"^(o |nam )?(" + w.replace("nam|", "") + r") cung\b", head)
        if not m and DIR_BARE:
            # đợt 20 (kiến thức chung): "ở dãy / hàng / cột trên", "cực bắc", "ở bắc" (hết cụm) -- không nhận "tầng trên", "đông người"
            m = re.search(r"\b(hang|day|dai|cot|khu|phan|vung|dau|cuc) (" + w + r")\b(?! (nguoi|duc|cua|xe))", head) \
                or re.search(r"^(o|nam o|tai|nam) (" + w.replace("nam|", "").replace("tren|", "").replace("|phai", "") + r")$", head)
        if m:
            if REF_NEG_DIR and re.search(NEG_DIR_LEFT, head[:m.start()]):
                return _FLIP_DIR[kind]                  # đợt 20: "KHÔNG nằm ở phía bắc", "không phải cái phía tây" -> bản kia
            return kind
    return None


# ---- đợt 20 (07/10 tối): giới từ chỉ NƠI ĐẾN hiếm / chưa từng có trong dữ liệu: "chuyển / mang / đưa / đem / chở ... SANG /
# VÀO / VỀ / RA X" (sau động từ giao hàng: 0 / 12 / 3 / 0 lần trong train, 0 / 2 / 0 / 0 trong validation; "tới / đến" ~470).
# Bộ gán nhãn cụm và các luật khung chỉ quen "tới / đến / qua" -> tên LẠ đứng sau "sang / vào / ra" bị bỏ sót hoàn toàn
# (scratch/h90_frames_novel.py: tên lạ trong khung câu mới tự soạn 74%, tên quen cùng khung 97%). Đổi các giới từ này thành
# "tới" trước khi đọc câu. Kiến thức tiếng Việt chung, không lấy từ test.
PREP_CANON = True
REF_CANON = True       # "lân cận X" -> "gần X", "phương bắc" -> "phía bắc" (h98 vòng 3: 0 / 8 câu được nhận tham chiếu)
CACH_MORE = True       # "cách Y không bao xa / một chút / một xíu" = gần, "cách Y cả một quãng / một đoạn khá xa" = xa
DIR_BARE = True        # "ở dãy / hàng trên", "cực bắc", "ở bắc" (hết cụm) = hướng (không có "phía / bên")
REF_TOWARD = True      # "X về phía / bên phía / mạn Y" = gần Y; "phía bên kia so với Y" = xa; "so với Y thì gần hơn" (h103)
DETACHED_REF = True    # (THỬ) "Giao tới X. Lưu ý chọn cái phía bắc." (tham chiếu tách rời tên; 0 câu train + val)
NEG_TYPO_REF = True    # "khôg xa / khng gần" (sửa lỗi gõ chữ khung) và viết tắt "ko / k / kg / hok / hông" ngay trước gần / xa
NEG_ABBR = {"ko", "k", "kg", "hok", "hong", "kh", "khog", "kog"}


def _neg_word(words, k):
    """Chữ phủ định (kể cả gõ sai / viết tắt đứng ngay trước gần / xa): tên địa điểm dừng ở đây."""
    w = words[k]
    if w in ("khong", "chang", "chua", "cha"):
        return True
    return NEG_TYPO_REF and w in NEG_ABBR | {"khng", "kong", "khonh"} and k + 1 < len(words)         and words[k + 1] in ("xa", "gan", "sat", "canh", "ke", "cach", "o", "nam", "qua", "may", "bao")
# (đợt 20, THỬ — h91) khung chặng tự soạn: "Chặng một: V. Chặng hai: G.", "G là điểm cuối, còn V là điểm dừng đầu tiên",
# "Lúc đầu định giao tới D nhưng giờ đổi sang G"
FRAME2_FIX = False   # thử: 0 dòng test đổi -> không đưa vào v41
LEG_SUBJ = r"(diem|noi|cho|chang) (cuoi|nhan|giao|den|dich|ket thuc|chot|dung|ghe|lay|trung chuyen|dau|mot|hai|thu)"
LEG_GOAL_RIGHT = r"^la (diem|noi|cho|chang) (cuoi|nhan|giao|den|dich|ket thuc|chot|hai|thu hai)\b"
LEG_VIA_RIGHT = r"^la (diem|noi|cho|chang) ((dung|ghe) )?(dau|dau tien|thu nhat|ghe|lay|trung chuyen|mot|dung)\b"
PLAN_CHANGE = r"\b(luc dau|ban dau|thoat dau|truoc do|luc truoc|ke hoach (cu|ban dau|truoc))\b( \w+){0,4} (dinh|tinh|du dinh|muon)\b"
_CANON_PREPS = {"sang": "sang", "vao": "vào", "ve": "về", "ra": "ra"}       # dạng có dấu đúng của từng giới từ
_DELIV_VERBS = {"mang", "giao", "chuyen", "gui", "dua", "dem", "cho", "ship", "trao", "goi"}         # "chở" (câu có dấu); "đem", "đưa"
_VERB_LEAD = {"roi", "moi", "khi", "hay", "cu", "thang", "di", "se", "nho",   # "rồi mới sang X", "trước khi vào X", "cứ thẳng sang X"
              "khong", "chua", "khoi"}                                        # "chứ không sang X", "khỏi sang X" (phủ định quen: "không tới")
_NO_PREV = {"ghe", "tat", "re", "ngang", "loi", "cong", "cua", "ra", "vao", "len", "xuong", "o", "buoi", "ban", "sach"}
_NO_NEXT = TIME_HEADS | {"sang", "toi", "mai", "nay", "nam", "thang", "ngay", "hom", "sau", "truoc", "trai", "phai", "phia", "huong",
                         "man", "khoi", "ngoai", "vao", "ra", "sao", "day", "do", "dau", "nua", "lai", "giup", "gium", "nhe", "nha",
                         "di", "tay", "dong", "bac", "tren", "duoi", "giua", "trong", "thu", "chu", "cuoi", "som", "khoang", "dip",
                         "ca", "truong", "luon", "ngay", "lenh", "quyet", "ve", "xe", "nhanh", "kip"}
_CANON_PUNCT = set(".,;:!?")
# động từ giao hàng chưa từng có trong train / validation ("đem", "chở": 0 lần; validation thêm "vận chuyển" 40 lần mà train
# không có -> test có thể thêm động từ mới) -> "mang" (động từ quen). Câu không dấu: "dem" (~ "đêm") chỉ đổi khi sau nó trong
# 6 từ có giới từ nơi đến.
_CANON_VERBS = {"đem": "mang", "chở": "mang"}
_DEST_AFTER = {"toi", "den", "sang", "qua", "vao", "ve", "ra"}
# trạng từ đuôi câu không thuộc tên lạ: "tới phòng pháp chế LIỀN nhé", "NGAY LẬP TỨC", "TRONG 15 PHÚT nữa" (0 lần "liền /
# lập tức" trong dữ liệu -> bộ gán nhãn nuốt chúng vào tên). "sân cầu lông TRONG NHÀ" vẫn giữ.
TAIL_FIX = True
_TAIL_TIME = {"vong", "phut", "giay", "gio", "tieng", "ngay", "hom", "buoi", "sang", "chieu", "toi", "trua", "tuan", "thang", "luc",
              "khoang", "nua", "vai", "it", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi", "may", "ngay"}
_TAIL_NEXT = {"nhe", "nha", "di", "giup", "gium", "dum", "nhanh", "ngay", "luon", "voi", "nhen", "nghen", "a", "ha"}


def _tail_stop(words, acc, k):
    """Từ thứ k mở đầu một trạng từ đuôi (không thuộc tên địa điểm)?"""
    if not TAIL_FIX or k >= len(words):
        return False
    w, nxt = words[k], (words[k + 1] if k + 1 < len(words) else None)
    a = acc[k] if acc is not None and len(acc) == len(words) else None
    if w == "lap" and nxt == "tuc":
        return a is None or a.isascii() or a == "lập"
    if w == "lien":
        if a is not None and not a.isascii():
            return a == "liền"
        return nxt is None or nxt in _TAIL_NEXT
    if w == "trong":
        return nxt is not None and (nxt.isdigit() or nxt in _TAIL_TIME)
    return False


# ---- đợt 20c: lỗi gõ ở CỤM then chốt của câu gây nhiễu / phủ định (h27 --typo: "hỏi phải qua X", "khỏi phiả qua X", "tránh hầm
# với X", "Ln trước giao ở X rồi", "X đng cửa rồi", "người nhận hông còn ở X" -> X thành điểm ghé). Sửa từng chữ riêng lẻ không
# được vì chữ gõ sai trùng chữ có nghĩa khác ("hai", "phía", "hầm", "trực"); xét CẢ CỤM: đa số chữ khớp đúng, chữ còn lại lệch
# đúng 1 ký tự so với cụm quen (kiến thức chung về các câu gây nhiễu; không lấy từ test).
CUE_TYPO_FIX = True
_CUE_PHRASES = ["khỏi phải qua", "khỏi phải ghé", "khỏi cần ghé", "khỏi cần qua", "khỏi cần tới", "không cần ghé", "không cần qua",
                "không cần tới", "không cần đến", "tránh nhầm với", "đừng nhầm với", "đừng mang tới", "đừng mang đến", "đừng giao tới",
                "đừng ghé qua", "lần trước giao ở", "lần trước đã giao ở", "hôm qua đã giao ở", "hôm qua giao ở", "thì bỏ qua",
                "đóng cửa rồi", "đã đóng cửa", "không còn ở", "người nhận đã rời", "không phải điểm nhận", "không phải nơi nhận",
                "hôm nay không nhận hàng", "không nhận hàng", "không ghé", "không vào", "bỏ qua",
                "đừng lạc sang", "cẩn thận nhầm sang", "khỏi tới", "khỏi ghé", "thì khỏi", "hôm trước giao ở",
                "không phải nơi giao", "không phải đích đến", "đã gửi cho", "đừng để nhầm", "không dành cho", "tuyệt đối không",
                "đừng ghé", "không phải giao cho", "người nhận đã rời", "không có ai nhận", "nhớ né", "tránh xa"]
_CUE_TOK = [(p.split(), _unaccent(p).split()) for p in _CUE_PHRASES]
# chữ THẬT của các cụm gấp / dễ vỡ / thời gian: không bao giờ được "sửa" thành chữ của cụm then chốt
_CUE_PROTECT = {"voi", "gap", "nhanh", "khan", "nhe", "vo", "be", "lam", "cho", "doi", "ngay", "lien", "som", "muon", "tre", "kip",
                "lau", "sao", "gi", "nua", "dau", "het", "duoc", "can", "the"}


def _dl1(a, b):
    return a != b and dl_distance(a, b, 1) <= 1


def fix_cue_typos(text):
    """Sửa lỗi gõ trong cụm then chốt: "Khỏi hải qua X" -> "Khỏi phải qua X". Chỉ sửa khi số chữ khớp đúng >= số chữ - 1 (cụm
    >= 3 chữ) hoặc cụm 2 chữ có chữ lệch dài >= 3 ký tự; chữ lệch phải cách đúng 1 ký tự (thiếu / thừa / đảo / thay)."""
    toks = list(re.finditer(r"[^\W\d_]+|[.,;:!?]", text))
    if not toks:
        return text
    acc = not text.isascii()
    words = [m.group() if m.group() in _CANON_PUNCT else _unaccent(m.group().lower()) for m in toks]
    rep = {}
    for p_acc, p_un in _CUE_TOK:
        n = len(p_un)
        for i in range(len(words) - n + 1):
            seg = words[i:i + n]
            if any(w in _CANON_PUNCT for w in seg) or seg == p_un:
                continue
            exact = sum(a == b for a, b in zip(seg, p_un))
            bad = [k for k in range(n) if seg[k] != p_un[k]]
            if len(bad) != 1 or not _dl1(seg[bad[0]], p_un[bad[0]]):
                continue
            k = bad[0]
            if seg[k] in _CUE_PROTECT:
                continue                                  # "không cần VỘI" không phải "không cần tới" gõ sai
            if n == 2 and (len(p_un[k]) < 4 or exact < 1):
                continue
            if n >= 3 and exact < n - 1:
                continue
            rep.setdefault(i + k, p_acc[k] if acc else p_un[k])
    if "hông" in text.lower() or "hong" in words:
        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else ""
            raw = toks[i].group().lower()
            if (raw == "hông" or (not acc and w == "hong")) and nxt in ("con", "vao", "can", "phai", "ghe", "toi", "den", "qua", "nhan",
                                                                           "giao", "mang", "duoc", "co", "di", "nen"):
                rep.setdefault(i, "không" if acc else "khong")
    if not rep:
        return text
    out, last = [], 0
    for i in sorted(rep):
        m = toks[i]
        r = rep[i]
        out.append(text[last:m.start()] + (r[:1].upper() + r[1:] if m.group()[:1].isupper() else r))
        last = m.end()
    return "".join(out) + text[last:]


def canon_preps(text):
    """"Chuyển gói hàng sang quầy chè ngay." -> "Chuyển gói hàng tới quầy chè ngay." (chỉ khi có động từ giao hàng trong 7 từ
    trước, không qua dấu câu, hoặc giới từ làm động từ sau "rồi / mới / khi / cứ / thẳng"). Không đổi: "sáng / vẽ / vé" (câu
    có dấu), giờ giấc ("vào lúc", "sáng nay"), hướng ("về phía", "sang trái"), "ghé / tạt vào X" (điểm ghé), "lối ra vào"."""
    toks = list(re.finditer(r"[^\W\d_]+|[.,;:!?]", text))
    if not toks:
        return text
    acc = not text.isascii()
    raws = [m.group().lower() for m in toks]
    words = [r if r in _CANON_PUNCT else _unaccent(r) for r in raws]
    out, last = [], 0
    for k, (w, raw, m) in enumerate(zip(words, raws, toks)):        # động từ lạ -> "mang"
        rep = _CANON_VERBS.get(raw) if acc else None
        if not acc and w == "dem":
            after = []
            for x in words[k + 1:k + 7]:
                if x in _CANON_PUNCT:
                    break
                after.append(x)
            if any(x in _DEST_AFTER for x in after) and (not after or after[0] not in ("nay", "qua", "mai", "khuya", "hom")) \
                    and not (k > 0 and words[k - 1] in ("ban", "nua", "ca", "suot", "toi", "dem")):
                rep = "mang"
        if rep:
            out.append(text[last:m.start()] + (rep.capitalize() if m.group()[:1].isupper() else rep))
            last = m.end()
    if out:
        text = "".join(out) + text[last:]
        toks = list(re.finditer(r"[^\W\d_]+|[.,;:!?]", text))
        raws = [m.group().lower() for m in toks]
        words = [r if r in _CANON_PUNCT else _unaccent(r) for r in raws]
    out, last = [], 0
    for k, (w, raw, m) in enumerate(zip(words, raws, toks)):
        if w not in _CANON_PREPS or k == 0 or k + 1 >= len(words):
            continue
        if acc and raw != _CANON_PREPS[w]:
            continue                                    # "sáng", "vẽ", "vé", "rạ"...: không phải giới từ
        nxt, prv = words[k + 1], words[k - 1]
        if nxt in _CANON_PUNCT or prv in _CANON_PUNCT or nxt in _NO_NEXT or prv in _NO_PREV or prv in HEADS:
            continue                                    # "cổng vào", "phòng sáng tạo", "lối ra": chữ thuộc tên địa điểm
        if not acc and w == "ve" and not (nxt in HEADS and (prv in _DELIV_VERBS or prv in _VERB_LEAD)):
            continue                                    # không dấu: "ve" = về / vệ / vé / vẽ ("bảo ve cổng", "mang ve xe"): chỉ "mang ve <nơi>"
        if not acc and w == "sang" and nxt in ("tao", "kien", "lap", "che", "suot", "loa"):
            continue                                    # "sáng tạo / sáng kiến / sáng lập" (không dấu)
        if prv == "dung" and acc and raws[k - 1] != "đừng":
            continue                                    # "dừng / dùng / đúng" + giới từ: không đổi
        ok = prv in _VERB_LEAD or prv == "dung"         # "đừng sang X"
        for q in range(k - 1, max(-1, k - 8), -1):
            if words[q] in _CANON_PUNCT:
                break
            if words[q] in _DELIV_VERBS and not (acc and words[q] in ("cho", "dem", "dua", "goi") and raws[q] not in ("chở", "đem", "đưa", "gởi")) and not (not acc and words[q] == "goi"):
                ok = True
                break
        if not ok:
            continue
        rep = "tới" if acc else "toi"
        out.append(text[last:m.start()] + (rep.capitalize() if m.group()[:1].isupper() else rep))
        last = m.end()
    return _ref_canon("".join(out) + text[last:], acc)


def _ref_canon(text, acc):
    if REF_CANON:
        # đợt 20: cách nói tham chiếu chưa có trong dữ liệu -> dạng quen: "lân cận X" = "gần X"; "phương bắc" = "phía bắc"
        text = re.sub(r"\b(l)ân cận\b", lambda m: "gần" if m.group(1) == "l" else "Gần", text, flags=re.I)
        text = re.sub(r"\b(p)hương (bắc|nam|đông|tây)\b", lambda m: ("p" if m.group(1) == "p" else "P") + "hía " + m.group(2), text, flags=re.I)
        if not acc:
            text = re.sub(r"\blan can\b", "gan", text, flags=re.I)
            text = re.sub(r"\bphuong (bac|nam|dong|tay)\b", r"phia \1", text, flags=re.I)
    return text


CANON_GUARD = True
SCOPE_GUARD = True     # phạm vi phủ định mới không được để lại một đích độ tin < 0,1 khi cách đọc cũ có đích >= 0,9


def _goal_plaus(r):
    """Độ tin (bộ phân loại vai) của lần nhắc mang vai đích chắc nhất; 0 nếu không có."""
    return max([x[3] for x in (r.get("_roles") or []) if x[1] == 0] or [0.0])
GOAL_PLAUS_GUARD = True  # đích còn lại độ tin < 0,1 mà lần nhắc bị phủ định có độ tin >= 0,9 -> lần nhắc đó là đích (h109)
GOAL_GUARD = True      # không còn đích nào sau các luật phủ định -> lần nhắc bộ phân loại vai chắc >= 0,9 là đích
DEBUG_NEG = False      # chẩn đoán: ghi lý do phủ định từng lần nhắc vào parser._negwhy (không đổi kết quả)


def _top_alts(rx):
    alts, depth, cur, esc = [], 0, "", False
    for ch in rx:
        if esc:
            cur += ch; esc = False; continue
        if ch == "\\":
            cur += ch; esc = True; continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            alts.append(cur); cur = ""
        else:
            cur += ch
    alts.append(cur)
    return [a for a in alts if a]


def _coherence(r):
    """Độ hợp lý của một kết quả đọc: (có lần nhắc mang vai đích, số lần nhắc KHÔNG bị phủ định)."""
    roles = r.get("_roles") or []
    has_goal = (not r.get("goal_from_text")) and any(x[1] == 0 for x in roles)
    return (has_goal, sum(1 for x in roles if not x[2]))


class MissionParser2:
    # ================= huấn luyện =================
    def fit(self, missions, verbose=False):
        self.lex = Lexicon(mine_lexicon(missions), ngram_counts(missions))
        cnt = Counter(t for m in missions for s in sentences(m["text"]) for t in tokens(s))
        self.freq = dict(cnt)
        # cặp từ liền nhau (để sửa lỗi gõ ra âm tiết HỢP LỆ theo từ đứng sau: "hông phải" -> "không phải")
        self.bigram = dict(Counter((a, b) for m in missions for s in sentences(m["text"])
                                   for a, b in zip(tokens(s), tokens(s)[1:])))
        # bảng khôi phục dấu (từ không dấu -> dạng có dấu hay gặp nhất, có xét từ đứng trước), học từ văn bản có dấu
        uni, bi = defaultdict(Counter), defaultdict(Counter)
        from nlp_knowledge import PLACES as _PL
        texts = [m["text"] for m in missions] + [x for v in _PL.values() for x in v]
        for t in texts:
            ws = re.findall(r"[^\W\d_]+", t.lower())
            if all(w.isascii() for w in ws):
                continue
            prev = "<s>"
            for w in ws:
                u = _unaccent(w)
                uni[u][w] += 1
                bi[(_unaccent(prev), u)][w] += 1
                prev = w
        self.acc_uni = {u: c.most_common(1)[0][0] for u, c in uni.items()}
        # dạng có dấu QUEN của một chữ: chiếm >= 10% số lần gặp và >= 3 lần (dạng gõ sai lẻ tẻ của dữ liệu như "hông" không tính)
        self.acc_forms = {u: {w for w, n in c.items() if n >= 3 and n >= 0.1 * sum(c.values())} for u, c in uni.items()}
        self.acc_bi = {k: c.most_common(1)[0][0] for k, c in bi.items() if sum(c.values()) >= 2}
        if TYPE_FIX:
            # bảng khôi phục dấu RIÊNG cho tên lạ (chỉ dùng trong restore): cặp từ liền nhau -> dạng có dấu của từ sau (biết từ
            # trước) và của từ trước (biết từ sau). Kho tên tự soạn được ưu tiên (đúng chính tả, đúng ngữ vực tên địa điểm),
            # rồi tới dữ liệu (cặp gặp >= 2 lần để tránh lỗi gõ lẻ tẻ).
            from nlp_knowledge import PLACES_MORE as _PM, TYPE_DESC as _TD

            def pair_tables(txts, minc):
                bl, br = defaultdict(Counter), defaultdict(Counter)
                for t in txts:
                    ws = re.findall(r"[^\W\d_]+", t.lower())
                    if all(w.isascii() for w in ws):
                        continue
                    for a, b in zip(ws, ws[1:]):
                        k = (_unaccent(a), _unaccent(b))
                        bl[k][b] += 1
                        br[k][a] += 1
                pick = lambda d: {k: c.most_common(1)[0][0] for k, c in d.items() if sum(c.values()) >= minc}
                return pick(bl), pick(br)
            kb_texts = [x for d in (_PL, _PM) for v in d.values() for x in v] + [x for v in _TD.values() for x in v]
            kl, kr = pair_tables(kb_texts, 1)
            dl_, dr_ = pair_tables([m["text"] for m in missions], 2)
            self.racc_l, self.racc_r = {**dl_, **kl}, {**dr_, **kr}
            # tên trong kho tự soạn (không dấu) -> loại, chỉ giữ tên thuộc đúng MỘT loại
            kbt = defaultdict(set)
            for d in (_PL, _PM):
                for t, v in d.items():
                    for x in v:
                        kbt[" ".join(tokens(_unaccent(x.lower())))].add(t)
            self.kb_type = {a: next(iter(ts)) for a, ts in kbt.items() if a and len(ts) == 1}
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
        rng = np.random.default_rng(FIT_SEED)
        Xl, Xr, Xs, yrole, Rr, ykind = [], [], [], [], [], []
        table = defaultdict(lambda: [0, 0, 0])
        G, yG = [], []
        alias_acc = defaultdict(set)      # tên gọi (không dấu) -> các dạng có dấu gặp trong dữ liệu
        phrase_acc = defaultdict(set)     # câu con (không dấu) -> các dạng có dấu
        main_acc = set()                  # các câu có nhắc địa điểm (dạng có dấu)
        ctx_rows = []                     # (ngữ cảnh quanh một lần nhắc đích/điểm ghé, loại)
        role_rows = []                    # (câu con có dấu, tên đang xét -> "nơi A", tên khác -> "nơi B"; vai)
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
                role_rows.append((self._role_text(me), ROLES.index(role)))
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
                if PHRASE_NEEDS_CUE and ((lab == 1 and not re.search(URGENT_POS, s)) or (lab == 2 and not re.search(FRAGILE_POS, s))):
                    lab = 0     # tương quan tình cờ ("hàng: chìa khóa xe" gặp 3 lần, cả 3 đều là đơn gấp): không có từ ngữ gấp / dễ vỡ
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
            if USE_PHRASES_MORE:
                from nlp_knowledge import PHRASES_MORE
                PHRASES = {c: list(PHRASES[c]) + [x for x in PHRASES_MORE[c] if x not in PHRASES[c]] for c in PHRASES}
            if USE_PLACES_MORE:
                from nlp_knowledge import PLACES_MORE
                PLACES = {t: list(PLACES[t]) + [x for x in PLACES_MORE.get(t, []) if x not in PLACES[t]] for t in PLACES}
            if not getattr(self, "use_knowledge", True):      # thí nghiệm: không dùng ví dụ viết tay
                PHRASES, PLACES = {}, {t: [] for t in PLACE_TYPES}
            for t, v in CANONICAL_ACC.items():
                for s in {v, _unaccent(v)} | {x for a in PLACES[t] for x in (a, _unaccent(a))}:
                    tx.append(s); ty.append(PLACE_TYPES.index(t))
            tv = E.encode(tx)
            self.e5_type = LogisticRegression(C=50, max_iter=3000).fit(tv, ty)
            # láng giềng gần nhất (tên đã biết giống nhất của từng loại) + câu mô tả từng loại: scratch/h29_typing_methods.py
            self.knn_vec, self.knn_lab = tv, np.array(ty)
            from nlp_knowledge import TYPE_DESC
            dd = [(d, PLACE_TYPES.index(t)) for t, lst in TYPE_DESC.items() for d in lst]
            if DESC_MORE:
                from nlp_knowledge import TYPE_DESC2
                dd += [(d, PLACE_TYPES.index(t)) for t, lst in TYPE_DESC2.items() for d in lst]
            self.desc_vec, self.desc_lab = E.encode([d for d, _ in dd], prefix="passage: "), np.array([k for _, k in dd])
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
            mrng = np.random.default_rng(FIT_SEED + 1)
            pool = sorted(main_acc)
            for v in [pool[i] for i in mrng.choice(len(pool), min(400, len(pool)), replace=False)]:
                px.append(v); py.append(0)
            self.e5_phrase = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(E.encode(px), py)
            # loại địa điểm đoán từ NGỮ CẢNH quanh nó (món hàng, người nhận...), không nhìn tên: chứng cứ bổ sung cho tên lạ
            self.e5_ctx = LogisticRegression(C=20, max_iter=3000).fit(E.encode([c for c, _ in ctx_rows]), [t for _, t in ctx_rows])
            # (đợt 18) VAI bằng NGHĨA cả câu con: test có nhiều khung câu lạ hơn hẳn validation (bộ phân loại n-gram không chắc
            # < 0,6 ở 15% cảnh test so với 2% validation) -> mô hình nghĩa tổng quát sang cách nói mới tốt hơn (như cờ dễ vỡ)
            rr = sorted(set(role_rows))
            self.e5_role = LogisticRegression(C=ROLE_E5_C, max_iter=3000, class_weight="balanced").fit(
                E.encode([x for x, _ in rr], prefix="query: "), [y for _, y in rr])
            # cùng ngữ cảnh đó nhưng khớp CHÍNH XÁC từ ngữ (món hàng, người nhận quen thuộc: "quả bóng", "hộp phấn", "hồ sơ"...)
            self.v_cx = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"\S+", min_df=2)
            self.cx_clf = LogisticRegression(C=10, max_iter=3000).fit(
                self.v_cx.fit_transform([_unaccent(c) for c, _ in ctx_rows]), [t for _, t in ctx_rows])
            E.save()
        if verbose:
            print("lexicon", len(self.lex.alias2type), "| roles", Counter(ROLES[y] for y in yrole), "| kinds", Counter(KINDS[y] for y in ykind))
        return self

    # ================= tiện ích =================
    def _sents(self, text):
        # (đã thử tự sửa lỗi gõ cho mọi từ lạ: làm hỏng các từ MỚI hợp lệ như "xong" -> "cong", nên bỏ)
        if SPATIAL_FIX2:
            text = re.sub(r"\bđáy\b", "dưới", text, flags=re.I)  # "ở đáy bản đồ" = phía dưới (chỉ dạng có dấu: "day" ~ "đây / dãy")
        out = sentences(text)
        if VIA_FIX and VIA_SEMI:
            out = [s.replace(";", ",") for s in out]          # dấu ";" là ranh giới vế (tag_tokens chỉ giữ "," và ":")
        if SPATIAL_FIX2:
            # "bên TAY phải" ("tay" không dấu ~ hướng "tây" -> hai hướng mâu thuẫn), "MÉ trái" (mé = phía); 0 lần trong train + val.
            # Chỉ dạng có "bên": "phía TÂY PHẢI được ghé trước" (tây + phải = cần) KHÔNG được đổi (h27, bản đầu đổi nhầm thành đông)
            out = [re.sub(r"\bme (tren|duoi|trai|phai|bac|nam|dong|tay)\b", r"phia \1", re.sub(r"\bben tay (trai|phai)\b", r"ben \1", s))
                   for s in out]
            # "phía tay phải / trái" (không có "bên"): "tay trái" luôn là tay; "tay phải" là tay trừ khi "phải" = CẦN
            # ("phía tây phải được ghé", "tây phải đi qua")
            out = [re.sub(r"\b(phia|o|nam|huong) tay trai\b", r"\1 ben trai",
                          re.sub(r"\b(phia|o|nam|huong) tay phai\b(?! (duoc|ghe|di|toi|den|qua|giao|mang|lay|nhan|co|la|vao|dung|re|tat|"
                                 r"chuyen|dua|xong|nho|can|ve))", r"\1 ben phai", s)) for s in out]
        return out

    @staticmethod
    def _tt(sent):
        tt = tag_tokens(sent)
        idx = [i for i, t in enumerate(tt) if t not in (",", ":")]
        return tt, [tt[i] for i in idx], idx

    def _main_text(self, text):
        keep = [s for s in self._sents(text) if not re.search(_neg(), _strip_flags(s))]
        return " . ".join(keep)

    def restore(self, text):
        """Đoán dấu cho cụm không dấu ("phong mot cua" -> "phòng một cửa") bằng bảng học từ dữ liệu; từ lạ giữ nguyên."""
        if not hasattr(self, "acc_uni"):
            return text
        if TYPE_FIX and hasattr(self, "racc_l"):
            ws = text.split()
            out = []
            for k, u in enumerate(ws):
                w = (self.racc_l.get((ws[k - 1], u)) if k > 0 else None) \
                    or (self.racc_r.get((u, ws[k + 1])) if k + 1 < len(ws) else None) or self.acc_uni.get(u, u)
                out.append(w)
            return " ".join(out)
        out, prev = [], "<s>"
        for u in text.split():
            w = self.acc_bi.get((prev, u)) or self.acc_uni.get(u, u)
            out.append(w); prev = u
        return " ".join(out)

    def span_probs(self, text, acc=None):
        """Phân bố loại địa điểm của một cụm tên gọi (lạ). Ba nguồn: n-gram ký tự so với tên đã biết,
        từ khóa viết tay, và vector nghĩa e5 (tin hơn khi cụm có dấu)."""
        kw_text = text
        if TYPE_FIX and "sinh vien" in text:
            # "sinh viên" là từ CHUNG trên campus (mọi nơi đều phục vụ sinh viên) nhưng tên ký túc xá của train hay có nó
            # ("sinh viên nội trú", "phòng ở sinh viên") -> các mô hình học kéo mọi tên lạ có "sinh viên" về ký túc xá
            # ("nơi sinh viên học", "phòng tiếp sinh viên"). Bỏ cụm này khỏi đầu vào của mô hình học; từ khóa và tra kho
            # vẫn xét cả tên ("làng sinh viên", "nhà sinh viên" vẫn là ký túc xá).
            ws = text.split()
            aw = acc.split() if acc and len(acc.split()) == len(ws) else None
            keep = [True] * len(ws)
            for k in range(len(ws) - 1):
                if ws[k] == "sinh" and ws[k + 1] == "vien":
                    keep[k] = keep[k + 1] = False
                    if k > 0 and ws[k - 1] == "cua":
                        keep[k - 1] = False
            if sum(keep) >= 2:
                text = " ".join(w for w, kp in zip(ws, keep) if kp)
                acc = " ".join(w for w, kp in zip(aw, keep) if kp) if aw else None
        p = self.type_clf.predict_proba(self.v_t.transform([text]))[0]
        full = np.full(10, 1e-3)
        full[self.type_clf.classes_] = p
        score = CHAR_W * np.log(full + 0.03) + (0.0 if getattr(self, "ablate_rules", False) else 2.0) * keyword_scores(kw_text)
        if getattr(self, "use_e5", False):
            s = acc if acc else text
            forms = None
            if RESTORE_ACC and s.isascii():
                s = self.restore(text)
                if E5_BOTH_FORMS and s != text:
                    # (đợt 19) tên lạ KHÔNG DẤU: dạng khôi phục dấu có thể sai nghĩa -> lấy trung bình chứng cứ e5 của cả dạng
                    # khôi phục lẫn dạng không dấu gốc (mô hình e5 đã học cả dạng không dấu của mọi tên)
                    forms = [s, text]
            pe = np.full(10, 1e-3)
            if forms is None:
                q = embedder().encode([s])
                pe[self.e5_type.classes_] = self.e5_type.predict_proba(q)[0]
                score += (1.0 if s != text else 0.5) * np.log(pe + 0.02)
                if KNN_W > 0 and hasattr(self, "knn_vec"):
                    score += KNN_W * np.log(_soft(_per_type_max(self.knn_vec @ q[0], self.knn_lab), KNN_T) + 0.02)
                if DESC_W > 0 and hasattr(self, "desc_vec"):
                    score += DESC_W * np.log(_soft(_per_type_max(self.desc_vec @ q[0], self.desc_lab), DESC_T) + 0.02)
            else:
                Q = embedder().encode(forms)
                for q, wq in zip(Q, (1.0 - E5_RAW_W, E5_RAW_W)):
                    pe = np.full(10, 1e-3)
                    pe[self.e5_type.classes_] = self.e5_type.predict_proba(q[None])[0]
                    score += wq * np.log(pe + 0.02)
                    if KNN_W > 0 and hasattr(self, "knn_vec"):
                        score += wq * KNN_W * np.log(_soft(_per_type_max(self.knn_vec @ q, self.knn_lab), KNN_T) + 0.02)
                    if DESC_W > 0 and hasattr(self, "desc_vec"):
                        score += wq * DESC_W * np.log(_soft(_per_type_max(self.desc_vec @ q, self.desc_lab), DESC_T) + 0.02)
        if TYPE_FIX and getattr(self, "kb_type", None):
            t = self.kb_type.get(kw_text)
            if t is not None:
                score[PLACE_TYPES.index(t)] += KB_BOOST       # tên có sẵn trong kho tự soạn ("nơi đóng học phí")
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
        person_spans = set()
        self._segs = []
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
                # chữ khung câu bị gõ sai ("gaio", "cnầ", "ần" trong "nơi cần giao"): không được nằm ở mép một tên lạ
                fw = self._fix_seq(words, acc_w) if NEW_FRAME_FIX else words
                frame_typo = lambda k: fw[k] != words[k] and (fw[k] in EDGE_STRIP or (fw[k] in KEY_CUE_SET and fw[k] not in HEADS))
                # khớp mờ (sửa lỗi gõ) chỉ được nhận khi nó không phải một mẩu của tên địa điểm dài hơn
                # (vd "an tap" trong "phòng ăn tập thể" không phải "sân tập" gõ sai)
                nb = lambda k: pw[k] > 0.5 and not (FUZZY_REF_OK and words[k] in REF_WORDS | ORI_WORDS | DIRS)
                ms = [m for m in ms if " ".join(words[m[0]:m[1]]) == m[3]
                      or not ((m[0] > 0 and nb(m[0] - 1)) or (m[1] < len(words) and nb(m[1])))]
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
                            and ((words[j] not in self.right_bound and maybe(j)) or self._o_name(words, idx, j, acc_w)
                                 or (SUBJ_TRIM and words[j] in ("truoc", "sau") and words[j - 1] in ("phia", "mat"))):
                        if self._o_name(words, idx, j, acc_w) and j + 1 < len(words) and words[j + 1] == "to" \
                                and j + 1 not in used and idx[j + 1] == idx[j] + 1:
                            j += 1                              # "ô tô": lấy cả hai chữ
                        j += 1; steps += 1
                    steps = 0
                    while i > 0 and steps < 3 and (i - 1) not in used and idx[i] == idx[i - 1] + 1 \
                            and words[i - 1] not in self.left_bound and maybe(i - 1):
                        i -= 1; steps += 1
                    if TAIL_FIX:                                # "... phòng pháp chế LIỀN nhé", "... TRONG 15 PHÚT nữa"
                        cut = next((k for k in range(i + 1, j) if _tail_stop(words, acc_w, k)), None)
                        if cut is not None:
                            j = cut
                    # gọt các từ khung ở hai đầu và các cụm chỉ hướng ở cuối ("... mạn trên", "... nằm phía đông", "... bên phải bản đồ")
                    changed = True
                    s_acc = EDGE_ACC_FIX and not "".join(acc_w).isascii() and len(acc_w) == len(words)
                    keep_edge = lambda k: s_acc and words[k] in EDGE_ACC and acc_w[k] not in EDGE_ACC[words[k]]
                    while changed and i < j:
                        if keep_edge(j - 1):
                            break
                        if SUBJ_TRIM and j - i >= 3 and words[j - 1] in ("truoc", "sau") and words[j - 2] in ("phia", "mat"):
                            break                               # "cổng phía trước", "cổng mặt sau": thuộc tên (không phải hướng bản đồ)
                        changed = False
                        if DETECT_RULES and words[j - 1] == "xa" and j - i >= 2 and (
                                acc_w[j - 1] == "xá" or (acc_w[j - 1] == "xa" and "".join(acc_w).isascii() and not (
                                    j < len(words) and (words[j] in HEADS or words[j] == "hon" or any(m[0] == j for m in ms))))):
                            break                               # "trạm xá", "bệnh xá": chữ cuối là "xá", không phải "xa"
                        if words[j - 1] == "o" and j - i >= 2 and self._o_name(words, idx, j - 1, acc_w):
                            break                               # "dãy phòng ở", "khu nhà ở": "ở" thuộc tên
                        if words[j - 1] in EDGE_STRIP or (NEW_FRAME_FIX and frame_typo(j - 1)):
                            j -= 1; changed = True              # cả chữ khung gõ sai ("Nơi nhna: X", "nơi cần gaio")
                        elif TYPE_FIX and j - i >= 2 and words[j - 1] in NAME_TAIL and acc_w[j - 1] != "nhà" \
                                and words[j - 2] not in NHA_KEEP and (j == len(words) or idx[j] != idx[j - 1] + 1):
                            j -= 1; changed = True              # trợ từ cuối câu: "... phòng quản lý sinh viên nha."
                        elif j - i >= 2 and words[j - 2:j] == ["ban", "do"]:
                            j -= 2; changed = True
                        elif j - i >= 2 and words[j - 1] in DIRS and words[j - 2] in ORI_WORDS:
                            j -= 2; changed = True
                        elif j - i >= 2 and words[j - 1] in DIRS and words[j - 2] in ("o", "nam"):
                            j -= 1; changed = True
                        elif words[j - 1] == "nam" and j - i >= 2 and j < len(words) and words[j] in ORI_WORDS | {"gan", "xa", "sat", "o", "canh"}:
                            j -= 1; changed = True          # "nằm" (động từ) đứng trước cụm chỉ hướng
                    while i < j and (words[i] in EDGE_STRIP or (NEW_FRAME_FIX and frame_typo(i))) and not keep_edge(i):
                        if CHO_HEAD and words[i] == "cho" and j - i >= 2 and (acc_w[i] == "chỗ" or (
                                acc_w[i] == "cho" and "".join(acc_w).isascii() and i > 0 and words[i - 1] in LOC_PREPS and pw[i] >= 0.3)):
                            break                               # "chỗ trả sách", "chỗ để xe": "chỗ" là từ mở đầu tên địa điểm
                        i += 1
                    if SUBJ_TRIM and i < j:
                        # "ban quản lý / cán bộ / cô <nơi>": phần người không thuộc tên nơi chốn
                        lead = next((len(pl) for pl in PERSON_LEADS if tuple(words[i:i + len(pl)]) == pl), 0)
                        if lead and i + lead < j - 1 and words[i + lead] in SUBJ_PLACE_HEADS:
                            i += lead
                            # phần nới sang phải bị giới hạn 4 bước tính cả phần người: nới tiếp qua các từ bộ gán nhãn chắc chắn
                            while j < len(words) and j - i < 6 and j not in used and idx[j] == idx[j - 1] + 1 and pw[j] >= 0.5:
                                j += 1
                    if i >= j or _all_ref(words[i:j]):
                        continue
                    # "tòa nhà ở của sinh viên", "phòng ở sinh viên": cụm tên kéo dài qua "(ở) của" hoặc (từ mở đầu + "ở") + tối đa 3 từ
                    if MERGE_RULES and j + 1 < len(words):
                        q = None
                        # (câu có dấu: chỉ "của" mới nối tên; "ở CỬA vào trường" là nơi khác -- đợt 16)
                        of = lambda k: not (HEAD_ACC_FIX and not "".join(acc_w).isascii() and acc_w[k] not in ("của", "cua"))
                        if words[j] == "cua" and of(j):
                            q = j + 1
                        elif words[j:j + 2] == ["o", "cua"] and of(j + 1):
                            q = j + 2
                        elif DETECT_RULES and words[j] == "o" and j - i <= 2 and all(w in HEADS for w in words[i:j])                                 and not any(m[0] == j + 1 for m in ms) and words[j + 1] not in HEADS | ORI_WORDS | REF_WORDS:
                            q = j + 1
                        if q is not None:
                            e = q
                            while e < len(words) and e - q < 3 and e not in used and idx[e] == idx[e - 1] + 1                                     and words[e] not in self.right_bound and words[e] not in EDGE_STRIP:
                                e += 1
                            if e > q and not any(e > m[0] and q < m[1] for m in ms):
                                j = e
                    if any(i < m[1] and j > m[0] for m in ms):
                        continue
                    if any(dl_distance(w, "robot", 1) <= 1 for w in words[i:j]):      # lời chào gõ sai ("orbot oi")
                        continue
                    span_text = " ".join(words[i:j])
                    if DIS_UNAVAIL and re.fullmatch(UNAVAIL_SPAN, span_text):
                        continue                              # "X đã KHÓA CỬA", "X ngừng HOẠT ĐỘNG": không phải tên địa điểm
                    kw = keyword_scores(span_text).max()
                    # cụm lạ chỉ được coi là địa điểm khi có dấu hiệu: mở đầu bằng "phòng / khu / nơi / nhà..." hoặc chứa từ khóa
                    # (nếu không, các cụm gấp/dễ vỡ lạ, món hàng lạ, động từ lạ sẽ bị nhận nhầm là địa điểm)
                    # HEAD_ACC (đợt 16): câu CÓ DẤU thì chữ đầu phải đúng là từ chỉ nơi chốn ("ký" túc xá, không phải "kỵ rung lắc")
                    head = words[i] in HEADS and not (HEAD_ACC_FIX and words[i] in HEAD_ACC and not acc_w[i].isascii()
                                                      and not "".join(acc_w).isascii() and acc_w[i] not in HEAD_ACC[words[i]])
                    if head and HEAD_ACC_FIX and kw == 0 and words[i] in HEAD_ACC and "".join(acc_w).isascii() and (
                            re.search(FRAGILE_POS, span_text) or re.search(URGENT_POS, span_text)):
                        head = False                          # câu không dấu: "ky rung lac" (kỵ rung lắc) là lời dặn, không phải tên
                    if SUBJ_TRIM and kw < 3 and (words[i] not in PLACE_HEADS or not head) and (
                            re.search(FRAGILE_POS, span_text) or re.search(URGENT_POS, span_text)):
                        continue                              # "đĩa sứ của khoa": cụm nói về HÀNG (dễ vỡ / gấp), không phải tên nơi
                    if kw == 0 and not head:
                        strong =DETECT_RULES and i > 0 and words[i - 1] in LOC_PREPS and idx[i] == idx[i - 1] + 1                             and min(pw[i:j]) >= 0.9 and not any(w in ITEM_HEADS for w in words[i:j])
                        if not strong:
                            continue
                    if words[i] in ITEM_HEADS and words[i] not in PLACE_HEADS and kw < 3:
                        continue                              # "hộp thuốc", "khay cơm"...: món hàng
                    if j - i == 1 and kw == 0 and self.span_probs(span_text, " ".join(acc_w[i:j])).max() < 0.4:
                        continue
                    if NEW_FRAME_FIX and j - i == 1 and words[i] in LONE_BAD:
                        continue                              # "Nơi nhận: X": "nơi" / "khu" / "phòng" đứng một mình không phải tên
                    if NEW_FRAME_FIX and "oi" in words[i:j]:
                        continue                              # lời gọi "Bạn ơi", "Anh ơi" không phải tên địa điểm
                    if NEW_FRAME_FIX and words[i] in TIME_HEADS:
                        continue                              # "giờ học", "buổi học", "giờ ăn trưa": thời gian, không phải địa điểm
                    ms.append((i, j, None, " ".join(words[i:j])))
                    used.update(range(i, j))
                ms.sort()
                if FRAME_STRONG:
                    ms = self._fix_spans(words, idx, acc_w, ms)
                # "tòa / khu / dãy" + tên đã biết ("tòa giảng đường"): một địa điểm, mang loại của tên đã biết
                if MERGE_RULES:
                    merged = []
                    for m in ms:
                        if merged and merged[-1][2] is None and m[2] is not None and merged[-1][1] == m[0] \
                                and m[0] - merged[-1][0] <= 2 and all(w in HEADS for w in words[merged[-1][0]:m[0]]):
                            merged[-1] = (merged[-1][0], m[1], m[2], m[3])
                        else:
                            merged.append(m)
                    ms = merged
                if FRAME_RULE or GOAL_FRAME_RULE or ANCHOR_RULE:
                    new = self._frame_spans(words, ms, pw, idx, acc_w)
                    if new:
                        ms = [m for m in ms if not (m[2] is None and any(a <= m[0] and m[1] <= b for a, b in new))]
                        ms += [(a, b, None, " ".join(words[a:b])) for a, b in new]
                        ms.sort()
                if PERSON_RULE:
                    # (FRAME_STRONG: tìm khung trên chuỗi đã sửa lỗi gõ chữ khung: "X cần nhn hộp giấy" = "cần nhận")
                    ps = self._person_subject(tt, fw if PERSON_TYPO_FIX and len(fw) == len(words) else words, idx, ms)
                    if ps is not None:
                        ms.append((ps[0], ps[1], None, " ".join(words[ps[0]:ps[1]])))
                        ms.sort()
                        person_spans.add((si, ps[0]))
                if NEW_FRAME_FIX and len(ms) >= 2:
                    # đồng vị ngữ "Giao tới X, nơi mọi người đang chờ": tên lạ mở đầu bằng "nơi / chỗ" ngay sau dấu phẩy,
                    # sát sau một địa điểm khác -> chỉ mô tả lại địa điểm đó, không phải địa điểm mới (đợt 9, h27e_misc.py)
                    # ("Bỏ qua X, nơi khám bệnh MỚI LÀ điểm nhận": cụm là chủ ngữ của vế mới -> giữ)
                    ms = [m for q, m in enumerate(ms) if not (
                        q > 0 and m[2] is None and words[m[0]] in ("noi", "cho") and ms[q - 1][1] == m[0]
                        and m[0] > 0 and idx[m[0]] != idx[m[0] - 1] + 1 and "la" not in words[m[0]:m[1] + 3])]
            if ITEM_FIX and ms:
                if ITEM_LINE.match(sent):
                    ms = []                                     # dòng "Hàng: X" chỉ nêu MÓN HÀNG ("Hàng: sách thư viện")
                else:
                    # tên CHỒNG LÊN cụm món hàng: "mang sách thư viện tới bãi xe", "hàng cần giao là thuốc y tế", cả khớp mờ
                    # vắt qua mép cụm ("là hóa chất" ~ "lab hóa"). Địa điểm thật đứng SAU giới từ nên không chồng lên cụm.
                    slots = self._item_slots(words, idx, ms)
                    ms = [m for m in ms if not any(m[0] < b and m[1] > a for a, b in slots)]
            if not ms:
                plain.append(sent)
                plain_acc.append(" ".join(acc_w))
            punct = [k for k, t in enumerate(tt) if t in (",", ":")]
            if ms and not known_only and punct:
                # các vế (giữa hai dấu phẩy) không chứa địa điểm nào: có thể là ý gấp / dễ vỡ gắn vào câu giao hàng
                cuts = [-1] + punct + [len(tt)]
                for lo, hi in zip(cuts, cuts[1:]):
                    wi = [k for k in range(len(words)) if lo < idx[k] < hi]
                    if len(wi) >= 2 and not any(m[0] <= k < m[1] for m in ms for k in wi):
                        self._segs.append((" ".join(words[k] for k in wi), " ".join(acc_w[k] for k in wi)))
            # "dung" không dấu = đừng / dừng / dùng / đúng. Nếu câu CÓ dấu thì chỉ "đừng" mới là phủ định: các chữ khác đổi
            # thành "dungx" trong bản dùng để xét luật ("phải dừng lại ở X" là điểm ghé, không phải "đừng lại X")
            tt2, words2 = tt, words
            if DUNG_CHECK and ms and any(not w.isascii() for w in acc_w):
                tt2, words2 = list(tt), list(words)
                for k, w_ in enumerate(words):
                    if w_ == "dung" and acc_w[k] != "đừng":
                        words2[k] = "dungx"
                        tt2[idx[k]] = "dungx"
            acc_tt = list(tt2)                                  # dạng có dấu cùng vị trí với tt2 (dấu câu giữ nguyên)
            for wi, ti in enumerate(idx):
                if wi < len(acc_w):
                    acc_tt[ti] = acc_w[wi]
            for (i, j, t, a) in ms:
                # đoạn giữa hai dấu phẩy chứa lần nhắc này (để xét phủ định trong phạm vi hẹp)
                lo = max([k for k in punct if k < idx[i]], default=-1)
                hi = min([k for k in punct if k > idx[j - 1]], default=len(tt))
                out.append({"si": si, "toks": words, "acc": acc_w, "i": i, "j": j, "type": t, "text": a, "ms": ms,
                            "seg": " ".join(tt2[lo + 1:hi]), "sent": " ".join(words2), "person": (si, i) in person_spans,
                            "sentp": " ".join(tt2),                 # giữ dấu phẩy: "X đã, rồi mới đi tiếp" không thành "đã rồi"
                            "seg_acc": acc_tt[lo + 1:hi], "sentp_acc": acc_tt, "idx": idx})
        return out, plain, plain_acc

    @staticmethod
    def _fix_spans(words, idx, acc, ms):
        """FRAME_STRONG (đợt 16), sửa ranh giới các lần nhắc:
          (1) cụm lạ nuốt cả quan hệ + mốc ("nơi gặp khách SÁT nơi nhận thuốc"): tách thành hai cụm ở từ quan hệ;
          (2) tên quen + đuôi lạ liền sau ("phòng học lớn | tầng hai", "phòng thực hành | hóa ở xa Y"): một tên, giữ loại quen."""
        n = len(words)
        brk = lambda k: idx is not None and 0 < k < n and idx[k] != idx[k - 1] + 1
        has_acc = acc is not None and len(acc) == n and not "".join(acc).isascii()
        REL = ("sat", "gan", "canh", "ke", "xa", "cach")
        out = []
        for m in ms:
            i, j, t, al = m
            cut = None
            if t is None:
                for k in range(i + 1, j - 1):
                    w = words[k]
                    if w not in REL or brk(k) or (has_acc and not acc[k].isascii() and acc[k] not in CUT_ACC.get(w, (acc[k],))):
                        continue
                    if w == "xa" and (not has_acc or acc[k] != "xa") and words[k - 1] in ("tram", "benh", "tuc", "y"):
                        continue                                  # "trạm xá", "bệnh xá", "ký túc xá"
                    r = k + 2 if w == "cach" and k + 1 < j and words[k + 1] == "xa" else k + 1
                    if r < j and (words[r] in HEADS or words[r] in ("noi", "cho")) and k - i >= 1:
                        cut = (k, r)
                        break
            if cut:
                out += [(i, cut[0], None, " ".join(words[i:cut[0]])), (cut[1], j, None, " ".join(words[cut[1]:j]))]
            else:
                out.append(m)
        ms, out = out, []
        starts = {m[0] for m in ms}
        for q, m in enumerate(ms):
            if out and out[-1][2] is not None and m[2] is None and out[-1][1] == m[0] and not brk(m[0]) and m[1] - m[0] <= 3 \
                    and not any(w in REF_WORDS or w in ORI_WORDS or w in DIRS for w in words[m[0]:m[1]]) \
                    and not (NEG_SPAN_STOP and _neg_word(words, m[0])):   # "giảng đường | chẳng xa Y"
                out[-1] = (out[-1][0], m[1], out[-1][2], out[-1][3])          # tên quen + cụm lạ liền sau
                continue
            if m[2] is not None:
                e = m[1]
                while e < n and e - m[1] < 2 and not brk(e) and e not in starts and words[e] not in EDGE_STRIP \
                        and words[e] not in FRAME_CUT and words[e] not in GOAL_FRAME_END and words[e] not in DIRS \
                        and not _tail_stop(words, acc, e) and not (NEG_SPAN_STOP and _neg_word(words, e)):
                    e += 1
                if e > m[1] and _ref_after(words, e):
                    m = (m[0], e, m[2], m[3])                                   # "phòng thực hành hóa | ở xa Y"
            out.append(m)
        if NAME_QUAL_FIX:
            # (3) đợt 18: phần bổ nghĩa ĐUÔI TÊN ("tầng hai", "khu A", "cơ sở hai", "dành cho cán bộ", "tòa nhà chính", "của trường",
            # "mới xây"...) thuộc tên đứng trước: nuốt vào (cả các cụm lạ nằm trọn trong nó). Tên trên test dài hơn validation
            # (5-7 chữ: 16,6% so với 7,5%); trước đây phần đuôi tách ra thành "địa điểm" riêng -> điểm ghé / mốc giả.
            ms, out = out, []
            for m in ms:
                if out and m[0] < out[-1][1]:
                    continue                                                    # đã bị nuốt vào tên trước
                e = m[1]
                for _ in range(2):
                    if e >= n or brk(e):
                        break
                    mm = re.match(NAME_QUAL, " ".join(words[e:e + 7]))
                    if not mm:
                        break
                    L = len(mm.group(0).split())
                    if any(brk(x) for x in range(e + 1, e + L))                             or any(x[0] < e + L and x[1] > e + L for x in ms if x[0] >= e):     # cắt ngang một tên khác
                        break
                    e += L
                m2 = (m[0], e, m[2], m[3] if m[2] is not None else " ".join(words[m[0]:e])) if e > m[1] else m
                out.append(m2)
        return out

    @staticmethod
    def _o_name(words, idx, e, acc=None):
        """Chữ "o" ở vị trí e THUỘC tên địa điểm (TYPE_FIX): "ô tô" ("nơi đỗ ô tô"), hoặc "ở" đứng cuối tên ngay sau
        phòng / nhà / khu / chỗ / dãy... ("dãy phòng ở", "khu nhà ở") -- khi sau nó là hết câu, dấu câu hay trợ từ."""
        if not TYPE_FIX or words[e] != "o":
            return False
        n = len(words)
        a = acc[e] if acc is not None and e < len(acc) else "o"
        if a not in ("o", "ô", "ở"):
            return False
        if e + 1 < n and words[e + 1] == "to" and a != "ở":
            return True
        brk = lambda k: idx is not None and 0 < k < n and idx[k] != idx[k - 1] + 1
        return e > 0 and words[e - 1] in O_IN_NAME and a != "ô" and (
            e + 1 == n or brk(e + 1) or words[e + 1] in GOAL_FRAME_END | NAME_TAIL
            or (FRAME_STRONG and tuple(words[e + 1:e + 3]) in O_QUAL)
            or (SUBJ_TRIM and tuple(words[e + 1:e + 3]) in (("can", "nhan"), ("dang", "cho"), ("dang", "can")))
            or (SUBJ_TRIM and e + 2 < n and words[e + 1] == "nam"
                and words[e + 2] in ORI_WORDS | {"gan", "xa", "sat", "canh", "o", "ke", "cach"})) \
            or (SUBJ_TRIM and a != "ô" and e >= 3 and words[e - 2:e] == ["sinh", "vien"] and words[e - 3] in O_IN_NAME
                and (e + 1 == n or brk(e + 1)))       # "dãy phòng sinh viên ở." (chỗ sinh viên ở)

    @staticmethod
    def _frame_spans(words, ms, pw, idx=None, acc=None):
        """Dò tên địa điểm lạ theo khung câu (bộ gán nhãn hay bỏ sót tên mới):
          - "ghé / tạt qua <X> ... lấy / nhận / trước": X là điểm ghé;
          - "(mang / giao / chuyển...) ... tới / đến / qua <X>", "đích đến là <X>", "ở <X>": X là địa điểm;
          - "<địa điểm> (nằm) gần / xa / sát / cạnh <Y> (hơn)": Y là mốc.
        X / Y phải kết thúc GỌN (ở từ kết thúc, dấu câu, cuối câu, từ chỉ vị trí hoặc tên đã biết) trong tối đa 5 từ.
        Cụm lạ ngắn hơn nằm trọn trong X / Y (vd "tòa" trong "tòa nhà ở của sinh viên") sẽ được thay."""
        n = len(words)
        out = []
        known = [m for m in ms if m[2] is not None]
        brk = lambda e: idx is not None and 0 < e < n and idx[e] != idx[e - 1] + 1     # có dấu câu giữa e-1 và e
        # REF_SLOT: câu CÓ DẤU cho biết chữ không dấu là từ khung hay chữ của tên ("kệ" / "kế", "tài" / "tại", "cửa" / "của")
        has_acc = REF_SLOT and acc is not None and len(acc) == n and not "".join(acc).isascii()
        other = lambda e: has_acc and words[e] in CUT_ACC and not acc[e].isascii() and acc[e] not in CUT_ACC[words[e]]
        cut_word = lambda e: words[e] in FRAME_CUT and not other(e) \
            and not (SUBJ_TRIM and words[e] == "tai" and e + 1 < n and words[e + 1] in TAI_NOUN)   # "góc tài liệu" (không dấu)
        cua = lambda e: e < n and words[e] in ("cua", "sinh") and not (has_acc and words[e] == "cua" and acc[e] not in ("của", "cua")) \
            and not (SUBJ_TRIM and words[e] == "cua" and e + 1 < n and words[e + 1] in CUA_DOOR)   # "ở cửa chính trường"
        is_cut = lambda e: cut_word(e) and not (words[e] == "o" and cua(e + 1)) \
            and not MissionParser2._o_name(words, idx, e)             and not (SUBJ_TRIM and words[e] in ("phia", "mat") and e + 1 < n and words[e + 1] in ("truoc", "sau")) \
            and not (SUBJ_TRIM and words[e] == "xa" and e > 0 and words[e - 1] in ("tram", "benh")
                     and (not has_acc or acc[e] == "xá"))     # "cổng phía trước", "trạm xá nằm gần ..."
        starts_known = lambda e: any(x[0] == e for x in known)

        def scan(s, ends, first_ok=False):
            if (cut_word(s) and not first_ok) or words[s] in ends or words[s] in DIRS or words[s] in ("cach", "hon", "nhat"):
                return None                                   # "ở xa X", "ở gần X": chưa phải tên địa điểm
            e = s
            while e < n and e - s < 7:
                if e > s and (brk(e) or words[e] in ends or is_cut(e) or starts_known(e) or _tail_stop(words, acc, e)
                              or (NEG_SPAN_STOP and _neg_word(words, e)
                                  and not (e + 1 < n and words[e + 1] in ("gian", "khi", "quan", "phan", "dung")))):
                    # "nơi / chỗ" + việc làm ("chỗ lấy thuốc", "nơi nhận hồ sơ"): "nơi / chỗ" đứng một mình không phải tên,
                    # nên "lấy / nhận" ngay sau nó thuộc tên chứ không phải khung "đến X lấy Y"
                    if not (TYPE_FIX and e == s + 1 and words[s] in ("noi", "cho") and words[e] in ("lay", "nhan", "lanh")
                            and not brk(e) and not starts_known(e) and e + 1 < n and not brk(e + 1)):
                        break
                e += 1
            if e < n and e - s >= 7:
                return None                                   # bị cắt vì quá dài: không gọn
            if e < n and starts_known(e):
                return None                                   # "khu" + tên đã biết: mốc / đích là tên đã biết
            if e - s >= 2 and words[e - 1] == "nha":
                e -= 1                                        # trợ từ "nha" cuối câu
            return e if 1 <= e - s <= 6 else None

        def ref_follows(e):
            """REF_SLOT: ngay sau cụm là một tham chiếu không gian ("... gần / xa / sát / cách xa Y", "... (nằm / ở) phía bắc",
            "... mạn dưới"): cụm đứng trước tham chiếu là TÊN địa điểm dù không mở đầu bằng phòng / khu / nơi..."""
            q = e
            while q < n and q - e < 2 and words[q] in ("nam", "o"):
                q += 1
            if q < n and words[q] in ("gan", "xa", "sat", "canh", "ke", "cach"):
                return True
            return q + 1 < n and words[q] in ORI_WORDS and words[q + 1] in DIRS

        def ok(s, e, strong=False):
            for m in ms:
                if s < m[1] and e > m[0] and not (m[2] is None and s <= m[0] and m[1] <= e):
                    return False
            if any(s < b and e > a for a, b in out):
                return False
            if SUBJ_TRIM and e - s == 1 and words[s] in LONE_BAD:
                return False                                  # "... xa nhà": "nhà" đứng một mình không phải tên địa điểm
            mp = float(np.mean(pw[s:e]))
            if words[s] in ITEM_HEADS and words[s] not in PLACE_HEADS and mp < 0.5:
                return False
            if words[s] not in HEADS and mp < 0.3 and not (strong and any(
                    w not in EDGE_STRIP and w not in GOAL_FRAME_END and w not in DIRS and w not in ORI_WORDS for w in words[s:e])):
                return False
            return not _all_ref(words[s:e])

        for q in range(n):
            w = words[q]
            s = None
            strong = False
            if FRAME_RULE and (w == "ghe" or (q > 0 and w in ("qua", "vao") and words[q - 1] in ("tat", "re", "ghe", "tien"))
                               or (TAIL_FIX and w in ("qua", "vao", "sang") and (q == 0 or brk(q)))):   # đợt 20: "Qua X lấy ..., sau đó ..."
                s = q + 1
                if s < n and words[s] in ("qua", "vao"):
                    s += 1
                # "ghé kệ sách chung lấy ...": sau "ghé" chữ "ke" là "kệ" (đầu tên), không phải "kế bên" (SUBJ_TRIM, h48 train)
                ke_head = SUBJ_TRIM and s + 1 < n and words[s] == "ke" and words[s + 1] not in ("ben", "canh", "sat", "can")                     and (not has_acc or acc[s] == "kệ")
                e = scan(s, FRAME_END, first_ok=ke_head) if s < n else None
                # "ghé X (nằm) xa / gần Y lấy ...": cụm tham chiếu đẩy chữ "lấy" ra xa hơn
                far_ok = FRAME_STRONG and e is not None and ref_follows(e)
                if not any(words[x] in FRAME_END for x in range(s + 1, min(s + (15 if far_ok else 9), n))):
                    continue
                # "ghé / tạt qua <X> lấy / nhận / trước": X chắc chắn là địa điểm (điểm ghé) -> không đòi từ mở đầu quen
                strong = FRAME_STRONG and e is not None and e < n and words[e] in FRAME_END and not brk(e) \
                    and not (has_acc and w == "ghe" and acc[q] != "ghé")
            elif GOAL_FRAME_RULE and q > 0 and words[q - 1:q + 1] in ((["diem", "giao"], ["diem", "nhan"], ["dich", "den"])
                                                                  + ((["nguoi", "nhan"],) if RECIP_FRAME else ())
                                                                  + ((["chang", "mot"], ["chang", "hai"], ["chang", "dau"], ["chang", "cuoi"],
                                                                      ["diem", "ghe"], ["diem", "dung"], ["noi", "nhan"], ["noi", "giao"],
                                                                      ["diem", "dau"], ["diem", "cuoi"], ["dia", "chi"]) if FRAME2_FIX else ())) \
                    and q + 1 < n and words[q + 1] not in ("la", "cua", "o") and (idx is None or brk(q + 1)):
                s = q + 1                                     # "Điểm giao: X" (dấu hai chấm)
                e = scan(s, GOAL_FRAME_END)
                strong = GOAL_SLOT_STRONG and e is not None and (e == n or brk(e))
            elif FRAME2_FIX and w == "la" and q > 0 and q + 2 < n and words[q + 1] in ("diem", "noi", "cho", "chang") \
                    and re.match(LEG_SUBJ, " ".join(words[q + 1:q + 5])):
                # (đợt 20, thử) "X là điểm cuối / nơi nhận", "Còn V là điểm dừng đầu tiên": X đứng đầu vế, trước "là"
                s = max([k for k in range(q) if k == 0 or brk(k)] or [0])
                while s < q - 1 and words[s] in ("con", "va", "nhung", "thi", "rieng", "tuc"):
                    s += 1
                e = q if 1 <= q - s <= 6 else None
                strong = e is not None
            elif FRAME_STRONG and q == 1 and words[:2] == ["hang", "cho"] and q + 1 < n:
                # "Hàng cho X: Y" (khung của validation, 16/300 câu; 0 câu train): X là nơi nhận / người nhận
                s = q + 1
                e = scan(s, GOAL_FRAME_END)
                strong = e is not None and (e == n or brk(e) or ref_follows(e))
            elif GOAL_FRAME_RULE and q > 0 and w in ("toi", "den", "la", "qua", "o"):
                if w == "la" and words[max(0, q - 2):q] not in ((["dich", "den"], ["diem", "giao"], ["diem", "nhan"], ["noi", "nhan"])
                                                                + ((["nguoi", "nhan"],) if RECIP_FRAME else ())) \
                        and not (FRAME_STRONG and words[max(0, q - 3):q] == ["noi", "can", "giao"]):
                    continue
                if w in ("toi", "den", "qua") and not any(v in ("mang", "giao", "chuyen", "gui", "dua", "van") for v in words[max(0, q - 7):q]):
                    continue
                if w == "qua" and words[q - 1] in ("tat", "re", "ghe", "tien", "di", "hom"):
                    continue
                if w == "o" and (other(q) or words[q - 1] in O_IN_NAME or cua(q + 1)
                                 or any(m[0] < q < m[1] for m in ms) or brk(q)):
                    continue                                  # "khu ở của sinh viên": "ở" nằm trong tên
                s = q + 1
                e = scan(s, GOAL_FRAME_END) if s < n else None
                # "đích đến là / nơi cần giao là X" (rồi hết vế): X chắc chắn là địa điểm
                strong = FRAME_STRONG and w == "la" and e is not None and (e == n or brk(e) or words[e] in GOAL_FRAME_END)
                # (đợt 18) "chuyển thùng hàng tới <X>." -- X >= 2 chữ, trọn tới hết vế: chắc là địa điểm ("góc tài liệu tham khảo")
                strong = strong or (SUBJ_TRIM and w in ("toi", "den") and e is not None and e - s >= 2
                                    and (e == n or brk(e) or words[e] in GOAL_FRAME_END) and words[s] not in ITEM_HEADS)
            elif RECIP_FRAME and q > 0 and w == "cho" and not (has_acc and acc[q] != "cho") and q + 1 < n \
                    and any(v in ("mang", "giao", "chuyen", "gui", "dua", "van") for v in words[max(0, q - 7):q]):
                # (đợt 18) "gửi chồng giáo trình CHO thầy chủ nhiệm.": người / nơi nhận đứng sau "cho" (trọn tới hết vế) --
                # chỉ nhận khi cụm có từ khóa nghề / nơi chốn ("bảo vệ", "thủ thư", "y tá", "chủ nhiệm"...), không nhận "cho mình"
                s = q + 1
                e = scan(s, GOAL_FRAME_END)
                if e is None or not (e == n or brk(e) or words[e] in GOAL_FRAME_END) or words[s] in PERSON_STOP0 \
                        or keyword_scores(" ".join(words[s:e])).max() < 2:
                    continue
                strong = True
            else:
                continue
            if e is not None and ok(s, e, strong=strong or (REF_SLOT and ref_follows(e))):
                out.append((s, e))
        if ANCHOR_RULE:
            for m in sorted(ms + [(a, b, None, None) for a, b in out], key=lambda x: (x[0], x[1])):
                q = m[1]
                if q < n and words[q] in ("nam", "o"):
                    q += 1
                if q < n and words[q] == "cach" and q + 1 < n and words[q + 1] == "xa":
                    q += 1
                if q >= n or words[q] not in ("gan", "xa", "sat", "canh", "ke") or other(q):
                    continue
                s = q + 1
                if s < n and words[s] in ("voi", "ben", "canh") + (("vach",) if VACH_FIX else ()):
                    s += 1
                e = scan(s, ANCHOR_END) if s < n else None
                # "<địa điểm> gần / xa / sát <Y> hơn / giúp / , ...": Y là mốc (địa điểm) dù không mở đầu bằng từ quen
                strong = FRAME_STRONG and e is not None and (e == n or brk(e) or words[e] in ANCHOR_END)
                if e is not None and ok(s, e, strong=strong):
                    out.append((s, e))
        return out

    @staticmethod
    def _person_subject(tt, words, idx, ms):
        """Khung "<người nhận> (ở <nơi>) cần nhận / đang chờ / đang cần ...": chủ ngữ là người nhận = đích.
        Trả về (i, j) của chủ ngữ nếu nó chưa được nhận là địa điểm, ngược lại None."""
        k = next((q for q in range(1, len(words) - 1) if words[q:q + 2] in (["can", "nhan"], ["dang", "cho"], ["dang", "can"])), None)
        if k is None:
            return None
        punct = [q for q, t in enumerate(tt) if t in (",", ":") and q < idx[k]]
        s0 = next((q for q in range(len(words)) if idx[q] > punct[-1]), k) if punct else 0
        e = next((q for q in range(s0, k) if words[q] == "o"
                  and not (SUBJ_TRIM and q > s0 and words[q - 1] in O_IN_NAME
                           and (q + 1 == k or tuple(words[q + 1:q + 3]) in O_QUAL))), k)   # "<người> ở <nơi>": chỉ lấy phần người
        if SUBJ_TRIM:
            # "Cán bộ / Ban quản lý / Nhân viên <nơi> cần nhận": bỏ phần người đứng trước từ mở đầu tên nơi chốn
            # ("khu phòng ở", "nơi ôn bài") -- phần người làm lệch loại đoán từ tên (h48 train, tên tự soạn)
            h = next((q for q in range(s0 + 1, min(s0 + 4, e - 1)) if words[q] in SUBJ_PLACE_HEADS
                      and (all(w not in PLACE_HEADS for w in words[s0:q]) or tuple(words[s0:q]) in PERSON_LEADS)), None)
            trimmed = h is not None and not any(h < m[1] and e > m[0] for m in ms)
            if trimmed:
                s0 = h
        else:
            trimmed = False
        subj = words[s0:e]
        if not 1 <= len(subj) <= 5 or any(s0 < m[1] and e > m[0] for m in ms):
            return None
        if (subj[0] in PERSON_STOP0 and not trimmed) or all(w in PERSON_STOP for w in subj) or any(dl_distance(w, "robot", 1) <= 1 for w in subj):
            return None
        return s0, e

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
    def _role_text(me):
        """Câu con CÓ DẤU, tên đang xét thay bằng "nơi A", các tên khác bằng "nơi B" (cho bộ phân loại vai bằng nghĩa)."""
        acc, i, ms = me["acc"], me["i"], me["ms"]
        out, k = [], 0
        while k < len(acc):
            hit = next((m for m in ms if m[0] == k), None)
            if hit:
                out.append("nơi A" if hit[0] == i else "nơi B"); k = hit[1]
            else:
                out.append(acc[k]); k += 1
        return " ".join(out)

    @staticmethod
    def _anchor_of(me):
        nxt = [x for x in me["ms"] if x[0] >= me["j"]]
        if nxt and nxt[0][0] - me["j"] <= 3:
            return nxt[0]
        # mốc đứng xa hơn một chút ("X ở tận phía xa Y", "X không nằm gần Y"): nhận nếu giữa hai tên có từ gần / xa / cách
        if CACH_RULE and nxt and nxt[0][0] - me["j"] <= 5 and any(t in ("gan", "xa", "cach", "sat", "ke", "canh", "giap")
                                                                  + (("phia", "huong", "kia", "voi", "tu") if REF_TOWARD else ())
                                                                  for t in me["toks"][me["j"]:nxt[0][0]]):
            return nxt[0]
        return None

    @staticmethod
    def _masked_sent(me):
        """(câu không dấu, câu có dấu) với các tên địa điểm thay bằng "x": luật gấp / dễ vỡ không được khớp vào chữ của
        TÊN ("chỗ để xe" ~ "chờ", "phòng cấp cứu" ~ "cấp cứu" -> gấp). Bản cũ (NEW_FRAME_FIX=False) giữ nguyên câu."""
        ws, acc = me["sent"].split(), list(me["acc"])
        if not NEW_FRAME_FIX or len(ws) != len(acc):
            return me["sent"], " ".join(me["acc"])
        for m in me["ms"]:
            for q in range(m[0], m[1]):
                ws[q] = acc[q] = "x"
        if ITEM_FIX:
            for a, b in MissionParser2._item_slots(me["toks"], me.get("idx"), me["ms"]):
                for q in range(a, b):
                    ws[q] = acc[q] = "x"                      # món hàng ("giao thuốc cấp cứu tới X") không mang cờ
        return " ".join(ws), " ".join(acc)

    @staticmethod
    def _item_slots(words, idx=None, ms=()):
        """Các khoảng [a, b) là cụm MÓN HÀNG (ITEM_FIX): ngay sau động từ giao (giao / gửi / đưa / mang / chuyển), ngay trước
        giới từ chỉ đích (tới / đến / qua / sang / về / vào / cho), không có dấu câu hay giới từ khác chen giữa, dài 1-6 từ.
        "Mang sách thư viện tới bãi xe": [sách thư viện]; "Giao hàng ở thư viện tới X", "Mang từ X tới Y": không có."""
        if not ITEM_FIX:
            return []
        n = len(words)
        out = []
        brk = lambda k: idx is not None and 0 < k < n and idx[k] != idx[k - 1] + 1
        # từ dừng của cụm món hàng, trừ khi là chữ đầu của danh từ ghép không dấu ("tai lieu" = tài liệu, "o to" = ô tô)
        stop = lambda k: words[k] in SLOT_END and not (k + 1 < n and (words[k], words[k + 1]) in (("tai", "lieu"), ("o", "to")))
        cstart = lambda k: max([0] + [e for e in range(1, k + 1) if brk(e)])      # đầu vế chứa vị trí k
        cend = lambda k: next((e for e in range(k + 1, n) if brk(e)), n)          # cuối vế chứa vị trí k
        in_name = {k for m in ms for k in range(m[0], m[1])}   # "phòng giáo vụ" ~ "giao vu", "khu gửi xe": không phải động từ
        for q, w in enumerate(words):
            if q in in_name:
                continue
            # (a) "giao <món> tới / đến / qua / cho / ở <nơi>"; "lấy / nhận <món> ở / tại / từ <nơi>"
            if w in DELIVER_VERBS or (w in PICK_VERBS and not (q > 0 and words[q - 1] in PICK_NOUN)):
                ends = DELIVER_END if w in DELIVER_VERBS else PICK_END
                b = q + 1
                while b < n and b - q <= 7 and not brk(b) and not stop(b):
                    b += 1
                if q + 1 < b <= q + 7 and b < n and not brk(b) and words[b] in ends:
                    out.append((q + 1, b))
                elif ITEM_FIX2 and w in PICK_VERBS and (any(m[1] == q for m in ms) or (
                        q > 1 and words[q - 1] in ("de", "va", "ma") and any(m[1] == q - 1 for m in ms))) \
                        and q + 1 < b <= q + 7 and (b == n or brk(b) or words[b] in ITEM_TAIL_END):
                    out.append((q + 1, b))      # (e) "ghé X lấy <món>" tới hết vế ("... lấy thẻ thư viện."): món, không phải nơi
            # (b) "hàng cần giao là <món>" (tới hết vế); KHÔNG phải "nơi lấy hàng là X", "điểm giao hàng là X" (VIA_FIX)
            mi = ITEM_IS.search(" ".join(words[max(0, q - 4):q])) if w == "la" else None
            if mi:
                before = " ".join(words[max(0, q - 4):q])[:mi.start()].split()
                if not (VIA_FIX and before and before[-1] in ITEM_NOT_BEFORE):
                    b = cend(q)
                    if 1 <= b - (q + 1) <= 7:
                        out.append((q + 1, b))
            # (c) "(đang / vẫn / còn) chờ / đợi <món>" (tới hết vế; "đang chờ ở X" thì không)
            if w in ("cho", "doi") and q > 0 and words[q - 1] in ("dang", "van", "con") and q + 1 < n and not brk(q + 1) \
                    and not stop(q + 1):
                b = cend(q)
                if b - (q + 1) <= 7:
                    out.append((q + 1, b))
            # (d) "<món> phải được đưa / cần chuyển (giúp) tới <nơi>": cụm đầu vế (không chứa "ở / tại": người ở đâu đó)
            if w in DELIVER_VERBS:
                k = q
                while k - 1 >= 0 and not brk(k) and words[k - 1] in MODALS:
                    k -= 1
                r = q + 1
                while r < n and words[r] in ("giup", "gium", "ho", "dum"):
                    r += 1
                if k < q and r < n and words[r] in DEST_PREPS:
                    s0 = cstart(k)
                    if s0 < k <= s0 + 7 and not any(words[x] in ("o", "tai", "tu") and stop(x) for x in range(s0, k)):
                        out.append((s0, k))
        return out

    def _dir_hint(self, s, core=False):
        """Ngữ cảnh có từ chỉ hướng / vị trí (kể cả gõ sai 1 ký tự)? Bản mới: bỏ các nghĩa khác ("phải được", "nhẹ tay"...)
        và chỉ coi là gõ sai khi từ HIẾM ("được" gần "dưới" 1 ký tự nhưng là từ rất quen, không phải lỗi gõ)."""
        for t in _dir_text(s).split():
            if len(t) < 2:
                continue
            hints = DIR_CORE if core else DIR_HINTS
            if t in hints:
                return True
            if NEW_FRAME_FIX and self.freq.get(t, 0) >= 20:
                continue
            if any(dl_distance(t, d, 1) <= 1 for d in hints):
                return True
        return False

    def _fix_cue(self, t):
        """Chữ then chốt của các luật (phủ định / ghé / khung nhãn) bị gõ sai -> chữ đúng. Chỉ sửa từ HIẾM (không có trong
        từ vựng quen) cách đúng MỘT chữ then chốt 1 lỗi đảo / rơi / thay; từ quen ("hỏng", "tòa"...) giữ nguyên."""
        if self._khong_typo(t):
            return "khong"
        if not NEW_FRAME_FIX or len(t) < 3 or t in KEY_CUE_SET or t == "dungx" or _valid_syllable(t):
            return t
        c = [q for q in KEY_CUES if _typo1(t, q)]
        return c[0] if len(c) == 1 else t

    @staticmethod
    def _contrast_clause(me, w):
        """Vế (giữa các từ tương phản "mà (là)", "chứ", "nhưng", "thay vì") chứa lần nhắc này, trong đoạn giữa hai dấu phẩy.
        -> (lo, hi) vị trí từ của vế; None nếu đoạn không có từ tương phản. "mà / chứ" có dấu khác ("mã", "chú bảo vệ")
        hoặc nằm trong tên thì không tính."""
        segw = [t for t in me["seg"].split() if t not in (",", ":")]
        sw = me["sent"].split()                               # cùng dạng với seg (có "dungx"), cùng vị trí với w
        i, j, n = me["i"], me["j"], len(segw)
        s0 = next((s for s in range(max(0, j - n), min(i, len(sw) - n) + 1) if sw[s:s + n] == segw), None)
        if s0 is None or len(sw) != len(w):
            return None
        acc = [a.lower() for a in me["acc"]] if len(me["acc"]) == len(w) else list(w)
        inside = lambda k: any(m[0] <= k < m[1] for m in me["ms"])
        ok = {"ma": ("mà", "ma"), "chu": ("chứ", "chu"), "nhung": ("nhưng", "nhung")}
        b = [k for k in range(s0, s0 + n) if not inside(k) and (
            (w[k] in ok and acc[k] in ok[w[k]]) or (w[k] == "thay" and k + 1 < len(w) and w[k + 1] == "vi"))]
        if not b:
            return None
        # "mà (là) / chứ / nhưng" chỉ là từ nối (bỏ khỏi vế sau); "thay vì Y" GIỮ lại trong vế của Y (= Y bị phủ định)
        lo = max([k if w[k] == "thay" else k + 1 for k in b if k < i] + [s0])
        hi = min([k for k in b if k >= j] + [s0 + n])
        return lo, hi

    def _fix_seq(self, toks, accs=None):
        """Sửa chữ then chốt gõ sai trong một dãy từ: (1) âm tiết sai cấu trúc (_fix_cue); (2) âm tiết HỢP LỆ do rơi / đảo
        chữ ("hông phải", "nầm với", "ôm qua"): cặp (chữ, từ sau) chưa từng gặp nhưng (chữ then chốt, từ sau) gặp >= 5 lần.
        accs: dạng có dấu cùng vị trí (nếu có). Câu có dấu: chỉ sửa khi dạng có dấu của chữ CHƯA từng gặp ("hông" vs "hỏng",
        "đã" là từ quen -> không sửa); câu không dấu: chỉ sửa chữ hiếm (< 50 lần)."""
        out = [self._fix_cue(t) for t in toks]
        bg = getattr(self, "bigram", None)
        if not (NEW_FRAME_FIX and BIGRAM_FIX and bg):
            return out
        if accs is not None and len(accs) != len(toks):
            accs = None
        forms = getattr(self, "acc_forms", {})
        for k in range(len(out) - 1):
            t, nxt = out[k], out[k + 1]
            if t in KEY_CUE_SET or t == "dungx" or not t.isalpha():
                continue
            a = accs[k].lower() if accs is not None else t
            unseen_acc = TYPO_FIX2 and a != t and not a.isascii() and a not in forms.get(t, ())
            prv = out[k - 1] if k > 0 else None
            # (đợt 18) ngữ cảnh HAI PHÍA: "đừng _ với" -> "nhầm" ("Đừng hầm với X"): cả (từ trước, chữ then chốt) và (chữ then
            # chốt, từ sau) đều rất quen, còn (từ trước, t) và (t, từ sau) chưa từng gặp -> sửa dù t là từ có thật
            two = []
            if TYPO_FIX2 and prv is not None and not bg.get((prv, t), 0) and not bg.get((t, nxt), 0):
                two = [q for q in KEY_CUES if bg.get((prv, q), 0) >= 10 and bg.get((q, nxt), 0) >= 10 and _typo1(t, q)]
            if len(two) == 1:
                out[k] = two[0]
                continue
            if len(t) < 2 and not unseen_acc:
                continue
            if bg.get((t, nxt), 0) and not unseen_acc and not (TYPO_FIX2 and a == t and self.freq.get(t, 0) >= 50):
                continue
            strong_ctx = False
            if a != t:
                if a in forms.get(t, ()):
                    continue                                   # dạng có dấu quen ("đã", "hỏng"): từ thật
            elif self.freq.get(t, 0) >= 50:
                # không dấu: từ quen thì không đoán là gõ sai -- trừ khi (chữ then chốt, từ sau) quen gấp >= 30 lần (t, từ sau):
                # "hong can ghe" = "không cần ghé" ("khong can" 191 lần, "hong can" 1 lần -- chính là một lỗi gõ của dữ liệu)
                if not TYPO_FIX2:
                    continue
                strong_ctx = True
            need = 30 * max(1, bg.get((t, nxt), 0)) if strong_ctx else 5
            c = sorted((q for q in KEY_CUES + KEY_CUES2 if bg.get((q, nxt), 0) >= need and _typo1(t, q)),
                       key=lambda q: -bg.get((q, nxt), 0))
            if len(c) == 1:
                out[k] = c[0]
            elif TYPO_FIX2 and len(c) > 1 and bg.get((c[0], nxt), 0) >= 3 * bg.get((c[1], nxt), 0):
                out[k] = c[0]                                  # "ã rời" -> "đã rời" (gấp 3 lần "ra rời")
        return out

    def _khong_typo(self, t):
        """t có phải chữ "không" gõ sai ("khng", "khogn")? Từ có thật ("hỏng", "thong (thả)", "phòng", "chống"...) thì KHÔNG."""
        if len(t) < 4 or t == "khong" or dl_distance(t, "khong", 1) > 1:
            return False
        if KHONG_STRICT and (t in REAL_WORDS_NEAR_KHONG or self.freq.get(t, 0) >= KHONG_FREQ):
            return False
        return True

    def _phrase_flags(self, s, acc=None):
        """-> (gấp?, dễ vỡ?) của một câu con không chứa địa điểm:
        bảng cụm đã học -> khớp mờ -> luật từ khóa -> vector nghĩa e5 -> phân loại ký tự."""
        if DNEG_FIX and re.search(_DNEG, s):
            # đợt 20: phủ định KÉP triệt tiêu ("đơn này không phải không gấp" = gấp, "không phải là không dễ vỡ" = dễ vỡ)
            s = re.sub(r"\s+", " ", re.sub(_DNEG, " ", s)).strip()
            s = re.sub(r"\b(dau|ca)\s*$", "", s).strip()     # "... không phải không gấp ĐÂU": "đâu" đi cùng phủ định đã bỏ
            acc = None
        s_in = s                                              # (trước khi khử nhập nhằng che chữ "cẩn" ~ "cần")
        if ITEM_FIX and ITEM_LINE.match(s):
            item = s.split(":", 1)[1]
            if not (re.search(ITEM_FRAGILE_PRED, item) or re.search(ITEM_URGENT_PRED, item)):
                self._ev.append((False, False, False, False))  # "Hàng: bình hoa", "Hàng: thuốc cấp cứu": chỉ nêu món hàng
                return False, False
        if NEW_FRAME_FIX and ROAD_CTX.search(s) and _road_free(s) != s:
            # "Đường hơi trơn, đi cẩn thận nhé": bỏ cụm dặn đi đường TRƯỚC khi tách vế (vế "đi cẩn thận nhé" đứng riêng
            # không còn thấy "đường trơn" ở vế kia)
            return self._phrase_flags(re.sub(r"\s+", " ", _road_free(s)).strip(" ,"), None)
        lab = self.phrase.get(s)
        how = "bảng"
        if lab is None:
            how = "bảng mờ"
            best, bl = 3, None
            for p, l in self.phrase.items():
                if abs(len(p) - len(s)) <= 2:
                    d = dl_distance(s, p, 2)
                    if d < best:
                        best, bl = d, l
            if bl is not None and best <= 2:
                lab = bl
        parts = re.split(r"[:,] ", s)
        if lab is None and PART_OR and len(parts) > 1:
            # câu nhiều vế ("hàng dễ vỡ, <vế lạ>"): xét TỪNG vế rồi gộp (trước đây chỉ lấy nhãn của vế đã học, bỏ qua vế lạ)
            accw = acc.split() if acc else None
            if accw is not None and len(accw) != sum(len(tokens(x)) for x in parts):
                accw = None
            ru = rf = False
            pos = 0
            seq = []                      # theo thứ tự các vế: True = vế gấp, False = vế nói rõ không gấp
            for part in parts:
                n = len(tokens(part))
                pa = " ".join(accw[pos:pos + n]) if accw else None
                pos += n
                if n:
                    pu, pf = self._phrase_flags(part.strip(), pa)
                    ru, rf = ru or pu, rf or pf
                    if pu:
                        seq.append(True)
                    elif re.search(URGENT_NEG, part):
                        seq.append(False)
            if True in seq and False in seq:
                # câu trộn: một vế gấp + một vế "không gấp". MIXED_POLICY: "urgent" | "calm" | "last" (vế cuối quyết định) | "first"
                ru = {"urgent": True, "calm": False, "last": seq[-1], "first": seq[0]}[MIXED_POLICY]
            return ru, rf
        if lab is None:
            how = "bảng (vế)"
            for part in parts:
                if self.phrase.get(part):
                    lab = self.phrase[part]
        if lab is not None:
            # (gấp có căn cứ chắc?, dễ vỡ có căn cứ chắc?, câu nói rõ KHÔNG gấp?, câu nói rõ KHÔNG dễ vỡ?)
            self._ev.append((lab == 1, lab == 2, lab == 0 and bool(re.search(URGENT_NEG, s)), lab == 0 and bool(re.search(FRAGILE_NEG, s))))
            self._trace(how, lab == 1, lab == 2)
            return lab == 1, lab == 2
        s = disambiguate(s, acc)
        # chữ "không" gõ sai ("khng", "khogn"...) -> "khong", để các luật phủ định vẫn bắt được
        s = " ".join("khong" if self._khong_typo(t) else t for t in s.split())
        u = bool(re.search(URGENT_POS, s)) and not re.search(URGENT_NEG, s)
        f = bool(re.search(FRAGILE_POS, _road_free(s))) and not re.search(FRAGILE_NEG, s)
        if NEW_FRAME_FIX and f and re.search(CAREFUL_OTHER, s_in) and not re.search(FRAGILE_POS, re.sub(r"\bcan than\b", " ", s)):
            f = False                                         # "cẩn thận kẻo trễ / lạc / nhầm": dặn dò, không nói về hàng
                                                              # (xét trên câu gốc: khử nhập nhằng có thể che "lạc" ~ "lắc")
        u_rule, f_rule = u, f
        silent = not u and not f and not re.search(URGENT_NEG + "|" + FRAGILE_NEG, s)
        if getattr(self, "ablate_rules", False):              # thí nghiệm: coi như không có luật từ khóa
            u, f, silent = False, False, True
        delivery = re.search(r"\b(can nhan|dang cho|giao|mang|gui|dua|chuyen|ghe|toi|den|hang:)", s)
        if getattr(self, "use_e5", False) and len(tokens(s)) <= 12:
            # Kết hợp luật từ khóa với mô hình nghĩa (5 lớp: trung tính / gấp / dễ vỡ / không gấp / không dễ vỡ).
            # Mô hình nghĩa chỉ được tin khi câu có dấu (ngưỡng 0.6) hoặc khi nó rất chắc (0.8).
            txt = acc if acc else s
            th = E5_TH[0] if txt != s else E5_TH[1]
            if E5_RESTORE and txt.isascii():
                # câu KHÔNG DẤU: thêm dấu bằng bảng học từ train/validation rồi mới hỏi e5 (e5 hiểu câu có dấu tốt hơn nhiều:
                # "hang de sut me" 0,66 trung tính -> "hàng dễ sứt mẻ" 0,93 dễ vỡ); ngưỡng ở giữa hai ngưỡng cũ
                r_ = self.restore(s)
                if r_ != s:
                    pu_ = self.e5_phrase.predict_proba(embedder().encode([txt]))[0]
                    pr_ = self.e5_phrase.predict_proba(embedder().encode([r_]))[0]
                    # lấy bản (gốc / đã thêm dấu) mà e5 chắc hơn: thêm dấu sai ("nhanh chắn ... nhẹ") không được làm mất cờ
                    if pr_.max() > pu_.max():
                        txt, th = r_, E5_TH_R
            pe = self.e5_phrase.predict_proba(embedder().encode([txt]))[0]
            k = int(self.e5_phrase.classes_[int(np.argmax(pe))])
            sure = pe.max() > th
            if k in (1, 2) and re.match(r"(hang|mon|kien|don)\b", s) and not (E5_ITEM_GUARD2 and not re.match(ITEM_DESC, s)):
                sure = False                                  # câu mô tả món hàng ("hàng: bưu kiện") hay bị nhận nhầm là gấp
            # từ phủ định chung ("chẳng", "đâu", "khỏi"...) mà không thuộc mẫu phủ định kép ("không được chậm trễ")
            gen_neg = bool(re.search(GEN_NEG, s)) and not re.search(DOUBLE_NEG, s) and not getattr(self, "no_gen_neg", False)
            has_neg = bool(re.search(GEN_NEG, s))
            if u:
                if (gen_neg and not (sure and k == 1)) or (sure and k == 3 and pe.max() > 0.75 and (gen_neg or not VETO_NEEDS_NEG)):
                    u = False                                 # có từ "gấp" nhưng cả câu mang nghĩa phủ định
            elif E5_POS_URGENT and sure and k == 1 and not re.search(URGENT_NEG, s) and not gen_neg \
                    and not (E5_CROSS_GUARD and f_rule):
                u = True
            if f:
                if (gen_neg and not (sure and k == 2)) or (sure and k == 4 and pe.max() > 0.75 and (gen_neg or not VETO_NEEDS_NEG)):
                    f = False
            elif E5_POS_FRAGILE and sure and k == 2 and not re.search(FRAGILE_NEG, s) and not gen_neg and not (
                    NEW_FRAME_FIX and (ROAD_CTX.search(s_in) or re.search(r"\b(can than|nhe nhang)\b", s_in))) \
                    and not (E5_CROSS_GUARD and u_rule):
                f = True                                      # (câu có "cẩn thận / nhẹ nhàng" mà luật đã xác định là nghĩa khác,
                                                              #  hoặc câu dặn đi đường: mô hình nghĩa không được tự nhận dễ vỡ)
            self._ev.append((u and u_rule, f and f_rule, not u and (bool(re.search(URGENT_NEG, s)) or (sure and k == 3)),
                             not f and (bool(re.search(FRAGILE_NEG, s)) or (sure and k == 4))))
            self._trace("luật" if (u_rule or f_rule) else "e5", u, f)
            return u, f
        if silent and len(tokens(s)) <= 7 and not delivery:
            pr = self.phrase_clf.predict_proba(self.v_p.transform([s]))[0]
            k = int(np.argmax(pr))
            if pr[k] > 0.6:
                self._ev.append((False, False, False, False))
                self._trace("ký tự", k == 1, k == 2)
                return k == 1, k == 2
        self._ev.append((u, f, not u and bool(re.search(URGENT_NEG, s)), not f and bool(re.search(FRAGILE_NEG, s))))
        self._trace("luật", u, f)
        return u, f

    def _trace(self, how, u, f):
        """Chẩn đoán (không đổi kết quả): ghi lại cách quyết định cờ gấp / dễ vỡ, nếu bật self._ftrace (list)."""
        tr = getattr(self, "_ftrace", None)
        if tr is not None and (u or f):
            tr.append((how, u, f))

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
            if me["type"] is None and not me.get("person") and me["toks"][me["i"]] not in PLACE_HEADS and keyword_scores(me["text"]).max() < 3:
                continue
            prev = [q for q in range(k) if ments[q]["si"] == me["si"] and ments[q]["j"] <= me["i"]]
            if prev:
                q = prev[-1]
                gap = me["toks"][ments[q]["j"]:me["i"]]
                if len(gap) <= 3 and any(g in ("gan", "xa", "sat", "canh", "ke", "cach") for g in gap) and not negs[q]:
                    role[k] = 3                                # "A gần B", "A cách xa B" -> B là mốc
                    continue
                if ent and ent[-1][-1] == q and (gap in (["o"], ["tai"], ["cua"], ["o", "tai"]) or (
                        SAME_ENT_HEADS and 2 <= len(gap) <= 3 and gap[0] in ("o", "tai") and all(g in HEADS for g in gap[1:])) or (
                        SUBJ_TRIM and 2 <= len(gap) <= 3 and gap[-1] in ("o", "tai") and not any(
                            g in EDGE_STRIP or g in KEY_CUE_SET or g in DIRS or g in REF_WORDS or g in ITEM_HEADS
                            or g in ("tra", "hang", "do", "di", "ve") for g in gap[:-1])
                        and all(me["idx"][x] == me["idx"][x - 1] + 1 for x in range(ments[q]["j"], me["i"] + 1)))):
                    # (đợt 17) "phòng đọc YÊN TĨNH ở khu đọc sách": phần đuôi tên chưa nhận ra nằm giữa tên quen và "ở"
                    ent[-1].append(k)                          # "thủ thư ở (khu) thư viện": người nhận + nơi làm việc là một
                    if SAME_ENT_HEADS and role[q] in (0, 1):
                        role[k] = role[q]                      # cùng vai trò với người nhận
                    continue
            ent.append([k])
        if len(ent) == 1:
            # (chỉ khi KHÔNG còn lần nhắc nào khác chưa bị phủ định: "Nhân viên quầy ăn cần nhận ..." là tên lạ bị bỏ khỏi
            #  `ent` nhưng vẫn là đích -- validation, bộ đọc chỉ học train)
            if SUBJ_TRIM and all(role[k] == 1 for k in ent[0]) and not any(
                    not negs[q] and role[q] != 3 and q not in ent[0] for q in range(len(ments))):
                for k in ent[0]:
                    role[k] = 0                                # chỉ một nơi mà lại là điểm ghé -> không có đích: nó là đích
                                                               # (train + validation: luôn có đích được nhắc; "tới cổng trước giúp mình")
            if all(role[k] == 2 for k in ent[0]):
                for k in ent[0]:
                    role[k] = 0
            return
        if len(ent) < 2:
            return
        has_via = [any(role[k] == 1 for k in e) for e in ent]
        has_goal = [any(role[k] == 0 for k in e) for e in ent]
        valid = sum(has_via) == 1 and sum(g and not v for g, v in zip(has_goal, has_via)) >= 1
        if valid and not (CUE_OVERRIDE and len(ent) == 2):
            return                                             # bộ phân loại đã cho một kết quả hợp lệ
        top = sorted(ent, key=lambda e: -max(P[k, 0] + P[k, 1] for k in e))[:2]
        top.sort(key=lambda e: e[0])
        A, B = top
        # "qua" chỉ là động từ ghé khi đi sau "tạt / rẽ / ghé / trước tiên" ("gửi X qua Y": Y là đích)
        VIA_V = r"\b(ghe|tat|re|lay|nhan|lanh|dung|vong)\b|\b(tat|re|ghe|tien) qua\b" if CUE_OVERRIDE \
            else r"\b(ghe|tat|re|lay|nhan|lanh|dung|vong|qua)\b"
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

        p_a = float(np.log(max(P[k, 1] for k in A) + 1e-3) + np.log(max(P[k, 0] for k in B) + 1e-3))
        p_b = float(np.log(max(P[k, 1] for k in B) + 1e-3) + np.log(max(P[k, 0] for k in A) + 1e-3))
        s_a_via, s_b_via = (0.0, 0.0) if valid else (p_a, p_b)     # hợp lệ: chỉ xét dấu hiệu trong câu
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
            if NEW_FRAME_FIX:
                l = re.sub(r"\bnguoi nhan\b", "nguoinhan", l)    # "người nhận ở X": chữ "nhận" không phải "nhận hàng" (ghé)
            # động từ gần tên địa điểm nhất quyết định
            mv = [m.end() for m in re.finditer(VIA_V, l)]
            mg = [m.end() for m in re.finditer(GOAL_V, l)]
            if NEW_FRAME_FIX:
                if re.fullmatch(r"hang (dang )?(o|tai|nam o)", l):
                    mv.append(len(l))                          # đầu câu "Hàng ở A, người nhận ở B": nơi có hàng = ghé
                if re.search(r"\bnguoinhan (dang )?(o|tai)$", l):
                    mg.append(len(l))
            if mv and (not mg or mv[-1] > mg[-1]):
                s_a_via += 1.5 * sign
            elif mg:
                s_a_via -= 1.5 * sign
        if valid:
            # chỉ đảo kết quả của bộ phân loại khi dấu hiệu trong câu NGƯỢC LẠI và rất mạnh
            cls_a_via = any(role[k] == 1 for k in A)
            cue = s_a_via - s_b_via
            if not ((cls_a_via and cue <= -CUE_MARGIN) or (not cls_a_via and cue >= CUE_MARGIN)):
                return
            s_a_via, s_b_via = cue, 0.0
        via_e, goal_e = (A, B) if s_a_via >= s_b_via else (B, A)
        for e in ent:
            for k in e:
                role[k] = 1 if e is via_e else (0 if e is goal_e else 2)

    # ================= dự đoán =================
    def parse(self, text, present=None):
        """present: tập các loại địa điểm CÓ trên bản đồ (nếu biết). Đích / điểm ghé / mốc luôn có trên bản đồ, còn
        địa điểm gây nhiễu thì hay vắng mặt -> tên ĐÃ BIẾT mà loại không có trên bản đồ chắc chắn là gây nhiễu."""
        global NEG_SCOPE_FIX
        if CUE_TYPO_FIX:
            text = fix_cue_typos(text)
        if NEG_SCOPE_FIX and SCOPE_GUARD:
            # đợt 20: phạm vi phủ định mới ("đừng chuyển / gửi ... tới X") chỉ được dùng khi nó không biến kết quả thành vô lý:
            # đích còn lại rất khó tin (< 0,1) trong khi cách đọc cũ có đích chắc chắn (>= 0,9). Câu thật "Đừng chuyển tới X,
            # hãy mang tới Y": Y vẫn có độ tin đích 0,92-0,96 (scratch/h108_dung_scope.py) -> không bị chặn.
            r = self._parse_scoped(text, present)
            if _goal_plaus(r) < 0.1:
                NEG_SCOPE_FIX = False
                try:
                    r_old = self._parse_scoped(text, present)
                finally:
                    NEG_SCOPE_FIX = True
                if _goal_plaus(r_old) >= 0.9:
                    return r_old
            return r
        return self._parse_scoped(text, present)

    def _parse_scoped(self, text, present=None):
        t0 = _ref_canon(text, not text.isascii()) if REF_CANON else text
        if not PREP_CANON:
            return self._parse_core(t0, present)
        t1 = canon_preps(text)
        r1 = self._parse_core(t1, present)
        if not CANON_GUARD or t1 == t0:
            return r1
        # đợt 20 (CANON_GUARD): câu đã đổi giới từ chỉ được dùng khi KHÔNG làm kết quả kém hợp lý hơn câu gốc -- mất lần nhắc
        # mang vai đích, hoặc thêm địa điểm bị phủ định (trên test, đổi giới từ từng làm 3 / 3 bản đồ nó chạm tới thành
        # "không có đích" / "đích độ tin 0,99 bị phủ định": cảnh 547, 661, 822 -- kiểm tra bằng scratch/h107_scene_struct.py)
        r0 = self._parse_core(t0, present)
        return r0 if _coherence(r1) < _coherence(r0) else r1

    def _parse_core(self, text, present=None):
        self._negwhy = []
        ments, plain, plain_acc = self._mentions(text)
        self._ev = []
        dis_marks = []
        flags = [self._phrase_flags(s, a) for s, a in zip(plain, plain_acc)]
        if SEG_FLAGS:
            flags += [self._phrase_flags(s, a) for s, a in self._segs]
        # câu có nhắc địa điểm cũng có thể kèm ý gấp / dễ vỡ (hoặc là câu gấp/dễ vỡ bị nhận nhầm có địa điểm) -> xét bằng luật từ khóa
        for s in {disambiguate(*self._masked_sent(me)) for me in ments}:
            flags.append((bool(re.search(URGENT_POS, s)) and not re.search(URGENT_NEG, s),
                          bool(re.search(FRAGILE_POS, _road_free(s))) and not re.search(FRAGILE_NEG, s)))
        gp = self.goal_clf.predict_proba(self.v_g.transform([self._main_text(text)]))[0]
        text_probs = np.full(10, 1e-3)
        text_probs[self.goal_clf.classes_] = gp
        res = {"goal": None, "goal_ref": None, "via": None, "via_ref": None,
               "urgent": any(f[0] for f in flags), "fragile": any(f[1] for f in flags), "conf": 0.0,
               "goal_dist": text_probs / text_probs.sum(), "via_dist": None, "goal_known": False, "via_known": False,
               "excluded": set(), "excluded_via": set(), "goal_anchor_dist": None, "via_anchor_dist": None}
        if CALM_WINS:
            # câu nói rõ "không gấp" thắng một câu khác chỉ được MÔ HÌNH NGHĨA đoán là gấp (không có từ khóa); tương tự dễ vỡ
            n_plain = len(plain) + (len(self._segs) if SEG_FLAGS else 0)
            if res["urgent"] and any(e[2] for e in self._ev) and not (any(e[0] for e in self._ev) or any(f[0] for f in flags[n_plain:])):
                res["urgent"] = False
            if res["fragile"] and any(e[3] for e in self._ev) and not (any(e[1] for e in self._ev) or any(f[1] for f in flags[n_plain:])):
                res["fragile"] = False
        if not ments:
            res["goal"] = PLACE_TYPES[int(np.argmax(text_probs))]
            return res
        ctx = [self._ctx(me) for me in ments]
        X = hstack([self.v_l.transform(["L " + c[0] for c in ctx]), self.v_r.transform(["R " + c[1] for c in ctx]),
                    self.v_s.transform([c[2] for c in ctx])]).tocsr()
        P = self.role_clf.predict_proba(X)
        if ROLE_E5_W > 0 and hasattr(self, "e5_role") and getattr(self, "use_e5", False):
            Pe = np.full((len(ments), 4), 1e-3)
            Pe[:, self.e5_role.classes_] = self.e5_role.predict_proba(embedder().encode([self._role_text(me) for me in ments], prefix="query: "))
            Q = P * (Pe + 0.02) ** ROLE_E5_W
            P = Q / Q.sum(1, keepdims=True)
        kindP = self.kind_clf.predict_proba(self.v_k.transform([c[3] for c in ctx]))
        role = P.argmax(1)
        ablate = getattr(self, "ablate_via", False)       # thí nghiệm: giả vờ bộ phân loại không biết khung câu "điểm ghé" nào
        if ablate:
            P = P.copy(); P[:, 1] = 0.25; P[:, 0] = 0.25
            role[role == 1] = 0
        # ---- luật tổng quát (kiến thức tiếng Việt) bổ sung cho khung câu lạ ----
        fz = lambda s: " ".join(self._fix_seq(s.split()))
        negs = [False] * len(ments)
        for k, me in enumerate(ments):
            w, i, j = me["toks"], me["i"], me["j"]
            cl = None
            if NEW_FRAME_FIX:
                w = self._fix_seq(w, me["acc"])               # "khong hge X", "Đừng mnag tới X", "hông phải X": chữ then chốt gõ sai
                seg = " ".join(self._fix_seq(me["seg"].split(), me.get("seg_acc")))
                sent = " ".join(self._fix_seq(me["sentp"].split(), me.get("sentp_acc")))
                cl = self._contrast_clause(me, w)
                if cl is not None:                            # "Không phải A mà là B", "tới A chứ không phải B": phủ định chỉ trong vế
                    raw = me["sent"].split()                  # (giữ "dungx": "đúng / dừng" không phải "đừng")
                    seg = " ".join(self._fix_seq(raw[cl[0]:cl[1]], me["acc"][cl[0]:cl[1]]))
            else:
                seg, sent = fz(me["seg"]), fz(me["sent"])     # "khng", "khogn"... -> "khong"
            seg_raw = seg
            seg, sent = _strip_flags(seg), _strip_flags(sent)  # "..., không cần vội": nói về hàng, không phủ định địa điểm
            if VIA_FIX and VIA_MASK:
                # che chữ của các tên ĐÃ BIẾT khi xét phủ định: "Điểm lấy hàng: nhà thi ĐẤU." ("đấu" không dấu ~ "đâu").
                # Không che tên lạ: ranh giới của nó có thể nuốt cả chữ phủ định ("dãy phòng nội trú không phải điểm")
                for m in me["ms"]:
                    a = " ".join(me["toks"][m[0]:m[1]]) if m[2] is not None else ""
                    if a:
                        pat = r"(?<![a-z])" + re.escape(a) + r"(?![a-z])"
                        seg, sent = re.sub(pat, "x", seg), re.sub(pat, "x", sent)
            if REF_NEG_FIX:                               # "X không phải cái gần Y", "X không xa Y": phủ định THAM CHIẾU, không phủ định X
                seg, sent = re.sub(NEG_REL, " ", seg), re.sub(NEG_REL, " ", sent)
            if REF_NEG_DIR:                               # "X không nằm ở phía bắc": phủ định HƯỚNG, X vẫn là đích
                seg, sent = re.sub(NEG_DIR_PHRASE, " ", seg), re.sub(NEG_DIR_PHRASE, " ", sent)
            neg = re.search(_neg(), seg) or (re.search(_neg(), sent) and "," not in me["seg"] and len(me["ms"]) == 1)
            if FRAME2_FIX and not neg and re.search(PLAN_CHANGE, " ".join(w[:i])) \
                    and re.search(r"\b(nhung|ma)\b( \w+){0,4} (doi|chuyen|thay)\b", " ".join(w[j:])):
                neg = True                                    # "Lúc đầu định giao tới D nhưng giờ đổi sang G": D là kế hoạch cũ
            if NEW_FRAME_FIX and not neg and re.search(AVOID_LEFT2 if VIA_FIX else AVOID_LEFT, " ".join(w[max(0, i - 3):i])):
                neg = True                                    # "né / tránh X", "tránh đường qua X": không phải điểm dừng
            if VIA_FIX and VIA_KHOI and not neg and re.search(r"\bkhoi (can |phai )?(den|toi|qua|ghe|vao|di|giao|mang|dua|tat|re)\b", seg_raw):
                neg = True                                    # "Khỏi cần đến X" ("khỏi cần" bị _strip_flags gỡ như "khỏi cần nhẹ tay")
            if VIA_FIX and VIA_CHO and not neg and re.search(r"\bchớ (ghé|qua|tới|đến|vào|đi|giao|mang|đưa|dừng)\b", " ".join(me["acc"]).lower()):
                neg = True                                    # "Chớ ghé X" (chỉ khi có dấu: "cho ghe" không dấu ~ "chỗ ghé: X")
            negs[k] = bool(neg)
            if DEBUG_NEG and neg:
                why = "seg" if re.search(_neg(), seg) else ("sent" if re.search(_neg(), sent) else "khác (né / khỏi / chớ / kế hoạch cũ)")
                alts = _top_alts(_neg())
                hit = next((a_ for a_ in alts if re.search(a_, seg if why == "seg" else sent)), None)
                self._negwhy.append((k, why, hit))
            if neg and role[k] != 3:
                role[k] = 2                                   # câu phủ định / chuyện đã qua -> gây nhiễu
                continue
            if cl is not None and role[k] in (2, 3) and re.search(
                    r"\b(giao|mang|dua|gui|chuyen|dem|cho)( hang| do)? (o|toi|den|tai|sang|ve|qua|cho)$|\bla$",
                    " ".join(w[max(cl[0], i - 4):i])):
                role[k] = 0                                   # "Không giao ở Y mà giao ở X", "Không phải Y mà là X": vế sau là đích
            # (luật "mốc không có từ quan hệ -> đích" bị BỎ: tên món hàng "thẻ thư viện" được bộ phân loại gán mốc để vô hiệu,
            #  luật đó biến nó thành đích — validation học train 1,0000 -> 0,9983. Chỉ giữ các khung tường minh dưới đây.)
            if NEW_FRAME_FIX and role[k] in (1, 2) and not neg and not ablate and j < len(w) and w[j] == "sau" \
                    and not (j + 1 < len(w) and w[j + 1] in ("khi", "do", "cung", "day", "nay", "nua")):
                role[k] = 0                                   # "A trước, B sau": B là chặng sau = đích
            if NEW_FRAME_FIX and not ablate:                  # khung nhãn tường minh: áp cho mọi vai trò (kể cả đang bị gán mốc)
                lab = " ".join(w[max(0, i - 6):i])
                if FRAME2_FIX:                                # "X là điểm cuối" / "V là điểm dừng đầu tiên" (nhãn đứng SAU tên)
                    rgt = " ".join(w[j:j + 6])
                    if re.match(LEG_GOAL_RIGHT, rgt):
                        role[k] = 0
                        continue
                    if re.match(LEG_VIA_RIGHT, rgt):
                        role[k] = 1
                        continue
                if re.search(GOAL_LABEL, lab) or re.search(GOAL_LABEL2, lab):
                    role[k] = 0                               # "Nơi nhận: X", "Điểm đến cuối cùng: X", "chặng cuối X", "sau đó X"
                    continue
                if re.search(VIA_LABEL, lab) or re.search(VIA_LABEL2, lab):
                    role[k] = 1                               # "Điểm ghé: X", "Nơi lấy hàng: X", "Chặng đầu X"
                    continue
                if PICKUP_RULE and re.search(r"\b(den|toi|qua|vao|sang|ra)$", lab) and re.match(r"(lay|nhan|lanh)\b", " ".join(w[j:j + 2])) \
                        and not (NHAN_VIEN_FIX and w[j:j + 2] == ["nhan", "vien"]) \
                        and not re.search(r"\b(nguoi nhan|nguoi can|khach|minh|toi|ban ay|ho|em|anh|chi|thay|co)( ay)? (se|dang|tu|co the|muon|sap)\b",
                                          " ".join(w[max(0, i - 8):i])):
                    role[k] = 1                               # "Đến X lấy đồ cho ...": nơi ROBOT lấy hàng = điểm ghé
                    continue                                  # ("Người nhận sẽ đến X lấy hàng": X là đích, không áp dụng)
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
                if re.search(VIA_LEFT2 if NEW_FRAME_FIX else VIA_LEFT, left) \
                        or (re.search(r"\b(lay|nhan|lanh)\b", far_left) and re.match(r"(xong|roi thi|truoc)\b", right)) \
                        or (re.match(r"truoc\b(?! khi)", right) and not (FRAME_STRONG and re.match(TRUOC_TIME, right))):
                    role[k] = 1                               # "ghé X", "lấy ... ở X xong", "X trước" -> điểm ghé
            elif VIA_NOT_ANCHOR and role[k] == 3 and not neg and not ablate \
                    and re.search(VIA_LEFT2 if NEW_FRAME_FIX else r"\b(ghe|ghe qua|tat qua|tat vao|re qua|tat ngang)$",
                                  " ".join(w[max(0, i - 4):i])):
                role[k] = 1                                   # "..., nhớ ghé X": ngay sau động từ ghé thì không thể là mốc gần / xa
            elif NEW_FRAME_FIX and role[k] == 3 and not neg and not ablate \
                    and re.search(r"\b(sang|toi|den|ve|di|vao|qua|giao|mang|dua|gui|dem|chuyen)$", " ".join(w[max(0, i - 3):i])) \
                    and not re.search(r"\b(gan|xa|sat|canh|ke|cach|voi|giap|hon)\b", " ".join(w[max(0, i - 2):i])):
                role[k] = 0                                   # "rồi đem sang X", "xong mới đi X": sau động từ chuyển động thì không phải mốc
            if NEW_FRAME_FIX and role[k] in (0, 2) and not neg and not ablate and (
                    re.search(r"\b(lay|nhan|mang|dua|chuyen|di)( hang| do)? tu$", " ".join(w[max(0, i - 3):i])) or (i == 1 and w[0] == "tu")):
                role[k] = 1                                   # "Từ X mang hàng sang Y", "lấy hàng từ X": nơi lấy hàng = điểm ghé
            if VIA_FIX and VIA_ROLE and role[k] in (0, 2, 3) and not neg and not ablate:
                nxt = " ".join(w[j:j + 8])
                if i > 0 and w[i - 1] == "tu" and re.search(r"\b(sang|toi|den|ve|qua)\b", nxt) \
                        and any(x[0] > j for x in me["ms"]):
                    role[k] = 1                               # "Chở thêm đồ TỪ X SANG Y": nơi đi = nơi lấy hàng
                elif re.search(STORED_AT, " ".join(w[max(0, i - 7):i])) and len(ments) >= 2:
                    role[k] = 1                               # "Đồ đang để ở X, nhờ mang sang Y": nơi đồ đang nằm = điểm ghé
                elif re.search(r"\b(hai|2) chang\b", " ".join(w[:i])) and j < len(w) and w[j] == "va" \
                        and any(x[0] == j + 1 for x in me["ms"]):
                    role[k] = 1                               # "Đơn gồm hai chặng: X và Y": chặng thứ nhất
                elif re.search(r"\b(hai|2) chang\b", " ".join(w[:i])) and i > 0 and w[i - 1] == "va" \
                        and any(x[1] == i - 1 for x in me["ms"]):
                    role[k] = 0                               # ... chặng thứ hai = đích (không phải mốc sau "và")
        if DIS_RULE:
            # Địa điểm đứng RIÊNG trong một câu có ý phủ định / quá khứ, câu không có động từ ghé và không có từ chỉ thứ tự
            # -> không phải điểm dừng (gây nhiễu viết theo cách mới). Kiểm tra trên train + validation: scratch/h13_dis_rule.py
            for k, me in enumerate(ments):
                if negs[k] or role[k] == 3 or len(me["ms"]) != 1:
                    continue
                st = fz(me["sent"])
                if NEW_FRAME_FIX:
                    st = " ".join(self._fix_seq(me["sent"].split(), me["acc"]))
                    st = _strip_flags(st)                     # "X cần hộp giấy, không gấp lắm": "không" nói về độ gấp
                if VIA_FIX and VIA_MASK and me["type"] is not None:        # che chữ của tên đã biết: "Điểm lấy hàng: nhà thi ĐẤU" ("đấu" ~ "đâu")
                    a = " ".join(me["toks"][me["i"]:me["j"]])
                    if a:
                        st = re.sub(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", "x", st)
                if DIS_UNAVAIL:
                    st = re.sub(TIME_TRUOC, " ", st)          # "từ TUẦN TRƯỚC", "lần trước": thời gian, không phải "X trước" (điểm ghé)
                if re.search(ORD_CUE, st) or re.search(VIA_VERB, st) or re.search(DOUBLE_NEG, st):
                    continue
                # (chữ báo "không hoạt động" phải nằm NGOÀI tên: "phòng hội nghị", "khu nghỉ ngơi" là tên)
                st_u = re.sub(r"(?<![a-z])" + re.escape(" ".join(me["toks"][me["i"]:me["j"]])) + r"(?![a-z])", "x", st) if DIS_UNAVAIL else st
                if re.search(GEN_NEG, st) or re.search(PAST_CUE, st) or (DIS_UNAVAIL and re.search(UNAVAIL_CUE, st_u)
                                                                         and not re.search(r"\bhoi nghi\b|\bnghi (ngoi|le|tet|he|trua)\b", st_u)):
                    dis_marks.append(k)
            # không đánh dấu nếu làm mất hết ứng viên đích
            keep = [k for k in range(len(ments)) if not negs[k] and role[k] != 3 and k not in dis_marks]
            if DEBUG_NEG:
                self._negwhy += [(k, "DIS_RULE", None) for k in dis_marks] if (keep or DIS_RULE == "force") else []
            if keep or DIS_RULE == "force":
                for k in dis_marks:
                    role[k] = 2
                    negs[k] = True
        if PRESENT_RULE and present is not None:
            for k, me in enumerate(ments):
                if DEBUG_NEG and me["type"] is not None and me["type"] not in present:
                    self._negwhy.append((k, "PRESENT_RULE (loại không có trên bản đồ)", me["type"]))
                if me["type"] is not None and me["type"] not in present:
                    role[k] = 2
                    negs[k] = True
        if ROLE_LM_THR < 1.0 and ments:
            # (đợt 18) bộ phân loại vai ĐỌC CẢ CÂU (src/role_lm.py, e5-small + LoRA): khi nó RẤT chắc mà khác vai do luật /
            # bộ phân loại n-gram chọn -> theo nó (rồi các ràng buộc cấu trúc bên dưới vẫn áp dụng)
            Pl = _role_lm().predict([role_lm.marked_input(me, text) for me in ments])
            res["_lm"] = [round(float(x.max()), 3) for x in Pl]
            for k in range(len(ments)):
                q = int(Pl[k].argmax())
                if Pl[k, q] >= ROLE_LM_THR and q != role[k]:
                    if q == 2 or q == 3:
                        negs[k] = negs[k] or q == 2
                    elif negs[k]:
                        negs[k] = False
                    role[k] = q
        if ANCHOR_NEEDS_REL:
            # mốc gần / xa LUÔN đứng ngay sau từ quan hệ (258/258 mốc train + validation): không có thì không phải mốc
            for k, me in enumerate(ments):
                if role[k] == 3 and not any(x in ("gan", "xa", "sat", "canh", "ke", "cach", "giap")
                                            + (("phia", "huong", "kia", "voi", "tu", "man") if REF_TOWARD else ())
                                            for x in me["toks"][max(0, me["i"] - 4):me["i"]]):
                    # tên QUEN (chắc là địa điểm): theo vai trò khả dĩ nhất còn lại; cụm LẠ (có thể chẳng phải địa điểm): trung tính
                    if me["type"] is not None:
                        # (đã bị phủ định / "né X": gây nhiễu -- chuỗi kiểm tra v33c, h27 "Nhớ né {X} ra." 1500 -> 1369)
                        role[k] = 2 if negs[k] else int(np.argmax(P[k, :3]))   # cụm LẠ: giữ nguyên (có thể chẳng phải địa điểm; vai mốc = trung tính)
        self._structure(ments, role, negs, P)
        if GOAL_GUARD and ments and not any(role[k] == 0 for k in range(len(ments))):
            # đợt 20: MỌI yêu cầu đều có nơi giao (DE_BAI mục 4). Luật phủ định đã gạt hết mọi lần nhắc mà bộ phân loại vai
            # vẫn RẤT chắc một lần nhắc là đích (>= 0,9; vd "đừng tới X muộn", "X ..., không liên quan ...") -> nhận lần nhắc
            # đó làm đích thay vì đoán loại đích từ cả câu. (Train + validation: không cảnh nào hết đích.)
            cand = [k for k in range(len(ments)) if role[k] != 3 and P[k, 0] >= 0.9
                    and not (present is not None and ments[k]["type"] is not None and ments[k]["type"] not in present)]
            if cand:
                k = max(cand, key=lambda q: P[q, 0])
                role[k] = 0
                negs[k] = False
        if GOAL_PLAUS_GUARD and ments:
            # đợt 20 (h109): đích còn lại RẤT khó tin (độ tin vai đích < 0,1) trong khi một lần nhắc bị luật phủ định gạt đi lại
            # rất chắc là đích (>= 0,9): luật phủ định nhiều khả năng bắt nhầm ("đừng tới X muộn", "đừng để X chờ"). Câu gây
            # nhiễu thật ("Đừng chuyển tới D, hãy mang tới G") thì G vẫn có độ tin >= 0,9 -> không áp dụng. Validation: 0 cảnh.
            gk = [k for k in range(len(ments)) if role[k] == 0]
            if gk and max(P[k, 0] for k in gk) < 0.1:
                cand = [k for k in range(len(ments)) if negs[k] and role[k] == 2 and P[k, 0] >= 0.9
                        and not (present is not None and ments[k]["type"] is not None and ments[k]["type"] not in present)]
                if cand:
                    k = max(cand, key=lambda q: P[q, 0])
                    role[k] = 0
                    negs[k] = False
                    for q in gk:
                        role[q] = 1 if P[q, 1] >= P[q, 2] else 2
        dist = [None] * len(ments)
        for k, me in enumerate(ments):
            if me["type"] is not None:
                d = np.full(10, 1e-4); d[PLACE_TYPES.index(me["type"])] = 1.0
            else:
                d = self.span_probs(me["text"], " ".join(me["acc"][me["i"]:me["j"]]))
                if getattr(self, "use_e5", False) and hasattr(self, "e5_ctx") and getattr(self, "ctx_weight", CTX_W) > 0:
                    pc = np.full(10, 1e-3)
                    pc[self.e5_ctx.classes_] = self.e5_ctx.predict_proba(embedder().encode([self._ctx_text(me)]))[0]
                    d = d * (pc + 0.02) ** getattr(self, "ctx_weight", CTX_W)
                if CTX_NG_W > 0 and hasattr(self, "cx_clf"):
                    px_ = np.full(10, 1e-3)
                    px_[self.cx_clf.classes_] = self.cx_clf.predict_proba(self.v_cx.transform([_unaccent(self._ctx_text(me))]))[0]
                    # CTX_NG_LOWCONF: chỉ dùng khi tên gọi tự nó không đủ chắc (không có từ khóa quen)
                    if not CTX_NG_LOWCONF or (d / d.sum()).max() < CTX_NG_LOWCONF:
                        d = d * (px_ + 0.02) ** CTX_NG_W
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

        # chẩn đoán (không đổi kết quả): vai trò cuối cùng của từng lần nhắc
        res["_roles"] = [("quen" if me["type"] is not None else "lạ", int(role[k]), bool(negs[k]), round(float(P[k, 0]), 2))
                         for k, me in enumerate(ments)]
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
            d = d * (res["goal_dist"] ** GOAL_TEXT_W)
            res["goal_known"] = any(ments[k]["type"] is not None and (role[k] == 0 or k == g) for k in range(len(ments)))
            if VIA_EXCLUDES_GOAL and not res["goal_known"]:
                # đích và điểm ghé không bao giờ cùng loại (0 / 781 cảnh train + validation có điểm ghé)
                for k in range(len(ments)):
                    if role[k] == 1 and ments[k]["type"] is not None:
                        d[PLACE_TYPES.index(ments[k]["type"])] *= 1e-3
            res["goal_dist"] = d / d.sum()
        else:
            res["goal_from_text"] = True        # chẩn đoán: không có lần nhắc nào làm đích -> loại đích lấy từ bộ phân loại cả câu
            d = res["goal_dist"].copy()
            for t in res["excluded"]:
                d[PLACE_TYPES.index(t)] *= 1e-3
            if VIA_TYPE_FIX:
                # KHÔNG tìm thấy tên đích, chỉ có điểm ghé (tên đã biết): bộ phân loại cả câu hay đoán đích = loại của chính điểm
                # ghé (tên nó nằm trong câu) rồi điểm ghé bị bỏ vì "cùng loại đích". Đích và điểm ghé không bao giờ cùng loại.
                for k in range(len(ments)):
                    if role[k] == 1 and ments[k]["type"] is not None:
                        d[PLACE_TYPES.index(ments[k]["type"])] *= 1e-6
            res["goal_dist"] = d / d.sum()
        res["goal"] = PLACE_TYPES[int(np.argmax(res["goal_dist"]))]
        # ---- điểm ghé ----
        via_c = [k for k in range(len(ments)) if role[k] == 1 and not (ments[k]["type"] is not None and ments[k]["type"] == res["goal"])]
        v = max(via_c, key=lambda k: (ments[k]["type"] is not None, P[k, 1])) if via_c else None
        if SUBJ_TRIM and v is not None and ments[v]["type"] is None and res["goal_known"]                 and dist[v][PLACE_TYPES.index(res["goal"])] > 0.9:
            # (đợt 18) điểm ghé là tên LẠ mà chắc chắn cùng loại với đích đã biết: đích và điểm ghé không bao giờ cùng loại
            # (0 / 781 cảnh) -> đó là CÙNG một nơi ("phòng ăn giáo viên tầng trệt ở bếp ăn đang chờ"): không có điểm ghé
            v = None
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
                # (đợt 18) lần nhắc tên QUEN bị gán điểm ghé nhưng cùng loại đích (đích và điểm ghé không bao giờ cùng loại:
                # 0/781 cảnh) là CÙNG một nơi với đích: "Nơi ăn nhẹ ở canteen nằm phía đông" -> tham chiếu thuộc đích
                same_goal = SUBJ_TRIM and key == "goal" and role[k] == 1 and ments[k]["type"] is not None                     and ments[k]["type"] == res["goal"] and not negs[k]
                if not (k == main or same_goal or (role[k] == role_i and (ments[k]["type"] is None or ments[k]["type"] == res[key]))):
                    continue
                kk = int(np.argmax(kindP[k]))
                kind, pr = (KINDS[kk], float(kindP[k, kk])) if kk else (None, 0.0)
                if kind is not None and (pr < 0.7 or "X" in ctx[k][3].split()):
                    kind, pr = None, 0.0        # bộ phân loại không chắc, hoặc ngữ cảnh có từ lạ -> để luật từ khóa quyết định
                if DIR_GUARD and kind in ("north", "south", "east", "west") and not (self._dir_hint(ctx[k][3]) and (not DIR_GUARD2 or self._dir_hint(ctx[k][3], core=True)
                                                                                      or not re.search(r"(phia|mat|ben|dang) (truoc|sau)", ctx[k][3]))):
                    kind, pr = None, 0.0        # hướng mà ngữ cảnh không có từ chỉ hướng / vị trí nào (kể cả gõ sai)
                # luật từ khóa trên các từ GỐC ngay sau tên địa điểm (bộ phân loại không hiểu từ chỉ hướng lạ như "mạn dưới")
                me = ments[k]
                tk = me["toks"]
                if NEG_TYPO_REF:
                    # đợt 20: chữ khung gõ sai / viết tắt ("khôg xa", "khng gần", "ko xa", "hông gần") -> dạng chuẩn trước luật
                    fx = self._fix_seq(list(me["toks"]), me["acc"])
                    tk = list(me["toks"])
                    # CHỈ nhận chữ được sửa thành "không" (bộ sửa lỗi gõ theo cặp từ có thể đổi chữ đúng: "bên phía" -> "bên phải")
                    tk = [("khong" if (fx[q_] == "khong" or (x in NEG_ABBR and q_ + 1 < len(tk) and tk[q_ + 1] in
                                                             ("xa", "gan", "sat", "canh", "ke", "cach", "o", "nam", "qua", "may", "bao")))
                           else x) for q_, x in enumerate(tk)]
                raw, q, post = [], me["j"], ""
                while q < len(tk) and len(raw) < (6 if CACH_RULE else 5):
                    hit = next((x for x in me["ms"] if x[0] == q), None)
                    if hit:
                        raw.append("PLC")
                        if hit[1] < len(tk) and tk[hit[1]] == "hon":
                            raw.append("hon")
                        post = " ".join(tk[hit[1]:hit[1] + 5])
                        break
                    raw.append(tk[q]); q += 1
                rk = rule_direction(" ".join(raw), post)
                dirs4 = ("north", "south", "east", "west")
                if rk is not None and (kind is None or pr < 0.9 or (DIR_RULE_WINS and rk in dirs4 and kind in dirs4 and rk != kind)
                                       or (NEW_FRAME_FIX and rk in ("near", "far") and kind in dirs4)
                                       # đợt 20: "KHÔNG ở gần Y" = xa Y: luật hiểu phủ định, bộ phân loại n-gram thì không
                                       or (REF_NEG_FIX and rk in ("near", "far") and kind in ("near", "far") and rk != kind
                                           and (re.search(NEG_REL, " ".join(raw)) or ("cach" in raw and re.match(NEG_POST, post))))):
                    # "kế bên / gần phía Y hơn" + mốc: luật thấy từ quan hệ ngay trước mốc -> thắng hướng do bộ phân loại đoán từ "bên"/"phía"
                    kind, pr = rk, max(pr, 0.6)     # "phía nam" viết rõ thắng bộ phân loại đọc "north" (0.9)
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
        if DETACHED_REF and res["goal"] is not None and res["goal_ref"] is None and not res["via"]:
            # (đợt 20, THỬ) tham chiếu TÁCH RỜI tên, trong câu / vế KHÔNG có địa điểm: "Lưu ý chọn cái phía bắc", "lấy bản ở phía nam"
            for s in plain:
                m = re.search(r"\b(cai|cho|ban|noi|toa|khu|dia diem)( (o|nam|ma|dang))? (phia|ben|man|huong|goc|mien|mep|ria|nua) "
                              r"(bac|nam|dong|tay|tren|duoi|trai|phai)\b", s)
                if m and not re.search(GEN_NEG, s[:m.start()]):
                    kd = {"bac": "north", "tren": "north", "nam": "south", "duoi": "south", "dong": "east", "phai": "east",
                          "tay": "west", "trai": "west"}[m.group(5)]
                    res["goal_ref"] = (kd, None)
                    break
        return res

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


ROLE_LM_THR = 1.0    # đợt 18: ngưỡng tin bộ phân loại vai đọc cả câu (1.0 = tắt)
ROLE_LM_PATH = "cache/role_lm_final.pt"
_ROLE_LM = None


def _role_lm():
    global _ROLE_LM
    if _ROLE_LM is None:
        import sys as _sys
        if "F:/pylibs" not in _sys.path:
            _sys.path.append("F:/pylibs")
        _ROLE_LM = role_lm.RoleLMPredictor(ROLE_LM_PATH)
    return _ROLE_LM


ROLE_E5_W = 0.0     # đợt 18: trọng số bộ phân loại VAI bằng nghĩa (e5) nhân vào bộ phân loại n-gram (0 = tắt)
ROLE_E5_C = 5.0
CTX_NG_W = 0.0      # trọng số manh mối ngữ cảnh khớp chính xác từ ngữ (0 = tắt)
CTX_NG_LOWCONF = 0  # nếu > 0: chỉ áp dụng khi độ chắc của tên gọi < ngưỡng này
CTX_W = 0.8         # trọng số manh mối NGỮ CẢNH (món hàng, người nhận quanh tên lạ) khi đoán loại tên lạ: phép đo khắt khe 0.9851 -> 0.9876
                    # 0.5 -> 0.8 (06/10): tên hoàn toàn mới trong câu (h28) 0,956 -> 0,958 / 0,921 -> 0,927; giấu tên công bằng giữ 0,9983
GOAL_TEXT_W = 0.3   # trọng số của bộ phân loại "cả câu -> loại đích" (món hàng, người nhận...) khi chốt loại đích
REACH_RULE = True   # đích / điểm ghé phải TỚI ĐƯỢC (đúng 2300/2300 cảnh train + validation); không thì chọn ứng viên kế tiếp
CLEAR_REF = True    # đợt 16: tham chiếu luôn RÕ RÀNG (DE_BAI mục 4; đúng 100% train + validation): bắc/nam -> hai bản lệch >= 2 hàng,
                    # đông/tây -> >= 2 cột, gần/xa X -> X chỉ có 1 bản và khoảng cách tới X chênh >= 1 ô. Dùng để chốt loại của
                    # TÊN LẠ (đích / điểm ghé / mốc) sao cho tham chiếu rõ ràng trên bản đồ (scratch/h41_consistency.py)
NOREF_PRIOR = False  # đợt 16: câu KHÔNG có tham chiếu mà loại có 2 bản phân biệt được bằng hướng: chỉ 32/198 câu validation (16%)
                    # bỏ tham chiếu -> tên lạ ít khả năng là loại đó (Bayes, scratch/h42_noref_prior.py). TẮT: tên tự soạn +0,26%
                    # nhưng validation thật 1,0000 -> 0,9990 (cảnh 144: "phng bác sĩ" 0,50 bị lật sang loại 1 bản 0,15)
NOREF_W = {"one": 1.0, "none": 1.0, "anc": 0.3, "sep": 0.16}
ANCHOR_EXCL_FIX = True  # đợt 16: mốc gần/xa của điểm ghé KHÔNG bị loại khỏi ứng viên đích (chỉ mốc của đích mới khác loại đích)
JOINT_GV = True         # đợt 16: đích lạ + điểm ghé lạ -> chọn cặp loại (khác nhau) có tích xác suất lớn nhất


def _ref_ok(L, t, kind, a):
    """Tham chiếu (kind, mốc a) có RÕ RÀNG với loại t trên bản đồ L không (theo DE_BAI mục 4)."""
    ps = L.get(t) or []
    if len(ps) != 2:
        return False
    (r1, c1), (r2, c2) = ps
    if kind in ("north", "south"):
        return abs(r1 - r2) >= 2
    if kind in ("east", "west"):
        return abs(c1 - c2) >= 2
    A = L.get(a) or [] if a is not None else []
    if len(A) != 1 or a == t:
        return False
    (ar, ac), = A
    return abs(math.hypot(r1 - ar, c1 - ac) - math.hypot(r2 - ar, c2 - ac)) >= 1 - 1e-9


def _ref_after(words, e):
    """Ngay từ vị trí e là một tham chiếu không gian: "(nằm / ở) gần / xa / sát / cạnh / kế / cách ...", "(nằm / ở) phía / mạn + hướng"."""
    n, q = len(words), e
    while q < n and q - e < 2 and words[q] in ("nam", "o"):
        q += 1
    if q < n and words[q] in ("gan", "xa", "sat", "canh", "ke", "cach"):
        return True
    return q + 1 < n and words[q] in ORI_WORDS and words[q + 1] in DIRS


def _pair_kind(L, t):
    """'one' (1 bản) | 'sep' (2 bản lệch >= 2 hàng hoặc cột) | 'anc' (chỉ phân biệt được bằng gần/xa) | 'none'."""
    ps = L.get(t) or []
    if len(ps) < 2:
        return "one"
    if len(ps) > 2 or any(_ref_ok(L, t, k, None) for k in ("north", "east")):
        return "sep"
    return "anc" if any(_ref_ok(L, t, "near", u) for u in L if u != t) else "none"


def resolve_with_map(m, landmarks, world=None):
    """Chốt loại đích / điểm ghé / mốc bằng cách chỉ xét các loại CÓ trên bản đồ (landmarks: {loại: [vị trí]}).
    Trả về mission dạng mà strategy.py dùng."""
    present = [t for t in PLACE_TYPES if landmarks.get(t)]
    out = {"goal": m["goal"], "goal_ref": m["goal_ref"], "via": m["via"], "via_ref": m["via_ref"],
           "urgent": m["urgent"], "fragile": m["fragile"]}
    if not present:
        return out
    two = {t for t in present if len(landmarks[t]) >= 2}
    one = {t for t in present if len(landmarks[t]) == 1}
    P = PLACE_TYPES.index
    fixed = set()        # tham chiếu đã được chốt cùng loại (CLEAR_REF) -> không sửa mốc lần nữa

    def choose(dist, allowed, need_two=False):
        cand = [t for t in present if t in allowed]
        if need_two and any(t in two for t in cand):
            cand = [t for t in cand if t in two]
        if not cand:
            cand = present
        return max(cand, key=lambda t: dist[P(t)])

    def choose_ref(key, dist, allowed, known_type=None):
        """CLEAR_REF: chọn cặp (loại, mốc) có xác suất lớn nhất trong các cặp làm tham chiếu RÕ RÀNG (và bản được chỉ tới phải
        tới được, nếu có world). None nếu không có cặp nào."""
        ref, ad = m[key + "_ref"], m.get(key + "_anchor_dist")
        types = [known_type] if known_type else [t for t in present if t in allowed]
        pairs = []
        for t in types:
            if ref[0] in ("near", "far"):
                if ad is None or max(ad) > 0.99:      # mốc là tên quen: giữ nguyên
                    anchors = [(ref[1], 1.0)]
                else:
                    anchors = [(u, float(ad[P(u)])) for u in one if u != t]
            else:
                anchors = [(None, 1.0)]
            for a, pa in anchors:
                if _ref_ok(landmarks, t, ref[0], a):
                    pairs.append(((1.0 if known_type else float(dist[P(t)])) * pa, t, a))
        for s, t, a in sorted(pairs, key=lambda x: -x[0]):
            if world is None or not REACH_RULE:
                return s, t, a
            from strategy import INF, q_values
            c = dict(out, goal=t, goal_ref=(ref[0], a), via=None, via_ref=None) if key == "goal" else \
                dict(out, via=t, via_ref=(ref[0], a))
            if min(q_values(world, c, 0)) < INF:
                return s, t, a
        return None

    def choose_noref(dist, allowed):
        cand = [t for t in present if t in allowed] or present
        return max(cand, key=lambda t: dist[P(t)] * NOREF_W[_pair_kind(landmarks, t)])

    excluded = set(m.get("excluded") or ())
    vr = m.get("via_ref")
    if ANCHOR_EXCL_FIX and vr and vr[0] in ("near", "far") and vr[1] and vr[1] not in (m.get("excluded_via") or ()) \
            and not (m.get("goal_ref") and m["goal_ref"][1] == vr[1]):
        # mốc của ĐIỂM GHÉ có thể chính là loại đích ("ghé cổng chính ở xa nơi giữ xe ... rồi giao tới bãi đỗ xe": val cảnh 208)
        excluded.discard(vr[1])
    via_known_type = m["via"] if m.get("via_known") else None
    goal_known = m.get("goal_known") and m["goal"] in present
    joint = False
    if JOINT_GV and not goal_known and m["via"] is not None and not (m.get("via_known") and m["via"] in present) \
            and m.get("via_dist") is not None:
        # đích lạ + điểm ghé lạ: chọn CẶP (đích, ghé) khác loại có tích xác suất lớn nhất, thỏa tham chiếu rõ ràng và tới được.
        # (trước đây chọn đích trước rồi ép điểm ghé sang loại khác: đích đoán sai kéo theo chặng đầu sai)
        def options(key, dist, allowed):
            ref, ad, opts = m[key + "_ref"], m.get(key + "_anchor_dist"), []
            if ref is not None and CLEAR_REF:
                for t in allowed:
                    if ref[0] in ("near", "far"):
                        anchors = [(ref[1], 1.0)] if ad is None or max(ad) > 0.99 else [(u, float(ad[P(u)])) for u in one if u != t]
                    else:
                        anchors = [(None, 1.0)]
                    opts += [(float(dist[P(t)]) * pa, t, (ref[0], a), True) for a, pa in anchors if _ref_ok(landmarks, t, ref[0], a)]
            if not opts:
                cand = [t for t in allowed if ref is None or t in two] or list(allowed)
                opts = [(float(dist[P(t)]), t, ref, False) for t in cand]
            return opts
        ag = [t for t in present if t not in excluded] or present
        av = [t for t in present if t not in set(m.get("excluded_via") or ())] or present
        pairs = sorted(((sg * sv, g_, gr, cg, v_, vr_, cv) for sg, g_, gr, cg in options("goal", m["goal_dist"], ag)
                        for sv, v_, vr_, cv in options("via", m["via_dist"], av) if g_ != v_), key=lambda x: -x[0])
        from strategy import INF, q_values
        for _, g_, gr, cg, v_, vr_, cv in pairs[:60]:
            c = dict(out, goal=g_, goal_ref=gr, via=v_, via_ref=vr_)
            if world is None or not REACH_RULE or min(q_values(world, c, 0)) < INF:
                out.update(goal=g_, goal_ref=gr, via=v_, via_ref=vr_)
                fixed.update(k for k, f in (("goal", cg), ("via", cv)) if f)
                joint = True
                break
    # ---- đích ----
    allowed = {t for t in present if t not in excluded and t != via_known_type} or set(present)
    b = choose_ref("goal", m["goal_dist"], allowed, m["goal"] if goal_known else None) if CLEAR_REF and m["goal_ref"] and not joint else None
    if joint:
        pass
    elif b:
        out["goal"], out["goal_ref"] = b[1], (m["goal_ref"][0], b[2])
        fixed.add("goal")
    elif not goal_known:
        if NOREF_PRIOR and m["goal_ref"] is None:
            out["goal"] = choose_noref(m["goal_dist"], allowed)
        else:
            # có tham chiếu phương hướng -> loại đích phải có 2 bản trên bản đồ
            out["goal"] = choose(m["goal_dist"], allowed, need_two=m["goal_ref"] is not None)
    # ---- điểm ghé ----
    if m["via"] is not None and not joint:
        via_known = m.get("via_known") and m["via"] in present
        if not via_known and m.get("via_dist") is None:
            out["via"], out["via_ref"] = None, None
        else:
            ex_via = set(m.get("excluded_via") or ())
            allowed = {t for t in present if t != out["goal"] and t not in ex_via} or {t for t in present if t != out["goal"]}
            b = choose_ref("via", m["via_dist"], allowed, m["via"] if via_known else None) \
                if CLEAR_REF and m["via_ref"] and (via_known or allowed) else None
            if b:
                out["via"], out["via_ref"] = b[1], (m["via_ref"][0], b[2])
                fixed.add("via")
            elif not via_known:
                if not allowed:
                    out["via"] = None
                elif NOREF_PRIOR and m["via_ref"] is None:
                    out["via"] = choose_noref(m["via_dist"], allowed)
                else:
                    out["via"] = choose(m["via_dist"], allowed, need_two=m["via_ref"] is not None)
        if out["via"] == out["goal"]:
            out["via"], out["via_ref"] = None, None
    # ---- mốc của gần/xa: phải là loại chỉ có MỘT bản trên bản đồ ----
    for key in ("goal", "via"):
        ref = out[key + "_ref"]
        if ref and ref[0] in ("near", "far") and key not in fixed:
            a = m.get(key + "_anchor_dist")
            if ref[1] not in one and a is not None:
                cand = [t for t in one if t != out[key]]
                out[key + "_ref"] = (ref[0], max(cand, key=lambda t: a[P(t)])) if cand else None
            elif ref[1] not in present:
                out[key + "_ref"] = None
    if REACH_RULE and world is not None:
        out = _make_reachable(out, m, world, present)
    return out


def _make_reachable(out, m, world, present):
    """Đề bài luôn cho đích (và điểm ghé) tới được. Nếu lựa chọn hiện tại không có đường cho robot thường thì nó sai:
    thử lần lượt các cách sửa, từ ít can thiệp nhất, và nhận cách đầu tiên có đường."""
    from strategy import INF, q_values
    ok = lambda c: min(q_values(world, c, 0)) < INF
    if ok(out):
        return out
    P = PLACE_TYPES.index
    cands = []
    # 1) bỏ tham chiếu không gian (có thể đã chọn nhầm bản trong hai địa điểm cùng loại)
    if out["goal_ref"] or out["via_ref"]:
        cands += [dict(out, goal_ref=None), dict(out, via_ref=None), dict(out, goal_ref=None, via_ref=None)]
    # 2) đổi loại điểm ghé (nếu là tên gọi lạ), theo thứ tự xác suất
    if out["via"] is not None and not m.get("via_known") and m.get("via_dist") is not None:
        for t in sorted(present, key=lambda t: -m["via_dist"][P(t)]):
            if t not in (out["via"], out["goal"]):
                cands.append(dict(out, via=t, via_ref=None))
    # 3) đổi loại đích (nếu là tên gọi lạ), theo thứ tự xác suất
    if not m.get("goal_known"):
        for t in sorted(present, key=lambda t: -m["goal_dist"][P(t)]):
            if t != out["goal"] and t not in (m.get("excluded") or ()):
                cands.append(dict(out, goal=t, goal_ref=None, via=None if out["via"] == t else out["via"]))
    # 4) bỏ điểm ghé
    if out["via"] is not None:
        cands.append(dict(out, via=None, via_ref=None))
    for c in cands:
        if ok(c):
            return c
    return out
