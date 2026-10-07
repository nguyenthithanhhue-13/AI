# Cập nhật đợt 17 (07/10, 0h-2h): v34c, sửa lỗi của v33

**v33 có lỗi**: luật "mốc phải đứng sau từ quan hệ" (`ANCHOR_NEEDS_REL`) biến địa điểm đang bị né ("Nhớ né X ra.") thành
ĐIỂM GHÉ khi bộ đọc học lại (h27 1.500 -> 1.369). Đã sửa: lần nhắc bị phủ định luôn là gây nhiễu. v33 không nên nộp.

Phép thử h48 trên **train** (`--split=train`, 2.000 câu thật, tên tự soạn) bắt được các lỗi chung (sửa tổng quát, đo trên
train / validation / câu tự soạn, không đọc test):

1. `GOAL_SLOT_STRONG`: "Điểm giao: X." -- X trọn vế luôn là địa điểm ("góc tra cứu" trước đây bị bỏ vì không mở đầu bằng phòng/khu).
2. `SUBJ_TRIM`: "Ban quản lý / Cán bộ / Nhân viên <nơi> cần nhận": bỏ phần người ("khu phòng ở", "nơi ôn bài yên tĩnh");
   "<nơi> ở cần nhận": "ở" thuộc tên; "nhà" đứng một mình không phải mốc ("... xa nhà"); "dãy phòng sinh viên ở."
3. Cụm nói về HÀNG không phải địa điểm: "trứng gà", "đĩa sứ của khoa" (có dấu hiệu dễ vỡ / gấp, không có từ khóa nơi chốn mạnh);
   "trứng" có dấu không phải "trung (tâm)". Trước đây "trứng gà" còn bị thành ĐIỂM GHÉ.
4. "cổng phía trước": "phía trước / sau" thuộc tên (không phải hướng bản đồ, không phải "X trước" = điểm ghé) (`DIR_GUARD2`).
5. Chỉ một nơi duy nhất (không còn lần nhắc nào khác) thì nơi đó là đích, không phải điểm ghé.
6. "trạm xá / bệnh xá" không bị cắt ở chữ "xá" ~ "xa" (khung "Điểm giao:").
7. Câu có dấu: chữ khung ở mép tên chỉ bị gọt khi đúng là dạng khung ("tới" chứ không phải "tối", "về" / "vẽ", "bên" / "bến") (`EDGE_ACC_FIX`).
8. "X <đuôi tên> ở Y" (cùng một nơi): "Phòng đọc yên tĩnh ở khu đọc sách mạn trái" -> tham chiếu thuộc đích.
9. Cờ (`FLAG_MORE3`): "nhạy với va đập", "không chịu được va đập", "mà vỡ là...", "chỗ xóc", "giòn", "kẻo nứt", "trong nửa tiếng",
   "không để chờ lâu được". Bộ h52 C (viết trước khi sửa; các câu sai của C đã được nhìn khi viết luật): dễ vỡ 22/32 -> 31/32.
10. Từ khóa loại (`TYPE_FIX3`): "hỗ trợ / tiếp sinh viên", "xác nhận" -> văn phòng; "khi ốm / người mệt", "nhiệt độ" -> y tế;
    "lưu trữ luận văn" -> thư viện; "thầy giáo / giảng viên / bộ môn" -> giảng đường; "đá cầu" -> thể thao. Kho tên
    `PLACES_MORE3` (121 tên kiểu mô tả chức năng, đã bỏ mọi tên trùng / gần trùng bộ h31). **Cả hai đổi 0 dòng test.**

Đã thử và không dùng: `CTX_W` 0,4 / 0,8 / 1,2 ngang nhau trên h48 -> giữ 0,8. Suy luận loại tên lạ bằng NHIỀU cảnh test cùng lúc
(mỗi tên lạ lặp lại ở nhiều cảnh; loại thật phải có trên mọi bản đồ đó): trên train với tên tự soạn lặp lại 791 -> 799/800 cảnh,
nhưng đây là "tự học không nhãn trên test" (DE_BAI 8.4) -> KHÔNG áp dụng cho test.

Chẩn đoán tổng hợp trên test (chỉ đếm): tỉ lệ có điểm ghé / tham chiếu chặng đầu / hòa giữa các hướng của 10 robot gần
validation; không cảnh nào hết đường; 10 / 1.200 bản đồ có chú giải không chắc.

# Cập nhật lượt 5 (06/10, khuya): tên gọi lạ đặt trong khung câu THẬT

## Phép thử mới: `scratch/h48_alias_swap.py`

Lấy câu validation thật, thay tên gọi của đích (tùy chọn cả điểm ghé, mốc gần/xa, địa điểm gây nhiễu) bằng tên TỰ SOẠN cùng
loại (bộ `h31` A–D, không lấy từ test), giữ bản đồ thật và nhãn thật, đo cả 10 robot. Đây là kiểu khác biệt chính của test
(khoảng 36% đích là tên gọi lạ) đặt vào đủ mọi khung câu, cờ, lỗi gõ của dữ liệu thật. Các phép thử cũ hoặc dùng tên lạ
trong câu tự soạn đơn giản (h31, h28), hoặc dùng câu lạ với tên quen (h27).

| phép đo (số cảnh đúng cả 10 robot) | v32a | v34c |
|---|---|---|
| chỉ đổi tên đích (897 câu) | 885 | 890 |
| đổi đích + điểm ghé + mốc (1.196 câu) | 1.171 | 1.186 |
| đổi cả địa điểm gây nhiễu (1.196 câu) | – | 1.186 |

Các ca còn sai chủ yếu là tên tự soạn nhập nhằng ("nơi đo nhiệt độ": phòng thí nghiệm hay trạm y tế?) hoặc câu không dấu
nhập nhằng thật ("kho luu tru tai lieu": lưu trữ / lưu trú, tài / tại).

## Lỗi hệ thống tìm được và cách sửa (đều là quy tắc chung, đo trên train / validation / câu tự soạn)

1. `CLEAR_REF`: DE_BAI mục 4 nói tham chiếu luôn RÕ RÀNG: bắc / nam thì hai bản lệch ≥ 2 hàng, đông / tây ≥ 2 cột, gần / xa X
   thì X chỉ có 1 bản và chênh khoảng cách ≥ 1 ô (đúng 100% train + validation, `scratch/h41_consistency.py`). Dùng để chốt
   loại tên lạ của đích / điểm ghé / mốc sao cho tham chiếu rõ ràng và bản được chỉ tới phải tới được.
2. `ANCHOR_EXCL_FIX`: mốc gần/xa của ĐIỂM GHÉ có thể chính là loại đích ("ghé cổng chính ở xa nơi giữ xe … rồi giao tới
   bãi đỗ xe", validation cảnh 208). Trước đây khi đích là tên lạ, loại này bị loại khỏi ứng viên đích -> đích sai, rồi điểm
   ghé bị ép sang loại khác -> chặng đầu sai.
3. `JOINT_GV`: đích lạ + điểm ghé lạ: chọn CẶP loại (khác nhau) có tích xác suất lớn nhất, thỏa tham chiếu rõ ràng và tới
   được (trước đây chọn đích trước rồi ép điểm ghé).
4. `REF_SLOT`, `FRAME_STRONG`: khung câu chắc chắn chứa tên địa điểm ("Hàng cho X: Y" — khung của validation, 16/300 câu;
   "ghé X lấy / trước"; "tới X gần Y hơn / phía bắc"; "đích đến là / nơi cần giao là X"): nhận cụm X lạ dù không mở đầu bằng
   phòng / khu / nơi. Trước đây X bị bỏ, mốc Y bị nhận làm ĐÍCH.
5. Dấu tiếng Việt khử nhập nhằng từ khung (câu có dấu): kệ / kế, tài / tại, cửa / của, kỵ / ký (`HEAD_ACC`), "nhân viên"
   không phải "nhận".
6. Ranh giới tên (`_fix_spans`): tách cụm lạ nuốt cả "sát / gần + mốc"; ghép tên quen + đuôi lạ ("phòng học lớn tầng hai",
   "phòng thực hành hóa ở xa Y"); "chỗ ở nội trú", "khu ở tập thể" (`O_QUAL`).
7. `ITEM_FIX2`: "ghé X lấy thẻ thư viện": món hàng có chữ tên địa điểm không phải điểm ghé.
8. `TRUOC_TIME`: "giao tới X trước 10 giờ" là hạn giờ, không phải "ghé X trước" (hồi quy do h27e bắt được, đã sửa).
9. `PERSON_TYPO_FIX`: khung "<người> cần nhận / đang chờ" được tìm trên chuỗi ĐÃ sửa lỗi gõ chữ khung ("cần nhn" = "cần
   nhận"); khi chữ đầu chủ ngữ gõ sai ("Ngừoi nhận đang chờ…"), bản cũ coi nó là tên người nhận lạ (thêm một "địa điểm"
   giả) và câu gấp đứng đó bị che -> mất cờ gấp. Trên test: 3 cảnh, chỉ robot 6 đổi (đếm cấu trúc, không đọc câu).
   `NHAN_VIEN_FIX`: "đến X nhân viên…" không phải "đến X NHẬN".
10. Cờ (bộ câu mới `scratch/h52_flags_fresh.py`: bộ A dùng để sửa, bộ B viết sau và giữ lại để đo):
   thêm "coi chừng làm bể", "rơi là hỏng", "có thể bị vỡ", "dằn xóc"; "đi chậm" không phải gấp, "không được đi chậm" là gấp;
   e5 không tự thêm cờ thứ hai trong một vế mà luật đã thấy cờ kia ("nâng niu giúp" ≠ gấp); e5 được dùng cho câu nói về
   TÍNH CHẤT hàng ("Hàng nhạy với va đập": e5 0,98 mà trước đây bị chặn vì câu mở đầu bằng "hàng").
   Bộ A: dễ vỡ 28/40 -> 40/40. Bộ B (giữ lại): dễ vỡ 22/32 -> 24/32, gấp 19/20, không gấp / không dễ vỡ 100%.

Đã thử và BỎ: `NOREF_PRIOR` (Bayes "câu không có tham chiếu -> ít khả năng là loại có 2 bản phân biệt được"): tên tự soạn
+0,26% nhưng validation thật 1,0000 -> 0,9990 (một tên gõ sai 0,50 bị lật) -> tắt.

## Ảnh hưởng trên test (chỉ đếm, không đọc câu)

v33 khác v32a ở 35 dòng / 11 cảnh (v34c: 45 dòng / 16 cảnh). Các sửa về cờ không đổi dòng test nào (tỉ lệ cờ test giữ nguyên 538 dễ vỡ / 438 gấp).
Mỗi lỗi hệ thống ở trên chỉ chạm 1–3 cảnh test: phần điểm còn thiếu trên bảng xếp hạng không đến từ một nguyên nhân lớn
nào mà các phép thử tự soạn tìm được.

## Kiểm tra (chuỗi đầy đủ, `checks_v33*.log`)

Chuỗi v34c (`checks_v34c.log`): h22 fair 0,9992 / norm 1,0000 / validation 1,0000; h27 vòng 1-4, lỗi gõ, h27c --v7, h27d, h27e: 100%; h27c 720/720 (sau khi sửa gộp nhầm "nhận hàng tại A, trả hàng tại B"); h36 200/200; h34 e2e 1,0000; h45 hướng 319/320, gần/xa 197/200; h48 890/897, 1.186/1.196 (cả gây nhiễu); h48 train 1.982/2.000; h52 A 100%, B dễ vỡ 29/32, C dễ vỡ 31/32.

## Lưu ý về cách chọn bài (DE_BAI mục 7)

Xếp hạng cuối dùng bài có điểm CÔNG KHAI cao nhất (bằng điểm thì lấy bài nộp sau cùng); test trộn cảnh được chấm với cảnh
không chấm; tối đa 5 lần nộp / ngày. Một bản chỉ "được tính" khi điểm công khai của nó bằng hoặc cao hơn bản tốt nhất.

# Kết quả bảng xếp hạng lượt 4 (06/10 tối)

**v31c = 0,9733** (thấp hơn v28 = v19 = 0,9778). v31c khác v28 ở 64 dòng, trong đó 44 dòng robot 7 do tắt cờ dễ vỡ chỉ
do e5 bật (`E5_POS_FRAGILE = False`). Mức tụt khớp với "gần như mọi dòng đổi nằm trong tập công khai đều từ đúng thành
sai" -> **các cờ dễ vỡ e5 tự bật trên test phần lớn ĐÚNG**: test có câu dặn dễ vỡ kiểu mới mà luật không bắt được, tỉ lệ dễ
vỡ thật của test (~44,8%) cao hơn train / validation (~39%). Lập luận dựa trên câu tự soạn (mục h32 dưới đây) đã sai ->
bật lại `E5_POS_FRAGILE = True`. Bài học: tỉ lệ cờ trên test KHÁC train / validation; đừng dùng tỉ lệ của dữ liệu có nhãn
làm "chuẩn" cho test.

Các cỡ tập công khai khớp cả ba điểm 0,9600 / 0,9778 / 0,9733: 45, 90, 135, 180... cảnh (10 dòng / cảnh).

v32a (`predictions_v32a_nlp_via_e5on.json`, CV như v28): giữ cờ e5; thêm `VIA_FIX`. So với v28: 92 dòng / 20 cảnh. Quy về
nhóm sửa (`scratch/h38_attribute.py`, tắt từng công tắc lúc đọc câu, chỉ đếm): dấu ";" là ranh giới vế (`VIA_SEMI`)
13 cảnh / 71 dòng (test có câu dùng ";", train / validation không có; trước đây `tag_tokens` bỏ mất ";" nên hai vế dính
nhau), món hàng 3 cảnh / 11 dòng, đoán loại tên mới 1 cảnh, sửa nhỏ không công tắc 3 cảnh.

# Cập nhật lượt 4 (06/10, chiều): tên gọi mới và MÓN HÀNG mới

Bảng xếp hạng vẫn 0,9778 với v28 (= v19). Ước lượng từ công thức chấm (macro accuracy 10 robot): nếu tập công khai
khoảng 360 cảnh thì còn khoảng 80 câu trả lời robot sai. v29 (sửa cờ dễ vỡ theo ngữ cảnh "cẩn thận kẻo trễ",
"đường trơn đi cẩn thận"...) **trùng v28 trên test (0 dòng khác)**, nên không nộp.

## Test "lạ" cỡ nào? (chỉ thống kê gộp đầu ra của hệ thống, không đọc câu / ảnh test)

| | validation, bộ đọc học CHỈ train | test, bộ đọc final |
|---|---|---|
| đích là tên gọi đã biết | 65,7% | 64,1% |
| câu hoàn toàn không dấu | 24,7% | 20,7% |
| độ tự tin đích < 0,999 | 4,0% | 8,2% |

Mức "tên lạ" của test giống mức validation từng có so với train, và validation đọc đúng loại 102/103 tên lạ. Nhưng cách
bộ sinh đặt tên mới (validation thêm 2-3 tên / loại so với train: "nơi phục vụ bữa trưa", "phòng bác sĩ", "chốt cổng",
"dãy phòng nội trú"...) cho thấy test cũng thêm vài tên / loại và **mỗi tên lặp lại ở hàng chục cảnh**. Vì vậy chỉ cần
đoán sai một tên là mất cả cụm cảnh (khoảng 1-1,5% điểm).

## Đoán loại tên mới: `scratch/h31_style_aliases.py` (tên tự soạn theo đúng các kiểu đặt tên của validation)

Các lỗi hệ thống tìm được (sửa dưới `TYPE_FIX`):
- khôi phục dấu sai trước khi đưa vào e5: "noi lam thu tuc" thành "nơi làm thủ **túc**" (e5 đoán ký túc xá), "khu tra cuu"
  thành "khu **trả** cứu" (đoán trạm y tế). Sửa: bảng khôi phục nhìn cả từ đứng TRƯỚC và đứng SAU, học thêm từ kho tên tự soạn;
- cắt sai ranh giới tên: "chỗ lấy thuốc" thành "chỗ" + "thuốc", "dãy phòng **ở**" bị bỏ chữ "ở", "nơi đỗ **ô tô**" (không dấu "o to")
  bị cắt ở "o", trợ từ "nha" cuối câu dính vào tên;
- tên có sẵn trong kho tự soạn ("nơi đóng học phí") vẫn bị đoán sai: thêm tra thẳng kho (`KB_BOOST`);
- chữ "sinh viên" kéo mọi tên lạ về ký túc xá ("nơi sinh viên học", "phòng tiếp sinh viên"): bỏ khỏi đầu vào mô hình học
  (từ khóa vẫn xét cả tên);
- thêm vài từ khóa: học phí, đóng dấu, đo đạc, vi sinh, huyết áp, giữa kỳ...

| bộ tên tự soạn | trước | sau |
|---|---|---|
| A (150 tên, dùng để sửa) | 0,9583 | 0,9883 |
| B (100 tên, viết sau A, dùng để sửa "sinh viên") | 0,9350 | 0,9812 |
| **C (100 tên, viết sau cùng, CHỈ đo)** | **0,9663** | **0,9750** |
| h22 fair (tên của dữ liệu bị giấu) | 0,9983 | 0,9991 |

## MÓN HÀNG mới: `scratch/h33_items.py`

Kiểm chứng trên train + validation:
- kho lời dặn rất nhỏ và cố định: 7 câu dễ vỡ, 8 câu gấp, vài câu "không gấp / không vỡ"; lời dặn luôn là câu riêng
  (0 / 1.145 câu giao hàng chứa từ gấp / dễ vỡ);
- **món hàng độc lập với cờ**: mọi món có tỉ lệ dễ vỡ quanh tỉ lệ chung 0,40 (hóa chất 0,26, máy chiếu 0,54, linh kiện
  điện tử 0,46...); dòng "Hàng: X" (221 lần) đều trung tính;
- món hàng tương quan với loại đích ("thẻ thư viện", "túi sơ cứu", "bộ dụng cụ thí nghiệm"), nên test có thể có món mới
  chứa chữ giống tên địa điểm.

Món tự nghĩ có chữ giống tên địa điểm ("sách thư viện", "thuốc y tế", "vé gửi xe"...), đích / điểm ghé ghép ngẫu nhiên:

| khung câu | đích + điểm ghé đúng: trước | sau |
|---|---|---|
| dòng "Hàng: X" | 260 / 480 | 480 / 480 |
| "giao X tới Y" | 394 / 480 | 476 / 480 |
| "Lấy X ở A xong thì..." | 456 / 480 | 480 / 480 |
| "hàng cần giao là X", "Có đơn giao X ở Y", "X phải được đưa tới Y", "... đang chờ X" | 793 / 960 (bản đầu) | 956 / 960 |

Dòng "Hàng: X" với món mới còn bật cờ nhầm 24 / 480 lần ("Hàng: bánh kem" thành dễ vỡ do e5, "Hàng: cốc thủy tinh" do luật,
"Hàng: thuốc cấp cứu" thành gấp); sau sửa 0 / 480.
Sửa (`ITEM_FIX`): dòng "Hàng: X" không có địa điểm và không mang cờ (trừ lời dặn rõ ràng "dễ vỡ", "gấp"...); tên chồng lên
cụm món hàng (sau động từ giao / lấy, trước giới từ chỉ đích; sau "hàng cần giao là"; sau "đang chờ"; chủ ngữ của
"phải được đưa tới") bị bỏ. Chữ nằm trong một tên ("phòng **giáo** vụ" ~ "giao", "khu **gửi** xe") không được coi là động từ
(bản đầu quên điều này, validation tụt 1,0000 xuống 0,9963; đã sửa, các chỉ số validation trùng hệt bản cũ).

## Tham chiếu không gian cách nói mới (`scratch/h35_spatial.py`)

Gần / xa: 240 / 240. Hướng: "ở phía bên **tay** phải" sai 12 / 12 ("tay" không dấu bị đọc là hướng "tây", mâu thuẫn với
"phải"; "bên tay trái" đúng nhờ may vì "tây" và "trái" cùng hướng) và "ở **mé** trái / phải" sai 12 / 12 ("mé" = phía).
Sửa (`SPATIAL_FIX2`, chuẩn hóa câu; hai dạng này có 0 lần trong train + validation): đông / tây 120 / 120.

## Hồi quy bắt được trong lúc làm (h27, câu thật của train + validation chỉ thay một câu con)

- "Nơi cần giao **là** cổng chính **ở** phía trên": bản đầu coi "là cổng chính" là món hàng (giữa "giao" và "ở"), mất đích.
  Sửa: "là" chặn cụm món hàng.
- "Đưa đến **tận** trạm xá": chữ "tận" bị nhận là một địa điểm (xác suất loại 0,44, vừa qua ngưỡng 0,4 của cụm một chữ
  sau khi học lại). Sửa: gọt "tận" ở mép tên.
- "Nhưng X ở phía **tây phải** được ghé trước": bản đầu của `SPATIAL_FIX2` đổi mọi "tay phai" thành "phai" (đông).
  Sửa: chỉ đổi dạng "bên tay trái / phải". (Không câu test nào bị ảnh hưởng: v31b trùng v31.)
- "Sau khi lấy đồ ở X mạn dưới **thì** mới giao": từ khóa mới "thi" (thi cử) trùng hư từ "thì" không dấu -> cụm "dưới thì
  mới" thành tên địa điểm, tham chiếu của điểm ghé thành "tây". Bỏ từ khóa "thi" và "con dau" (~ "còn đâu"). v31c trùng v31b.

## Bản v31 (`predictions_v31c_items_spatial_e5frag.json`)

v31 = v30 + tắt "e5 tự bật dễ vỡ" (`E5_POS_FRAGILE = False`) + `SPATIAL_FIX2` + các sửa hồi quy ở trên + "càng sớm càng
hay" (gấp) + "đã giao nhanh" (lời cảm ơn, không phải gấp) + "đi đường cẩn thận" (dặn đi đường, không phải dễ vỡ).
So với v28: 64 dòng / 50 cảnh, trong đó 44 dòng của robot 7. Tỉ lệ dễ vỡ dự đoán trên test 44,8% -> 38,7%.

## Cờ dễ vỡ / gấp trên test (`scratch/h30_flag_trace.py`, chỉ đếm gộp theo NGUỒN bật cờ)

Validation (bộ đọc chỉ học train): 110 / 110 cờ dễ vỡ đúng, mọi cờ do bảng cụm hoặc luật bật, e5 không tự bật lần nào.
Test: e5 tự bật dễ vỡ ở 74 cảnh (6,2%); bỏ nhóm này thì tỉ lệ dễ vỡ 38,7%, khớp train / validation (~39%), còn nếu giữ
thì là 44,8%. Con số này chỉ dùng làm gợi ý chẩn đoán, KHÔNG dùng để chọn luật. Giả thuyết đầu (món hàng mới ở dòng
"Hàng: X") SAI: sửa món hàng không đổi cờ nào trên test. Quyết định dựa trên câu tự soạn `scratch/h32_e5_flags.py`:
e5 tự bật dễ vỡ nhầm ~3% câu không dễ vỡ (6 / 200 câu trung tính, 2 / 60 câu chào / kết như "xin cảm ơn rất nhiều",
2 / 60 câu "hàng chắc" như "đồ nhựa dẻo") và bắt đúng ~10% câu dễ vỡ cách nói mới (12 / 120). Mỗi yêu cầu có nhiều câu
không dễ vỡ (chào, kết, không gấp, hàng chắc) hơn câu dễ vỡ ~5 lần, nên kỳ vọng số cờ nhầm gấp đôi số cờ đúng; xác suất
e5 không tách được hai nhóm (nhầm 0,80-0,99, đúng 0,73-0,96); trên validation e5 tự bật dễ vỡ 0 lần. -> tắt.
Lật cờ dễ vỡ đổi nước đi của robot 7 ở 59% cảnh validation; lật cờ gấp đổi robot 6 ở 68%.

# Cập nhật lượt 3 (06/10, sáng)

## Chiến thuật robot: các trọng số bị "ghim" chặt, không phải nguồn mất điểm

`scratch/h26_policy_margin.py` quét từng trọng số (giữ nguyên các trọng số khác) trên 2.300 cảnh train + validation
với bản đồ / yêu cầu đúng, tìm khoảng giá trị vẫn khớp 100% nhãn:

| robot | trọng số | đang dùng | khoảng khớp 100% |
|---|---|---|---|
| 5 | phạt rẽ / quay đầu | 3 / 30 | đúng 3 / đúng 30 (2,75 hay 3,25 đã sai) |
| 7 (dễ vỡ) | đông / rẽ / quay đầu | 6 / 1,5 / 15 | chỉ đúng giá trị đang dùng |
| 8 | rẽ phải / rẽ trái / quay đầu | 0,5 / 4,5 / 9 | chỉ đúng giá trị đang dùng |
| 2, 3 (khô), 4, 6 (đông) | | | chỉ đúng giá trị đang dùng |
| 1 | đông | 5 | 5 – 5,75 |
| 3 (mưa) | mái che | −0,8 | −0,84 – −0,80 |
| 6 (không gấp) | mái che | −0,3 | −0,35 – −0,05 |

Gần như mọi trọng số lệch một bước nhỏ là sai ngay vài dòng → đây là giá trị thật của bộ sinh dữ liệu. Ba khoảng
còn hở không phân biệt được bằng bất kỳ cảnh nào trong 2.300 cảnh, nên ảnh hưởng tới test gần như bằng 0.
**Kết luận: 2,2% điểm mất trên bảng xếp hạng nằm ở đọc ảnh hoặc đọc câu, không ở chiến thuật.**

## NLP: khung câu mới tự soạn (`scratch/h27_paraphrase.py`)

Đề mục 4: validation và test có "khung câu và cụm từ mới". So khung câu (thay tên địa điểm bằng X, món hàng bằng Y):
**219 khung của validation không có trong train** (514/1.221 câu con), gồm câu gây nhiễu mới ("Bỏ qua X, không phải
ở đó"), cụm gấp / dễ vỡ mới ("ưu tiên số một, đi ngay", "hàng kỵ va đập"), khung đích mới ("Đích đến là X, hàng cần
giao là Y", "Y phải được chuyển tới X"), khung ghé mới ("Chưa đi thẳng được: ghé X trước, rồi…"). Test nhiều khả năng
có bộ khung mới RIÊNG (tên gọi của test cũng khác hẳn validation).

Phép đo mới: lấy cảnh train + validation có câu con thuộc khung quen, thay câu con đó bằng một cách nói khác CÙNG
NGHĨA do mình tự soạn (kiến thức tiếng Việt chung, không nhìn test), giữ nguyên nhãn, đo lại. Đợt 1 (39 cách nói):
34 cách đúng 100%, 5 cách làm tụt điểm:

| cách nói mới | điểm robot (bản đồ đúng) | cả 6 trường đúng |
|---|---|---|
| "Nơi nhận: X." | 0,820 | 21% |
| "Trên đường đi nhớ ghé ngang X." | 0,932 | 79% |
| "Không giao ở X." | 0,976 | 69% |
| "X hôm nay không nhận hàng." | 0,991 | 98% |

Gỡ lỗi còn thấy các lỗi nặng hơn mà dữ liệu chưa từng lộ (train / validation luôn tách "Không cần vội." thành câu riêng):
"Giao hộp giấy tới thư viện, **không cần vội**." → đích thành canteen, vì luật phủ định cấp câu coi "không cần" là phủ
định địa điểm duy nhất trong câu. Một câu như vậy làm sai cả 10 robot.

Sửa tổng quát (`NEW_FRAME_FIX`):
1. Khung nhãn "Nơi nhận / Địa chỉ giao / Điểm đến (cuối cùng) / Người nhận ở: X" → đích; "Điểm ghé / Nơi lấy hàng: X" → ghé.
2. Từ mở đầu chung chung đứng một mình ("nơi", "khu", "phòng"...) không phải tên địa điểm; "ngang" không thuộc tên.
3. Phủ định mở rộng: "không giao / không nhận / không mang / không được đưa / khỏi qua X".
4. **Bỏ cụm gấp / dễ vỡ trước khi xét phủ định** ("không cần vội", "không quá gấp", "không lo vỡ", "đừng đi chậm"):
   chúng nói về hàng, không phủ định địa điểm. Cụm có "nhầm" luôn giữ lại (không dấu "vội" = "với": "đừng nhầm với X").
5. Động từ ghé mới: "ghé ngang", "đi ngang qua", "rẽ vào", "dừng chân / dừng lại ở"; "đừng đi chậm" là gấp.

Đợt 2 (36 cách nói mới: động từ "đem / chở / ship / trao", địa điểm đứng đầu câu, khung bị động, người nhận "đang đợi /
chờ nhận / muốn nhận", cụm gấp / dễ vỡ gộp vào câu đích, 6 kiểu tham chiếu không gian). Mọi khung đích mới đúng 100%.
Lỗi tìm ra và cách sửa (tổng quát, vẫn dưới `NEW_FRAME_FIX`):

| lỗi | ví dụ | sửa |
|---|---|---|
| chữ đa nghĩa bị đọc thành hướng | "X **phải** được ghé trước" → phía đông; "nhớ nhẹ **tay**" → phía tây | bỏ nghĩa khác ("phải được", "nhẹ tay", "trên đường", "đông người", "bác bảo vệ", "nằm gần") trước khi xét manh mối hướng; chỉ coi là gõ sai khi từ HIẾM ("được" cách "dưới" 1 chữ nhưng là từ rất quen) |
| sửa lỗi gõ quá tay | "kế bên **cổng** trường" → "kế bên **đông** trường" (mất cả mốc) | lỗi gõ của dữ liệu chỉ là đảo / rơi chữ: từ thật chỉ khác từ chỉ hướng do THAY một chữ (cổng, hầm, dãy, bãi, trạm, cuối...) không bao giờ bị "sửa" |
| cụm vị trí bị nhận là tên địa điểm | "ở **nửa trên bản đồ**", "ở **phần** phía trên", "phía trên **cùng**" | cụm toàn từ chỉ vị trí (thêm nửa / phần / cùng / tít / hẳn / vùng) không phải tên |
| luật "gần / kế bên + mốc" thua bộ phân loại | "kế bên cổng", "gần phía X hơn" → phía đông | luật thấy từ quan hệ ngay trước mốc thì thắng hướng do bộ phân loại đoán |
| phủ định chưa biết | "Nhớ **né** X ra", "**Tránh** X ra", "X **đóng cửa** rồi", "**không còn ở** X", "X **không có ai** nhận", "X **thì khỏi**" | thêm các mẫu này |
| cờ gấp / dễ vỡ nhầm | "**Ưu tiên** ghé X trước" → gấp; "**Cẩn thận** nhầm sang X" → dễ vỡ; "Tránh **va**n phòng..." → dễ vỡ ("tránh va") | "ưu tiên" + động từ ghé là thứ tự; "cẩn thận" + nhầm / lạc / sai không phải dễ vỡ; "tránh va" cần ranh giới từ |

Đợt 3 (câu ghép, đảo trật tự, chen lời nhắn) và vòng **lỗi gõ** (gõ sai MỌI câu thay mới, kiểu lỗi của dữ liệu:
đảo 2 chữ kề nhau / rơi 1 chữ). Sửa thêm:

| lỗi | ví dụ | sửa |
|---|---|---|
| lời gọi bị nhận là tên địa điểm | "**Bạn ơi**, phòng lab nhé" (đích cụt) | cụm có "ơi" không phải tên |
| phủ định cấp câu bỏ mất dấu phẩy | "qua X **đã, rồi** mới đi tiếp" → "đã rồi" ~ "đã rời" | xét phủ định trên câu còn dấu phẩy |
| luật gấp / dễ vỡ quét cả chữ của TÊN | "chỗ để xe" ~ "chờ", "phòng **cấp cứu**" → gấp | che tên địa điểm trước khi xét |
| ghé "trước tiên / đầu tiên + qua X" | | thêm vào động từ ghé |
| chữ then chốt gõ sai | "Không **ầcn** tới X", "Chớ giao **nầhm**", "Đừng **mnag** tới", "bỏ **qau**", "Khong **iao** o X", "Noi **nhna**: X" | chữ KHÔNG phải âm tiết tiếng Việt hợp lệ (phụ âm đầu + vần + phụ âm cuối) mà cách đúng một chữ then chốt một lỗi đảo / rơi chữ -> sửa thành chữ đó; âm tiết hợp lệ ("giờ", "hỏng", "tòa") không bao giờ bị sửa |

| lỗi gõ ra âm tiết HỢP LỆ | "**hông** phải X", "Tránh **nầm** với X", "đã lấy hàng **ôm** qua" | sửa theo từ đứng sau: cặp (chữ, từ sau) chưa từng gặp mà (chữ then chốt, từ sau) gặp ≥ 5 lần; câu có dấu chỉ sửa khi dạng có dấu không phải dạng quen (≥ 10% và ≥ 3 lần: "hỏng", "đã" là quen; "hông" không), câu không dấu chỉ sửa từ hiếm |
| chữ khung gõ sai bị gộp vào tên lạ | "X là **nơi cần gaio**" → tên lạ "nơi cần gaio" → điểm ghé | gọt chữ khung gõ sai ở mép tên lạ (dùng bản đã sửa lỗi gõ của cả câu) |
| câu tương phản: phủ định lan sang cả câu | "**Không phải** canteen **mà là** phòng lab", "Giao tới phòng lab **chứ không phải** canteen" → đích sai (cả 10 robot) | phạm vi phủ định = vế chứa địa điểm, giữa các từ tương phản "mà (là) / chứ / nhưng / thay vì" (dùng dấu để không nhầm "chú bảo vệ", "mã") |
| đợt 5 (`scratch/h27b_contrast.py`): câu tương phản gộp đích + gây nhiễu | "Giao tới X chứ **không phải** Y" → đích bị gán "phía đông" ("phải" ≠ "bên phải"); "Mang tới X **thay vì** Y" → Y thành điểm ghé; "Không giao ở Y **mà giao ở** X", "Không phải Y **mà là** X gần Z hơn" → X bị bộ phân loại gán gây nhiễu / mốc | "không / chẳng / đâu / chứ không + phải" không là manh mối hướng; "thay vì" là phủ định và giữ trong vế của Y; trong câu có từ tương phản, địa điểm KHÔNG bị phủ định đứng ngay sau "là" / "giao ở, mang tới…" là đích |
| đợt 4: gây nhiễu mới | "X **đã có hàng** rồi", "Hàng **không dành cho** X", "Đừng **lạc** sang X" ("lạc" không dấu = "lắc": luật dễ vỡ "đừng lắc" còn xóa mất), "**Không đi qua** X" | thêm mẫu hẹp (không đụng khung đích "Có đơn giao Y ở X"); "đừng lắc" + sang / vào / tới / qua không phải dễ vỡ |

Kết quả cuối (mã hiện tại): đợt 1 58/58, đợt 2 65/65, đợt 3 23/23 phép thử đạt điểm 1,0000; vòng lỗi gõ 98 phép thử
trung bình 0,9913 → 0,9933 sau khi sửa theo cặp từ (trước mọi sửa lỗi gõ: thấp nhất 0,706). Hồi quy trên dữ liệu thật không đổi sau mọi lần sửa:
giấu tên gọi công bằng 0,9983 / thường 0,9999 / validation học train 1,0000; validation đầu-cuối 6 cấu hình đều 1,0000.

## Đoán loại tên gọi HOÀN TOÀN MỚI (`scratch/h28_knowledge_holdout.py`, `scratch/h29_typing_methods.py`)

Giấu 1/5 tên tự viết (357 tên không trùng dữ liệu), học lại, đoán loại tên bị giấu: chỉ dựa vào tên đúng 92,2% (có dấu)
/ 91,0% (không dấu); trong câu có món hàng theo phân bố thật 94,2% / 92,1%. Đây là ứng viên lớn nhất cho phần điểm còn
mất (test có ~36% đích là tên gọi mới). Sai thường gặp: "phòng + X" bị kéo về văn phòng ("phòng robot", "phòng chế tạo").
So các cách (h29, chỉ dựa vào tên, có dấu / không dấu): hiện tại 0,922 / 0,910; e5 hồi quy 0,911 / 0,814; láng giềng
gần nhất (tên đã biết giống nhất của từng loại) 0,894 / 0,831; câu mô tả loại 0,772 / 0,726; tích hiện tại × láng giềng
× mô tả 0,939 / 0,912. Chọn cấu hình bằng HAI thước đo cùng lúc (tên hoàn toàn mới h28 + giấu tên gọi công bằng):

| cấu hình | h28 trong câu | giấu tên công bằng | validation |
|---|---|---|---|
| cũ | 0,942 / 0,921 | 0,9983 | 1,0000 |
| CHAR_W 0,5 + láng giềng + mô tả 0,5 + CTX 0,8 | 0,967 / 0,935 | 0,9979 | 1,0000 |
| **láng giềng 1 + mô tả 1 + CTX 0,8 (chọn)** | **0,958 / 0,927** | **0,9983** | **1,0000** |

CHAR_W 0,5 cho h28 cao nhất nhưng làm kém phép đo công bằng (lặp lại ở mọi tổ hợp) nên không dùng. Cấu hình chọn
không làm kém thước đo nào; 146 câu thử khung mới vẫn đạt 1,0000. Kho tên tự viết mở rộng 348 → 559 mục
(`nlp_knowledge.PLACES_MORE2`: biến thể chính tả "kí túc xá", viết tắt "phòng CTSV", từ mượn "sân futsal", dạng mô tả).

## Thứ tự hai chặng (đợt 6, `scratch/h27c_order.py`, 12 cách nói × 30 cặp địa điểm × có / không dấu)

8/12 cách nói đúng ngay ("Bước 1: ghé A… Bước 2: giao tới B", "Điểm thứ nhất là A, điểm cuối là B", "Nhận hàng tại A,
trả hàng tại B"...). 4 cách sai 100% và đã sửa: "**Từ** A mang hàng **sang** B", "Lấy hàng ở A rồi **đem sang** B",
"Ghé A xong mới **đi** B" (B bị bộ phân loại gán MỐC: địa điểm ngay sau động từ chuyển động, không có từ quan hệ
khoảng cách, không thể là mốc; "từ X" là nơi lấy hàng), "Hàng ở A, **người nhận ở** B" ("nhận" trong "người nhận" không
phải "nhận hàng"; "Hàng ở A" đầu câu = nơi lấy hàng).

Đợt 7 (`h27c_order.py --v7`, 15 cách nói nữa): 11/15 đúng ngay; sửa "Ghé A, **sau đó** B", "A **trước**, B **sau**",
"Đến A **lấy đồ cho người ở** B", "**Chặng đầu** A, **chặng cuối** B". Luật chung: địa điểm bị gán MỐC mà không có từ quan
hệ khoảng cách (gần / xa / sát / cạnh / kế / cách / giáp / đối diện; không tính chữ thuộc tên khác: "ký túc **xá**") thì
không phải mốc. Đợt 6: 720/720, đợt 7: 900/900. v26 (đợt 6) khác v23 ở 2 cảnh test.
**Bài học:** luật tổng quát "mốc không có từ quan hệ → đích" làm validation học train tụt 1,0000 → 0,9983: tên món hàng
"lấy **thẻ thư viện**" bị bộ phân loại gán mốc (cách nó vô hiệu một "địa điểm" giả), luật biến nó thành đích. Đã bỏ;
thay bằng các khung nhãn tường minh ("chặng cuối X", "sau đó X", "cho người ở X") áp cho mọi vai trò. Mọi sửa đều phải
qua hồi quy (giấu tên công bằng / thường / validation học train) trước khi tạo bản nộp.
Luật "đến X **lấy** đồ → X là điểm ghé" bị thu hẹp: không áp dụng khi chủ ngữ là người khác ("Người nhận **sẽ đến** X lấy
hàng", "Khách đang tới X nhận hàng": X là ĐÍCH vì người nhận tự tới lấy).

Đợt 8 (`scratch/h27d_multi.py`): câu gộp BỐN địa điểm (đích có tham chiếu không gian, điểm ghé có mốc, địa điểm gây nhiễu,
câu tương phản, khung nhãn thứ tự), 8 mẫu × 360 hoán vị × có / không dấu: **5.760/5.760**.

Đợt 9 (`scratch/h27e_misc.py`: câu hỏi "… được không?", "Có ai ở X không?", mục đích "ghé A để lấy…", lịch sự, giờ giấc):
11/12 đúng ngay; sửa **đồng vị ngữ** "Giao tới X, **nơi mọi người đang chờ**" (cụm "nơi / chỗ …" ngay sau dấu phẩy sát sau
một địa điểm chỉ mô tả lại nó) — trừ khi cụm là chủ ngữ của vế mới ("Bỏ qua A, nơi khám bệnh **mới là** điểm nhận");
"đích" không đứng ở mép tên lạ ("nơi khám bệnh là đích"). Đợt 9: 720/720.

## v23 = v24 = v25 trên test

Ba bản chỉ khác nhau ở các sửa khung câu mới / lỗi gõ / câu tương phản, và cho dự đoán TRÙNG HOÀN TOÀN trên 12.000 dòng
test: test không chứa các cách nói đó (bảng xếp hạng chung cuộc cũng chấm trên cảnh của cùng file test). Thay đổi
cuối cùng có tác động lên test là đoán loại tên mới (v22 → v23: 4 cảnh). Giả thuyết: test dùng khung câu như train /
validation nhưng TÊN GỌI mới → phần mất điểm còn lại chủ yếu ở đoán loại tên mới (~5% sai × ~36% đích là tên mới).

## Bản nộp v22 (`outputs/cac_ban_nop_cu/predictions_v22_nlp_newframes_edgestrong.json`)

NLP với mọi sửa khung câu mới ở trên + CNN đoạn đường học thêm ảnh làm méo mạnh (final, cách gộp "cnn,cnn,cnn") +
các sửa CV lượt 2 (lấp ô lưới, sàn màu, đoạn hiếm luôn hỏi CNN). Khác v19 ở 91/12.000 dòng (27/1.200 cảnh).
CNN đoạn đường final từng phân kỳ (loss 0,24 → 1,66): lô cuối mỗi epoch chỉ có 3 mẫu (408.579 = 1.596×256 + 3), BatchNorm
trên 3 mẫu làm gradient nhảy vọt. Sửa: cắt chuẩn gradient + bỏ lô cuối < 32 mẫu (cả 4 script huấn luyện CNN).

## CV dưới độ méo cực nặng (blur 1,2 + JPEG 25 chồng lên validation)

Mô hình dev hiện tại: bản đồ đúng 292/300, điểm robot 0,9957. Chỉ 2 cảnh làm sai robot: cảnh 167 (đọc địa điểm, kiểu
sketch) và cảnh 171 (đọc chú giải → 55 đoạn sai trạng thái, kiểu print). Mức này nặng hơn hẳn validation (test được mô tả
"khó hơn train" như validation) nhưng chỉ ra chỗ gãy đầu tiên → huấn luyện lại CNN chú giải + CNN giao lộ với dữ liệu
làm méo mạnh (trước đó bị hết bộ nhớ hai lần).

## Máy: thiếu bộ nhớ ảo

Ổ C: còn 0,2 GB nên file trang của Windows không nới được; các ứng dụng khác đã chiếm ~23 GB cam kết / trần ~30 GB.
Chạy chồng 2 việc nặng (huấn luyện CNN + đo độ bền 6 tiến trình) là hết bộ nhớ ("paging file is too small").
Từ giờ chạy tuần tự. Huấn luyện lại CNN giao lộ / chú giải bằng dữ liệu làm méo mạnh bị dừng giữa chừng vì lỗi này
(mô hình cũ không bị ghi đè).

# Cập nhật lượt 2 (06/10, sau khi bản v19 được 0,9778 trên bảng xếp hạng)

## Vì sao 0,9778 thấp hơn mọi phép đo cũ

1. **Phép đo "giấu tên gọi" cũ lạc quan**: bảng từ khóa / ví dụ viết tay chứa sẵn đúng các tên bị giấu. Phép đo mới
   `scratch/h1_alias_holdout.py --fair` bỏ cả các từ khóa và ví dụ khớp tên bị giấu: v19 chỉ được **0,9894**.
2. **Ảnh validation méo hơn train** (JPEG 45–75 ở 34% ảnh so với 60–85 ở 11%; xoay > 2° ở 23% so với 4%; mờ ở 40%
   so với 21%). Test có thể còn méo hơn. Phép đo `scratch/h24_cv_stress.py` chồng thêm JPEG / xoay / mờ lên validation.

## NLP (đo bằng phép đo công bằng; train / validation vẫn 1,0000)

| bước | giấu tên gọi công bằng | giấu tên gọi thường |
|---|---|---|
| v19 | 0,9894 | 0,9975 |
| giữ "chỗ" ở đầu tên lạ ("chỗ trả sách") | 0,9913 | |
| người nhận làm đích ("Thư ký khoa cần nhận...") | 0,9922 | 0,9975 |
| điểm ngay sau "ghé" không phải mốc; điểm ghé đã biết loại loại trừ loại đó khỏi đích (0/781 cảnh trùng) | 0,9941 | 0,9979 |
| dò tên lạ theo khung câu ("ghé X lấy...", "mang ... tới X", "gần / xa Y hơn", "ở X", "Điểm giao: X") | 0,9955 | 0,9985 |
| hướng chỉ được nhận khi ngữ cảnh có từ chỉ hướng; khôi phục dấu cho tên lạ không dấu | 0,9956 | 0,9988 |
| thêm ~350 cách gọi địa điểm / người làm việc (`nlp_knowledge.PLACES_MORE`) | 0,9967 | 0,9988 |
| khung câu kết thúc gọn, chữ trùng tên (nha~nhà, cho~chỗ, thi~thí...) không là từ kết thúc | **0,9978** | **0,9998** |

Các luật dò theo khung: 0 báo nhầm trên 2.300 câu train + validation. Validation NLP 5-fold: 0,9943 → 1,0000.
Dò tham số (`scratch/h22_sweep.py`, chấm đồng thời 3 phép đo): điểm gần như không đổi theo CTX_W, CUE_MARGIN,
GOAL_TEXT_W — phần còn lại là các ca cụ thể, không phải tham số.

## CV

- **Lấp ô trống của lưới bằng CNN giao lộ** (`GRID_FILL`): trên 1.567 ô trống thật, CNN cho p(không phải giao lộ)
  ≥ 0,935 (cả khi nén JPEG 35) → ngưỡng 0,5 không nhận nhầm ô nào. Sửa val cảnh 53 (mất một giao lộ khi ảnh bị méo).
- **Sàn cho chứng cứ màu** (`COLOR_FLOOR` 1e-4 → 0,02): ô màu nhỏ trong chú giải bị JPEG làm loang không còn lật được
  loại địa điểm mà CNN đọc chữ rất chắc (val cảnh 190 khi nén JPEG 35).
- **Đoạn đường hiếm mà hệ trọng (đóng / một chiều / bậc thang) luôn được CNN đọc lại** (`EDGE_CNN_RARE`): MLP từng đọc
  một đoạn thường thành "đóng" với xác suất 0,999 (val cảnh 75). Gộp theo từng đầu ra (`EDGE_HEAD_MODE`): kiểu nét lấy
  CNN, bậc thang / một chiều lấy trung bình MLP + CNN, vì CNN đọc bậc thang kém khi ảnh mờ.
  Validation ảnh gốc: mọi thành phần CV 300/300.

# Cập nhật ngày 05–06/10: train và validation đều đạt 100%

## Kết quả (macro accuracy, khi dự đoán KHÔNG dùng scenes.json)

| phép đo | trước | sau |
|---|---|---|
| validation: CV thật + mission đúng | 0,9917 | **1,0000** |
| validation: CV thật + NLP chỉ học train (cách nói lạ) | 0,9880 | **1,0000** |
| validation: bản đồ đúng + NLP chỉ học train | 0,9957 | **1,0000** |
| train: CV thật + NLP cấu hình nộp bài | — | **1,0000** (cả 10 robot) |
| train: CV thật + mission đúng (mô hình dev, học train) | 0,9965 | 0,9999 |
| NLP "giấu tên gọi" (5 nhóm, train + validation; gần với test nhất) | ~0,985 | **0,9973** |
| NLP 5-fold trên validation | 0,9893 | 0,9943 |

Từng khâu CV trên validation (mô hình học train): giao lộ, đoạn đường, robot, hướng mũi, thời tiết, chú giải
đều 300/300 cảnh đúng. Trên train: tất cả 2000/2000 trừ loại địa điểm (5 cảnh lệch, chỉ 1 cảnh làm sai bước đi).

Chiến thuật 10 robot không đổi (vẫn 100% với thông tin đúng).

## Các thay đổi, theo mức quan trọng

### CV

1. **CNN cho đoạn đường** (`src/cv_edge.py`). MLP cũ sai 1 / 10 / 16 mảnh (kiểu nét / bậc thang / một chiều) trên
   26.738 mảnh validation; CNN sai 0 / 0 / 0. CNN chậm trên CPU nên chỉ được hỏi khi MLP không chắc (xác suất < 0,999,
   khoảng 4,7% số đoạn) — cách này bắt được 23/23 lỗi của MLP.
2. **Lớp "không phải giao lộ" cho CNN giao lộ** (`src/cv_cnn.py`, `src/cv_data.py`). Bộ dò quét dày báo nhầm ở 53/300
   ảnh validation (chữ tiêu đề, vạch bậc thang, mũi tên). Một đỉnh nhầm lọt vào lưới làm gộp hai cột và lệch chỉ số cả
   bản đồ (cảnh 81). Với đỉnh dò sạch thì khớp lưới đúng 300/300. CNN giờ có lớp thứ 16 học từ các mảnh "nhiều mực"
   không phải giao lộ; `drop_nonnodes` loại các đỉnh đó trước khi khớp lưới.
3. **CNN giao lộ được tập với tâm lệch tới 20% unit** (trước 7%), và được đọc thêm một lần tại tâm đã căn. Bộ dò
   hay lệch 10–20px ở nhãn chữ rộng của kiểu "print".
4. **Trọng số CNN khi ghép với MLP: 0,8** (trước là trung bình hình học 0,5). Soi từng ca sai: CNN đúng (≥ 0,95) trong
   mọi ca, MLP sai và tự tin. Train: 0,5 → 0,9977; 0,8 → 0,9999; 1,0 (chỉ CNN) → 0,9963.
5. **CNN đọc chú giải** (`src/cv_swatch.py`): lỗi đọc ý nghĩa dòng từ ~143 xuống 0–1 trên 17.635 dòng validation;
   3 ảnh train bị đọc sai hoán vị màu đường (93 đoạn sai trạng thái) nay đúng hết.
6. **CNN thời tiết** (`src/cv_weather.py`): 0/1800 lỗi. MLP cũ đọc biểu tượng "nhiều mây" thành mưa (cảnh 104).
7. **Vị trí biểu tượng thời tiết**: chỉ ở dòng chú giải hoặc góc trên phải (x/W 0,90–0,97, y/H 0,02–0,09 trên 100%
   ảnh train + validation). Trước đây đĩa vàng của robot bị lấy làm biểu tượng thời tiết rồi bị xóa khỏi danh sách giao lộ
   (cảnh 191: mất robot, cả 10 robot sai).
8. Sửa lỗi: `import os` nằm trong `Models.__init__` làm `os` thành biến cục bộ; khởi tạo mô hình crash và pool treo.

### NLP

1. **Sửa lỗi gõ của từ chỉ hướng** (`nlp.repair_spatial`): "phía ren" → "phía trên", "bên tria" → "bên trái",
   "pia tây" → "phía tây". Chỉ sửa trong khung chỉ hướng; trên 2.300 yêu cầu train + validation sửa đúng 5 chỗ, không
   sửa nhầm chỗ nào.
2. **Lỗi gõ của "không"**: ngưỡng tần suất bảo vệ từ thật tăng 5 → 20. Từ thật gần "không" xuất hiện ≥ 54 lần
   (phong 960, hong 199, chong 54); lỗi gõ ≤ 5 lần (khogn 5, khng 4...). Ngưỡng cũ làm "khôgn cần ghé X" thành điểm ghé.
3. **Khớp mờ tên gọi nhiều từ khi chỉ lệch một từ** ("hong lab" ~ "phong lab"): trước đây bị chặn vì "hong" là từ
   thông dụng. Cụm quen thuộc ("cong van") vẫn bị chặn như cũ.

## Môi trường

- PyTorch bản CUDA cài ở `F:\pylibs` (ổ C gần đầy): chạy huấn luyện với `PYTHONPATH=F:/pylibs`.
  Khi dự đoán chỉ cần `onnxruntime`, không cần PyTorch.
- GTX 1650 4 GB: **không** dùng `cudnn.benchmark`, AMP hay channels_last (chậm 6–10 lần); không để cả mảng dữ liệu
  trên GPU (tràn bộ nhớ, chậm 10 lần).
- Đặt `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` khi chạy trên Windows (code in tiếng Việt).
- `cv_swatch.py` xuất ONNX bằng trình xuất mới (`dynamo=True`, cần gói `onnxscript`, đã cài vào `F:\pylibs`) vì
  `AdaptiveAvgPool2d((1, 4))` không xuất được bằng trình cũ. File `swatch_cnn.onnx` đi kèm `swatch_cnn.onnx.data`
  (trọng số ngoài) — phải giữ cả hai. Đã kiểm tra ONNX khớp PyTorch 100% trên 2.000 dòng chú giải.
- Đường dẫn dữ liệu: `src/common.py` tự tìm `delivery_public` ở thư mục cha nếu không có trong `AI/`.

## File nộp

`outputs/predictions.json` (bản sao `outputs/cac_ban_nop_cu/predictions_v19_cnn_cv_typo.json`): hợp lệ theo
`check_submission.py`, tổng tham số 131.244.682. Khác bản 0,9600 ở 194 / 12.000 đáp án (54 / 1.200 cảnh).
Bản trước khi chạy lượt này được giữ ở `outputs/cac_ban_nop_cu/predictions_truoc_v19.json`.

## Chạy lại

```powershell
python src/cv_data.py train ; python src/cv_data.py validation        # cắt mảnh (gồm cả dữ liệu CNN giao lộ)
$env:PYTHONPATH="F:/pylibs"
python src/cv_cnn.py dev 8 ; python src/cv_edge.py dev 6 ; python src/cv_swatch.py dev 12 ; python src/cv_weather.py dev 14
python src/run_cv.py validation dev ; python src/eval_cv.py validation dev
python src/pipeline.py eval dev                                         # validation end-to-end
python src/run_cv.py train dev ; python scratch/q8_train_final.py       # train end-to-end (cấu hình nộp bài)
python scratch/h1_alias_holdout.py                                      # NLP giấu tên gọi

python src/cv_cnn.py final 8 ; python src/cv_edge.py final 6 ; python src/cv_swatch.py final 12 ; python src/cv_weather.py final 14
python src/pipeline.py test final ; python src/check_submission.py final
```

## Tuân thủ quy định

- Không đọc câu hay xem ảnh test, không gán nhãn, không dùng bộ đếm tổng hợp của test để chọn luật trong lượt này.
  Mọi sửa đổi được tìm ra và đo trên train / validation. Test chỉ dùng để chạy dự đoán.
- Tổng tham số vẫn dưới 200 triệu (xem `python src/check_submission.py final`).

## Lưu ý trung thực

Train và validation 100% không có nghĩa bảng xếp hạng sẽ 100%. Test có tên gọi và khung câu chưa từng gặp
(khoảng 35% cảnh có đích là tên gọi lạ). Phép đo gần với tình huống đó nhất là "giấu tên gọi": 0,9973.
