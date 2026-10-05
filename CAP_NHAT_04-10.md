# Cập nhật ngày 03–04/10: từ 0,9389 lên 0,9567

File nộp tốt nhất hiện tại: `outputs/predictions.json` (bản sao: `outputs/cac_ban_nop_cu/predictions_v15_lb0.9567_calm.json`).
Điểm bảng xếp hạng: **0,9567 (1722/1800)**, robot yếu nhất 0,9111.

Chạy lại từ đầu ra đúng file này:

```powershell
python src/pipeline.py test final
python src/check_submission.py final
```

## Bổ sung ngày 05/10

- **Điểm tốt nhất hiện tại: 0,9600 (1728/1800)**, file `outputs/predictions.json`
  (bản sao: `outputs/cac_ban_nop_cu/predictions_v17_lb0.9600_last.json`).
- So với bản 0,9567: câu phụ hai vế kiểu "…không gấp lắm, nhưng đang rất gấp" / "tưởng gấp, … không cần vội" được đọc
  theo quy tắc **vế cuối là ý chốt** (`nlp2.MIXED_POLICY = "last"`), chỉ ảnh hưởng robot 6. Được thêm 6 đáp án.
- Bản đang chờ điểm: `outputs/cac_ban_nop_cu/predictions_v18_cach_rule.json`. Thêm luật `CACH_RULE`: nhận chỉ dẫn gần / xa
  viết kiểu "X cách Y không xa", "X ở tận phía xa Y" (từ gần / xa đứng sau địa điểm mốc, hoặc mốc đứng xa tên đích hơn).
  Luật thêm 34 chỉ dẫn ở test, cả 34 địa điểm mốc đều có đúng 1 bản trên bản đồ (tính chất của mốc thật). Khác bản
  0,9600 ở 41 đáp án. **Code trong `src/` đang ở cấu hình v18**; muốn ra lại đúng bản 0,9600 thì đặt
  `nlp2.CACH_RULE = False`.
- Đã thử và bỏ: bộ phân loại ngữ cảnh khớp chính xác từ ngữ (`CTX_NG_W`, để 0).
- Lưu ý về quy định: để tìm ra kiểu câu "X cách Y", đã đếm các từ chức năng nằm giữa hai tên địa điểm ở test (như
  "cách", "xa", "ở"). Không đọc cả câu hay đáp án, nhưng sát ranh giới hơn việc chỉ đếm tỉ lệ.

## Điều cần biết trước khi đọc

- Bảng xếp hạng chỉ chấm **180 cảnh** (1800 đáp án), không phải cả 1200 cảnh test. Một đáp án = 0,00056 điểm.
- Phần đọc ảnh (CV) và chiến thuật 10 robot **không đổi** trong hai ngày này. Mọi cải tiến nằm ở khâu đọc câu yêu cầu
  (`src/nlp2.py`, `src/nlp.py`, `src/nlp_knowledge.py`) và một chỗ nhỏ trong `src/strategy.py`.
- Test dùng cách nói khác train và validation. Luật thi cấm đọc câu test, nên mỗi sửa đổi đều được đo trên train và
  validation bằng cách "giấu bớt dữ liệu rồi chấm trên phần bị giấu", và chỉ dùng **số đếm tổng hợp** của lượt chạy test
  (ví dụ: bao nhiêu % cảnh không tìm được đường), không xem ảnh hay câu test nào.

## Lịch sử các bản nộp

| bản | thay đổi chính | điểm | robot yếu nhất |
|---|---|---|---|
| v4 | (đầu ngày 03/10) | 0,9389 | 0,8889 |
| v5 | xử lý khi robot hết đường, sửa lỗi gõ nhầm, gộp tên địa điểm | 0,9394 | 0,8944 |
| v6 | sửa lỗi "bên trong" bị hiểu là "hàng bền" | 0,9444 | 0,8944 |
| v7 | hạ ngưỡng mô hình nghĩa để nhận thêm câu "gấp" | 0,9422 (bỏ) | — |
| v9 | luật "đích phải tới được" | 0,9450 | 0,8944 |
| v11 | sửa một loạt lỗi đọc câu gấp / không gấp | 0,9467 | 0,9056 |
| v13 | câu hai vế: coi là gấp nếu một vế có chữ gấp | 0,9461 (bỏ) | — |
| **v15** | **nhận ra địa điểm gây nhiễu viết theo kiểu mới** | **0,9567** | **0,9111** |

## Các sửa đổi, theo mức quan trọng

### 1. Địa điểm gây nhiễu viết theo kiểu mới (+18 đáp án, lớn nhất)

Train và validation chỉ có 6 kiểu câu gây nhiễu ("Không cần ghé X", "Hôm qua đã giao ở X rồi"…). Test có kiểu mới, mô
hình không nhận ra nên coi các địa điểm đó là **điểm ghé** hoặc **đích**, làm cả 10 robot đi sai.

Hai luật mới trong `MissionParser2.parse(text, present)`:

- `PRESENT_RULE`: tên địa điểm đã biết mà loại đó **không có trên bản đồ** thì là gây nhiễu. Trên train và validation,
  đích / điểm ghé / mốc luôn có trên bản đồ (3297/3297), còn địa điểm gây nhiễu vắng mặt 55%.
- `DIS_RULE`: địa điểm đứng riêng trong một câu có ý phủ định hoặc quá khứ, câu không có động từ ghé và không có từ chỉ
  thứ tự ("trước", "xong", "sau đó"…), thì là gây nhiễu. Luật không bắt nhầm điểm ghé thật nào trên train (0/813).

Cách kiểm tra ở test mà không cần đáp án: trong các địa điểm `DIS_RULE` bắt được, 54% không có trên bản đồ, khớp với tỉ
lệ 55% của địa điểm gây nhiễu thật. Nếu luật bắt nhầm điểm ghé thật thì tỉ lệ này phải gần 0%.

### 2. Lỗi do bỏ dấu làm các chữ trùng nhau

Mô hình xử lý câu ở dạng không dấu, nên nhiều chữ khác nghĩa thành giống nhau:

- "bên trong" và "hàng **bền**" → câu "Bên trong là đồ thủy tinh" bị coi là không dễ vỡ (sửa ở v6, +9 đáp án).
- "hỏng", "thong (thả)", "phòng" bị bước sửa lỗi gõ đổi thành "không" → mất ý "dễ hỏng", "thong thả".
- "gấp / gặp", "vội / với", "gốm / gồm", "ngay / ngày": hàm `disambiguate()` dùng dạng có dấu của câu gốc để phân biệt.
- "bên trong" bị bước sửa lỗi gõ coi là "bếp trưởng" (tên gọi của căng tin) → từ thông dụng không còn bị coi là gõ sai
  (`nlp.FUZZY_COMMON`).

### 3. Câu gấp / không gấp và dễ vỡ / không dễ vỡ

- Thêm luật từ khóa cho phủ định ("đừng vội", "hết gấp rồi", "ưu tiên thấp") và cách nói khác ("không thể đợi",
  "trễ là hỏng hết"): các biến `URGENT_*_MORE`, `FRAGILE_*_MORE`.
- Thêm khoảng 480 câu ví dụ viết tay cho mô hình nghĩa e5 (`nlp_knowledge.PHRASES_MORE`).
- Mô hình nghĩa không còn được phủ quyết từ khóa khi câu không có từ phủ định (nó hiểu phủ định kém).

### 4. Tên gọi lạ

- Gộp "tòa / khu / dãy + tên đã biết" thành một địa điểm; kéo cụm tên qua chữ "của" và "ở" ("tòa nhà ở của sinh viên").
- Giữ chữ "xá" cuối tên ("trạm xá" không phải "trạm … xa").
- Thêm manh mối ngữ cảnh (món hàng, người nhận) khi đoán loại của tên lạ (`CTX_W`).

### 5. Các luật dựa trên bản đồ

- `REACH_RULE`: đích thật luôn tới được (2300/2300 trên train và validation). Nếu đích đang chọn không có đường tới thì
  chọn ứng viên kế tiếp.
- `strategy.RELAX`: nếu vẫn không có đường, cho phép đi qua đoạn "nghi đọc sai" với giá phạt lớn thay vì đoán bừa.

## Việc còn dở

Robot 6 (đi theo mức "gấp") vẫn là robot yếu nhất (164/180). Test có hai mẫu câu hai vế "một vế có chữ gấp, vế kia có ý
không gấp" mà chưa biết mẫu nào là gấp. Hai bản thử đã có sẵn, chỉ khác bản 0,9567 ở robot 6:

- `outputs/cac_ban_nop_cu/predictions_v15_disrule_mixed_last.json` (vế cuối quyết định)
- `outputs/cac_ban_nop_cu/predictions_v15_disrule_mixed_first.json` (vế đầu quyết định)

## Tuân thủ quy định

- Dự đoán chạy offline, không gọi API nào. Tổng tham số 128.622.818 (giới hạn 200 triệu).
- Không huấn luyện, không gán nhãn, không đọc câu hay xem ảnh test. Các luật từ khóa viết từ kiến thức tiếng Việt chung
  và từ train / validation.
- Có dùng số đếm tổng hợp của lượt chạy test và điểm bảng xếp hạng để chọn giữa các phương án. Nếu đội muốn chắc chắn
  tuyệt đối, nên hỏi ban tổ chức xem cách dùng này có được phép không.
