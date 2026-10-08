# Vòng private test (08/10) — nhánh `private_cases`

Đề vòng private (`DE_BAI.md` mới): train / validation / test đều là bộ mới; **chiến thuật 10 robot đổi** (trọng số, điều kiện,
phá hòa riêng từng robot); nơi giao có thể **chỉ mô tả qua bản đồ** (`north_most` / `south_most` / `west_most` / `east_most` /
`anchor_near`); câu có đính chính / đối chiếu; bản đồ 8–14 địa điểm. Nộp ở mục "Campus Courier · Private test" (`COURIER2`).

## Kết quả

| file (`outputs/private_result/`) | nội dung | validation (CV thật, NLP + chiến thuật chỉ học train) | bảng xếp hạng |
|---|---|---|---|
| `predictions_p4_hybrid_tv.json` | chiến thuật lai theo điều kiện (đêm / mưa / gấp / dễ vỡ / điểm ghé) | ~0,83 | **0,7744** |
| `predictions_p6_nlpkb.json` | + kho tên gọi vòng 1 vào từ điển | 0,839 | 0,7583 (kho tên gọi làm hại trên test → bỏ) |
| `predictions_p11_goaltype.json` | + **loại nơi giao** là điều kiện ẩn (R2, R3, R7) + luật R0 | 0,867 | **0,8056** |
| **`predictions_p12_tie.json`** (= `predictions.json`) | + thứ tự phá hòa riêng từng robot (R8 ưu tiên rẽ phải) | **0,871** | chưa nộp |

Từng robot (p12, validation, thông tin từ CV + đọc câu): R0 0,957 · R1 0,863 · R2 0,890 · R3 0,847 · R4 0,860 · R5 0,900 ·
R6 0,820 · R7 0,830 · R8 0,850 · R9 0,893.

## Cách chạy lại

```powershell
# dữ liệu ở ..\delivery_public (bộ mới); CV dùng lại mô hình vòng 1 (outputs/models_v1final, 98,3% bản đồ đúng trên validation mới)
$env:NLP_LEX="canon"
python src/run_cv.py validation v1final ; python src/run_cv.py test v1final     # CV (~23 phút cho test, PROCS=6)
python src/pipeline_p2.py fit trainonly     # bộ đọc câu + mô hình học máy chỉ học train (để đo validation)
python src/pipeline_p2.py eval v1final      # validation end-to-end
python src/pipeline_p2.py fit final         # học train + validation
python src/pipeline_p2.py test v1final      # -> outputs/private_result/predictions.json
```

Tham số chiến thuật nằm trong `outputs/strategy_hybrid.json` (chỉ học train, dùng cho `eval`) và
`outputs/strategy_hybrid_final.json` (học train + validation, dùng cho `test`); chúng được dò bằng các script `scratch/p2_*.py`
(`p2_cd3.py` theo nhóm điều kiện, `p2_lex.py` cho R0, `p2_a37_r9dist.py` cho R9, `p2_tie2.py` cho thứ tự phá hòa,
`p2_cond.py` tự dò điều kiện ẩn).

## Thành phần mới

| file | vai trò |
|---|---|
| `src/mapref.py` | nhận diện nơi giao mô tả qua bản đồ (đầu ngữ chung + so sánh nhất; sửa lỗi gõ theo tần suất) và tìm vị trí: cực trị trong mọi địa điểm / gần mốc nhất theo Manhattan. Validation 300/300, câu thử tự soạn 630/630 |
| `src/nlp3.py` | bộ đọc vòng 2: thay cụm mô tả bằng tên loại ở vị trí đó rồi dùng `nlp2.MissionParser2` (học lại trên dữ liệu mới); từ điển = tên khai thác + 10 tên chuẩn |
| `src/nlp.py` | ranh giới câu "(tin trước ghi … là nhầm)", "đổi lại: / thay vào đó: / đúng ra:"; khung câu nhiễu mới (đính chính, hủy đơn) |
| `src/strategy.py` | đích chốt theo vị trí (`("pos", (r, c))`) |
| `src/strat_ml.py` | chiến thuật lai: Dijkstra trên (giao lộ, hướng) với tham số theo nhóm điều kiện; luật thứ tự từ điển cho R0; tham lam cho R9; nhận diện ảnh BAN ĐÊM theo độ sáng |
| `src/pipeline_p2.py` | ghép CV + NLP + chiến thuật |

## Phát hiện về chiến thuật (từ nhãn train)

- **R0**: ít đoạn nhất → ít đi XUYÊN giao lộ có địa điểm (trừ đích / điểm ghé) → (mưa: ít đoạn không mái che, ít đông; khô: ít
  đông, ít quay đầu) → thứ tự hướng. Validation 0,987 với thông tin đúng.
- **Kiểu vẽ `night` = ban đêm**: robot đổi cách đi (R1 không ngại đường đông, R5 không ngại rẽ). Nhận diện bằng độ sáng ảnh
  (đêm ≤ 49, ngày ≥ 221; 0 / 700 lỗi).
- **Loại nơi giao** là điều kiện ẩn của R2, R3, R7 (`scratch/p2_cond.py`): R2 0,78 → 0,90, R3 0,66 → 0,86.
- **R8** đi rất khác nhau khi có / không có điểm ghé; khi hòa ưu tiên rẽ phải.
- **R9** tham lam: khoảng cách đường chim bay tới điểm đến + phạt đông / đi xuyên địa điểm theo đêm × dễ vỡ × gấp.
- Mọi lần đi vòng đều đúng +2 đoạn; phần còn sai (10–15% ở R1, R4, R6, R7, R8) chưa giải thích được.

## Tuân thủ quy định

- Test chỉ dùng để chạy dự đoán. Không đọc câu hay xem ảnh test, không gán nhãn, không học trên test. Luật và tham số được tìm và
  đo trên train / validation; phép thử tên gọi / cách nói mới là câu tự soạn (`scratch/p2_n4*.py` – `p2_n7*.py`).
- Thứ duy nhất lấy từ lượt chạy test là các **bộ đếm tổng hợp** của chính hệ thống (`scratch/p2_health.py`: tỉ lệ đích là tên
  đã biết, tỉ lệ mô tả qua bản đồ…) để biết test khác validation; không dùng để chọn luật cho từng câu. Nếu cần chắc chắn tuyệt
  đối, đội nên hỏi BTC về việc dùng bộ đếm như vậy.
- Không gọi API ngoài; tổng tham số không đổi so với vòng 1 (dưới 200 triệu).
