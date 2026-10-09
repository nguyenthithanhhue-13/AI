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
| `predictions_p12_tie.json` | + thứ tự phá hòa riêng từng robot (R8 ưu tiên rẽ phải) | 0,871 | chưa nộp |
| `predictions_p13_ml4.json` | + R5, R7 dùng bộ phân loại "ml4" (đặc trưng bước + đêm + chi phí mô hình lai + loại nơi giao) | 0,876 | chưa nộp |
| `predictions_p14_vin.json` | + R1, R2, R4, R6 dùng BỘ TÌM ĐƯỜNG CÓ HÀM CHI PHÍ HỌC ĐƯỢC (`src/vin.py`, mạng nhỏ + lặp giá trị, gộp 3 mạng) | 0,884 | chưa nộp |
| `predictions_p15_vinbig.json` | + mạng lớn (hid 128) cho R1, R4 (gộp với mạng nhỏ), R8; bộ nhận diện mô tả qua bản đồ thêm "tít / đầu phía / hơn tất cả / bậc nhất" (test: 464 -> 487 cảnh nhận ra) | 0,888 | **0,8189** |
| `predictions_p16_mapref3.json` | + bộ nhận diện mô tả qua bản đồ TỔNG QUÁT (`mapref._generic`: dấu so sánh nhất + từ hướng / từ gần trong một vế câu; "gần nhất với X", "phía bắc xa nhất", "mạn bắc nhất", "cách X ngắn nhất"...; và đầu ngữ chung + gần / phía khi "nhất" bị bỏ). Câu thử tự soạn: vòng 2 50,6% -> 100% (3060), vòng 3 end-to-end 1788/1788, không "nhất" 1200/1200; train / validation không đổi. Test: nhận ra mô tả qua bản đồ 487 -> 608 cảnh; khác p15 296 dòng / 96 cảnh | 0,888 | 0,8189 (= p15: các cảnh đổi gần như chắc là cảnh MỒI không chấm) |
| `predictions_p17_hyper.json` | + mô hình SIÊU TUYẾN TÍNH trên mặt Pareto (`src/hyper.py`): chi phí đường TUYẾN TÍNH theo [số đoạn, đông, mái che, không mái, xuyên địa điểm (gộp + từng loại), bậc thang, rẽ R/L/B] + phạt rẽ bước đầu, trọng số = softplus(MLP(điều kiện cảnh)); khớp train ~0,998 với mọi robot. GỘP log-xác suất (method `hv`): R1, R2, R6, R8 = vin + hyper đầy đủ + hyper (cờ + loại đích); R3 = 2 bộ hyper; R7 = hyper (cờ + loại đích); R0, R4, R5, R9 giữ như p15. Khác p16 871 dòng | 0,894 | **0,8261** |
| `predictions_p18_r9abs.json` | + R9 luật tham lam gọn (`strat_ml._greedy2_move`, dò ở `scratch/p2_greedy_exact_fix.py`): khoảng cách tới điểm đến (gấp: Manhattan, không gấp: Euclid lưới) + phạt đông / mái che / vào địa điểm (+ theo loại) / rẽ, theo nhóm (đêm, gấp, dễ vỡ); HÒA -> thứ tự hướng TUYỆT ĐỐI LÊN, PHẢI, XUỐNG, TRÁI. R9 thông tin đúng: train 0,916 -> 0,969, validation 0,897 -> 0,953. Khác p17 97 dòng | **0,9003** | chưa nộp |
| **`predictions_p19_sel.json`** (= `predictions.json`) | + bộ siêu tuyến tính với ĐIỀU KIỆN RIÊNG từng robot (chọn theo độ nhạy `scratch/p2_hyper_sens.py`: R1 mưa+đêm+gấp, R2 mưa+đêm+loại đích, R3 mưa+loại đích, R4 mưa+gấp+vỡ+loại đích, R5 chỉ đêm, R6/R7/R8 nhiều hơn; `outputs/hypersel_*.npz`), gộp 4 mô hình cho R1, R2, R3, R4, R8; R5 = chỉ bộ điều kiện riêng. Khác p17 (đã nộp) 351 dòng (2,9%) | **0,9033** | chưa nộp |

Từng robot (p14: R1 0,893 · R2 0,893 · R4 0,867 · R6 0,867; còn lại như p13) (p13, validation, thông tin từ CV + đọc câu): R0 0,957 · R1 0,863 · R2 0,890 · R3 0,847 · R4 0,860 · R5 0,927 ·
R6 0,820 · R7 0,850 · R8 0,850 · R9 0,893.

## Cách chạy lại

```powershell
# dữ liệu ở ..\delivery_public (bộ mới); CV dùng lại mô hình vòng 1 (outputs/models_v1final, 98,3% bản đồ đúng trên validation mới)
$env:NLP_LEX="canon"
python src/run_cv.py validation v1final ; python src/run_cv.py test v1final     # CV (~23 phút cho test, PROCS=6)
python src/pipeline_p2.py fit trainonly     # bộ đọc câu + mô hình học máy chỉ học train (để đo validation)
python scratch/p2_ml4_train.py trainonly 5 7   # bộ phân loại ml4 cho R5, R7 (bản final: ... final 5 7)
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
| `src/hyper.py` | mô hình siêu tuyến tính: mặt Pareto các đường (số đoạn <= ngắn nhất + 5) + trọng số theo điều kiện (huấn luyện `scratch/p2_hyper.py`, mặt Pareto `scratch/p2_pareto.py`, xuất `scratch/p2_hyper_export.py`) |
| `src/vin.py` | dự đoán bằng numpy của bộ tìm đường có hàm chi phí học được (huấn luyện: `scratch/p2_vin_data.py`, `scratch/p2_vin2.py`, xuất: `scratch/p2_vin_export.py`) |

## Phát hiện về chiến thuật (từ nhãn train)

- **R0**: ít đoạn nhất → ít đi XUYÊN giao lộ có địa điểm (trừ đích / điểm ghé) → (mưa: ít đoạn không mái che, ít đông; khô: ít
  đông, ít quay đầu) → thứ tự hướng. Validation 0,987 với thông tin đúng.
- **Kiểu vẽ `night` = ban đêm**: robot đổi cách đi (R1 không ngại đường đông, R5 không ngại rẽ). Nhận diện bằng độ sáng ảnh
  (đêm ≤ 49, ngày ≥ 221; 0 / 700 lỗi).
- **Loại nơi giao** là điều kiện ẩn của R2, R3, R7 (`scratch/p2_cond.py`): R2 0,78 → 0,90, R3 0,66 → 0,86.
- **R8** đi rất khác nhau khi có / không có điểm ghé; khi hòa ưu tiên rẽ phải.
- **R9** tham lam: khoảng cách đường chim bay tới điểm đến + phạt đông / đi xuyên địa điểm theo đêm × dễ vỡ × gấp.
- Mổ mạng chi phí (`scratch/p2_vin_probe.py`): phạt đi XUYÊN địa điểm khác nhau THEO LOẠI: R1 né ký túc xá / sân thể thao / căn tin, R2 né bãi xe / sân thể thao (+ phòng y tế), R6 né ký túc xá / căn tin, R8 né thư viện / giảng đường / phòng hành chính / cổng, R4 như nhau. Chi phí tuyến tính theo đặc trưng đường với trọng số phụ thuộc điều kiện khớp train ~0,998.
- Mọi lần đi vòng đều đúng +2 đoạn; phần còn sai (10–15% ở R1, R4, R6, R7, R8) chưa giải thích được.

## Tuân thủ quy định

- Test chỉ dùng để chạy dự đoán. Không đọc câu hay xem ảnh test, không gán nhãn, không học trên test. Luật và tham số được tìm và
  đo trên train / validation; phép thử tên gọi / cách nói mới là câu tự soạn (`scratch/p2_n4*.py` – `p2_n7*.py`).
- Thứ duy nhất lấy từ lượt chạy test là các **bộ đếm tổng hợp** của chính hệ thống (`scratch/p2_health.py`: tỉ lệ đích là tên
  đã biết, tỉ lệ mô tả qua bản đồ…) để biết test khác validation; không dùng để chọn luật cho từng câu. Nếu cần chắc chắn tuyệt
  đối, đội nên hỏi BTC về việc dùng bộ đếm như vậy.
- Không gọi API ngoài; tổng tham số không đổi so với vòng 1 (dưới 200 triệu).
