# HƯỚNG DẪN DUYỆT human_queue.csv — làm từng bước (10-15 phút/tuần)

## Queue là gì?
Mọi quyết định tự động của hệ thống khử trùng tên đều nằm trong file này để bạn kiểm tra.
Máy KHÔNG tự thay người ở những dòng P1/P2 — đây chính là phần phán đoán pháp lý.

## File: `human_queue.csv` (mở bằng Excel/Google Sheets — phân cách bằng dấu `;`)
Cột: `priority | method | score | name_a | name_b | class | action`

## Duyệt theo 4 mức ưu tiên

### 🔴 P1_review_merge_no_support (184 dòng) — QUAN TRỌNG NHẤT
Đây là các cặp hệ thống ĐÃ GỘP nhưng không có bằng chứng đồ thị (không dùng chung chunk/văn bản).
**Cách duyệt**: đọc cặp tên ở name_a/name_b, tự hỏi: "Đây có phải CÙNG MỘT thực thể/khái niệm không?"
- Đúng cùng một → không làm gì (merge giữ nguyên)
- Khác nhau (như `chương VII` vs `Chương VIII` — loại này guard đã chặn, nhưng còn sót case lạ) → đánh dấu `REJECT` ở cột action
**Mẹo nhanh**: chỉ cần nhìn name_a vs name_b khác nhau ở Con số/La Mã/từ phủ định (không/không-hại/hủy/hoãn) → gần như chắc chắn phải REJECT.

### 🟠 P2_conflicts (21 dòng) — soi kỹ
Các cụm entity bị gộp chéo class (vd 1 cụm vừa là Organ vừa là Concept) hoặc cụm >4 thành viên.
Quyết định: tách / giữ / sửa class.

### 🟡 P3_rejected_skim (294 dòng) — lướt nhanh
Cặp hệ thống quyết định KHÔNG gộp (LLM nói khác nhau + đã re-check bằng ngữ cảnh: 81 confirmed).
Sai ở đây chỉ là "thiếu 1 node", không nguy hiểm. Lướt tìm case hiển nhiên như `kim khí quí/quý` nếu muốn cứu thêm.

### 🟢 P4_auto_ok (352 dòng) — khỏi duyệt
Merge đã có topology/context chứng minh. Chỉ mang tính minh bạch.

## Sau khi duyệt xong
1. Tạo file `approved_pairs.jsonl` (mỗi dòng: `{"name_a": "...", "name_b": "...", "reason": "human-verified"}`)
   cho các cặp bạn CHẤP NHẬN gộp mà hệ thống chưa gộp.
2. Đặt file vào `~/vicivil_tkg/graph/` trên A30 → chạy `python3 -m vtg.run_all --llm` → cơ chế tier0
   sẽ gộp đúng các cặp bạn duyệt, ghi provenance "human-approved", rollback được.
3. Các cặp bạn REJECT ở P1 → đưa vào `rejected_by_human.jsonl` → build sau sẽ tách riêng.

## Bộ ảnh minh họa (figures/)
| Ảnh | Nội dung |
|---|---|
| fig1 | Timeline hiệu lực 33 văn bản 1995-2027 |
| fig2 | Đồ thị tri thức tại 3 mốc as_of (2005/2015/2026) |
| fig3 | Funnel khử trùng tên 14,099→13,857 + merge theo tầng |
| fig4 | Continual embedding: 6 chiến lược × 3 kỷ nguyên |
| fig5 | Sơ đồ pipeline realtime (cron 6h đang chạy) |
| fig6 | Headline benchmark v1 (TCR, Cite-Acc) |
| fig7 | 97.7% cạnh entity có interval hiệu lực |
| fig8 | 157 amendment facts cấp ĐIỀU theo luật đích/năm |
| fig9 | Phễu thẩm định 1,151 cặp (guard/LLM/context) |
| fig10 | Donut 4 mức ưu tiên hàng chờ (851 dòng) |

## Quy ước an toàn của hệ thống (đã verify)
- KHÔNG BAO GIỜ đoán ngày hiệu lực / không gộp khi thiếu bằng chứng → chỉ cần người cho P1/P2
- Mọi merge có `merge_log.jsonl` (method + confidence + evidence) → rollback được
- Mọi artifact có SHA256 trong `SHA256SUMS`; thay đổi ghi `CHANGES.json`
- SHACL conforms=True; drift 0 predicate; reproduced GPU-free
