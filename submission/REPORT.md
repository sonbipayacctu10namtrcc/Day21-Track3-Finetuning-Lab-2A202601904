# Lab 21 — Evaluation Report

**Họ tên**: Hà Xuân Sơn  **MSSV**: 2A202601904  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16 GB (Colab Free, ~14.6 GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định, 250 mẫu train) |
| Train / val | 225 / 25 (seed 42, `train_frac=0.9`) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*: `rendered` cho thấy
khối `<think>...</think>` rỗng vẫn được giữ nguyên trong generation prompt, `verdict`:
"reasoning preserved — safe to train on traces".

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3158 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1051 |
| (c) LoRA fine-tune | 0.965 | 0.522 | 1.000 | 1415 |

**(b) có thật sự mạnh hơn (a) không?** Có, rõ rệt — chỉ đổi prompt (cùng model, chưa train)
đã đưa `target` từ 0.000 lên 0.765 và `format` từ 0.000 lên 1.000, đồng thời latency giảm
gần 3 lần (3158 → 1051 ms, vì model không còn lan man mà trả JSON gọn ngay). Đây là bằng
chứng prompt engineering một mình đã đi được phần lớn quãng đường trước khi cần fine-tune.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** Không — `optimized_prompt_sha` ghi trong
`baselines_frozen.json` (`719e74d3b6232053`) khớp đúng với sha của `OPTIMIZED_PROMPT` hiện
tại trong `src/labkit/generate.py`, tức là prompt đóng băng dùng để đo (b) chính là prompt
gốc của repo, không bị chỉnh sau khi biết kết quả fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6276 | 0.965 | 905.3 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5370 | 0.970 | 807.3 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 942.9 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1005.0 | 7.09 |

> Cột **target** và cột **train loss** ở đây cho **cùng thứ tự** giữa `correct` và
> `attn_only` (attn_only thấp loss hơn *và* target cao hơn) — không phải trường hợp hai cột
> mâu thuẫn nhau như deck cảnh báo, nhưng chính vì vậy nó là kết quả đáng nói ở 4.1.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct` (32,456,704 so với
32,464,896 — lệch chưa tới 0.03%, đúng thiết kế "matched"). Trên tập target nó không thua:
0.970 so với 0.965 của `correct`, tức là hoà (chênh 0.005 = đúng 1/50 ticket, nằm trong
nhiễu đo lường của bộ eval 50 mẫu). Thứ tự đó khớp với thứ tự theo train loss
(`attn_only` 0.537 < `correct` 0.628, tức loss thấp hơn). Điều này KHÔNG ủng hộ câu chuyện
"vị trí gắn adapter quan trọng hơn rank" mà deck ngụ ý ở §10.2 — ít nhất là trên bài toán
JSON-triage đơn giản này: khi đã khớp số tham số huấn luyện, chỉ gắn vào `q,v` (attention)
cho kết quả không phân biệt được với gắn vào toàn bộ lớp linear. Kết luận thận trọng hơn:
trên tác vụ này, *lượng tham số huấn luyện* mới là biến quan trọng đo được rõ ràng nhất;
*vị trí* gắn có thể chỉ quan trọng với tác vụ khó hơn hoặc bộ eval lớn hơn 50 mẫu.**

**4.2 — `wrong_lr` chỉ đổi đúng learning rate (1e-4 → 1e-5, tức 1/10). Train loss cuối cao
hơn hẳn: 1.5704 so với 0.6276 của `correct` — mô hình chưa kịp học trong 30 step với bước
nhảy nhỏ như vậy. Nhưng đây chỉ là phần nổi: `target` rơi thẳng xuống 0.000 và `format`
cũng xuống 0.000 — mô hình không sinh ra được JSON hợp lệ nữa, và latency phình lên
5098 ms (gấp 3,6 lần `correct`) vì nó sinh output dài, lan man thay vì JSON gọn. Nếu chỉ
nhìn con số loss 1.57 mà không biết LR đứng sau nó, người ta dễ kết luận nhầm rằng "mô hình
học chậm nhưng vẫn học được, chỉ cần train thêm step" — trong khi thực tế là một sụp đổ
hoàn toàn về format và task, loss một mình không phản ánh được điều đó.**

**4.3 — `qlora` tiết kiệm 12.01 − 7.09 = 4.92 GB VRAM đỉnh (~41%), nhưng trả giá bằng
`target` thấp hơn (0.940 so với 0.965, −0.025), loss cuối cao hơn (0.7058 so với 0.6276),
và latency chậm hơn (1765 ms so với 1415 ms, +25%, do chi phí dequantize 4-bit lúc sinh).
Trên tier T4 này, `correct` (16-bit) đã chạy vừa trong 12.01 GB — tức là T4 free (~14.6 GB
khả dụng) không hề cần QLoRA để fit. Vì vậy số đo của tôi ủng hộ khuyến nghị "không dùng
QLoRA cho dòng model này" **trên tier T4**: QLoRA mua một khoảng đệm VRAM không ai cần,
đổi lấy một khoản phí chính xác và tốc độ nhỏ nhưng nhất quán. Trên tier hẹp VRAM hơn
(LAPTOP 8 GB) mà không có 4-bit thì có thể không fit được — lúc đó đánh đổi này mới đáng
cân nhắc lại.**

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.236` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ): Bản fine-tune thắng áp đảo trên đúng tác vụ nó được train: `target`
tăng từ 0.765 (baseline (b), prompt đã tối ưu) lên 0.965 (+0.200). Nhưng cổng hồi quy vẫn
báo FAILED, vì năng lực tổng quát (15 câu hỏi kiến thức/chỉ dẫn phổ thông, đo bằng
`regression`) sụt từ 0.758 xuống 0.522 — giảm 0.236, gấp hơn 10 lần ngưỡng cho phép 0.020.
Đây là catastrophic forgetting kinh điển: chỉ 30 step / 2 epoch trên 225 mẫu JSON-triage
hẹp đã kéo hành vi model quá mạnh về phía "luôn trả JSON gọn", làm hỏng khả năng trả lời
câu hỏi ngoài miền dữ liệu train. (`valid_trace_rate = 0.00` không phải tín hiệu xấu riêng
ở đây — corpus mặc định không có reasoning trace nào để giữ lại, nên chỉ số này luôn bằng 0
với mọi run, kể cả `correct`.) Đây đúng là lý do lab có cổng hồi quy: nếu chỉ nhìn `target`,
tôi đã đóng gói và ship checkpoint này như một chiến thắng rõ ràng.

---

## 6. Định tính — bắt buộc có cả ca THUA

> **Cần hoàn tất trước khi nộp**: `results/qualitative.json` chỉ lưu prediction của (c),
> không lưu prediction của (b) cho từng ticket — phải tự sinh lại (b) cho các dòng dưới rồi
> mới đánh dấu ✅/❌ đúng. 7 ticket duy nhất mà (c) không đạt điểm tuyệt đối (ft_score=0.75,
> tức đúng 3/4 trường) là index **3, 5, 12, 26, 39, 41, 46** trong `eval_target.jsonl` —
> **6/7 trong số đó sai đúng một kiểu: nhãn thật `urgency=thap`, model đoán `trung_binh`**
> (ngoại lệ là index 26: sai `intent` — nhãn `hoan_tien`, model đoán `doi_tra`, ticket dùng
> từ "trả lại" vốn mơ hồ giữa đổi/trả và hoàn tiền). Đây là dấu hiệu chung đáng viết vào ô
> "mẫu chung" bên dưới dù chưa biết (b) làm gì với các ticket này.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "...đặt chuột không dây... Cho tôi trả lại..." (i=0) | `doi_tra / cao / chuột không dây / tich_cuc` | `<chạy lại (b), xem hướng dẫn>` | đúng cả 4 trường (ft_score=1.0) | |
| 2 | "...đặt ốp lưng điện thoại... Hoàn tiền..." (i=1) | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | `<chạy lại (b)>` | đúng cả 4 trường (ft_score=1.0) | |
| 3 | "...đặt bình giữ nhiệt... Chưa thấy tiền..." (i=3) | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | `<chạy lại (b)>` | sai `urgency` → `trung_binh` (ft_score=0.75) | |
| 4 | "...đặt nồi chiên không dầu... Thiếu phụ kiện..." (i=5) | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | `<chạy lại (b)>` | sai `urgency` → `trung_binh` (ft_score=0.75) | |
| 5 | "...đặt máy xay sinh tố... Trả lại tiền..." (i=26) | `hoan_tien / thap / máy xay sinh tố / trung_tinh` | `<chạy lại (b)>` | sai `intent` → `doi_tra` (ft_score=0.75) | |

Có mẫu chung nào ở các ca FT thua không? **Có** — 6 trên 7 ticket mà fine-tune không đạt
điểm tuyệt đối đều lệch đúng một hướng: model đoán `urgency=trung_binh` trong khi nhãn thật
là `thap`, bất kể sản phẩm hay nội dung ticket là gì. Model có vẻ học được một prior lệch
("mặc định là trung bình") thay vì đọc tín hiệu khẩn cấp thật trong câu, có thể vì tập train
225 mẫu không đủ đa dạng cách diễn đạt "không gấp" để model phân biệt chắc `thap` khỏi
`trung_binh`.

*(Chạy snippet dưới trong Colab — cần model + tokenizer + `generate.OPTIMIZED_PROMPT` vẫn
còn trong session — để lấy cột (b) cho 5 ticket trên, rồi điền nốt và đổi Nhận xét thành
✅ FT thắng / ❌ FT thua cho từng dòng:*

```python
import json, pathlib
from labkit import evaluate as ev, generate

idx = [0, 1, 3, 5, 26]
target = [json.loads(l) for l in open("data/eval_target.jsonl", encoding="utf-8")]
rows = [target[i] for i in idx]
preds_b, _ = generate.generate_batch(model, tok, [r["input"] for r in rows],
                                      system=generate.OPTIMIZED_PROMPT, label="(b) qual check")
for i, p, r in zip(idx, preds_b, rows):
    s_b = ev.triage_field_accuracy(p, r["label"])
    print(i, "b_score=", round(s_b, 2), "|", p.replace("\n", " ")[:90])
```
*)*

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bản fine-tune này **không nên deploy nguyên trạng**. Nó thắng rõ
ràng trên đúng việc nó được train để làm — JSON triage của ticket CSKH, +20 điểm target so
với baseline đã prompt tốt — nhưng đánh đổi là một khoản mất năng lực tổng quát vượt xa
ngưỡng chấp nhận được (−0.236 so với tolerance 0.020), nghĩa là model này sẽ trả lời tệ hơn
hẳn cho bất kỳ câu hỏi nào ngoài JSON triage, kể cả những câu đơn giản mà bản gốc trả lời
tốt. Đòn bẩy thật sự trong lab này, theo đúng thứ tự đo được: (1) **learning rate** — sai
một số 10 lần (`wrong_lr`) sập cả target lẫn format về 0, cú sập mạnh nhất trong cả bốn run;
(2) **lượng tham số huấn luyện**, không phải vị trí gắn adapter — `attn_only` khớp tham số
với `correct` thì cho kết quả tương đương, không thua; (3) **chất lượng/độ đa dạng dữ liệu**
— lỗi hệ thống "đoán urgency=trung_binh" ở 6/7 ca sai của (c) cho thấy 225 mẫu train chưa đủ
đa dạng cách diễn đạt mức độ khẩn cấp; (4) **mask** đã đúng ngay từ đầu (supervised_fraction
0.41, câu hỏi bị mask hoàn toàn) nên không phải nguyên nhân của regression lần này. Điều
quan trọng nhất lab này chứng minh được: đo một mình `target` là không đủ — nếu không có
cổng hồi quy, một checkpoint sụp đổ năng lực tổng quát vẫn trông như một chiến thắng.

**Ba điều tôi học được** (cụ thể, không generic):
1. Khi so khớp số tham số huấn luyện giữa hai cách gắn adapter khác nhau (`attn_only` r=283
   so với `text-linear` r=16), sự khác biệt về *vị trí* gần như biến mất trên một tác vụ đủ
   hẹp/dễ — số tham số mới là biến chi phối rõ nhất trên dữ liệu này, không phải câu chuyện
   "attention quan trọng hơn" mà tôi hình dung trước khi đo.
2. Loss huấn luyện một mình có thể đánh lừa hoàn toàn: `wrong_lr` có loss cuối 1.57 — nghe
   như "học chậm" — nhưng thực chất là model mất khả năng sinh JSON hợp lệ (`format=0`),
   một sụp đổ chức năng không thấy được nếu chỉ nhìn đường loss.
3. Một checkpoint có `target` tăng vẫn có thể là một checkpoint tệ hơn để ship — cổng hồi
   quy (regression gate) không phải thủ tục hình thức, nó bắt đúng trường hợp catastrophic
   forgetting mà lab này đo được thật (−0.236, gấp hơn 10 lần ngưỡng).

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% dữ liệu "replay" (câu hỏi kiến thức chung,
đúng gợi ý ở deck §14.3) vào tập train để xem có kéo `regression` về trong ngưỡng mà không
làm hỏng `target` không, và thử thu hẹp lỗi hệ thống `urgency=thap→trung_binh` bằng cách bổ
sung thêm ví dụ train có nhãn `thap` diễn đạt theo nhiều cách khác nhau.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
