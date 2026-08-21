# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hai điều. Thứ nhất, baseline (b) — chỉ đổi prompt, chưa train gì — đã kéo `target` từ 0.000
lên 0.765 và `format` từ 0.000 lên 1.000. Phần lớn "phép màu" mà tôi nghĩ là nhờ fine-tune
hoá ra đến từ prompt engineering. Thứ hai, ở NB4: `attn_only` (chỉ gắn adapter vào q,v,
nhưng nâng rank để khớp đúng số tham số huấn luyện với `correct`) không hề thua `correct`
— thậm chí nhỉnh hơn một chút (0.970 so với 0.965). Tôi cứ đinh ninh vị trí gắn adapter
(deck nhấn mạnh all-linear tốt hơn attention-only) mới là yếu tố quyết định; số đo thật lại
nói rằng trên bài toán này, số tham số huấn luyện mới là biến ảnh hưởng rõ nhất.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở GPU hay train — mà ở bước Smoke (NB `verify.py --smoke`) trên Colab, trước khi
chạm tới GPU: 3 unit test fail với `ModuleNotFoundError: No module named 'tests.fake_tokenizer'`,
dù cùng bộ test đó pass 100% trên máy cá nhân. Nguyên nhân gần như chắc chắn là một package
trùng tên `tests` đã có sẵn trong site-packages của Colab, che khuất namespace package
`tests/` của repo (vì thư mục đó không có `__init__.py`). Fix là thêm một file
`tests/__init__.py` rỗng để biến nó thành package thật, ưu tiên tìm thấy trước. Tôi không dự
đoán trước sẽ mất thời gian ở đây — nghĩ vướng mắc đầu tiên sẽ là OOM hay lệch package ML
(torch/peft/bitsandbytes), không phải một quirk import-resolution thuần Python.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ "target tăng" gần như đồng nghĩa với "model tốt hơn, có thể ship". NB5 cho
thấy rõ đó là hai trục khác nhau: bản fine-tune của tôi tăng target +0.200 nhưng đồng thời
regression (năng lực tổng quát) sụt −0.236 — gấp hơn 10 lần ngưỡng cho phép — nên cổng hồi
quy báo FAILED. Nếu không có bước NB2 đóng băng eval + NB5 chấm hai baseline trước, tôi đã
đóng gói và tự tin rằng đây là một chiến thắng rõ ràng.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng Claude Code (Claude trong VS Code) để: (a) lên kế hoạch chạy trên Colab web từng bước
(chọn GPU, form NB3, quản lý ngân sách thời gian); (b) chẩn đoán và fix lỗi
`ModuleNotFoundError: tests.fake_tokenizer` ở mục 2; (c) đọc trực tiếp các file trong
`results/` (`token_stats.json`, `mask_proof.json`, `baselines_frozen.json`, `runs.csv`,
`autopsy.json`, `verdict.json`, `qualitative.json`) và soạn phần lớn `REPORT.md` từ đúng số
đo, không bịa. Chỗ nó sai/chưa hoàn chỉnh: nguyên nhân gốc của lỗi `tests.fake_tokenizer` chỉ
là một giả thuyết hợp lý (package `tests` trùng tên trong site-packages của Colab che khuất
namespace package của repo) — nó chưa bao giờ được xác nhận bằng lệnh chẩn đoán thật
(`import tests; print(tests.__file__)`), fix áp dụng trước rồi mới hoạt động, chứ không chứng
minh chắc chắn thủ phạm cụ thể là package nào. Ngoài ra, ở mục 6 REPORT.md, nó không thể tự
điền cột "(b) prompt" vì `qualitative.json` không lưu prediction của baseline (b) theo từng
ticket — AI chỉ tổ chức được đúng phần dữ liệu đã tồn tại, không tự sinh ra phần thiếu.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng tập eval và đo baseline đã prompt-engineer kỹ **trước khi** train bất cứ gì —
đúng thứ tự NB2 làm trong lab này. Lab đã cho thấy prompt tốt một mình có thể đi được phần
lớn quãng đường, và nếu đo baseline sau khi đã thấy kết quả fine-tune, tôi sẽ vô thức chỉnh
prompt cho tới khi mình thắng — tự lừa chính mình về việc fine-tune có thật sự cần thiết hay
không.
