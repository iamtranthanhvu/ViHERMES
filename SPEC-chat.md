# SPEC: Web chat thật trên pipeline B3, thay cho web demo giả

> Trạng thái: **Bản 3, đã chốt các câu mở theo yêu cầu "chạy thật, không giả"; chờ bạn xác nhận để sang kế hoạch.**
> Đây là spec riêng; `SPEC.md` (thí nghiệm B1–B4) giữ nguyên. Làm trên nhánh `feature/chat`
> (tách từ `baseline/3-lora`). `main` là code paper gốc, không đụng.

## 1. Objective

Thay `demo/index.html` (trang giả: câu trả lời soạn sẵn, so khớp từ khóa, các bước "đang suy nghĩ" chạy theo đồng hồ) bằng một **web chat thật**: người dùng gõ câu hỏi tiếng Việt, server chạy **pipeline B3 thật** và trả về câu trả lời có trích dẫn nguyên văn. **Giao diện giữ y như demo giả**: cùng HTML/CSS, sidebar, màn hình chào, thẻ câu hỏi mẫu, sáng/tối, ô nhập, khối "Nguồn", nút sao chép và 👍/👎.

**Chỉ dùng B3, không có cấu hình nào khác trong sản phẩm:**

- Retriever: BM25 + dense `BAAI/bge-m3` gốc (không fine-tune), đồ thị SQLite của corpus.
- Conductor: `Qwen/Qwen3.5-4B` + LoRA `iamtranthanhvu/vihermes-qwen35-4b-conductor-lora` qua vLLM.
- Interpreter và Auditor dùng cùng server vLLM (model gốc, không adapter), như khi đánh giá.
- Không cần judge, nên chỉ một server vLLM.

Mỗi câu hỏi **độc lập** (không nhớ hội thoại), như pipeline đã đánh giá.

### Điều hệ thống phải nói thật

Số đo B3 trên 25 câu test: **7/25 được trả lời, 18/25 bị từ chối**; F1 0,218, Judge 0,299 (nhiễu ±20 điểm). Vì vậy:

1. Từ chối là hành vi **bình thường, hiển thị rõ** với lý do tiếng Việt và các văn bản đã tìm thấy; không che bằng câu trả lời bịa. (Demo giả đã có cách hiển thị này; giữ nguyên.)
2. Corpus chỉ có **998 đoạn văn bản** của dataset ViHERMES, không phải toàn bộ luật y tế. Ghi rõ trên giao diện.
3. Công cụ tra cứu, **không phải tư vấn pháp lý**; giữ dòng ghi chú cố định dưới ô nhập như demo.
4. **Bỏ mọi dấu vết "mô phỏng"**: nhãn "Bản demo · dữ liệu mô phỏng" ở sidebar đổi thành tên model và số đoạn văn bản thật lấy từ `/api/health`.

### Ba kết quả (như `status` trong demo giả)

| Kết quả | Khi nào | Hiển thị |
| :-- | :-- | :-- |
| `answered` | Auditor chấp nhận | Câu trả lời, khối "Nguồn · N văn bản" (tiêu đề + đoạn trích nguyên văn) |
| `refused` | Auditor từ chối / Conductor bỏ qua | Lý do tiếng Việt + gợi ý hỏi cụ thể hơn + khối "Văn bản đã tìm thấy" (chỉ tiêu đề) |
| `clarification` | Interpreter thấy câu hỏi không phải về quy định | Câu hỏi làm rõ |

## 2. Giả định (sửa ngay nếu sai)

1. Chạy trên **host GPU thuê** (Vast.ai) hoặc máy có GPU. **Sản phẩm không có chế độ giả**: không model giả, không câu trả lời soạn sẵn, không tiến trình giả. Máy không có GPU thì không chạy được chat (và nói rõ lý do).
2. Adapter ghim theo commit `ca7768c356aa271d8c3a1337fe7204115728429e`; corpus là `contexts.csv` đã dựng, 998 đoạn.
3. Một người dùng hoặc nhóm nhỏ. Không đăng nhập, **không lưu gì phía server**.
4. **Lịch sử ở sidebar chỉ nằm trong bộ nhớ trình duyệt** (mất khi tải lại trang), như demo giả. Câu hỏi y tế là dữ liệu nhạy cảm nên không ghi `localStorage` hay server. 👍/👎 chỉ đổi trạng thái nút, không gửi đi đâu.
5. Backend dùng **`http.server` của stdlib** (3 endpoint, không thêm dependency). Trước đây tôi nghiêng về FastAPI; đổi vì không cần và khỏi phải xin thêm dependency.
6. Giao diện vẫn là **một file HTML tĩnh** (không framework, không build), do backend phục vụ.
7. Server xử lý **tuần tự** (hàng đợi, tối đa 3 câu chờ), vì chỉ một vLLM và ~10 giây/câu.

## 3. Commands

```bash
bash scripts/chat.sh            # host GPU: cài, tải adapter, bật vLLM + chat, in địa chỉ và lệnh SSH tunnel
bash scripts/chat.sh stop       # dừng tất cả
```

`scripts/chat.sh` chạy theo thứ tự, bước nào xong rồi thì bỏ qua (chạy lại an toàn):

1. Kiểm tra GPU và `python3` ≥ 3.10; thiếu thì dừng, nói rõ thiếu gì.
2. `make setup`.
3. Chưa có adapter ở `LORA_PATH`: hỏi **một lần** HF token READ bằng `read -rsp` (không hiện, không ghi file, không in), tải đúng revision ghim, rồi xóa biến. Đã có `HF_TOKEN` trong môi trường thì dùng luôn.
4. `scripts/serve.sh lora` rồi `vihermes chat --host 127.0.0.1 --port 8080` ở nền (`nohup`, log vào `outputs/logs/`).
5. Đợi `GET /api/health` trả `ok`, in `ssh -N -L 8080:127.0.0.1:8080 -p <PORT> root@<IP>` và địa chỉ `http://127.0.0.1:8080`.

Chat chỉ nghe `127.0.0.1`; mở ra ngoài cần hỏi trước. Mọi cổng/model/đường dẫn đọc từ `configs/host.env`, **không thêm file cấu hình mới**.

Kiểm thử (đây là hạ tầng test, **không phải** chế độ của sản phẩm; chỉ chạy trong CI/Mac để kiểm tra mã không cần GPU): `pytest tests/test_chat.py`, `make dryrun-chat`, `ruff check . && ruff format --check .`.

## 4. Giao diện API

| Method | Đường dẫn | Mô tả |
| :-- | :-- | :-- |
| `GET` | `/` | Trang chat tĩnh |
| `GET` | `/api/health` | `{"ok": true, "model": "...", "lora": "...", "corpus_units": 998, "mode": "b3"}`; `503` nếu vLLM chưa sẵn sàng |
| `POST` | `/api/ask` | `{"question": "..."}` → kết quả dưới đây |

```json
{
  "status": "answered | refused | clarification",
  "answer": "…",
  "sources": [{"id": "…", "title": "…", "quote": "…"}],
  "reason": "insufficient | quote | semantic | other",
  "seconds": 9.7
}
```

- Hình dạng `status` / `sources[{id,title,quote}]` / `reason` **giống dữ liệu của demo giả**, nên đoạn JS vẽ giao diện giữ nguyên, chỉ thay nguồn dữ liệu.
- `title` lấy từ node trong kho (`Node.title`); `quote` là đoạn nguyên văn Auditor đã xác minh cho đoạn đó; khi `refused` không cần `quote`.
- `status` suy từ `Answer.abstained` / `Answer.reasons`. `reason` ánh xạ từ mã kỹ thuật (`semantic_grounding_failed` → `semantic`, trích dẫn không nguyên văn → `quote`, thiếu bằng chứng/trích dẫn → `insufficient`); mã lạ → `other` và giao diện hiện câu chung, không lộ chuỗi thô.
- Lỗi: `422` câu hỏi rỗng hoặc > **500 ký tự** (khớp `maxlength` của ô nhập); `429` hàng đợi đầy; `503` vLLM không phản hồi; `500` không kèm stack trace. Giao diện hiện thông báo tiếng Việt cho từng mã.
- `trace` không trả ra ngoài (chỉ khi `--debug`).

## 5. Project Structure

```text
src/vihermes/chat/__init__.py
src/vihermes/chat/service.py          Dựng pipeline B3 một lần; ask(question) -> dict (chỉ gọi Conductor.run)
src/vihermes/chat/server.py           HTTP stdlib, validate, hàng đợi, ánh xạ lý do
src/vihermes/chat/static/index.html   Chuyển từ demo/index.html, bỏ dữ liệu giả, gọi /api/ask
src/vihermes/cli.py                   Thêm lệnh con `chat`, dùng lại pipeline(), conductor(), language_model()
scripts/chat.sh
make/chat.mk                          dryrun-chat
tests/test_chat.py
docs/chat.md                          Cách chạy, phạm vi, giới hạn
demo/index.html                       XÓA ở task cuối, sau khi bản thật đã chạy được
```

Phần phải bỏ khỏi trang: `ITEMS` (JSON soạn sẵn), `match()`/`resolve()`, `STEPS` chạy theo đồng hồ, `wait()` giả lập, nhãn "mô phỏng". Phần giữ: toàn bộ CSS, markup, theme, sidebar, hoạt ảnh gõ chữ, `sourcesUI`, `actionsUI`, `REASONS`, `HINT`.

## 6. Code Style

Python ≥3.10, ruff (dòng 100, E4/E7/E9/F/I), hàm nhỏ, ít chú thích, chỉ giải thích "vì sao".

```python
def classify(answer):
    if "clarification_requested" in answer.reasons:
        return "clarification"
    return "refused" if answer.abstained else "answered"
```

Trang web: chỉ dùng `textContent` cho nội dung do model hoặc người dùng sinh. Không `innerHTML` với chúng (demo giả chỉ dùng `innerHTML` cho icon SVG cố định; giữ nguyên chỗ đó).

## 7. Testing Strategy

pytest theo mẫu `tests/test_core.py`; không cần GPU hay mạng.

| Mức | Nội dung |
| :-- | :-- |
| Đơn vị | `classify`, ánh xạ lý do (đã biết và lạ), cắt độ dài, gắn `title`/`quote` cho nguồn |
| API | Conductor giả (`--generator extractive`, embedding `hash`) trên `tests/fixtures/dryrun`: hợp lệ, rỗng, > 500 ký tự, JSON sai, vLLM chết, hàng đợi đầy |
| An toàn đầu ra | Trả lời chứa `<script>`: JSON giữ nguyên; kiểm tra bằng grep rằng trang không `innerHTML` nội dung động |
| Không còn dữ liệu giả | Test grep: `static/index.html` không chứa `ITEMS`, `resolve(`, "mô phỏng" |
| Từ đầu đến cuối | `make dryrun-chat` thoát mã 0 |
| Thủ công trên host | Mở trang, hỏi 5 câu từ tập test: thấy đủ `answered` / `refused` / `clarification`, ≤ 20 giây/câu, giao diện khớp demo giả (so ảnh chụp trước/sau, sáng và tối, mobile) |

Không test chất lượng câu trả lời: đã đo bằng B3 trên 25 câu.

## 8. Boundaries

- **Always:**
  - Giữ disclaimer, phạm vi 998 đoạn, và giữ nguyên bước kiểm tra của Auditor.
  - Mọi câu `answered` có nguồn và trích đoạn nguyên văn.
  - `HF_TOKEN` chỉ trong bộ nhớ script, nhập bằng `read -rsp`.
  - Chạy `pytest` + `ruff` trước mỗi commit; commit trên `feature/chat`.
- **Ask first:**
  - Thêm dependency.
  - Lưu câu hỏi (log, lịch sử server, `localStorage`).
  - Mở cổng ra Internet hoặc thêm xác thực; thêm nhớ hội thoại.
  - Đổi model, adapter, revision; đổi giao diện khác với demo giả.
- **Never:**
  - Sửa `main`, hay sửa logic Interpreter/Pathfinder/Auditor cho câu trả lời "tốt hơn".
  - Hạ ngưỡng hoặc bỏ Auditor để giảm từ chối.
  - Đưa câu trả lời soạn sẵn hoặc tiến trình giả vào bản thật.
  - Thêm chế độ giả/demo vào sản phẩm; `service.py` luôn dựng đúng pipeline B3 và từ chối khởi động nếu vLLM hoặc adapter thiếu.
  - Commit token hay đưa token vào `configs/host.env`.

## 9. Success Criteria

1. `make dryrun-chat` thoát mã 0 trên Mac, không GPU, không mạng.
2. Trên host GPU mới, `bash scripts/chat.sh` đưa chat lên trong ≤ 15 phút (gồm tải model lần đầu), `/api/health` trả `ok` với `mode: "b3"`; chạy lại không lỗi, không làm thừa.
3. Hỏi **một câu chưa từng có trong demo giả** vẫn nhận câu trả lời/từ chối thật từ pipeline (chứng minh không còn so khớp soạn sẵn).
4. Ba loại kết quả đều hiển thị đúng; câu `refused` có lý do tiếng Việt và danh sách văn bản.
5. "chào bạn" → `clarification`. Câu rỗng hoặc 501+ ký tự → `422`, server vẫn chạy.
6. Không có mã lý do thô, stack trace hay chuỗi "mô phỏng" trên trang.
7. Giao diện giống demo giả ở mọi trạng thái chính (chào, đang chờ, trả lời, từ chối, sáng/tối, mobile).
8. Token không xuất hiện trong log, `ps`, lịch sử shell, file repo.
9. pytest và ruff qua; `main` không đổi; `demo/index.html` đã được thay.

## 10. Các quyết định đã chốt (từ yêu cầu "chạy baseline thật")

1. **Tiến trình khi chờ:** một dòng chờ trung thực "Đang tra cứu…" kèm số giây đã chờ. Bỏ ba bước chạy theo đồng hồ (đó là tiến trình giả). Không sửa pipeline để phát sự kiện.
2. **Thẻ câu hỏi mẫu:** giữ 12 câu từ tập test, bấm là gửi lên pipeline thật; có thể bị từ chối, đúng như B3 đo.
3. **Không có chế độ thử trong sản phẩm** (mục 2.1). Test dùng Conductor giả chỉ trong pytest.
4. **Không đưa ra Internet công khai**; chỉ SSH tunnel. Cần mở thì làm spec riêng.
5. **Disclaimer chỉ tiếng Việt**, như demo giả.

Nếu bạn muốn khác điều nào ở trên, nói ngay; không thì sang bước lập kế hoạch (`tasks/plan.md`).
