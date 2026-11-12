# Kế hoạch: Web chat thật trên pipeline B3

Spec: `SPEC-chat.md` (đã duyệt). Danh sách task: `tasks/chat/todo.md`.
Kế hoạch B1–B4 (`tasks/plan.md`, `tasks/todo.md`) giữ nguyên, không liên quan.
Nhánh: `feature/chat`. Commit chỉ khi người dùng yêu cầu.

## Tổng quan

Thay `demo/index.html` (giả) bằng một chat chạy pipeline B3 thật. Backend mỏng bọc `Conductor.run` đã có; giao diện là chính file demo, bỏ dữ liệu giả và gọi `/api/ask`. Không sửa logic Interpreter, Pathfinder, Auditor.

## Đã đọc trong code (căn cứ cho kế hoạch)

- `cli.py` đã có `pipeline(args, store, stack)`, `conductor(args, pathfinder)`, `language_model(...)`: `chat` dùng lại, không viết lại. B3 chạy bằng `--embedding sentence-transformers --embedding-model $(EMBED_BASE) --conductor-model $(LORA_NAME)` (`make/b3.mk`).
- `Conductor.run(question)` trả `(Answer, Retrieval)`:
  - `Answer.abstained` + `reasons`; `clarification_requested` khi Interpreter thấy không phải câu quy định.
  - `Answer.sources` chỉ có khi trả lời (lọc theo id được trích). Khi từ chối, `sources=[]` nên **danh sách "văn bản đã tìm thấy" phải lấy từ `Retrieval.evidence`**.
  - Claim có `quotes{id: đoạn}` → là `quote` của nguồn.
- `Node.title` có trong kho (`models.py`) → là `title` của nguồn.
- `Conductor` không thread-safe chưa được kiểm chứng → server **tuần tự hóa** bằng khóa.

## Đồ thị phụ thuộc

```
service.classify + map_reason + build_sources   (thuần, test được)
        │
        ├── service.ask  ── dựng pipeline B3 ──┐
        │                                      │
        └── server (HTTP, validate, khóa) ─────┤── cli `chat`
                                               │
                        scripts/chat.sh ◄──────┘   (host GPU: vLLM + adapter + chat)
                               │
                  static/index.html (từ demo, gọi /api/ask)
                               │
                  hoàn thiện lỗi/mobile, xóa demo/index.html, docs
```

## Cắt dọc

Mỗi task kết thúc ở trạng thái chạy được và kiểm chứng được. Rủi ro cao (chạy B3 thật trên GPU) đưa lên **trước giao diện**, để lỗi lộ ra sớm bằng `curl`.

| Phase | Task | Kết quả chạy được |
| :-- | :-- | :-- |
| 1 Lõi | 1, 2 | `POST /api/ask` trả đúng hình dạng spec (Conductor giả trong test) |
| 2 Host thật | 3 | Một lệnh bật B3 thật; `curl` hỏi được câu thật |
| 3 Giao diện | 4, 5 | Trang demo cũ chạy trên dữ liệu thật; demo giả bị xóa |
| 4 Đóng | 6 | docs, `dryrun-chat`, checklist tiêu chí thành công |

## Quyết định kiến trúc

- `http.server` stdlib (`ThreadingHTTPServer`) + một `threading.Lock` quanh `ask`, hàng đợi tối đa 3 (đếm số yêu cầu đang chờ, vượt thì `429`). Không thêm dependency.
- `service.py` nhận Conductor đã dựng (`Service(conductor, meta)`), nên test tiêm Conductor giả; hàm dựng B3 thật nằm trong `cli.py`/`service.build`.
- `chat` **không có cờ giả**: dùng `--generator openai` cố định, thiếu vLLM thì thoát với lỗi rõ. Test không đi qua `chat` mà tạo `Service` trực tiếp.
- Tiến trình chờ ở UI: dòng "Đang tra cứu…" + đồng hồ giây, tính ở client (không giả bước).
- `index.html` đưa vào `static/` giữ nguyên CSS/markup; chỉ thay khối script dữ liệu và `ask()`.

## Rủi ro

| Rủi ro | Ảnh hưởng | Giảm thiểu |
| :-- | :-- | :-- |
| Không có GPU để chạy B3 thật trong phiên này | Cao | Task 3 có bước **người dùng chạy trên host**; trước đó mọi thứ kiểm bằng test + `curl` với Conductor giả; Checkpoint 2 chờ người dùng |
| `Conductor` dùng chung giữa luồng không an toàn | Trung | Khóa tuần tự; test 2 yêu cầu song song |
| Model trả JSON hỏng → `500` | Trung | Pipeline đã từ chối khi Auditor nhận JSON sai (commit `bc60c2d`); server bắt ngoại lệ còn lại → `500` không lộ stack |
| Sửa `index.html` làm lệch giao diện so với demo | Trung | Chụp ảnh trước/sau (sáng/tối/mobile) ở Task 4–5 |
| Câu hỏi y tế bị ghi vào log | Trung | Log chỉ ghi mã trạng thái và thời gian, **không ghi nội dung câu hỏi**; test kiểm tra |
| `chat.sh` làm rò HF token | Cao | `read -rsp`, `unset` sau khi tải; test kiểm bằng `bash -x` giả lập không thấy token |

## Câu hỏi cần người dùng (không chặn Phase 1)

- Host GPU nào dùng để thử Task 3 (Vast.ai mới hay host B3 cũ còn)? Nếu host cũ còn, `chat.sh` bỏ qua phần lớn bước cài.
