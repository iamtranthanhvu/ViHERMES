# Task list: Web chat thật trên B3

Kế hoạch: `tasks/chat/plan.md`. Spec: `SPEC-chat.md`.
Lệnh kiểm chứng chung: `pytest -q tests/test_chat.py` và `ruff check src tests && ruff format --check src tests` (offline, không GPU). Commit chỉ khi người dùng yêu cầu.

---

## Phase 1: Lõi backend

### Task 1: `service.py`: từ `Answer` thành JSON của spec
**Description:** Hàm thuần `classify`, `map_reason`, `build_sources` và lớp `Service(conductor, meta).ask(question)` trả `{status, answer, sources[{id,title,quote}], reason, seconds}`. Nguồn khi trả lời lấy `quote` từ `claims.quotes`; khi từ chối lấy danh sách văn bản từ `Retrieval.evidence` (chỉ `id`, `title`).
**Acceptance criteria:**
- [x] `answered` có ≥1 nguồn, mỗi nguồn có `title` (từ `Node.title`) và `quote` nguyên văn.
- [x] `refused` có `reason` ∈ {insufficient, quote, semantic, other} và nguồn chỉ có `id`+`title`.
- [x] `clarification_requested` → `status: "clarification"`.
- [x] Mã lý do lạ → `other`, không lộ chuỗi thô.
- [x] Không còn `trace` trong kết quả.
**Verification:**
- [x] `pytest -q tests/test_chat.py -k service` (Conductor giả, corpus `tests/fixtures/dryrun`)
- [x] `ruff check src tests`
**Dependencies:** None
**Files:** `src/vihermes/chat/__init__.py`, `src/vihermes/chat/service.py`, `tests/test_chat.py`
**Scope:** Small

### Task 2: `server.py` + lệnh `vihermes chat`: API chạy được
**Description:** HTTP stdlib phục vụ `GET /`, `GET /api/health`, `POST /api/ask`. Validate (`422` rỗng hoặc >500 ký tự, JSON sai), khóa tuần tự + hàng đợi tối đa 3 (`429`), `503` khi vLLM không phản hồi, `500` không stack. Lệnh `chat` trong `cli.py` dựng B3 (embedding bge-m3, `--conductor-model`) và **thoát lỗi rõ nếu vLLM/adapter thiếu**.
**Acceptance criteria:**
- [ ] `curl -XPOST /api/ask -d '{"question":"…"}'` trả đúng hình dạng spec mục 4.
- [ ] Các mã lỗi 422/429/503/500 đúng; server vẫn sống sau mỗi lỗi.
- [ ] Hai yêu cầu đồng thời không chạy `ask` cùng lúc (test bằng Conductor giả có độ trễ).
- [ ] Log không chứa nội dung câu hỏi.
- [ ] `vihermes chat` không có tùy chọn nào chọn chế độ giả/extractive.
**Verification:**
- [ ] `pytest -q tests/test_chat.py -k "api or queue or log"`
- [ ] `ruff check src tests`
- [ ] Thủ công: khởi chạy server với `Service` giả trong script test, `curl` ba kết quả
**Dependencies:** Task 1
**Files:** `src/vihermes/chat/server.py`, `src/vihermes/cli.py`, `tests/test_chat.py`
**Scope:** Medium

### Checkpoint 1: sau Task 1–2
- [ ] pytest + ruff qua
- [ ] `curl` ra đúng JSON ở cả ba trạng thái
- [ ] Duyệt với người dùng trước khi sang host

---

## Phase 2: B3 thật trên host GPU (rủi ro cao, làm trước giao diện)

### Task 3: `scripts/chat.sh`: một lệnh bật B3 thật
**Description:** Theo spec mục 3: kiểm GPU/python, `make setup`, tải adapter (hỏi HF token bằng `read -rsp` khi cần, ghim revision `ca7768c…`), `scripts/serve.sh lora`, bật `vihermes chat` nền, đợi `/api/health`, in địa chỉ + lệnh SSH tunnel; `stop` dừng cả hai. Bước nào xong thì bỏ qua. Thêm `LORA_FT_REVISION` vào `configs/host.env`.
**Acceptance criteria:**
- [ ] Chạy lần hai không làm gì thừa, không lỗi.
- [ ] Token không xuất hiện trong log, `ps`, file repo (kiểm bằng grep sau khi chạy).
- [ ] `/api/health` trả `ok`, `mode: "b3"`, `corpus_units: 998`.
- [ ] `curl` hỏi một câu từ tập test trả `answered` hoặc `refused` có nguồn, ≤ 20 giây.
- [ ] `chat.sh stop` dừng vLLM và chat.
**Verification:**
- [ ] `bash -n scripts/chat.sh` và `shellcheck scripts/chat.sh` (nếu có)
- [ ] Test offline: chạy script với `serve.sh`/`vihermes` giả, kiểm thứ tự bước và không lộ token
- [ ] **Người dùng chạy trên host GPU** và dán đầu ra `/api/health` + một câu `curl`
**Dependencies:** Task 2
**Files:** `scripts/chat.sh`, `configs/host.env`, `tests/test_chat_script.py`
**Scope:** Medium

### Checkpoint 2: B3 thật hoạt động
- [ ] Người dùng xác nhận trên host: health ok, 5 câu mẫu cho đủ trạng thái, thời gian chấp nhận được
- [ ] Nếu lỗi: dừng, sửa Task 1–3 trước khi đụng giao diện

---

## Phase 3: Giao diện

### Task 4: Chuyển `demo/index.html` thành trang thật
**Description:** Sao chép vào `src/vihermes/chat/static/index.html`, **giữ nguyên CSS/markup**. Bỏ `ITEMS`, `match`, `resolve`, `STEPS` theo đồng hồ, `wait()` giả lập. `ask()` gọi `POST /api/ask`; chờ hiện "Đang tra cứu…" + số giây. Vẽ `answered`/`refused`/`clarification` bằng `sourcesUI`, `REASONS`, `HINT` sẵn có. Nhãn sidebar lấy từ `/api/health`.
**Acceptance criteria:**
- [ ] Hỏi một câu không có trong 12 câu mẫu vẫn nhận kết quả từ server.
- [ ] Ba trạng thái hiển thị đúng như demo giả.
- [ ] Nội dung model sinh chỉ vào DOM bằng `textContent`.
- [ ] Không còn `ITEMS`, `resolve(`, chữ "mô phỏng" trong file.
**Verification:**
- [ ] `pytest -q tests/test_chat.py -k static` (grep các điều trên)
- [ ] Mở trang với server thật (Chrome): hỏi 3 câu; chụp ảnh sáng/tối
**Dependencies:** Task 2 (chạy được với server giả để dev; kiểm cuối với Task 3)
**Files:** `src/vihermes/chat/static/index.html`, `tests/test_chat.py`
**Scope:** Medium

### Task 5: Lỗi, mobile, xóa demo giả
**Description:** Thông báo tiếng Việt cho 422/429/503/500/mất mạng; khóa nút gửi khi đang chờ; `maxlength=500`. So giao diện với demo giả (ảnh chụp trước/sau, mobile 390px), rồi xóa `demo/index.html`.
**Acceptance criteria:**
- [ ] Mỗi mã lỗi hiện đúng câu tiếng Việt, không lộ chuỗi thô, không stack.
- [ ] Giao diện khớp demo giả ở các trạng thái: chào, chờ, trả lời, từ chối, sáng/tối, mobile.
- [ ] `demo/index.html` đã xóa; không còn tham chiếu.
**Verification:**
- [ ] `pytest -q tests/test_chat.py`
- [ ] So ảnh chụp bằng Chrome; `grep -rn "demo/index" .` rỗng
**Dependencies:** Task 4
**Files:** `src/vihermes/chat/static/index.html`, `demo/index.html` (xóa), `tests/test_chat.py`
**Scope:** Small

### Checkpoint 3: sau Task 4–5
- [ ] Người dùng mở trang trên host, hỏi tự do, thấy giao diện như cũ nhưng dữ liệu thật

---

## Phase 4: Đóng việc

### Task 6: docs + `make dryrun-chat` + rà tiêu chí
**Description:** `docs/chat.md` (cách chạy, phạm vi 998 đoạn, giới hạn B3 7/25). `make/chat.mk` với `dryrun-chat` (Conductor giả trong test, **không** là chế độ sản phẩm). Đi qua 9 tiêu chí thành công của spec mục 9.
**Acceptance criteria:**
- [ ] `make dryrun-chat` thoát mã 0, không GPU, không mạng.
- [ ] Từng tiêu chí 1–9 trong `SPEC-chat.md` có bằng chứng (lệnh/ảnh) ghi vào `docs/chat.md` hoặc PR.
- [ ] `main` không đổi (`git diff main -- ':!…'` chỉ ở nhánh `feature/chat`).
**Verification:**
- [ ] `make dryrun-chat`, `pytest -q`, `ruff check . && ruff format --check .`
**Dependencies:** Task 5
**Files:** `docs/chat.md`, `make/chat.mk`, `Makefile` (include), `README.md` (một dòng)
**Scope:** Small

### Checkpoint 4: hoàn tất
- [ ] Mọi tiêu chí thành công đạt
- [ ] Người dùng duyệt để commit và (nếu muốn) mở PR
