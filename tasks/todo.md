# Task list: ViHERMES local baselines (B1–B4)

Kế hoạch tổng thể: `tasks/plan.md`. Spec: `SPEC.md`.
Lệnh kiểm chứng chung cho mọi task: `pytest -q` và `ruff check src tests` (offline, không GPU). Commit chỉ khi người dùng yêu cầu.

---

## Phase 1: `baseline/1-local`

### Task 1: Module chia split chống rò rỉ + lệnh `vihermes split`
**Description:** Thêm `splits.py` với "test trước, cách ly sau": chọn câu test phân tầng theo hop, gom đoạn vàng, loại khỏi train/validation các câu chạm đoạn vàng của test (`quarantined`), chia phần còn lại train/validation. `test` có đúng `test_size` câu (mặc định 25, 5 mỗi hop) và cũng là tập báo cáo; `validation` lấy 12% phần sạch.
**Acceptance criteria:**
- [x] Cùng seed cho cùng kết quả và cùng hash; khác seed cho kết quả khác.
- [x] Mỗi câu thuộc đúng một trong `train/validation/test/quarantined`; tổng bằng số câu đầu vào.
- [x] Không đoạn vàng nào của câu test xuất hiện trong gold của câu train hoặc validation.
- [x] `test` có đúng 25 câu, 5 mỗi hop 1–5 (không còn khóa `test_100`).
- [x] `vihermes split` in thống kê: số câu từng phần theo hop, số bị cách ly (% tổng), số đoạn từng phần.
**Verification:**
- [x] `pytest -q tests/test_splits.py`
- [x] `ruff check src tests`
**Dependencies:** None
**Files likely touched:** `src/vihermes/splits.py`, `src/vihermes/cli.py`, `tests/test_splits.py`
**Estimated scope:** Small (3 files)

### Task 2: Chạy split trên dataset thật, review, chốt `configs/split.json`
**Description:** Chạy `vihermes split` trên `dataset/dataset.jsonl`, xem mức cách ly và kích thước train/validation thật. Đây là việc rủi ro, làm sớm để biết có cần đổi tỉ lệ trước khi xây thêm.
**Acceptance criteria:**
- [x] `configs/split.json` được tạo; chạy lần hai cho cùng hash.
- [x] Báo cáo thống kê gửi người dùng; **người dùng quyết**. Đã chốt (đổi từ 100 xuống 25): test đúng 25 câu, cách ly chặt (17,6%); train 1.110, validation 151.
- [x] Chỉ commit `configs/split.json` sau khi người dùng đồng ý.
**Verification:**
- [x] Chạy hai lần, so hash: `python -c "import json;print(json.load(open('configs/split.json'))['hash'])"`
- [x] `vihermes audit-data --qa dataset/dataset.jsonl` (so tổng 1.561)
**Dependencies:** Task 1
**Files likely touched:** `configs/split.json`
**Estimated scope:** XS (1 file)

### Task 3: Kiểm tra ánh xạ `evidence` ↔ `used_context_ids` (chỉ đọc)
**Description:** Script kiểm tra trên dữ liệu thật: khóa `context_n` trong `evidence` có khớp thứ tự `used_context_ids` không, và bao nhiêu đoạn trích là chuỗi con nguyên văn của đoạn vàng tương ứng. Kết quả quyết định thiết kế dữ liệu LoRA (Task 13).
**Acceptance criteria:**
- [x] In tỉ lệ mẫu mà mọi đoạn trích đều là chuỗi con của một đoạn vàng; tách theo hop.
- [x] Ghi rõ quy tắc ánh xạ khóa phát hiện được, hoặc ghi "không ánh xạ được" kèm phương án thay thế (khớp theo nội dung).
- [x] Không sửa dữ liệu.
**Verification:**
- [x] `python scripts/check_evidence.py dataset/dataset.jsonl`
**Dependencies:** None
**Files likely touched:** `scripts/check_evidence.py`
**Estimated scope:** XS (1 file)

**Kết quả (đã chạy trên dữ liệu thật):** khóa `context_k` ánh xạ theo **vị trí** với `used_context_ids[k-1]` khi đoạn trích là nguyên văn, nhưng **chỉ 216/1.561 bản ghi (13,8%)** có mọi đoạn trích khớp nguyên văn (hop 1: 212/312; hop 2: 2; hop 3: 0; hop 4: 2; hop 5: 0). Phần còn lại của `evidence` là **câu tóm tắt**, không phải trích dẫn. Hệ quả cho Task 13: không dùng `evidence` làm quote; cắt quote trực tiếp từ đoạn vàng.

### Task 4: `evaluate` theo split (`--split`, `--part`, `--resume`) + ghi config/timing
**Description:** Cho `evaluate` chạy trên một phần của `configs/split.json` (mặc định không dùng split: hành vi cũ). Thêm `--resume`. Ghi `config.json` đầy đủ (hash split, phiên bản thư viện, GPU, git commit) và `timing.json`. Chỉ số giữ nguyên: Recall@k (dùng `--ks 5`), F1, judge.
**Acceptance criteria:**
- [x] `--split F --part test` chỉ đánh giá đúng các id đó.
- [x] Không có `--split`: kết quả giống hệt trước khi sửa (các test hiện có qua).
- [x] `--resume` bỏ qua id đã có trong `predictions.jsonl`; `metrics.json` tính trên toàn bộ.
- [x] `config.json` có hash split, tên model, `base_url`, phiên bản `vllm`/`torch`/`transformers` (nếu có), tên GPU (null nếu không), git commit.
- [x] Không thêm chỉ số mới.
**Verification:**
- [x] `pytest -q tests/test_evaluation.py tests/test_cli.py`
- [x] Chạy `evaluate --dense-weight 0` trên fixture nhỏ, có và không `--split`.
**Dependencies:** Task 1
**Files likely touched:** `src/vihermes/evaluation.py`, `src/vihermes/cli.py`, `tests/test_evaluation.py`
**Estimated scope:** Small (3 files)

### Task 5: Model local: `base_url`, `extra_body`, `--conductor-model`, `--no-thinking`
**Description:** `OpenAILanguageModel` nhận `base_url` và `extra_body`; thêm `--base-url`, `--judge-base-url`, `--conductor-model`, `--no-thinking`. Tách LLM của Conductor khỏi Interpreter/Auditor trong `conductor()`. `SentenceEmbedder` nhận `max_seq_length`.
**Acceptance criteria:**
- [x] Không truyền cờ mới: hành vi và test cũ không đổi.
- [x] Test với server HTTP giả (mẫu có sẵn trong `tests/test_adapters.py`): request đến đúng `base_url`, body có `chat_template_kwargs.enable_thinking=false` khi bật.
- [x] Generator và judge dùng hai `base_url` khác nhau trong cùng một lần chạy `evaluate`.
- [x] Conductor dùng `--conductor-model`; Interpreter và Auditor dùng `--model`.
**Verification:**
- [x] `pytest -q tests/test_adapters.py tests/test_cli.py tests/test_agents.py`
**Dependencies:** None
**Files likely touched:** `src/vihermes/adapters/models.py`, `src/vihermes/cli.py`, `tests/test_adapters.py`, `tests/test_cli.py`
**Estimated scope:** Medium (4 files)

### Checkpoint A: nền tảng đánh giá
- [x] `pytest -q` và `ruff check src tests` sạch
- [x] `configs/split.json` đã được người dùng duyệt (T2)
- [x] Kết quả T3 đã được xem xét và thiết kế LoRA điều chỉnh nếu cần
- [x] Review với người dùng trước khi tiếp tục

---

### Task 6: Extra `train` + `hub.py` (publish/fetch ghim revision)
**Description:** Thêm extra `train` vào `pyproject.toml` (`peft`, `trl`, `datasets`, `accelerate`, `huggingface_hub`; **hỏi người dùng trước khi sửa**). Viết `hub.py`: `publish(path, repo_id)` trả commit hash; `fetch(repo_id, revision, dest)` tải đúng revision và kiểm tra đủ file. Thêm lệnh `vihermes publish` và `vihermes fetch`.
**Acceptance criteria:**
- [x] `pip install -e .` vẫn nhẹ (import trễ thư viện nặng).
- [x] Token chỉ đọc từ `HF_TOKEN`; không bao giờ in ra log hay ghi vào file.
- [x] `publish` ghi hash vào file meta; `fetch` từ chối khi thiếu file hoặc revision sai.
- [x] Repo luôn là private khi tạo; không tạo repo public.
**Verification:**
- [x] `pytest -q tests/test_hub.py` (dùng `huggingface_hub` giả, không gọi mạng)
**Dependencies:** None (cần người dùng đồng ý thêm dependency)
**Files likely touched:** `pyproject.toml`, `src/vihermes/hub.py`, `src/vihermes/cli.py`, `tests/test_hub.py`
**Estimated scope:** Medium (4 files)

### Task 7: `configs/host.env` + `scripts/serve.sh`
**Description:** `host.env`: id model kèm revision (ô trống để điền sau khi xác nhận trên HF), cổng, `gpu-memory-utilization`, chế độ judge, `HF_USER=iamtranthanhvu`, tên hai repo. `serve.sh base|lora|judge`: bật vLLM nền, log ở `outputs/logs/`, đợi `/v1/models` sẵn sàng.
**Acceptance criteria:**
- [x] Không chứa bí mật nào.
- [x] `serve.sh` báo lỗi rõ khi thiếu `vllm` hoặc đường dẫn adapter; không treo vô hạn (có timeout).
- [x] Chế độ `lora` bật `--enable-lora --lora-modules conductor=<path>`.
**Verification:**
- [x] `bash -n scripts/serve.sh`; `shellcheck scripts/serve.sh` (nếu có)
- [x] `grep -iE "hf_|token|key" configs/host.env` không thấy giá trị bí mật
**Dependencies:** None
**Files likely touched:** `configs/host.env`, `scripts/serve.sh`
**Estimated scope:** Small (2 files)

### Task 8: `vihermes pack` và `vihermes compare`
**Description:** `pack` đóng gói số liệu của một baseline (metrics, predictions, config, timing, train_meta; **không** trọng số, DB, log lớn) thành `report-bN.tar.gz`. `compare` đọc các gói, xuất `table.md` và `results.json`: dòng B1 là gốc, dòng khác kèm `(+/−)` so với B1, in đậm giá trị tốt nhất mỗi cột.
**Acceptance criteria:**
- [x] Bảng đúng 3 cột F1, LLM Judge, Recall@5, 4 chữ số thập phân, thang 0–1.
- [x] `compare` **từ chối** (lỗi rõ) nếu hash split, cấu hình judge hoặc hash corpus khác nhau giữa các gói.
- [x] Gói không chứa tệp `.safetensors`, `.db`, `.npz`.
- [x] Dưới bảng có ghi chú về tập test giữ lại và sai số của 25 câu.
**Verification:**
- [x] `pytest -q tests/test_report.py` (dữ liệu giả nhiều trường hợp: khớp, lệch hash, thiếu B1)
**Dependencies:** Task 4
**Files likely touched:** `src/vihermes/report.py`, `src/vihermes/cli.py`, `tests/test_report.py`
**Estimated scope:** Small (3 files)

### Task 9: Makefile `setup/smoke/b1/report/dryrun` + `docs/baselines.md`
**Description:** `Makefile` dùng `include make/*.mk`; file `make/b1.mk`. `setup` (cài đặt, `extract-contexts`, `ingest`, kiểm hash split), `smoke` (đánh giá 5 câu, kiểm tra `content` không có khối suy luận và JSON hợp lệ), `b1` (index gốc → đánh giá `test` → `pack`), `dryrun` (toàn chuỗi offline với encoder `hash`, LLM giả, fixture nhỏ). Mỗi bước bỏ qua nếu đã có artifact.
**Acceptance criteria:**
- [x] `make dryrun` chạy hết trên máy này không cần GPU/mạng và tạo `report-b1.tar.gz`.
- [x] Chạy `make dryrun` lần hai không làm lại bước đã xong.
- [x] `make -n b1` in đúng chuỗi lệnh và cờ model của B1.
- [x] `docs/baselines.md` có phần H1 ngắn gọn (vài lệnh).
**Verification:**
- [x] `make dryrun && make dryrun` (lần hai nhanh, không lỗi)
- [x] `make -n b1`
**Dependencies:** Task 4, 5, 7, 8
**Files likely touched:** `Makefile`, `make/b1.mk`, `docs/baselines.md`, `tests/fixtures/*` (fixture nhỏ)
**Estimated scope:** Medium (4 files)

### Checkpoint B: cổng người dùng (cần GPU)
- [x] Tất cả test và `make dryrun` qua trên máy phát triển
- [ ] Người dùng thuê H1, chạy `make setup && make smoke`: xác nhận Qwen3.5 chạy trên vLLM, tắt thinking đúng, JSON hợp lệ
- [ ] Nếu `smoke` lỗi: dừng, báo tôi; có thể phải đổi họ model (cập nhật spec)
- [ ] Người dùng chạy `make b1` và nhận `report-b1.tar.gz`

---

## Phase 2: `baseline/2-embed` (B2)

### Task 10: Dữ liệu fine-tune embedding
**Description:** `training/embed_data.py`: từ câu **train** tạo cặp (câu hỏi, từng đoạn vàng) và 3 hard negative lấy từ top-30 của retriever B1, bỏ mọi đoạn vàng của câu đó. Lệnh `vihermes make-embed-data`.
**Acceptance criteria:**
- [x] Không cặp dương nào chứa đoạn vàng của test, validation hoặc quarantined.
- [x] Không negative nào thuộc tập vàng của câu tương ứng.
- [x] Tái lập theo seed; xuất JSONL `{query, positive, negatives}`.
**Verification:**
- [x] `pytest -q tests/test_embed_data.py` (corpus giả, encoder `hash`)
**Dependencies:** Phase 1 (T1, T2, T4)
**Files likely touched:** `src/vihermes/training/embed_data.py`, `src/vihermes/cli.py`, `tests/test_embed_data.py`
**Estimated scope:** Small (3 files)

### Task 11: Huấn luyện `bge-m3`
**Description:** `training/embed.py`: `sentence-transformers` + `MultipleNegativesRankingLoss`, chọn checkpoint theo Recall@5 trên validation, lưu model và `train_meta.json` (seed, hash split, số cặp, siêu tham số, thời gian, loss, VRAM đỉnh). Lệnh `vihermes finetune-embed`.
**Acceptance criteria:**
- [x] Smoke test `--max-steps 2` với model rất nhỏ chạy xong, lưu model và `train_meta.json`.
- [x] Không dùng test ở bất kỳ bước nào (kiểm bằng test chặn truy cập id test).
- [x] Thư viện nặng chỉ import trong hàm.
**Verification:**
- [x] `pytest -q tests/test_embed_train.py` (bỏ qua nếu chưa cài extra `train`, qua `importorskip`)
**Dependencies:** Task 6, 10
**Files likely touched:** `src/vihermes/training/embed.py`, `src/vihermes/cli.py`, `tests/test_embed_train.py`
**Estimated scope:** Small (3 files)

### Task 12: `make b2`
**Description:** `make/b2.mk`: index gốc → dữ liệu → huấn luyện → `publish` lên `iamtranthanhvu/vihermes-bge-m3-ft` → index mới (`dense-b2.npz`) → đánh giá → `pack`. Mở rộng `dryrun` cho B2 (model nhỏ, không đẩy HF).
**Acceptance criteria:**
- [x] `make dryrun-b2` chạy offline hết chuỗi trừ `publish` (có chế độ giả).
- [x] Idempotent: bỏ qua bước đã có artifact.
- [x] `publish` thất bại thì dừng và báo rõ, không bỏ qua im lặng.
**Verification:**
- [x] `make dryrun-b2`; `make -n b2`
**Dependencies:** Task 9, 11
**Files likely touched:** `make/b2.mk`, `docs/baselines.md`
**Estimated scope:** Small (2 files)

---

## Phase 3: `baseline/3-lora` (B3)

### Task 13: Dữ liệu LoRA cho Conductor
**Description:** `training/lora_data.py`: mỗi mẫu là một lần gọi Conductor đúng định dạng suy luận (import `GENERATE` và `serialize_evidence` từ `agents.py`), đích là JSON `{claims, abstain}`. Vì Task 3 cho thấy `evidence` chủ yếu là câu tóm tắt, **quote được cắt trực tiếp từ đoạn vàng**: ưu tiên `evidence` nếu nó là chuỗi con nguyên văn; nếu không, chọn cửa sổ văn bản (câu hoặc cụm ≤ ~300 ký tự) có độ trùng từ cao nhất với câu trả lời tham chiếu và phần tóm tắt, lấy nguyên văn theo vị trí ký tự nên luôn thỏa `quote in node.text`. ~15% mẫu abstain (bỏ đoạn vàng khỏi evidence). Mẫu không chọn được quote đủ trùng bị loại và đếm lại. Câu mà truy xuất chưa đủ đoạn vàng vừa cho một mẫu abstain (bằng chứng như đã truy xuất) vừa cho một mẫu trả lời trong đó đoạn vàng thiếu được chèn thay cho đoạn xếp hạng thấp nhất (giữ nguyên số đoạn); tắt bằng `--no-inject-gold`. **Kết quả thật (BM25, k=5):** train 868 mẫu (738 trả lời, 130 abstain), validation 118; không mẫu nào bị loại vì thiếu quote; 6 mẫu quá dài (>24.000 ký tự). Lệnh `vihermes make-lora-data`.
**Acceptance criteria:**
- [x] Mọi mẫu `abstain=false` có mọi quote là chuỗi con của đoạn được cite (dùng đúng điều kiện của Auditor).
- [x] Không mẫu nào chứa câu hỏi hoặc đoạn vàng của test/quarantined.
- [x] Mẫu abstain không chứa đoạn vàng trong evidence.
- [x] System prompt trong dữ liệu bằng đúng `agents.GENERATE`.
- [x] In số mẫu giữ/loại.
**Verification:**
- [x] `pytest -q tests/test_lora_data.py`
**Dependencies:** Phase 1, Task 3
**Files likely touched:** `src/vihermes/training/lora_data.py`, `src/vihermes/cli.py`, `tests/test_lora_data.py`
**Estimated scope:** Small (3 files)

### Task 14: Huấn luyện LoRA `Qwen3.5-4B`
**Description:** `training/lora.py`: `peft` + `trl` `SFTTrainer`, LoRA bf16 (48GB), gradient checkpointing, loss chỉ trên phần assistant, tắt thinking trong chat template, chọn checkpoint theo eval loss trên validation, lưu adapter và `train_meta.json`. Lệnh `vihermes finetune-lora`.
**Acceptance criteria:**
- [x] Smoke test `--max-steps 2` với model rất nhỏ lưu được adapter.
- [x] Loss chỉ tính trên phần assistant (test kiểm mặt nạ nhãn).
- [x] Không dùng test; thư viện nặng import trễ.
**Verification:**
- [x] `pytest -q tests/test_lora_train.py` (`importorskip`)
**Dependencies:** Task 6, 13
**Files likely touched:** `src/vihermes/training/lora.py`, `src/vihermes/cli.py`, `tests/test_lora_train.py`
**Estimated scope:** Small (3 files)

### Task 15: `make b3` + chế độ `serve.sh lora`
**Description:** `make/b3.mk`: index gốc → dữ liệu → huấn luyện → `publish` lên `iamtranthanhvu/vihermes-qwen35-4b-conductor-lora` → bật vLLM chế độ LoRA → đánh giá với `--conductor-model conductor` → `pack`. Mở rộng `dryrun` cho B3.
**Acceptance criteria:**
- [x] `make dryrun-b3` chạy offline hết chuỗi trừ `publish` và vLLM (có chế độ giả).
- [x] Idempotent; lỗi dừng rõ ràng.
**Verification:**
- [x] `make dryrun-b3`; `make -n b3`
**Dependencies:** Task 9, 14
**Files likely touched:** `make/b3.mk`, `scripts/serve.sh`, `docs/baselines.md`
**Estimated scope:** Small (3 files)

### Checkpoint C: B2 và B3 chạy xong
- [ ] Người dùng chạy H2 (`make b2`) và H3 (`make b3`); hai model nằm trên HF (private) với commit hash được ghi
- [ ] `report-b2` và `report-b3` đã `scp` về máy

---

## Phase 4: `baseline/4-all` (B4)

### Task 16: Merge nhánh 2 + 3, giải quyết xung đột
**Description:** Từ `baseline/1-local`, merge `baseline/2-embed` và `baseline/3-lora`. Xung đột dự kiến ở `cli.py`, `docs/baselines.md`; giữ cả hai tập cờ.
**Acceptance criteria:**
- [x] `pytest -q` và `ruff check src tests` sạch sau merge.
- [x] Không đổi ngữ nghĩa các cờ đã có.
- [x] Chạy không bật cờ nào cho kết quả giống `baseline/1-local` (test hồi quy trên fixture).
**Verification:**
- [x] `pytest -q`; `ruff check src tests`; `make dryrun` (B1) vẫn qua
**Dependencies:** Task 12, 15
**Files likely touched:** `src/vihermes/cli.py`, `docs/baselines.md` (xung đột)
**Estimated scope:** Small

### Task 17: `make b4` (fetch) + dryrun đầy đủ + hoàn thiện docs
**Description:** `make/b4.mk`: `fetch` model và adapter đúng revision ghim, dựng index, đánh giá với cả hai, `pack`; **không có bước huấn luyện**. Hoàn thiện `docs/baselines.md` (H1–H4 và `make compare`).
**Acceptance criteria:**
- [x] `make dryrun-b4` chạy offline (fetch giả).
- [x] B4 không import/gọi `peft`/`trl` trên đường đánh giá (test).
- [x] `config.json` của B4 ghi repo id và commit hash của hai model.
- [x] `docs/baselines.md` mô tả đủ 4 host bằng vài lệnh.
**Verification:**
- [x] `make dryrun-b4`; `pytest -q`
**Dependencies:** Task 16
**Files likely touched:** `make/b4.mk`, `docs/baselines.md`, `tests/test_b4_no_training.py`
**Estimated scope:** Small (3 files)

### Checkpoint D: hoàn tất
- [ ] Người dùng chạy H4 (`make b4`)
- [ ] `make compare report-b1.tar.gz report-b2.tar.gz report-b3.tar.gz report-b4.tar.gz` ra `table.md` (từ chối nếu hash lệch)
- [ ] Mọi tiêu chí thành công trong `SPEC.md` mục 9 đạt; `main` không đổi
