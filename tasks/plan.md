# Implementation Plan: ViHERMES local baselines (B1–B4)

Nguồn: `SPEC.md` (đã duyệt). Chi tiết từng task (tiêu chí, kiểm chứng, file) ở `tasks/todo.md`.

## Overview

Viết code trên máy này (test offline, không GPU) để người dùng chạy trên 4 host Vast: H1 chạy B1, H2 fine-tune embedding rồi đẩy lên Hugging Face (B2), H3 huấn luyện LoRA rồi đẩy lên Hugging Face (B3), H4 tải hai model về và chạy B4. Kết quả gộp thành một bảng (F1, LLM Judge, Recall@5) kèm mức chênh so với B1. `main` không đổi.

## Architecture Decisions

- **Mở rộng, không viết lại:** thêm `splits.py`, `hub.py`, `report.py`, `training/`; sửa `evaluation.py`, `adapters/models.py`, `cli.py` theo hướng **tương thích ngược** (mặc định giữ hành vi cũ, các test hiện có phải qua).
- **Tránh xung đột khi merge B2 + B3 → B4:** hai nhánh cùng sửa `cli.py`, `Makefile`, `pyproject.toml`. Giải pháp:
  - Tạo extra `train` **một lần** ở Phase 1 (cả hai nhánh kế thừa).
  - Subcommand huấn luyện đặt trong module riêng (`training/embed.py`, `training/lora.py`), `cli.py` chỉ thêm một dòng đăng ký mỗi cái.
  - `Makefile` dùng `include make/*.mk`; B2 và B3 mỗi cái một file `.mk` riêng.
- **Hub dùng chung:** `publish`/`fetch` (ghim revision) viết một lần ở Phase 1 cho B2, B3, B4 cùng dùng.
- **Chạy lại được sau khi ngắt:** mỗi target Makefile bỏ qua bước đã có artifact; `evaluate --resume`.
- **Phát hiện lỗi Makefile trước khi thuê host:** target `make dryrun` chạy cả chuỗi offline (encoder `hash`, LLM giả, fixture nhỏ).
- **Quy ước git:** mỗi phase một nhánh theo spec. Tôi tạo nhánh khi bắt đầu phase; **chỉ commit khi bạn yêu cầu**. `SPEC.md` và `tasks/` hiện chưa theo dõi trên `main` và sẽ đi theo `baseline/1-local`.

## Dependency Graph

```text
T1 split ──┬─ T2 split thật (review) ─┐
           └─ T4 evaluate theo split ─┤
T3 kiểm tra evidence (đọc-only) ──────┼──────────────► T13 (LoRA data)
T5 model local ───────────────────────┤
                                      ├─ T6 train extra + hub ──► T10..T15, T17
                                      ├─ T7 host.env + serve.sh
                                      ├─ T8 compare + pack
                                      └─ T9 Makefile b1 + dryrun
                                               │
                         ┌─────────────────────┴─────────────────────┐
                    Phase 2 (B2): T10→T11→T12            Phase 3 (B3): T13→T14→T15
                         └─────────────────────┬─────────────────────┘
                                          T16 merge → T17 b4 + dryrun đầy đủ
```

Phase 2 và Phase 3 độc lập, làm song song được; chỉ gặp nhau ở T16.

## Task List (chỉ mục; chi tiết ở `tasks/todo.md`)

### Phase 1 — `baseline/1-local` (B1 + nền tảng dùng chung)
- [x] T1: Module chia split chống rò rỉ + lệnh `vihermes split`
- [x] T2: Chạy split trên dataset thật, review thống kê, chốt `configs/split.json`
- [x] T3: Kiểm tra ánh xạ `evidence` ↔ `used_context_ids` trên dữ liệu thật (chỉ đọc)
- [ ] T4: `evaluate` theo split (`--split --part test`, `--resume`, ghi config/timing)
- [ ] T5: Model local: `base_url`, `extra_body`, `--conductor-model`, `--no-thinking`

### Checkpoint A: nền tảng đánh giá
- [ ] T6: Extra `train` + `hub.py` (publish/fetch ghim revision)
- [ ] T7: `configs/host.env` + `scripts/serve.sh`
- [ ] T8: `vihermes pack` + `vihermes compare` (bảng, mức chênh, chốt chặn hash)
- [ ] T9: Makefile `setup/smoke/b1/report/dryrun` + `docs/baselines.md`

### Checkpoint B: **cổng người dùng**, thuê H1, chạy `make smoke` rồi `make b1`

### Phase 2 — `baseline/2-embed` (B2)
- [ ] T10: Dữ liệu fine-tune embedding
- [ ] T11: Huấn luyện `bge-m3`
- [ ] T12: `make b2` (huấn luyện → publish → index → đánh giá)

### Phase 3 — `baseline/3-lora` (B3)
- [ ] T13: Dữ liệu LoRA cho Conductor
- [ ] T14: Huấn luyện LoRA `Qwen3.5-4B`
- [ ] T15: `make b3` + chế độ `serve.sh lora`

### Checkpoint C: B2 và B3 chạy xong trên H2, H3; model nằm trên HF

### Phase 4 — `baseline/4-all` (B4)
- [ ] T16: Merge nhánh 2 + 3, giải quyết xung đột
- [ ] T17: `make b4` (fetch) + dryrun đầy đủ + hoàn thiện docs

### Checkpoint D: chạy H4, gộp 4 gói số liệu bằng `make compare` thành bảng cuối

## Risks and Mitigations

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| vLLM / PEFT / TRL chưa hỗ trợ kiến trúc Qwen3.5 | Cao | `make smoke` trên H1 trước khi chạy dài; nếu hỏng, đổi sang họ Qwen3 và báo bạn (đổi spec) |
| Cách thức tắt thinking cho Qwen3.5 không đúng | Cao | `smoke` kiểm tra `content` không chứa khối suy luận; có phương án `--reasoning-parser` |
| Cách ly rò rỉ làm mất quá nhiều dữ liệu train | Trung bình | T2 đo trước, bạn quyết ngưỡng; có thể hạ tỉ lệ test |
| `evidence` không phải trích dẫn nguyên văn (T3: chỉ 13,8% khớp) | **Đã xảy ra** | T13 cắt quote trực tiếp từ đoạn vàng theo độ trùng từ; quote nguyên văn theo cách xây dựng |
| 48GB không đủ cho 4B + adapter + judge 9B cùng lúc | Trung bình | Tùy chọn judge lượng tử hóa trong `host.env`, **cùng một cấu hình cho cả 4 host** |
| Xung đột merge ở `cli.py`, `Makefile`, `pyproject.toml` | Trung bình | Quyết định kiến trúc ở trên (extra chung, module riêng, `include make/*.mk`) |
| Lỗi Makefile chỉ lộ khi đã thuê host (tốn tiền) | Trung bình | `make dryrun` offline, `make -n`, `bash -n` |
| Bốn host cho số không so sánh được | Cao | Ghim revision model, phiên bản vLLM, hash split; `compare` từ chối nếu lệch |
| 100 câu quá nhiễu để thấy mức tăng | Trung bình | Đã chấp nhận; ghi chú dưới bảng; cờ để tăng số câu về sau |
| Lộ `HF_TOKEN` | Cao | Chỉ qua biến môi trường tại host; không ghi vào file/log; `.gitignore` đã chặn `.env` |

## Open Questions

1. ~~Ngưỡng cách ly~~ đã chốt ở T2: test đúng 100 câu, cách ly 39,5%, train 744, dev 100.
2. Judge lượng tử hóa hay hai lượt sinh/chấm; chọn khi chạy `smoke` trên H1.
3. Thêm dependency vào extra `train` (`peft`, `trl`, `datasets`, `accelerate`, `huggingface_hub`): cần bạn đồng ý ở T6 (đã nêu trong spec là "Ask first").
