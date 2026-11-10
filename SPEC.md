# SPEC: Cải tiến ViHERMES bằng fine-tune, báo cáo trong một bảng như paper

> Trạng thái: **Đã duyệt.** Kế hoạch ở `tasks/plan.md`.
> File nằm trong working tree của `main`, **không commit vào `main`** (paper gốc).

## 1. Objective

Lấy pipeline gốc của paper (ViHERMES), **chỉ đổi model sang local** và chạy ra chỉ số: đó là **chỉ số gốc (B1)**. Sau đó cải tiến bằng **fine-tune** và đo **tăng bao nhiêu so với B1**, trên dataset của paper. Kết quả ghi vào **một bảng** có cột `F1 ↑ | LLM Judge ↑ | Recall@5 ↑` (thang 0–1, 4 chữ số thập phân, in đậm giá trị tốt nhất mỗi cột), mỗi dòng cải tiến kèm mức chênh so với B1.

**Không so sánh với số của paper** (không có dòng paper trong bảng). Không có bảng ablation thành phần, không thêm chỉ số mới, không có reranker.

### Các dòng trong bảng

| Dòng | Mô tả | Cách có số liệu |
|---|---|---|
| **B1** (chỉ số gốc) | Pipeline gốc, thay model local, **không huấn luyện** | Chạy |
| **B2** + fine-tune embedding | B1 + `BAAI/bge-m3` fine-tune | Chạy |
| **B3** + LoRA Conductor | B1 + LoRA `Qwen3.5-4B` cho Conductor | Chạy |
| **B4** + cả hai | Ghép trọng số B2 và B3, **không huấn luyện lại** | Chạy |

Ví dụ định dạng (số minh họa, không phải kết quả):

| Method | F1 ↑ | LLM Judge ↑ | Recall@5 ↑ |
| :-- | --: | --: | --: |
| B1 (gốc) | 0.xxxx | 0.xxxx | 0.xxxx |
| B2 | 0.xxxx (+0.xxxx) | 0.xxxx (+0.xxxx) | **0.xxxx (+0.xxxx)** |

Mức chênh `(+/−)` tính trực tiếp từ B1, không phải chỉ số mới.

### Model local

| Vai trò | Paper | Local |
|---|---|---|
| Generator (Interpreter, Conductor, Auditor) | gpt-4o-mini | `Qwen/Qwen3.5-4B` |
| Judge | gpt-4o | `Qwen/Qwen3.5-9B` (không có bản 8B; xác nhận tên trên Hugging Face) |
| Embedding | text-embedding-3-small | `BAAI/bge-m3` |

## 2. Điều kiện để so sánh với B1 là công bằng

1. **Phải giữ lại tập test không dùng để huấn luyện.** Fine-tune dùng chính dataset của paper, nên nếu đánh giá trên câu đã học thì mức tăng so với B1 là giả. Cả B1–B4 đều đánh giá trên **cùng tập test giữ lại** (B1 không huấn luyện nhưng vẫn đánh giá trên tập này để so cùng điều kiện).
2. **Cùng judge, cùng corpus, cùng cấu hình** cho mọi dòng; chỉ khác phần được fine-tune.
3. **Độ nhiễu:** với 100 câu, sai số khoảng ±10 điểm % ở mức tổng, ±22 điểm % mỗi hop. Mức tăng nhỏ hơn mức này có thể chỉ là nhiễu; phải ghi chú dưới bảng.

Corpus nạp từ `contexts.csv` cho `0 edges` (không có đồ thị); điều này áp dụng đồng đều cho mọi dòng nên không ảnh hưởng so sánh với B1.

## 3. Phạm vi công việc

| # | Hạng mục | Nội dung | Nhánh |
|---|---|---|---|
| W1 | Split + đánh giá | Chia train/dev/test chống rò rỉ; `evaluate` theo split; xuất bảng đúng định dạng paper | `baseline/1-local` |
| W2 | Model local (B1) | `base_url` riêng cho judge, tắt thinking, `--conductor-model`, cấu hình bge-m3 | `baseline/1-local` |
| W3 | Fine-tune embedding (B2) | Dữ liệu (câu hỏi, đoạn vàng, hard negative) và fine-tune `bge-m3` | `baseline/2-embed` |
| W4 | LoRA Conductor (B3) | Dữ liệu từ `evidence` có sẵn và LoRA `Qwen3.5-4B` | `baseline/3-lora` |
| W5 | Ghép (B4) | Merge nhánh 2 + 3, dùng lại trọng số, chạy đánh giá | `baseline/4-all` |

Thứ tự: W1 → W2 → W3 và W4 (song song được) → W5.

```text
main                      paper gốc, không đụng
└─ baseline/1-local       W1 + W2
   ├─ baseline/2-embed    W3
   ├─ baseline/3-lora     W4
   └─ baseline/4-all      W5 (merge 2 + 3)
```

### W1 chi tiết: split và tập đánh giá

- Mỗi đoạn ngữ cảnh là bằng chứng của ~4–5 câu hỏi, nên chia ngẫu nhiên theo câu sẽ rò rỉ. Dùng **"test trước, cách ly sau"**: chọn câu test phân tầng theo hop, gom đoạn vàng của chúng, **loại khỏi train mọi câu khác chạm vào các đoạn đó** (`quarantined`), phần còn lại chia train/dev. Mức mất dữ liệu chưa đo; `vihermes split` in thống kê.
- **Đã chốt (sau khi đo trên dữ liệu thật):** test đúng **100 câu** (20 mỗi hop), cách ly chặt; `dev` = 12% phần sạch còn lại; seed 42. Kết quả thật: train 744, dev 100, test 100, quarantined 617 (39,5%), không rò rỉ. Test 311 câu sẽ cách ly 60% và chỉ còn 168 câu train, nên không dùng. Lưu `configs/split.json` kèm hash; cả 4 baseline dùng chung.
- **Tập báo cáo = chính tập `test` (100 câu, 20 câu mỗi hop), cố định theo seed.** Không dùng `--limit` vì dataset sắp xếp theo khối hop (sẽ chỉ lấy hop 1).
- Cả ba chỉ số của bảng (F1, LLM Judge, Recall@5) tính trên cùng 100 câu này. Với 100 câu, sai số khoảng ±10 điểm % ở mức tổng, ±22 điểm % mỗi hop; chênh lệch nhỏ giữa các dòng có thể chỉ là nhiễu và phải ghi chú.
- Chỉ số: dùng nguyên `evaluation.py` hiện có (`recall@5`, `f1`, `judge`). Không thêm chỉ số.
- Corpus retrieval vẫn là toàn bộ 998 đoạn.

### W2 chi tiết

- `OpenAILanguageModel` nhận `base_url` và `extra_body`; thêm `--base-url`, `--judge-base-url`, `--conductor-model`. Hiện `OpenAI()` dùng chung một `OPENAI_BASE_URL`, mà một server vLLM chỉ phục vụ một model gốc, nên chạy hai server (4B cổng 8000, 9B cổng 8001).
- Tắt thinking bằng `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`; **chưa kiểm chứng** với Qwen3.5.
- `conductor()` trong `cli.py` hiện dùng một `llm` cho cả ba vai; tách để Conductor dùng `--conductor-model`.
- Mặc định giữ hành vi cũ để `main` và demo không vỡ.

### W3 chi tiết (B2)

- Dữ liệu: câu train; cặp dương là (câu hỏi, từng đoạn vàng); âm tính khó lấy từ top-30 của retriever B1, bỏ mọi đoạn vàng của câu đó, chọn 3.
- Huấn luyện: `sentence-transformers`, `MultipleNegativesRankingLoss`, lr 1e-5, 1–2 epoch, `max_seq_length=512`. Chọn checkpoint theo Recall@5 trên dev. Build index riêng (`dense-b2.npz`).

### W4 chi tiết (B3)

- Mẫu = một lần gọi Conductor đúng định dạng lúc suy luận (`GENERATE`, `serialize_evidence` import từ `agents.py`); đích là JSON `{claims:[{text, citations, quotes}], abstain}`; `text` = `answer`, `quotes` lấy từ trường `evidence` của dataset, **phải thỏa `quote in node.text`**, mẫu không thỏa bị loại. Khoảng 15% mẫu abstain (bỏ đoạn vàng khỏi evidence).
- LoRA `peft` + `trl`, r=16, loss chỉ trên phần assistant. **GPU 48GB: LoRA bf16 (không cần QLoRA)**, bật gradient checkpointing, `max_seq_length` 4096–8192. Chọn checkpoint theo eval loss trên dev.
- Phục vụ: `vllm serve Qwen/Qwen3.5-4B --enable-lora --lora-modules conductor=<path>`; `--conductor-model conductor`. Interpreter và Auditor vẫn dùng 4B gốc.

### W5 chi tiết (B4)

- Chỉ merge, dùng lại `outputs/embed-ft/model` và adapter LoRA, **không huấn luyện**. Adapter được huấn luyện với ngữ cảnh từ retriever B1 nhưng chạy với retriever B2, có thể lệch nhẹ; ghi chú khi báo cáo.

## 4. Commands

```bash
pytest -q && ruff check src tests

vihermes split    --qa dataset/dataset.jsonl --output configs/split.json --seed 42   # chạy một lần, commit
vihermes evaluate --qa dataset/dataset.jsonl --split configs/split.json --part test \
  --answers --judge --ks 5 --output outputs/<b1|b2|b3|b4>      # + cờ model theo từng baseline
vihermes compare  report-b1.tar.gz report-b2.tar.gz report-b3.tar.gz report-b4.tar.gz > table.md
```

Cờ model cho B1 (trên host): `--embedding sentence-transformers --embedding-model BAAI/bge-m3 --model Qwen/Qwen3.5-4B --base-url http://localhost:8000/v1 --judge-model Qwen/Qwen3.5-9B --judge-base-url http://localhost:8001/v1`. B2 đổi `--embedding-model` và `--index`; B3 thêm `--conductor-model conductor`; B4 dùng cả hai.

`vihermes compare` xuất bảng Markdown: dòng B1 là chỉ số gốc, các dòng còn lại kèm mức chênh `(+/−)` so với B1, in đậm giá trị tốt nhất mỗi cột.

## 4a. Số liệu phải lưu cho báo cáo

Mọi số liệu ghi tự động vào `outputs/`, không phải chép tay. Không lưu trọng số model hay DB trong gói báo cáo.

| Artifact | Nội dung | Ghi bởi |
|---|---|---|
| `configs/split.json` | seed, tỉ lệ, số câu train/dev/test/quarantined (tổng và theo hop), số đoạn mỗi phần, hash (commit vào repo) | `split` |
| `outputs/<b>/metrics.json` | F1, LLM Judge, Recall@5: tổng và theo hop 1–5, kèm số câu `n` | `evaluate` |
| `outputs/<b>/predictions.jsonl` | từng câu: câu trả lời, `retrieved_ids`, điểm từng câu, thời gian | `evaluate` |
| `outputs/<b>/config.json` | mọi cờ chạy, id model, `base_url`, embedding/index, hash split, git commit, phiên bản `vllm`/`torch`/`transformers`/`sentence-transformers`/`peft`/`trl`, tên GPU và VRAM | `evaluate` |
| `outputs/<b>/timing.json` | thời gian build index, đánh giá, tổng | `evaluate`/`index` |
| `outputs/embed-ft/train_meta.json`, `outputs/lora-conductor/train_meta.json` | seed, hash split, số mẫu (và số mẫu bị loại), siêu tham số, thời gian huấn luyện, loss train/dev theo bước, VRAM đỉnh | `finetune-*` |
| `outputs/table.md`, `outputs/results.json` | bảng cuối (B1–B4 kèm mức chênh so với B1) và số thô để vẽ lại | `compare` |
| `outputs/report-bN.tar.gz` | gói số liệu của mỗi host (metrics, predictions, config, timing, train_meta; không có trọng số, DB, log lớn) để `scp` về máy | `make report` |

## 4b. Chạy trên host: 4 host, mỗi baseline một host

Mục tiêu: mỗi host chỉ gõ **một lệnh** (`make bN`), không phải nhớ chuỗi cờ.

| Host | Nhánh | Lệnh | Làm gì | Đưa lên / lấy về |
|---|---|---|---|---|
| H1 | `baseline/1-local` | `make b1` | Dựng index gốc, đánh giá | Gói số liệu `report-b1` |
| H2 | `baseline/2-embed` | `make b2` | Dựng index gốc, tạo dữ liệu, fine-tune `bge-m3`, **publish model**, dựng index mới, đánh giá | Model lên HF Hub (private); `report-b2` |
| H3 | `baseline/3-lora` | `make b3` | Dựng index gốc, tạo dữ liệu, LoRA, **publish adapter**, đánh giá | Adapter lên HF Hub (private); `report-b3` |
| H4 | `baseline/4-all` | `make b4` | **Tải** model từ H2 và adapter từ H3, dựng index, đánh giá (không huấn luyện) | `report-b4` |
| Máy bạn | bất kỳ | `make compare` | Gộp bốn gói số liệu thành bảng (không cần GPU) | `table.md`, `results.json` |

Thứ tự: H1, H2, H3 chạy **song song**; H4 chờ H2 và H3 publish xong. Các host không phụ thuộc nhau về dữ liệu: mỗi host tự dựng index gốc (998 đoạn, rẻ) để khai thác hard negative và ngữ cảnh huấn luyện.

### Điều kiện để số của 4 host so sánh được

- **Cùng split:** `configs/split.json` tạo **một lần** trên máy phát triển và **commit vào nhánh `baseline/1-local`** (file nhỏ, là dữ liệu đầu vào, không phải artifact), các nhánh khác kế thừa. Mỗi host kiểm hash trước khi chạy; lệch thì dừng.
- **Cùng cấu hình:** `configs/host.env` được commit: tên model **kèm revision (commit hash trên Hugging Face)**, phiên bản `vllm`, cách xử lý judge (lượng tử hóa hay không), seed, temperature = 0. Không để model hay thư viện tự cập nhật giữa các host.
- **Cùng loại GPU 48GB** cho mọi host (không bắt buộc cho điểm số, nhưng thời gian ghi trong báo cáo mới so sánh được).
- **`compare` từ chối tạo bảng** nếu hash split, cấu hình judge, hoặc hash corpus khác nhau giữa các gói số liệu.

### Chuyển model giữa host

- Dùng **Hugging Face Hub, repo private**: H2 đẩy `bge-m3` đã fine-tune (~2GB, ước lượng), H3 đẩy adapter LoRA (vài trăm MB, ước lượng), H4 tải về.
- Hai repo private: `<HF_USER>/vihermes-bge-m3-ft` (H2) và `<HF_USER>/vihermes-qwen35-4b-conductor-lora` (H3). `HF_USER` đặt trong `configs/host.env`.
- Hai token fine-grained giới hạn đúng hai repo: **write** cho H2/H3, **read-only** cho H4. Token qua biến môi trường `HF_TOKEN` nhập tại host, **không commit, không gửi qua chat**.
- Lệnh: `make publish` (H2, H3), `make fetch` (H4). `publish` ghi **commit hash** của lần đẩy vào `train_meta.json`; `fetch` tải **đúng revision đã ghim**, kiểm tra đủ file, ghi repo id và hash vào `config.json`.
- Phương án khác nếu không muốn dùng HF: `scp` về máy bạn rồi đẩy lên H4, hoặc bucket S3/GCS.

### Cơ chế chung mỗi host

- `configs/host.env`: tên model + revision, cổng, đường dẫn, seed, `gpu-memory-utilization`, cách xử lý judge.
- `Makefile`: `setup`, `b1`, `b2`, `b3`, `b4`, `publish`, `fetch`, `report`, `compare`. Mỗi target **bỏ qua bước đã có artifact**, nên chạy lại sau khi ngắt không mất công.
- `scripts/serve.sh <base|lora|judge>`: bật vLLM nền, ghi log vào `outputs/logs/`, đợi `/v1/models` sẵn sàng; `b3` và `b4` bật chế độ LoRA.
- `evaluate --resume`: bỏ qua câu đã có trong `predictions.jsonl`.
- Mỗi host tạo `outputs/report-bN.tar.gz`; bạn `scp` cả bốn về máy, chạy `make compare` để ra bảng cuối.
- Không cần tương tác giữa chừng. Lỗi dừng chương trình với thông báo rõ, không bỏ qua im lặng.

## 5. Project structure (dự kiến)

```text
src/vihermes/
  splits.py              chia split chống rò rỉ                  (W1)
  evaluation.py          --split/--part/--resume, ghi timing     (W1)
  adapters/models.py     base_url riêng, tắt thinking            (W2)
  cli.py                 cờ mới, lệnh split/compare/finetune-*   (W1–W4)
  training/              embed_data/train, lora_data/train       (W3, W4)
configs/split.json       split cố định, commit (tạo một lần)
configs/host.env         cấu hình model + revision, cổng, đường dẫn cho host
scripts/serve.sh         bật vLLM nền theo chế độ
Makefile                 setup, b1..b4, publish, fetch, report, compare
docs/baselines.md        hướng dẫn chạy ngắn trên host
tests/                   pytest offline, không GPU
outputs/                 artifact, không commit
```

## 6. Code style

Giữ phong cách hiện có: type hint `X | None`, dependency injection qua `ports.py`, dòng ≤ 100, `ruff` (E4, E7, E9, F, I), import thư viện nặng (torch, peft, trl) trễ bên trong hàm.

## 7. Testing strategy

- `pytest` offline với encoder `hash` và LLM giả; không cần GPU hay API.
- Chống rò rỉ: không đoạn vàng của câu test nào nằm trong gold của train hoặc dev.
- Split tái lập theo seed, đủ hop 1–5, mỗi câu thuộc đúng một phần.
- Chỉ số hiện có không đổi kết quả khi chạy không kèm `--split`.
- Dữ liệu LoRA: mọi `quote in node.text`; prompt hệ thống bằng đúng `agents.GENERATE`.
- Script huấn luyện có chế độ smoke test (`--max-steps 2`, model nhỏ). Chạy thật trên host Vast.

## 8. Boundaries

- **Always:** `pytest` và `ruff` trước commit; cùng `configs/split.json` cho cả 4 baseline; seed cố định; ghi model (kèm revision), cấu hình, hash split vào `config.json`.
- **Ask first:** thêm dependency (`peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `huggingface_hub`) vào `pyproject.toml` (extra `train`); đổi cú pháp CLI hiện có; xây đồ thị cho corpus; thêm chỉ số hay bảng mới; **đẩy model lên dịch vụ bên ngoài** (HF Hub), chỉ dùng repo **private**.
- **Never:** sửa hoặc commit lên `main`; commit API key, `HF_TOKEN`, `.env`, trọng số model, `outputs/`; huấn luyện hoặc chọn siêu tham số bằng tập test; nới kiểm tra quote của Auditor để tăng điểm; tạo repo model public.

## 9. Success criteria

1. `outputs/table.md` là **một bảng** có 4 dòng (B1–B4), 3 cột F1, LLM Judge, Recall@5; B1 là chỉ số gốc, các dòng còn lại kèm mức chênh so với B1.
2. B1–B4 dùng cùng `split.json` và cùng 100 câu `test`.
3. Test chống rò rỉ qua.
4. B2 và B3 tái lập được từ lệnh trong `docs/baselines.md`; B4 không có bước huấn luyện.
5. Dưới bảng có ghi chú về tập test giữ lại và sai số của 100 câu (mục 2).
6. `main` không thay đổi.
7. Mỗi host chỉ cần `make setup` rồi **một** lệnh `make bN`; sau khi ngắt, chạy lại tiếp tục từ chỗ dở mà không làm lại bước đã xong.
8. Mọi artifact ở mục 4a được tạo tự động và gói trong `outputs/report-bN.tar.gz`.
9. `make compare` ở máy bạn từ chối tạo bảng nếu hash split, cấu hình judge hoặc hash corpus khác nhau giữa các gói.
10. H4 chạy B4 chỉ bằng model tải về từ H2 và H3, không có bước huấn luyện.

## 10. Open Questions

0. ~~Một nhánh gộp hay từng nhánh~~ **Đã chốt: 4 host, mỗi host một nhánh** (H1 `baseline/1-local`, H2 `baseline/2-embed`, H3 `baseline/3-lora`, H4 `baseline/4-all`). Model chuyển qua Hugging Face Hub private: bạn có tài khoản/token HF và đồng ý dùng repo private không?

1. Tên chính xác model judge (`Qwen/Qwen3.5-9B`?) và vLLM đã hỗ trợ Qwen3.5 chưa; cách tắt thinking.
2. ~~Mức mất dữ liệu do cách ly~~ **Đã chốt:** 39,5% (617 câu) với test 100 câu; còn train 744 và dev 100. Tăng số câu test về sau sẽ làm train giảm mạnh (test 200 câu: còn ~390 câu sạch).
3. ~~Map khóa `evidence`~~ **Đã kiểm tra (T3):** ánh xạ theo vị trí đúng, nhưng chỉ 13,8% bản ghi có `evidence` là trích dẫn nguyên văn; phần còn lại là câu tóm tắt. Dữ liệu LoRA cắt quote trực tiếp từ đoạn vàng theo độ trùng từ với câu trả lời tham chiếu.
4. PEFT/TRL và `--enable-lora` của vLLM có hỗ trợ kiến trúc Qwen3.5 không.
5. ~~GPU cho LoRA~~ **Đã chốt: 48GB** (LoRA bf16). Lưu ý khi đánh giá: Qwen3.5-4B + adapter (~9GB) và judge Qwen3.5-9B (~19GB bf16) chạy cùng lúc trên 48GB rất sát KV cache; cần chọn giữa lượng tử hóa judge (AWQ/FP8), hạ `--max-model-len`, hoặc chạy hai lượt (sinh câu trả lời trước, chấm điểm sau).
6. ~~Số câu đánh giá~~ **Đã chốt: 100 câu test** (20 mỗi hop); xem kết quả rồi quyết định có tăng không.
6. Nếu mức tăng nhỏ hơn sai số của 100 câu, có tăng số câu đánh giá (ví dụ toàn bộ ~312 câu test) cho các dòng cuối không.

