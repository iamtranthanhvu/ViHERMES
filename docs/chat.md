# Chat on the B3 pipeline

A web page where you type a Vietnamese question about health regulations and get an answer that
quotes the source text, or a clear refusal. It runs **only** the B3 pipeline from
[baselines.md](baselines.md): BM25 + base `BAAI/bge-m3`, and `Qwen/Qwen3.5-4B` with the Conductor
LoRA `iamtranthanhvu/vihermes-qwen35-4b-conductor-lora`, served by one vLLM. There is no offline or
mock mode; without a GPU the chat does not start.

## Run

On a GPU host (for example a rented Vast.ai machine):

```bash
git clone https://github.com/iamtranthanhvu/ViHERMES.git && cd ViHERMES
git checkout feature/chat
bash scripts/chat.sh          # sets up, downloads the adapter, starts vLLM and the chat
```

The script asks once for a Hugging Face token with READ access to the adapter repo (typed hidden,
never written to a file or log), then prints an SSH command to run on your own machine and the
address to open (`http://127.0.0.1:8080`). Run it again any time: finished steps are skipped.
`bash scripts/chat.sh stop` stops everything. The chat listens on `127.0.0.1` only.

Ports, models and paths come from `configs/host.env`; the adapter commit is `CHAT_LORA_REVISION`.

## What to expect

B3 answered **7 of 25** test questions and refused 18 (F1 0.218, Judge 0.299; about ±20 points of
noise on 25 questions). Refusing is normal: the page shows why and the documents that were found.
The corpus is the **998 passages** of the ViHERMES dataset, not all health law. This is a lookup
tool, not legal advice. Each question is independent (no conversation memory). One question runs
at a time, about 10 seconds each on an RTX 4090; at most three wait.

Nothing is stored on the server and logs hold no question text. The history in the sidebar lives
only in the open browser tab.

## API

| Method | Path | |
| :-- | :-- | :-- |
| `GET` | `/api/health` | `{"ok", "model", "lora", "corpus_units", "mode": "b3"}`; `503` if vLLM is down |
| `POST` | `/api/ask` | `{"question"}` (1 to 500 characters) returns `status` (`answered`, `refused`, `clarification`), `answer`, `sources`, and `reason` when refused |

Errors: `422` bad question, `429` queue full, `503` model not ready, `500` anything else.

## Tests

`make dryrun-chat` runs the chat tests offline. They drive the real Conductor and Auditor against
`scripts/fake_vllm.py`; that fake exists only for tests. Whether the real model answers well on a
host is checked by hand (`/api/health`, then a few questions).
