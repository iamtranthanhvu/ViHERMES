# Running the baselines on a host

One baseline per host. Everything is driven by `make`; settings live in
[`configs/host.env`](../configs/host.env). `main` stays the original paper code, so use the
baseline branch named below.

| Host | Branch | Command | Result |
| :-- | :-- | :-- | :-- |
| H1 | `baseline/1-local` | `make b1` | `outputs/report-b1.tar.gz` |
| H2 | `baseline/2-embed` | `make b2` | `outputs/report-b2.tar.gz` and a fine-tuned `bge-m3` on Hugging Face |
| H3 | `baseline/3-lora` | `make b3` | `outputs/report-b3.tar.gz` and a LoRA adapter on Hugging Face |

## Before the first run

1. Open `configs/host.env` and fill every `*_REVISION` with the full commit hash shown on the
   model's Hugging Face page, so all hosts load identical weights. Leave `JUDGE_QUANT` the same
   on every host (empty means bf16).
2. Use a 48 GB GPU, Python 3.10 or newer, and `make` on the host.

## H1: B1 (local models, no training)

```bash
git clone <repo> ViHERMES && cd ViHERMES
git checkout baseline/1-local
tmux new -s run                 # keeps the job alive if the connection drops

make setup                      # install, build the corpus, check the split
make smoke                      # first time only: 5 questions, checks thinking is off and JSON is valid
make b1                         # full evaluation of the 100 test questions
```

`make b1` starts the two vLLM servers, evaluates, then stops them. If it is interrupted, run
`make b1` again: finished steps are skipped and the evaluation resumes where it stopped.

Copy the result to your machine:

```bash
scp -P <PORT> root@<IP>:ViHERMES/outputs/report-b1.tar.gz .
```

## H2: B2 (fine-tuned embedding)

```bash
git clone <repo> ViHERMES && cd ViHERMES
git checkout baseline/2-embed
tmux new -s run
export HF_TOKEN=...             # the WRITE token for iamtranthanhvu/vihermes-bge-m3-ft; type it here, never commit it
make setup
make b2
```

`make b2` builds the base index, makes training pairs from the train split, fine-tunes
`bge-m3`, **uploads it to the private repo** (and records the commit hash in
`outputs/embed-ft/train_meta.json`), indexes with the tuned model, evaluates and packs. It
stops immediately if `HF_TOKEN` is missing, so you do not lose hours of training to a failed
upload. If it is interrupted, run `make b2` again: the trained model is kept and finished steps
are skipped.

Copy `outputs/report-b2.tar.gz` to your machine and keep the commit hash printed by the upload;
H4 downloads exactly that revision.

## H3: B3 (LoRA for the Conductor)

```bash
git clone <repo> ViHERMES && cd ViHERMES
git checkout baseline/3-lora
tmux new -s run
export HF_TOKEN=...             # the WRITE token for iamtranthanhvu/vihermes-qwen35-4b-conductor-lora
make setup
make b3
```

`make b3` builds the base index, makes training samples from the train split (the exact
Conductor prompt, answers quoted verbatim from the gold passages, plus abstentions), stops any
vLLM server (training needs the whole GPU), trains a LoRA adapter, **uploads it to the private
repo** (commit hash recorded in `outputs/lora-conductor/train_meta.json`), serves the generator
with the adapter, evaluates with `--conductor-model conductor` and packs. The retriever is the
B1 one: B3 changes only how the Conductor writes. It stops immediately if `HF_TOKEN` is missing.

`outputs/lora-conductor/train_meta.json` also records how many training tokens each sample has;
if many samples are dropped for exceeding `LORA_MAX_SEQ`, raise it in `configs/host.env`.

## Putting the table together (any machine, no GPU)

Put the four `report-bN.tar.gz` files in one folder, then:

```bash
make compare            # writes outputs/compare/table.md and results.json
```

`compare` refuses to build the table if the runs used a different split, corpus, judge or
generator model, or a different number of questions.

## Rehearsal without a GPU

```bash
make dryrun          # B1
make dryrun-b2       # B2 (needs the train extra: pip install -e '.[local-models,train]')
make dryrun-b3       # B3 (needs the train extra)
```

Runs the whole chain on a tiny synthetic dataset with a fake model server (and, for B2 and B3, a tiny
random model and a local folder instead of Hugging Face). Use it to check the Makefile and
scripts before renting a host. On a Mac, set `VIHERMES_DEVICE=cpu` for any run that loads an
embedding model: the automatic MPS device can crash.

## If something fails

| Symptom | What to do |
| :-- | :-- |
| `serve.sh: ... timed out` | Look at `outputs/logs/*.log`; lower `MAX_MODEL_LEN` or `GEN_GPU_UTIL` in `configs/host.env` |
| `smoke_check: the reply contains a <think> block` | The model ignores the thinking switch; add `--reasoning-parser` to the vLLM command in `scripts/serve.sh` |
| `does not match ... split` | The dataset or the code differs from the one the split was made with; do not continue |
| Out of memory with both servers | Set `JUDGE_QUANT=fp8` (on every host) or lower `GEN_GPU_UTIL` / `JUDGE_GPU_UTIL` |
