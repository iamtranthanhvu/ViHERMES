# Running the baselines on a host

One baseline per host. Everything is driven by `make`; settings live in
[`configs/host.env`](../configs/host.env). `main` stays the original paper code, so use the
baseline branch named below.

| Host | Branch | Command | Result |
| :-- | :-- | :-- | :-- |
| H1 | `baseline/1-local` | `make b1` | `outputs/report-b1.tar.gz` |

(H2 to H4 are added with the branches `baseline/2-embed`, `baseline/3-lora`, `baseline/4-all`.)

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

## Putting the table together (any machine, no GPU)

Put the four `report-bN.tar.gz` files in one folder, then:

```bash
make compare            # writes outputs/compare/table.md and results.json
```

`compare` refuses to build the table if the runs used a different split, corpus, judge or
generator model, or a different number of questions.

## Rehearsal without a GPU

```bash
make dryrun
```

Runs the whole B1 chain on a tiny synthetic dataset with a fake model server. Use it to check
the Makefile and scripts before renting a host.

## If something fails

| Symptom | What to do |
| :-- | :-- |
| `serve.sh: ... timed out` | Look at `outputs/logs/*.log`; lower `MAX_MODEL_LEN` or `GEN_GPU_UTIL` in `configs/host.env` |
| `smoke_check: the reply contains a <think> block` | The model ignores the thinking switch; add `--reasoning-parser` to the vLLM command in `scripts/serve.sh` |
| `does not match ... split` | The dataset or the code differs from the one the split was made with; do not continue |
| Out of memory with both servers | Set `JUDGE_QUANT=fp8` (on every host) or lower `GEN_GPU_UTIL` / `JUDGE_GPU_UTIL` |
