# ViHERMES baselines. Run `make help` for the targets.
#
# Per host: `make setup` once, then one target (b1 here; b2, b3, b4 come from make/*.mk).
# Settings live in configs/host.env; override any of them on the command line,
# e.g. `make b1 GEN_PORT=9000`.

SHELL := /bin/bash
.DEFAULT_GOAL := help
.EXPORT_ALL_VARIABLES:

# BM25 adds up scores over a set of query words, and a set of strings iterates in a different
# order in every process unless the hash seed is fixed. The sums then differ in their last bits,
# which reorders passages with almost equal scores, so two identical runs could rank differently.
# A fixed seed makes every host and every re-run retrieve the same passages in the same order.
PYTHONHASHSEED := 0

include configs/host.env

VENV ?= .venv
VH := $(VENV)/bin/vihermes
PY := $(VENV)/bin/python
QA ?= dataset/dataset.jsonl
OUT ?= outputs
SPLIT ?= configs/split.json

ifdef DRYRUN
# Offline rehearsal: a tiny dataset, a hash encoder and scripts/fake_vllm.py instead of vLLM.
QA := tests/fixtures/dryrun/qa.jsonl
OUT := outputs/dryrun
SPLIT := $(OUT)/split.json
LOG_DIR := $(OUT)/logs
VLLM_BIN := $(CURDIR)/scripts/fake_vllm.py
GEN_MODEL := fake-4b
JUDGE_MODEL := fake-9b
GEN_PORT := 18080
JUDGE_PORT := 18081
EMBED_BASE_FLAGS := --embedding hash
else
EMBED_BASE_FLAGS := --embedding sentence-transformers --embedding-model $(EMBED_BASE) \
	--max-seq-length $(EMBED_MAX_SEQ_LENGTH) \
	$(if $(EMBED_REVISION),--embedding-revision $(EMBED_REVISION))
endif

DB := $(OUT)/vihermes.db
CONTEXTS := $(OUT)/contexts.csv
GEN_URL := http://127.0.0.1:$(GEN_PORT)/v1
JUDGE_URL := http://127.0.0.1:$(JUDGE_PORT)/v1

# One evaluation of the held-out test set; the caller adds the embedding, index, model
# overrides and --output.
EVAL = $(VH) evaluate --qa $(QA) --split $(SPLIT) --part test --db $(DB) \
	--dense-weight 0.5 -k 5 --ks 5 --answers --judge --no-thinking --resume \
	--model $(GEN_MODEL) --base-url $(GEN_URL) \
	--judge-model $(JUDGE_MODEL) --judge-base-url $(JUDGE_URL)
VLLM_ENV := $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(VLLM_BIN)))))
SERVE := scripts/serve.sh
VLLM_VERSION = "$$(cat $(LOG_DIR)/vllm.version 2>/dev/null)"

.PHONY: help setup setup-vllm smoke b1 compare dryrun

help:
	@echo "make setup      install the project and vLLM, build the corpus, verify the split"
	@echo "make smoke      first-run check: serve the models, evaluate 5 questions, inspect them"
	@echo "make b1         B1: local models, no training -> $(OUT)/report-b1.tar.gz"
	@echo "make compare    merge report-b*.tar.gz into $(OUT)/compare/table.md (no GPU needed)"
	@echo "make dryrun     rehearse the whole B1 chain offline with a fake model server"
	@echo "Settings: configs/host.env (override on the command line)."

# ---- setup -------------------------------------------------------------------------------

setup: $(VENV)/.installed setup-vllm $(DB) $(OUT)/.split-ok

$(VENV)/.installed: pyproject.toml
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[openai,local-models,dataset,train]'
	touch $@

# vLLM lives in its own virtualenv so its torch does not clash with sentence-transformers.
# `make setup SKIP_VLLM=1` skips it (a machine without a GPU).
setup-vllm:
ifndef SKIP_VLLM
	test -x $(VLLM_BIN) || { python3 -m venv $(VLLM_ENV) && $(VLLM_ENV)/bin/pip install -q vllm; }
endif

$(CONTEXTS): $(QA)
	mkdir -p $(OUT)
	$(VH) extract-contexts --qa $(QA) --output $@

$(DB): $(CONTEXTS)
	rm -f $@ $@-shm $@-wal
	$(VH) ingest --input $(CONTEXTS) --db $@

ifdef DRYRUN
$(SPLIT): $(QA)
	mkdir -p $(OUT)
	$(VH) split --qa $(QA) --output $@ --test-size 10
endif

# Every host must prove it uses the committed split on the same dataset.
$(OUT)/.split-ok: $(SPLIT) $(QA)
	$(VH) split --qa $(QA) --check $(SPLIT)
	touch $@

# ---- B1: local models, no training ---------------------------------------------------------

$(OUT)/dense-b1.npz: $(DB) $(OUT)/.split-ok
	$(VH) index --db $(DB) --index $@ --force $(EMBED_BASE_FLAGS)

$(OUT)/b1/metrics.json: $(DB) $(OUT)/dense-b1.npz $(OUT)/.split-ok
	$(SERVE) base
	$(SERVE) judge
	export VIHERMES_VLLM_VERSION=$(VLLM_VERSION); \
	$(EVAL) $(EMBED_BASE_FLAGS) --index $(OUT)/dense-b1.npz --output $(OUT)/b1
	$(SERVE) stop

$(OUT)/report-%.tar.gz: $(OUT)/%/metrics.json
	$(VH) pack --run $(OUT)/$* --name $* --output $@ --extra $(SPLIT) $(EXTRA_$*)

b1: $(OUT)/report-b1.tar.gz

# Five questions, started fresh each time (--resume would otherwise reuse old answers).
smoke: $(DB) $(OUT)/dense-b1.npz $(OUT)/.split-ok
	$(SERVE) base
	$(SERVE) judge
	rm -f $(OUT)/smoke/predictions.jsonl
	export VIHERMES_VLLM_VERSION=$(VLLM_VERSION); \
	$(EVAL) $(EMBED_BASE_FLAGS) --index $(OUT)/dense-b1.npz --output $(OUT)/smoke --limit 5
	$(PY) scripts/smoke_check.py --base-url $(GEN_URL) --model $(GEN_MODEL) --run $(OUT)/smoke
	$(SERVE) stop

# ---- results -----------------------------------------------------------------------------

REPORTS ?= $(sort $(wildcard report-b*.tar.gz $(OUT)/report-b*.tar.gz))
compare:
	$(VH) compare $(REPORTS) --output-dir $(OUT)/compare
	@cat $(OUT)/compare/table.md

# ---- offline rehearsal ---------------------------------------------------------------------

# The first line removes only the fixed rehearsal directory.
dryrun:
	rm -rf outputs/dryrun
	$(MAKE) DRYRUN=1 b1
	$(MAKE) DRYRUN=1 compare REPORTS=outputs/dryrun/report-b1.tar.gz
	$(MAKE) -q DRYRUN=1 outputs/dryrun/report-b1.tar.gz || { echo "dryrun: a second run would redo finished work"; exit 1; }
	@echo "dryrun ok"

-include make/*.mk
