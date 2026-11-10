# B3: B1 plus a LoRA adapter for the Conductor (host H3).
#   make b3        -> $(OUT)/report-b3.tar.gz
# Steps: base index -> training samples -> stop vLLM (training needs the GPU) -> LoRA
#        -> publish the adapter to the private Hugging Face repo -> serve with the adapter
#        -> evaluate -> pack. Finished steps are skipped.
# The retriever is the B1 one on purpose: B3 changes only how the Conductor writes.

LORA_DIR := $(OUT)/lora-conductor
EXTRA_b3 := $(LORA_DIR)/train_meta.json $(LORA_DIR)/data/data_stats.json

ifdef DRYRUN
# A tiny random Llama and a local directory stand in for Qwen3.5-4B and Hugging Face.
LORA_BASE := $(OUT)/tiny-lm
LORA_TRAIN_ARGS := --max-steps 2 --batch-size 1 --grad-accum 1 --max-seq-length 6000 \
	--rank 4 --alpha 8 --no-gradient-checkpointing
LORA_PREREQ := $(OUT)/tiny-lm/config.json
LORA_PATH := $(CURDIR)/$(LORA_DIR)/adapter
VIHERMES_HUB_DIR := $(OUT)/hub
VIHERMES_DEVICE := cpu
else
LORA_BASE := $(GEN_MODEL)
LORA_TRAIN_ARGS := --epochs $(LORA_EPOCHS) --lr $(LORA_LR) --batch-size $(LORA_BATCH) \
	--grad-accum $(LORA_GRAD_ACCUM) --max-seq-length $(LORA_MAX_SEQ) --rank $(LORA_RANK) \
	--alpha $(LORA_ALPHA) $(if $(GEN_REVISION),--revision $(GEN_REVISION))
LORA_PREREQ :=
endif

.PHONY: b3 check-hf-lora dryrun-b3

# Fail in seconds, not after hours of training, when the upload could not work.
check-hf-lora:
ifndef VIHERMES_HUB_DIR
	@test -n "$$HF_TOKEN" || { echo "b3: export HF_TOKEN (the write token for $(HUB_LORA_REPO)) first"; exit 1; }
endif

$(OUT)/tiny-lm/config.json:
	$(PY) scripts/make_tiny_lm.py --output $(OUT)/tiny-lm

$(LORA_DIR)/data/train.jsonl: $(DB) $(OUT)/dense-b1.npz $(OUT)/.split-ok
	mkdir -p $(LORA_DIR)
	$(VH) make-lora-data --qa $(QA) --split $(SPLIT) --db $(DB) --index $(OUT)/dense-b1.npz \
		$(EMBED_BASE_FLAGS) --dense-weight 0.5 --seed $(SEED) --output-dir $(LORA_DIR)/data

# Training needs the whole GPU, so any vLLM server is stopped first.
$(LORA_DIR)/adapter/adapter_config.json: $(LORA_DIR)/data/train.jsonl $(LORA_PREREQ) | check-hf-lora
	$(SERVE) stop
	$(VH) finetune-lora --data-dir $(LORA_DIR)/data --qa $(QA) --split $(SPLIT) --base $(LORA_BASE) \
		--output $(LORA_DIR) --seed $(SEED) $(LORA_TRAIN_ARGS)

$(LORA_DIR)/.published: $(LORA_DIR)/adapter/adapter_config.json
	$(VH) publish --path $(LORA_DIR)/adapter --repo $(HUB_LORA_REPO) --meta $(LORA_DIR)/train_meta.json
	touch $@

$(OUT)/b3/metrics.json: $(DB) $(OUT)/dense-b1.npz $(LORA_DIR)/.published $(OUT)/.split-ok
	$(SERVE) lora
	$(SERVE) judge
	export VIHERMES_VLLM_VERSION=$(VLLM_VERSION); \
	$(EVAL) $(EMBED_BASE_FLAGS) --index $(OUT)/dense-b1.npz --conductor-model $(LORA_NAME) --output $(OUT)/b3
	$(SERVE) stop

b3: $(OUT)/report-b3.tar.gz

dryrun-b3:
	rm -rf outputs/dryrun
	$(MAKE) DRYRUN=1 b3
	$(MAKE) -q DRYRUN=1 outputs/dryrun/report-b3.tar.gz || { echo "dryrun-b3: a second run would redo finished work"; exit 1; }
	@echo "dryrun-b3 ok"
