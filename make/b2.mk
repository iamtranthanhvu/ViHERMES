# B2: B1 plus a fine-tuned embedding model (host H2).
#   make b2        -> $(OUT)/report-b2.tar.gz
# Steps: base index -> training pairs -> fine-tune -> publish to the private Hugging Face repo
#        -> index with the tuned model -> evaluate -> pack. Finished steps are skipped.

EMBED_FT_DIR := $(OUT)/embed-ft
EXTRA_b2 := $(EMBED_FT_DIR)/train_meta.json

ifdef DRYRUN
# A tiny random model and a local directory stand in for bge-m3 and Hugging Face.
FT_BASE := $(OUT)/tiny-st
FT_TRAIN_ARGS := --max-steps 2 --batch-size 4 --max-seq-length 32
FT_FLAGS := --embedding sentence-transformers --embedding-model $(EMBED_FT_DIR)/model --max-seq-length 32
VIHERMES_HUB_DIR := $(OUT)/hub
VIHERMES_DEVICE := cpu
FT_PREREQ := $(OUT)/tiny-st/modules.json
else
FT_BASE := $(EMBED_BASE)
FT_TRAIN_ARGS := --epochs $(EMBED_EPOCHS) --lr $(EMBED_LR) --batch-size $(EMBED_BATCH) \
	--max-seq-length $(EMBED_MAX_SEQ_LENGTH) --negatives $(EMBED_NEGATIVES) \
	$(if $(EMBED_REVISION),--revision $(EMBED_REVISION))
FT_FLAGS := --embedding sentence-transformers --embedding-model $(EMBED_FT_DIR)/model \
	--max-seq-length $(EMBED_MAX_SEQ_LENGTH)
FT_PREREQ :=
endif

.PHONY: b2 check-hf dryrun-b2

# Fail in seconds, not after hours of training, when the upload could not work.
check-hf:
ifndef VIHERMES_HUB_DIR
	@test -n "$$HF_TOKEN" || { echo "b2: export HF_TOKEN (the write token for $(HUB_EMBED_REPO)) first"; exit 1; }
endif

$(OUT)/tiny-st/modules.json:
	$(PY) scripts/make_tiny_model.py --output $(OUT)/tiny-st

$(EMBED_FT_DIR)/data.jsonl: $(DB) $(OUT)/dense-b1.npz $(OUT)/.split-ok
	mkdir -p $(EMBED_FT_DIR)
	$(VH) make-embed-data --qa $(QA) --split $(SPLIT) --db $(DB) --index $(OUT)/dense-b1.npz \
		$(EMBED_BASE_FLAGS) --dense-weight 0.5 --negatives $(EMBED_NEGATIVES) --seed $(SEED) --output $@

$(EMBED_FT_DIR)/model/modules.json: $(EMBED_FT_DIR)/data.jsonl $(FT_PREREQ) | check-hf
	$(VH) finetune-embed --data $< --qa $(QA) --split $(SPLIT) --db $(DB) --base $(FT_BASE) \
		--output $(EMBED_FT_DIR) --seed $(SEED) $(FT_TRAIN_ARGS)

$(EMBED_FT_DIR)/.published: $(EMBED_FT_DIR)/model/modules.json
	$(VH) publish --path $(EMBED_FT_DIR)/model --repo $(HUB_EMBED_REPO) --meta $(EMBED_FT_DIR)/train_meta.json
	touch $@

$(OUT)/dense-b2.npz: $(DB) $(EMBED_FT_DIR)/.published
	$(VH) index --db $(DB) --index $@ --force $(FT_FLAGS)

$(OUT)/b2/metrics.json: $(DB) $(OUT)/dense-b2.npz $(OUT)/.split-ok
	$(SERVE) base
	$(SERVE) judge
	export VIHERMES_VLLM_VERSION=$(VLLM_VERSION); \
	$(EVAL) $(FT_FLAGS) --index $(OUT)/dense-b2.npz --output $(OUT)/b2
	$(SERVE) stop

b2: $(OUT)/report-b2.tar.gz

dryrun-b2:
	rm -rf outputs/dryrun
	$(MAKE) DRYRUN=1 b2
	$(MAKE) -q DRYRUN=1 outputs/dryrun/report-b2.tar.gz || { echo "dryrun-b2: a second run would redo finished work"; exit 1; }
	@echo "dryrun-b2 ok"
