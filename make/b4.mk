# B4: B1 plus the fine-tuned embedding (from B2) and the LoRA adapter (from B3) (host H4).
#   make b4        -> $(OUT)/report-b4.tar.gz
# Nothing is trained here. The two models are downloaded from the private Hugging Face repos at
# the exact commits the B2 and B3 hosts recorded (EMBED_FT_REVISION and LORA_FT_REVISION in
# configs/host.env), then indexed, served and evaluated. Finished steps are skipped.

B4_EMBED := $(OUT)/models/embed
B4_LORA := $(OUT)/models/lora
EXTRA_b4 := $(OUT)/b4/models.json

ifdef DRYRUN
# B2 and B3 published into a local directory; read the revisions they recorded.
EMBED_FT_REVISION = $(shell $(PY) -c "import json;print(json.load(open('$(OUT)/embed-ft/train_meta.json'))['hub']['revision'])")
LORA_FT_REVISION = $(shell $(PY) -c "import json;print(json.load(open('$(OUT)/lora-conductor/train_meta.json'))['hub']['revision'])")
B4_FLAGS := --embedding sentence-transformers --embedding-model $(B4_EMBED) --max-seq-length 32
VIHERMES_HUB_DIR := $(OUT)/hub
VIHERMES_DEVICE := cpu
else
B4_FLAGS := --embedding sentence-transformers --embedding-model $(B4_EMBED) \
	--max-seq-length $(EMBED_MAX_SEQ_LENGTH)
endif

.PHONY: b4 check-hf-read dryrun-b4

# Downloading needs a token that can read the two repos.
check-hf-read:
ifndef VIHERMES_HUB_DIR
	@test -n "$$HF_TOKEN" || { echo "b4: export HF_TOKEN (a read token for $(HUB_EMBED_REPO) and $(HUB_LORA_REPO)) first"; exit 1; }
endif

$(B4_EMBED)/hub.json: | check-hf-read
	@test -n "$(EMBED_FT_REVISION)" || { echo "b4: set EMBED_FT_REVISION in configs/host.env to the hub.revision in outputs/embed-ft/train_meta.json from the B2 host"; exit 1; }
	$(VH) fetch --repo $(HUB_EMBED_REPO) --revision $(EMBED_FT_REVISION) --dest $(B4_EMBED)

$(B4_LORA)/hub.json: | check-hf-read
	@test -n "$(LORA_FT_REVISION)" || { echo "b4: set LORA_FT_REVISION in configs/host.env to the hub.revision in outputs/lora-conductor/train_meta.json from the B3 host"; exit 1; }
	$(VH) fetch --repo $(HUB_LORA_REPO) --revision $(LORA_FT_REVISION) --dest $(B4_LORA)

$(OUT)/dense-b4.npz: $(DB) $(B4_EMBED)/hub.json
	$(VH) index --db $(DB) --index $@ --force $(B4_FLAGS)

# serve.sh looks for the adapter at LORA_PATH; here it is the downloaded one.
$(OUT)/b4/metrics.json: LORA_PATH := $(B4_LORA)
$(OUT)/b4/metrics.json: $(DB) $(OUT)/dense-b4.npz $(B4_LORA)/hub.json $(OUT)/.split-ok
	mkdir -p $(OUT)/b4
	$(PY) scripts/merge_hub_info.py --embedding $(B4_EMBED) --adapter $(B4_LORA) --output $(OUT)/b4/models.json
	$(SERVE) lora
	$(SERVE) judge
	export VIHERMES_VLLM_VERSION=$(VLLM_VERSION); \
	$(EVAL) $(B4_FLAGS) --index $(OUT)/dense-b4.npz --conductor-model $(LORA_NAME) --output $(OUT)/b4
	$(SERVE) stop

b4: $(OUT)/report-b4.tar.gz

# Runs B1, B2 and B3 first (they publish to the local hub), then B4 from what they published,
# then merges the four reports, which exercises the table's comparability checks.
dryrun-b4:
	rm -rf outputs/dryrun
	$(MAKE) DRYRUN=1 b1
	$(MAKE) DRYRUN=1 b2
	$(MAKE) DRYRUN=1 b3
	$(MAKE) DRYRUN=1 b4
	$(MAKE) DRYRUN=1 compare REPORTS="outputs/dryrun/report-b1.tar.gz outputs/dryrun/report-b2.tar.gz outputs/dryrun/report-b3.tar.gz outputs/dryrun/report-b4.tar.gz"
	$(MAKE) -q DRYRUN=1 outputs/dryrun/report-b4.tar.gz || { echo "dryrun-b4: a second run would redo finished work"; exit 1; }
	@echo "dryrun-b4 ok"
