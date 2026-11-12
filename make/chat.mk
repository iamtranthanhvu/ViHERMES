# Chat on the B3 pipeline. Running it is `bash scripts/chat.sh` (needs a GPU host).
#   make dryrun-chat   check the chat code offline: its tests run the real pipeline against the
#                      fake model server in scripts/fake_vllm.py. This is a test, not a mode of
#                      the product; the chat itself has no offline mode.

.PHONY: dryrun-chat

dryrun-chat:
	$(PY) -m pytest -q tests/test_chat.py tests/test_chat_server.py tests/test_chat_cli.py \
		tests/test_chat_page.py tests/test_chat_script.py
	@echo "dryrun-chat ok"
