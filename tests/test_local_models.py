import csv
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("openai")

from vihermes.adapters.models import OpenAILanguageModel, SentenceEmbedder  # noqa: E402

TEXT = "hồ sơ số 0"


def reply_for(system):
    if "Interpreter" in system:
        return {"regulatory": True, "use_graph": False, "document_ids": []}
    if "Conductor" in system:
        return {
            "abstain": False,
            "claims": [{"text": "Hồ sơ số 0.", "citations": ["c0"], "quotes": {"c0": TEXT}}],
        }
    if "Auditor" in system:
        return {"supported": True, "reasons": []}
    return {"correctness": 1, "completeness": 1, "consistency": 1, "reason": "ok"}


def start_server(tag, log):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = body["messages"][0]["content"]
            log.append(
                {
                    "tag": tag,
                    "path": self.path,
                    "model": body["model"],
                    "chat_template_kwargs": body.get("chat_template_kwargs"),
                    "role": next(
                        (r for r in ("Interpreter", "Conductor", "Auditor") if r in system), "judge"
                    ),
                }
            )
            payload = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(reply_for(system))},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


@pytest.fixture
def servers():
    log = []
    gen, gen_url = start_server("gen", log)
    judge, judge_url = start_server("judge", log)
    yield log, gen_url, judge_url
    gen.shutdown()
    judge.shutdown()


def test_language_model_uses_base_url_and_needs_no_api_key(servers, monkeypatch):
    log, url, _ = servers
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    llm = OpenAILanguageModel("qwen", base_url=url)
    assert llm.json("system Auditor", {"q": 1}) == {"supported": True, "reasons": []}
    assert log[0]["path"].endswith("/chat/completions") and log[0]["model"] == "qwen"
    assert log[0]["chat_template_kwargs"] is None


def test_extra_body_disables_thinking(servers, monkeypatch):
    log, url, _ = servers
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    OpenAILanguageModel("qwen", base_url=url, extra_body=extra).json("Auditor", {})
    assert log[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_default_model_still_reads_the_environment(servers, monkeypatch):
    log, url, _ = servers
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    OpenAILanguageModel("gpt-4o-mini").json("Auditor", {})
    assert log[0]["model"] == "gpt-4o-mini" and log[0]["chat_template_kwargs"] is None


class FakeSentenceTransformer:
    def __init__(self, model, revision=None, device=None):
        self.model, self.revision, self.device = model, revision, device
        self.max_seq_length = 8192

    def encode(self, texts, batch_size, normalize_embeddings):
        import numpy as np

        return np.ones((len(texts), 3), dtype=np.float32)


def test_sentence_embedder_can_limit_sequence_length(monkeypatch):
    import types

    fake = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    plain = SentenceEmbedder("BAAI/bge-m3")
    limited = SentenceEmbedder("BAAI/bge-m3", max_seq_length=512)
    assert (
        plain.model.max_seq_length == 8192 and plain.identity == "sentence-transformers:BAAI/bge-m3"
    )
    assert limited.model.max_seq_length == 512
    assert (
        limited.identity != plain.identity
    )  # truncation changes the vectors, so the index key too


def test_sentence_embedder_pins_a_model_revision(monkeypatch):
    import types

    fake = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    plain = SentenceEmbedder("BAAI/bge-m3")
    pinned = SentenceEmbedder("BAAI/bge-m3", revision="a" * 40)
    assert plain.model.revision is None and pinned.model.revision == "a" * 40
    assert pinned.identity != plain.identity


def test_sentence_embedder_device_can_be_forced_from_the_environment(monkeypatch):
    import types

    fake = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.delenv("VIHERMES_DEVICE", raising=False)
    assert SentenceEmbedder("m").model.device is None  # library default
    monkeypatch.setenv("VIHERMES_DEVICE", "cpu")
    assert SentenceEmbedder("m").model.device == "cpu"


def cli(*args, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)],
        capture_output=True,
        text=True,
        **kwargs,
    )


def test_generator_judge_and_conductor_use_separate_endpoints_and_models(servers, tmp_path):
    log, gen_url, judge_url = servers
    contexts = tmp_path / "contexts.csv"
    with contexts.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "title", "text"])
        writer.writeheader()
        writer.writerow({"context_id": "c0", "title": "Quy định", "text": TEXT})
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps(
            {"id": "q0", "question": TEXT, "answer": "Hồ sơ số 0.", "hop_level": 1,
             "used_context_ids": ["c0"]}
        ),
        encoding="utf-8",
    )  # fmt: skip
    db = tmp_path / "g.db"
    assert cli("ingest", "--input", contexts, "--db", db).returncode == 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    result = cli(
        "evaluate", "--qa", qa, "--db", db, "--dense-weight", 0, "--ks", 5,
        "--model", "base-4b", "--conductor-model", "conductor", "--base-url", gen_url,
        "--judge-model", "judge-9b", "--judge-base-url", judge_url,
        "--answers", "--judge", "--no-thinking", "--output", tmp_path / "out",
        env=env,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    seen = {(e["tag"], e["role"], e["model"]) for e in log}
    assert seen == {
        ("gen", "Interpreter", "base-4b"),
        ("gen", "Conductor", "conductor"),
        ("gen", "Auditor", "base-4b"),
        ("judge", "judge", "judge-9b"),
    }
    assert all(e["chat_template_kwargs"] == {"enable_thinking": False} for e in log)
    metrics = json.loads(result.stdout)["overall"]
    assert metrics["recall@5"] == 1 and metrics["judge"] == 1 and metrics["f1"] > 0
