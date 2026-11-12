import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE_VLLM = ROOT / "scripts" / "fake_vllm.py"


def vh(*args, **kwargs):
    return subprocess.Popen(
        [sys.executable, "-m", "vihermes", *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        json.dumps({"id": "D", "title": "Thông tư", "text": "Điều 1. Hồ sơ gồm bệnh án."})
    )
    db = tmp_path / "graph.db"
    done = subprocess.run(
        [sys.executable, "-m", "vihermes", "ingest", "--input", docs, "--db", db],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return db


@pytest.fixture
def fake_model():
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, FAKE_VLLM, "serve", "qwen", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(50):
        try:
            urllib.request.urlopen(url + "/models", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait()


def chat_args(db, url, *extra):
    return [
        "chat", "--db", db, "--embedding", "hash", "--dense-weight", 0, "--model", "qwen",
        "--base-url", url, "--conductor-model", "qwen", "--no-thinking", "--port", 0, *extra,
    ]  # fmt: skip


def test_chat_has_no_offline_generator_switch(corpus, fake_model):
    proc = vh(*chat_args(corpus, fake_model, "--generator", "extractive"))
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 2 and "--generator" in err


def test_chat_refuses_to_start_without_adapter_name(corpus, fake_model):
    proc = vh("chat", "--db", corpus, "--embedding", "hash", "--base-url", fake_model)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 2 and "--conductor-model" in err


def test_chat_refuses_to_start_when_vllm_is_down(corpus):
    proc = vh(*chat_args(corpus, f"http://127.0.0.1:{free_port()}/v1"))
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 2 and "scripts/serve.sh lora" in err


def test_chat_answers_through_the_real_pipeline(corpus, fake_model):
    proc = vh(*chat_args(corpus, fake_model))
    try:
        line = proc.stdout.readline()
        assert line.startswith("vihermes chat: http://"), line
        base = line.split(": ", 1)[1].strip()
        health = json.load(urllib.request.urlopen(base + "/api/health", timeout=10))
        assert health["ok"] and health["mode"] == "b3" and health["corpus_units"] >= 1
        request = urllib.request.Request(
            base + "/api/ask",
            json.dumps({"question": "Hồ sơ gồm gì?"}).encode(),
            {"Content-Type": "application/json"},
        )
        result = json.load(urllib.request.urlopen(request, timeout=30))
        assert result["status"] == "answered"
        assert result["sources"][0]["title"] and result["sources"][0]["quote"]
    finally:
        proc.terminate()
        proc.wait()
