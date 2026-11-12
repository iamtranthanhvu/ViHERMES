import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "chat.sh"
TOKEN = "hf_SECRET_TOKEN_123"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None, reason="needs bash and curl"
)

# One stub plays make, serve.sh and the vihermes command; it records how it was called.
STUB = f"""#!{sys.executable}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
who, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({{"who": who, "args": args,
                        "has_token": "HF_TOKEN" in os.environ}}) + "\\n")
if who == "vh" and args[0] == "fetch":
    dest = args[args.index("--dest") + 1]
    os.makedirs(dest, exist_ok=True)
    open(os.path.join(dest, "adapter_config.json"), "w").write("{{}}")
if who == "vh" and args[0] == "chat":
    port = int(args[args.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("make", "serve", "vh", "nvidia-smi"):
        path = bin_dir / name
        path.write_text(STUB)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    values = {
        **os.environ,
        "MAKE_BIN": str(bin_dir / "make"),
        "SERVE_SH": str(bin_dir / "serve"),
        "VH_BIN": str(bin_dir / "vh"),
        "NVIDIA_SMI": str(bin_dir / "nvidia-smi"),
        "CALLS": str(tmp_path / "calls.jsonl"),
        "LOG_DIR": str(tmp_path / "logs"),
        "LORA_PATH": str(tmp_path / "adapter"),
        "OUT": str(tmp_path / "out"),
        "CHAT_PORT": str(free_port()),
        "CHAT_TIMEOUT": "20",
    }
    values.pop("HF_TOKEN", None)
    yield values
    subprocess.run(["bash", SCRIPT, "stop"], env=values, capture_output=True, cwd=ROOT)


def start(env, stdin=""):
    return subprocess.run(
        ["bash", SCRIPT], env=env, input=stdin, capture_output=True, text=True, cwd=ROOT, timeout=60
    )


def calls(env):
    path = Path(env["CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_first_run_does_every_step_in_order_and_never_leaks_the_token(env, tmp_path):
    done = start(env, TOKEN + "\n")
    assert done.returncode == 0, done.stderr
    steps = [(c["who"], c["args"][0]) for c in calls(env)]
    assert steps == [
        ("make", "setup"),
        ("make", f"{env['OUT']}/dense-b1.npz"),
        ("vh", "fetch"),
        ("serve", "lora"),
        ("vh", "chat"),
    ]
    fetch = next(c for c in calls(env) if c["args"][0] == "fetch")
    assert fetch["has_token"] and TOKEN not in " ".join(fetch["args"])
    assert not any(c["has_token"] for c in calls(env) if c["args"][0] != "fetch")
    assert "ca7768c356aa271d8c3a1337fe7204115728429e" in " ".join(fetch["args"])
    for path in [tmp_path / "calls.jsonl", *(tmp_path / "logs").glob("*")]:
        assert TOKEN not in path.read_text(errors="ignore")
    assert TOKEN not in done.stdout + done.stderr
    assert f"127.0.0.1:{env['CHAT_PORT']}" in done.stdout and "ssh -N -L" in done.stdout


def test_chat_gets_the_b3_arguments(env):
    start(env, TOKEN + "\n")
    chat = next(c["args"] for c in calls(env) if c["args"][0] == "chat")
    pairs = dict(zip(chat[1:], chat[2:]))
    assert pairs["--conductor-model"] == "conductor"
    assert pairs["--embedding-model"] == "BAAI/bge-m3"
    assert pairs["--embedding"] == "sentence-transformers"
    assert "--generator" not in chat


def test_second_run_repeats_nothing(env):
    assert start(env, TOKEN + "\n").returncode == 0
    before = len([c for c in calls(env) if c["args"][0] in ("fetch", "chat")])
    done = start(env, "")
    assert done.returncode == 0, done.stderr
    after = calls(env)
    assert len([c for c in after if c["args"][0] in ("fetch", "chat")]) == before
    assert "already running" in done.stdout


def test_stop_stops_chat_and_vllm(env):
    start(env, TOKEN + "\n")
    done = subprocess.run(
        ["bash", SCRIPT, "stop"], env=env, capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert calls(env)[-1]["args"] == ["stop"]
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", int(env["CHAT_PORT"]))) != 0


def test_no_gpu_stops_before_doing_anything(env):
    env["NVIDIA_SMI"] = "/no/such/nvidia-smi"
    done = start(env)
    assert done.returncode == 1 and "GPU" in done.stderr
    assert calls(env) == []


def test_empty_token_is_refused(env):
    done = start(env, "\n")
    assert done.returncode == 1 and "token" in done.stderr
    assert not [c for c in calls(env) if c["args"][0] in ("serve", "chat")]
