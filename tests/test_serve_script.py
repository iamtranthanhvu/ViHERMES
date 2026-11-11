import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVE = ROOT / "scripts" / "serve.sh"
HOST_ENV = ROOT / "configs" / "host.env"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None, reason="needs bash and curl"
)

FAKE_VLLM = f"""#!{sys.executable}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if "--version" in args:
    print("0.0.fake")
    sys.exit(0)
with open(os.environ["FAKE_VLLM_ARGS"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if os.environ.get("FAKE_VLLM_CRASH"):
    sys.exit(3)
if os.environ.get("FAKE_VLLM_NEVER"):
    time.sleep(600)
port = int(args[args.index("--port") + 1])

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({{"data": [{{"id": args[1]}}]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path):
    fake = tmp_path / "vllm"
    fake.write_text(FAKE_VLLM)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    values = {
        **os.environ,
        "VLLM_BIN": str(fake),
        "LOG_DIR": str(tmp_path / "logs"),
        "GEN_PORT": str(free_port()),
        "JUDGE_PORT": str(free_port()),
        "LORA_PATH": str(adapter),
        "SERVE_TIMEOUT": "20",
        "FAKE_VLLM_ARGS": str(tmp_path / "args.jsonl"),
    }
    yield values
    subprocess.run(["bash", str(SERVE), "stop"], env=values, capture_output=True)


def serve(env, *args, **overrides):
    return subprocess.run(
        ["bash", str(SERVE), *args], env={**env, **overrides}, capture_output=True, text=True
    )


def launched(env):
    return [json.loads(line) for line in Path(env["FAKE_VLLM_ARGS"]).read_text().splitlines()]


def test_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(SERVE)]).returncode == 0


def test_unknown_mode_prints_usage_and_exits_2(env):
    result = serve(env, "bogus")
    assert result.returncode == 2 and "usage" in result.stderr.lower()


def test_missing_vllm_binary_is_a_clear_error(env):
    result = serve(env, "base", VLLM_BIN="/nonexistent/vllm")
    assert result.returncode != 0 and "vllm" in result.stderr.lower()


def test_base_mode_waits_until_ready_and_passes_settings(env):
    result = serve(env, "base", GEN_REVISION="a" * 40)
    assert result.returncode == 0, result.stderr
    args = launched(env)[-1]
    assert args[:2] == ["serve", "Qwen/Qwen3.5-4B"]
    assert args[args.index("--port") + 1] == env["GEN_PORT"]
    assert args[args.index("--revision") + 1] == "a" * 40
    assert "--enable-lora" not in args
    # Qwen3.5 needs a Mamba state block per sequence; the default of 256 does not fit beside a 9B.
    assert args[args.index("--max-num-seqs") + 1] == "8"
    assert (Path(env["LOG_DIR"]) / "vllm.version").read_text().strip() == "0.0.fake"


def test_serving_twice_does_not_start_a_second_server(env):
    assert serve(env, "base").returncode == 0
    assert serve(env, "base").returncode == 0
    assert len(launched(env)) == 1


def test_lora_mode_serves_the_adapter_and_replaces_a_base_server(env):
    assert serve(env, "base").returncode == 0
    result = serve(env, "lora")
    assert result.returncode == 0, result.stderr
    args = launched(env)[-1]
    assert "--enable-lora" in args
    assert args[args.index("--lora-modules") + 1] == f"conductor={env['LORA_PATH']}"
    assert len(launched(env)) == 2


def test_lora_mode_needs_an_adapter(env, tmp_path):
    result = serve(env, "lora", LORA_PATH=str(tmp_path / "missing"))
    assert result.returncode != 0 and "adapter" in result.stderr.lower()


def test_judge_mode_uses_the_judge_model_and_port(env):
    assert serve(env, "judge", JUDGE_QUANT="fp8").returncode == 0
    args = launched(env)[-1]
    assert args[1] == "Qwen/Qwen3.5-9B"
    assert args[args.index("--port") + 1] == env["JUDGE_PORT"]
    assert args[args.index("--quantization") + 1] == "fp8"


def test_a_crashing_server_fails_fast_with_the_log(env):
    result = serve(env, "base", FAKE_VLLM_CRASH="1")
    assert result.returncode != 0 and "exited" in result.stderr


def test_a_server_that_never_gets_ready_times_out_and_is_killed(env):
    result = serve(env, "base", FAKE_VLLM_NEVER="1", SERVE_TIMEOUT="3")
    assert result.returncode != 0 and "timed out" in result.stderr


def test_stop_ends_servers_and_is_safe_to_repeat(env):
    assert serve(env, "base").returncode == 0
    assert serve(env, "stop").returncode == 0
    assert serve(env, "stop").returncode == 0
    probe = subprocess.run(
        ["curl", "-sf", f"http://127.0.0.1:{env['GEN_PORT']}/v1/models"], capture_output=True
    )
    assert probe.returncode != 0


def test_host_env_has_no_secrets_and_the_agreed_repositories():
    text = HOST_ENV.read_text()
    assert not re.search(r"hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{10,}", text)
    assert "HF_TOKEN=" not in text.replace("# HF_TOKEN=", "")
    assert "HF_USER=iamtranthanhvu" in text
    assert "vihermes-bge-m3-ft" in text and "vihermes-qwen35-4b-conductor-lora" in text
    # `include`d by make, so no inline comments (they would become part of a value)
    assert not re.search(r"^[A-Z_]+=.*\s#", text, re.M)
