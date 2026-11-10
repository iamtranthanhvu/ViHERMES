import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE = ROOT / "scripts" / "fake_vllm.py"
SMOKE = ROOT / "scripts" / "smoke_check.py"

needs_tools = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None or shutil.which("curl") is None,
    reason="needs make, bash and curl",
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server():
    started = []

    def start(**env):
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, str(FAKE), "serve", "fake-4b", "--port", str(port)],
            env={**os.environ, **env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                return f"http://127.0.0.1:{port}/v1"
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("fake vllm did not start")

    yield start
    for proc in started:
        proc.terminate()


def smoke(*args):
    return subprocess.run(
        [sys.executable, str(SMOKE), *map(str, args)], capture_output=True, text=True
    )


def write_predictions(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_smoke_check_accepts_a_clean_server_and_run(fake_server, tmp_path):
    url = fake_server()
    rows = [
        {"id": f"q{i}", "answer": {"abstained": False, "trace": ["Auditor: passed"]}}
        for i in range(5)
    ]
    write_predictions(tmp_path / "run", rows)
    result = smoke("--base-url", url, "--model", "fake-4b", "--run", tmp_path / "run")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "abstention rate 0/5" in result.stdout


def test_smoke_check_rejects_reasoning_blocks_in_content(fake_server):
    url = fake_server(FAKE_VLLM_THINK="1")
    result = smoke("--base-url", url, "--model", "fake-4b")
    assert result.returncode != 0
    assert "think" in result.stderr.lower()


def test_smoke_check_reports_an_unreachable_server():
    result = smoke("--base-url", f"http://127.0.0.1:{free_port()}/v1", "--model", "m")
    assert result.returncode != 0 and "reach" in result.stderr.lower()


def test_smoke_check_rejects_a_run_with_invalid_structured_answers(fake_server, tmp_path):
    url = fake_server()
    rows = [
        {
            "id": "q0",
            "answer": {"abstained": True, "trace": ["Conductor: invalid structured answer"]},
        }
    ]
    write_predictions(tmp_path / "run", rows)
    result = smoke("--base-url", url, "--model", "fake-4b", "--run", tmp_path / "run")
    assert result.returncode != 0 and "invalid structured answer" in result.stderr


def test_smoke_check_rejects_an_empty_run(fake_server, tmp_path):
    url = fake_server()
    write_predictions(tmp_path / "run", [])
    result = smoke("--base-url", url, "--model", "fake-4b", "--run", tmp_path / "run")
    assert result.returncode != 0 and "no predictions" in result.stderr.lower()


def make(*args, **env):
    return subprocess.run(
        ["make", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **env}
    )


@needs_tools
def test_make_help_lists_the_targets():
    result = make("help")
    assert result.returncode == 0
    for target in ("setup", "smoke", "b1", "compare", "dryrun"):
        assert target in result.stdout


@needs_tools
def test_make_b1_plan_uses_the_agreed_models_and_flags():
    result = make("-n", "-B", "outputs/b1/metrics.json")
    assert result.returncode == 0, result.stderr
    plan = result.stdout
    for expected in (
        "--model Qwen/Qwen3.5-4B",
        "--judge-model Qwen/Qwen3.5-9B",
        "--judge-base-url http://127.0.0.1:8001/v1",
        "--part test",
        "--index outputs/dense-b1.npz",
        "--embedding-model BAAI/bge-m3",
        "--no-thinking",
        "--resume",
        "--ks 5",
    ):
        assert expected in plan, expected


@needs_tools
def test_dryrun_runs_the_whole_b1_chain_offline_and_is_idempotent():
    result = make("dryrun")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    report = ROOT / "outputs" / "dryrun" / "report-b1.tar.gz"
    assert report.exists()
    table = (ROOT / "outputs" / "dryrun" / "compare" / "table.md").read_text()
    assert "| B1 (baseline) |" in table
    metrics = json.loads((ROOT / "outputs" / "dryrun" / "b1" / "metrics.json").read_text())
    assert metrics["overall"]["n"] == 10
    for key in ("f1", "judge", "recall@5"):
        assert metrics["overall"][key] is not None
    again = make("-q", "DRYRUN=1", "outputs/dryrun/report-b1.tar.gz")
    assert again.returncode == 0, "a second run would redo finished work"
