import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

needs_tools = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None or shutil.which("curl") is None,
    reason="needs make, bash and curl",
)


def make(*args, env=None):
    return subprocess.run(
        ["make", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})}
    )


@needs_tools
def test_make_b3_plan_trains_publishes_then_evaluates_with_the_adapter():
    result = make("-n", "-B", "outputs/b3/metrics.json", env={"HF_TOKEN": "x"})
    assert result.returncode == 0, result.stderr
    plan = result.stdout
    order = [
        "make-lora-data",
        "serve.sh stop",
        "finetune-lora",
        "publish --path outputs/lora-conductor/adapter --repo iamtranthanhvu/vihermes-qwen35-4b-conductor-lora",
        "scripts/serve.sh lora",
        "--conductor-model conductor",
        "--output outputs/b3",
    ]
    positions = [plan.index(item) for item in order]
    assert positions == sorted(positions), "steps are out of order"
    assert "--index outputs/dense-b1.npz" in plan  # B3 changes the generator, not the retriever


@needs_tools
def test_b3_stops_early_without_a_hugging_face_token():
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "VIHERMES_HUB_DIR")}
    result = subprocess.run(
        ["make", "outputs/lora-conductor/adapter/adapter_config.json"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert result.returncode != 0 and "HF_TOKEN" in result.stdout + result.stderr


def has_train_extra():
    return (
        subprocess.run(
            [sys.executable, "-c", "import transformers, peft"], capture_output=True
        ).returncode
        == 0
    )


@needs_tools
@pytest.mark.skipif(not has_train_extra(), reason="needs the train extra")
def test_dryrun_b3_trains_publishes_and_evaluates_with_the_adapter_offline():
    result = make("dryrun-b3")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    out = ROOT / "outputs" / "dryrun"
    assert (out / "report-b3.tar.gz").exists()
    meta = json.loads((out / "lora-conductor" / "train_meta.json").read_text())
    assert meta["hub"]["repo"] == "iamtranthanhvu/vihermes-qwen35-4b-conductor-lora"
    assert len(meta["hub"]["revision"]) == 40
    config = json.loads((out / "b3" / "config.json").read_text())
    assert config["conductor_model"] == "conductor" and config["model"] == "fake-4b"
    metrics = json.loads((out / "b3" / "metrics.json").read_text())
    assert metrics["overall"]["n"] == 10 and metrics["overall"]["f1"] is not None
