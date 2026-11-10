import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REV_E, REV_L = "e" * 40, "1" * 40

needs_tools = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("bash") is None or shutil.which("curl") is None,
    reason="needs make, bash and curl",
)


def make(*args, env=None):
    return subprocess.run(
        ["make", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})}
    )


@needs_tools
def test_make_b4_plan_downloads_both_models_and_never_trains():
    result = make(
        "-n", "-B", "outputs/b4/metrics.json",
        f"EMBED_FT_REVISION={REV_E}", f"LORA_FT_REVISION={REV_L}", env={"HF_TOKEN": "x"},
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    plan = result.stdout
    order = [
        f"fetch --repo iamtranthanhvu/vihermes-bge-m3-ft --revision {REV_E} --dest outputs/models/embed",
        "--index outputs/dense-b4.npz --force",  # indexed with the downloaded embedding model
        f"fetch --repo iamtranthanhvu/vihermes-qwen35-4b-conductor-lora --revision {REV_L} --dest outputs/models/lora",
        "scripts/serve.sh lora",  # the adapter is on disk before it is served
        "--conductor-model conductor",
    ]
    positions = [plan.index(item) for item in order]
    assert positions == sorted(positions), "steps are out of order"
    assert (
        plan.rindex("--output outputs/b4") > positions[-1]
    )  # the evaluation writes its results last
    assert "--embedding-model outputs/models/embed" in plan
    for forbidden in (
        "finetune-embed",
        "finetune-lora",
        "make-embed-data",
        "make-lora-data",
        "publish",
    ):
        assert forbidden not in plan, f"B4 must not run {forbidden}"


@needs_tools
def test_b4_asks_for_the_pinned_revisions_instead_of_guessing():
    env = {k: v for k, v in os.environ.items() if k not in ("VIHERMES_HUB_DIR",)}
    result = subprocess.run(
        ["make", "outputs/models/embed/hub.json", "HF_TOKEN=x"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert result.returncode != 0
    assert "EMBED_FT_REVISION" in result.stdout + result.stderr


@needs_tools
def test_b4_stops_early_without_a_token():
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "VIHERMES_HUB_DIR")}
    result = subprocess.run(
        ["make", "outputs/models/embed/hub.json", f"EMBED_FT_REVISION={REV_E}"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert result.returncode != 0 and "HF_TOKEN" in result.stdout + result.stderr


def test_evaluation_never_loads_the_training_libraries(tmp_path):
    """B4 has no training step: importing the CLI and running an evaluation must not pull in
    peft or trl (they are imported lazily, only by the finetune commands)."""
    qa = ROOT / "tests" / "fixtures" / "dryrun" / "qa.jsonl"
    code = f"""
import sys
from vihermes.cli import main
def run(*argv):
    sys.argv = ["vihermes", *map(str, argv)]
    main()
run("extract-contexts", "--qa", {str(qa)!r}, "--output", {str(tmp_path / "c.csv")!r})
run("ingest", "--input", {str(tmp_path / "c.csv")!r}, "--db", {str(tmp_path / "g.db")!r})
run("evaluate", "--qa", {str(qa)!r}, "--db", {str(tmp_path / "g.db")!r}, "--dense-weight", 0,
    "--ks", 5, "--limit", 3, "--output", {str(tmp_path / "out")!r})
print("LOADED", sorted(m for m in ("peft", "trl", "datasets") if m in sys.modules))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "LOADED []"


def has_train_extra():
    return (
        subprocess.run(
            [sys.executable, "-c", "import transformers, peft, sentence_transformers"],
            capture_output=True,
        ).returncode
        == 0
    )


@needs_tools
@pytest.mark.skipif(not has_train_extra(), reason="needs the train extra")
def test_dryrun_b4_runs_all_four_baselines_offline_and_builds_the_table():
    result = make("dryrun-b4")
    assert result.returncode == 0, result.stdout[-3500:] + result.stderr[-3500:]
    out = ROOT / "outputs" / "dryrun"
    models = json.loads((out / "b4" / "models.json").read_text())
    assert models["embedding"]["repo"] == "iamtranthanhvu/vihermes-bge-m3-ft"
    assert models["adapter"]["repo"] == "iamtranthanhvu/vihermes-qwen35-4b-conductor-lora"
    published = json.loads((out / "embed-ft" / "train_meta.json").read_text())["hub"]["revision"]
    assert models["embedding"]["revision"] == published  # B4 used exactly what B2 published
    config = json.loads((out / "b4" / "config.json").read_text())
    assert config["conductor_model"] == "conductor" and "models/embed" in config["embedding_model"]
    table = (out / "compare" / "table.md").read_text()
    for label in (
        "B1 (baseline)",
        "B2 (+ fine-tuned embedding)",
        "B3 (+ LoRA Conductor)",
        "B4 (+ both)",
    ):
        assert label in table
