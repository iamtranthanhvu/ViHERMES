import json
import subprocess
import sys
import tarfile

import pytest

from vihermes.report import compare, pack

SPLIT = {"hash": "d" * 64, "part": "test"}
JUDGE = {"model": "Qwen/Qwen3.5-9B", "quant": "bf16", "revision": None}


def make_run(root, name, f1, judge_score, recall, **overrides):
    run = root / name
    run.mkdir(parents=True)
    config = {
        "model": "Qwen/Qwen3.5-4B",
        "split": SPLIT,
        "corpus": "c" * 64,
        "judge": JUDGE,
        "environment": {"git_commit": "abc"},
    }
    config.update({k: v for k, v in overrides.items() if k != "n"})
    metrics = {
        "overall": {
            "n": overrides.get("n", 100),
            "f1": f1,
            "judge": judge_score,
            "recall@5": recall,
        },
        "by_hop": {"1": {"n": 20, "f1": f1, "judge": judge_score, "recall@5": recall}},
        "metadata": {"split": SPLIT},
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "predictions.jsonl").write_text('{"id": "q0"}\n')
    (run / "timing.json").write_text('{"seconds_this_run": 1.5}')
    (run / "weights.safetensors").write_bytes(b"x" * 10)  # must never be packed
    return run


@pytest.fixture
def reports(tmp_path):
    runs = {
        "b1": make_run(tmp_path, "b1", 0.5000, 0.4000, 0.6000),
        "b2": make_run(tmp_path, "b2", 0.5100, 0.4000, 0.7000),
        "b3": make_run(tmp_path, "b3", 0.4900, 0.4500, 0.6000),
    }
    return {name: pack(run, name, tmp_path / f"report-{name}.tar.gz") for name, run in runs.items()}


def test_pack_keeps_numbers_and_leaves_weights_out(tmp_path):
    run = make_run(tmp_path, "b1", 0.5, 0.4, 0.6)
    extra = tmp_path / "train_meta.json"
    extra.write_text("{}")
    out = pack(run, "b1", tmp_path / "r.tar.gz", extra=[extra])
    with tarfile.open(out) as tar:
        names = sorted(tar.getnames())
    assert names == [
        "b1/config.json",
        "b1/metrics.json",
        "b1/predictions.jsonl",
        "b1/timing.json",
        "b1/train_meta.json",
    ]


def test_pack_refuses_weight_files_and_incomplete_runs(tmp_path):
    run = make_run(tmp_path, "b1", 0.5, 0.4, 0.6)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"x")
    with pytest.raises(ValueError, match="weights|binary|not allowed"):
        pack(run, "b1", tmp_path / "r.tar.gz", extra=[weights])
    (run / "metrics.json").unlink()
    with pytest.raises(ValueError, match="metrics.json"):
        pack(run, "b1", tmp_path / "r.tar.gz")


def test_table_has_paper_columns_deltas_and_bold_best(reports, tmp_path):
    result = compare([reports["b1"], reports["b2"], reports["b3"]], tmp_path / "out")
    lines = (tmp_path / "out" / "table.md").read_text().splitlines()
    assert lines[0] == "| Method | F1 ↑ | LLM Judge ↑ | Recall@5 ↑ |"
    assert lines[1] == "| :-- | --: | --: | --: |"
    b1, b2, b3 = lines[2:5]
    assert b1 == "| B1 (baseline) | 0.5000 | 0.4000 | 0.6000 |"
    assert (
        b2
        == "| B2 (+ fine-tuned embedding) | **0.5100 (+0.0100)** | 0.4000 (+0.0000) | **0.7000 (+0.1000)** |"
    )
    assert (
        b3
        == "| B3 (+ LoRA Conductor) | 0.4900 (-0.0100) | **0.4500 (+0.0500)** | 0.6000 (+0.0000) |"
    )
    assert result["rows"][1]["delta"]["recall@5"] == pytest.approx(0.1)


def test_tied_best_values_are_all_bold(tmp_path):
    runs = [make_run(tmp_path, n, 0.5, 0.4, 0.6) for n in ("b1", "b2")]
    packed = [pack(r, r.name, tmp_path / f"{r.name}.tar.gz") for r in runs]
    compare(packed, tmp_path / "out")
    rows = (tmp_path / "out" / "table.md").read_text().splitlines()[2:4]
    assert all(row.count("**") == 6 for row in rows)


def test_notes_state_sample_size_split_and_judge(reports, tmp_path):
    compare([reports["b1"], reports["b2"]], tmp_path / "out")
    text = (tmp_path / "out" / "table.md").read_text()
    assert "100 questions" in text and "d" * 8 in text and "Qwen/Qwen3.5-9B" in text
    assert "noise" in text


def test_compare_also_reads_plain_run_directories(tmp_path):
    runs = [make_run(tmp_path, n, 0.5, 0.4, 0.6) for n in ("b1", "b2")]
    result = compare(runs, tmp_path / "out")
    assert [r["name"] for r in result["rows"]] == ["b1", "b2"]


@pytest.mark.parametrize(
    "override, message",
    [
        ({"split": {"hash": "e" * 64, "part": "test"}}, "split"),
        ({"corpus": "f" * 64}, "corpus"),
        ({"judge": {**JUDGE, "quant": "fp8"}}, "judge"),
        ({"model": "other-model"}, "model"),
        ({"n": 99}, "questions"),
    ],
)
def test_compare_refuses_runs_that_are_not_comparable(tmp_path, override, message):
    good = make_run(tmp_path, "b1", 0.5, 0.4, 0.6)
    bad = make_run(tmp_path, "b2", 0.5, 0.4, 0.6, **override)
    packed = [pack(r, r.name, tmp_path / f"{r.name}.tar.gz") for r in (good, bad)]
    with pytest.raises(ValueError, match=message):
        compare(packed, tmp_path / "out")
    assert not (tmp_path / "out" / "table.md").exists()


def test_compare_needs_the_baseline_and_the_guard_fields(tmp_path):
    only_b2 = make_run(tmp_path, "b2", 0.5, 0.4, 0.6)
    with pytest.raises(ValueError, match="baseline"):
        compare([only_b2], tmp_path / "out")
    b1 = make_run(tmp_path, "b1", 0.5, 0.4, 0.6)
    config = json.loads((b1 / "config.json").read_text())
    del config["corpus"]
    (b1 / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="corpus"):
        compare([b1, only_b2], tmp_path / "out")


def test_missing_metric_shows_n_a(tmp_path):
    b1 = make_run(tmp_path, "b1", 0.5, None, 0.6)
    compare([b1], tmp_path / "out")
    assert "n/a" in (tmp_path / "out" / "table.md").read_text()


def test_cli_pack_and_compare(tmp_path):
    runs = [make_run(tmp_path, n, 0.5, 0.4, 0.6) for n in ("b1", "b2")]
    for run in runs:
        out = tmp_path / f"report-{run.name}.tar.gz"
        result = subprocess.run(
            [sys.executable, "-m", "vihermes", "pack", "--run", str(run), "--name", run.name,
             "--output", str(out)],
            capture_output=True, text=True,
        )  # fmt: skip
        assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "vihermes", "compare", str(tmp_path / "report-b1.tar.gz"),
         str(tmp_path / "report-b2.tar.gz"), "--output-dir", str(tmp_path / "cmp")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cmp" / "results.json").exists()
    bad = subprocess.run(
        [sys.executable, "-m", "vihermes", "compare", str(tmp_path / "report-b2.tar.gz"),
         "--output-dir", str(tmp_path / "cmp2")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert bad.returncode == 2 and "baseline" in bad.stderr
