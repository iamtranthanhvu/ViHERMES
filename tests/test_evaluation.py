import csv
import json
import subprocess
import sys

import pytest

from vihermes.agents import Pathfinder
from vihermes.data import load_qa
from vihermes.evaluation import evaluate, run_environment
from vihermes.graph import SQLiteGraph
from vihermes.models import Node, QAExample
from vihermes.retrieval import GraphPropagator, HybridRetriever
from vihermes.splits import make_split, save_split


class CountingFinder:
    """Wraps a Pathfinder and records which questions were actually retrieved."""

    def __init__(self, inner):
        self.inner, self.seen = inner, []

    def retrieve(self, plan):
        self.seen.append(plan.query)
        return self.inner.retrieve(plan)


@pytest.fixture
def setup():
    with SQLiteGraph() as graph:
        for i in range(3):
            graph.upsert_node(Node(f"c{i}", f"d{i}", title="t", text=f"hồ sơ số {i}"))
        finder = CountingFinder(
            Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
        )
        examples = [QAExample(f"q{i}", f"hồ sơ số {i}", "", [f"c{i}"], 1) for i in range(3)]
        yield finder, examples


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_resume_skips_finished_questions_and_matches_a_fresh_run(setup, tmp_path):
    finder, examples = setup
    evaluate(examples[:2], finder, tmp_path / "run", ks=(5,))
    finder.seen.clear()
    resumed = evaluate(examples, finder, tmp_path / "run", ks=(5,), resume=True)
    assert finder.seen == ["hồ sơ số 2"]
    rows = lines(tmp_path / "run" / "predictions.jsonl")
    assert [r["id"] for r in rows] == ["q0", "q1", "q2"]
    fresh = evaluate(examples, finder, tmp_path / "fresh", ks=(5,))
    assert resumed["overall"]["recall@5"] == fresh["overall"]["recall@5"] == 1
    assert resumed["overall"]["n"] == 3


def test_without_resume_the_run_starts_over(setup, tmp_path):
    finder, examples = setup
    evaluate(examples, finder, tmp_path, ks=(5,))
    finder.seen.clear()
    evaluate(examples, finder, tmp_path, ks=(5,))
    assert len(finder.seen) == 3
    assert len(lines(tmp_path / "predictions.jsonl")) == 3


def test_resume_survives_a_truncated_last_line(setup, tmp_path):
    finder, examples = setup
    evaluate(examples[:2], finder, tmp_path, ks=(5,))
    with (tmp_path / "predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "q2", "recall@5"')
    finder.seen.clear()
    report = evaluate(examples, finder, tmp_path, ks=(5,), resume=True)
    assert finder.seen == ["hồ sơ số 2"]
    assert report["overall"]["n"] == 3
    assert [r["id"] for r in lines(tmp_path / "predictions.jsonl")] == ["q0", "q1", "q2"]


def test_resume_redoes_rows_missing_a_requested_metric(setup, tmp_path):
    finder, examples = setup
    evaluate(examples, finder, tmp_path, ks=(1,))
    finder.seen.clear()
    report = evaluate(examples, finder, tmp_path, ks=(5,), resume=True)
    assert len(finder.seen) == 3
    assert report["overall"]["recall@5"] == 1
    assert len(lines(tmp_path / "predictions.jsonl")) == 3


def test_metadata_and_timing_are_written(setup, tmp_path):
    finder, examples = setup
    evaluate(
        examples, finder, tmp_path, ks=(5,), metadata={"split": {"hash": "abc", "part": "test"}}
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["metadata"]["split"]["hash"] == "abc"
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["examples"] == 3 and timing["evaluated_now"] == 3 and timing["resumed"] == 0
    assert timing["seconds_this_run"] >= 0


def test_run_environment_reports_versions_without_crashing():
    env = run_environment()
    assert env["python"].count(".") >= 1
    assert "numpy" in env["packages"]
    assert "gpu" in env and "git_commit" in env


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


@pytest.fixture
def corpus(tmp_path):
    contexts = tmp_path / "contexts.csv"
    with contexts.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "title", "text"])
        writer.writeheader()
        for i in range(12):
            writer.writerow({"context_id": f"c{i}", "title": "Quy định", "text": f"hồ sơ số {i}"})
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"q{i}",
                    "question": f"hồ sơ số {i}",
                    "answer": "",
                    "hop_level": 1,
                    "used_context_ids": [f"c{i}"],
                }
            )
            for i in range(12)
        ),
        encoding="utf-8",
    )
    db = tmp_path / "g.db"
    assert cli("ingest", "--input", contexts, "--db", db).returncode == 0
    split = make_split(load_qa(qa), test_size=5)
    save_split(split, tmp_path / "split.json")
    return qa, db, split, tmp_path


def evaluate_cli(corpus, *extra, out="out"):
    qa, db, _, tmp = corpus
    return cli(
        "evaluate", "--qa", qa, "--db", db, "--dense-weight", 0, "--ks", 5,
        "--output", tmp / out, *extra,
    )  # fmt: skip


def test_cli_evaluates_only_the_requested_split_part(corpus):
    _, _, split, tmp = corpus
    result = evaluate_cli(corpus, "--split", tmp / "split.json", "--part", "test")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["overall"]["n"] == len(split["test"]) == 5
    ids = {r["id"] for r in lines(tmp / "out" / "predictions.jsonl")}
    assert ids == set(split["test"])
    config = json.loads((tmp / "out" / "config.json").read_text())
    assert config["split"] == {"hash": split["hash"], "part": "test"}
    assert "packages" in config["environment"]


def test_cli_without_split_keeps_old_behaviour(corpus):
    result = evaluate_cli(corpus)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["overall"]["n"] == 12


def test_cli_rejects_a_tampered_split(corpus):
    _, _, _, tmp = corpus
    data = json.loads((tmp / "split.json").read_text())
    data["test"] = data["test"][:-1]
    (tmp / "split_bad.json").write_text(json.dumps(data))
    result = evaluate_cli(corpus, "--split", tmp / "split_bad.json")
    assert result.returncode != 0 and "hash" in result.stderr


def test_cli_resume_flag(corpus):
    _, _, _, tmp = corpus
    first = evaluate_cli(corpus, "--limit", 4)
    assert first.returncode == 0, first.stderr
    second = evaluate_cli(corpus, "--resume")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["overall"]["n"] == 12
    assert len(lines(tmp / "out" / "predictions.jsonl")) == 12
