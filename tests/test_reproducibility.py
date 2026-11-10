import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "dryrun" / "qa.jsonl"


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def run_b1(tmp, name, env):
    """The B1 pipeline with deterministic parts only: hash encoder, extractive answers."""
    d = tmp / name
    d.mkdir()
    steps = [
        ("extract-contexts", "--qa", FIXTURE, "--output", d / "c.csv"),
        ("ingest", "--input", d / "c.csv", "--db", d / "g.db"),
        ("split", "--qa", FIXTURE, "--output", d / "s.json", "--test-size", 10),
        ("index", "--db", d / "g.db", "--index", d / "d.npz", "--embedding", "hash", "--force"),
        (
            "evaluate", "--qa", FIXTURE, "--split", d / "s.json", "--part", "test", "--db", d / "g.db",
            "--index", d / "d.npz", "--embedding", "hash", "--dense-weight", 0.5, "-k", 5,
            "--ks", 5, "--answers", "--generator", "extractive", "--output", d / "out",
        ),
    ]  # fmt: skip
    for step in steps:
        done = cli(*step, env=env)
        assert done.returncode == 0, done.stderr
    rows = [
        json.loads(line)
        for line in (d / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row.pop("seconds")
    return rows, json.loads((d / "out" / "metrics.json").read_text())["overall"]


def test_a_fixed_hash_seed_makes_retrieval_repeatable(tmp_path):
    """BM25 sums scores over a set of query words; its order changes with the hash seed, which
    reorders passages whose scores differ only in the last float bits. Fixing the seed pins it."""
    a, metrics = run_b1(tmp_path, "a", {"PYTHONHASHSEED": "0"})
    b, _ = run_b1(tmp_path, "b", {"PYTHONHASHSEED": "0"})
    assert a == b
    assert metrics["recall@5"] == 1.0 and metrics["f1"] > 0
    first = a[0]
    assert set(first["retrieved_ids"]) >= set(first["gold_context_ids"])  # gold is retrieved


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_the_makefile_exports_a_fixed_hash_seed(tmp_path):
    extra = tmp_path / "show.mk"
    extra.write_text("show:\n\t@printenv PYTHONHASHSEED\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    result = subprocess.run(
        ["make", "-f", "Makefile", "-f", str(extra), "show"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"
