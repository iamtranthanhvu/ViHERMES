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


# Top-5 passages (as sets) of the 10 test questions of the fixture, recorded on baseline/1-local
# with the deterministic B1 settings above. Every later branch must still retrieve exactly these
# when none of its new flags is used. Regenerate only if the retrieval code changes on purpose.
B1_TOP5 = [
    ["0006-fix-0042", "0066-fix-0462", "0124-fix-0868", "0207-fix-1449", "0305-fix-2135"],
    ["0006-fix-0042", "0031-fix-0217", "0066-fix-0462", "0244-fix-1708", "0305-fix-2135"],
    ["0048-fix-0336", "0244-fix-1708", "0278-fix-1946", "0301-fix-2107", "0316-fix-2212"],
    ["0050-fix-0350", "0244-fix-1708", "0278-fix-1946", "0301-fix-2107", "0373-fix-2611"],
    ["0041-fix-0287", "0046-fix-0322", "0229-fix-1603", "0244-fix-1708", "0278-fix-1946"],
    ["0104-fix-0728", "0244-fix-1708", "0280-fix-1960", "0301-fix-2107", "0308-fix-2156"],
    ["0035-fix-0245", "0045-fix-0315", "0102-fix-0714", "0243-fix-1701", "0347-fix-2429"],
    ["0059-fix-0413", "0099-fix-0693", "0249-fix-1743", "0300-fix-2100", "0313-fix-2191"],
    ["0011-fix-0077", "0018-fix-0126", "0059-fix-0413", "0189-fix-1323", "0310-fix-2170"],
    ["0102-fix-0714", "0133-fix-0931", "0183-fix-1281", "0200-fix-1400", "0374-fix-2618"],
]


def test_the_default_pipeline_still_retrieves_what_baseline_1_retrieved(tmp_path):
    rows, metrics = run_b1(tmp_path, "regression", {"PYTHONHASHSEED": "0"})
    assert metrics["recall@5"] == 1.0
    assert [sorted(r["retrieved_ids"]) for r in rows] == B1_TOP5
