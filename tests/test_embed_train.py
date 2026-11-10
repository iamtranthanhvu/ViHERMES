import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vihermes.training.embed import validation_recall_at_k

FIXTURE = Path(__file__).parent / "fixtures" / "dryrun" / "qa.jsonl"


def test_validation_recall_at_k_scores_the_gold_passages_in_the_top_k():
    # query vectors point at their own passage; passage "c" is the decoy for both queries
    corpus = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0.6, 0.6, 0]}
    queries = {"q1": [1, 0, 0], "q2": [0, 1, 0]}

    def encode(texts):
        table = {
            **{f"text-{k}": v for k, v in corpus.items()},
            **{f"ask-{k}": v for k, v in queries.items()},
        }
        return np.array([table[t] for t in texts], dtype=np.float32)

    validation = [("ask-q1", ["a"]), ("ask-q2", ["b", "c"])]
    texts = {k: f"text-{k}" for k in corpus}
    assert validation_recall_at_k(encode, validation, texts, k=1) == pytest.approx((1.0 + 0.5) / 2)
    assert validation_recall_at_k(encode, validation, texts, k=3) == 1.0


def test_validation_recall_ignores_questions_without_gold_passages():
    def encode(texts):
        return np.eye(len(texts), dtype=np.float32)

    assert validation_recall_at_k(encode, [("q", [])], {"a": "x"}, k=1) is None


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


@pytest.fixture
def prepared(tmp_path):
    contexts, db, split, data = (
        tmp_path / "contexts.csv", tmp_path / "g.db", tmp_path / "split.json", tmp_path / "data.jsonl",
    )  # fmt: skip
    assert run_cli("extract-contexts", "--qa", FIXTURE, "--output", contexts).returncode == 0
    assert run_cli("ingest", "--input", contexts, "--db", db).returncode == 0
    assert run_cli("split", "--qa", FIXTURE, "--output", split, "--test-size", 10).returncode == 0
    made = run_cli(
        "make-embed-data", "--qa", FIXTURE, "--split", split, "--db", db, "--dense-weight", 0,
        "--output", data,
    )  # fmt: skip
    assert made.returncode == 0, made.stderr
    return db, split, data, tmp_path


def test_cli_finetune_embed_smoke_with_a_tiny_model(prepared, tiny_st_dir):
    pytest.importorskip("datasets")
    db, split, data, tmp = prepared
    out = tmp / "embed-ft"
    result = run_cli(
        "finetune-embed", "--data", data, "--qa", FIXTURE, "--split", split, "--db", db,
        "--base", tiny_st_dir, "--output", out, "--max-steps", 2, "--batch-size", 4,
        "--max-seq-length", 32, "--epochs", 1,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    meta = json.loads((out / "train_meta.json").read_text())
    split_hash = json.loads(split.read_text())["hash"]
    assert meta["split_hash"] == split_hash and meta["seed"] == 42
    assert meta["steps"] == 2 and meta["pairs"] > 0
    assert (
        meta["validation_recall_at_5"]["base"] is not None
        and meta["validation_recall_at_5"]["best"] is not None
    )
    assert meta["seconds"] >= 0 and "packages" in meta
    assert (out / "model" / "modules.json").exists()
    # Load it in a fresh process: torch and scikit-learn (imported by other tests) can clash over
    # OpenMP when they share one process on macOS.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from sentence_transformers import SentenceTransformer as S;"
            "print(S(sys.argv[1], device='cpu').encode(['hồ sơ']).shape[0])",
            str(out / "model"),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr[-1500:]
    assert probe.stdout.strip().splitlines()[-1] == "1"


def test_training_refuses_rows_that_use_a_forbidden_passage(prepared, tiny_st_dir, tmp_path):
    pytest.importorskip("datasets")
    from vihermes.training.embed import train_embedding

    _, _, data, _ = prepared
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    forbidden = {rows[0]["positive_id"]}
    corpus = {r["positive_id"]: r["positive"] for r in rows}
    with pytest.raises(ValueError, match="forbidden"):
        train_embedding(
            rows, [("q", [rows[1]["positive_id"]])], corpus, tmp_path / "out",
            base=str(tiny_st_dir), max_steps=1, batch_size=2, max_seq_length=32,
            forbidden=forbidden,
        )  # fmt: skip
    assert not (tmp_path / "out" / "model").exists()
