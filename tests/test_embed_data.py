import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from vihermes.graph import SQLiteGraph
from vihermes.models import Node, QAExample
from vihermes.retrieval import HybridRetriever, units
from vihermes.splits import make_split
from vihermes.training.embed_data import build_embed_data

FIXTURE = Path(__file__).parent / "fixtures" / "dryrun" / "qa.jsonl"


@pytest.fixture
def world():
    rng = random.Random(1)
    with SQLiteGraph() as graph:
        for i in range(150):
            graph.upsert_node(
                Node(f"c{i}", f"d{i}", title="t", text=f"hồ sơ loại a{i} giấy tờ b{i} cơ quan c{i}")
            )
        examples = []
        for n in range(60):
            hop = n % 3 + 1
            gold = [f"c{i}" for i in rng.sample(range(150), hop)]
            question = " ".join(f"hồ sơ loại a{g[1:]}" for g in gold)
            examples.append(QAExample(f"q{n}", question, "answer", gold, hop))
        split = make_split(examples, test_size=9, validation=0.2)
        retriever = HybridRetriever(graph, weight=0)
        texts = {n.id: n.text for n in units(graph)}
        yield examples, split, retriever, texts


def by_id(examples):
    return {e.id: e for e in examples}


def test_positives_come_only_from_train_questions_and_never_from_test_passages(world):
    examples, split, retriever, texts = world
    rows, stats = build_embed_data(examples, split, retriever, texts)
    q = by_id(examples)
    test_gold = {c for i in split["test"] for c in q[i].gold_context_ids}
    assert rows and {r["qid"] for r in rows} <= set(split["train"])
    for row in rows:
        assert row["positive_id"] in q[row["qid"]].gold_context_ids
        assert row["positive_id"] not in test_gold
        assert row["positive"] == texts[row["positive_id"]]
    assert stats["pairs"] == len(rows)


def test_one_row_per_question_passage_pair(world):
    examples, split, retriever, texts = world
    rows, _ = build_embed_data(examples, split, retriever, texts)
    q = by_id(examples)
    expected = sum(len(q[i].gold_context_ids) for i in split["train"])
    assert len(rows) == expected
    assert len({(r["qid"], r["positive_id"]) for r in rows}) == len(rows)


def test_negatives_exclude_the_questions_gold_and_all_test_passages(world):
    examples, split, retriever, texts = world
    rows, _ = build_embed_data(examples, split, retriever, texts, negatives=3, pool=30)
    q = by_id(examples)
    test_gold = {c for i in split["test"] for c in q[i].gold_context_ids}
    for row in rows:
        ids = row["negative_ids"]
        assert 0 < len(ids) <= 3 and len(set(ids)) == len(ids)
        assert not set(ids) & set(q[row["qid"]].gold_context_ids)
        assert not set(ids) & test_gold
        assert row["negatives"] == [texts[i] for i in ids]


def test_rows_are_reproducible_and_seed_dependent(world):
    examples, split, retriever, texts = world
    a, _ = build_embed_data(examples, split, retriever, texts, seed=1)
    b, _ = build_embed_data(examples, split, retriever, texts, seed=1)
    c, _ = build_embed_data(examples, split, retriever, texts, seed=2)
    assert a == b and a != c


def test_split_referring_to_unknown_questions_is_rejected(world):
    examples, split, retriever, texts = world
    with pytest.raises(ValueError, match="absent"):
        build_embed_data(examples[:-5], split, retriever, texts)


def test_gold_missing_from_the_corpus_is_skipped_and_counted(world):
    examples, split, retriever, texts = world
    broken = dict(texts)
    victim = by_id(examples)[split["train"][0]].gold_context_ids[0]
    del broken[victim]
    rows, stats = build_embed_data(examples, split, retriever, broken)
    assert all(r["positive_id"] != victim for r in rows)
    assert stats["skipped_missing_gold"] >= 1


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


def test_cli_make_embed_data_on_the_fixture_without_leaks(tmp_path):
    contexts, db, split_file, out = (
        tmp_path / "contexts.csv", tmp_path / "g.db", tmp_path / "split.json", tmp_path / "data.jsonl",
    )  # fmt: skip
    assert run_cli("extract-contexts", "--qa", FIXTURE, "--output", contexts).returncode == 0
    assert run_cli("ingest", "--input", contexts, "--db", db).returncode == 0
    assert run_cli(
        "split", "--qa", FIXTURE, "--output", split_file, "--test-size", 10
    ).returncode == 0  # fmt: skip
    result = run_cli(
        "make-embed-data", "--qa", FIXTURE, "--split", split_file, "--db", db,
        "--dense-weight", 0, "--output", out,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["pairs"] == len(rows) > 0
    split = json.loads(split_file.read_text())
    from vihermes.data import load_qa

    q = {e.id: e for e in load_qa(FIXTURE)}
    test_gold = {c for i in split["test"] for c in q[i].gold_context_ids}
    assert not {r["positive_id"] for r in rows} & test_gold
    assert not {n for r in rows for n in r["negative_ids"]} & test_gold
