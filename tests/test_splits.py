import json
import random
import subprocess
import sys

import pytest

from vihermes.models import QAExample
from vihermes.splits import load_split, make_split, save_split


def synthetic(per_hop=150, pool=3000, seed=0):
    rng = random.Random(seed)
    examples = []
    for hop in range(1, 6):
        for i in range(per_hop):
            gold = [f"c{rng.randrange(pool)}" for _ in range(hop)]
            examples.append(
                QAExample(
                    f"h{hop}-{i}", f"question {hop}-{i}", "answer", list(dict.fromkeys(gold)), hop
                )
            )
    return examples


def parts(split):
    return {name: set(split[name]) for name in ("train", "dev", "test", "quarantined")}


def test_split_is_deterministic_and_seed_sensitive():
    examples = synthetic()
    a, b = make_split(examples, seed=7), make_split(examples, seed=7)
    assert a == b and a["hash"] == b["hash"]
    assert make_split(examples, seed=8)["hash"] != a["hash"]


def test_every_question_in_exactly_one_part():
    examples = synthetic()
    split = make_split(examples)
    p = parts(split)
    assert sum(len(v) for v in p.values()) == len(examples)
    assert set().union(*p.values()) == {e.id for e in examples}
    names = list(p)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not p[left] & p[right]


def test_no_test_gold_context_leaks_into_train_or_dev():
    examples = synthetic()
    by_id = {e.id: e for e in examples}
    split = make_split(examples)
    test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}
    for name in ("train", "dev"):
        used = {c for i in split[name] for c in by_id[i].gold_context_ids}
        assert not used & test_gold, name
    assert split["stats"]["leaked_contexts"] == 0


def test_quarantined_questions_touch_test_contexts():
    examples = synthetic(pool=400)
    by_id = {e.id: e for e in examples}
    split = make_split(examples)
    test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}
    assert split["quarantined"]
    for qid in split["quarantined"]:
        assert set(by_id[qid].gold_context_ids) & test_gold


def test_test_is_stratified_and_report_subset_has_20_per_hop():
    examples = synthetic()
    by_id = {e.id: e for e in examples}
    split = make_split(examples)
    test_hops = [by_id[i].hop_level for i in split["test"]]
    assert all(test_hops.count(h) == 30 for h in range(1, 6))
    report = split["test_100"]
    assert set(report) <= set(split["test"])
    assert [by_id[i].hop_level for i in report].count(3) == 20
    assert len(report) == 100


def test_report_subset_uses_whole_hop_when_test_is_small():
    examples = synthetic(per_hop=40)
    split = make_split(examples)
    assert set(split["test_100"]) == set(split["test"])


def test_dev_is_stratified_and_not_empty():
    split = make_split(synthetic())
    assert split["stats"]["by_hop"]["dev"]["3"] > 0
    assert split["dev"] and split["train"]


def test_invalid_ratios_rejected():
    with pytest.raises(ValueError):
        make_split(synthetic(), test=0.0)
    with pytest.raises(ValueError):
        make_split(synthetic(), test=0.6, dev=0.5)


def test_save_and_load_roundtrip(tmp_path):
    split = make_split(synthetic())
    path = tmp_path / "split.json"
    save_split(split, path)
    assert load_split(path) == split


def test_load_rejects_tampered_file(tmp_path):
    split = make_split(synthetic())
    path = tmp_path / "split.json"
    save_split(split, path)
    data = json.loads(path.read_text())
    data["test"] = data["test"][:-1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_split(path)


def test_cli_split_command(tmp_path):
    qa = tmp_path / "qa.jsonl"
    rows = [
        {
            "id": e.id,
            "question": e.question,
            "answer": e.answer,
            "hop_level": e.hop_level,
            "used_context_ids": e.gold_context_ids,
        }
        for e in synthetic(per_hop=60)
    ]
    qa.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "split.json"
    result = subprocess.run(
        [sys.executable, "-m", "vihermes", "split", "--qa", str(qa), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    assert stats["records"] == 300 and "quarantined_fraction" in stats
    assert load_split(out)["hash"] == stats["hash"]
