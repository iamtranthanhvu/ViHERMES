import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "dryrun" / "qa.jsonl"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_tiny_lm.py"

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from vihermes.training.lora import collate, encode_example, train_lora  # noqa: E402


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_tiny_lm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_tiny_lm(tmp_path_factory.mktemp("tiny_lm"))


@pytest.fixture(scope="module")
def tokenizer(tiny_lm):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(tiny_lm)


def message_row(
    qid="q1", user='{"question": "a1", "evidence": []}', answer='{"claims": [], "abstain": true}'
):
    return {
        "messages": [
            {"role": "system", "content": "You are the Conductor."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"qid": qid, "kind": "abstain_natural", "part": "train"},
    }


def test_only_the_answer_carries_loss(tokenizer):
    row = message_row()
    encoded = encode_example(tokenizer, row["messages"], max_seq_length=512, no_thinking=True)
    ids, labels = encoded["input_ids"], encoded["labels"]
    assert len(ids) == len(labels)
    answer = tokenizer(
        row["messages"][2]["content"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    assert ids[-len(answer) :] == answer
    assert labels[-len(answer) :] == answer
    assert set(labels[: -len(answer)]) == {-100}
    assert answer[-1] == tokenizer.eos_token_id  # the model learns to stop


def test_the_prompt_is_built_exactly_as_at_inference(tokenizer):
    row = message_row()
    prompt_text = tokenizer.apply_chat_template(
        row["messages"][:2], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    expected = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = encode_example(tokenizer, row["messages"], 512, no_thinking=True)
    assert encoded["input_ids"][: len(expected)] == expected
    thinking = encode_example(tokenizer, row["messages"], 512, no_thinking=False)
    assert len(thinking["input_ids"]) < len(
        encoded["input_ids"]
    )  # the template adds a marker only when off


def test_examples_longer_than_the_limit_are_rejected(tokenizer):
    assert (
        encode_example(tokenizer, message_row()["messages"], max_seq_length=10, no_thinking=True)
        is None
    )


def test_collate_pads_with_ignore_labels():
    batch = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ]
    out = collate(batch, pad_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def rows(n, prefix):
    return [
        message_row(f"{prefix}{i}", user=f'{{"question": "q{i}", "evidence": []}}')
        for i in range(n)
    ]


def test_training_smoke_saves_an_adapter_and_run_metadata(tiny_lm, tmp_path):
    out = tmp_path / "lora"
    record = train_lora(
        rows(6, "t"), rows(3, "d"), out, base=str(tiny_lm), epochs=1, max_steps=2, batch_size=2,
        grad_accum=1, max_seq_length=256, rank=4, alpha=8, seed=7, meta={"split_hash": "abc"},
    )  # fmt: skip
    assert (out / "adapter" / "adapter_config.json").exists()
    assert list((out / "adapter").glob("adapter_model.*"))
    saved = json.loads((out / "train_meta.json").read_text())
    assert saved == json.loads(json.dumps(record))
    assert saved["seed"] == 7 and saved["split_hash"] == "abc" and saved["steps"] == 2
    assert saved["rows"] == {"train": 6, "dev": 3} and saved["dropped_too_long"] == {
        "train": 0,
        "dev": 0,
    }
    assert saved["dev_loss"]["best"] is not None and saved["trainable_params"] > 0
    assert saved["params"]["rank"] == 4 and "packages" in saved and saved["seconds"] >= 0


def test_the_saved_adapter_loads_onto_the_base_model(tiny_lm, tmp_path):
    out = tmp_path / "lora"
    train_lora(rows(4, "t"), rows(2, "d"), out, base=str(tiny_lm), max_steps=1, batch_size=2,
               grad_accum=1, max_seq_length=256, rank=4, alpha=8)  # fmt: skip
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, transformers, peft;"
            "m = transformers.AutoModelForCausalLM.from_pretrained(sys.argv[1]);"
            "p = peft.PeftModel.from_pretrained(m, sys.argv[2]);"
            "print(sum(1 for n, _ in p.named_parameters() if 'lora_' in n))",
            str(tiny_lm), str(out / "adapter"),
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert probe.returncode == 0, probe.stderr[-1500:]
    assert int(probe.stdout.strip().splitlines()[-1]) > 0


def test_rows_too_long_for_the_limit_are_dropped_and_counted(tiny_lm, tmp_path):
    long_row = message_row("long", user=json.dumps({"question": "x " * 400, "evidence": []}))
    record = train_lora(
        rows(4, "t") + [long_row], rows(2, "d"), tmp_path / "o", base=str(tiny_lm), max_steps=1,
        batch_size=2, grad_accum=1, max_seq_length=256, rank=4, alpha=8,
    )  # fmt: skip
    assert record["dropped_too_long"]["train"] == 1 and record["rows"]["train"] == 4


def test_training_fails_when_nothing_fits(tiny_lm, tmp_path):
    with pytest.raises(ValueError, match="fit"):
        train_lora([message_row(user=json.dumps({"question": "x " * 400, "evidence": []}))], rows(1, "d"), tmp_path / "o", base=str(tiny_lm),
                   max_seq_length=64)  # fmt: skip


def test_forbidden_questions_and_passages_are_refused(tiny_lm, tmp_path):
    leaky_passage = message_row(
        "p1", user=json.dumps({"question": "q", "evidence": [{"id": "secret-1", "text": "t"}]})
    )
    with pytest.raises(ValueError, match="forbidden question"):
        train_lora(rows(3, "t"), rows(1, "d"), tmp_path / "a", base=str(tiny_lm),
                   forbidden_qids={"t1"}, max_seq_length=256)  # fmt: skip
    with pytest.raises(ValueError, match="forbidden passage"):
        train_lora([leaky_passage], rows(1, "d"), tmp_path / "b", base=str(tiny_lm),
                   forbidden_passages={"secret-1"}, max_seq_length=256)  # fmt: skip
    assert not (tmp_path / "a" / "adapter").exists()


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


def test_cli_finetune_lora_on_the_fixture(tiny_lm, tmp_path):
    contexts, db, split, data, out = (
        tmp_path / "c.csv", tmp_path / "g.db", tmp_path / "s.json", tmp_path / "data", tmp_path / "lora",
    )  # fmt: skip
    assert run_cli("extract-contexts", "--qa", FIXTURE, "--output", contexts).returncode == 0
    assert run_cli("ingest", "--input", contexts, "--db", db).returncode == 0
    assert run_cli("split", "--qa", FIXTURE, "--output", split, "--test-size", 10).returncode == 0
    made = run_cli(
        "make-lora-data", "--qa", FIXTURE, "--split", split, "--db", db, "--dense-weight", 0,
        "--output-dir", data,
    )  # fmt: skip
    assert made.returncode == 0, made.stderr
    result = run_cli(
        "finetune-lora", "--data-dir", data, "--qa", FIXTURE, "--split", split, "--base", tiny_lm,
        "--output", out, "--max-steps", 2, "--batch-size", 1, "--grad-accum", 1,
        "--max-seq-length", 6000, "--rank", 4, "--alpha", 8, "--no-gradient-checkpointing",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    meta = json.loads((out / "train_meta.json").read_text())
    assert meta["split_hash"] == json.loads(split.read_text())["hash"]
    assert meta["steps"] == 2 and meta["rows"]["train"] > 0
    assert (out / "adapter" / "adapter_config.json").exists()
