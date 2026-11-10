import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from vihermes.agents import GENERATE, Auditor, Pathfinder, serialize_evidence
from vihermes.data import load_qa, records
from vihermes.graph import SQLiteGraph
from vihermes.models import Claim, Node, Plan, QAExample
from vihermes.retrieval import GraphPropagator, HybridRetriever, units
from vihermes.splits import make_split
from vihermes.training.lora_data import best_quote, build_lora_rows, exact_span

FIXTURE = Path(__file__).parent / "fixtures" / "dryrun" / "qa.jsonl"
TEXT = (
    "Điều 5. Hồ sơ đăng ký gồm đơn đề nghị và bản sao giấy tờ. "
    "Thời hạn xử lý là năm ngày làm việc kể từ khi nhận đủ hồ sơ. "
    "Cơ quan tiếp nhận có trách nhiệm hướng dẫn người nộp hồ sơ."
)


def test_best_quote_is_an_exact_slice_aligned_with_the_reference():
    quote = best_quote(TEXT, "thời hạn xử lý hồ sơ là năm ngày", max_chars=80)
    assert quote and quote in TEXT and len(quote) <= 80
    assert "Thời hạn xử lý" in quote


def test_best_quote_needs_real_overlap_with_the_reference():
    assert best_quote(TEXT, "hoàn toàn khác biệt xyz", max_chars=80) is None
    assert best_quote("", "hồ sơ", max_chars=80) is None


def test_best_quote_handles_long_flattened_text_without_sentences():
    blob = " ".join(f"cột{i} giá{i}" for i in range(400)) + " tổng cộng 438 triệu đồng"
    quote = best_quote(blob, "tổng cộng 438 triệu đồng", max_chars=120)
    assert quote in blob and "438" in quote and len(quote) <= 120


def test_exact_span_only_accepts_verbatim_text():
    assert exact_span("Hồ sơ đăng ký gồm đơn", TEXT) == "Hồ sơ đăng ký gồm đơn"
    assert exact_span("Cung cấp thông tin về hồ sơ", TEXT) is None
    assert exact_span("   ", TEXT) is None


class SupportingModel:
    """Semantic verifier that always agrees, so only the deterministic Auditor checks matter."""

    def json(self, system, payload):
        return {"supported": True, "reasons": []}


@pytest.fixture
def world():
    with SQLiteGraph() as graph:
        examples, raw = [], []
        for i in range(160):
            graph.upsert_node(
                Node(
                    f"c{i}",
                    f"d{i}",
                    title=f"Quy định {i}",
                    text=f"Điều {i}. Hồ sơ loại a{i} gồm giấy tờ b{i} do cơ quan c{i} cấp. "
                    f"Thời hạn xử lý hồ sơ loại a{i} là d{i} ngày làm việc.",
                )
            )
        for n in range(90):
            hop = n % 3 + 1
            gold = [f"c{(n * 7 + 11 * k) % 160}" for k in range(hop)]
            tags = [g[1:] for g in gold]
            examples.append(
                QAExample(
                    f"q{n}",
                    "Hồ sơ "
                    + " và hồ sơ ".join(f"loại a{t}" for t in tags)
                    + " gồm gì và thời hạn bao lâu?",
                    " ".join(
                        f"Hồ sơ loại a{t} gồm giấy tờ b{t} và thời hạn xử lý d{t} ngày."
                        for t in tags
                    ),
                    gold,
                    hop,
                    {"original_id": f"q{n}", "row": n},
                )
            )
            raw.append(
                {
                    "evidence": {
                        f"context_{k + 1}": f"Cung cấp thông tin về hồ sơ loại a{t}."
                        for k, t in enumerate(tags)
                    }
                }
            )
        split = make_split(examples, test_size=9, validation=0.2)
        pathfinder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph), k=5)
        nodes = {n.id: n for n in units(graph)}
        yield examples, raw, split, pathfinder, nodes


def build(world, part="train", **options):
    examples, raw, split, pathfinder, nodes = world
    return build_lora_rows(examples, raw, split, part, pathfinder, nodes, **options)


def payload_of(row):
    return json.loads(row["messages"][1]["content"])


def target_of(row):
    return json.loads(row["messages"][2]["content"])


def test_rows_use_exactly_the_inference_prompt_and_payload(world):
    rows, _ = build(world)
    assert rows
    examples, _, _, pathfinder, _ = world
    by_id = {e.id: e for e in examples}
    for row in rows[:10]:
        assert row["messages"][0] == {"role": "system", "content": GENERATE}
        question = payload_of(row)["question"]
        assert question == by_id[row["meta"]["qid"]].question
        result = pathfinder.retrieve(Plan(question))
        live = {
            "question": question,
            "evidence": serialize_evidence(result),
            "edges": [asdict(e) for e in result.edges],
        }
        shown = {e["id"] for e in payload_of(row)["evidence"]}
        assert shown <= {e["id"] for e in live["evidence"]}  # only test passages may be missing


def test_answer_samples_pass_the_auditors_deterministic_checks(world):
    rows, _ = build(world)
    answers = [r for r in rows if r["meta"]["kind"] == "answer"]
    assert answers
    examples, _, _, pathfinder, _ = world
    by_id = {e.id: e for e in examples}
    auditor = Auditor(SupportingModel())
    for row in answers:
        target = target_of(row)
        assert target["abstain"] is False and len(target["claims"]) == 1
        claim = target["claims"][0]
        evidence = payload_of(row)["evidence"]
        nodes = {e["id"]: e for e in evidence}
        assert (
            set(claim["citations"])
            == set(claim["quotes"])
            == set(by_id[row["meta"]["qid"]].gold_context_ids)
        )
        for key, quote in claim["quotes"].items():
            assert quote in nodes[key]["text"]
        # rebuild a Retrieval holding exactly the evidence shown to the model
        from vihermes.models import Evidence, Retrieval

        shown = Retrieval(
            [Evidence(Node(e["id"], e["document_id"], text=e["text"]), 1.0) for e in evidence]
        )
        assert auditor.audit(by_id[row["meta"]["qid"]].question, [Claim(**claim)], shown) == []


def test_abstain_samples_have_an_empty_answer(world):
    rows, stats = build(world, abstain_fraction=0.3)
    abstains = [r for r in rows if r["meta"]["kind"].startswith("abstain")]
    assert abstains and stats["abstain"] == len(abstains)
    for row in abstains:
        assert target_of(row) == {"claims": [], "abstain": True}
        examples = {e.id: e for e in world[0]}
        shown = {e["id"] for e in payload_of(row)["evidence"]}
        assert (
            not set(examples[row["meta"]["qid"]].gold_context_ids) <= shown
        )  # evidence is insufficient


def test_abstain_fraction_is_respected(world):
    rows, stats = build(world, abstain_fraction=0.2)
    fraction = stats["abstain"] / len(rows)
    assert 0.1 <= fraction <= 0.3
    none, stats0 = build(world, abstain_fraction=0.0)
    assert not [r for r in none if r["meta"]["kind"] == "abstain_removed"]


def test_no_test_question_or_test_passage_reaches_the_training_data(world):
    examples, _, split, _, _ = world
    by_id = {e.id: e for e in examples}
    test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}
    for part in ("train", "validation"):
        rows, _ = build(world, part)
        assert rows and {r["meta"]["qid"] for r in rows} <= set(split[part])
        for row in rows:
            assert not {e["id"] for e in payload_of(row)["evidence"]} & test_gold
            assert (
                not set(target_of(row)["claims"] and target_of(row)["claims"][0]["citations"])
                & test_gold
            )


def test_overlong_prompts_are_dropped_and_counted(world):
    rows, stats = build(world, max_prompt_chars=100)
    assert rows == [] and stats["dropped_too_long"] > 0


def test_questions_without_an_alignable_quote_are_dropped(world):
    examples, raw, split, pathfinder, nodes = world
    garbled = [
        QAExample(e.id, e.question, "zzz qqq www", e.gold_context_ids, e.hop_level, e.metadata)
        for e in examples
    ]
    blank = [{"evidence": {}} for _ in raw]  # no summaries either, so nothing can align
    rows, stats = build_lora_rows(
        garbled, blank, split, "train", pathfinder, nodes, abstain_fraction=0.0, min_overlap=3
    )
    assert [r for r in rows if r["meta"]["kind"] == "answer"] == []
    assert stats["dropped_no_quote"] > 0


def test_rows_are_reproducible(world):
    a, _ = build(world, seed=1)
    b, _ = build(world, seed=1)
    c, _ = build(world, seed=2)
    assert a == b and a != c


def test_the_exact_evidence_span_is_preferred_when_it_is_verbatim(world):
    examples, raw, split, pathfinder, nodes = world
    verbatim = []
    for e in examples:
        verbatim.append({"evidence": {"context_1": nodes[e.gold_context_ids[0]].text[:40]}})
    rows, stats = build_lora_rows(
        examples, verbatim, split, "train", pathfinder, nodes, abstain_fraction=0.0
    )
    assert stats["exact_evidence_quotes"] > 0
    row = next(r for r in rows if r["meta"]["kind"] == "answer")
    qid = row["meta"]["qid"]
    gold0 = next(e for e in examples if e.id == qid).gold_context_ids[0]
    assert target_of(row)["claims"][0]["quotes"][gold0] == nodes[gold0].text[:40].strip()


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vihermes", *map(str, args)], capture_output=True, text=True
    )


def test_cli_make_lora_data_on_the_fixture(tmp_path):
    contexts, db, split_file, out = (
        tmp_path / "c.csv",
        tmp_path / "g.db",
        tmp_path / "s.json",
        tmp_path / "lora",
    )
    assert run_cli("extract-contexts", "--qa", FIXTURE, "--output", contexts).returncode == 0
    assert run_cli("ingest", "--input", contexts, "--db", db).returncode == 0
    assert (
        run_cli("split", "--qa", FIXTURE, "--output", split_file, "--test-size", 10).returncode == 0
    )
    result = run_cli(
        "make-lora-data", "--qa", FIXTURE, "--split", split_file, "--db", db, "--dense-weight", 0,
        "--output-dir", out,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    train = [
        json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train) == stats["train"]["rows"] > 0 and validation
    assert (
        json.loads((out / "data_stats.json").read_text())["split_hash"]
        == json.loads(split_file.read_text())["hash"]
    )
    split = json.loads(split_file.read_text())
    qa = {e.id: e for e in load_qa(FIXTURE)}
    test_gold = {c for i in split["test"] for c in qa[i].gold_context_ids}
    for row in train + validation:
        assert row["meta"]["qid"] not in split["test"]
        assert (
            not {e["id"] for e in json.loads(row["messages"][1]["content"])["evidence"]} & test_gold
        )
    assert records(FIXTURE)  # the raw reader the builder relies on


@pytest.fixture
def narrow(world):
    """The same data, but retrieval keeps only 2 passages, so hop-3 questions come back incomplete."""
    examples, raw, split, _, nodes = world
    graph_nodes = list(nodes.values())
    with SQLiteGraph() as graph:
        for n in graph_nodes:
            graph.upsert_node(n)
        yield (
            examples,
            raw,
            split,
            Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph), k=2),
            nodes,
        )


def test_missing_gold_is_injected_for_answer_samples_without_growing_the_prompt(narrow):
    examples, raw, split, pathfinder, nodes = narrow
    by_id = {e.id: e for e in examples}
    rows, stats = build_lora_rows(
        examples, raw, split, "train", pathfinder, nodes, abstain_fraction=0.0
    )
    injected = [r for r in rows if r["meta"].get("injected")]
    assert injected and stats["answer_injected"] == len(injected)
    for row in rows:
        shown = [e["id"] for e in payload_of(row)["evidence"]]
        gold = by_id[row["meta"]["qid"]].gold_context_ids
        assert set(gold) <= set(shown)  # answers are only written when every gold passage is shown
        assert len(shown) <= max(2, len(gold))  # the prompt keeps the retrieval size
    assert {r["meta"]["qid"] for r in rows} >= {r["meta"]["qid"] for r in injected}


def test_injected_answers_still_pass_the_auditor(narrow):
    examples, raw, split, pathfinder, nodes = narrow
    by_id = {e.id: e for e in examples}
    rows, _ = build_lora_rows(
        examples, raw, split, "train", pathfinder, nodes, abstain_fraction=0.0
    )
    auditor = Auditor(SupportingModel())
    from vihermes.models import Evidence, Retrieval

    for row in [r for r in rows if r["meta"].get("injected")][:15]:
        evidence = payload_of(row)["evidence"]
        shown = Retrieval(
            [Evidence(Node(e["id"], e["document_id"], text=e["text"]), 1.0) for e in evidence]
        )
        claim = Claim(**target_of(row)["claims"][0])
        assert auditor.audit(by_id[row["meta"]["qid"]].question, [claim], shown) == []


def test_incomplete_retrieval_still_yields_a_natural_abstention_for_the_same_question(narrow):
    examples, raw, split, pathfinder, nodes = narrow
    rows, _ = build_lora_rows(
        examples, raw, split, "train", pathfinder, nodes, abstain_fraction=0.5
    )
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["meta"]["qid"], set()).add(r["meta"]["kind"])
    assert any({"answer", "abstain_natural"} <= kinds for kinds in by_kind.values())


def test_injection_can_be_turned_off(narrow):
    examples, raw, split, pathfinder, nodes = narrow
    rows, stats = build_lora_rows(
        examples, raw, split, "train", pathfinder, nodes, abstain_fraction=0.0, inject_gold=False
    )
    assert stats["answer_injected"] == 0 and not any(r["meta"].get("injected") for r in rows)
