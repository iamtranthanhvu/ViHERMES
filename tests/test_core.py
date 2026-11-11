import json

import numpy as np
import pytest

from vihermes.adapters.models import HashEmbedder
from vihermes.agents import Auditor, Conductor, Interpreter, Pathfinder
from vihermes.construction import sample_tuples, validate_generated
from vihermes.data import extract_contexts, load_qa
from vihermes.evaluation import evaluate, recall_at_k, token_f1
from vihermes.graph import SQLiteGraph
from vihermes.ingestion import ingest, parse_document
from vihermes.models import Claim, Edge, Evidence, Node, Retrieval
from vihermes.retrieval import GraphPropagator, HybridRetriever, NumpyDenseIndex, units


@pytest.fixture
def graph():
    with SQLiteGraph() as g:
        yield g


def put(graph, *nodes):
    for n in nodes:
        graph.upsert_node(n)


def node(key, text="hồ sơ y tế", **kwargs):
    return Node(key, key.split("::")[0], title=key, text=text, **kwargs)


def test_parse_hierarchy_and_direction():
    ns, es = parse_document(
        "02/2025/TT-BYT",
        "Thông tư mới",
        "Điều 1. Sửa đổi khoản 2 Điều 3 của Thông tư số 01/2020/TT-BYT.\n1. Hồ sơ điện tử.",
    )
    assert {n.id for n in ns} == {
        "02/2025/TT-BYT",
        "02/2025/TT-BYT::article:1",
        "02/2025/TT-BYT::article:1::clause:1",
    }
    assert Edge("01/2020/TT-BYT::article:3::clause:2", "02/2025/TT-BYT::article:1", "AMENDS") in es
    assert len([e for e in es if e.relation == "HAS_CLAUSE"]) == 1
    assert all("status" not in n.metadata for n in ns)


def test_placeholder_resolved_without_losing_edges(graph):
    graph.upsert_edge(Edge("old", "new", "REPLACES"))
    assert graph.get("old").placeholder
    graph.upsert_node(node("old"))
    graph.upsert_node(Node("old", "old", placeholder=True))
    assert not graph.get("old").placeholder
    assert graph.edges("old")[0].target == "new"


def test_typed_propagation_and_one_hop_references(graph):
    put(graph, *(node(k) for k in ["old", "mid", "new", "ref", "drift"]))
    for e in [
        Edge("old", "mid", "AMENDS"),
        Edge("mid", "new", "REPLACES"),
        Edge("new", "ref", "REFERS_TO"),
        Edge("ref", "drift", "REFERS_TO"),
    ]:
        graph.upsert_edge(e)
    result = GraphPropagator(graph).expand([Evidence(graph.get("old"), 1)])
    assert result.evidence[0].node.id == "new"
    assert result.evidence[0].terminal
    assert {e.node.id for e in result.evidence} == {"old", "mid", "new", "ref"}


def test_cycles_and_limits(graph):
    put(graph, *(node(k) for k in ["a", "b", "c"]))
    for e in [Edge("a", "b", "AMENDS"), Edge("b", "a", "AMENDS"), Edge("b", "c", "REFERS_TO")]:
        graph.upsert_edge(e)
    result = GraphPropagator(graph, max_nodes=2).expand([Evidence(graph.get("a"), 1)])
    assert len(result.evidence) <= 2
    assert any("cycle" in w for w in result.warnings)
    assert not any(e.terminal for e in result.evidence if e.node.id in {"a", "b"})
    result = GraphPropagator(graph, max_depth=1).expand([Evidence(graph.get("a"), 1)])
    assert any("depth" in w for w in result.warnings)


def test_reference_evidence_includes_document_provenance_without_reference_drift(graph):
    put(graph, *(node(k) for k in ["seed", "reference", "drift"]))
    document = Node(
        "source-document",
        "source-document",
        kind="Document",
        title="Referenced regulation",
        metadata={"issuing_authority": "Ministry of Health"},
    )
    put(graph, document)
    for edge in [
        Edge("seed", "reference", "REFERS_TO"),
        Edge("reference", "drift", "REFERS_TO"),
        Edge(document.id, "reference", "HAS_CLAUSE"),
    ]:
        graph.upsert_edge(edge)
    result = GraphPropagator(graph).expand([Evidence(graph.get("seed"), 1)])
    evidence = {e.node.id: e for e in result.evidence}
    assert document.id in evidence
    assert evidence[document.id].node.metadata == document.metadata
    assert evidence[document.id].origin == "provenance"
    assert "drift" not in evidence


def test_dense_sparse_weight_and_cache(graph, tmp_path):
    put(graph, node("a", "hồ sơ bệnh án"), node("b", "học phí đại học"))

    class Dense:
        def scores(self, query, ids):
            return np.array([0.0, 1.0])

    assert HybridRetriever(graph, Dense(), 0).seeds("bệnh án", 1)[0].node.id == "a"
    assert HybridRetriever(graph, Dense(), 1).seeds("bệnh án", 1)[0].node.id == "b"
    cache = tmp_path / "index.npz"
    NumpyDenseIndex(units(graph), HashEmbedder(), cache)
    put(graph, node("b", "changed"))
    with pytest.raises(ValueError, match="rebuild"):
        NumpyDenseIndex(units(graph), HashEmbedder(), cache)


def test_auditor_rejects_unknown_quotes_and_synthesis(graph):
    n = node("a", "Hồ sơ gồm bản điện tử.")
    retrieved = Retrieval([Evidence(n, 1)])
    auditor = Auditor()
    assert auditor.audit("hồ sơ?", [Claim("x", ["fake"], {"fake": "x"})], retrieved)
    assert auditor.audit("hồ sơ?", [Claim("Miễn phí.", ["a"], {"a": n.text})], retrieved)
    assert not auditor.audit("hồ sơ?", [Claim(n.text, ["a"], {"a": n.text})], retrieved)


def test_auditor_checks_dates():
    n = node("a", "Hồ sơ điện tử.", metadata={"effective_date": "2999-01-01"})
    assert "source_date_conflict:a" in Auditor().audit(
        "?", [Claim(n.text, ["a"], {"a": n.text})], Retrieval([Evidence(n, 1)])
    )


def test_semantic_verifier_can_reject_exact_quotes():
    class LLM:
        def json(self, system, payload):
            return {"supported": False, "reasons": ["omitted exception"]}

    n = node("a")
    assert "semantic_grounding_failed" in Auditor(LLM()).audit(
        "?", [Claim("x", ["a"], {"a": n.text})], Retrieval([Evidence(n, 1)])
    )


def test_truncated_verifier_reply_rejects_the_answer_instead_of_crashing():
    import json

    class LLM:
        def json(self, system, payload):
            return json.loads('{"supported": false, "reasons": ["a long rambling reason')

    n = node("a")
    errors = Auditor(LLM()).audit(
        "?", [Claim("x", ["a"], {"a": n.text})], Retrieval([Evidence(n, 1)])
    )
    assert "semantic_grounding_failed" in errors
    assert "verifier_output_not_valid_json" in errors


def test_conductor_abstention_and_grounded_answer(graph):
    put(graph, node("a", "Hồ sơ gồm bản điện tử."))
    pathfinder = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
    engine = Conductor(Interpreter(), pathfinder, Auditor())
    assert engine.ask("xin chào").abstained
    assert engine.ask("xyzzy").abstained
    answer = engine.ask("hồ sơ")
    assert not answer.abstained and answer.sources[0]["id"] == "a"


def test_data_duplicate_ids_preserved_and_contexts_extracted(tmp_path):
    p = tmp_path / "qa.jsonl"
    row = {
        "id": "q1",
        "question": "Q",
        "answer": "A",
        "hop_level": 1,
        "used_context_ids": ["c"],
        "context_block": "[CONTEXT 1]\nID: c\nTitle: Title\nBody:\nText",
    }
    p.write_text("\n".join(json.dumps(row) for _ in range(2)))
    qa = load_qa(p)
    assert len({r.id for r in qa}) == 2
    assert all(r.metadata["original_id"] == "q1" for r in qa)
    assert extract_contexts(p, tmp_path / "contexts.csv") == 1


def test_metrics_and_denominators(graph, tmp_path):
    assert token_f1("a a b", "a b b") == pytest.approx(2 / 3)
    assert recall_at_k(["a", "a"], ["a", "b"], 2) == 0.5
    assert recall_at_k([], [], 5) is None
    put(graph, node("a"))
    from vihermes.models import QAExample

    examples = [QAExample("q", "y tế", "", ["a"], 1)]
    pf = Pathfinder(HybridRetriever(graph, weight=0), GraphPropagator(graph))
    report = evaluate(examples, pf, tmp_path)
    assert report["overall"]["recall@5"] == 1
    assert report["by_hop"]["1"]["n"] == 1


def test_tuple_constraints_and_validation():
    nodes = [node(str(i)) for i in range(10)]
    tuples = sample_tuples(nodes, 3, 4, overlap=0.34)
    for a in tuples:
        assert len({n.title for n in a}) == 3
        for b in tuples:
            if a is not b:
                assert len({n.id for n in a} & {n.id for n in b}) / 3 <= 0.34
    result = {
        "question": "Q",
        "answer": "A",
        "reasoning": "R",
        "hop_level": 3,
        "used_context_ids": [n.id for n in tuples[0]],
        "evidence": {n.id: n.text for n in tuples[0]},
    }
    assert validate_generated(result, tuples[0])
    result["evidence"][tuples[0][0].id] = "fabricated"
    with pytest.raises(ValueError):
        validate_generated(result, tuples[0])


def test_ingest_validates_before_writing(graph, tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps({"id": "a", "title": "A", "text": "Text"})
        + "\n"
        + json.dumps({"id": "b", "title": "B", "text": ""})
    )
    with pytest.raises(ValueError):
        ingest(path, graph)
    assert not graph.nodes()


def test_shared_validity_target_is_not_a_cycle(graph):
    put(graph, *(node(key) for key in ["a", "b", "current"]))
    graph.upsert_edge(Edge("a", "current", "AMENDS"))
    graph.upsert_edge(Edge("b", "current", "SUPPLEMENTS"))
    result = GraphPropagator(graph).expand(
        [
            Evidence(graph.get("current"), 1),
            Evidence(graph.get("a"), 0.9),
            Evidence(graph.get("b"), 0.8),
        ]
    )
    assert not any("cycle" in warning for warning in result.warnings)
    assert result.evidence[0].node.id == "current"
    assert result.evidence[0].terminal


def test_context_ingest_preserves_document_metadata(graph, tmp_path):
    import csv

    document = Node(
        "01/2020/TT-BYT",
        "01/2020/TT-BYT",
        "Document",
        "Thông tư 01/2020/TT-BYT",
        "Full original document",
        {"effective_date": "2020-07-01"},
    )
    graph.upsert_node(document)
    path = tmp_path / "contexts.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "title", "text"])
        writer.writeheader()
        writer.writerow({"context_id": "c1", "title": document.title, "text": "Hồ sơ."})
    first = ingest(path, graph)
    second = ingest(path, graph)
    assert first == second
    assert graph.get(document.id) == document
    assert len(graph.edges(document.id)) == 1
