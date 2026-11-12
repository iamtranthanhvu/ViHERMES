import pytest

from vihermes.chat.service import Service, classify, map_reason
from vihermes.models import Answer, Claim, Evidence, Node, Retrieval


def clause(key, title, text):
    return Node(key, "doc", "Clause", title=title, text=text)


def evidence(*nodes):
    return Retrieval([Evidence(n, 1.0) for n in nodes])


class FakeConductor:
    def __init__(self, answer, retrieval):
        self.result = (answer, retrieval)

    def run(self, question):
        return self.result


def service(answer, retrieval):
    return Service(FakeConductor(answer, retrieval), {"model": "m", "lora": "l", "corpus_units": 2})


def answered():
    node = clause("c1", "Thông tư 01", "Điều 5. Lãnh đạo gồm Giám đốc và các Phó Giám đốc.")
    claim = Claim("Gồm Giám đốc.", ["c1"], {"c1": "Lãnh đạo gồm Giám đốc"})
    sources = [{"id": "c1", "title": "Thông tư 01", "text": node.text}]
    return Answer("q", "Gồm Giám đốc. [c1]", [claim], sources), evidence(node)


def refused(reasons):
    node = clause("c2", "Quyết định giá", "Phụ lục III giá dịch vụ")
    return Answer("q", "Chưa đủ", [], [], True, reasons), evidence(node)


def test_answered_has_titled_sources_with_verbatim_quote():
    out = service(*answered()).ask("q")
    assert out["status"] == "answered"
    assert out["answer"] == "Gồm Giám đốc."  # no raw "[c1]" node ids
    assert out["sources"] == [
        {"id": "c1", "title": "Thông tư 01", "quote": "Lãnh đạo gồm Giám đốc"}
    ]
    assert "reason" not in out and "trace" not in out
    assert out["seconds"] >= 0


def test_refused_lists_retrieved_documents_without_quotes():
    out = service(*refused(["semantic_grounding_failed"])).ask("q")
    assert out["status"] == "refused"
    assert out["reason"] == "semantic"
    assert out["sources"] == [{"id": "c2", "title": "Quyết định giá"}]


def test_clarification_status_and_no_sources():
    answer = Answer("hi", "Bạn muốn tìm hiểu?", [], [], True, ["clarification_requested"])
    out = service(answer, Retrieval()).ask("hi")
    assert out["status"] == "clarification"
    assert out["sources"] == []
    assert out["answer"] == "Bạn muốn tìm hiểu?"


@pytest.mark.parametrize(
    "reasons,expected",
    [
        (["insufficient_evidence"], "insufficient"),
        (["unknown_citation:x"], "insufficient"),
        (["unsupported_quote:c1"], "quote"),
        (["semantic_grounding_failed", "unsupported_quote:c1"], "semantic"),
        (["superseded_source:c1"], "other"),
        (["something_new"], "other"),
        ([], "other"),
    ],
)
def test_reason_mapping_never_leaks_raw_codes(reasons, expected):
    assert map_reason(reasons) == expected


def test_classify():
    assert classify(Answer("q", "", [], [], True, ["clarification_requested"])) == "clarification"
    assert classify(Answer("q", "", [], [], True, ["x"])) == "refused"
    assert classify(Answer("q", "a", [], [])) == "answered"


def test_refused_sources_are_capped_and_skip_empty_nodes():
    nodes = [clause(f"c{i}", f"T{i}", "text") for i in range(8)]
    nodes.append(Node("d", "d", "Document", title="Doc"))
    out = service(Answer("q", "", [], [], True, ["insufficient_evidence"]), evidence(*nodes)).ask(
        "q"
    )
    assert len(out["sources"]) == 5
    assert all(s["id"].startswith("c") for s in out["sources"])
