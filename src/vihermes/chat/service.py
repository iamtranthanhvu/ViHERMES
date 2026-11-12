"""Turns a Conductor answer into the JSON the chat page shows."""

import time

MAX_REFUSED_SOURCES = 5
QUOTE_FALLBACK_CHARS = 300


def classify(answer):
    if "clarification_requested" in answer.reasons:
        return "clarification"
    return "refused" if answer.abstained else "answered"


def map_reason(reasons):
    # Order matters: the semantic verdict outranks the quote errors that usually accompany it.
    if "semantic_grounding_failed" in reasons:
        return "semantic"
    if any(r.startswith("unsupported_quote:") for r in reasons):
        return "quote"
    if any(
        r in ("insufficient_evidence", "uncited_claim") or r.startswith("unknown_citation:")
        for r in reasons
    ):
        return "insufficient"
    return "other"


def answered_sources(answer):
    quotes = {}
    for claim in answer.claims:
        for key, quote in claim.quotes.items():
            quotes.setdefault(key, quote)
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "quote": quotes.get(s["id"]) or s["text"][:QUOTE_FALLBACK_CHARS],
        }
        for s in answer.sources
    ]


def found_sources(retrieval):
    found = [
        {"id": e.node.id, "title": e.node.title}
        for e in retrieval.evidence
        if e.node.kind != "Document" and e.node.text
    ]
    return found[:MAX_REFUSED_SOURCES]


class Service:
    def __init__(self, conductor, meta):
        self.conductor, self.meta = conductor, meta

    def ask(self, question):
        started = time.perf_counter()
        answer, retrieval = self.conductor.run(question)
        status = classify(answer)
        # The pipeline appends raw node ids like [D::article:1]; the page shows sources instead.
        text = "\n\n".join(c.text for c in answer.claims) if status == "answered" else answer.answer
        out = {"status": status, "answer": text}
        if status == "answered":
            out["sources"] = answered_sources(answer)
        else:
            out["sources"] = [] if status == "clarification" else found_sources(retrieval)
        if status == "refused":
            out["reason"] = map_reason(answer.reasons)
        out["seconds"] = round(time.perf_counter() - started, 1)
        return out


def served_models(base_url, timeout=2):
    """Model ids the vLLM server lists, or [] when it cannot be reached."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []
