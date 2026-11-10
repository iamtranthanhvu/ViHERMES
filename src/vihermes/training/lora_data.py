"""Training samples for the Conductor LoRA.

Each sample is one real Conductor call: the exact system prompt and user payload the pipeline
sends at inference time, and a target answer whose quotes are cut verbatim from the cited
passages (so the Auditor's quote check passes by construction). Questions whose evidence is
insufficient become abstentions, so the model also learns when to refuse.
"""

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict

from ..agents import GENERATE, serialize_evidence
from ..models import Evidence, Plan, Retrieval
from ..retrieval import tokens

WORD = re.compile(r"\S+")
ABSTAIN = {"claims": [], "abstain": True}


def exact_span(candidate, text):
    """The candidate if it occurs verbatim in text (ignoring edge whitespace), else None."""
    candidate = (candidate or "").strip()
    return candidate if candidate and candidate in text else None


def _windows(text, max_chars):
    """Word-aligned (start, end) windows of at most max_chars, each overlapping the next by half."""
    words = [(m.start(), m.end()) for m in WORD.finditer(text)]
    spans, i = [], 0
    while i < len(words):
        start, j = words[i][0], i
        while j + 1 < len(words) and words[j + 1][1] - start <= max_chars:
            j += 1
        spans.append((start, min(words[j][1], start + max_chars)))
        if j + 1 >= len(words):
            break
        i += max(1, (j - i + 1) // 2)
    return spans


def best_quote(text, reference, max_chars=300, min_overlap=2):
    """The window of ``text`` that best matches ``reference``, cut verbatim; None if none fits.

    Words that are rare within the passage count more, so a window holding the distinctive words
    of the reference beats one full of common words. Returns an exact slice of ``text``.
    """
    ref = {t for t in tokens(reference) if len(t) > 1}
    spans = _windows(text, max_chars)
    if not ref or not spans:
        return None
    sets = [ref & set(tokens(text[a:b])) for a, b in spans]
    df = defaultdict(int)
    for found in sets:
        for token in found:
            df[token] += 1
    best = None
    for (a, b), found in zip(spans, sets):
        if len(found) < min_overlap:
            continue
        score = sum(math.log(1 + len(spans) / df[t]) for t in found)
        key = (-score, b - a, a)
        if best is None or key < best[0]:
            best = (key, a, b)
    return text[best[1] : best[2]].strip() if best else None


def _summary(raw, index):
    evidence = (raw or {}).get("evidence")
    return str(evidence.get(f"context_{index}", "")) if isinstance(evidence, dict) else ""


def _percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else None


def _inject(nodes, gold, nodes_by_id, rng):
    """Put the missing gold passages into ``nodes`` in place of the lowest-ranked non-gold ones."""
    present = {e.node.id for e in nodes}
    missing = [g for g in gold if g not in present]
    keep = [e for e in nodes if e.node.id in set(gold)]
    others = [e for e in nodes if e.node.id not in set(gold)]
    room = max(0, len(nodes) - len(keep) - len(missing))
    chosen = keep + others[:room] + [Evidence(nodes_by_id[g], 0.0) for g in missing]
    rng.shuffle(chosen)  # injected passages must not always sit last
    return chosen, missing


def _sample(question, qid, part, kind, evidence, edges, target):
    payload = {
        "question": question,
        "evidence": serialize_evidence(Retrieval(evidence, edges, [])),
        "edges": [asdict(e) for e in edges],
    }
    return {
        "messages": [
            {"role": "system", "content": GENERATE},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "meta": {"qid": qid, "kind": kind, "part": part},
    }


def build_lora_rows(
    examples,
    raw,
    split,
    part,
    pathfinder,
    nodes_by_id,
    seed=42,
    abstain_fraction=0.15,
    max_quote_chars=300,
    max_prompt_chars=24000,
    min_overlap=2,
    inject_gold=True,
):
    """Return ``(rows, stats)`` for the questions of ``part`` ("train" or "dev").

    ``raw`` holds the dataset records (same order as the file) for the summary in ``evidence``;
    ``nodes_by_id`` maps passage id to its Node. Passages that answer any test question are
    removed from the evidence shown, and questions outside ``part`` are never read, so nothing
    from the test set reaches training.

    An answer is only written when every gold passage is in the evidence. With ``inject_gold``,
    a question whose retrieval missed some gold passages also gets an answer sample in which the
    missing ones replace the lowest-ranked non-gold passages (the prompt keeps its size); its
    real, incomplete retrieval still becomes an abstention.
    """
    if part not in ("train", "dev"):
        raise ValueError("part must be 'train' or 'dev'")
    if not 0 <= abstain_fraction < 1:
        raise ValueError("abstain_fraction must be in [0, 1)")
    by_id = {e.id: e for e in examples}
    absent = [i for p in (part, "test") for i in split[p] if i not in by_id]
    if absent:
        raise ValueError(f"Split refers to {len(absent)} questions absent from the dataset")
    texts = {i: n.text for i, n in nodes_by_id.items()}
    test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}
    rng = random.Random(f"{seed}:{part}")
    order = {q: n for n, q in enumerate(split[part])}
    counts = defaultdict(int)
    answers, natural, removed = [], [], []

    for qid in split[part]:
        example = by_id[qid]
        gold = example.gold_context_ids
        found = pathfinder.retrieve(Plan(example.question))
        shown = [e for e in found.evidence if e.node.id not in test_gold]
        ids = {e.node.id for e in shown}
        edges = [e for e in found.edges if e.source in ids and e.target in ids]
        make = lambda kind, evidence, links, target: _sample(  # noqa: E731
            example.question, qid, part, kind, evidence, links, target
        )
        injected = []
        if not set(gold) <= ids:
            natural.append(make("abstain_natural", shown, edges, ABSTAIN))
            if not inject_gold or any(g not in nodes_by_id for g in gold):
                continue
            shown, injected = _inject(shown, gold, nodes_by_id, rng)
            edges = []  # relations between the original passages no longer describe this view
        if not example.answer.strip() or any(g not in texts for g in gold):
            counts["dropped_no_quote"] += 1
            continue
        index = example.metadata.get("row", -1)
        record = raw[index] if 0 <= index < len(raw) else {}
        quotes = {}
        for k, gid in enumerate(gold, 1):
            summary = _summary(record, k)
            quote = exact_span(summary, texts[gid])
            if quote:
                counts["exact_evidence_quotes"] += 1
            else:
                reference = f"{example.answer} {summary}"
                quote = best_quote(texts[gid], reference, max_quote_chars, min_overlap)
                counts["windowed_quotes"] += bool(quote)
            if not quote:
                break
            quotes[gid] = quote
        if len(quotes) < len(gold):
            counts["dropped_no_quote"] += 1
            continue
        claim = {"text": example.answer.strip(), "citations": list(gold), "quotes": quotes}
        built = make("answer", shown, edges, {"claims": [claim], "abstain": False})
        if injected:
            built["meta"]["injected"] = injected
        answers.append(built)
        rest = [e for e in shown if e.node.id not in set(gold)]
        if rest:
            links = [e for e in edges if e.source not in set(gold) and e.target not in set(gold)]
            removed.append(make("abstain_removed", rest, links, ABSTAIN))

    lengths = []

    def short_enough(row):
        size = len(row["messages"][0]["content"]) + len(row["messages"][1]["content"])
        if size > max_prompt_chars:
            counts["dropped_too_long"] += 1
            return False
        lengths.append(size)
        return True

    answers = [r for r in answers if short_enough(r)]
    natural = [r for r in natural if short_enough(r)]
    removed = [r for r in removed if short_enough(r)]

    wanted = (
        round(abstain_fraction / (1 - abstain_fraction) * len(answers)) if abstain_fraction else 0
    )
    pool = natural + removed
    rng.shuffle(pool)
    chosen = pool[:wanted]
    rows = sorted(answers + chosen, key=lambda r: (order[r["meta"]["qid"]], r["meta"]["kind"]))
    stats = {
        "questions": len(split[part]),
        "rows": len(rows),
        "answer": len(answers),
        "answer_injected": sum(bool(r["meta"].get("injected")) for r in answers),
        "abstain": len(chosen),
        "abstain_natural": sum(r["meta"]["kind"] == "abstain_natural" for r in chosen),
        "abstain_removed": sum(r["meta"]["kind"] == "abstain_removed" for r in chosen),
        "abstain_candidates": len(pool),
        "dropped_too_long": counts["dropped_too_long"],
        "dropped_no_quote": counts["dropped_no_quote"],
        "exact_evidence_quotes": counts["exact_evidence_quotes"],
        "windowed_quotes": counts["windowed_quotes"],
        "prompt_chars": {
            "p50": _percentile(lengths, 0.5),
            "p90": _percentile(lengths, 0.9),
            "max": max(lengths) if lengths else None,
        },
    }
    return rows, stats
