"""Training pairs for the embedding model: (question, gold passage, hard negatives).

Only train-split questions are used. A passage that answers any test question is never used,
neither as a positive nor as a negative, so the model cannot learn the test passages.
"""

import random


def build_embed_data(examples, split, retriever, texts, negatives=3, pool=30, seed=42):
    """Return ``(rows, stats)``.

    ``retriever.seeds(query, k)`` supplies the candidates (the B1 retriever, so negatives are
    the passages the untrained system actually confuses with the answer). ``texts`` maps
    passage id to text. Each row is one (question, gold passage) pair with its own negatives.
    """
    by_id = {e.id: e for e in examples}
    absent = [i for part in ("train", "test") for i in split[part] if i not in by_id]
    if absent:
        raise ValueError(f"Split refers to {len(absent)} questions absent from the dataset")
    test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}

    rows, skipped, without_negatives = [], 0, 0
    for qid in split["train"]:
        example = by_id[qid]
        gold = set(example.gold_context_ids)
        candidates = []
        for item in retriever.seeds(example.question, pool):
            pid = item.node.id
            if pid in gold or pid in test_gold or pid not in texts or pid in candidates:
                continue
            candidates.append(pid)
        # A string seed is stable across runs and independent of the order questions are visited.
        chosen = random.Random(f"{seed}:{qid}").sample(candidates, min(negatives, len(candidates)))
        without_negatives += not chosen
        for pid in example.gold_context_ids:
            if pid not in texts or pid in test_gold:
                skipped += 1
                continue
            rows.append(
                {
                    "qid": qid,
                    "query": example.question,
                    "positive_id": pid,
                    "positive": texts[pid],
                    "negative_ids": chosen,
                    "negatives": [texts[n] for n in chosen],
                }
            )
    stats = {
        "questions": len(split["train"]),
        "pairs": len(rows),
        "skipped_missing_gold": skipped,
        "questions_without_negatives": without_negatives,
        "mean_negatives": round(sum(len(r["negative_ids"]) for r in rows) / len(rows), 2)
        if rows
        else 0,
    }
    return rows, stats
