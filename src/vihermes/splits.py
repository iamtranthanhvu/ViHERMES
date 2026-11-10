"""Leak-free train/dev/test splits: choose test first, then quarantine anything that touches it."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

PARTS = ("train", "dev", "test", "quarantined")


def _digest(split):
    body = {k: split[k] for k in (*PARTS, "test_100", "seed", "ratios")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _by_hop(examples):
    groups = defaultdict(list)
    for example in examples:
        groups[example.hop_level].append(example)
    return dict(sorted(groups.items()))


def make_split(examples, seed=42, test=0.2, dev=0.1, report_size=100):
    """Return a split dict.

    Test questions are drawn first, stratified by hop. Every other question that shares a gold
    context with a test question is quarantined (kept out of train and dev), so no passage that
    answers a test question is ever seen during training or model selection.
    """
    if not 0 < test < 1 or not 0 <= dev < 1 or test + dev >= 1:
        raise ValueError("Use 0 < test < 1, 0 <= dev < 1 and test + dev < 1")
    if report_size < 1:
        raise ValueError("report_size must be positive")
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique")
    rng = random.Random(seed)
    groups = _by_hop(sorted(examples, key=lambda e: e.id))
    order = {e.id: i for i, e in enumerate(examples)}

    test_ids = []
    for group in groups.values():
        pool = list(group)
        rng.shuffle(pool)
        size = max(1, round(len(pool) * test)) if len(pool) > 1 else 0
        test_ids += [e.id for e in pool[:size]]
    test_set = set(test_ids)
    test_gold = {c for e in examples if e.id in test_set for c in e.gold_context_ids}

    quarantined, candidates = [], defaultdict(list)
    for example in examples:
        if example.id in test_set:
            continue
        if set(example.gold_context_ids) & test_gold:
            quarantined.append(example.id)
        else:
            candidates[example.hop_level].append(example)

    dev_ids = []
    totals = {hop: len(group) for hop, group in groups.items()}
    for hop, group in sorted(candidates.items()):
        pool = sorted(group, key=lambda e: e.id)
        rng.shuffle(pool)
        dev_ids += [e.id for e in pool[: min(round(totals[hop] * dev), max(len(pool) - 1, 0))]]
    dev_set = set(dev_ids)
    train_ids = [e.id for h in sorted(candidates) for e in candidates[h] if e.id not in dev_set]

    per_hop = max(1, report_size // max(len(groups), 1))
    hop_of = {e.id: e.hop_level for e in examples}
    report = []
    for hop in groups:
        pool = sorted(i for i in test_ids if hop_of[i] == hop)
        rng.shuffle(pool)
        report += pool[:per_hop]

    def ordered(values):
        return sorted(values, key=order.__getitem__)

    split = {
        "seed": seed,
        "ratios": {"test": test, "dev": dev, "train": round(1 - test - dev, 6)},
        "train": ordered(train_ids),
        "dev": ordered(dev_ids),
        "test": ordered(test_ids),
        "quarantined": ordered(quarantined),
        "test_100": ordered(report),
    }
    by_id = {e.id: e for e in examples}
    context_sets = {
        name: {c for i in split[name] for c in by_id[i].gold_context_ids} for name in PARTS
    }
    split["stats"] = {
        "records": len(examples),
        "report_size": len(report),
        "sizes": {name: len(split[name]) for name in PARTS},
        "quarantined_fraction": round(len(quarantined) / len(examples), 4),
        "by_hop": {
            name: {str(h): sum(hop_of[i] == h for i in split[name]) for h in groups}
            for name in (*PARTS, "test_100")
        },
        "contexts": {name: len(values) for name, values in context_sets.items()},
        "leaked_contexts": len(
            context_sets["test"] & (context_sets["train"] | context_sets["dev"])
        ),
    }
    split["hash"] = _digest(split)
    return split


def save_split(split, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_split(path):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if split.get("hash") != _digest(split):
        raise ValueError(f"Split file {path} does not match its hash; it was edited or corrupted")
    return split
