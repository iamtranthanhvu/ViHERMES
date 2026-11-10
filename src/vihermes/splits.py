"""Leak-free train/dev/test splits: choose test first, then quarantine anything that touches it."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

PARTS = ("train", "dev", "test", "quarantined")


def _digest(split):
    body = {k: split[k] for k in (*PARTS, "seed", "params")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def make_split(examples, seed=42, test_size=100, dev=0.12):
    """Return a split dict.

    ``test`` holds ``test_size`` questions, stratified evenly across hop levels; it is also the
    evaluation set reported for every baseline. Every other question that shares a gold context
    with a test question is quarantined (kept out of train and dev), so no passage that answers
    a test question is seen during training or model selection. ``dev`` is the fraction of the
    remaining clean questions held out, stratified by hop, for checkpoint selection.
    """
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if not 0 <= dev < 1:
        raise ValueError("dev must be in [0, 1)")
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique")
    rng = random.Random(seed)
    order = {e.id: i for i, e in enumerate(examples)}
    groups = defaultdict(list)
    for example in sorted(examples, key=lambda e: e.id):
        groups[example.hop_level].append(example)
    groups = dict(sorted(groups.items()))
    per_hop = max(1, test_size // max(len(groups), 1))

    test_ids = []
    for group in groups.values():
        pool = list(group)
        rng.shuffle(pool)
        # Never take a whole hop group: some questions must remain for training.
        test_ids += [e.id for e in pool[: min(per_hop, len(pool) // 2)]]
    test_set = set(test_ids)
    test_gold = {c for e in examples if e.id in test_set for c in e.gold_context_ids}

    quarantined, clean = [], defaultdict(list)
    for example in examples:
        if example.id in test_set:
            continue
        if set(example.gold_context_ids) & test_gold:
            quarantined.append(example.id)
        else:
            clean[example.hop_level].append(example)

    dev_ids, train_ids = [], []
    for hop, group in sorted(clean.items()):
        pool = sorted(group, key=lambda e: e.id)
        rng.shuffle(pool)
        size = min(round(len(pool) * dev), max(len(pool) - 1, 0))
        dev_ids += [e.id for e in pool[:size]]
        train_ids += [e.id for e in pool[size:]]

    def ordered(values):
        return sorted(values, key=order.__getitem__)

    split = {
        "seed": seed,
        "params": {"test_size": test_size, "dev": dev},
        "train": ordered(train_ids),
        "dev": ordered(dev_ids),
        "test": ordered(test_ids),
        "quarantined": ordered(quarantined),
    }
    by_id = {e.id: e for e in examples}
    context_sets = {
        name: {c for i in split[name] for c in by_id[i].gold_context_ids} for name in PARTS
    }
    split["stats"] = {
        "records": len(examples),
        "sizes": {name: len(split[name]) for name in PARTS},
        "quarantined_fraction": round(len(quarantined) / len(examples), 4),
        "by_hop": {
            name: {str(h): sum(by_id[i].hop_level == h for i in split[name]) for h in groups}
            for name in PARTS
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
