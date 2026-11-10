"""Read-only check: how do `evidence` spans relate to `used_context_ids` and `context_block`?

Decides how LoRA training targets can be built: a quote is usable only if it is an exact
substring of the passage it will be cited against.

    python scripts/check_evidence.py dataset/dataset.jsonl
"""

import json
import re
import sys
from collections import defaultdict

BLOCK = re.compile(
    r"\[CONTEXT\s+\d+\]\s*\nID:\s*([^\n]+)\nTitle:\s*([^\n]+)\nBody:\s*\n(.*?)(?=\n\[CONTEXT\s+\d+\]|\Z)",
    re.S,
)


def contexts_of(row):
    return [(m.group(1).strip(), m.group(3).strip()) for m in BLOCK.finditer(row["context_block"])]


def analyse(rows):
    stats = defaultdict(lambda: defaultdict(int))
    for row in rows:
        hop = row["hop_level"]
        s = stats[hop]
        s["records"] += 1
        passages = dict(contexts_of(row))
        gold = row["used_context_ids"]
        evidence = row.get("evidence") or {}
        s["gold_in_block"] += all(g in passages for g in gold)
        s["evidence_count_eq_hop"] += len(evidence) == hop
        spans = list(evidence.values())
        in_some = [any(span in text for text in passages.values()) for span in spans]
        s["all_spans_in_some_passage"] += bool(spans) and all(in_some)
        # Position-based mapping: evidence key context_k <-> k-th used_context_id.
        positional = True
        for key, span in evidence.items():
            m = re.fullmatch(r"context_(\d+)", key)
            if not m or not 1 <= int(m.group(1)) <= len(gold):
                positional = False
                break
            if span not in passages.get(gold[int(m.group(1)) - 1], ""):
                positional = False
                break
        s["positional_mapping_ok"] += positional and bool(evidence)
        # Content-based mapping: every span is found in exactly one gold passage.
        unique = True
        for span in spans:
            owners = [g for g in gold if span in passages.get(g, "")]
            if len(owners) != 1:
                unique = False
                break
        s["content_mapping_unique"] += unique and bool(spans)
    return stats


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    stats = analyse(rows)
    fields = [
        "records",
        "gold_in_block",
        "evidence_count_eq_hop",
        "all_spans_in_some_passage",
        "positional_mapping_ok",
        "content_mapping_unique",
    ]
    print("hop | " + " | ".join(fields))
    total = defaultdict(int)
    for hop in sorted(stats):
        print(f"{hop}   | " + " | ".join(str(stats[hop][f]) for f in fields))
        for f in fields:
            total[f] += stats[hop][f]
    print("all | " + " | ".join(str(total[f]) for f in fields))
    n = total["records"]
    for f in fields[1:]:
        print(f"{f}: {100 * total[f] / n:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset.jsonl")
