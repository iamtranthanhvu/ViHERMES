"""Regenerate tests/fixtures/dryrun/qa.jsonl: a tiny synthetic dataset in the released format.

Only for offline dry runs of the Makefile; it has nothing to do with the research benchmark.
Question IDs repeat across hops on purpose, as in the released dataset.

    python scripts/make_dryrun_fixture.py
"""

import json
import random
from pathlib import Path

PER_HOP, POOL, SEED = 16, 400, 0
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "dryrun" / "qa.jsonl"


def passage(i):
    return {
        "id": f"{i:04d}-fix-{(i * 7) % 9973:04d}",
        "title": f"Quyết định số {i}/QĐ-BYT về hồ sơ loại a{i}",
        "text": (
            f"Điều {i}. Hồ sơ loại a{i} gồm giấy tờ b{i} do cơ quan c{i} cấp, "
            f"thời hạn xử lý d{i} ngày làm việc."
        ),
    }


def main():
    rng = random.Random(SEED)
    pool = [passage(i) for i in range(POOL)]
    rows = []
    for hop in range(1, 6):
        for n in range(1, PER_HOP + 1):
            chosen = rng.sample(pool, hop)
            block = "\n".join(
                f"[CONTEXT {k}]\nID: {c['id']}\nTitle: {c['title']}\nBody:\n{c['text']}"
                for k, c in enumerate(chosen, 1)
            )
            tags = [c["title"].rsplit(" ", 1)[1] for c in chosen]
            rows.append(
                {
                    "id": f"q{n}",
                    "question": "Hồ sơ "
                    + " và hồ sơ ".join(f"loại {t}" for t in tags)
                    + " gồm gì?",
                    "answer": " ".join(f"Hồ sơ loại {t} gồm giấy tờ." for t in tags),
                    "hop_level": hop,
                    "evidence": {
                        f"context_{k}": f"Cung cấp thông tin về hồ sơ loại {t}."
                        for k, t in enumerate(tags, 1)
                    },
                    "reasoning": "Ghép thông tin từ các đoạn.",
                    "used_context_ids": [c["id"] for c in chosen],
                    "context_block": block,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
