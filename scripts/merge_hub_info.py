#!/usr/bin/env python3
"""Write one models.json that says which Hugging Face revisions a run used.

`vihermes fetch` leaves a hub.json (repo and commit hash) next to each downloaded model; B4 uses
two models, so their records are merged here and shipped inside the report.

    python scripts/merge_hub_info.py --embedding DIR --adapter DIR --output FILE
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = {
        name: json.loads((Path(folder) / "hub.json").read_text(encoding="utf-8"))
        for name, folder in (("embedding", args.embedding), ("adapter", args.adapter))
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
