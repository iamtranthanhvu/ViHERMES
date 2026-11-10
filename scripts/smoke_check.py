#!/usr/bin/env python3
"""First-run check that a served model behaves the way the pipeline needs.

1. The server answers a JSON-mode chat request with thinking disabled, and the reply holds
   plain JSON (no <think> block).
2. With --run DIR, the predictions of a short evaluation contain answers, no invalid structured
   answers, and report the abstention rate so a model that refuses everything is noticed.

Exit status is non-zero with a message on stderr when something is wrong. Standard library only.
"""

import argparse
import json
import urllib.error
import urllib.request


def chat(base_url, model):
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": 'Return JSON {"ok": true}.'},
            {"role": "user", "content": "{}"},
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def check_server(base_url, model):
    try:
        content = chat(base_url, model)
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"smoke_check: cannot reach {base_url}: {error}")
    if "<think" in content or "</think" in content:
        raise SystemExit(
            "smoke_check: the reply contains a <think> block, so thinking is not disabled; "
            "try a --reasoning-parser or another chat-template switch"
        )
    try:
        parsed = json.loads(content)
    except ValueError:
        raise SystemExit(f"smoke_check: the reply is not JSON: {content[:200]!r}")
    if not isinstance(parsed, dict):
        raise SystemExit("smoke_check: the reply is JSON but not an object")
    print(f"server ok: {model} answered with plain JSON")


def check_run(directory):
    path = f"{directory}/predictions.jsonl"
    try:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    except OSError as error:
        raise SystemExit(f"smoke_check: cannot read {path}: {error}")
    if not rows:
        raise SystemExit(f"smoke_check: no predictions in {path}")
    invalid = sum(
        any("invalid structured answer" in t for t in (r.get("answer") or {}).get("trace", []))
        for r in rows
    )
    abstained = sum(bool((r.get("answer") or {}).get("abstained")) for r in rows)
    if invalid:
        raise SystemExit(
            f"smoke_check: {invalid} of {len(rows)} answers were an invalid structured answer "
            "(the model did not return the JSON schema the Conductor asks for)"
        )
    print(f"run ok: {len(rows)} predictions, abstention rate {abstained}/{len(rows)}")
    if abstained == len(rows):
        print("warning: every answer abstained; check the model and retrieval before a long run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", help="evaluation output directory to inspect")
    args = parser.parse_args()
    check_server(args.base_url, args.model)
    if args.run:
        check_run(args.run)


if __name__ == "__main__":
    main()
