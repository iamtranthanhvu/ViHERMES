#!/usr/bin/env python3
"""A stand-in for `vllm serve` that answers the pipeline's prompts with canned JSON.

Used by `make dryrun` and the tests so the Makefile, scripts/serve.sh and the CLI can be
exercised end to end without a GPU or a model. It understands the arguments serve.sh passes
and records nothing else. Standard library only.

    fake_vllm.py serve MODEL --port N [any other vllm flags are accepted and ignored]
    fake_vllm.py --version

Set FAKE_VLLM_THINK=1 to prefix replies with a <think> block (to test the smoke check).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply_for(system, payload):
    if "Interpreter" in system:
        return {"regulatory": True, "use_graph": False, "document_ids": []}
    if "Conductor" in system:
        evidence = [e for e in payload.get("evidence", []) if e.get("text")]
        if not evidence:
            return {"abstain": True, "claims": []}
        first = evidence[0]
        quote = first["text"][:60]
        return {
            "abstain": False,
            "claims": [{"text": quote, "citations": [first["id"]], "quotes": {first["id"]: quote}}],
        }
    if "Auditor" in system:
        return {"supported": True, "reasons": []}
    if "Evaluate a Vietnamese regulatory QA answer" in system:
        return {"correctness": 0.5, "completeness": 0.5, "consistency": 0.5, "reason": "fake"}
    return {"ok": True}


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"object": "list", "data": [{"id": model, "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = body["messages"][0]["content"]
            try:
                payload = json.loads(body["messages"][-1]["content"])
            except ValueError:
                payload = {}
            content = json.dumps(reply_for(system, payload), ensure_ascii=False)
            if os.environ.get("FAKE_VLLM_THINK"):
                content = "<think>let me think</think>" + content
            self._send(
                200,
                {
                    "id": "fake",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", model),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        def log_message(self, *args):
            pass

    return Handler


def main(argv):
    if "--version" in argv:
        print("0.0.0-fake")
        return 0
    if len(argv) < 2 or argv[0] != "serve" or "--port" not in argv:
        print("usage: fake_vllm.py serve MODEL --port N", file=sys.stderr)
        return 2
    port = int(argv[argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(argv[1])).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
