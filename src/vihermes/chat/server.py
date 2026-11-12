"""A small HTTP front for the chat page. One question runs at a time: there is a single vLLM."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_QUESTION_CHARS = 500
MAX_BODY_BYTES = 16 * 1024
MAX_WAITING = 3
STATIC = Path(__file__).parent / "static" / "index.html"
# Names, not imports: openai is optional and the server must not need it to start.
UNREACHABLE = {"APIConnectionError", "APITimeoutError"}


class Busy(Exception):
    pass


class Gate:
    """Runs one job at a time and refuses new ones once `limit` are running or waiting."""

    def __init__(self, limit):
        self.limit, self.count = limit, 0
        self.guard, self.turn = threading.Lock(), threading.Lock()

    def run(self, job):
        with self.guard:
            if self.count >= self.limit:
                raise Busy
            self.count += 1
        try:
            with self.turn:
                return job()
        finally:
            with self.guard:
                self.count -= 1


def unreachable(error):
    return isinstance(error, ConnectionError) or type(error).__name__ in UNREACHABLE


def make_server(service, host="127.0.0.1", port=8080, ready=lambda: True, page=STATIC):
    gate = Gate(MAX_WAITING)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind="application/json; charset=utf-8"):
            data = (
                body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            )
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def fail(self, code, name):
            self.send(code, {"error": name})

        def do_GET(self):
            if self.path == "/":
                if not page.exists():
                    return self.fail(404, "not_found")
                return self.send(200, page.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/health":
                if not ready():
                    return self.send(503, {"ok": False, "error": "model_unavailable"})
                return self.send(200, {"ok": True, **service.meta})
            self.fail(404, "not_found")

        def do_POST(self):
            if self.path != "/api/ask":
                return self.fail(404, "not_found")
            question = self.question()
            if question is None:
                return self.fail(422, "invalid_question")
            try:
                result = gate.run(lambda: service.ask(question))
            except Busy:
                return self.fail(429, "busy")
            except Exception as error:  # the page must get a clean code, never a stack trace
                print(f"ask failed: {type(error).__name__}", file=sys.stderr)
                return self.fail(
                    *((503, "model_unavailable") if unreachable(error) else (500, "failed"))
                )
            self.send(200, result)

        def question(self):
            try:
                size = int(self.headers.get("Content-Length", ""))
                if not 0 < size <= MAX_BODY_BYTES:
                    return None
                value = json.loads(self.rfile.read(size))["question"]
            except (ValueError, KeyError, TypeError):
                return None
            if not isinstance(value, str):
                return None
            value = value.strip()
            return value if 0 < len(value) <= MAX_QUESTION_CHARS else None

        def log_request(self, code="-", size="-"):
            # Method, path and status only: questions are health data and must not reach logs.
            print(f"{self.command} {self.path.split('?')[0]} {code}", file=sys.stderr)

    return ThreadingHTTPServer((host, port), Handler)
