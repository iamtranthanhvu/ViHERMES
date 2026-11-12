import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from vihermes.chat.server import MAX_QUESTION_CHARS, make_server

GOOD = {"status": "answered", "answer": "ok", "sources": [], "seconds": 0.1}


class FakeService:
    meta = {"model": "m", "lora": "l", "corpus_units": 2, "mode": "b3"}

    def __init__(self, result=GOOD, delay=0.0, error=None):
        self.result, self.delay, self.error = result, delay, error
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.asked = []

    def ask(self, question):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.asked.append(question)
            time.sleep(self.delay)
            if self.error:
                raise self.error
            return self.result
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(service, ready=lambda: True, page=None):
        page = page or tmp_path / "index.html"
        server = make_server(service, "127.0.0.1", 0, ready, page)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def call(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def ask(base, question):
    code, body = call(base + "/api/ask", {"question": question})
    return code, json.loads(body)


def test_ask_returns_service_result(serve):
    base = serve(FakeService())
    assert ask(base, "  Câu hỏi?  ") == (200, GOOD)


@pytest.mark.parametrize("question", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1), 5, None])
def test_invalid_question_is_422_and_server_survives(serve, question):
    service = FakeService()
    base = serve(service)
    assert ask(base, question)[0] == 422
    assert service.asked == []
    assert ask(base, "ok")[0] == 200


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"q": 1}', b""])
def test_bad_body_is_422(serve, raw):
    base = serve(FakeService())
    assert call(base + "/api/ask", raw=raw)[0] == 422


def test_limit_is_inclusive(serve):
    base = serve(FakeService())
    assert ask(base, "x" * MAX_QUESTION_CHARS)[0] == 200


def test_unreachable_model_is_503_and_other_errors_are_500_without_trace(serve):
    base = serve(FakeService(error=ConnectionError("boom")))
    assert ask(base, "q") == (503, {"error": "model_unavailable"})
    base = serve(FakeService(error=RuntimeError("secret detail")))
    code, body = ask(base, "q")
    assert (code, body) == (500, {"error": "failed"})


def test_health_ok_and_503_when_model_not_ready(serve):
    code, body = call(serve(FakeService()) + "/api/health")
    assert code == 200 and json.loads(body) == {"ok": True, **FakeService.meta}
    code, body = call(serve(FakeService(), ready=lambda: False) + "/api/health")
    assert code == 503 and json.loads(body)["ok"] is False


def test_serves_page_and_404(serve, tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html>hi</html>", encoding="utf-8")
    base = serve(FakeService(), page=page)
    assert call(base + "/") == (200, b"<html>hi</html>")
    assert call(base + "/nope")[0] == 404


def test_questions_run_one_at_a_time(serve):
    service = FakeService(delay=0.15)
    base = serve(service)
    results = []
    threads = [threading.Thread(target=lambda: results.append(ask(base, "q")[0])) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200, 200, 200]
    assert service.peak == 1


def test_queue_full_is_429(serve):
    service = FakeService(delay=0.5)
    base = serve(service)
    results = []
    threads = [threading.Thread(target=lambda: results.append(ask(base, "q")[0])) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [200, 200, 200, 429, 429]


def test_log_has_no_question_text(serve, capsys):
    base = serve(FakeService())
    ask(base, "bí mật sức khỏe của tôi")
    err = capsys.readouterr().err
    assert "POST /api/ask 200" in err
    assert "bí mật" not in err
