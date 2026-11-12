import re
from pathlib import Path

PAGE = (
    Path(__file__).resolve().parent.parent / "src" / "vihermes" / "chat" / "static" / "index.html"
)
html = PAGE.read_text(encoding="utf-8")


def test_page_has_no_prepared_answers_or_simulation():
    for forbidden in ("ITEMS", "resolve(", "match(", "mô phỏng", "STEPS", 'id="data"'):
        assert forbidden not in html, forbidden


def test_page_calls_the_real_endpoints():
    assert 'fetch("/api/ask"' in html and 'fetch("/api/health")' in html


def test_inner_html_is_only_used_for_fixed_icons():
    lines = [
        line for line in html.splitlines() if "innerHTML" in line or "insertAdjacentHTML" in line
    ]
    assert lines
    assert all("svg(" in line for line in lines), lines


def test_every_error_status_has_a_message():
    for code in ("422", "429", "503"):
        assert re.search(rf"\b{code}: \"", html), code


def test_sample_questions_are_only_questions_without_answers():
    block = html[html.index("const SAMPLES") : html.index("const REASONS")]
    assert block.count('"question"') == 12 and '"answer"' not in block
