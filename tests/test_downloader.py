"""scripts/download_brazil.py with the network and sleeps mocked out."""

import gzip
import importlib.util
import io
import json
import urllib.error
from email.message import Message

import pytest

from symptompred.datasets import PROJECT_ROOT


@pytest.fixture
def dl(monkeypatch):
    spec = importlib.util.spec_from_file_location("download_brazil", PROJECT_ROOT / "scripts" / "download_brazil.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sleeps = []
    monkeypatch.setattr(module, "polite_sleep", module.sleeps.append)
    monkeypatch.setattr(module, "log", lambda msg: None)
    monkeypatch.setattr(module.random, "uniform", lambda a, b: b)  # deterministic: max jitter / backoff
    return module


def fake_api(total_rows, calls, errors=None):
    """A fake fetch_page serving `total_rows` rows; `errors` = {call number: exception} raised first."""
    def fetch(year, offset, limit, user_agent, timeout):
        calls.append(offset)
        if errors and len(calls) in errors:
            raise errors[len(calls)]
        rows = [{"id": i} for i in range(offset, min(offset + limit, total_rows))]
        return rows, 0.1
    return fetch


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("http://x", code, "err", headers, io.BytesIO())


def args(dl, tmp_path, *extra):
    return dl.parse_args(["--out", str(tmp_path), "--limit", "10", "--rest-every", "0", *extra])


def read_ids(tmp_path, year=2020):
    ids = []
    for f in sorted((tmp_path / str(year)).glob("offset_*.json.gz")):
        with gzip.open(f, "rt") as fh:
            ids += [r["id"] for r in json.load(fh)]
    return ids


def test_pages_by_row_offset_and_marks_complete(dl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "fetch_page", fake_api(25, calls))
    assert dl.download_year(2020, args(dl, tmp_path)) == "complete"
    assert calls == [0, 10, 20]  # offset is a row offset, not a page number
    assert read_ids(tmp_path) == list(range(25))
    marker = json.loads((tmp_path / "2020" / "_COMPLETE.json").read_text())
    assert marker["rows"] == 25
    # A second run skips the completed year entirely
    calls.clear()
    assert dl.download_year(2020, args(dl, tmp_path)) == "complete" and calls == []


def test_resumes_by_skipping_existing_pages(dl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "fetch_page", fake_api(35, calls))
    assert dl.download_year(2020, args(dl, tmp_path, "--max-pages", "2")) == "max-pages"
    assert calls == [0, 10]
    calls.clear()
    assert dl.download_year(2020, args(dl, tmp_path)) == "complete"
    assert calls == [20, 30]  # pages already on disk aren't fetched again
    assert read_ids(tmp_path) == list(range(35))


def test_exact_multiple_ends_on_empty_page(dl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "fetch_page", fake_api(20, calls))
    assert dl.download_year(2020, args(dl, tmp_path)) == "complete"
    assert calls == [0, 10, 20]


def test_retries_with_backoff_and_honours_retry_after(dl, tmp_path, monkeypatch):
    calls = []
    errors = {1: http_error(503), 2: http_error(429, retry_after=900), 3: TimeoutError("slow")}
    monkeypatch.setattr(dl, "fetch_page", fake_api(5, calls, errors))
    assert dl.download_year(2020, args(dl, tmp_path, "--backoff-base", "10", "--max-backoff", "600")) == "complete"
    assert calls == [0, 0, 0, 0]
    assert dl.sleeps == [10, 900, 40]  # exponential (10, 20, 40), but Retry-After wins when longer


def test_non_retryable_error_stops(dl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "fetch_page", fake_api(5, calls, {1: http_error(404)}))
    assert dl.download_year(2020, args(dl, tmp_path)) == "failed"
    assert calls == [0]


def test_gives_up_after_max_failures(dl, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "fetch_page", fake_api(5, calls, {i: http_error(500) for i in range(1, 10)}))
    assert dl.download_year(2020, args(dl, tmp_path, "--max-failures", "2")) == "failed"
    assert len(calls) == 3


def test_adaptive_pacing_never_faster_than_the_server(dl, tmp_path, monkeypatch):
    def slow(year, offset, limit, user_agent, timeout):
        return [{"id": i} for i in range(offset, min(offset + limit, 15))], 8.0
    monkeypatch.setattr(dl, "fetch_page", slow)
    dl.download_year(2020, args(dl, tmp_path, "--delay", "3", "--jitter", "0.3"))
    assert dl.sleeps == [pytest.approx(8.0 * 1.3)]


def test_limit_is_validated(dl):
    with pytest.raises(SystemExit):
        dl.parse_args(["--limit", "5000"])


def test_retry_after_parsing(dl):
    assert dl.retry_after_seconds(http_error(429, retry_after=30)) == 30
    assert dl.retry_after_seconds(http_error(429)) is None
    assert dl.retry_after_seconds(http_error(429, retry_after="nonsense")) is None
