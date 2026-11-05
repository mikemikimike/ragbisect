import io
import json
import socket
import threading
import urllib.error
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ragbisect.llm import LLM, LLMError


@pytest.mark.parametrize("kind", ["completion", "embedding"])
def test_dropped_connection_retries_and_caches_success(tmp_path, monkeypatch, kind):
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            type(self).calls += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.calls == 1:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if self.path.endswith("/embeddings"):
                response = {"data": [{"index": 0, "embedding": [0.5]}]}
            else:
                response = {"choices": [{"message": {"content": "answer"}}]}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ragbisect.llm.time.sleep", lambda _: None)
    try:
        llm = LLM(cache_path=tmp_path / "cache.sqlite", timeout=2)
        if kind == "completion":
            assert llm.complete("question") == "answer"
            assert llm.complete("question") == "answer"
        else:
            assert llm.embed(["question"]) == [[0.5]]
            assert llm.embed(["question"]) == [[0.5]]
        assert Handler.calls == 2
        assert llm.usage.calls == 1
        assert llm.usage.cache_hits == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "error",
    [RemoteDisconnected("closed"), ConnectionResetError("reset"),
     urllib.error.URLError("unreachable"), TimeoutError("timed out")],
)
def test_network_errors_use_existing_retry_backoff(tmp_path, monkeypatch, error):
    calls = []
    delays = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr("ragbisect.llm.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("ragbisect.llm.time.sleep", delays.append)
    llm = LLM(cache_path=tmp_path / "cache.sqlite")
    assert llm._post("http://example.test", {}, {}) == {"ok": True}
    assert len(calls) == 2
    assert delays == [1.0]


@pytest.mark.parametrize("status, attempts", [(401, 1), (429, 2)])
def test_http_error_retry_policy_is_preserved(tmp_path, monkeypatch, status, attempts):
    calls = []
    delays = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("http://example.test", status, "error", {}, io.BytesIO(b"error"))
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr("ragbisect.llm.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("ragbisect.llm.time.sleep", delays.append)
    llm = LLM(cache_path=tmp_path / "cache.sqlite")
    if status == 401:
        with pytest.raises(LLMError, match="HTTP 401"):
            llm._post("http://example.test", {}, {})
    else:
        assert llm._post("http://example.test", {}, {}) == {"ok": True}
    assert len(calls) == attempts
    assert delays == ([1.0] if status == 429 else [])


def test_exhausted_connection_errors_are_not_cached(tmp_path, monkeypatch):
    calls = []
    delays = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 12:
            raise ConnectionResetError("reset")
        return io.BytesIO(b'{"choices": [{"message": {"content": "answer"}}]}')

    monkeypatch.setattr("ragbisect.llm.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("ragbisect.llm.time.sleep", delays.append)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(cache_path=tmp_path / "cache.sqlite")
    for _ in range(2):
        with pytest.raises(LLMError, match="network error") as exc:
            llm.complete("question")
        assert isinstance(exc.value.__cause__, ConnectionResetError)
    assert len(calls) == 12
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0] * 2
    assert llm.usage.calls == llm.usage.cache_hits == 0
    assert llm.complete("question") == "answer"
    assert llm.complete("question") == "answer"
    assert len(calls) == 13
    assert llm.usage.calls == llm.usage.cache_hits == 1
