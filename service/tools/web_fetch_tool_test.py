"""Tests for the web fetch tool, hermetic via loopback HTTP."""

import contextlib
import threading
import time
from http import server as http_server

from service.backends import base
from service.tools import web_fetch_tool


class _Handler(http_server.BaseHTTPRequestHandler):
    """Serves fixed routes for fetch tests; unlisted paths 404."""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/ok":
            body = b"hello web"
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/large":
            body = b"x" * 1000
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/slow":
            time.sleep(2)
            body = b"late"
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError):
                self.wfile.write(body)
        elif self.path == "/redir":
            self.send_response(302)
            self.send_header("location", "/ok")
            self.end_headers()
        elif self.path == "/binary":
            body = b"\x00\x01\x02binary"
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"missing")


def _base_url() -> tuple[str, http_server.ThreadingHTTPServer]:
    httpd = http_server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", httpd


def test_success_returns_body_without_workdir() -> None:
    """Fetch works with a null workdir, unlike bash/read."""
    root, httpd = _base_url()
    try:
        result = web_fetch_tool.build_web_fetch_tool().handler(
            {"url": f"{root}/ok"}, base.ToolContext(workdir=None)
        )
    finally:
        httpd.shutdown()
    assert result.failure_reason is None
    assert "hello web" in result.output


def test_missing_url_is_an_error() -> None:
    """Empty url fails with a reason instead of fetching."""
    result = web_fetch_tool.build_web_fetch_tool().handler(
        {"url": ""}, base.ToolContext(workdir=None)
    )
    assert result.failure_reason == "missing url"


def test_non_http_scheme_is_rejected() -> None:
    """file/ftp URLs are refused without any network call."""
    for url in ("file:///etc/hostname", "ftp://example.com/x", "not a url"):
        result = web_fetch_tool.build_web_fetch_tool().handler(
            {"url": url}, base.ToolContext(workdir=None)
        )
        assert result.failure_reason == "invalid scheme"


def test_404_reports_failure_reason() -> None:
    """HTTP errors keep the status and any body."""
    root, httpd = _base_url()
    try:
        result = web_fetch_tool.build_web_fetch_tool().handler(
            {"url": f"{root}/absent"}, base.ToolContext(workdir=None)
        )
    finally:
        httpd.shutdown()
    assert result.failure_reason == "http 404"


def test_truncation_is_marked() -> None:
    """Oversized bodies are cut with a marker, not silently."""
    root, httpd = _base_url()
    try:
        result = web_fetch_tool.build_web_fetch_tool(max_bytes=10).handler(
            {"url": f"{root}/large"}, base.ToolContext(workdir=None)
        )
    finally:
        httpd.shutdown()
    assert result.failure_reason is None
    assert result.output.startswith("--- ")
    assert "<truncated," in result.output


def test_redirect_is_followed() -> None:
    """One redirect chain resolves to the target body."""
    root, httpd = _base_url()
    try:
        result = web_fetch_tool.build_web_fetch_tool().handler(
            {"url": f"{root}/redir"}, base.ToolContext(workdir=None)
        )
    finally:
        httpd.shutdown()
    assert result.failure_reason is None
    assert "hello web" in result.output


def test_binary_is_rejected() -> None:
    """Non-text bodies fail instead of returning garbage."""
    root, httpd = _base_url()
    try:
        result = web_fetch_tool.build_web_fetch_tool().handler(
            {"url": f"{root}/binary"}, base.ToolContext(workdir=None)
        )
    finally:
        httpd.shutdown()
    assert result.failure_reason == "binary content"


def test_timeout_reports_failure_reason() -> None:
    """A slow endpoint fails with a timeout reason."""
    root, httpd = _base_url()
    try:
        tool = web_fetch_tool.build_web_fetch_tool(timeout_s=1)
        result = tool.handler({"url": f"{root}/slow"}, base.ToolContext(workdir=None))
    finally:
        httpd.shutdown()
    assert result.failure_reason is not None
    assert "timed out" in result.failure_reason
