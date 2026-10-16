"""Web fetch tool: GET a URL over HTTP(S), return truncated text."""

import urllib.parse

import httpx

from service.backends import base
from service.tools import registry

from typing import Any

# Machine name referenced by model tool_use blocks.
WEB_FETCH_TOOL_NAME = "web_fetch"
# Default cap on fetch runtime; override with --web-timeout.
DEFAULT_WEB_TIMEOUT_S = 30
# Default cap on body text kept in history and logs; override with --web-max-bytes.
DEFAULT_WEB_MAX_BYTES = 32768
# Cap on redirects followed per fetch.
_MAX_REDIRECTS = 5


def _truncate(body: str, max_bytes: int) -> str:
    """Cuts body to its head, marking the cut when one happened."""
    if (
        len(body) <= max_bytes
        and len(body.encode("utf-8", errors="ignore")) <= max_bytes
    ):
        return body
    return f"<truncated, showing first {max_bytes} of ~{len(body)} chars>\n{body[:max_bytes]}"


def _fetch(url: str, timeout_s: int) -> httpx.Response:
    """GETs one URL, following a bounded redirect chain."""
    client = httpx.Client(
        timeout=timeout_s, follow_redirects=True, max_redirects=_MAX_REDIRECTS
    )
    try:
        return client.get(url, headers={"User-Agent": "peer-agent web_fetch"})
    finally:
        client.close()


def build_web_fetch_tool(
    timeout_s: int = DEFAULT_WEB_TIMEOUT_S,
    max_bytes: int = DEFAULT_WEB_MAX_BYTES,
) -> registry.Tool:
    """Builds the registry entry for the web fetch tool."""

    def _handle(
        tool_input: dict[str, Any], context: base.ToolContext
    ) -> registry.ToolResult:
        _ = context
        url = str(tool_input.get("url", "")).strip()
        if not url:
            return registry.ToolResult(
                output="<error: missing required 'url'>",
                failure_reason="missing url",
            )
        try:
            parsed = urllib.parse.urlparse(url)
        except ValueError:
            return registry.ToolResult(
                output=f"<error: invalid url: {url}>",
                failure_reason="invalid url",
            )
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return registry.ToolResult(
                output=f"<error: only http(s) URLs may be fetched: {url}>",
                failure_reason="invalid scheme",
            )
        try:
            response = _fetch(url, timeout_s)
        except httpx.TimeoutException:
            return registry.ToolResult(
                output=f"<error: timed out after {timeout_s}s: {url}>",
                failure_reason=f"timed out after {timeout_s}s",
            )
        except httpx.TooManyRedirects:
            return registry.ToolResult(
                output=f"<error: too many redirects: {url}>",
                failure_reason="too many redirects",
            )
        except httpx.ConnectError as exc:
            return registry.ToolResult(
                output=f"<error: connection failed for {url}: {exc}>",
                failure_reason="connection failed",
            )
        except httpx.RequestError as exc:
            return registry.ToolResult(
                output=f"<error: request failed for {url}: {exc}>",
                failure_reason="request failed",
            )
        raw = response.content
        if b"\x00" in raw[:8192]:
            return registry.ToolResult(
                output=f"<error: non-text content at {url}>",
                failure_reason="binary content",
            )
        text = response.text
        if response.status_code >= 400:
            body = _truncate(text, max_bytes) if text else "<no body>"
            return registry.ToolResult(
                output=f"{url} -> HTTP {response.status_code}\n{body}",
                failure_reason=f"http {response.status_code}",
            )
        body = _truncate(text, max_bytes) if text else "<ok, empty body>"
        header = f"--- {url} (HTTP {response.status_code}, {len(text)} chars) ---"
        return registry.ToolResult(output=f"{header}\n{body}")

    return registry.Tool(
        name=WEB_FETCH_TOOL_NAME,
        description=(
            "Fetch an http(s) URL and return its body as text, truncated. "
            "Use this instead of curl via bash for single web fetches. "
            "Works without a workdir."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "http(s) URL to fetch.",
                }
            },
            "required": ["url"],
        },
        handler=_handle,
    )
