"""MCP stdio wrapper exposing the peer-agent service as a tool."""

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import httpx
from mcp.server import fastmcp

from service import server

from typing import Any

# Long timeout: replies may include several tool turns.
_REQUEST_TIMEOUT_S = 300.0
# Mutable runtime config (contents set from --port in main()).
_CONFIG = {"port": server.DEFAULT_PORT}

MCP = fastmcp.FastMCP("peer-agent")


@MCP.tool()
def peer_chat(
    prompt: str,
    conversation_id: str | None = None,
    cwd: str | None = None,
) -> str:
    """Ask the peer agent; pass back conversation_id to continue the thread."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "cwd": cwd if cwd is not None else os.getcwd(),
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = httpx.post(
        f"http://127.0.0.1:{_CONFIG['port']}/chat",
        json=payload,
        timeout=_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return json.dumps(
        {"text": body["text"], "conversation_id": body["conversation_id"]}
    )


def main() -> None:
    """Runs the MCP stdio server against the peer-agent HTTP service."""
    parser = argparse.ArgumentParser(description="Peer-agent MCP wrapper.")
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    args = parser.parse_args()
    _CONFIG["port"] = args.port
    MCP.run()


if __name__ == "__main__":
    main()
