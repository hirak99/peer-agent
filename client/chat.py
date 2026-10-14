"""Thin CLI client for the peer-agent service."""

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import httpx

from service import server

from typing import Any

# Long timeout: replies may include several tool turns.
_REQUEST_TIMEOUT_S = 300.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the peer agent.")
    parser.add_argument("prompt", help="Prompt to send.")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--null-cwd", action="store_true")
    return parser.parse_args()


def _run_stream(base_url: str, payload: dict[str, Any]) -> None:
    """Prints streamed tokens live, then the conversation id."""
    with httpx.stream(
        "POST", f"{base_url}/chat/stream", json=payload, timeout=_REQUEST_TIMEOUT_S
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event: dict[str, Any] = json.loads(line[len("data: ") :])
            kind = event.get("type")
            if kind == "token":
                print(str(event.get("text", "")), end="", flush=True)
            elif kind == "done":
                print()
                print(
                    f"conversation_id={event.get('conversation_id')}", file=sys.stderr
                )
            else:
                print(f"[{kind}] {json.dumps(event)}", file=sys.stderr)


def main() -> None:
    """Sends one prompt, prints the reply, reports the conversation id."""
    args = _parse_args()
    payload: dict[str, Any] = {
        "prompt": args.prompt,
        "cwd": None if args.null_cwd else args.cwd,
    }
    if args.conversation_id is not None:
        payload["conversation_id"] = args.conversation_id
    base_url = f"http://127.0.0.1:{args.port}"
    if args.stream:
        _run_stream(base_url, payload)
        return
    response = httpx.post(f"{base_url}/chat", json=payload, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    print(str(body["text"]))
    print(f"conversation_id={body['conversation_id']}", file=sys.stderr)


if __name__ == "__main__":
    main()
