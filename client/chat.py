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

# Long timeout: replies may include many tool turns.
_REQUEST_TIMEOUT_S = 1800.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the peer agent.")
    parser.add_argument("prompt", help="Prompt to send.")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--null-cwd", action="store_true")
    parser.add_argument(
        "--host-max-tool-turns",
        type=int,
        default=None,
        help="Host-controlled tool-turn budget (default: unlimited).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Auto-send 'continue' while the server asks for continuation.",
    )
    compact_group = parser.add_mutually_exclusive_group()
    compact_group.add_argument(
        "--compact", action="store_true", help="Force compaction this turn."
    )
    compact_group.add_argument(
        "--no-compact", action="store_true", help="Inhibit compaction this turn."
    )
    return parser.parse_args()


def _build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "cwd": None if args.null_cwd else args.cwd,
    }
    if args.conversation_id is not None:
        payload["conversation_id"] = args.conversation_id
    if args.host_max_tool_turns is not None:
        payload["host_max_tool_turns"] = args.host_max_tool_turns
    if args.compact:
        payload["compact"] = True
    elif args.no_compact:
        payload["compact"] = False
    return payload


def _run_stream(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Prints streamed tokens live, then the conversation id."""
    body: dict[str, Any] = {}
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
                    f"conversation_id={event.get('conversation_id')} "
                    f"status={event.get('status', 'done')}",
                    file=sys.stderr,
                )
                if event.get("hint"):
                    print(f"hint={event['hint']}", file=sys.stderr)
                body = {
                    "text": "",
                    "conversation_id": event.get("conversation_id"),
                    "status": event.get("status", "done"),
                    "hint": event.get("hint"),
                }
            else:
                print(f"[{kind}] {json.dumps(event)}", file=sys.stderr)
    return body


def main() -> None:
    """Sends one prompt, prints the reply, reports the conversation id."""
    args = _parse_args()
    base_url = f"http://127.0.0.1:{args.port}"
    prompt = args.prompt
    payload = _build_payload(args, prompt)
    first = True
    while True:
        if args.stream:
            body = _run_stream(base_url, payload)
        else:
            response = httpx.post(
                f"{base_url}/chat", json=payload, timeout=_REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
            body = response.json()
            print(str(body["text"]))
            print(
                f"conversation_id={body['conversation_id']} "
                f"status={body.get('status', 'done')}",
                file=sys.stderr,
            )
            if body.get("hint"):
                print(f"hint={body['hint']}", file=sys.stderr)
        args.conversation_id = body["conversation_id"]
        if not args.loop or body.get("status", "done") == "done":
            break
        if first:
            print(
                "continuation requested; sending 'continue' (--loop)", file=sys.stderr
            )
            first = False
        prompt = "continue"
        payload = _build_payload(args, prompt)


if __name__ == "__main__":
    main()
