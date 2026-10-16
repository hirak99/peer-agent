# Peer Agent skill

Talk to the peer agent over loopback HTTP. The service owns history;
you only send the new prompt.

## Endpoint

Base: `http://127.0.0.1:8471` (override `--port` on both ends if changed).

The human manages the server; never start it yourself (no background
processes, no port juggling — one server serves all callers).

## Preflight

Before your first call, check the server is up:

```sh
curl -s --max-time 5 localhost:8471/health
# {"status":"ok"}
```

If that fails, stop and tell the human the peer-agent server is down and
ask them to start it. Do not attempt to launch it.

## Chat

New thread:

```sh
curl -s localhost:8471/chat -H 'content-type: application/json' \
  -d '{"prompt":"YOUR PROMPT","cwd":"/abs/host/workdir"}'
# {"text":"...","conversation_id":"..."}
```

Continue it (no need to resend history):

```sh
curl -s localhost:8471/chat -H 'content-type: application/json' \
  -d '{"prompt":"FOLLOW-UP","conversation_id":"ID FROM LAST REPLY","cwd":"/abs/host/workdir"}'
# {"text":"...","conversation_id":"...","status":"done"}
```

Or use the CLI: `python client/chat.py "PROMPT" [--conversation-id ID] [--stream] [--cwd DIR] [--null-cwd] [--loop] [--host-max-tool-turns N]`.
The CLI defaults `--cwd` to its own working directory; `--null-cwd`
sends `null` for pure Q&A turns (no filesystem tools that turn, though
web fetching still works).

## Tool budget (opt-in)

Unlimited by default. To checkpoint long runs, set
`"host_max_tool_turns": N` on a call (server flag
`--host-max-tool-turns N` sets the default).

When the budget is spent the reply has `status:
continuation_required` with partial text. To get the full answer,
resume with the same id until `done` — do not stop at the first
partial text:

```sh
curl -s localhost:8471/chat -H 'content-type: application/json' \
  -d '{"prompt":"continue","conversation_id":"SAME_ID","cwd":"/same/dir"}'
```

`--loop` auto-sends `continue` while the server asks for it. You may
also redirect with a new prompt or stop instead of continuing.

## Notes

- `cwd` is required on every call: an absolute, existing directory, or `null`.
  Relative/missing paths return 400, a missing field returns 422.
- Unknown `conversation_id` returns 404; drop the id and retry to start fresh.
- The peer works with files, runs commands, and fetches the web
  server-side — state the outcome you want, not the tool to use.
- Long threads auto-compact (old turns → summary, recent kept).
  On failure just resend — the next turn compacts first.
  Manual/forced compaction: `POST /compact '{"conversation_id":"ID"}'`,
  or `"compact":true/false` per call.
