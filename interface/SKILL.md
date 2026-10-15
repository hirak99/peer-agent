# Peer Agent skill

Talk to the stronger peer agent over loopback HTTP. The service owns history;
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

Long replies yield instead of crashing. If `status` is
`continuation_required`, resume with the same id until `done`:

```sh
curl -s localhost:8471/chat -H 'content-type: application/json' \
  -d '{"prompt":"continue","conversation_id":"SAME_ID","cwd":"/same/dir"}'
```

Weak models: do not stop at the first partial text. Loop on
`continuation_required` — that is the gate. Per-reply tool budget defaults
to 30; pass `"max_tool_turns": 100` (or explicit `null` for unbounded)
to raise it for a known-long task.

Or use the CLI: `python client/chat.py "PROMPT" [--conversation-id ID] [--stream] [--cwd DIR] [--null-cwd] [--loop] [--max-tool-turns N]`.
`--loop` auto-sends `continue` while the server asks for it.
The CLI defaults `--cwd` to its own working directory; `--null-cwd` sends
`null` for pure Q&A turns (the peer then gets no `bash` tool that turn).

## Notes

- `cwd` is required on every call: an absolute, existing directory, or `null`.
  Relative/missing paths return 400, a missing field returns 422.
- Unknown `conversation_id` returns 404; drop the id and retry to start fresh.
- The peer can run `bash` server-side; ask it to run commands when useful.
- Every turn is logged to `logs/<conversation_id>.jsonl`.
- Prefer one thread per task so the peer keeps context.
