# Peer Agent skill

Talk to the stronger peer agent over loopback HTTP. The service owns history;
you only send the new prompt.

## Endpoint

Base: `http://127.0.0.1:8471` (override `--port` on both ends if changed).

Start it first (needs `ANTHROPIC_API_KEY` in its environment):

```sh
python -m service.server --port 8471
```

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
```

Or use the CLI: `python client/chat.py "PROMPT" [--conversation-id ID] [--stream] [--cwd DIR] [--null-cwd]`.
The CLI defaults `--cwd` to its own working directory; `--null-cwd` sends
`null` for pure Q&A turns (the peer then gets no `bash` tool that turn).

## Notes

- `cwd` is required on every call: an absolute, existing directory, or `null`.
  Relative/missing paths return 400, a missing field returns 422.
- Unknown `conversation_id` returns 404; drop the id and retry to start fresh.
- The peer can run `bash` server-side; ask it to run commands when useful.
- Every turn is logged to `logs/<conversation_id>.jsonl`.
- Prefer one thread per task so the peer keeps context.
