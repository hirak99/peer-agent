# Peer Agent

Stateful peer-agent service. A host agent sends a `prompt` (plus an optional
`conversation_id`); the service holds history server-side, runs the model
(Anthropic by default) with a server-side `bash` tool loop, and returns the
final `text` with the `conversation_id`.

## Run

```sh
cp .env.example .env   # then put your key in .env (git-ignored)
python -m service.server --port 8471            # default backend: anthropic
python -m service.server --backend stub --port 8471   # offline smoke test
```

## Model

Default is `claude-sonnet-5-5`. Pick another with `--model`:

```sh
python -m service.server --model claude-haiku-5-5 --port 8471   # cheap, good for testing
python -m service.server --model claude-opus-5-5 --port 8471    # strongest, priciest
```

The active model is recorded on every `response` line in `logs/`, so a
mid-thread model switch stays attributable.

## Chat

```sh
python client/chat.py "explain this repo" --port 8471
python client/chat.py "go on" --conversation-id <id> --port 8471
python client/chat.py "long answer" --stream --port 8471
```

Raw HTTP:

```sh
curl -s localhost:8471/chat -H 'content-type: application/json' \
  -d '{"prompt":"hi","cwd":"/abs/host/workdir"}'
# {"text":"...","conversation_id":"..."}
```

## MCP (Option B)

Stdio wrapper over the same HTTP service:

```sh
python interface/mcp_server.py --port 8471
```

opencode project config (`opencode.jsonc`):

```jsonc
{"mcp": {"servers": {"peer-agent": {
  "type": "local",
  "command": ["python", "interface/mcp_server.py", "--port", "8471"],
}}}}
```

This exposes a `peer_chat(prompt, conversation_id?)` tool.

## Skill (Option A)

See `interface/SKILL.md` for the plain-HTTP instructions any agent can follow.

## Layout

- `service/server.py` — FastAPI app: `POST /chat`, `POST /chat/stream`, `GET /health`
- `service/backends/` — `base.py` interface, `anthropic_backend.py`, `stub_backend.py`
- `service/tools/` — `registry.py` (open for new tools), `bash_tool.py`
- `service/store.py`, `service/event_log.py` — `conversations/*.json`, `logs/*.jsonl`
- `client/chat.py`, `interface/mcp_server.py`

## License

Apache-2.0, Copyright 2026 Arnab Bose. See `LICENSE`.
