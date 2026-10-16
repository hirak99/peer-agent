# Peer Agent

Stateful peer-agent service. A host agent sends a `prompt` (plus an optional
`conversation_id`); the service holds history server-side, runs the model
(Anthropic by default) with a server-side `bash` tool loop, and returns the
final `text` with the `conversation_id`.

## Quick Start

- `cp .env.example .env` and put your key in `.env`.
- Start the service from this directory: `python -m service.server --port 8471`.

- Start any coding agent **from your project directory** (not here).
- Ask your agent to read the `<abs-path-to-this-repo>/interface/SKILL.md`.
- Now you can ask your agent to talk to peer agent, e.g. "validate your plan with peer agent".

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

## Prompt caching

Every request marks ephemeral cache breakpoints on the system prompt, the
last tool definition, and the history tail, so repeated history is served
from cache (5-min TTL) instead of rebilled. Breakpoints attach by default and
suspend only after `n = _CACHE_IDLE_DISABLE_AFTER` (in
`service/backends/anthropic_backend.py`) consecutive gaps over 5 minutes.
See `docs/prompt-caching.md` for pricing, the breakeven derivation, and the
keepalive-vs-1h analysis.

Watch `cached_prompt_tokens` vs `cache_creation_input_tokens` on `response`
lines in `logs/`.


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

## Compaction

Long threads auto-compact: when `est_history + max_tokens + effective_margin`
exceeds the model's input budget (read from the Models API at startup, 1M for
current Claude 5.5 models), old turns are summarized and replaced, keeping the
last `--compact-keep-messages` (default 20). The effective margin is
`max(--compact-margin-tokens, 5% of window)`, so the fixed default (20000)
scales up on large-context models. Full history stays in `logs/`;
compactions are logged as `compaction` events. Per-turn override:
`{"compact": true/false}`. Manual, no model turn: `POST /compact
{"conversation_id": "..."}` (CLI: `python client/chat.py "/compact"
--conversation-id <id> --port 8471`). On context-overflow the turn
retries once after compaction instead of failing. Server knobs:
`--compact-model` (e.g. haiku for cheap summaries), `--compact-summary-max-tokens`,
`--compact-context-window`.

## Skill (Option A)

See `interface/SKILL.md` for the plain-HTTP instructions any agent can follow.

## MCP (Option B)

Stdio wrapper over the same HTTP service:

```sh
python interface/mcp_server.py --port 8471
```

opencode project config (`opencode.jsonc`). Use absolute paths — the wrapper
runs from the host agent's project directory, not from here:

```jsonc
{"mcp": {"servers": {"peer-agent": {
  "type": "local",
  "command": ["/abs/path/to/venv/python", "/abs/path/to/peer-agent/interface/mcp_server.py", "--port", "8471"],
}}}}
```

This exposes a `peer_chat(prompt, conversation_id?, cwd?)` tool.

## Layout

- `service/server.py` — FastAPI app: `POST /chat`, `POST /chat/stream`, `POST /compact`, `GET /health`
- `service/compaction.py` — margin trigger, tool-pair-safe split, summary helpers
- `service/backends/` — `base.py` interface, `anthropic_backend.py`, `stub_backend.py`
- `service/tools/` — `registry.py` (open for new tools), `bash_tool.py`, `read_tool.py`, `edit_tool.py`
- `service/store.py`, `service/event_log.py` — `conversations/*.json`, `logs/*.jsonl`
- `client/chat.py`, `interface/mcp_server.py`

## License

Apache-2.0, Copyright 2026 Arnab Bose. See `LICENSE`.
