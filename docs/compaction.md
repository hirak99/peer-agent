# Compaction

Long threads are summarized so work can continue past the context window.
Two paths, one switch; logs always keep the full pre-compaction history
(append-only JSONL) plus a `compaction` event with pre/post counts.

## Paths

**New-thread (default).** A separate `summarize()` call distills the
compactable prefix (per-message truncated to 6000 chars) into a handoff,
then history is spliced to summary + recent tail. Pros: bounded input,
cheaper/different summarizer via `--compact-model`, failed calls touch
nothing, prompt controlled by a dedicated system prompt. Cons: cold prefill
of the prefix on every compaction.

**In-context (`--compact-in-context`).** A transient summary request runs
one tool-less turn against the live history, then the exchange is popped
and the same splice applies. Any failure (no split point, turn error,
empty reply) silently falls back to new-thread. Pros: reuses the warm KV
cache — no cold prefill. Cons: full (untruncated) history in the prompt,
summary thinking billed on the executor, needs headroom for the exchange.

Overflow recovery inside a turn always uses new-thread: by definition the
history doesn't fit, so in-context can't run there.

## Recommendation

Hosted models: stay on new-thread (prefill is fast and cached server-side;
`--compact-model haiku` keeps summaries cheap). Local models with slow
prefill (llama.cpp Qwen and kin): pass `--compact-in-context` — the
compaction prompt is already in cache, so summaries come back in seconds
instead of re-prefilling tens of thousands of tokens cold. Raise
`--compact-margin-tokens` if summaries ever flirt with overflow.

## Knobs

`--compact-keep-messages` (default 20, tool-pair-safe split),
`--compact-margin-tokens` (default 20000; effective margin is
`max(margin, 5% of window)`), `--compact-model`,
`--compact-summary-max-tokens`, `--compact-context-window`,
`--compact-in-context`. Per-turn: `{"compact": true}` forces,
`{"compact": false}` inhibits. Manual: `POST /compact
{"conversation_id": "..."}` (CLI: `python client/chat.py "/compact"
--conversation-id <id>`).
