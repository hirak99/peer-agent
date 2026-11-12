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

## When compaction fires

**Trigger (exact, preferred).** `--compact-trigger-tokens N` compacts
before a turn when the last billed input reached N. The billed count is
`response.usage.total_input_tokens`: fresh tokens plus cache reads and
writes for the call (cache counters accumulate across tool-loop calls,
while the input total keeps the latest per-call value). The old
`total_tokens` (fresh-in + out, cache excluded) is removed; use
`total_input_tokens` (+ `completion_tokens` for output). No reserve,
margin, or estimation is subtracted. Example: `--compact-trigger-tokens
300000` on a 1M model compacts a large model early at 300k billed.

**Window formula (fallback).** Without a trigger (or when no billed
usage exists yet — first turn, or backends that report none),
`est_history + max_output_tokens + effective_margin > window` decides,
where `est_history` is a `chars/4` heuristic over stored messages and the
effective margin is `max(--compact-margin-tokens, 5% of window)`. The
server logs a stderr warning whenever it falls back to the estimate while
a trigger is set. Providers without a metadata API take their true input
budget from `--model-context-window`; Anthropic reads it from the Models
API at startup.

## Knobs

`--compact-trigger-tokens` (exact billed-input threshold; beats the
window formula, no reserve/margin subtracted),
`--compact-keep-messages` (default 20, tool-pair-safe split),
`--compact-margin-tokens` (default 20000; effective margin is
`max(margin, 5% of window)`; window-formula only), `--compact-model`,
`--compact-summary-max-tokens` (default 8000),
`--max-output-tokens` (default 8192; per-call output budget, also the
window-formula reserve), `--model-context-window` (true input budget for
providers without a metadata API),
`--compact-in-context`. Deprecated aliases `--max-tokens`,
`--context-window`, `--compact-context-window` still parse with a stderr
warning. Per-turn: `{"compact": true}` forces,
`{"compact": false}` inhibits. Manual: `POST /compact
{"conversation_id": "..."}` (CLI: `python client/chat.py "/compact"
--conversation-id <id>`).

## Truncation refusal

A summary that exhausts `--compact-summary-max-tokens` (or comes back
empty) is never persisted: proactive compaction skips the splice and
continues the turn on uncompacted history, `POST /compact` fails with
502, and overflow recovery fails the turn instead of writing a cut-off
handoff. Raise `--compact-summary-max-tokens` and retry.

## Backups and recovery

Every compaction snapshots the pre-compaction history to
`conversations/backups/<conversation_id>/<UTC-timestamp>.json` before
overwriting it (newest 5 kept, older pruned). The `compaction` log event
carries the snapshot in `backup_path` plus the `summary_budget` used.
History writes are atomic (temp file + rename).

To recover: stop the server, then copy the snapshot back over
`conversations/<conversation_id>.json`. Snapshots for proactive and
recovery compactions predate the triggering turn's prompt — resend that
prompt afterwards (it is the `prompt` event just before the `compaction`
event in `logs/<id>.jsonl`). Manual `/compact` snapshots are exact.
