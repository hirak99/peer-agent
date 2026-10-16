# Prompt caching

Peer-agent marks ephemeral cache breakpoints on the system prompt, the
last tool definition, and the history tail on every request, so
repeated history is served from cache (5-min TTL) instead of
rebilled. Breakpoints attach by default and suspend only after
`n = _CACHE_IDLE_DISABLE_AFTER` (in
`service/backends/anthropic_backend.py`) consecutive gaps over
5 minutes. Per-request copies keep stored history clean.

Watch `cached_prompt_tokens` vs `cache_creation_input_tokens` on
`response` lines in `logs/`.

## Pricing notation

Per MTok of cached prefix `P`, with base input price `B`:

- `w5 = 1.25`: 5-minute cache-write multiplier.
- `w1 = 2.0`: 1-hour cache-write multiplier.
- `r`: cache-read multiplier — `0.05` on Opus 5.5 / Sonnet 5.5
  (`$0.20/M` / `$0.10/M`), `0.025` on Fable 5.1 / Mythos 5.1,
  `0.1` elsewhere.

Current absolutes ($/MTok, standard global tier, Oct 2026):

| Model            | Base write | 5m write | 1h write | Read | Output |
|------------------|------------|----------|----------|------|--------|
| Opus 5.5         | 4.00       | 5.00     | 8.00     | 0.20 | 20.00  |
| Sonnet 5.5       | 2.00       | 2.50     | 4.00     | 0.10 | 10.00  |
| Haiku 5.5 <=100k | 0.10       | 0.125    | 0.20     | 0.01 | 0.50   |
| Haiku 5.5 >100k  | 0.50       | 0.625    | 1.00     | 0.05 | 2.50   |

(Haiku tier counts the full prompt incl. cache reads/writes.)

Minimum cacheable prefix is 512 tokens on the 5.5 models; shorter
prefixes silently skip caching (`cache_creation_input_tokens: 0`).
A cache hit refreshes the TTL, measured from request start, so
generation time counts against it.

## Gate breakeven

Attaching a breakpoint costs `(w - 1) * B * P` extra on a miss and
saves `(1 - r) * B * P` on a hit. Breakeven hit probability `h`:

```
(w - 1) * (1 - h) = (1 - r) * h
=> h = (w - 1) / (w - r)
```

For 5-minute writes on Opus/Sonnet 5.5: `h = 0.25 / 1.20 = 0.2083`.
Tune `n` so that after `n` consecutive over-window misses,
`P(next request hits) < h`. We use `n = 3` as the judgment call
that three 5+ minute gaps imply `P(hit) < 0.2083`. Slow bursts
past `n` stay attached; fully dead sessions waste one extra write
per breakpoint level on the resume.

## Keepalive vs 1-hour TTL (considered, not implemented)

Alternative: re-send the unchanged prefix with `max_tokens: 0`
every ~4.5 minutes while idle (bills only a cache read, no
output) to hold the 5-minute entry, up to ~1 hour, instead of
paying `2x` for the 1-hour TTL.

Per MTok held for a full hour (13.3 reads + initial 5m write):

- `r = 0.05`: `1.25 + 13.3*0.05 = 1.92x` vs `2.0x` for 1h — keepalive wins ~4%.
- `r = 0.1`: `1.25 + 1.33 = 2.58x` vs `2.0x` — 1h wins.
- `r = 0.025`: `1.25 + 0.33 = 1.58x` vs `2.0x` — keepalive wins.

Rescuing one hit is worth `(1 - r) * B * P`; the full-hour
keepalive run costs `N * r * B * P`. Breakeven resume probability:

```
P(resume within 1h) > N * r / (1 - r)
```

That is ~0.70 for `r = 0.05`, >1 (never pays) for `r = 0.1`,
~0.34 for `r = 0.025`. Peer-agent gaps look bimodal — tight
tool-loop turns that already hit, then hours/days idle — so a
1-hour hold still misses the next-day resume after paying for
13 reads, while the gate pays nothing while silent. Keepalive
also needs per-conversation timers, byte-identical prefix
replay, restart persistence, and a token-size floor; without
the `max_tokens: 0` shape, reply output dominates small
prompts.

Revisit if `logs/` show many resumes landing in the 5–60min
band. Cheaper first step than keepalives: selective `ttl: "1h"`
on the stable system/tools prefix with the volatile tail at
5 minutes (longer TTLs must come first).
