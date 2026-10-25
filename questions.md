# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q21 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31 and R32 of responses.md, 2026-10-25, and accepted.)

## Q22. R32: three points the API settles differently from R32's text (no call is made until "Q22 confirmed")

(1) **Temperature 0 cannot be sent.** Since 2026-10-21 a Messages request carrying `temperature`, `top_p` or `top_k` returns HTTP 400 on
every model. Default: no sampling parameter; the three repeats measure the row's determinism as R27 asks, and RESULTS states that
temperature could not be set. (2) **Prompt caching will not engage**: the frozen prompt is about 415 tokens (estimate), below
claude-sonnet-5-5's 512-token cacheable minimum; the cache_control marker is kept (harmless) and the usage fields (cache read / write
tokens) are logged, so it is visible either way. The estimate of about 2.14 M input / 0.19 M output tokens then costs about $6.2 at
$2 / $10 per M (the prices in the API skill's table cached 2026-10-06; cache write $2.50, cache read $0.20), within the $20 cap.
(3) **Thinking and effort.** claude-sonnet-5-5 runs adaptive thinking by default (it cannot be disabled except by `between_tools`), and
thinking tokens are billed as output. Default: `effort: low` (the reading task is short; it keeps the cost near the estimate), and no
server-side refusal fallback (so that every answer comes from the named model; a refusal is recorded as such). Prices are passed on the
command line as read on the pricing page at call time and written into the log. The script (eval/llm_reference.py --call) is committed;
it reads ANTHROPIC_API_KEY only and never writes the key, headers or environment to a log.
