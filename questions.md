# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q19 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23 and R29 of responses.md, 2026-10-25, and accepted.)

## Q20. The LLM reference row (R27 item 2): waiting for the owner's confirmation of the settings proposed in R29

R29 proposes: provider OpenRouter with the DGX's key; model `anthropic/claude-sonnet-5-5`; prompt caching if exposed; temperature 0;
a budget cap of 20 USD at which the script stops and reports (spent amount from the provider's usage fields logged every 100 calls);
the synthetic sentences may be sent (no personal data). Estimated cost at the frozen prompt: about 2.14 M input and 0.19 M output tokens
for 4,800 calls, i.e. a few dollars at mid-size prices. R29 makes the call conditional on a later response saying "Q20 confirmed".
Default: no call; the row stays PLANNED with the prompt and items committed. Not on the critical path (API-bound; runs in a free slot
alongside the equal-size block once confirmed).
