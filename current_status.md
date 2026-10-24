# current status (rewritten at every judgment point)

- 2026-10-24: repository created; no run yet. Next: run 0 (`./run.sh data/make_splits.py --seeds 5`), then run 1 (reader and 7b / 7b-bis on the
  frozen splits; exit check: within seed variance of the main repository). Open decisions: the small pretrained encoder; the timing of round 2.
- 2026-10-24 (later): the real-data round prepared (INSTRUCTIONS_REAL_2026-10-24.md, RUNBOOK section 4): SNIPS, MASSIVE and TOPv2 as
  real user commands annotated with frames; loaders, conversion (round trips tested on gold), frozen split conditions (base with
  sizes, new slot values and unseen templates; held-out templates; out-of-domain intents; TOPv2 depth), the reader through the sop tool,
  the encoders with BIO + intent + nested heads, the shared scorer and the tables. SNIPS splits committed (ids only); MASSIVE and TOPv2 are
  downloaded and split on the DGX. Runs 8 onward after judgment point 2 unless the reviewing session decides otherwise. No result yet.
