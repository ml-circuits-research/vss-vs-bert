# current status (rewritten at every judgment point)

- 2026-10-24: repository created; no run yet. Next: run 0 (`./run.sh data/make_splits.py --seeds 5`), then run 1 (reader and 7b / 7b-bis on the
  frozen splits; exit check: within seed variance of the main repository). Open decisions: the small pretrained encoder; the timing of round 2.
- 2026-10-24 (later): the real-data round prepared (INSTRUCTIONS_REAL_2026-10-24.md, RUNBOOK section 4): SNIPS, MASSIVE and TOPv2 as
  real user commands annotated with frames; loaders, conversion (round trips tested on gold), frozen split conditions (base with
  sizes, new slot values and unseen templates; held-out templates; out-of-domain intents; TOPv2 depth), the reader through the sop tool,
  the encoders with BIO + intent + nested heads, the shared scorer and the tables. SNIPS splits committed (ids only); MASSIVE and TOPv2 are
  downloaded and split on the DGX. Runs 8 onward after judgment point 2 unless the reviewing session decides otherwise. No result yet.
- 2026-10-25 (DGX, between judgment points; rewritten at JP 1): runs 0, 1a, 1b, 1c done and reproduce the main repository (RESULTS 0-1c);
  R6 depth-3 metrics in place (commit 3ab0ef9); pre-registration tag prereg-2026-10-25 on 7c2035a (R9.1); run 2a (BERT-base, WordPiece)
  done (RESULTS 2a); 2b running (13:00-); R10 rules on the saved seed-0 models done. Open items:
  - JP 1: the recursive tagger that can express an unseen depth (questions Q5, accepted as a proposal by R7): decided at JP 1; if run,
    reported beside the flat tagger, never instead of it.
  - JP 1 onward: the wider BERT-base grid on seed 0 (R9.5: lr {1e-5, 2e-5, 3e-5, 5e-5, 1e-4} x epochs {3, 5, 10, 20}, warm-up 10%) on the
    unseen-depth and new-identity axes, as a stated additional analysis beside the pre-registered rows.
  - before run 6: blind error analysis in eval/error_analysis.py (R9.3: model names removed, rows of all models shuffled together with a
    fixed seed, the key in a separate file committed after the classification), its own commit.
  - real-data round (after JP 2): run 8.0 first, the reproduction of published BERT numbers on SNIPS and MASSIVE en-US with standard
    splits and metrics (R8), written into INSTRUCTIONS_REAL in its own commit when the round starts.
  - the independent review of the BERT code (R9.6) arrived as R10: deviations declared in RESULTS, R10 rows beside the pre-registered ones.
  - JP 1: seeds 5-9 (R13) for every model of runs 1-3, same selected configuration, splits data/basic_seed5-9.json committed before any
    model sees them; never seed selection.
  - write-up, threats to validity (R12, Q10): "selection cost of the reader in this repository: 0; its configuration was developed in the
    main repository on the same worlds (cited rounds), so the reader's design is not blind to these worlds, while BERT's architecture is".
  - cost rows (R12, Q8): training and selection wall-clock from the logs for every row, GPU energy from 2b on, thread count and batch.
  - after JP 1: BERT-tiny / BERT-mini at the reader's size (R19; RESULTS 'Deviation R19'): both load with transformers 5.20.2 and are
    cached; grid lr {5e-5, 1e-4, 3e-4} x epochs {5, 10, 20}; interpretations (a)/(b)/(c) written in RESULTS before the run.
  - JP 1 question (R15): extending the frozen worlds to depths 4 and 5 for every model (training still at 0-2); its cost and any
    unfairness to either side are written in questions.md at JP 1, not before.
  - before JP 1, in order (R15-R18): 2b, 3a, 3b (pre-registered); the R10 rows of runs 2-3 with the R11 1b / 1c re-runs; the R14 rows;
    the R17 rows (out-of-world sets committed after 2b, before any model sees them; Q16 open).
