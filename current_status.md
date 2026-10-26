# current status (rewritten at every judgment point)

## 2026-10-26 19:15 — close of the day (R93)

SNIPS, per-intent configuration (seven worlds, intent anchor, one-vs-rest negatives; R81 models, seeds 0-2):
- The demo reading (R90 item 1, MEASURED): dispatch by the anchor percentile 0.966 (one-world reader 0.977, BERT-tiny 0.985); frame =
  R88 line C (in-context kinds, admission from them, presence by kinds, kind_run spans, presence queries) span F1 0.892 (typer alone
  0.886; R81 0.824); refusal = line A's explained residual of the chosen world above its dev-calibrated 2% threshold: 0.460 of the
  held-out intent refused at 0.016 in-domain false rejection.
- Causes on record: R75's thresholds at 1.0 came from the frame-incomplete clause (Q33); the readout's losses were presence decisions
  between sibling questions that admit each other's lexicon kinds (R86); presence by kinds closes most of them.
- R85 (vorder on AddToPlaylist): need signal fired on one world; under the paired test the route is rejected on 2 of 3 seeds and wins
  nothing on its marked rows.
- R92 step 1: one config per world reproduces the frame but not the refusal (0.377: the refusal pass reads with (B)'s admission sets).
- Export: logs/export/snips_demo_seed2/ (seed 2; 7 worlds via export-web, typer 2.8 MB, reading.json with R90's two-config reading);
  the single-config export of R92 is declared, not run.

Open for tomorrow (R93): R90 item 2's outcome (above: 1 of 3 accepted), the demo export with the single config (needs the refusal
pass to read the stored admission sets), R78 (2) on the per-intent configuration, the SGD-DST baseline. Nothing runs.

## 2026-10-25 22:30 — JP 1 (the experiment stops here by the owner's rule R48)

What was measured (RESULTS 'JP 1: synthesis' holds both tables, the confirmatory results, the regime statements and the limitations):
- Pre-registered rows (tag prereg-2026-10-25): 0, 1a, 1b, 1c, 2a, 2b, 3a; 3b not run (R28).
- The main comparison at the reader's size, MRC form: BERT-tiny (4.4 M) and BERT-mini (11.2 M), pretrained / from scratch / kinds given.
  No MRC row answers the level-3 question it was never asked (X3 0.000); pretrained inner frame 0.97-1.00; from scratch 0.14-0.17 (the
  withdrawal condition is not met: the from-scratch claim stands).
- The reader 1a (3.3 M): depth-3 exact match 0.98-0.99, new formulations 0.83-0.87 (below the rule and BERT). With the R46 decoder and the
  verb-order route variant b (R58; 3.47 M at its used keys, 11.7 M allocated): new formulations 1.000 on the claim set (seeds 5-9);
  variant a (R56) rejected by its own rule. Template 4 at k = 0 with route b + R46: 1.000.
- Refusal (decoder sop @ c126cbe): route b + R46, explained residual, claim set: random 0.984, reference 0.928 rejected at 0.000
  false rejection: the demo criterion is met for those sources on all five claim seeds (also by 1a + R46); threshold 0, every in-world
  sentence fully explained; nonce not re-scored through the tool.
- Query emission: the reader reports unsupported identities 0.98 (MRC 0.66-0.84) and, with R54's rule, undecided roles 0.997-0.999.
- Confirmatory: all pairs Holm-rejected except out-of-world AUROC vs BERT-mini pretrained; (i)-(ii) decided by the untrained question.
- Legend growth: the composed question costs the reader 0 examples; BERT-tiny needs 500 examples and an lr choice; a new form costs the
  reader a 2.3-min retraining. Cost at batch 1 / 1 thread: reader 8.2 ms, BERT-tiny 12.2, BERT-mini 15.8, BERT-base 103.5.
- LLM reference (claude-sonnet-5-5, 10-shot): 0.95-1.00 per axis, depth-3 exact match 0.955, 2.95 USD in total.

Not run by R48 / R51 (each in RESULTS): BERT-base MRC, BERT-small, the sensitivity grid, the 'all' rows, BERT-mini kinds seeds 3-4 and
growth, R30 items 2 / 4 / 6c, the deferred tagger rows, the real-data and multilingual rounds, the trained coverage head of R26.

sop/ EQUAL checks done: every per-role and one-to-one accuracy row EQUAL (labelled sop); the explained residual, re-checked after R64's
fix (0b4878b), R65's (6c2c6ee) and R66's (f1af943), is EQUAL sentence by sentence on the world's sets; on the random out-of-world
sets the tool rejects 0.91-0.92 against 0.99 (verb treatment and occurrence bound). No sentence of the world repeats an identity. Reader growth at
k = 5 / 10 / 20 done (20 examples: 0.98; 50: 1.000). Nothing runs; the experiment has stopped (R48, R62). Threats to validity: RESULTS 'Limitations' (rule-solvable world; claim at 4-11 M; refusal not met; reader changes
designed on the inspected seeds; decoder provenance).

## Earlier entries

- 2026-10-24: repository created; runs 0-1 planned. 2026-10-25 (morning): runs 0, 1a-1c, 2a, 2b, 3a; the R10 rules; the derived-row queue.
