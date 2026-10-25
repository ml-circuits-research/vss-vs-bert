# current status (rewritten at every judgment point)

## 2026-10-25 22:40 — JP 1 (the experiment stops here by the owner's rule R48)

What was measured (RESULTS 'JP 1: synthesis' holds both tables, the confirmatory results, the regime statements and the limitations):
- Pre-registered rows (tag prereg-2026-10-25): 0, 1a, 1b, 1c, 2a, 2b, 3a; 3b not run (R28).
- The main comparison at the reader's size, MRC form: BERT-tiny (4.4 M) and BERT-mini (11.2 M), pretrained / from scratch / kinds given.
  No MRC row answers the level-3 question it was never asked (X3 0.000); pretrained inner frame 0.97-1.00; from scratch 0.14-0.17 (the
  withdrawal condition is not met: the from-scratch claim stands).
- The reader 1a (3.3 M): depth-3 exact match 0.98-0.99, new formulations 0.83-0.87 (below the rule and BERT). With the R46 decoder and the
  verb-order route variant b (R58, 11.7 M): new formulations 1.000 on the claim set (seeds 5-9); variant a (R56) rejected by its own rule.
- Refusal: no model meets the demo criterion on the claim set; the best row (route b + R46, explained residual) rejects 0.99 / 0.93 / 1.00
  at 2.1% false rejection (design set 1.8%).
- Query emission: the reader reports unsupported identities 0.98 (MRC 0.66-0.84) and, with R54's rule, undecided roles 0.997-0.999.
- Confirmatory: all pairs Holm-rejected except out-of-world AUROC vs BERT-mini pretrained; (i)-(ii) decided by the untrained question.
- Legend growth: the composed question costs the reader 0 examples; BERT-tiny needs 500 examples and an lr choice; a new form costs the
  reader a 2.3-min retraining. Cost at batch 1 / 1 thread: reader 8.2 ms, BERT-tiny 12.2, BERT-mini 15.8, BERT-base 103.5.
- LLM reference (claude-sonnet-5-5, 10-shot): 0.95-1.00 per axis, depth-3 exact match 0.955, 2.95 USD in total.

Not run by R48 / R51 (each in RESULTS): BERT-base MRC, BERT-small, the sensitivity grid, the 'all' rows, BERT-mini kinds seeds 3-4 and
growth, R30 items 2 / 4 / 6c, the deferred tagger rows, the real-data and multilingual rounds, the trained coverage head of R26.

Open: sop/ EQUAL checks (R58 item 1, unblocked by R59) are running and are reported in RESULTS when done; questions.md has no open
blocking question. Threats to validity: RESULTS 'Limitations' (rule-solvable world; claim at 4-11 M; refusal not met; reader changes
designed on the inspected seeds; decoder provenance).

## Earlier entries

- 2026-10-24: repository created; runs 0-1 planned. 2026-10-25 (morning): runs 0, 1a-1c, 2a, 2b, 3a; the R10 rules; the derived-row queue.
