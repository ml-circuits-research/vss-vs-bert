# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q10 were answered by R1, R2, R5, R7, R11 and R12 of responses.md, 2026-10-25, and accepted.)

## Q11. Seed count: more seeds for precision on the axes that vary (never to pick a seed)

Run 1a shows that the reader is insensitive to the seed on composition (seen, new ids, depths 1-3: std <= 0.002; FULL at depth 3
0.982 +- 0.010) but not on new formulations: seen transitions 0.864 +- 0.069 (per seed 0.93 0.82 0.77 0.95 0.84), unseen transitions
0.868 +- 0.040. The spread follows the learned gates: seeds 0, 3, 4 open {'.', by, that, to}; seed 1 lacks 'by'; seed 2 lacks '.' and opens
'from', and is the lowest (0.77 / 0.79). The main repository's 7c shows the same spread (0.848 +- 0.064). With 5 seeds the 95% t
interval on that axis is about +-0.09, too wide to separate two models closer than that; BERT at 1k on WikiSQL is expected to vary as
much (INSTRUCTIONS section 9). A seed is a data split and an initialisation together (basic_seed<s>.json).

Proposal, as a stated additional analysis after JP 1 (the pre-registered 5-seed rows stay as reported):
(a) seeds 5-9 for EVERY model of runs 1-3 (reader, 7b-bis, 7b, BERT-base both tokenizer conditions, DistilBERT pretrained and
scratch, pre-registered and R10 rows), with frozen splits basic_seed5-9.json made by data/make_splits.py and committed before any model
sees them, each model's configuration fixed at the one selected on seed 0 (no new grid); report 5 and 10 seeds side by side;
(b) the per-seed gate sets of the reader printed beside its per-seed new-formulation numbers (already in the logs), and for BERT the
per-seed list, so the source of the variance is visible for every model;
(c) never: choosing a seed, dropping a seed, or tuning the reader's gates on report rows (rules 25, R9.2). If a variance-reduction
change of the reader is wanted (for example a gate decision averaged over restarts, chosen on calibration rows only), it is a NEW model
row, declared before its run, and BERT gets the matching treatment (an ensemble of the same size), never instead of the pre-registered row.
Cost on this machine: about 2 min per seed for the reader, about 4-14 min per seed per BERT row (no grid).
Default: (a) and (b) after JP 1; (c) not run unless you ask. Needed at JP 1.
