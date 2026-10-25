# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q5 were answered by R1, R2, R5 and R7 of responses.md, 2026-10-25, and accepted.)

## Q6. R10 item 4 (the 'whole' initialisation) and the R10 rows of run 2

R10's Order puts item 4 with "3 / 4 / 5 / real round", but the 'whole' tokenizer condition exists only in run 2 (2b); runs 3-5 are
WordPiece. The R10 rows of run 2 include "both tokenizer conditions", so the R10 'whole' row is the only place item 4 can apply.
Default: the R10 'whole' row of run 2 uses item 4 (each new token initialised as the mean of its WordPiece pieces' embeddings), behind
the flag `--whole_init pieces` (default `resize`, the tagged mean-resizing behaviour, which RESULTS will state for 2b). Say if item 4
should instead stay out of run 2.

## Q7. R10 item 1 for runs 1b / 1c, and margins under the one-to-one rule

(a) The 7b / 7b-bis transformers of runs 1b / 1c were not saved (no --save_model in models/transformer_7b.py), so the one-to-one decoding
cannot be applied to them without retraining. Default: 1b / 1c keep the per-role rule only, stated in RESULTS; the one-to-one rule is
applied to the reader (1a, seed 0 from the saved route book), to 2a (seed 0 from its saved model) and to every later run (all seeds,
computed in the run). (b) Under the one-to-one rule the filler of a role need not be the role's top-1 candidate. Default margin: the
assigned candidate's score minus the best score of any OTHER candidate for that role (it can be negative, which then marks the decision
as unsure); the per-role rule's margins are unchanged. Calibration blocks are reported for both rules.
