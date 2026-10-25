# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q24 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35 and R39 of responses.md, 2026-10-25, and accepted.)

## Q25 (2026-10-25, experimental session): the 'all six questions' MRC rows (Q18 (1b)) in the equal-size block

R35 left the 'all' rows (all six questions in training, level 3 always NULL) waiting for a response; the handover queue mentions "the
all-questions row as declared in R29". **Default until answered:** the block runs the 'zs' rows only (the main row); the 'all' rows run
for BERT-tiny and BERT-mini pretrained (two rows, 5 seeds, same fixed configuration) only after an explicit "Q25: yes"; otherwise they are
recorded as not run with this reason. Needed before: the end of the 'zs' block.

## Note (2026-10-25, transparency, no decision needed): execution changes that alter no number

Commit 33dc482: bert_mrc.py prints one progress line per epoch (the 30-min log-stall watchdog of run.sh killed the BERT-tiny 5e-5 grid
chain), `--save_all` (every seed's model saved, for R30 item 8's Mahalanobis and matched coverage), `--seed_start` (BERT-mini rows split
into two runs under the 6 h cap); data/make_splits.py `--seed_start` (seeds 5-9 without rewriting 0-4). RESULTS 'Execution notes for the
equal-size block'.
