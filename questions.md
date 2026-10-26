# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33) and R80 (Q34) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q35 (2026-10-26 13:15, experimental session): R80 measured; the anchor-percentile dispatch reaches 0.863, under 0.9; option 2 to be declared

R80 (eval/r80_dispatch.py, code 52c5718, saved R79 models, CPU, seeds 0-2, both conditions; RESULTS, last commit 60e27e1). Test:

| rule | intent | span F1 | residual AUROC |
|---|---|---|---|
| residual raw (= R79) | 0.731 | 0.720 | 0.747 |
| residual percentile | 0.659 | 0.668 | 0.730 |
| **anchor percentile** | **0.863 +- 0.011** [0.861 0.851 0.877] | 0.802 | 0.849 |
| anchor, then residual | 0.854 | 0.794 | 0.850 |
| sum of percentile ranks | 0.855 | 0.784 | 0.797 |
| structure percentile (added: differs from the anchor in 3 worlds whose core also has a slot) | 0.817 | 0.785 | 0.792 |

Rejection: the residual rule rejects 0.020 of held-out intents; the second line "every world's anchor percentile < 2%" rejects
0.280 +- 0.022 at 0.038 in-domain false rejection (ood_intents test; 0.047 on base test). Head-frame exact 0.462 -> 0.535.
Against R80: "well above 0.716" met; "anchor alone >= 0.9" not met on any seed. By R80's rule option 2 (one-vs-rest negatives in
training) is next, to be declared by you; option 3 stays the fallback. R78 (1) has not started.
**Default until answered:** nothing runs; R75 and R79 world folders are kept.
