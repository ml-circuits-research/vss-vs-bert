# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q25 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39 and R40 of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40.)

## Note (2026-10-25, transparency; proceeding as declared, objections welcome): seeds 5-9 data for R43's claim

R43 makes the AUROC on seeds 5-9 decide, but seeds 5-9 have no reference-world splits and no out-of-world sets yet, and no 1a reader.
Proceeding, as declared in RESULTS 'Fresh seeds 5-9 for the reader 1a and its out-of-world sets': the same generators with
`--seed_start 5` (reference splits -> out-of-world sets, committed with sha256 before any model sees them) and 1a trained on the fresh
splits (R13's seeds 5-9 for the reader; tagged apart). One stated adaptation in residuals_explained: the basic-world decoder emits no
event lemma, so the first token of an event-verb kind is taken as the lemma. If either is not what you meant, say so; the 1a seeds 0-4
row (design set) runs meanwhile.
