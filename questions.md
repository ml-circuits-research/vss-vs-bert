# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q15 were answered by R1, R2, R5, R7, R11, R12, R13 and R18 of responses.md, 2026-10-25, and accepted.)

## Q16. R17 item 3: a [CLS] depth head trained on depths 0-2 cannot output depth 3

Training covers depths 0-2 only, so a depth CLASSIFIER on [CLS] has no trained class for 3: the same "untrained label" situation as X3
in R6, here for the decision that selects which roles are read. The reader's level-opener count gives 3 for a depth-3 sentence
directly. Default: a classification head over 0-3 (class 3 untrained), reported as such, beside a second BERT variant that is a
REGRESSION head on [CLS] (one output, squared error on the depth, rounded and clipped to 0-3 at decoding), which can extrapolate; both
on the same 5 seeds, 2a configuration. Stated in RESULTS before the rows run. Needed before the R17 rows (after run 3).
