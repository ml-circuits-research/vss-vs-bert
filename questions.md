# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35), R84 (Q36), R85 (Q37), R87 (Q38) and R88 (Q39) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q40 (2026-10-26 18:30, experimental session): R88 line (B) is in; (C) `presence_by_kind` can be added to sop

As R88 asks ("say when (B) is in"): lines (A) and (B) are declared with their R83 paragraphs and running (sop @ 4a0eba8, presence_query,
`spans: kind_run`, token_kinds = the typer's in-context tags; RESULTS, 76a62b0). Smoke test of (B): the recomputed admission sets shrink
to 1-3 kinds per question (restaurant_name 19 -> 1, movie_name 35 -> 1, city 16 -> 1). Results in about 30 minutes.
Also measured (R84 item 3, b467e4b, base test): presence_query alone lowers chosen-frame abstention 0.528 -> 0.450 (17.5% of
utterances carry a presence query), dispatch 0.966 and span F1 0.824 unchanged (expectation met); the full contract raises it to 0.821
(Q38's cause; the R87 kind_run re-score is running). Two runs of this session were invalid and are recorded in RESULTS: the R87
re-score dispatch runs crashed on a field-name clash (fixed, relaunched); the R85 AddToPlaylist activation profiled training rows that
included R81's negatives (need share 1.2% < 2%, no trial; fixed to positives only, own tag r85act, relaunched).
**Default until answered:** (C) is re-scored as soon as `presence_by_kind` is announced in sop/SOURCE.
