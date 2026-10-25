# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1 and Q2 were answered by R1 and R2 of responses.md, 2026-10-25, and accepted.)

## Q3. Which main-repository row is run 1a's reference (exit check)?

RUNBOOK run 1a names RESULTS 6c (new ids 1.000, depth 3 1.000). But vss/run_reader_basic.py runs v8_gates with gates LEARNED from the
closed start, which is the main repository's RESULTS 7c row "bundle, learned, start closed" (depth 3 0.996, A 0.986, C 0.991); 6c's
1.000 is the hand-gate row. Run 1a gives new ids 1.000 and depth 3 0.997 ± 0.002 (A 0.990, C 0.989): within seed variance of 7c
learned-closed on every column, 0.003 below the 6c hand-gate row at depth 3 (RESULTS 1a). Default: the reference is 7c learned-closed,
the gate is MET, and runs 1b / 1c proceed (they do not depend on 1a). I do not start run 2 until you answer. If you want the 6c
hand-gate row as the reference, the reader would have to be run with hand gates (a different model from the one the script names).
