# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q23 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34 and R35 of responses.md, 2026-10-25, and accepted.)

## Q24. R38: the error dump is taken from the REPORT rows, so an improved reader designed from it cannot be scored on those same rows

R38 asks for per-item dumps of 1a and R14 on the new-formulation REPORT rows, read to design an improved reader, which then "runs here
as above", i.e. on the same frozen splits. Reading the errors of a report set and then scoring a model designed from them on that same
set is the post hoc pattern that rule 25 and R9 guard against, even if no number is tuned: the design has seen the test items (which
templates, which transitions, which gate fails where). A reviewer would discount any gain on those rows.
Default (keeps R38's intent, removes the objection): (1) the dumps are made as R38 asks, from the five seeds' report rows, and the JOURNAL
entry names exactly what was looked at; (2) BEFORE the dump is read, fresh frozen evaluation splits are drawn and committed for the
axes the improvement targets (new formulations seen / unseen transitions, plus depth 3 and new ids as controls), with new identities
and a new rng, never shown to anyone: the R13 splits basic_seed5-9.json already serve this, generated now, before the dump is read;
(3) the improved reader's CLAIM is made on those fresh splits, against 1a and the BERT rows run on the same fresh splits (the equal-size
models, as R13 already plans); (4) its numbers on the inspected seeds 0-4 are reported too, labelled "design set (inspected)".
The templates are shared between old and fresh splits (the generator has only these forms), so state that the held-out generalisation
is over identities, draws and transitions, not over new templates. Needed before the dump is read.

