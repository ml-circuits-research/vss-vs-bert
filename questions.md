# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50 and R59 (Q29) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q30 (2026-10-25 23:25, experimental session): sop's residual_explained has no "role not decoded -> 1.0" clause

R42 defines the explained residual as 1.0 "when no token is explained or a role of the depth is not decoded"; common/decode.py does so;
sop/infer.py residual_explained returns 1.0 only when nothing is explained. With the sentence-candidates decoder this decides the
reference-world rejection (RESULTS 'Decoder provenance'): 0.88-0.95 with the clause, 0.01-0.04 without. **Default until answered:** the
tool is the reference; the paper uses the tool's numbers; the transcription's rows are kept, labelled as differing. Nothing waits on it.
