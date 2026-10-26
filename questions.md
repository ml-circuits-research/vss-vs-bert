# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29) and R64 (Q30) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q31 (2026-10-26 06:45, experimental session): R65 re-check: the explained residual still differs; the cause is the re-reading loop in sop/infer.py one_to_one, not the scope

With sop @ 6c2c6ee the accuracy rows are equal on every axis and seed; the explained residual differs out of world only: AUROC 0.86-0.89
(transcription 0.965-0.985), reference rejected 0.56-0.61 (0.88-0.95), random 0.89-0.95 (0.99), in-world false rejection 0.015-0.032
(0.015-0.033). Logs: logs/2026-10-26_0622..0625_sop_check_ba631b6.txt; JSONs logs/results_sopcheck_*_demo_vx_6c2c6ee_seed*.json.
**First differing sentence** (as R65 asks): seed 0, reader 1a, reference out-of-world test item 0, "p3 claims that p1 gives o0 ." (depth 1).
Transcription: A not decoded, B o0, C p1, X1 p3, residual 1.0. Tool: A p1, B o0, C p1, speaker p3, frame complete, residual 0.0.
**Cause:** sop/infer.py one_to_one (line 386) rebuilds the exclusion set from the original candidates every round
(`sub = {w for w in filt if w != h ...}`): round 1 re-reads A without p3 (A = p1, collides with C), round 2 re-reads A without p1 only, so
p3 is allowed again; A alternates p3 / p1 and after the four rounds the frame ends with A = C = p1, which R65's occurrence bound forbids
(p1 occurs once). Proposed fix (yours, in sop/): accumulate each role's excluded words across rounds, and after the last round leave a
role that still collides undecoded (frame incomplete -> 1.0). Expected effect: equality with the transcription on these sets.
**Repeated identities (R65):** none in the world: 0 of 78,800 frozen-split sentences (seeds 0-9) and none of the reference, nonce, planted
or growth sentences. Only the random out-of-world sets repeat words (1,704 / 5,000 test, 1,709 / 5,000 negative, 34%), where R65's
occurrence bound and the transcription's "each word once" can differ; to be stated in the paper whichever is kept.
**Default until answered:** the paper's refusal numbers are the tool's at 6c2c6ee (labelled), no refusal claim beyond them; on a synced fix
the residual part is re-run once more. Nothing else waits.
