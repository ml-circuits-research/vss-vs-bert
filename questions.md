# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35), R84 (Q36), R85 (Q37), R87 (Q38), R88 (Q39), R89 (Q40) and R90 (Q41) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q42 (2026-10-26 21:20, experimental session, close of day per R93): two tool points that wait for tomorrow

Everything of today is in RESULTS (last commit 371bbf5); nothing runs. Two points need sop/ (yours) before the demo export is final:
1. **R92 step 1: the one-config reading reproduces R90's frame but not its refusal.** Frame exact (dispatch 0.966, span F1 0.892,
   abstention 0.297); refusal 0.377 held-out vs 0.460 at the same 2% target. Cause (stated before the run): the `refusal` pass reads with
   the schema's (B) admission sets, while line A used the stored ones, so the one-config refusal is line B's (0.380). Separation unchanged
   (AUROC 0.926 vs 0.925), only the residual's scale moves. Needed: a refusal block that can also restore the stored admission sets.
2. **The typer export (R91) at SNIPS scale:** 56,376 kept rows, 2.8 MB (expected well under 1 MB), and the re-loaded export matches the
   in-memory typer on 0.990 of 200 dev_rep utterances, not 1.000 (int8 quantisation, presumably).
Also recorded: R85 under the paired test (sop 25de418): accepted on seed 0 (17 wins / 7 losses, noise 9.8), rejected on seeds 1-2, 0 wins
on the marked rows in every seed: "need signal fired on one world, route rejected on 2 of 3 seeds". The R84 item 4 / R91 export exists at
logs/export/snips_demo_seed2/ (seed 2: dev dispatch tied 0.9657 in all seeds, best dev span F1 0.8941; 7 worlds x 15.9 MB + typer;
reading.json = R90's two-config reading); the R92 single-config export is declared-but-unrun.
**Default until answered:** nothing runs; tomorrow starts from R93's list.
