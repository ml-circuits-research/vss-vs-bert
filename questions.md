# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30) and R66 (Q31) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q32 (2026-10-26 07:05, experimental session): after R66 the residual matches on the world's sets; two points for the paper's refusal numbers

R66 re-check (sop @ f1af943; logs/2026-10-26_0637..0640_sop_check_0dfe5fa.txt): the explained residual is equal sentence by sentence to
the transcription on every calibration, in-world and reference sentence checked (1a seeds 0 and 5, route b seed 1; |diff| < 1e-3). The
paper's refusal rows (decoder: sop @ f1af943) are in RESULTS 'Decoder provenance': route b on the claim set AUROC 0.960, random 0.916,
reference 0.930, false rejection 2.03% (the demo criterion met on 2 of 5 seeds, missed on the mean by 0.03 points).
1. **The calibration threshold is exactly 0.0** on every seed (more than 98% of calibration residuals are 0), so "false rejection" is the
   share of in-world residuals above floating-point noise, which differs between float64 (tool) and float32 (transcription): 0.024 vs
   0.028 (1a seed 5), 0.031 vs 0.017 (route b seed 1). The demo-criterion verdict for route b therefore rests on rounding. A possible
   fix is a fixed numerical tolerance (e.g. residual <= 1e-6 counts as 0) declared before it is applied, or reporting the criterion as
   undecided at this resolution. This is a protocol change, so it is not applied without your answer.
2. **Random out-of-world sets:** tool 0.91-0.92 rejected vs transcription 0.99. 297 of 315 (1a seed 5) and 316 of 336 (route b seed 1)
   differing sentences have two or more verbs: with verbs configured as non-content kinds the tool explains every verb, the transcription
   (R44) explains only the first verb as the event lemma. The other 18 / 20 are R66's occurrence bound. In-world sentences have one verb
   and no repeated identity. The paper states both numbers; to be decided by you whether "verbs explained" should cover only the event
   verb (a change in sop/).
**Default until answered:** the paper uses the tool's numbers as measured, with the threshold 0.0 and both random-set numbers stated;
no tolerance is introduced; the demo criterion for route b is reported as "missed on the mean by 0.03 points, met on 2 of 5 seeds". Nothing
runs.
