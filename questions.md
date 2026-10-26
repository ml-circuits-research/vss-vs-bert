# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32) and R79 (Q33) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q34 (2026-10-26 12:45, experimental session): the per-intent dispatcher is an untrained argmin over uncalibrated residuals; three options, cheapest first

Preliminary R79 seed 0 (logs/2026-10-26_1145/1146_run_reader_real_dd1da21.txt; the worker's full report follows): dispatch intent 0.716
(R75 0.595; reference one world 0.977), chosen frame abstains 0.601, held-out rejected 0.020, in-domain false rejection 0.000-0.003. By R79's
own rule (dispatch < 0.9) R78 (1) goes to the one-world configuration. The question is what the per-intent configuration needs, since it
is the one the demo is to show.
What the dispatcher is now: not a vote and not trained; argmin over the 7 worlds' explained residuals, ties by the min margin. Each world
was trained only on its own intent's utterances, so (hypothesis, not measured) its residual is calibrated on nothing outside it, and the
residuals of worlds of different legend size / lexicon coverage are on different scales. The one-world reader's closed-value intent role
is in effect a gate trained discriminatively on all utterances, which is why it reaches 0.977.
Options (none implemented; each would be a declared row on seeds 0-2):
1. **Calibrated residuals, no training (CPU, saved R79 models):** replace each world's raw residual by its percentile in that world's own
   calibration distribution (or a z-score), then argmin; rejection unchanged (every world above its own threshold). Tests the scale
   hypothesis directly; minutes.
2. **One-vs-rest training of the worlds:** each world also sees the other intents' training utterances as negatives (its frame must not
   explain them: e.g. a loss term pushing their explained share down), so the residual is trained to discriminate. This keeps routing by
   fit and the refusal-by-no-world property; costs one retraining per world.
3. **A learned gate** (the R79 hybrid, or a small classifier over the 7 worlds' residual / margin vectors): this is a top-1 mixture of
   experts with independently trained experts. It should recover the intent accuracy, but the refusal then depends on the gate having a
   reject option, and the "no world explains it" argument becomes a gate decision.
The session's recommendation: 1 first (it decides whether the problem is scale or discrimination), then 2 if 1 is not enough; 3 is the
fallback R79 already names. **Default until answered:** nothing beyond R79's own rule runs.
