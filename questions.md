# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q25 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39 and R40 of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44.)

## Q26. Two execution changes to shorten the equal-size block (no number changes; not applied until accepted) (main session, 16:55)

Measured at 16:51: BERT-tiny MRC takes about 19 min per seed (1,158 s, seed 1 of the from-scratch row) with six jobs sharing the GPU,
and each block job runs its five seeds one after another, so a BERT-tiny job lasts about 1.6 h and a BERT-mini job, 2-3x slower per
seed, about 4 h; mini from scratch and mini with kinds have not started (slots full). Expected end of the block at this pace: about
22:00-23:00, JP 1 about 01:00-02:00 (estimates). The GPU reads 92% "utilisation" but draws only about 34 W and memory use is 22 of 119
GB: as measured in Q17, the small models are launch-bound, not arithmetic-bound, so more concurrent processes can still add throughput.

(a) **One job per seed** instead of one job per row with five sequential seeds: each (model, variant, seed) becomes its own run.sh job
with its own log; a row is the five per-seed logs, aggregated by eval/compare.py as now. The per-seed numbers are identical (seeds are
fixed per process, fp32, determinism per process as R22 states); only the scheduling changes. Jobs already running finish as they are.
(b) **More slots for the small models**: raise the cap from 6 to 12 for BERT-tiny / BERT-mini jobs only, with R23's memory rule
unchanged (VSA_MEM_FRACTION = 0.9 / slots, 40 GB post-launch guard), after one measurement as R23 prescribes: samples per second of 12
concurrent tiny jobs against 6; keep 12 if the total gain is at least 1.3x, else stay at 6. BERT-base stays alone. Each run keeps one
CPU core (20 cores available).
Cost rows: per-seed training times from concurrent runs are already marked "concurrent (N runs)" and are not used as cost numbers
(R22 / R23); nothing changes there.
Default: neither (a) nor (b) until accepted; the queue continues as it is. Needed as soon as possible (it only helps for the jobs not
yet started: mini scratch, mini kinds, the 'all' rows, BERT-base MRC is unaffected).

