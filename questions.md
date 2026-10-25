# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q7 were answered by R1, R2, R5, R7 and R11 of responses.md, 2026-10-25, and accepted.)

## Q8. Training cost, measured the same way for every model (presentation; no number of any axis changes)

Measured so far on the same GB10, fp32: the reader trains in 134-136 s per seed (run 1a), with no grid. BERT-base (run 2a, in progress)
takes about 247 / 410 / 820 s per grid point at 3 / 5 / 10 epochs, i.e. about 75 min for the 9-point selection on seed 0, then one
configuration per later seed. The cost table currently has parameters, size and CPU ms per sentence, not the cost of training.
Proposal: add, for every model and row, (a) training wall-clock per seed, (b) the wall-clock of model selection (the whole grid on seed
0), (c) their sum per 5-seed row, and (d) GPU energy (power from nvidia-smi sampled during training, integrated), all on this machine
and stated as such. Training cost is then reported beside inference cost, so neither is the only cost shown. Default: add (a)-(c) from
the logs (the per-point times are printed already), and (d) from the next runs on, starting with 2b; no rerun for it. Needed before JP 1.

## Q9. Numerical precision: fp32 everywhere, including the robustness rows

All runs so far train in fp32. bf16 would roughly halve the BERT time, but it changes numbers and would be a deviation; the extra hours
do not matter for this study. Proposal: fp32 for EVERY training row, the pre-registered ones, the R10 rows and the R9.5 wider grid
(20 points on seed 0), so that all rows are comparable with one another and with the training costs of Q8. The fp16 / int8 rows of
R10.13 stay as inference-cost rows only. Default: fp32 everywhere, stated in RESULTS. Needed before the R9.5 grid (after JP 1).

## Q10. Tuning budget: the reader has none in this repository, BERT has a 9-point grid

The reader runs with the main repository's configuration (no grid here); every neural model is tuned on a grid on seed 0. This
asymmetry favours BERT, so it does not weaken the comparison, but it has to be visible. Proposal: state it in RESULTS and in the cost
table (selection cost of the reader = 0 here; its configuration was fixed in the main repository's earlier rounds, cited), and add no
grid for the reader. Default: as proposed. Needed before JP 1.
