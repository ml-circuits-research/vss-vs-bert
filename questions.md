# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q16 were answered by R1, R2, R5, R7, R11, R12, R13, R18 and R20 of responses.md, 2026-10-25, and accepted.)

## Q17. R22 item 3 (three concurrent runs): two points to check before relying on it

(1) **Speed.** There is one GPU. During 2a / 2b it ran at 89-94% utilisation with one BERT-base run, so three runs time-share the same
SMs: the expected gain in total throughput is small (perhaps 1.1-1.5x), not 3x, and every run's wall-clock and energy become harder to
read. (2) **Memory and the hang the author has seen on this machine** (memory not released, a forced restart needed). run.sh caps each
process's CUDA allocator at VSA_MEM_FRACTION 0.5 of 120 GB; three processes at 0.5 can claim 150% of the unified memory, which is the
condition under which the GB10 has hung before (main repository, 2026-10-15).
Default: implement the slots as R22 says, with VSA_MEM_FRACTION 0.25 per process when more than one slot is in use and the 40 GB
post-launch guard; then measure once, on the first derived rows: samples per second of 1 run alone against 2 concurrent runs (same
script, same seed range). If 2 concurrent runs give less than 1.3x the throughput of one, go back to one slot and say so in RESULTS;
never 3 slots unless 2 gave the gain. Needed before the first derived row (after 3b).

