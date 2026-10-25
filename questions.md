# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q16 were answered by R1, R2, R5, R7, R11, R12, R13, R18 and R20 of responses.md, 2026-10-25, and accepted.)

## Q17. R22 item 3 (three concurrent runs): two points to check before relying on it

(1) **Speed (corrected by the main session after a measurement, 13:45).** nvidia-smi's 89-94% "utilisation" is the share of time a
kernel is running, not the share of the GPU's arithmetic in use. Measured during 2a / 3a: one Python process at ~105% CPU (one of 20
cores), and BERT-base at about 110 ms per step of batch 32 x ~16 tokens, i.e. roughly 2-3 TFLOP/s in fp32, a small fraction of the GB10's
fp32 peak: small matrices (512 tokens per step) and one-thread kernel launches, not arithmetic, set the speed. So concurrent runs CAN give
a real gain here, and they are the only speed-up that changes no number (a larger batch, TF32 or torch.compile would each change numbers
or the fixed batch 32 of INSTRUCTIONS section 4, and are not proposed). (2) **Memory and the hang the author has seen on this machine** (memory not released, a forced restart needed). run.sh caps each
process's CUDA allocator at VSA_MEM_FRACTION 0.5 of 120 GB; three processes at 0.5 can claim 150% of the unified memory, which is the
condition under which the GB10 has hung before (main repository, 2026-10-15).
Default: implement the slots as R22 says, with VSA_MEM_FRACTION 0.25 per process when more than one slot is in use and the 40 GB
post-launch guard; then measure once, on the first derived rows: samples per second of 1 run alone against 2 concurrent runs (same
script, same seed range). If 2 concurrent runs give less than 1.3x the throughput of one, go back to one slot and say so in RESULTS;
3 slots only if 2 gave the gain and memory stays above the guard. Needed before the first derived row (after 3b).

