# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35), R84 (Q36), R85 (Q37), R87 (Q38), R88 (Q39) and R89 (Q40) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q41 (2026-10-26 19:10, experimental session): R88 A/B/C measured; demo reading = C by R88's rule, at a cost in refusal; three points

R88 / R87 written up (RESULTS, 58a794c, d68d4cc; logs 2026-10-26_1610.._1627, code 13c1fec). Base test, anchor dispatch 0.966 in every line:

| line | span F1 | head-frame exact | abstains | held-out refused at 2% FR (test FR) |
|---|---|---|---|---|
| R81 | 0.824 | 0.565 | 0.528 | 0.183 (0.008) |
| R87 B (contract + kind_run) | 0.833 | - | 0.653 | 0.170 |
| A: in-context kinds | 0.826 | - | 0.422 | **0.460** (0.016) |
| B: + admission from in-context kinds | 0.870 | 0.636 | 0.299 | 0.380 (0.017) |
| C: + presence_by_kind | **0.892 +- 0.001** | **0.699** | 0.298 | 0.267 (0.017) |

Typer alone 0.886. False fills / misses per seed (all worlds): R81 224 / 165, A 193 / 177 (expectation "sibling fills fall by most" not
met: admission sets still wide), B 36 / 223, C 54 / 121 (as expected). Under C, 6 of 7 worlds at or above the typer alone; BookRestaurant
0.911 vs 0.921 (R81 0.731). Presence decided by: no admitted kind 10176, unique kind 4647, threshold (shared kinds) 1077. C's residual
AUROC 0.907 (R81 0.941). R87's line B abstains 0.653, not at or below C's: expectation not met.
1. **Demo reading.** By R88's rule (best in-domain span F1 at unchanged dispatch) it is C; but C refuses held-out intents least of the
   three (0.267 vs B 0.380, A 0.460 at 2% FR). On ood_intents in-domain B is marginally better (0.859 vs 0.853). Keep C as the rule says,
   or weigh refusal?
2. **R85 AddToPlaylist, seed 0** (corrected: verb kinds ['w:add']): accepted under the 82585ea acceptance rule (FULL 0.528 -> 0.571 on
   417 selection rows) but 0/4 on the four same-kind-competition rows, so the gain is not on the rows the route is for. Seeds 1-2 run under
   the same pinned rule. sop 2f3806f (SOURCE 6c735f9) made the acceptance a paired test; should seed 0's acceptance be re-run under it
   (~15 min GPU; the trial models are not kept)?
3. **run.sh**: its post-launch guard appended a "# post-launch guard" line into two old logs (0953 / 0955) at 16:29 (`tee -a "$out"`
   picked the wrong file); restored to their committed version. A fix in run.sh is a protocol-file change, so not made without you.
**Defaults until answered:** demo reading C (R88's rule); seeds 1-2 of R85 under the pinned rule; no run.sh change; item 4 (the export)
waits for your go and for the typer export in sop.
