# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q26 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40 and R45 of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44.)

## Note for the author (2026-10-25 17:15, experimental session): H1 / H2 / H3 diagnosis of the reader on new formulations (design set)

Source: results/dump_reader_basic_seed0-4.json (1a), results/dump_reader_basic_cues_seed0-4.json (R14), every seed EQUAL to its stored
row; eval/diagnose_reader.py (patterns fixed in 175b473 before the dumps were read), logs/2026-10-25_1705_diagnose_reader_eb89dbb.txt,
results/diagnosis_reader_basic*.json; the cell breakdown below from the same dumps. New-formulation report rows only (1,800 items, 5,400
role decisions per seed). Seeds 0-4 are now the inspected design set (R39). R37 is not yet run.

Wrong role decisions per seed, by (template, role) cell (the cells carry the mechanism; "absent" = the decoded filler is an identity not in
the sentence; "swap" = another role's filler):

| row, seed | open gates | wrong | T4:C | T7:A | T10:A+C | T4:B + T11:B | rest |
|---|---|---|---|---|---|---|---|
| 1a s0 | . by that to | 418 | 166 | 173 | 71 | 0 | 8 |
| 1a s1 | . that to | 563 | 174 | 171 | 195 | 0 | 23 |
| 1a s2 | by from that to | 934 | 158 | 177 | 245 | 305 | 49 |
| 1a s3 | . by that to | 431 | 178 | 169 | 51 | 0 | 33 |
| 1a s4 | . by that to | 526 | 160 | 167 | 172 | 0 | 27 |
| R14 s0-4 | . by claims from says that to | 933 / 993 / 882 / 888 / 996 | 156 / 176 / 134 / 174 / 160 | 178 / 170 / 165 / 173 / 176 | 254 / 264 / 262 / 198 / 294 | 305 / 309 / 248 / 280 / 304 | 40-74 |

Pattern counts of eval/diagnose_reader.py (not exclusive), 1a seeds 0-4: H3 (gold ranked 2nd and recovered by the one-to-one rule) 175 /
206 / 50 / 162 / 198; H2 (cell wrong on every seed) 339 / 345 / 335 / 347 / 327; H1 (cell wrong on some seed, right on another, with a
cue gate differing) 0 / 0 / 305 / 0 / 0; none 32 / 154 / 289 / 33 / 134. R14 seeds 0-4: H3 44 / 28 / 36 / 38 / 30; H2 678 / 705 / 644 /
697 / 687; none 190-306. Median margin of wrong decisions 0.010-0.024 against 0.25-0.29 for right ones (both rows). Gold candidate rank
on errors, 1a: 2nd in 36-72% (seed 2: 21%); R14: 2nd in 16-24%. 50-item confident-error sample (tau 0.000, so every error is confident):
1a 28 wrong role (swap) / 22 candidate not in the sentence; R14 12 / 38; missed role and false fill cannot occur with the depth given.

What is observed, cell by cell:
- **T4:C** ('{A} V {C} {B} .', the recipient right after the verb without 'to'; new transition verb>P): wrong on every seed of both rows
  (cell error 0.74-0.99); in 1a 812 of 836 errors are swaps with A, gold ranked 2nd in 730. Independent of the gate set: H2-consistent
  (the position "P right after the verb" is not encoded as C by any route; its deposit lands nearest to A).
- **T7:A** ('it is {B} that {A} V to {C} .', the agent after a cleft 'that'): wrong on every seed of both rows (0.92-0.99); 807 of 857
  errors decode an identity absent from the sentence (gold ranked >= 4th). Independent of the gate set as far as these readers vary,
  but 'that' is open in every reader (it is the level opener), so this data cannot separate "routes" from "the one 'that' gate opens a
  level address here too": consistent with H2 and with a fixed-gate form of H1; not separable on the design set.
- **T4:B + T11:B** (the object last, right after a P): wrong in exactly the readers whose 'from' gate is open (1a seed 2 and all five
  R14 seeds, cell error 0.46-0.99) and right in all four readers with 'from' closed (0.000), including T4, which contains no 'from'. So
  the effect is of the gate SET on what the routes learn in training, not of a token's gate at reading time: H1-consistent in that form.
  This cell pair is the largest part of R14's loss against 1a (about 290 of the about 360 extra errors per seed).
- **T10:A+C** ('from {A} , {C} VR {B} .', A/C swaps and absent fillers): 245-294 with 'from' open, 51-195 with it closed; seeds 0, 3, 4
  have the same open set and give 71 / 51 / 172, so the gate set does not determine it: mixed, partly seed noise in the routes.
- **H3** (decoding): errors are low-margin and, in 1a, often near misses (the swaps of T4:C and T10); the one-to-one rule recovers
  50-206 decisions per seed (+0.01-0.03 slot accuracy, RESULTS) and in R14 near misses are rare (28-44). No cell is explained by
  decoding alone; it is the symptom of the swaps, not a separate cause.

Reading, stated as observed: the evidence favours **H2** for the shared core of the loss (T4:C and T7:A: about 330-345 errors on every
1a seed and the same with the gates fixed, i.e. 60-80% of 1a's errors on seeds 0, 1, 3, 4), and **H1 in the form "opening 'from' in
training breaks the object-last reading"** for the seed-dependent part and for R14's drop (T4:B + T11:B appear if and only if 'from' is
open). **H3** is not supported as a cause. T7:A's mechanism (routes vs the single opener gate of 'that') is not separable with these
readers. Prediction for R37 written now (it opens 'from' as a trainable initialisation): if T4:B + T11:B errors appear on its seeds,
the R14 loss comes from the set ('from' open), not from the hardness; if they do not, hardness matters. No design of a fix here.
