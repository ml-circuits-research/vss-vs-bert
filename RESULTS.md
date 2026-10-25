# RESULTS — one numbered section per run (command, commit, log, table with mean ± std and the per-seed list, calibration of rule 18, label, reading)

Sections are added in the order of INSTRUCTIONS_2026-10-24.md section 7. A result in the in-between band of
Table 3 is reported with both readings. Reference numbers from the main repository (for the exit check of run 1): RESULTS 6c reader
new ids 1.000, depth 3 1.000; 7b transformer new ids 0.215 ± 0.012, depth 3 0.390 ± 0.024; 7b-bis 0.253 ± 0.034, 0.393 ± 0.023;
8c WikiSQL reader sel 0.816 / 0.760 / 0.632 at 56k / 5k / 1k, transformer 0.543 / 0.510 / 0.415, BoW+LR 0.499 / 0.502 / 0.475.

## 0. Frozen splits of the basic world (data/make_splits.py; logs/2026-10-25_0748_make_splits_1617108.txt)

Command: `./run.sh data/make_splits.py --seeds 5`, commit 1617108, on the DGX (NVIDIA GB10, Python 3.12.3, torch 2.14.1+cu130).
Result: seeds 0-4, 7,880 sentences each; gold tags map to gold frames at 1.000 on every seed (exit check MET). Axes per seed: seen 960,
new ids 960, new form / seen transitions 360, new form / unseen transitions 1,080, new form / role cues seen 360, depth 1 720, depth 2
seen 720, depth 3 unseen 720, calibration 2,000; vocabulary 66. Label: MEASURED.

The second exit check of RUNBOOK run 0 ("`git diff --stat data/` shows no change") is met on content, not on bytes: the regenerated
files have the same JSON content as the committed ones on all five seeds (parsed objects equal), but the key order of one mapping (word
-> category) differs, because it is built from a Python set whose iteration order depends on the per-process string hash. The committed
files were therefore kept unchanged (`git checkout data/`), and every model reads them. Not a deviation of the splits; recorded so that
the byte diff is not mistaken for a change of data. From commit c8e4e96 on, every run has PYTHONHASHSEED=0 (responses R3); run 0 ran
without it and its content was verified equal to the committed files.

## 1a. The VSS reader on the frozen basic-world splits (vss/run_reader_basic.py; logs/2026-10-25_0750_run_reader_basic_fade9d5.txt)

Command: `./run.sh vss/run_reader_basic.py --seeds 5 --save_model`, commit fade9d5 (code identical to c8e4e96 except run.sh's
PYTHONHASHSEED line), D 1024, N 30,000, batch 8, lr 0.02, device cuda. This run started before responses R3 was read, so it ran
**without PYTHONHASHSEED=0**. The code reading of R3 (JOURNAL 2026-10-25) found no index, id or random draw in the reader's path that
depends on set or hash order (every category or vocabulary list is sorted before use), so the hash seed cannot change its numbers; it is
not repeated. Model: v8_gates.V8, codes path + chainL + bigram, gates learned from the all-closed start (the main repository's RESULTS
7c row "bundle, learned, start closed"). Learned OPEN gates: seeds 0, 3, 4 {'.', by, that, to}; seed 1 {'.', that, to}; seed 2 {by, from,
that, to}.

| axis (mean role accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new ids | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., seen trans. | 0.864 ± 0.069 | 0.93 0.82 0.77 0.95 0.84 |
| new form., UNSEEN trans. | 0.868 ± 0.040 | 0.89 0.89 0.79 0.88 0.89 |
| new form., ROLE CUES seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 1 | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 2 (seen) | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 3 (UNSEEN) | 0.997 ± 0.002 | 1.00 0.99 1.00 1.00 0.99 |
| FULL frame at depth 3 (UNSEEN) | 0.982 ± 0.010 | 0.982 0.971 0.999 0.986 0.971 |

Depth 3 per role: A 0.990 ± 0.007, B 1.000, C 0.989 ± 0.006, X1 1.000, X2 1.000, X3 1.000.
Calibration (rule 18; tau set on the calibration rows): new ids tau 0.000, coverage 1.000, confident-wrong 0.000 (per seed 0.000 x5),
accuracy among answered 1.000; depth 3 tau 0.000, coverage 1.000, confident-wrong 0.003 (per seed 0.003 0.006 0.000 0.003 0.006),
accuracy among answered 0.997. Cost: 3.32 M parameters (phases), route book 13.3 MB at 32 bits (1.7 MB at 4 bits), 9.0-12.5 ms per
sentence on CPU. Label: MEASURED.

Exit check. The variant run is the main repository's RESULTS 7c "bundle, learned, start closed" (5 seeds): seen 1.000, new ids 1.000,
seen trans. 0.848 ± 0.064, UNSEEN trans. 0.870 ± 0.038, role cues 1.000, depth 1 / 2 1.000, depth 3 0.996 (A 0.986, C 0.991). Every
column here is within one seed std of it: MET. RUNBOOK run 1a names RESULTS 6c (1.000 / 1.000) as the reference; 6c is the hand-gate
row of the same model (7c "bundle, hand gates (fixed)", depth 3 1.000 on all six roles). Against it, new ids is met and depth 3 is 0.003
lower (0.997 ± 0.002), the same A / C shortfall that 7c reports for the learned gates. The reference was settled by responses R5
(2026-10-25): 7c learned-closed is the reference of this run (the RUNBOOK's pointer to 6c was corrected), the gate is MET, no
hand-gate variant is run.

Reading: the reader reproduces the main repository on the frozen splits: new identities and the unseen depth 3 at 0.997-1.000 role
accuracy, 0.982 FULL frame at depth 3, with near-zero confident-wrong at full coverage; new formulations stay at 0.86-0.87, as in 7c.

## 1b. Transformer control 7b-bis, word categories at the input (models/transformer_7b.py --variant cat; logs/2026-10-25_0809_transformer_7b_c8e4e96.txt)

Command: `./run.sh models/transformer_7b.py --variant cat --seeds 5`, commit c8e4e96 (PYTHONHASHSEED=0), device cuda. Model: the
main repository's v22 TModelCat (0.80 M parameters), objective -cos(S, S_world), decoded with the world's addresses on the frozen splits.
Grid on seed 0 by validation cos of the training distribution (lr 1e-3 / 3e-4, dropout 0 / 0.1: 0.918 / 0.857 / 0.963 / 0.859):
chosen lr 3e-4, dropout 0; validation cos per seed 0.963 0.963 0.923 0.958 0.962.

| axis (mean role accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new ids | 0.251 ± 0.048 | 0.22 0.22 0.27 0.34 0.21 |
| new form., seen trans. | 0.821 ± 0.133 | 0.89 1.00 0.88 0.67 0.67 |
| new form., UNSEEN trans. | 0.786 ± 0.126 | 0.68 0.75 0.63 0.96 0.91 |
| new form., ROLE CUES seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 1 | 0.978 ± 0.044 | 1.00 1.00 0.89 1.00 1.00 |
| depth 2 (seen) | 0.803 ± 0.002 | 0.80 0.81 0.80 0.80 0.80 |
| depth 3 (UNSEEN) | 0.390 ± 0.021 | 0.43 0.38 0.38 0.37 0.40 |
| FULL frame at depth 3 (UNSEEN) | 0.000 ± 0.001 | 0.001 0.000 0.000 0.000 0.000 |

Depth 3 per role: A 0.081 ± 0.026, B 0.099 ± 0.059, C 0.101 ± 0.039, X1 1.000, X2 0.999 ± 0.002, X3 0.061 ± 0.010.
Calibration (rule 18; tau on calibration rows, mean 0.038): new ids coverage 0.016, confident-wrong 0.010 (per seed 0.023 0.009 0.001
0.006 0.011), accuracy among answered 0.366; depth 3 coverage 0.347, confident-wrong 0.013 (per seed 0.017 0.013 0.007 0.015 0.014),
accuracy among answered 0.963. Label: MEASURED.

Exit check against the main repository's RESULTS 7b-bis (new ids 0.253 ± 0.034, depth 3 0.393 ± 0.023): 0.251 ± 0.048 and 0.390 ±
0.021, MET. Reading: as in 7b-bis, the control fits the seen rows and fails on new identities (0.25, chance 0.167) and on the unseen
depth (0.39; it keeps the two outer speakers X1 X2 and loses the inner frame and X3, FULL 0.000).
