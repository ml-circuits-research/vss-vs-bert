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
the byte diff is not mistaken for a change of data.
