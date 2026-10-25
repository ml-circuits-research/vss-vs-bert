# RESULTS — one numbered section per run (command, commit, log, table with mean ± std and the per-seed list, calibration of rule 18, label, reading)

Sections are added in the order of INSTRUCTIONS_2026-10-24.md section 7. A result in the in-between band of
Table 3 is reported with both readings. Reference numbers from the main repository (for the exit check of run 1): RESULTS 6c reader
new ids 1.000, depth 3 1.000; 7b transformer new ids 0.215 ± 0.012, depth 3 0.390 ± 0.024; 7b-bis 0.253 ± 0.034, 0.393 ± 0.023;
8c WikiSQL reader sel 0.816 / 0.760 / 0.632 at 56k / 5k / 1k, transformer 0.543 / 0.510 / 0.415, BoW+LR 0.499 / 0.502 / 0.475.

**Decision before run 3 (responses R1, 2026-10-25): the small pretrained encoder is `distilbert-base-uncased`.** Reasons: it has the
same WordPiece vocabulary and tokenizer as BERT-base, so both tokenizer conditions stay comparable; it is the standard small checkpoint a
reviewer expects. Measured on the DGX: `prajjwal1/bert-small` does not load with the pinned transformers 5.20.2 (the checkpoint has no
fast-tokenizer file and the slow-to-fast conversion fails); no workaround is added, the pin is not changed, and bert-small is not run.
The from-scratch twin of run 3b is DistilBERT's architecture (6 layers, same width), randomly initialised, with the same tokenizer.

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

## 1c. Transformer control 7b, plain input (models/transformer_7b.py --variant plain; logs/2026-10-25_0817_transformer_7b_b4f0739.txt)

Command: `./run.sh models/transformer_7b.py --variant plain --seeds 5`, commit b4f0739 (code identical to c8e4e96; PYTHONHASHSEED=0),
device cuda. Model: the main repository's v9 TModel (0.79 M parameters). Grid on seed 0 by validation cos (lr 1e-3 / 3e-4, dropout
0 / 0.1: 0.963 / 0.867 / 0.962 / 0.916): chosen lr 1e-3, dropout 0; validation cos per seed 0.963 0.917 0.920 0.903 0.877.

| axis (mean role accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 0.972 ± 0.056 | 1.00 1.00 1.00 1.00 0.86 |
| new ids | 0.215 ± 0.015 | 0.23 0.19 0.23 0.20 0.23 |
| new form., seen trans. | 0.769 ± 0.122 | 1.00 0.67 0.66 0.74 0.77 |
| new form., UNSEEN trans. | 0.780 ± 0.062 | 0.82 0.88 0.73 0.74 0.73 |
| new form., ROLE CUES seen | 0.968 ± 0.048 | 1.00 1.00 0.96 1.00 0.88 |
| depth 1 | 0.837 ± 0.094 | 1.00 0.75 0.79 0.88 0.76 |
| depth 2 (seen) | 0.803 ± 0.003 | 0.80 0.81 0.80 0.80 0.80 |
| depth 3 (UNSEEN) | 0.392 ± 0.025 | 0.44 0.38 0.38 0.37 0.40 |
| FULL frame at depth 3 (UNSEEN) | 0.000 ± 0.000 | 0.000 0.000 0.000 0.000 0.000 |

Depth 3 per role: A 0.083 ± 0.027, B 0.104 ± 0.071, C 0.103 ± 0.043, X1 1.000, X2 1.000, X3 0.065 ± 0.012.
Calibration (rule 18; tau on calibration rows, mean 0.047): new ids coverage 0.017, confident-wrong 0.010 (per seed 0.037 0.005 0.001
0.008 0.000), accuracy among answered 0.387; depth 3 coverage 0.340, confident-wrong 0.005 (per seed 0.016 0.001 0.005 0.003 0.002),
accuracy among answered 0.984. Label: MEASURED.

Exit check against the main repository's RESULTS 7b (new ids 0.215 ± 0.012, depth 3 0.390 ± 0.024): 0.215 ± 0.015 and 0.392 ±
0.025, MET. Seed 4 fits the seen row at 0.86 only (validation cos 0.877, the lowest), with the grid fixed on seed 0 as written. Reading:
the same picture as 1b and the main repository: new identities near chance and the unseen depth at 0.39 with FULL 0.000; the category
input of 1b changes neither axis (0.251 vs 0.215 new ids, 0.390 vs 0.392 depth 3). Run 1 gate (R4): 1a, 1b and 1c all reproduce.

## 2a (stopped). BERT-base, WordPiece: started and stopped before any number (logs/2026-10-25_0825_bert_basic_9140ce8.txt)

`./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --seeds 5 --save_model` started at 08:25 (commit 9140ce8)
and was stopped with SIGTERM by the main session within minutes, before the first grid point finished, because responses R6 (the
depth-3 metrics below) arrived and must be in place before any BERT number exists. The log is kept and committed; it holds no result.
The run is started again after the R6 commit, as run 2a.

## Depth-3 metrics added before run 2 (R6, 2026-10-25; commit 3ab0ef9; logs/2026-10-25_0828_depth3_saved_reader_3ab0ef9.txt)

Reason. The token-tagging head of models/bert_basic.py has the labels O, A, B, C, X1, X2, X3, and training covers depths 0-2 only, so
the label X3 never occurs in training. At depth 3 the tagger must output X3 for the third speaker with untrained weights for that
label: X3 is at chance by construction, and the six-role FULL frame at depth 3 is near 0 whatever the encoder composes. The reader and
the 7b transformer write a state that the world's addresses decode, so they are not limited in the same way. A deviation of the
REPORTING only (no model changes), written before any BERT number: for the depth-3 axis, every model's report (reader, transformer,
taggers, through common/report.py) now also gives FULL over the roles whose labels occur in training (A, B, C, X1, X2) and FULL of the
inner frame (A, B, C). The six-role FULL and the per-role table are unchanged.

Runs 1a-1c. Their JSONs (logs/results_*.json) store scores, not per-item predictions, so the two numbers cannot be recomputed from them.
Run 1a saved its seed-0 route book (--save_model): eval/depth3_saved_reader.py decodes the frozen seed-0 depth-3 split with it (no
training) and reproduces the stored six-role FULL exactly (0.9819, EQUAL), with FULL over the trained labels 0.982 and FULL of the inner
frame 0.982 (seed 0 only; the reader's depth-3 errors are all on A or C, so the three FULL measures coincide). Seeds 1-4 of 1a and all
seeds of 1b / 1c have the per-role table only; they are not re-run. For 1b and 1c, the per-role accuracies bound the inner FULL from
above (FULL <= the lowest of A, B, C per seed): 1b <= 0.133 0.064 0.071 0.028 0.075 (mean 0.074), 1c <= 0.133 0.054 0.069 0.026 0.085
(mean 0.074). The trained-label FULL is bounded by the same numbers. These are bounds, not measurements.

Reading of the unseen-depth row of INSTRUCTIONS Table 3 (as R6 asks; INSTRUCTIONS_2026-10-24.md itself is not edited, see questions.md
Q4): the comparison with a tagger at depth 3 is made on the inner frame (A B C) and on the trained labels (A B C X1 X2), with the
thresholds of Table 3 (> 0.9 contradicts, < 0.5 confirms, in between both readings) applied to these two FULL numbers and to the mean
role accuracy over the same roles. The six-role FULL is reported for every model, with the note that a tagger cannot output a label it
was never trained on.

## Pre-registration (R9, 2026-10-25): tag prereg-2026-10-25 on commit 7c2035a

Before any BERT number was written, the main session tagged commit 7c2035a as `prereg-2026-10-25` (annotated, pushed; responses R9.1).
Fixed at the tag: INSTRUCTIONS_2026-10-24.md with its Table 3 interpretations; the grids of the scripts (models/bert_basic.py: lr
{2e-5, 3e-5, 5e-5} x epochs {3, 5, 10}, batch 32, selection on seed 0 by validation of the training distribution); the R6 depth-3 metrics
(FULL over the trained labels A B C X1 X2 and over the inner frame A B C, beside the six-role FULL); the R8 reproduction check for the
real-data round (run 8.0). The code of run 2a (commit 4429ea4, logs/2026-10-25_0830_bert_basic_4429ea4.txt) equals the tagged code:
`git diff --stat 4429ea4 7c2035a` lists responses.md only. Any later change of a criterion, metric or grid is a deviation listed here
with its date, with the tagged version reported beside it (R9.1). From here on: every run, seed and stopped run stays in RESULTS with
its log (R9.2), and the axes where BERT is better are stated with the same prominence as those where the reader is (R9.4).

## Deviations after the independent review (R10), declared 2026-10-25 before any R10 row runs (code: commit 3b06af9)

Responses R10 (the findings of an independent review of the BERT baselines, made from the side of BERT) and R11 (Q6, Q7). The
pre-registered rows (tag prereg-2026-10-25) stay exactly as they are and are reported. Every item below is implemented behind a flag
whose default reproduces the tagged code, or is ADDITIVE (a new field next to the unchanged ones). Rows run with the flags are labelled
"reviewed (R10)" (results files `*_r10*`) and are reported beside the pre-registered rows, never instead of them.

1. **One-to-one decoding** (additive, every model): a second decoding rule beside the per-role argmax. The fillers of one sentence are
   always distinct in the world (a property equally available to every model; it is stated as such). The roles are assigned to distinct
   candidates maximising the summed score (cosine for the reader and the 7b transformers, log-probability for a tagger), exactly
   (Hungarian, eval/metrics.one_to_one). Role margin = the assigned candidate's score minus the best score of any other candidate for
   that role (negative = unsure); the sentence's assignment margin = best total minus the second-best assignment's total (R11). Both
   rules get their calibration blocks. Applied to 1a (seed 0, saved route book) and 2a (seed 0, saved model) with eval/r10_saved.py,
   without retraining; to 1b / 1c by a re-run with the pre-registered selected configuration and saved models (R11, below); to every
   later run in the run itself.
2. **Positions never seen in training.** (a) Additive, every model: the depth-3 role accuracy split by whether the gold filler's
   position was reached by a training sentence (words for the reader and the 7b transformers: the longest training sentence has 15
   words, 75% of depth-3 sentences are longer; first sub-token index for a tagger, against its longest training encoding). (b) Flag
   `--pos_offset K`: random position-id offsets 0..K per training sentence (K = 10 for the R10 scratch row of run 3b; optional for the
   pretrained encoders).
3. **Model selection ties.** Additive: the validation token cross-entropy of every grid point is printed. Flag `--select r10`: among
   the grid points with the best validation FULL, the lowest validation cross-entropy (the tagged rule takes the first, i.e. the least
   trained, point on a tie). If R10 selects another configuration, its 5 seeds are the R10 row.
4. **'whole' initialisation** (R11: applies to the R10 'whole' row of run 2): the pre-registered 'whole' rows use transformers 5.20.2
   `resize_token_embeddings`, whose default `mean_resizing=True` initialises the new rows from the mean and covariance of the existing
   embeddings, NOT "randomly" as INSTRUCTIONS section 4 says; this is the actual initialisation of the pre-registered row 2b. Flag
   `--whole_init pieces`: each new whole-word token = the mean of its WordPiece pieces' embeddings.
5. **Weight decay** (flag `--wd_groups`): no decay on biases and LayerNorm weights (the standard BERT recipe), in every R10 row.
6. **Temperature scaling** (flag `--temperature`): one temperature for the tagger's softmax, fitted on the calibration rows (token
   cross-entropy); the calibration block of the temperature-scaled tagger is reported beside the softmax one. Additive for every model:
   the depth-3 calibration block over the trained roles (A B C X1 X2) beside the six-role one. Stated for every model: the calibration
   rows are the training distribution (near 100% correct), so no threshold or temperature fitted on them can calibrate the
   out-of-distribution axes (new identities, depth 3); the calibration numbers on those axes measure transfer, not fit.
7-10. Reference-world predicate, WikiSQL aggregator and max_len, step counts at 1k / 5k, cased multilingual inputs: implemented when
   their runs come (runs 4, 5 and the real-data round), each in its own commit and listed here before its run.
11. Stated: the depth of a sentence is given at decoding to every model (the roles read are those of its depth), and there is no event /
   depth head on the basic world. This is equal for all models and a deviation from INSTRUCTIONS section 3. The taggers have no CRF
   (standard for BERT token classification).
12. Additive: the taggers' occupancy at depth 3 is also given over the trained labels (X3 is missed by construction).
13. Additive: the cost of a tagger also at fp16 (bytes), with int8 dynamic quantization (bytes and CPU ms per sentence), the CPU thread
   count, and the embedding table's size (the scratch twin carries the 30,522-row WordPiece table for a 66-word world).
14. Flag `--seed_per_config`: torch is seeded before every grid configuration, so a selected configuration can be re-run alone.
15. Flag `--trunc_none`: a word truncated away gets no label and no prediction (the tagged code points it at [CLS]; it never triggers on
   this world, max_len 96).
`--r10` = `--select r10 --wd_groups --seed_per_config --trunc_none --temperature`. The R10 rows of run 2: BERT-base pretrained, `--r10`,
WordPiece and whole (the latter with `--whole_init pieces`). Order (R10, R11): 2a (pre-registered) finishes; eval/r10_saved.py on 1a and
2a; 1b / 1c re-run for R10 (same configuration, models saved); 2b (pre-registered); the R10 rows of run 2; stop before run 3.
