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
16. **Correction of item 1, 2026-10-25 (commit 767cf1f; accepted by R16)**: for a role with a SINGLE candidate in the sentence, the
   one-to-one margin was "score - 0.0", which with a tagger's log-probabilities is log p <= 0 and made a certain decision count as unsure
   (found on the saved 2a model, before any one-to-one row was reported). Now "score - floor", as the per-role rule's "top1 - 0": the
   floor is **0 for the reader and the 7b transformers (cosine scores)** and **log 1e-30 = -69.1 for a tagger (log-probabilities)**.
   Consequence to keep in mind: a tagger's single-candidate margin is about 69 by construction; it is NOT a stronger decision than the
   reader's "top1 - 0", and calibration blocks that set reader and tagger margins side by side say so. Every one-to-one number reported
   in this file uses the corrected margin.
`--r10` = `--select r10 --wd_groups --seed_per_config --trunc_none --temperature`. The R10 rows of run 2: BERT-base pretrained, `--r10`,
WordPiece and whole (the latter with `--whole_init pieces`). Order (R10, R11): 2a (pre-registered) finishes; eval/r10_saved.py on 1a and
2a; 1b / 1c re-run for R10 (same configuration, models saved); 2b (pre-registered); the R10 rows of run 2; stop before run 3.

## Cost accounting and precision (responses R12, 2026-10-25)

- Q8: every row's cost gets (a) the training wall-clock per seed, (b) the selection wall-clock (the seed-0 grid), (c) their sum per
  5-seed row, read from the logs; (d) the GPU energy from run 2b on, from a sidecar sampler (eval/power_sampler.sh, one nvidia-smi
  sample per second while the run holds the lock; eval/power_energy.py integrates it), with no change to model code; the CPU thread count
  and the batch size in every cost row. All labelled "this machine (GB10), fp32". Runs 0-2a have no power samples.
- Q9: every training row (pre-registered, R10, R9.5) is trained in fp32; fp16 and int8 appear only as inference-cost rows.
- Q10: selection cost of the reader in this repository: 0; its configuration was developed in the main repository on the same worlds
  (cited rounds), so the reader's design is not blind to these worlds, while BERT's architecture is. (The cited rounds: the main
  repository's RESULTS 6c, 7c and the rounds before them, on the same basic world.) This sentence goes into the threats-to-validity part
  of the write-up.

## 2a. BERT-base pretrained, WordPiece, pre-registered (models/bert_basic.py; logs/2026-10-25_0830_bert_basic_4429ea4.txt)

Command: `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --seeds 5 --save_model`, commit 4429ea4 (code
identical to the pre-registration tag prereg-2026-10-25 on 7c2035a; PYTHONHASHSEED=0), device cuda, fp32, batch 32, N 30,000, 1.58
WordPiece pieces per world word, 109.5 M parameters. Started 08:30, finished 10:02:40 (the network loss of 09:57, JOURNAL 2026-10-25,
did not affect it). Grid on seed 0 (fixed, INSTRUCTIONS section 4): every one of the 9 points reached validation FULL 1.000 (lr 2e-5 /
3e-5 / 5e-5 x 3 / 5 / 10 epochs: 247 / 410 / 820 s, 249 / 405 / 813 s, 246 / 410 / 816 s); the pre-registered rule (strict `>`) kept the
first point, lr 2e-5, 3 epochs (the tie R10 item 3 describes). Seeds 1-4: 246 s each.

| axis (mean role accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new ids | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., seen trans. | 0.999 ± 0.002 | 0.99 1.00 1.00 1.00 1.00 |
| new form., UNSEEN trans. | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., ROLE CUES seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 1 | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 2 (seen) | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 3 (UNSEEN) | 0.961 ± 0.063 | 1.00 1.00 0.84 0.98 0.99 |
| FULL frame at depth 3 (UNSEEN), six roles | 0.765 ± 0.376 | 1.000 0.978 0.017 0.886 0.943 |
| FULL at depth 3 over A B C X1 X2, trained labels (R6) | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |
| FULL at depth 3 over A B C, inner frame (R6) | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |

Depth 3 per role: A, B, C, X1, X2 1.000 on every seed; X3 0.765 ± 0.376 (per seed 1.00 0.98 0.02 0.89 0.94).
Calibration (rule 18; tau on calibration rows): new ids tau 0.000, coverage 1.000, confident-wrong 0.000 (per seed 0.000 x5), accuracy
among answered 1.000; depth 3 tau 0.000, coverage 1.000, confident-wrong 0.039 (per seed 0.000 0.004 0.164 0.019 0.009), accuracy among
answered 0.961. Occupancy (free tags): no false filler on absent roles on any axis; at depth 3 'missed_present' 0.167 on every seed,
i.e. one of the six roles (X3) counted as missed by the free-tag occupancy rule (R10 item 12 adds the occupancy over the trained labels).
Cost: 109.5 M parameters, 438 MB at 32 bits, 30.5-31.3 ms per sentence on CPU (torch default, 20 threads), batch 32; training 246 s per
seed, selection 4,416 s (the seed-0 grid), 5-seed row 4,416 + 4 x 246 = 5,400 s, this machine (GB10), fp32; no power samples (the
sidecar starts at 2b). Label: MEASURED.

Gate (R4, INSTRUCTIONS Table 3): seen 1.000 >= 0.95, MET; the other rows are interpreted.

Observation against R6's premise. R6 expected X3 "at chance by construction", because the label never occurs in training. It is not at
chance: under the pre-registered per-role argmax over the world's candidates, X3 is the correct speaker on 4 of 5 seeds (0.89-1.00) and
fails on seed 2 (0.02). The X3 output row is untrained; why the argmax lands on the one person token left by the trained roles is not
established by this run (the R10 one-to-one rule and the position split, applied to the saved seed-0 model, are the next measurement).
Stated as observed, not explained.

Reading (Table 3 bands; R9.4: BERT's results first). BERT-base fine-tuned on the basic world is at ceiling on every axis whose labels
it was trained on: new identities 1.000 (> 0.9: the WordPiece pieces carry the role), new formulations 0.999 / 1.000 (the reader 0.864 /
0.868, run 1a: BERT is better on this axis by 0.13, at 109.5 M parameters against the reader's 3.3 M, 1.8x the per-seed training
time (246 s vs 134-136 s) and 8x the 5-seed row with selection (5,400 s vs about 680 s), this machine, fp32), and the unseen depth 3 over the trained labels and over the inner frame at 1.000 (> 0.9: by the reading written in Table 3 for
depth 3, this contradicts, for a pretrained encoder, the paper's same-input claim made against the from-scratch transformer 7b,
whose inner frame is <= 0.074 on the same splits, runs 1b / 1c). The six-role FULL at depth 3 (0.765 ± 0.376) is in the in-between
band; both readings: (i) the frame of the unseen depth is composed except for the one label never trained, which comes out right on 4
seeds and wrong on one; (ii) the unseen-depth frame is not reliably produced as a whole, and the across-seed spread (0.017-1.000) is
itself the result. On this row the reader is at 0.982 ± 0.010 with every seed above 0.97. Calibration: at full coverage BERT has
confident-wrong 0.039 at depth 3 (0.164 on seed 2) against the reader's 0.003.

## R10 rules on the saved seed-0 models of 1a and 2a (eval/r10_saved.py; logs/2026-10-25_1205_r10_saved_767cf1f.txt)

Command: `./run.sh eval/r10_saved.py --reader logs/models/reader_basic_seed0.pt --tagger logs/models/bert_bert-base-uncased_pre_wordpiece_seed0`,
commit 767cf1f. No training: the saved seed-0 route book of 1a and the saved seed-0 tagger of 2a decode the frozen seed-0 split; the
per-role rule recomputed from them equals the stored run on both (EQUAL). Seed 0 only (the runs saved seed 0 only), so this is one seed,
not a row with an interval. Label: MEASURED (seed 0).

| seed 0 | per-role rule: depth 3 roles / FULL six / trained / inner | one-to-one rule: depth 3 roles / FULL six / trained / inner | depth 3 by filler position, per-role rule: seen / never reached |
|---|---|---|---|
| reader (1a) | 0.997 / 0.982 / 0.982 / 0.982 | 0.999 / 0.996 / 0.996 / 0.996 | 0.996 (n 3,960) / 1.000 (n 360) words, training max 15 |
| BERT-base WordPiece (2a) | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 (n 4,305) / 1.000 (n 15) sub-tokens, training max 23 |

New formulations under the one-to-one rule (seen / unseen transitions): reader 0.959 / 0.921 (per-role 0.934 / 0.893 on this seed),
BERT 1.000 / 1.000 (per-role 0.994 / 1.000). Calibration of the one-to-one rule (tau on calibration rows): reader new ids coverage 1.000,
confident-wrong 0.000; depth 3 coverage 0.997, confident-wrong 0.001. BERT new ids 1.000 / 0.000; depth 3 1.000 / 0.000.

Margin version: every one-to-one number above uses the corrected single-candidate margin (R10 item 16, commit 767cf1f): floor 0
for the reader's cosines, log 1e-30 for the tagger. The tagger's one-to-one coverage of 1.000 is therefore partly by construction on
roles with one candidate (a margin of about 69); it is not a stronger decision than the reader's, whose margins are cosine gaps.

What this measurement can and cannot say about the X3 observation of 2a: the saved model is seed 0, on which X3 was already 1.00 under
the per-role rule; seed 2 (X3 0.02) was not saved, so the one-to-one rule cannot be measured on the failing seed from saved models. It
will be measured on every seed in the R10 rows of run 2, where the rule is computed in the run. On BERT's seed 0, only 15 depth-3 role
decisions have their filler at a sub-token position never reached in training; on the reader's, 360 at an unreached word position,
all correct.

Correction made before this log (commit 767cf1f; the first run of the script, logs/2026-10-25_1201_r10_saved_8bf061c.txt, is kept and
is superseded on ONE field only). For a role with a single candidate in the sentence, the one-to-one margin was "score - 0.0", which
with a tagger's log-probabilities is log p <= 0, i.e. a certain decision counted as unsure (25% of the calibration decisions, 33% of new
ids); the first log therefore shows the tagger's one-to-one calibration at coverage 0.740 on new ids. Fixed as the per-role rule does it
("top1 - 0" in probability, so "score - log 1e-30" in log-probability; 0 for the reader's cosines, unchanged). Every other number of the
first log equals this one.

## Interpretation of run 3b, written before it runs (responses R15, 2026-10-25)

Run 3b is the from-scratch twin of the small encoder (DistilBERT's architecture, 6 layers, same width and tokenizer, randomly
initialised, the scratch grid of INSTRUCTIONS section 4), and the direct test of "pretraining is what composes here" after 2a. The
reading applied to its depth-3 FULL over the inner frame (A B C), mean over the five seeds, with the per-seed list:
- **<= 0.1, near the from-scratch transformers of 1b / 1c** (their inner FULL is bounded above by 0.074): the composition of the unseen
  depth seen in 2a (inner frame 1.000) comes from pretraining, not from the architecture or the tagging head.
- **Near 1.0 (> 0.9)**: the composition comes from the tagging head over a BERT-shaped encoder, not from pretraining; the narrowing of
  the paper's claim (R15) would then be about the architecture and output head, not about pretraining.
- **In between (0.1-0.9)**: both readings are written, with the per-seed list. 
The seen-row gate applies first: if 3b's seen FULL is below 0.95 it is under-trained, its rows are reported and not interpreted, and
neither reading above is applied (R4). Run 3a (DistilBERT pretrained) is read with the same Table 3 bands as 2a.

## Deviation R14: rows "gates given" (reader) and "cues given" (BERT), declared 2026-10-25 before they run (code: commit 767cf1f)

Rationale (responses R14). The reader's learned gates differ by seed in run 1a (seeds 0, 3, 4 open {'.', by, that, to}; seed 1 lacks
'by'; seed 2 lacks '.' and opens 'from'), and this is the source of its spread on new formulations (0.864 ± 0.069, 0.868 ± 0.040). On
the training sentences several gate sets score alike, so the data do not fix them. The words that open a role's address ('to' for the
recipient, 'by' for the agent of a passive, 'from' for the source, the sentence end) and the level openers ('says', 'that') are
treated as a GIVEN of the legend, like the word categories (rule 20): the legend already states which roles exist and which question
each answers, and the cue words those questions rest on are the same kind of knowledge, not a choice made on report rows. The main
repository gives the relation-marker gates in the reference world in the same way (v16 --rel_gates open; reader.gates_given in
tool/sop/legend.py).
- Reader row "gates given": `vss/run_reader_basic.py --gates_given cues`: v8_gates.V8.fix_gates(['to', 'by', 'from', '.', 'says',
  'that'], open) (logit +8, gradient masked); every other gate learned from the closed start as before. The same list on every seed.
- Matching BERT row "cues given": `models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --cues given`: one learned cue
  embedding (hidden size, init N(0, 0.02)) added to the input embedding of every token that is one of the same six words as a whole
  token (the next token does not continue it with '##'); everything else as the pre-registered 2a, the same grid, 5 seeds. This is the
  equivalent of telling BERT which words open roles.
- 7b / 7b-bis with the cue feature: only if the budget allows after JP 1; RESULTS will say whether it was run.
- These rows are reported beside the learned ones. The main claims are made on the pre-registered rows; the given-cue rows show how much
  of the new-formulation variance is this degeneracy. The expected effect is on new formulations; any change on depth 3 or new ids is
  reported as such.

## Deviation R17: out-of-world rejection, a residual for the taggers, depth not given (declared 2026-10-25 before any R17 row; responses R17)

Declared additional rows, for every model, after the pre-registered run 3 and before JP 1, never replacing a pre-registered row. They
measure the parts of q-formalisation the pre-registered rows measure only partly ("not given", abstention outside the training
distribution, the residual). Open points with their defaults: questions.md Q12-Q15.
1. **Out-of-world rejection.** Frozen sets data/oow_seed<s>.json (data/make_oow_sets.py, committed before any model sees them): 500 per
   source per seed, (a) random sequences of the world's vocabulary, (b) reference-world sentences whose form is not a basic-world form,
   (c) in-world sentences with one content word replaced by a nonce word; a disjoint 'neg' part for training / calibration. Criterion
   (the demo specification): rejected share >= 0.90 on each source, with false rejections <= 2% on the in-world report rows (the axes of
   runs 1-3). Reader: rejected when its residual is above a threshold set on calibration rows (Q12). BERT, side by side: (i) no training:
   rejected when the largest candidate probability of any role is below a threshold set on the in-world calibration rows at the 2%
   false-rejection point; (ii) with an out-of-world class trained on the 'neg' part, the reader's threshold then also set with the 'neg'
   part, so neither side has information the other lacks. Reported per source: rejected share and false rejections, mean ± std over
   seeds with the per-seed list.
2. **A residual for the taggers**, defined once: the share of content tokens (tokens whose world kind is not a function word) that the
   tagger assigns to no role with probability >= 0.5. Reported beside the reader's residual; it is an analogue, not the same quantity.
3. **Depth not given.** The reader estimates the depth from the count of level-opener tokens ('says', 'claims' followed by 'that'; the
   sop tool's default); BERT from a depth head on [CLS] trained with the tagger (INSTRUCTIONS section 3). FULL at depth 3 (six roles,
   trained labels, inner) with the estimated depth, beside the given-depth row, plus the depth accuracy itself.
Reading, written now: if BERT with the trained out-of-world class (ii) rejects at >= 0.90 with <= 2% false rejections on a source, BERT
meets the criterion there, whatever the reader does; the same for the reader. Results are reported per source, so that a source on which
a model rejects by construction (the reader's lexicon on (c), Q14) is not mixed with a learned rejection.
R17 as settled by responses R18 (2026-10-25; Q12-Q15):
- The reader has no status on the basic world (no event classifier, no unknown-event): for the reader, "rejected" means "residual above
  its threshold" and nothing else. Residual = the sop tool's (tool/sop/infer.py): ||S - a rec|| / ||S||, rec = the world's state of the
  decoded frame at the given (or, in item 3, estimated) depth, a = the best real scale; computed in common/decode.py. Thresholds:
  (i) the 2% false-rejection point of the in-world calibration rows; (ii) the best separating threshold with the 'neg' negatives under
  <= 2% false rejection on the calibration rows.
- Source (b) is the weakest of the three: a person reads most of its sentences as basic-world sentences with C "not given". It is kept,
  reported per source and never pooled. Its 'neg' part is 320 items on seed 0 because the pool of reference sentences of a non-basic form
  is exhausted.
- Source (c), two reader rows: primary "by construction" (a word outside the legend cannot be encoded; rejected, reported as such, NOT
  counted as a learned rejection), and "nonce as placeholder" (one fixed random vector of category 'O' per seed for every nonce word, so
  that the reader decides from its state; the only version of (c) on which its learned rejection can be compared with BERT's). For BERT
  (c) is a real test in both rows.
- Scope: item 1 (i) and item 2 for the reader, 7b-bis / 7b and every BERT / DistilBERT row; item 1 (ii) and item 3 as new 5-seed rows for
  BERT-base WordPiece (2a configuration) and DistilBERT pretrained (3a configuration) with the matching reader rows; item 3 also for 7b-bis
  from its saved R11 re-run models, with the same level-opener count as the reader (a property of the token sequence, not of the model).
  The from-scratch encoder and the controls' trained OOW class are out unless asked.

## Deviation R19: small pretrained encoders at the reader's size (declared 2026-10-25; runs after JP 1)

Declared by responses R19, run after JP 1 (not before). BERT-tiny (google/bert_uncased_L-2_H-128_A-2) and BERT-mini
(google/bert_uncased_L-4_H-256_A-4), pretrained, fine-tuned on the basic world, 5 seeds, both tokenizer conditions, the one-to-one rule and
the position split as every row, the cue feature in a second row (R14). Grid for these models, fixed now and not extended: lr {5e-5,
1e-4, 3e-4} x epochs {5, 10, 20}, batch 32, warm-up 10%, selection on seed 0 with the R10 tie-break (the larger learning rates are the
published practice for the small BERTs, Turc et al. 2019). Load check on the DGX (2026-10-25, transformers 5.20.2, no training): both load
with their fast BertTokenizer and are cached; 4,385,920 and 11,170,560 encoder parameters; 'p7 says that o3 .' -> p ##7 says that o ##3 .
Interpretations, written before the run (R19, verbatim): (a) inner-frame FULL at depth 3 < 0.9 for the small encoders while the reader
stays at 0.98: at equal size, the reader composes the unseen depth and the pretrained encoders do not; (b) >= 0.9: pretraining composes
on this world even at 4 M parameters, and the reader's case rests on the six-role FULL, the calibration, the residual and the cost; (c)
in between, both readings. Cost rows for both (parameters, bytes at fp32 / fp16 / int8, CPU ms per sentence with the thread count) on
the same table as the accuracy. These rows do not change the pre-registered claims.

## 2b. BERT-base pretrained, whole-word tokens, pre-registered (models/bert_basic.py --tokenizer whole; logs/2026-10-25_1207_bert_basic_f2afe4e.txt)

Command: `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer whole --seeds 5`, commit f2afe4e. The code differs from
the tag prereg-2026-10-25 only by default-off flags and additive outputs (R10, R14, the margin correction of R10 item 16), so the
pre-registered numbers are those of the tagged code; the additive R10 fields (one-to-one rule, position split) are computed in the run.
Device cuda, fp32, batch 32, N 30,000; the 66 world words added as whole tokens (1.00 piece per word) with transformers 5.20.2
`resize_token_embeddings` (mean_resizing=True: new rows from the mean and covariance of the existing embeddings, not "randomly
initialised" as INSTRUCTIONS section 4 says; R10 item 4). Grid on seed 0: all 9 points validation FULL 1.000 (token cross-entropy 0.00007
at lr 2e-5 / 3 epochs down to 0.00000 at 5e-5 / 10); the pre-registered rule kept the first point, lr 2e-5, 3 epochs.

| axis (mean role accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new ids | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., seen trans. | 0.993 ± 0.015 | 1.00 0.96 1.00 1.00 1.00 |
| new form., UNSEEN trans. | 0.977 ± 0.018 | 1.00 0.98 0.99 0.96 0.95 |
| new form., ROLE CUES seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 1 | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 2 (seen) | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 3 (UNSEEN) | 0.892 ± 0.107 | 0.72 1.00 0.91 1.00 0.84 |
| FULL frame at depth 3 (UNSEEN), six roles | 0.495 ± 0.444 | 0.000 1.000 0.460 1.000 0.014 |
| FULL at depth 3 over A B C X1 X2, trained labels (R6) | 0.858 ± 0.281 | 0.296 1.000 0.994 1.000 1.000 |
| FULL at depth 3 over A B C, inner frame (R6) | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |

Depth 3 per role: A, B, C, X1 1.000; X2 0.858 ± 0.281 (seed 0 0.30); X3 0.495 ± 0.444. Position split (R10 item 2a): every depth-3
error is at a sub-token position seen in training (0.890 seen / 1.000 never reached). One-to-one rule (R10 item 1, corrected margin):
depth 3 per role 0.839, FULL six 0.516 ± 0.445 (0.000 1.000 0.565 1.000 0.014), trained labels 0.516 ± 0.445, inner 1.000; the rule does
not help this row (it lowers X2 from 0.858 to 0.516). Calibration (rule 18): new ids coverage 1.000, confident-wrong 0.000; depth 3
coverage 1.000, confident-wrong 0.108 (per seed 0.284 0.000 0.091 0.000 0.164), over the trained labels 0.028 (0.141 0.000 0.001 0.000
0.000). Occupancy: no false filler on any axis; at depth 3 'missed_present' 0.167 (X3), 0.000 over the trained labels.
Cost (this machine, GB10, fp32): 109.5 M parameters (embedding table 30,560 rows), 438 MB at fp32, 219 MB at fp16, 182 MB int8 dynamic;
CPU 28.3-30.5 ms per sentence (int8 16.5-18.8 ms), 20 threads, batch 32; training 228-231 s per seed, selection 4,104 s (the 9 grid
points), 5-seed row 4,104 + 4 x 229 = 5,019 s; GPU energy of the whole run 78.0 Wh (5,223 s sampled, mean 53.7 W, peak 90.3 W, idle
draw at start 4.3 W; eval/power_energy.py on logs/power_run2b.csv). Label: MEASURED.

Gate (R4): seen 1.000 >= 0.95, MET.

Reading (Table 3 bands; BERT's results first). With whole-word tokens BERT-base is again at ceiling on new identities (1.000: > 0.9),
near it on new formulations (0.993 / 0.977, above the reader's 0.864 / 0.868 by 0.11-0.13), and at 1.000 on the inner frame of the unseen
depth on every seed (> 0.9, as 2a). The trained-label FULL at depth 3 (0.858 ± 0.281) is in the in-between band: (i) the unseen depth
is composed over the trained labels on 4 seeds (0.994-1.000) and the outer speaker X2 is misplaced on one (seed 0, 0.30); (ii) with whole
tokens the frame of the unseen depth is not reliably produced beyond the inner clause. The six-role FULL (0.495 ± 0.444, two seeds near 0)
is in the in-between band with the same two readings. Against 2a (WordPiece), the whole-token condition lowers depth 3 (0.892 vs 0.961
per role) and does not change the inner frame. The reader on the same rows: six-role FULL 0.982 ± 0.010, confident-wrong at depth 3
0.003 against 0.108 here, at 3.3 M parameters against 109.5 M.
Added by responses R21 (2026-10-25). The one-to-one rule LOWERS this row (X2 0.858 -> 0.516, trained-label FULL 0.858 -> 0.516): it is
a finding against the rule, not a free improvement for any model; the pre-registered per-role numbers stay the main ones. Observed, not
explained: in 2b every depth-3 error is on the OUTER speakers (X2 0.858 ± 0.281, per seed 0.30 1.00 0.99 1.00 1.00; X3 0.495 ± 0.444)
while the inner frame is 1.000 on every seed, the mirror image of the 7b controls' failure shape at the same depth (inner roles lost,
A / B / C 0.08-0.10, outer speakers X1 / X2 kept at 1.000, runs 1b / 1c).
