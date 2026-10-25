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
A / B / C 0.08-0.10, outer speakers X1 / X2 kept at 0.999-1.000, X3 0.06-0.07, runs 1b / 1c).
R17 item 3 for BERT, as settled by responses R20 (2026-10-25; Q16), stated before the rows run. Three depth variants on the same 5 seeds,
the 2a configuration (lr 2e-5, 3 epochs, no grid), each its own row: (a) `--depth_head cls`: a classifier on [CLS] over 0-3, class 3
never in training (an untrained class, as X3 in R6), reported as such; (b) `--depth_head reg`: one regression output on [CLS], squared
error, rounded and clipped to 0-3; (c) `--depth_head count`: a token-level binary head "this word opens a level", trained on the gold
level openers (the 'that' opening a reported clause, which occurs at every training depth 1-2 and is absent at depth 0), depth = the
number of words it tags, so it can output 3 as naturally as a count. On this world the gold openers are exactly the 'that' right after
a reporting verb ('says', 'claims'): the opener count equals the gold depth on every in-world axis. In the same table the reader's (and
7b-bis's) depth is that count as a RULE on the token kinds, not a learned head; the rows report the depth accuracy and FULL at depth 3
(six roles, trained labels, inner) with the decided depth beside the given-depth rows.

## Derived rows: inherited configurations and shared heads (responses R22, 2026-10-25; RUNBOOK section 1b)

On 2a and 2b every one of the 9 grid points reached validation FULL 1.000, so the pre-registered selection never chose anything but the
first point of the grid (the tie, R10 item 3). By R22 item 1 the derived rows (R10, R11, R14, R17, R20) run without a grid, with the
configuration selected on seed 0 by the pre-registered row of the same model and tokenizer condition; each row names the selection it
inherits. R10 item 3 (ties to the lowest validation cross-entropy) is therefore reported, not run: on 2b it would have chosen lr 5e-5, 10
epochs (cross-entropy 0.00000 against 0.00007 for the chosen lr 2e-5, 3 epochs); 2a ran with the tagged code, which did not print the
cross-entropy. Deviation (R22 item 2): the add-on heads share the encoder: one run per model and seed trains the tagger together with the
cue feature (BERT-base, R14), the out-of-world class (R17 ii) and the three depth heads (R20), losses summed with equal weights, each head
decoded as its own row. The R14 tagger row is therefore the cue-given tagger of that multi-head run, not a tagger trained alone with
the cue feature; it is labelled so. The pre-registered rows ran alone and are unchanged. Concurrency (R22 item 3, questions Q17): at most
two runs at once, and only after a measured gain >= 1.3x; cost rows of concurrent runs are marked "concurrent (N runs)".

## Deviation R24: BERT-MRC on the basic world (declared 2026-10-25 before it runs; code models/bert_mrc.py; responses R24, R25; questions Q18)

A new row family, not a change of the pre-registered tagger rows: the question-based form of the task (QA-SRL, He et al. 2015; MRC
extraction, Li et al. 2020). bert-base-uncased before JP 1 (R25); DistilBERT-MRC after JP 1 with R19 (Q18 (4) default).
- **Questions, frozen here** (one per role and level, verb-neutral, Q18 (2) default): A "who gives?", B "what is given?", C "to whom is
  it given?", X1 "who says it?", X2 "who says it at level 2?", X3 "who says it at level 3?" (X1 is the outermost speaker, level 1).
- **Input and head**: "[CLS] question [SEP] sentence [SEP]", start / end logits over the sentence's sub-tokens, NULL = the [CLS]
  position; one forward pass per (sentence, question).
- **Training**: the same 30k sentences as every model (rng 200+seed, depths 0-2). Row 'zs' (primary, Q18 (1a) default): the questions of
  A, B, C, level 1 and level 2 on every sentence, NULL for a level absent at that depth; the level-3 question is never seen in training.
  Row 'all' (Q18 (1b)): all six questions, level 3 always NULL in training. No out-of-world negatives in either row.
- **Grid**: the pretrained grid of INSTRUCTIONS section 4 (lr {2e-5, 3e-5, 5e-5} x epochs {3, 5, 10}) on seed 0, selected by validation
  FULL, ties to the lowest validation span loss (R24: the R10 tie-break if every point is 1.000). The seed-0 grid is about 54 epochs over
  150,000 (sentence, question) examples, longer than run.sh's 6 h cap (R4: the cap is not raised), so it runs as one run per learning rate
  (`--grid_only --lrs <lr>`); `--select` reads the three files and prints the choice; the five-seed run then uses that configuration
  (seed 0 retrained at it).
- **Decoding**: a role's answer = the candidate word (the world's candidate list, as for every model) with the highest start + end logit
  (its first and last sub-token), or NULL if the [CLS] score is higher; margin = best minus second best over the candidates and NULL;
  the threshold rule of rule 18 on the calibration rows, as every model. No one-to-one row for MRC (each question is answered alone; the
  rule needs one score table per sentence over all roles, which this form does not produce in one pass).
- **Depth**: given (the questions of the sentence's depth, as every model) and NOT given (all six questions; NULL = not given; depth = the
  number of answered level questions), both reported; occupancy (false fillers on absent roles, missed present roles) under the second.
- **R17 (i) in the same run**: on the frozen out-of-world sets (depth from the level openers), rejected = some question of that depth
  answered NULL (NULL refusal) and, second, the smallest NULL gap below its 2% false-rejection point on the calibration rows; residual
  analogue = the share of content words that answer no question.
- **Cost rows**: CPU ms per sentence with all six questions (six passes) and the thread count, parameters, bytes at fp32 / fp16, training
  time per seed, energy from the sidecar (for the batch when runs are concurrent).
Interpretations, written before the run (R24, verbatim in substance): inner-frame FULL at depth 3 >= 0.9 means the question form composes
as the tagger did (expected); the six-role FULL and the level-3 question are the new information (a question never asked in training, row
'zs'); confident-wrong under NULL training against the reader's residual is the calibration comparison a reviewer will trust most; the cost
ratio with the legend's size is the argument that survives whatever the accuracy says.

## Deviation R26: the query-emission row, for every model (declared 2026-10-25; runs after JP 1 and after the format gains query wires)

Declared by responses R26 now, so that the planted worlds are built and frozen before any model is tuned with them in view; run only after
JP 1 and after the reader's format emits queries (demo plan 2.8). Scored on the OUTPUT, not on the mechanism.
- **Data**: a planted-questions world (data/planted_seed<s>.json, data/make_planted_sets.py, questions Q19 default): 300 'unsupported'
  sentences per seed (an extra in-vocabulary identity appended as ', <id> .', which no question of the legend asks for; gold = that slot
  is `unsupported`) and 300 'undecided' (a second identity of the same class right after one role's filler; gold = that role is
  `undecided`); the out-of-world sets (data/oow_seed<s>.json), gold = `unknown`; the normal frozen splits, gold = values.
- **Scoring of every decision**: correct value, rightly queried, needlessly queried, or INVENTED (a value emitted where the gold is a
  query, or a wrong value emitted confidently); the invented share is the hallucination rate.
- **Query sources**: the reader: margins (undecided), unplaced words, the residual (unknown) and a coverage test on the state
  (unsupported: a content token whose identity is not recovered by any decoded slot), a quantity of the state. BERT-MRC: NULL for not
  given, the margin for undecided, and, so that it is not disadvantaged, a COVERAGE head (a token-level "covered by some answer" tagger)
  trained on the same planted negatives the reader's residual threshold is calibrated on, for unsupported and unknown, a trained head.
  Both are reported as what they are.
- **Criteria, written now (the demo's)**: values emitted under the threshold = 0 for every model by construction of the row; planted
  unsupported slots reported as queries >= 0.90; the hallucination rate reported per axis.

## 3b: not run (deviation from the pre-registration, responses R28, 2026-10-25)

Run 3b (DistilBERT's architecture from scratch, pre-registered at tag prereg-2026-10-25) is NOT executed; recorded here before its
scheduled time (it was to start after 3a). Reason (R28, the author's decision): its purpose, "is it pretraining or the form that composes
the unseen depth?", is answered better by variant (b) (the same architecture from scratch) at every small size of the equal-size block
(BERT-tiny, BERT-mini, BERT-small; R28 item 1), while 3b on DistilBERT would cost about an hour for one point at a size the paper no
longer compares at. The two readings written for 3b (section 'Interpretation of run 3b') are applied to the (b) rows instead, per size.
A pre-registered row that is not run is reported as such here and in STATUS, never silently dropped.

## Deviation R28: the main comparison at the reader's size, in the MRC form (declared 2026-10-25)

The paper's question is q-formalisation (R28): the legend's answers, "not given", a margin per decision, a residual, a query where the
model does not know, in one pass, in a model that fits a browser. Main comparison: the reader (3.3 M parameters; 13 MB, 1.7 MB at 4 bits)
against BERT-MRC (deviation R24's form) at small sizes: BERT-tiny (google/bert_uncased_L-2_H-128_A-2, 4.39 M), BERT-mini
(L-4_H-256_A-4, 11.17 M), BERT-small (L-4_H-512_A-8, 28.76 M; the official Google upload loads with transformers 5.20.2, checked
2026-10-25, fast BertTokenizer), each in three variants: (a) pretrained, (b) the same architecture from scratch, (c) pretrained with the
kinds given (R27 item 1). R19's grid (lr {5e-5, 1e-4, 3e-4} x epochs {5, 10, 20}, batch 32, warm-up 10%, selection on seed 0, ties to
the lowest validation loss) for every variant; 5 seeds; every declared axis: seen, new ids, new formulations, depth 3 (per role, six-role
FULL, trained labels, inner), depth not given (all questions, NULL), out-of-world rejection (R17: NULL as trained, and the threshold),
the residual analogue, the symmetric cost (R27 item 4). BERT-base and DistilBERT become reference rows, kept as measured (0, 1a-1c, 2a,
2b, 3a); the R10 / R14 / R17 / R20 tagger rows on them are deferred without date; the one BERT-base row still run is R24 MRC on
bert-base-uncased, the large reference of the MRC form (5 seeds, both depth variants). The reader's R10 and R14 rows stay. The
interpretations of R19 (a)/(b)/(c) apply to each size; R15's two readings of 3b apply to each (b) row.

## Regime statements for the WikiSQL and real-data rows (responses R27 item 5, 2026-10-25; written before those runs)

- Run 5 (WikiSQL): this repository's regime (the question without the table, the columns as candidates) is below the published
  table-conditioned systems (about 0.90 selected-column accuracy with the table). The comparison is BERT against the reader in the same
  regime, never against WikiSQL's state of the art; the same sentence goes into RESULTS 5 and the paper.
- Runs 8-12 (SNIPS, MASSIVE, TOPv2): the reader's kinds come from skip-gram clusters on the training pool (learned, unsupervised), not
  given; every real-data row says so (INSTRUCTIONS_REAL note of 2026-10-25).

## Declared NOW, before any result of the equal-size block is read (responses R30 items 3, 7, 8, 12; 2026-10-25)

**Confirmatory comparisons (R30 item 8), pre-specified.** Pairs: the reader against each equal-size encoder of R28 in the MRC form
(BERT-tiny, BERT-mini, BERT-small, each pretrained and from scratch: six pairs) and against BERT-base MRC (one pair). Three comparisons per
pair, on the five shared seeds (seed 0's frozen splits and so on):
- (i) six-question exact match at depth 3 with the depth NOT given (the frame exact match with the levels as labels; the reader with
  the depth from its level-opener count, MRC with all six questions and NULL = not given);
- (ii) the confident-error rate at matched coverage: the reader's coverage on depth 3 (depth not given) under its threshold set on the
  calibration rows; the other model's threshold chosen on the same calibration rows to give that coverage; confident error = a wrong
  answered decision, over all decisions;
- (iii) the out-of-world AUROC, in-world report rows against out-of-world sources (a) random and (b) reference pooled (source (c)
  reported apart, the reader rejecting it by construction): the reader's residual (and margin), the encoder's Mahalanobis score on its
  encoder states (and MSP).
Test: the difference reader minus model, its p-value from the two-level bootstrap (seeds x items, 1,000 resamples, two-sided), Holm's
correction over the three comparisons of each pair at alpha 0.05; mean ± std, the 4-df paired t, the seeds-ahead count and a sign
test are reported beside it and not relied on (the seeds are bimodal). Every other row declared in R10-R27 is EXPLORATORY.

**Withdrawal condition (R30 item 3), written before the block runs.** If an encoder of the reader's size trained FROM SCRATCH reaches an
inner frame (A B C) exact match >= 0.99 at depth 3 with the depth not given (mean over the five seeds), the claim "same-size models
trained from scratch do not compose this world" is WITHDRAWN, not re-narrowed. Until the block runs, the from-scratch evidence at BERT-base
size is the 7b control only (3b is not run, R28).

**Terminology (R30 items 3, 7), from here on.** "FULL" is reported as **exact match** (the six-role FULL at depth 3 is "frame exact
match with the levels as labels", not "composition"); per-role accuracy as **slot accuracy**; a wrong value given with a margin above the
threshold is a **confident error**, a value given where the gold is absent is a **false fill**; "hallucination" appears at most once,
defined as their sum. Earlier sections keep their words; their numbers mean the same.

**Layout (R30 item 12).** The tables of the write-up (eval/compare.py) are split in two: the pre-registered rows (tag prereg-2026-10-25:
0, 1a-1c, 2a, 2b, 3a; 3b not run) and the declared rows, each with its R number and the date of its declaration (R6, R10, R14, R17,
R19, R20, R22, R24 (tag prereg-2026-10-25b), R26, R27, R28, R30, all 2026-10-25). Nothing is removed. The headline table is the one with the
depth NOT given / all questions, for every model; the given-depth rows beside it, labelled (R30 item 5).

## Rule ceiling, positional and window-LR rows (responses R30 item 1; eval/rule_ceiling.py; logs/2026-10-25_1430_rule_ceiling_503cf7c.txt)

Command: `VSA_SLOTS=2 ./run.sh eval/rule_ceiling.py --seeds 5`, commit 503cf7c, CPU, 5 seeds, the frozen splits. (The first run,
logs/2026-10-25_1428_rule_ceiling_6464f8e.txt, stopped with a TypeError after the rule's seed 0; kept.) The rule was written from the
legend's 16 forms and checked on training sentences only (exact match 1.000 on 3,000 of seed 0's) before any report row was scored.

| row (slot accuracy) | seen | new ids | new form. seen / UNSEEN trans. | role cues | depth 1 / 2 | depth 3 | exact match depth 3: six / trained / inner |
|---|---|---|---|---|---|---|---|
| rule (given kinds + says / claims / that / to / by / from) | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 1.000 | 1.000 / 1.000 / 1.000 |
| positional (training frequency of length x position) | 0.767 ± 0.020 | 0.767 ± 0.020 | 0.667 / 0.611 | 0.833 | 0.850 / 0.880 | 0.39 | 0.000 / 0.000 / 0.000 |
| window LR (previous word, kind, next word) | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 0.75 | 0.000 / 0.501 ± 0.017 / 1.000 |

Per seed: the rule is 1.000 on every axis and seed, also with the depth NOT given (its own speaker count; exact match of all six roles
1.000 on every axis); the positional row's per-seed lists are in the log (seen 0.79 0.75 0.75 0.75 0.79); window LR at depth 3: X2
0.47-0.52, X3 0.00 on every seed. Label: MEASURED.

Consequence, written as R30 item 1 asks. On every axis of the basic world a hand-written rule over the given kinds and six function words
reaches exact match 1.000, with or without the depth given, and a logistic regression over a three-word window reaches 1.000 on every
axis except depth 3 (inner frame 1.000 there too). These axes therefore measure whether a model learned this rule, not composition. The
reader's 0.86-0.87 on new formulations is BELOW the rule (and below BERT-base, 0.98-1.00). The composition evidence must come from axes
where the rule fails or does not apply (depth not given beyond the training depths, out-of-world sets, the planted sets) and from the
equal-size block; the depth-3 rows of the taggers measure the frame exact match with the levels as labels.

## 3a. DistilBERT pretrained, WordPiece, pre-registered (models/bert_basic.py; logs/2026-10-25_1336_bert_basic_7a1500a.txt)

Command: `./run.sh models/bert_basic.py --model distilbert-base-uncased --tokenizer wordpiece --seeds 5`, commit 7a1500a (code
differing from the tag prereg-2026-10-25 only by default-off flags and additive outputs, as 2b), device cuda, fp32, batch 32, N 30,000,
66.4 M parameters, 1.58 pieces per world word. The small encoder of R1. Grid on seed 0: all 9 points validation FULL 1.000 (token
cross-entropy 0.00012 at lr 2e-5 / 3 epochs down to 0.00001); the pre-registered rule kept lr 2e-5, 3 epochs (R10's tie-break would
have chosen lr 3e-5 or 5e-5 at 10 epochs, cross-entropy 0.00001).

| axis (slot accuracy) | mean ± std | per seed (0-4) |
|---|---|---|
| seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new ids | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., seen trans. | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| new form., UNSEEN trans. | 0.985 ± 0.030 | 1.00 0.93 1.00 1.00 1.00 |
| new form., ROLE CUES seen | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 1 | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 2 (seen) | 1.000 ± 0.000 | 1.00 1.00 1.00 1.00 1.00 |
| depth 3 (UNSEEN) | 0.938 ± 0.063 | 0.85 0.97 1.00 0.87 1.00 |
| exact match depth 3, six roles (levels as labels) | 0.631 ± 0.379 | 0.113 0.797 1.000 0.244 1.000 |
| exact match depth 3, trained labels A B C X1 X2 | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |
| exact match depth 3, inner frame A B C | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |

Depth 3 per role: A, B, C, X1, X2 1.000; X3 0.631 ± 0.379. Position split: every depth-3 error at a sub-token position seen in
training. One-to-one rule: exact match six 0.633 ± 0.379, inner 1.000. Calibration: new ids coverage 1.000 / confident error 0.000;
depth 3 coverage 1.000 / confident error 0.062 (per seed 0.148 0.034 0.000 0.126 0.000), over the trained labels 0.000. Occupancy: no
false fill; at depth 3 'missed_present' 0.167 (X3), 0.000 over the trained labels. Cost (this machine, GB10, fp32): 66.4 M parameters,
266 MB at fp32, 138 MB int8 dynamic; CPU 17.6-18.7 ms per sentence at 20 threads, batch 1 (int8 10.5-16.3 ms; seed 1's 50.5 ms
coincided with other CPU processes on the machine, the smoke tests and the planted-set generation, and is not a cost number);
training 142-143 s per seed, selection 2,549 s, 5-seed row 2,549 + 4 x 142 = 3,117 s; GPU energy of the run 49.1 Wh (3,242 s sampled,
mean 54.5 W, peak 82.0 W). The GPU part ran alone; CPU jobs ran beside it from 14:24 to 14:32. Label: MEASURED.

Gate (R4): seen 1.000 >= 0.95, MET.

Reading (Table 3 bands; BERT's results first; R30 terminology). DistilBERT at 66 M is at ceiling on new identities (1.000) and new
formulations (1.000 / 0.985, above the reader's 0.864 / 0.868), and at exact match 1.000 on the inner frame and on the trained labels at
depth 3 on every seed (> 0.9: as 2a). The six-role frame exact match with the levels as labels (0.631 ± 0.379) is in the in-between band:
(i) the unseen level-3 label comes out right on two seeds (1.000) and partly on two (0.797, 0.244); (ii) the frame with the untrained
level label is not reliably produced. By the rule ceiling (above), these axes measure whether a model learned the world's rule; DistilBERT
did, as BERT-base did. The reader on the same rows: six-role exact match 0.982 ± 0.010, confident error at depth 3 0.003 against 0.062.

## Concurrency check of run.sh's slots (responses R23; eval/throughput.py; logs/2026-10-25_14*_throughput_1c37d49.txt)

Training throughput of BERT-MRC (batch 32, fp32, the same script and data), samples per second, summed over concurrent runs:
- **bert-base-uncased**: alone 368.4; six concurrent 57.3 + 58.7 + 57.4 + 69.0 + 60.0 + 61.4 = 363.8 (0.99x); three concurrent
  117.7 + 125.9 + 123.6 = 367.2 (1.00x). Below 1.3x at 6 and at 3: BERT-base rows run ONE at a time (the GPU is compute-bound at this
  size; peak reserved 2.52 GB per process).
- **BERT-tiny** (L-2_H-128): 5,000 steps alone 2,806.1; six concurrent 1,554.9 + 1,531.0 + 1,537.1 + 1,539.9 + 1,545.5 + 1,543.4 = 9,251.8
  (3.30x): the small models are launch-bound, and their rows run six at a time. (A first attempt with 300 steps overlapped too little to
  measure anything, about 3 s per run, and is not used.)
Per R22 / R23 the cost rows per seed come only from runs that ran alone; the energy of concurrent runs is reported for the batch.

## R11 re-runs of 1b / 1c with saved models, and R17 (i) on them and on the saved 2a seed-0 tagger (exploratory; R30 item 8)

Logs: logs/2026-10-25_1507_transformer_7b_699a9ec.txt (1b re-run), logs/2026-10-25_1508_transformer_7b_699a9ec.txt (1c re-run),
logs/2026-10-25_1516_r17_reader_bb13fee.txt and _1517_ (R17 on them), logs/2026-10-25_1507_r17_tagger_699a9ec.txt (R17 on 2a, seed 0).
Concurrent runs (six slots; GPU energy for the batch in logs/power_queue_2026-10-25_1506.csv). Label: MEASURED.

**R11 re-runs (same pre-registered configuration, models saved; exit check: the per-role rule reproduces 1b / 1c).** 7b-bis
(lr 3e-4, dropout 0): new ids 0.246 ± 0.051, depth 3 0.390 ± 0.021 (pre-registered 1b: 0.251 ± 0.048, 0.390 ± 0.021; within seed
variance, MET; not bit-identical, GPU non-determinism). 7b (lr 1e-3, dropout 0): 0.215 ± 0.015, 0.392 ± 0.025, identical to 1c (MET).
One-to-one rule on the same models (R10 item 1): depth-3 exact match six roles 0.000, inner frame 0.002 ± 0.003 (7b-bis) and 0.003 ±
0.002 (7b); slot accuracy at depth 3 0.399 / 0.401. The rule does not change the controls' picture.

**R17 (i), out-of-world rejection with thresholds at the 2% false-rejection point of the in-world calibration rows** (the depth from the
level openers, which equals the gold depth on every in-world axis, 1.000):

| model | rejected: random | reference | nonce (placeholder) | nonce (by construction) | false rejection, pooled in-world report axes |
|---|---|---|---|---|---|
| 7b-bis (residual) | 0.997 ± 0.002 | 0.524 ± 0.168 | 0.976 ± 0.041 | 1.000 | 0.374 ± 0.039 |
| 7b (residual) | 0.971 ± 0.025 | 0.340 ± 0.186 | 0.789 ± 0.132 | 1.000 | 0.353 ± 0.018 |
| BERT-base WordPiece, 2a seed 0 only (MSP: min over roles of the max candidate probability) | 1.000 | 0.982 | 0.954 (a real test) | n/a | 0.391 |

The false rejections are not spread evenly: per axis (7b-bis / 7b / 2a seed 0) seen 0.000 / 0.015 / 0.000, new ids 1.000 / 1.000 /
1.000, depth 3 1.000 / 1.000 / 1.000, new formulations 0.19-0.34 / 0.06-0.30 / 0.35-0.50. A threshold set at 2% false rejection on the
training distribution (where every model is near 100% correct and its scores saturate: 2a's threshold is a probability of 0.9998)
rejects every new-identity and every depth-3 sentence for all three models: on this world the in-world "out-of-distribution" axes and
the out-of-world sets are not separated by a training-distribution threshold. None meets the demo criterion (>= 0.90 per source with <=
2% false rejection). The confirmatory comparison on this point is therefore the AUROC (R30 item 8 (iii)), not a single threshold.
The tagger's residual analogue (content tokens with no role probability >= 0.5) is degenerate in-world (0.11-0.25 on every in-world axis,
because the verbs are content words that no role takes) and is reported but not interpreted (R30 item 7).

## Deviation R34 / R35: no grid for BERT-base MRC and for the equal-size block; one sensitivity table (declared 2026-10-25, before any of these rows is read)

Reason (questions Q23 point 3, written by the main session): on this world the validation set cannot discriminate. Every grid point
measured so far (2a, 2b, 3a; BERT-tiny MRC at three learning rates) gives validation FULL 1.000, the MRC span loss prints as 0.00000,
and the selection rule (best validation FULL, ties to the lowest validation loss) chooses by numerical noise. The validation draw is not
leaked: on seed 0, 80 of 2,000 validation sentences (4.0%) also occur among the 29,359 distinct training sentences. The cause is the
world itself: the rule ceiling (above) reaches exact match 1.000 on the training distribution. **The saturation is symmetric**: the reader
was selected under the same rule, in the same world, with the same validation draw; its rows are not re-selected (R34, R35).
- **BERT-base MRC** (the large reference): 2a's configuration (lr 2e-5, 3 epochs), no grid, 5 seeds (R34 (B)).
- **Equal-size block, all nine variants** (BERT-tiny / mini / small x pretrained / from scratch / kinds given): no grid; one configuration
  fixed by rule before any report row is read, the middle point of R19's grid: lr 1e-4, 10 epochs, batch 32; 5 seeds each; the same for
  every model.
- **Convergence rule** (R35): a from-scratch variant whose last-epoch training loss is not below the pretrained variant's by more than a
  factor of 10 at 10 epochs (i.e. has not converged as the pretrained one has) gets ONE extra row at 20 epochs, labelled; no further
  tuning. The validation and training losses are printed at full float precision from commit b5e0aee.
- **One sensitivity table, not a selection**: BERT-mini pretrained, seed 0 only, the full R19 grid (lr {5e-5, 1e-4, 3e-4} x epochs {5,
  10, 20}), the report-axis numbers for all nine points; read only after the fixed-configuration rows; it never picks a configuration.
  If its spread on depth 3 (depth not given) or out-of-world exceeds the seed spread, that is written here and in the limitations.
- The BERT-tiny grid chains that had already started (pretrained, lr 5e-5 / 1e-4 / 3e-4 x 5 / 10 / 20 epochs, seed 0, grid_only) are
  kept and reported as what they are (validation numbers only; no selection is made from them).
- The 'all six questions in training' rows (Q18 (1b)) are not in R35's block; they wait for a response.

## R14: the reader with the role-cue gates given (vss/run_reader_basic.py --gates_given cues; logs/2026-10-25_1506_run_reader_basic_699a9ec.txt; exploratory)

Command: `./run.sh vss/run_reader_basic.py --gates_given cues --seeds 5 --save_model`, commit 699a9ec, concurrent (six slots; training
511-545 s per seed against 134-136 s alone in 1a: the time is not a cost number). Gates given open on every seed: to, by, from, '.',
says, that; every other gate learned from the closed start, and the learned set is the same on all five seeds: the given six plus 'claims'.

| axis (slot accuracy) | gates given (R14) | learned gates (1a) |
|---|---|---|
| seen / new ids | 1.000 / 1.000 | 1.000 / 1.000 |
| new form., seen trans. | 0.764 ± 0.029 [0.76 0.76 0.76 0.82 0.73] | 0.864 ± 0.069 [0.93 0.82 0.77 0.95 0.84] |
| new form., UNSEEN trans. | 0.789 ± 0.011 [0.79 0.78 0.81 0.79 0.78] | 0.868 ± 0.040 [0.89 0.89 0.79 0.88 0.89] |
| role cues seen; depth 1; depth 2 | 1.000; 1.000; 1.000 | 1.000; 1.000; 1.000 |
| depth 3 (UNSEEN) | 1.000 ± 0.000 | 0.997 ± 0.002 |
| exact match depth 3, six roles / trained / inner | 0.999 ± 0.001 / 0.999 / 0.999 | 0.982 ± 0.010 (all three) |

One-to-one rule: exact match six roles 1.000 on every seed. Calibration: new ids and depth 3 coverage 1.000, confident error 0.000.
Cost: 3.32 M parameters, route book 13.3 MB (1.7 MB at 4 bits); CPU 16 ms per sentence measured under concurrent load. Label: MEASURED.

Reading. Giving the cue gates removes the gate degeneracy (one gate set on every seed) and the seed spread on new formulations (std
0.069 -> 0.029, 0.040 -> 0.011), and lifts the depth-3 exact match to 0.999; but it LOWERS the mean on new formulations by 0.08-0.10
(0.764 / 0.789 against 0.864 / 0.868): the learned gate sets that differed by seed (seed 2 opened 'from', seed 1 closed 'by') were, on
average, better for unseen forms than the given set. The R14 row is therefore not an improvement of the reader on that axis; it shows that
the 1a spread was the degeneracy, as R14 expected, and that removing it costs accuracy on new formulations. The matching BERT row (cue
feature) is deferred (R25 / R28). The rule ceiling (1.000 on every axis) stands above both.

## Amendment R36 to deviation R34 / R35 (declared 2026-10-25, before any row of the equal-size block runs; the R35 text above is kept as written)

- **5 epochs, not 10**: the fixed configuration of every block variant is lr 1e-4, **5 epochs**, batch 32, 5 seeds (BERT-tiny's
  validation span loss is already below printed precision at 5 epochs on every learning rate tried). The convergence clause now reads:
  if a from-scratch variant's last-epoch training loss is more than 10x the pretrained variant's at the same size, ONE labelled row at
  20 epochs for that variant; nothing else is tuned.
- **BERT-small after tiny and mini, and only if they leave a question**: tiny (4.4 M) and mini (11 M) bracket the reader (3.3 M); the three
  BERT-small variants run only if the picture differs between tiny and mini, otherwise they are recorded as not run with that reason.
- GPU order: the six tiny and mini rows in parallel (the LLM reference beside them); then BERT-base MRC alone (2a's configuration); then
  the 2a re-run with every seed saved and R17 on it; then the BERT-mini seed-0 sensitivity grid.

## Execution notes for the equal-size block (written 2026-10-25 16:10, before any block row runs; commit 33dc482; no number changes)

- Every block run is `models/bert_mrc.py --model <tiny|mini> --train_questions zs [--scratch | --kinds given] --lrs 1e-4 --epochs 5
  --save_all` (R36), 5 seeds. BERT-tiny: one run of 5 seeds per variant. BERT-mini: each variant split into two runs (`--seeds 3`, and
  `--seed_start 3 --seeds 2`) so that no run approaches run.sh's 6 h cap (R4: the cap is not raised); the five per-seed JSONs are pooled
  for the row. Seeds and data are those of a single run (the seed sets everything; the split changes no number).
- `--save_all` keeps every seed's model (logs/models/, untracked) so that R30 item 8 (ii) and (iii) (matched coverage, Mahalanobis on
  encoder states) and R17 (i) are computed on the saved models without retraining.
- bert_mrc.py now prints one line per epoch: run.sh's log-stall watchdog (30 min without log growth) killed the BERT-tiny 5e-5 grid chain
  at its 10-epoch point (below), and a BERT-mini run of 5 epochs would print nothing for longer than that.
- Only the 'zs' rows (training questions of levels 0-2, the main row of Q18 (1a)) run in the block; the 'all six questions' rows wait for
  the answer to questions Q25 (R35 left them pending).
- BERT-small: not run unless tiny and mini disagree (R36), decided after both are read.
- Concurrency: up to six queue runs plus the LLM reference (API-bound) in a seventh run.sh slot (VSA_SLOTS=7: memory fraction 0.129 per
  process; measured peaks 0.2 GB for BERT-tiny, 2.5 GB for BERT-base). Cost rows from these runs are concurrent and are not cost numbers.

## R17 (i), the residual and the R10 rules on the reader 1a, five seeds (eval/r17_reader.py; logs/2026-10-25_1507_r17_reader_699a9ec.txt; exploratory)

Command: `VSA_SLOTS=6 ./run.sh eval/r17_reader.py --seeds 5`, commit 699a9ec, concurrent (six slots; retraining 515-620 s per seed is not
a cost number). Each seed's reader is retrained by 1a's code and checked against the stored 1a row: per-role numbers EQUAL on all five
seeds (max difference 0.0). The depth is estimated from the level openers (no depth given): depth accuracy 1.000 on every in-world axis
and seed; exact match at depth 3 with the estimated depth 0.982 ± 0.010 [0.982 0.971 0.999 0.986 0.971], identical to the given-depth row.
Thresholds: (i) residual, (ii) residual with the 'neg' part, both at the 2% false-rejection point of the calibration rows (0.546-0.600).

| quantity | mean ± std | per seed (0-4) |
|---|---|---|
| rejected, random | 1.000 ± 0.000 | 1.000 1.000 1.000 1.000 1.000 |
| rejected, reference | 0.989 ± 0.009 | 0.982 0.986 1.000 0.978 1.000 |
| rejected, nonce (placeholder) / (by construction) | 1.000 / 1.000 | 1.000 on every seed |
| false rejection, pooled in-world report axes | 0.435 ± 0.015 | 0.456 0.418 0.436 0.447 0.418 |
| false rejection: seen / new ids | 0.021 ± 0.017 / 0.083 ± 0.047 | 0.046 0.032 0.000 0.005 0.023 / 0.121 0.116 0.000 0.060 0.118 |
| false rejection: new form. seen trans. / UNSEEN trans. / role cues | 0.881 ± 0.146 / 1.000 / 0.883 ± 0.144 | 1.000 0.681 1.000 1.000 0.725 / 1.000 x5 / 1.000 0.683 1.000 1.000 0.733 |
| false rejection: depth 1 / depth 2 / depth 3 | 0.032 ± 0.026 / 0.001 / 1.000 | 0.006 0.036 0.060 0.060 0.000 / - / 1.000 x5 |

(ii) gives the same numbers as (i) on every seed. Residual means (fraction of the sentence not explained by the decoded frame):
calibration 0.432, seen 0.408, new ids 0.435, depth 1 / 2 0.454 / 0.431, new formulations 0.768 (seen trans.) / 0.900 (UNSEEN) / 0.675
(role cues), depth 3 0.723; out-of-world random 0.987, reference 0.774, nonce 0.852 (std 0.003-0.029 over seeds).

R10 rules on the same five readers (the one-to-one rule): slot accuracy new formulations, seen trans. 0.870 ± 0.089 [0.96 0.79 0.77 0.99
0.84] (per-role rule 0.864), UNSEEN trans. 0.894 ± 0.048 [0.92 0.93 0.80 0.90 0.92] (per-role 0.868); exact match at depth 3 0.995 ±
0.003 [0.996 0.992 1.000 0.996 0.993] (per-role 0.982); confident error at depth 3 0.000-0.002. Label: MEASURED.

Reading. The reader's residual rejects the out-of-world sources at 0.98-1.00, but a threshold at 2% false rejection on the training
distribution also rejects every depth-3 and every UNSEEN-transition sentence and most new-formulation sentences (pooled false rejection
0.435): as for the controls and 2a (section R11 above), a training-distribution threshold does not separate the in-world generalisation
axes from the out-of-world sets, and the demo criterion (>= 0.90 per source with <= 2% false rejection) is NOT met by the reader either.
The residual is graded (depth 3 0.72 and UNSEEN formulations 0.90 lie between in-distribution 0.41-0.45 and random 0.99), so the
confirmatory comparison remains the AUROC (R30 item 8 (iii)). The one-to-one rule raises the depth-3 exact match to 0.995 and
new formulations by 0.01-0.03, within the seed spread.

## Runs stopped on 2026-10-25 around 16:02-16:03 (kept, partial; logs untracked until this commit)

- **BERT-tiny MRC grid chains** (pretrained, zs, seed 0, `--grid_only --epochs 5,10,20`; logs/2026-10-25_1509_bert_mrc_699a9ec.txt
  lr 5e-5, logs/2026-10-25_1518_bert_mrc_bb13fee.txt lr 1e-4, logs/2026-10-25_1519_bert_mrc_bb13fee.txt lr 3e-4). The 5-epoch point of
  each chain completed: validation exact match 1.000 and span loss 0.00000 (printed at 5 decimals; the full-precision print came with
  b5e0aee, after these runs started) at lr 5e-5 (1,385 s), 1e-4 (1,306 s) and 3e-4 (1,309 s), concurrent, 4.4 M parameters, 150,000
  examples per epoch, peak reserved 0.2 GB. The 10- and 20-epoch points did not finish: the lr 1e-4 and 3e-4 chains were stopped by the
  main session at 16:03 because R35 / R36 removed the grid; the lr 5e-5 chain was killed at about the same time by run.sh's log-stall
  watchdog (its log: "log did not advance for 30 min"; the script printed nothing during a 10-epoch point, fixed in 33dc482). Status:
  kept, partial, stopped per R36; no selection is made from them; no grid JSON was written.
- **LLM reference row** (eval/llm_reference.py --call, commit bb13fee; logs/2026-10-25_1514_llm_reference_bb13fee.txt): stopped
  unintentionally at 16:02 after 1,300 of 4,800 calls (repeat 1 of 3), 0.6221 USD spent at the last log line (at most about 0.65 USD by
  the stop). Cause: the run was a child process of the previous experimental session's shell and was terminated with it. No result was
  written (the script writes a repeat's JSON when the repeat completes). Restarted at 16:12 from the start, detached from any session
  (setsid), with the same settings and prices read again on the pricing page at call time (input 2, output 10, 5-minute cache write 2.5,
  cache read 0.10 USD per M tokens, unchanged) and the cap lowered to 19.35 USD so that the two runs together stay within 20 USD. Note:
  the stopped run's log shows 819 cache-read tokens on every call, i.e. the prompt cache DID engage, against R33's expectation; the
  restarted run's log is read for the same fields.

## The 'all six questions' MRC rows of the equal-size block (responses R40, declared 2026-10-25 before they run)

BERT-tiny and BERT-mini PRETRAINED only, `--train_questions all` (all six questions in training, the level-3 question always NULL),
the fixed configuration (lr 1e-4, 5 epochs, batch 32), 5 seeds (BERT-mini split as the zs rows), after the 'zs' block; labelled
'all'. Sentence for the write-up (R40): a deployed MRC reader is trained with every question of its legend, so 'all' shows how it
behaves when a question that was always NULL in training becomes answerable (the realistic form of "not given" turning into
"given"); 'zs' is the fair capacity test. If 'all' refuses level 3 on every seed, that is the expected result and is written as such,
not as a failure of BERT.

## Deviation R37: the reader with the six cue gates given as a trainable initialisation (declared 2026-10-25 before it runs; code 4bc0a1e)

Diagnostic row (exploratory), same seeds and frozen splits as 1a / R14: `./run.sh vss/run_reader_basic.py --gates_given init_cues
--seeds 5 --save_all`. The six R14 gates (to, by, from, '.', says, that) start OPEN at 1a's typical learned logit and stay trainable;
every other gate starts closed as in 1a. The typical learned logit, fixed by rule before the run: the median (torch.median, the lower
middle value) of the open learned logits of the saved 1a model (logs/models/reader_basic_seed0.pt, the only saved 1a model): open
gates '.' 1.522, by 2.301, to 8.203, that 8.444 -> 2.3011 (sigmoid 0.909). Reading written now (R37): if the row recovers 1a's mean on
new formulations (0.864 / 0.868) with R14's seed spread (std about 0.03 / 0.01), R14's loss comes from the HARDNESS of the given gates,
not from the set, and the demo default should be "given as initialisation"; if it does not, the loss comes from the set, and the learned
per-seed gate sets that beat it are listed here. Also found while writing it: R14's seed-0 model had been saved as
logs/models/{tag}_seed0.pt (a missing f-string); renamed to reader_basic_cues_seed0.pt (its logits: the six given gates at 8.0, as R14).

## Deviation R42 / R43: the explained-tokens residual (declared 2026-10-25 before it runs; code 6405a61)

A second rejection score for the reader, beside the compiled residual of every table (which stays). Definition (R42; reference
implementation vss-demo commit a1f4564, tool/sop/infer.py residual_explained, parity 55/55 there; vss-demo is NOT a submodule of this
repository, so common/decode.residuals_explained transcribes R42's definition): explained = tokens of a non-content kind (cue words,
openers, COP, punctuation) U fillers of the decoded roles (speakers included) U the event lemma; rec' = the reader's own state of the same
sentence restricted to the explained tokens (addresses from the full sentence, identities of unexplained tokens zeroed); residual' =
||S/|S| - a rec'/|rec'||, a = max(0, Re<rec', S>); 1.0 when nothing is explained or a role of the depth is not decoded; the nonce
placeholder is never explained. Stated adaptation: the basic-world decoder emits no lemma, so the event lemma is the first token of an
event-verb kind (one per in-world sentence). Check before any row: 0 (max 2e-15) on 200 calibration sentences with the gold frame, > 0
(min 0.37) when A is replaced by an absent identity.
Rationale as corrected by R43: it scores only what the decoded frame leaves unexplained and is insensitive to form novelty by
construction (0 whenever every token is placed): the score for the query wire of kind `unknown`; the compiled residual is the "form is
unfamiliar" signal. Both are reported. **Expected failure mode, written now (R43): random in-vocabulary sentences that the decoder still
fills into a frame leave few tokens unexplained and may pass.**
Rows: `eval/residual_explained.py --row 1a --seed_start 0` on the saved 1a readers (design set, inspected) and `--row 1a_fresh
--seed_start 5` on 1a trained on the fresh splits (the CLAIM, R39): AUROC in-world report rows vs out-of-world (random + reference pooled;
per source; nonce as placeholder apart) and the 2% calibration-threshold table, both residuals. The AUROC on seeds 5-9 decides which
score separates out-of-world better.

## Fresh seeds 5-9 for the reader 1a and its out-of-world sets (declared 2026-10-25; responses R13, R39, R43)

The claim of R42 / R43 (and later of any improved reader, R38 / R39) is made on seeds 5-9. That needs, with the SAME code and fixed
seeds as seeds 0-4: data/reference_seed5-9.json (`data/make_reference_splits.py --seed_start 5`, the reference world, the source (b) of
the out-of-world sets) and data/oow_seed5-9.json (`data/make_oow_sets.py --seed_start 5`), both generated and committed before any model
sees them (sha256 in the commit message), and the reader 1a trained on the fresh splits (`vss/run_reader_basic.py --seed_start 5 --seeds 5
--save_all`, named "VSS reader (reference) [fresh splits, seeds 5-9]", tag reader_basic_fresh, never pooled with seeds 0-4; R13's seeds
5-9 for the reader). Nobody opens their items.

## R17 (i), the residual and the R10 rules on the R14 reader (gates given), five seeds (eval/r17_reader.py --gates_given cues; logs/2026-10-25_1552_r17_reader_8ae6eeb.txt; exploratory)

Command: `VSA_SLOTS=6 ./run.sh eval/r17_reader.py --seeds 5 --gates_given cues`, commit 8ae6eeb, concurrent. Each seed retrained as R14
(236-472 s, concurrent) and EQUAL to the stored R14 row on all five seeds. Depth from the level openers: accuracy 1.000 on every axis;
exact match at depth 3 with the estimated depth 0.999 ± 0.001 [0.997 1.000 1.000 1.000 0.999], identical to the given depth.
Thresholds (i) = (ii) 0.540-0.574 (2% false rejection on the calibration rows). Label: MEASURED.

| quantity | R14 (gates given) | 1a (learned gates, section above) |
|---|---|---|
| rejected random / reference / nonce (placeholder) | 1.000 / 0.989 ± 0.009 [0.982 0.986 1.000 0.978 1.000] / 1.000 | 1.000 / 0.989 ± 0.009 (the same per-seed list) / 1.000 |
| false rejection, pooled in-world report axes | 0.437 ± 0.002 [0.436 0.436 0.436 0.440 0.438] | 0.435 ± 0.015 |
| false rejection: seen / new ids / depth 1 / depth 2 | 0.000 / 0.004 ± 0.005 / 0.066 ± 0.012 / 0.002 | 0.021 / 0.083 / 0.032 / 0.001 |
| false rejection: new form. seen trans. / UNSEEN / role cues / depth 3 | 0.998 / 1.000 / 0.998 / 1.000 | 0.881 / 1.000 / 0.883 / 1.000 |
| residual means: calibration / seen / new form. UNSEEN / depth 3 / random / reference / nonce | 0.422 / 0.391 / 0.910 / 0.725 / 0.989 / 0.833 / 0.850 | 0.432 / 0.408 / 0.900 / 0.723 / 0.987 / 0.774 / 0.852 |
| one-to-one rule: slot accuracy new form. seen / UNSEEN; exact match depth 3 | 0.768 [0.77 0.76 0.76 0.82 0.73] / 0.796 [0.80 0.78 0.82 0.79 0.79]; 1.000 on every seed | 0.870 / 0.894; 0.995 |

Reading. Giving the gates changes nothing in out-of-world rejection (the same rejection lists, per seed, as 1a: the 1.1-2.2% of reference
sentences that pass are passed by both readers) and leaves the training-distribution threshold rejecting every depth-3 and every
new-formulation sentence (pooled false rejection 0.437, with a smaller seed spread than 1a). The demo criterion is not met. The
one-to-one rule gives exact match 1.000 at depth 3 on every seed and adds 0.00-0.01 on new formulations.

## R10 row of 2a: BERT-base WordPiece with the reviewed code paths (models/bert_basic.py --r10; logs/2026-10-25_1508_bert_basic_699a9ec.txt; exploratory)

Command: `VSA_SLOTS=6 ./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --r10 --lrs 2e-5 --epochs 3 --seeds 5`,
commit 699a9ec, the inherited configuration of 2a (no grid, R22 item 1), concurrent with the six-slot batch (times are not cost numbers).
R10 flags in force: weight-decay groups, a seed per configuration, a truncated word read as None, the temperature rule. Label: MEASURED.

| axis | R10 row | 2a (pre-registered) |
|---|---|---|
| seen / new ids / new form. seen / UNSEEN / role cues / depth 1 / depth 2 | 1.000 / 1.000 / 0.999 ± 0.002 / 1.000 / 1.000 / 1.000 / 1.000 | the same |
| depth 3 slot accuracy | 0.961 ± 0.063 [1.00 1.00 0.84 0.98 0.99] | 0.961 ± 0.063 (the same list) |
| exact match depth 3, six roles (levels as labels) | 0.764 ± 0.376 [1.000 0.978 0.017 0.886 0.940] | 0.765 ± 0.376 [1.000 0.978 0.017 0.886 0.943] |
| exact match depth 3, trained labels / inner frame | 1.000 / 1.000 | 1.000 / 1.000 |
| one-to-one rule: depth-3 slot accuracy; exact match six | 0.930; 0.789 ± 0.387 [1.000 1.000 0.017 0.988 0.942] | (R10 rules on the saved seed 0: section above) |
| temperature rule: depth-3 X3; exact match six | 0.000 on every seed; 0.000 | — |
| confident error at depth 3 (tau 0.000, coverage 1.000) | 0.039 [0.000 0.004 0.164 0.019 0.010] | 0.039 [0.000 0.004 0.164 0.019 0.009] |

Reading. The reviewed code paths of R10 reproduce 2a to the third decimal (the only differences are one sentence on seed 4); the
review's implementation items do not change BERT-base's picture on this world. The one-to-one rule lifts seed 1 and seed 3 to exact
match 1.000 / 0.988 and leaves seed 2's level-3 failure (0.017) untouched; the temperature rule turns X3 off on every seed (the
temperature fitted on the calibration rows, where X3 never occurs, makes the untrained label lose everywhere).


## Deviation R46: the reader with "sentence candidates + one-to-one", decoder only (declared 2026-10-25 before it runs)

R46's text, verbatim:

**Proposed row, decoder only, no retraining** (R38 protocol; declared now; claim on seeds 5-9): **sentence candidates + one-to-one**:
the cleanup of each role is restricted to the words of the sentence whose kind the role admits (the tool already has this mode,
training/parity `--sentence-candidates`, introduced for words outside the catalog), and the one-to-one assignment of R10 is applied
over those candidates; margin = the gap between the best and the second in-sentence candidate, with R10's single-candidate floor. This
is the same assumption the taggers and MRC carry by construction (the answer is a span of the sentence), so it equalises the decoders;
it does not touch the chart or the routes. Expected effect on the design cells, written before the run: template 11 B (one O in the
sentence) and template 4 B become right by construction; template 4 C, template 7 A and template 10 C become right through one-to-one
(the other P is taken by the role whose address does hold it) whenever that other role is right, which the dumps show it is; the
in-world rows cannot fall (the gold is always in the sentence). Expected cost, also written now: the out-of-world "random in-vocabulary"
set will be filled into frames more often (the explained residual of R42 is what has to catch it, and the planted 'undecided' sets
measure the margin between two in-sentence candidates, which is the right test). Report coverage, confident error and the AUROC beside
exact match, as for every row.

Implementation: common/decode.decode_frozen(..., sentence_candidates=True) (default off; every reported row keeps the original
decoder), the one-to-one rule of R10 (eval/metrics.one_to_one) over those candidates; eval/sentence_candidates.py. The margin is
metrics.one_to_one's: the assigned candidate's score minus the best OTHER in-sentence candidate of the role (equal to best minus second
when the assignment keeps the role's top-1; negative when one-to-one overrides it), a single candidate against R10's floor 0. Rows: 1a
on seeds 0-4 (design set, inspected), R14 on seeds 0-4 (inspected), and the CLAIM on seeds 5-9 with the 1a readers trained on the fresh
splits (row 'VSS reader (reference) [fresh splits, seeds 5-9]'). Reported beside exact match: coverage and confident error (tau on the
calibration rows, rule 18), the out-of-world AUROC and the 2% threshold table for both residuals on the one-to-one frames. Per (template,
role) cell of the new-formulation rows on the design set (questions Q27): decided by the address vs decided by elimination vs wrong.
**The H2 cause (the address a filler receives follows the cue chain) stays unfixed by this row: it changes the decoder, not the chart.**

Does 1a show the same cells (R46's condition for running the row on 1a; eval/diagnose_reader.py on results/dump_reader_basic_seed0-4.json,
questions.md note of 17:15)? Yes for T4:C (cell error 0.88-0.99 on every seed) and T7:A (0.93-0.98 on every seed); T10:C on every seed
but graded (0.24-1.00); T4:B and T11:B only on seed 2, the one 1a reader with the 'from' gate open (0.84-0.86; 0.000 on the other four).
One difference from R14: in 1a the errors of T4:C and T10 are mostly SWAPS with A (the other role's filler, gold ranked 2nd), in R14
mostly a filler absent from the sentence. The row therefore runs on the 1a readers too.

**R46 first attempt failed (logs/2026-10-25_1710_sentence_candidates_be645c7.txt, and the parallel R14 run, same commit):** KeyError in
eval/metrics.one_to_one on out-of-world sentences that have fewer admissible words than roles (the assignment gave a role a column
outside its candidates); no row was written. Fix (separate commit): a local copy of the rule with one guard, a role left without one of
its own candidates is "not given" (margin 0); identical to metrics.one_to_one whenever every role receives its own candidate (every
in-world sentence). **Caption of the R46 row (R47):** it changes the decoder's candidate set, not the addresses; the H2 cause stays
unfixed; each role decision is reported as decided by the ADDRESS (the in-sentence winner is the catalog winner of the original
decoder), by RESTRICTION (the catalog winner was absent from the sentence) or by ELIMINATION (one-to-one assigned the remaining
candidate), with the accuracy in each; the margin of a decision by elimination is the single-candidate floor, not evidence.
**Note (R47):** for the demo's worlds the cue-gate default is withdrawn until TODO_DGX 2.9 is measured (opening 'from' in training
breaks another reading, a finding that applies to any given gate set).
**R46 second attempt failed (logs/2026-10-25_1715_sentence_candidates_6f40eed.txt, 1a, and _1716_, R14):** IndexError in the
compiled residual on the nonce-as-placeholder set: with sentence candidates the placeholder (kind O, R18) is an admissible B candidate
and was decoded, and the world's identity table had no row for it. Fix (separate commit): inside the placeholder context the world's
table is extended with the same vector as the model's (this process only). A frame holding the placeholder is R46's expected cost.

## Equal-size block (R28 / R35 / R36), BERT-tiny MRC pretrained and from scratch, 'zs' (models/bert_mrc.py; logs below; MEASURED)

Commands: `./run.sh models/bert_mrc.py --model google/bert_uncased_L-2_H-128_A-2 --train_questions zs [--scratch] --lrs 1e-4 --epochs 5
--save_all --seeds 5`, commit 79fff34, launched 16:08:22 (pretrained) and 16:08:42 (scratch), concurrent (6-7 runs). **Log note:** the
two runs started in the same minute, so run.sh gave them the same log name: logs/2026-10-25_1608_bert_mrc_79fff34.txt holds both outputs
interleaved under the second run's header (and _1609_ holds BERT-tiny kinds-given and BERT-mini pretrained seeds 0-2 the same way).
Each run's complete stdout is kept separately: logs/2026-10-25_1608_bert_mrc_79fff34__tiny_pre_stdout.txt, __tiny_scratch_stdout.txt,
logs/2026-10-25_1609_bert_mrc_79fff34__mini_pre_s0-2_stdout.txt (kinds given when it ends). From queue3 on, launches are 61 s apart.
Table: eval/mrc_table.py over the per-seed JSONs (mean ± std [seeds 0-4]).

| axis | BERT-tiny pretrained (4.4 M) | BERT-tiny from scratch (4.4 M) | reader 1a (3.3 M) |
|---|---|---|---|
| seen / new ids | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| new form. seen trans. | 0.884 ± 0.099 [0.753 0.836 0.833 1.000 1.000] | 0.628 ± 0.145 [0.500 0.833 0.667 0.707 0.431] | 0.864 ± 0.069 |
| new form. UNSEEN trans. | 0.968 ± 0.041 [1.000 0.898 1.000 0.944 1.000] | 0.806 ± 0.041 [0.819 0.804 0.804 0.737 0.865] | 0.868 ± 0.040 |
| new form. role cues | 1.000 | 0.715 ± 0.080 [0.667 0.806 0.694 0.806 0.602] | 1.000 |
| depth 1 / depth 2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| depth 3 slot accuracy | 0.829 ± 0.004 (X3 0.000 on every seed) | 0.642 ± 0.025 (A 0.256, C 0.593, X3 0.000) | 0.997 ± 0.002 |
| exact match depth 3, depth NOT given: six / trained / inner | 0.000 / 0.972 ± 0.027 / 0.972 ± 0.027 [0.969 0.928 0.965 1.000 1.000] | 0.000 / 0.173 ± 0.150 / 0.173 ± 0.150 [0.000 0.174 0.450 0.115 0.125] | 0.982 ± 0.010 (opener depth) |
| confident error depth 3 (tau 0.000, coverage 1.000) | 0.171 ± 0.004 (trained labels 0.005) | 0.358 ± 0.025 (trained labels 0.230) | 0.003 |
| occupancy, depth not given (false fill / missed) | 0.000 / 0.000 on seen, new ids, depth 3 | 0.000 / 0.000 | — |
| R17 (i) NULL refusal: random / reference / nonce | 0.285 / 0.000 / 0.227 (false rejection 0.000) | 0.289 / 0.005 / 0.268 (0.007) | — |
| R17 (i) threshold (2% cal.): random / reference / nonce | 0.992 / 0.943 / 0.954, false rejection pooled 0.280 ± 0.072 | 0.915 / 0.928 / 0.956, false rejection 0.506 ± 0.057 | residual: 1.000 / 0.989 / 1.000, false rejection 0.435 |
| last-epoch training loss | 5.2e-5 1.3e-4 2.9e-5 9.5e-5 3.3e-4 (mean 1.27e-4) | 1.2e-3 1.0e-3 1.0e-3 2.9e-4 1.4e-3 (mean 9.97e-4) | — |
| size, fp32 | 4.39 M parameters, 17.6 MB | the same | 3.32 M, 13.3 MB |

Concurrent runs: training 355-1,174 s per seed and CPU 28-30 ms per sentence with six passes are not cost numbers (R22 / R45).
Convergence clause (R36): the from-scratch row's last-epoch training loss is 7.8x the pretrained row's (row means; per seed 24x, 7.6x,
35x, 3.1x, 4.3x), not more than 10x: no 20-epoch row (the clause read on the row means, as the rows are compared; two seeds exceed 10x,
stated). Withdrawal condition (R30 item 3): BERT-tiny from scratch, inner-frame exact match at depth 3 with the depth not given 0.173 <
0.99: the claim stands for BERT-tiny (BERT-mini from scratch pending).

Reading (BERT's results first; no interpretation beyond the declared ones). The level-3 question, never asked in training ('zs'), is
answered on no item by either BERT-tiny row (X3 0.000 on every seed; the six-question exact match at depth 3 is 0.000), while it never
fills an absent role (occupancy 0.000) and the questions of levels 1-2 and the inner frame stay at 0.97 for the pretrained row
(R24's reading: the question form composes the inner frame, >= 0.9, pretrained only). From scratch at the same size the inner frame at
depth 3 falls to 0.173 and new formulations to 0.63-0.81. Pretrained BERT-tiny is ABOVE the reader on UNSEEN-transition formulations
(0.968 vs 0.868) and on role cues (1.000 vs 1.000, equal), at the reader's level on seen-transition formulations (0.884 vs 0.864). NULL
refusal rejects 0.00-0.29 of the out-of-world sentences; a calibration threshold on the NULL gap rejects 0.92-0.99 but with 0.28-0.51 false
rejection (the reader's residual: 0.99-1.00 at 0.435). The confirmatory comparisons (R30 item 8) are computed on the saved models when the
block finishes (eval/confirmatory.py).

## Concurrency check at 12 slots (responses R45; eval/throughput.py; logs/2026-10-25_1721_throughput_01a4715.txt, _1727_throughput_ecd7c47.txt)

R23's measurement at the slot count R45 allows: BERT-tiny MRC, 5,000 steps of batch 32, N concurrent copies launched together, under
the SAME background load in both conditions (the queue frozen at three runs: BERT-tiny kinds given, two reader trainings; two R46
decoder runs; the LLM row): N = 6: 556.5-558.8 samples/s each, total 3,344; N = 12: 403.1-405.3 each, total 4,847: 1.45x >= 1.3x, so
BERT-tiny / BERT-mini jobs run up to 12 at a time from 17:35 (memory fraction 0.9 / 16 per process in the queue, 0.19 GB peak reserved
each; the 40 GB guard unchanged; BERT-base alone). The copies of one N started in the same minute and share one run.sh log file; each
copy's stdout is in logs/2026-10-25_1721_throughput_01a4715__N6_stdout.txt and logs/2026-10-25_1727_throughput_ecd7c47__N12_stdout.txt.
No axis result; concurrent times are not cost numbers.

## Stopping rule R48 (declared by the owner, 2026-10-25 17:50): the declared set and the rows NOT RUN

The experiment stops when this set is measured, whatever the numbers say: (1) BERT-mini pretrained / from scratch / kinds given and
BERT-tiny kinds given; (2) the 'all' rows of BERT-tiny and BERT-mini pretrained (R40); (3) the reader rows R37, R46 (three decision
categories) and R42 / R43, each with its claim on seeds 5-9 and the 1a readers of seeds 5-9 (R37 is also trained on seeds 5-9 for its
claim); (4) the planted-sets row (R26 / Q19) for the reader and the small MRC models, 'undecided' and 'unsupported' as query emission;
(5) the three confirmatory comparisons (R30 item 8), reader vs each small model, risk-coverage and AURC, AUROC in- vs out-of-world for
every score the saved models allow, the cost table at equal precision (R30 item 10); (6) the LLM reference. Then JP 1, the synthesis.

**Not run: the declared set is complete (R48).** Each of these is recorded here with that reason and was removed from the queue at 17:56:
- BERT-base MRC (R24 / R34 (B)): the owner avoids training a 110 M model; the claim is stated at the reader's size; BERT-base stays as
  the measured tagger reference (2a, 2b, the R10 row).
- BERT-small MRC (R28 / R36), the BERT-mini sensitivity grid (R35), the 2a re-run with every seed saved and R17 (i) on it.
- R30 items 2 (depth 4-6, length control, primitive-to-role, seq2seq), 4 (nonce identities), 6c (7b with a tagging head).
- The deferred BERT-base tagger rows (R10 rows of 2b, 3a, 3b; R14 cue feature; R17 (ii); R20 depth heads), seeds 5-9 for the controls.
- The real-data round and the multilingual round (runs 4-12 of RUNBOOK): a later paper with its own pre-registration.
- The address-level fix of H2 (R47): tool-repository work after the stop.

## The planted-sets row (R26 / Q19, R48 item 4), declared 2026-10-25 before it runs (code eval/planted.py)

On the frozen planted sets data/planted_seed0-4.json (committed at e6c6f68, before any model saw them; seeds 0-4, the only planted
sets; they were never part of the design-set inspection), for the reader 1a (original decoder, and R46's sentence candidates + one-to-one)
and the saved small MRC models (BERT-tiny / BERT-mini x pretrained / from scratch / kinds given). Rules fixed now, the same for every
model: depth from the level openers (not given); **'undecided'** is emitted for the planted role when its margin is below the model's
threshold t = the 2% quantile of its role margins on the calibration rows (the R17 cap: at most 2% of in-distribution decisions
queried; rule 18's tau is 0.000 for every model on this world and would query nothing), otherwise its value counts as INVENTED;
**'unsupported'** is emitted for every identity token (kind P or O) placed in no role: the planted token is rightly queried if no role
took it and invented if one did; an original identity left unplaced is a needless query. Deviation from R26 as written, stated: **no
coverage head is trained for MRC** (the declared set of R48 contains no new training); every model uses the same untrained coverage
test. The 'unknown' source (the out-of-world sets) is scored by the AUROC rows of R30 item 8, not here. Criterion (R26, the demo's):
planted unsupported reported as queries >= 0.90; the hallucination rate (invented planted decisions + confident errors on the other
roles, over all decisions; the term used once, so defined) reported per set.

**Additions to the confirmatory script before it runs (R48 item 5):** AURC of the risk-coverage curve at depth 3 (depth not given;
margin as confidence) for the reader and every MRC model; AUROC of the reader's explained residual (R42) and of the MRC NULL gap beside
the pre-specified scores; the pairs are the reader against each of the six small MRC rows (Holm per pair over the three comparisons).

## Deviation R49 / R50: the cost of a new legend question (declared 2026-10-25 before it runs; in the R48 set between items 2 and 3)

Question: what the next legend question costs each system, in annotated examples, retraining time, hyperparameters to choose and
accuracy lost on what already worked. Examples frozen first: data/growth_seed0-9.json (data/make_growth_sets.py: per seed 500
depth-3 training-distribution sentences, 'level3', and 500 template-4 sentences, 'template4', training identities, rng 74,000 + seed,
disjoint from every frozen split, out-of-world and planted set; k = 50 is the first 50 of the 500), committed with sha256 before any
model sees them.
- **BERT-tiny and BERT-mini pretrained, 'zs'** (models/bert_mrc_growth.py), from each seed's saved block model, every condition from
  the saved model again: 'level3' = k depth-3 sentences annotated with all six legend questions (the new question is X3); 'template4' =
  k template-4 sentences annotated with the 'zs' questions; regimes (a) continued fine-tuning on the new examples only, the declared
  recipe lr 1e-4, 3 epochs, and the ONE alternative lr 3e-5; (b) replay, the new examples added to the block's 30,000 training sentences,
  1 epoch from the saved model, lr 1e-4; bert_mrc.train's optimiser (AdamW, warm-up 10%, batch 32). Hyperparameter choices counted: 2
  per regime (lr, epochs), plus the alternative lr measured for (a).
- **The reader**, compositional case: no run; the level-3 question is answered by the saved 1a readers with zero examples and zero
  retraining (slot accuracy at depth 3 0.997 ± 0.002, X3 1.000; exact match 0.982 with the depth not given): that line is the row.
  Non-compositional case (vss/reader_growth.py): the k template-4 sentences added to the reader's 30,000 training sentences at fixed
  random positions (rng 75,000 + seed), retrained in full with the same seed (its only regime; V8.train's loop replicated so that the
  sentences can be added; with k = 0 it must reproduce 1a EQUAL, checked on the first seed of each run); 0 hyperparameter choices.
- **Measures**: the new question (level3: X3 slot accuracy at depth 3 and the six-question exact match, depth not given; template4: the
  template-4 cell of the new-formulation rows, slot accuracy and exact match); every report axis before and after (forgetting = the
  drop, per axis, with the seed std); wall-clock of the training ALONE on the GPU (these runs are scheduled with nothing else on the
  GPU); the number of examples; the number of hyperparameter choices.
- **Seeds (R50 / Q28)**: seeds 0-4 are the design set (inspected; template 4 is one of its inspected cells); the claim is on seeds 5-9:
  the 1a readers of seeds 5-9 and BERT-tiny pretrained 'zs' trained on seeds 5-9 for this purpose (models/bert_mrc.py --seed_start s
  --seeds 1 --save_all, one job per seed); BERT-mini on seeds 5-9 only if the slots are idle (otherwise its growth rows are on the
  design set only and labelled so).
- **Table** (as R49 gives it): rows = system x regime; columns = examples, minutes alone, hyperparameter choices, new-question accuracy,
  worst drop on an old row. Statements limited to this world. Written now, whatever the result: **replay removes forgetting at the price
  of full retraining** (the sentence stands whether or not forgetting appears under (a)).

## Amendment R51 (the owner, 2026-10-25 18:30): BERT training cut to what the result needs

BERT training that still runs or starts: the BERT-mini rows already launched (finished and reported); BERT-tiny pretrained 'zs' on
seeds 5-9 (5 jobs; the growth claim and the confirmatory comparisons on fresh seeds: eval/confirmatory.py --seed_start 5, reader 1a
of seeds 5-9 against BERT-tiny of seeds 5-9); the growth regimes of R49 on BERT-tiny only. **Not run: R51**, each:
- the 'all' rows of R40 (BERT-tiny and BERT-mini pretrained; declared above; the growth rows answer the same objection with examples);
- BERT-mini kinds given, seeds 3 and 4 (seeds 0-2 had started and are reported as a partial row of 3 seeds; BERT-mini pretrained and
  from scratch have all five seeds);
- BERT-mini in the growth rows (R49) and in the seeds 5-9 rows;
- consequently: the planted row and the confirmatory comparisons for BERT-mini kinds given (3 seeds only; the paired 5-seed tests need
  5), recorded as not run.

## R41 check: the compiled residual against sentence length, calibration rows only (eval/residual_length_check.py; logs/2026-10-25_1631_residual_length_check_3d8814d.txt)

Command `./run.sh eval/residual_length_check.py --seeds 5 --row 1a`, commit 3d8814d; the saved / retrained 1a readers (EQUAL on every
seed), the 2,000 calibration sentences of each seed only. Spearman(residual, token count) -0.146 ± 0.046 [-0.152 -0.062 -0.154 -0.205
-0.156]; mean residual at depth 0 / 1 / 2: 0.409 ± 0.009 / 0.455 ± 0.014 / 0.431 ± 0.009 (not rising on any seed); within one depth the
correlation is NEGATIVE, -0.42 to -0.89 (longer sentences of a depth have a smaller residual). R41's criterion (> 0.5 and rising means)
fails on every seed: **R41's length mechanism is refuted** (R43). As R43 writes: the compiled residual has a floor of about 0.43 on the
training distribution, the share of the state that the compiler's reconstruction never contains (the routes' deposits), and above that
floor it rises with the novelty of the form (UNSEEN transitions 0.90, depth 3 0.72, random order 0.99), not with the token count.
Label: MEASURED (calibration only, no report row).

## R42 / R43: the explained-tokens residual on the design set (eval/residual_explained.py --row 1a; logs/2026-10-25_1727_residual_explained_ecd7c47.txt)

Seeds 0-4 (design set, inspected), the saved 1a readers (EQUAL), the original (catalog) decoder, depth from the openers, thresholds at
2% false rejection on the calibration rows. MEASURED; the claim on seeds 5-9 is a separate row.

| score | AUROC in vs out (random + reference) | AUROC random / reference / nonce ph. | rejected random / reference / nonce ph. at 2% cal. | false rejection pooled |
|---|---|---|---|---|
| compiled residual | 0.876 ± 0.005 | 0.984 / 0.768 / 0.842 | 1.000 / 0.989 / 1.000 | 0.435 ± 0.015 |
| explained residual | 0.730 ± 0.007 | 0.996 / **0.464** / 0.952 | 0.997 / **0.026** / 1.000 | 0.112 ± 0.017 |

Per axis, explained residual false rejection: seen 0.029, new ids 0.025, depth 1-2 0.022-0.023, depth 3 0.051, role cues 0.022, new
formulations 0.308 / 0.386 (where the frame is wrong, tokens are left unexplained). Reading: with the catalog decoder the explained
residual removes most false rejections of form novelty (0.435 -> 0.112) but does not see the reference-world sentences at all (AUROC
0.464, rejected 0.026: their identities are placed into some frame, so nothing is left unexplained); it is a detector of UNPLACED
content, as R43 says, not of an unfamiliar form. R43's expected failure mode (random sentences filled into a frame pass) does not
occur for random order (0.997 rejected); it occurs for the reference world instead.

## R37: the reader with the six cue gates given as a trainable initialisation (vss/run_reader_basic.py --gates_given init_cues; logs/2026-10-25_1720_run_reader_basic_01a4715.txt; design set)

Command `./run.sh vss/run_reader_basic.py --gates_given init_cues --seeds 5 --save_all`, commit 01a4715, concurrent (training 392-998 s
per seed: not a cost number). Initial logit 2.3011 on to, by, from, '.', says, that (trainable). Final logits on every seed: to 10.8-11.0,
from 10.1-10.3, '.' 10.2-10.7, that 11.4-11.7, by 5.6-5.8, **says -8.4 to -8.7 (closed)**; open set on every seed: . by from that to.

| axis (slot accuracy) | R37 (initialised) | R14 (fixed open) | 1a (learned from closed) |
|---|---|---|---|
| seen / new ids / role cues / depth 1 / depth 2 | 1.000 | 1.000 | 1.000 |
| new form. seen trans. | 0.778 ± 0.031 [0.79 0.76 0.76 0.83 0.74] | 0.764 ± 0.029 | 0.864 ± 0.069 |
| new form. UNSEEN trans. | 0.799 ± 0.007 [0.79 0.80 0.81 0.80 0.79] | 0.789 ± 0.011 | 0.868 ± 0.040 |
| depth 3 slot accuracy; exact match six | 1.000; 0.998 ± 0.002 | 1.000; 0.999 | 0.997; 0.982 |
| one-to-one: new form. seen / UNSEEN; depth-3 exact match | 0.777 / 0.807; 1.000 | 0.768 / 0.796; 1.000 | 0.870 / 0.894; 0.995 |
| confident error depth 3 (tau 0.000) | 0.000 | 0.000 | 0.003 |

Reading, as R37 wrote it before the run: the row does NOT recover 1a's mean on new formulations (0.778 / 0.799 against 0.864 / 0.868; it
keeps R14's small seed spread), so **the R14 loss comes from the SET, not from the hardness**. The prediction written in questions.md
(17:15) holds: T4:B + T11:B errors appear on every R37 seed (289 / 274 / 253 / 278 / 310), as on every R14 seed and on 1a seed 2, the
readers with 'from' open; T4:C (138-176) and T7:A (171-180) as in every reader. The learned per-seed gate sets that beat the given set
(R37 asks for the list): 1a seed 0 {. by that to} 0.93 / 0.89, seed 3 {. by that to} 0.95 / 0.88, seed 1 {. that to} 0.82 / 0.89, seed 4
{. by that to} 0.84 / 0.89 (new form. seen / UNSEEN); all of them leave 'from' (and 'says', which no reader keeps open when it may close
it) closed; the one 1a seed that opened 'from' (seed 2, {by from that to}) is at 0.77 / 0.79, the R14 / R37 level. The cue word a seed
found useful that the legend did not give: none; the useful choice was to NOT open 'from' and 'says'. Diagnosis files:
results/dump_reader_basic_cuesinit_seed0-4.json, results/diagnosis_reader_basic_cuesinit.json (H2 core cells T10:C, T4:B, T4:C, T7:A
on every seed; H3 22-53 per seed). The claim on seeds 5-9 is a separate run (queued).

## R46: sentence candidates + one-to-one, design set (eval/sentence_candidates.py; logs/2026-10-25_1717_sentence_candidates_01a4715.txt (1a), _1719_ (R14))

Decoder only, the saved readers (original decoder EQUAL to the stored rows on every seed). Caption (R47): it changes the decoder's
candidate set, not the addresses; the H2 cause stays unfixed; the margin of a decision by elimination is the single-candidate floor,
not evidence. MEASURED; design set (inspected); the claim on seeds 5-9 is a separate run.

| axis (slot accuracy) | 1a + R46 (one-to-one) | 1a original | R14 + R46 | R14 original |
|---|---|---|---|---|
| seen / new ids / role cues / depth 1 / depth 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| new form. seen trans. | 0.873 [0.959 0.791 0.780 0.994 0.839] | 0.864 [0.93 0.82 0.77 0.95 0.84] | 0.775 | 0.764 |
| new form. UNSEEN trans. | 0.975 [0.975 0.979 0.978 0.958 0.982] | 0.868 | 0.983 | 0.789 |
| depth 3 slot accuracy; exact match | 0.998; 0.995 | 0.997; 0.982 | 1.000; 1.000 | 1.000; 0.999 |
| coverage / confident error at depth 3 (tau 0.000) | 0.997 / 0.001 | 1.000 / 0.003 | — | 1.000 / 0.000 |

(The per-role in-sentence top-1 without one-to-one: 1a 0.873 / 0.919, R14 0.799 / 0.920.) Out-of-world on the one-to-one frames:

| score | 1a + R46: AUROC in vs out; rejected random / reference / nonce ph.; false rejection | R14 + R46 |
|---|---|---|
| compiled residual | 0.976 ± 0.004; 1.000 / 0.989 / 0.855; 0.435 | 0.971; 1.000 / 0.989 / 0.848; 0.437 |
| explained residual | 0.975 ± 0.003; 0.990 / 0.908 / 1.000; **0.026 ± 0.004** | 0.974; 0.990 / 0.907 / 1.000; 0.024 |

R47's decision categories, 1a, new formulations, per seed (address / restriction / elimination; accuracy in each): UNSEEN trans.
2,557-2,890 / 90-393 / 250-290 (0.976-0.990 / 0.967-1.000 / 0.728-0.890); seen trans. 842-1,029 / 0-125 / 51-122 (0.84-1.00 / 0.26 /
0.33-0.94); depth 3: 4,302-4,319 by the address, 0 by restriction, 1-18 by elimination; seen: all 2,880 by the address. Per cell (1a,
seeds pooled, 900 decisions each): T7:A 46 by the address, 414 by restriction (411 right), 440 by elimination (440 right), 6 wrong;
T4:C 259 / 9 / 632 (632 right), 198 wrong; T4:A 195 eliminations all wrong (198 wrong); T4:B and T11:B 154 / 151 by restriction, all
right, 0 wrong; T10:A 344 wrong and T10:C 344 wrong (a swap the elimination cannot resolve when both addresses hold the other filler).
R14 cells the same in kind (T4:B 850 and T11:B 596 decided by restriction, all right; T10 607 wrong in each of A and C).
Against R46's expectations written before the run: T11 B and T4 B right by construction: yes (0 errors in 1a and R14); T7 A right
through one-to-one: yes (6 / 13 wrong of 900); T4 C and T10 C right whenever the other role is right: partly (T4: 198 / 117 wrong in A
and C together; T10: 344 / 607); "the in-world rows cannot fall": seen, ids, depths unchanged, but new formulations with seen
transitions fall on seed 1 (0.82 -> 0.79) and seed 2 (stays 0.78), the T10 swaps; the mean rises 0.864 -> 0.873. The expected cost
(random sentences filled into frames more often) is not visible in the rejection: the explained residual on these frames rejects random
0.990, reference 0.908, nonce 1.000 at 2.6% false rejection, the closest any reader row comes to the demo criterion (>= 0.90 per source
at <= 2% false rejection; not met: 2.4-2.6%).

**R37, the author's reading (R52), added to the row:** with the six gates as a trainable initialisation 'from' stays open on every
seed and T4:B + T11:B fail on every seed: the loss comes from the gate SET, not from the hardness. A cue that is a genuine opener in
one frame ('from' as the giver of receive) breaks the object-last reading in another, because the routes compose addresses from the
cue chain (H2); giving gates adds no information the routes can use, it only fixes which route-level failures appear. R14 and R37 are
reported as measured; no further gate rows; the demo's given-gates default stays withdrawn (TODO_DGX 2.9); the route-level fix is
tool-repository work after the stop.

## Equal-size block, the remaining rows: BERT-tiny kinds given; BERT-mini pretrained, from scratch, kinds given (models/bert_mrc.py; MEASURED)

Commands as the block (lr 1e-4, 5 epochs, batch 32, 'zs', --save_all): BERT-tiny kinds given one run of 5 seeds (commit 79fff34,
launched 16:09:02; its output is interleaved in logs/2026-10-25_1609_bert_mrc_79fff34.txt, its own stdout in
logs/2026-10-25_1609_bert_mrc_79fff34__tiny_kinds_stdout.txt); BERT-mini pretrained seeds 0-2 in one run (the _1609_ log and its
stdout copy) and seeds 3, 4 as one job each, BERT-mini from scratch and kinds given one job per seed (R45; logs
logs/2026-10-25_1735_ to _1749_bert_mrc_*.txt, one per job); kinds given seeds 3-4 not run (R51). All concurrent (up to 12): training
times are not cost numbers. Table from eval/mrc_table.py (mean ± std [per seed]).

| axis | BERT-tiny kinds (4.4 M) | BERT-mini pretrained (11.2 M) | BERT-mini from scratch (11.2 M) | BERT-mini kinds, seeds 0-2 |
|---|---|---|---|---|
| seen / new ids / depth 1 / depth 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| new form. seen trans. | 0.931 ± 0.080 [0.833 0.990 0.833 1.000 1.000] | 1.000 | 0.729 ± 0.101 [0.667 0.561 0.816 0.819 0.785] | 1.000 |
| new form. UNSEEN trans. | 0.970 ± 0.041 | 0.982 ± 0.020 [1.000 0.944 0.980 1.000 0.985] | 0.891 ± 0.053 | 0.953 ± 0.035 |
| new form. role cues | 1.000 | 1.000 | 0.860 ± 0.063 | 1.000 |
| depth 3 slot accuracy (X3) | 0.786 ± 0.048 (0.000) | 0.833 ± 0.001 (0.000) | 0.635 ± 0.034 (0.000) | 0.827 (0.000) |
| exact match depth 3, depth NOT given: six / inner | 0.000 / 0.722 ± 0.278 [0.736 0.625 0.250 1.000 1.000] | 0.000 / 1.000 | 0.000 / 0.142 ± 0.121 [0.306 0.021 0.135 0.000 0.250] | 0.000 / 0.963 |
| confident error depth 3 (tau 0.000) | 0.214 (trained labels 0.057) | 0.167 (0.000) | 0.365 (0.238) | 0.173 (0.007) |
| R17 (i) NULL refusal random / reference / nonce | 0.285 / 0.000 / 0.243 | 0.288 / 0.038 / 0.272 | 0.286 / 0.000 / 0.301 | 0.275 / 0.000 / 0.253 |
| R17 (i) 2% threshold: random / reference / nonce; false rejection | 0.926 / 0.950 / 0.925; 0.259 | 0.991 / 0.922 / 0.908; 0.148 | 0.970 / 0.948 / 0.955; 0.484 | 0.987 / 0.889 / 0.899; 0.135 |
| last-epoch training loss (mean) | 3.5e-5 | 5.4e-6 | 3.3e-5 | 1.9e-6 |
| size fp32 | 17.6 MB | 44.7 MB | 44.7 MB | 44.7 MB |

Convergence clause (R36): BERT-mini from scratch / pretrained last-epoch loss 6.0x (row means), not more than 10x: no 20-epoch row.
**Withdrawal condition (R30 item 3), written before the block: an encoder of the reader's size trained FROM SCRATCH reaching an inner
frame >= 0.99 at depth 3 with the depth not given would withdraw the claim. Measured: BERT-tiny 0.173, BERT-mini 0.142: the claim
"same-size models trained from scratch do not compose this world" stands at 4.4 M and 11.2 M parameters.** BERT-small is not run
(R36: tiny and mini do not disagree; and R48).
Reading (BERT first). Pretrained BERT-mini is at ceiling on every new-formulation axis except UNSEEN transitions (0.982), above the reader
there (0.868; the R46 decoder 0.975), and reaches inner frame 1.000 at depth 3 with the depth not given, as BERT-base's tagger did; no MRC
row answers the level-3 question that was never asked in training (X3 0.000 on every seed of every row), so the six-question exact
match at depth 3 is 0.000 for every BERT row against the reader's 0.982 (one-to-one 0.995). Kinds given do not change the pretrained
picture (tiny: inner 0.722 against 0.972 without, with a large seed spread; mini seeds 0-2: 0.963 against 1.000). Pretraining is what
separates the rows at equal size: from scratch the inner frame falls to 0.14-0.17 and new formulations to 0.63-0.89. The confirmatory
comparisons follow in their own section.

**R46 / R42 on the design set, accepted by R53; three sentences written before any seeds 5-9 row is read:** (1) the design-set
numbers of the reader 1a with sentence candidates + one-to-one (UNSEEN formulations 0.868 -> 0.975, depth-3 exact match 0.995 at
coverage 0.997 with confident error 0.001, explained-residual AUROC 0.975, rejected 0.99 / 0.91 / 1.00 at 2.6% false rejection) are
DESIGN-SET numbers only; the claim is the seeds 5-9 row. (2) The demo criterion (>= 0.90 per source at <= 2% false rejection) is NOT
met: missed by 0.6 points of false rejection. (3) The largest remaining error is one cell, T10 (fronted 'from': A and C swapped, both
addresses holding the other filler; 344 wrong decisions in each of A and C over the five seeds), which elimination cannot resolve and
only the route fix can. Correction of fact to R53's wording, from the same files: T4 also keeps errors (198 wrong decisions in each of
A and C pooled, where both addresses hold A's filler and the elimination assigns it to C, leaving A wrong); every other cell has at most
13 wrong decisions of 900. Decision provenance stays in the table: T7:A is right by restriction (411) and elimination (440), not by
the address (43).

## 1a on the fresh splits, seeds 5-9 (R13 / R39; vss/run_reader_basic.py --seed_start 5; logs/2026-10-25_1721_run_reader_basic_01a4715.txt; CLAIM SET)

Command `./run.sh vss/run_reader_basic.py --seed_start 5 --seeds 5 --save_all`, commit 01a4715 (1a's code; row "VSS reader (reference)
[fresh splits, seeds 5-9]", tag reader_basic_fresh, never pooled with seeds 0-4), concurrent (training 533-1,000 s: not a cost
number). Learned open gates: seeds 5, 7, 9 {by from that to}; seed 6 {. by that to}; seed 8 {. that to}. MEASURED.

| axis (slot accuracy) | seeds 5-9 | seeds 0-4 (1a, design set) |
|---|---|---|
| seen / new ids / role cues / depth 1 / depth 2 | 1.000 | 1.000 |
| new form. seen trans. | 0.831 ± 0.055 [0.80 0.93 0.78 0.84 0.81] | 0.864 ± 0.069 |
| new form. UNSEEN trans. | 0.835 ± 0.051 [0.78 0.89 0.82 0.90 0.79] | 0.868 ± 0.040 |
| depth 3 slot accuracy; exact match six (levels as labels) | 0.999 ± 0.002; 0.992 ± 0.010 [1.000 0.975 1.000 0.986 0.999] | 0.997; 0.982 |
| one-to-one: new form. seen / UNSEEN; depth-3 exact match | 0.840 / 0.856; 0.998 ± 0.002 | 0.870 / 0.894; 0.995 |
| confident error at depth 3 (tau 0.000, coverage 1.000) | 0.001 | 0.003 |

The fresh seeds reproduce 1a's picture: depth 3 at ceiling, new formulations 0.83-0.84, lower by 0.03 in the mean, with three of the
five readers opening 'from' (1a: one of five).

## LLM reference, in-context (R27 item 2, R29, R32, R33; eval/llm_reference.py --call; logs/2026-10-25_1608_llm_reference_79fff34.txt): reference (in-context LLM), not a baseline

The restarted run (the first one stopped at 1,300 calls, above). Model claude-sonnet-5-5 through the Anthropic API (every response
reports that model; no fallback), 2026-10-25 16:08-18:51, thinking {type: between_tools} at effort low (the lowest setting; no
temperature control exists for this model), the frozen 10-shot prompt (eval/llm_reference_prompt.txt; ten training sentences of seed
0; not data-matched) as a cached system block, 200 report items per axis of seed 0 (1,600 items), three repeats, 4,800 calls, every
call end_turn, no refusal. Prices read on the pricing page at call time (USD per M tokens: input 2, output 10, 5-minute cache write 2.5,
cache read 0.10). Tokens per call: 29.2 input, 34.3-34.6 output, 819 cache read (the cache DID engage on every call after the first,
against R33's expectation; the cache-write field is 0.5 tokens per call on average in repeat 1). Spend 2.3275 USD (0.00048 per call);
with the stopped first run (0.62-0.65 USD) about 2.95 USD of the 20 USD cap. About 2.0 s per call (API-bound, beside the GPU runs).
Label: MEASURED.

| axis (seed 0, n 200 each) | slot accuracy, repeats 1 / 2 / 3 | exact match six roles | wrong values; false fills |
|---|---|---|---|
| seen | 0.997 / 0.997 / 0.997 | 0.995 | 2 / 2 / 2; 0 |
| new ids | 0.995 / 0.993 / 0.995 | 0.990 | 2 / 3 / 1; 0 |
| new form. seen trans. | 0.950 / 0.963 / 0.950 | 0.925 / 0.945 / 0.925 | 30 / 22 / 29; 0 |
| new form. UNSEEN trans. | 1.000 | 1.000 | 0; 0 |
| new form. role cues | 1.000 | 1.000 | 0; 0 |
| depth 1 | 0.978 / 0.980 / 0.975 | 0.955 / 0.960 / 0.950 | 17 / 14 / 18; 0 |
| depth 2 | 0.982 / 0.990 / 0.988 | 0.955 / 0.975 / 0.970 | 16 / 10 / 12; 0 |
| depth 3 (all six questions asked, depth not given) | 0.985 | 0.955 | 18 / 18 / 17; 0 |

Determinism: frames identical across the three repeats on 0.978 of the 1,600 items. Reading: the in-context reference answers the
level-3 question it was never shown in the ten examples (depth 3 exact match 0.955 with all six questions asked), never fills an absent
role (0 false fills), and is at 0.95-1.00 on every new-formulation axis; it is the accuracy ceiling of this row family, on one seed's
subset, at about 0.5 thousandths of a dollar and 2 s per sentence through an API, against a reader of 13 MB. It is not compared
statistically with the trained models (one seed, a 200-item subset, 10-shot).

## The planted-sets row, query emission (R26 / Q19 / R48 item 4; eval/planted.py; logs/2026-10-25_1848_ to _1855_planted_*.txt; MEASURED)

Rules as declared (depth from the openers; 'undecided' when the planted role's margin is below the 2% quantile of the model's
calibration margins; 'unsupported' for every identity placed in no role; no trained coverage head for any model). Seeds 0-4, the frozen
planted sets (300 + 300 per seed). mean ± std over 5 seeds; "needless" = a right value queried; "conf. err." = a wrong value at or above
the threshold; hallucination rate = invented planted decisions + confident errors, over all decisions.

| model | undecided: planted queried [per seed] | other roles: needless / conf. err. | halluc. | unsupported: planted queried [per seed] | needless 'unsupported' on original ids | other roles: needless / conf. err. | halluc. |
|---|---|---|---|---|---|---|---|
| reader 1a | 0.513 ± 0.163 [0.470 0.380 0.833 0.450 0.433] | 0.171 / 0.000 | 0.108 | **0.984 ± 0.005** [0.977 0.993 0.983 0.983 0.983] | 0.005 | 0.216 / 0.000 | 0.003 |
| reader 1a + R46 decoder | 0.522 ± 0.180 [0.470 0.380 0.877 0.450 0.433] | 0.154 / 0.000 | 0.106 | **0.983 ± 0.006** | 0.004 | 0.206 / 0.000 | 0.003 |
| BERT-tiny pretrained | **0.991 ± 0.013** [0.967 1.000 0.990 1.000 1.000] | 0.352 / 0.029 | 0.024 | 0.835 ± 0.116 [0.883 0.610 0.920 0.920 0.840] | 0.094 | 0.431 / 0.028 | 0.053 |
| BERT-tiny from scratch | 0.837 ± 0.063 | 0.223 / 0.072 | 0.092 | 0.705 ± 0.062 | 0.129 | 0.254 / 0.056 | 0.100 |
| BERT-tiny kinds given | 0.985 ± 0.010 | 0.313 / 0.031 | 0.027 | 0.663 ± 0.121 | 0.132 | 0.365 / 0.029 | 0.085 |
| BERT-mini pretrained | **1.000 ± 0.000** | 0.360 / 0.000 | 0.000 | 0.747 ± 0.066 [0.783 0.797 0.800 0.733 0.623] | 0.113 | 0.399 / 0.000 | 0.046 |
| BERT-mini from scratch | 0.945 ± 0.021 | 0.287 / 0.061 | 0.060 | 0.690 ± 0.087 | 0.134 | 0.307 / 0.049 | 0.097 |

Criterion (R26, the demo's): planted unsupported reported as queries >= 0.90: met by the reader (0.98 on every seed), by no MRC row
(0.66-0.84). Reading. The two kinds of query split the models: the reader leaves an appended identity unplaced (0.98) because no
address holds it, and the MRC rows answer some question with it in 16-34% of the sentences (and leave 9-13% of the sentences' own
identities unplaced, needless queries); an MRC model's margin flags two equally placed candidates (undecided 0.99-1.00 pretrained),
which the reader's margin does in only half the cases (0.51; the inserted identity sits in an address the role's readout reads with
little interference). Every MRC row pays for its undecided queries with 0.29-0.43 needless queries on the other roles (the 2%
calibration threshold sits on margins that fall on any perturbed sentence); the reader 0.15-0.22. The trained coverage head of R26 was
not built (no new training in the declared set), so the MRC 'unsupported' numbers are those of the untrained test, as for the reader.

## BERT-tiny pretrained 'zs' on the fresh splits, seeds 5-9 (R50 / R51; models/bert_mrc.py --seed_start s --seeds 1; logs/2026-10-25_1838_ to _1843_bert_mrc_*.txt; CLAIM SET)

One job per seed, the block's configuration, concurrent (training 297-664 s: not a cost number). For the growth claim and the
confirmatory comparisons on fresh seeds; never pooled with seeds 0-4. MEASURED.

| axis | BERT-tiny pretrained, seeds 5-9 | seeds 0-4 | reader 1a, seeds 5-9 |
|---|---|---|---|
| seen / new ids / role cues / depth 1 / depth 2 | 1.000 | 1.000 | 1.000 |
| new form. seen trans. | 0.834 ± 0.105 [0.667 1.000 0.833 0.833 0.838] | 0.884 | 0.831 |
| new form. UNSEEN trans. | 0.986 ± 0.021 | 0.968 | 0.835 |
| depth 3 slot accuracy (X3) | 0.824 ± 0.017 (0.000) | 0.829 (0.000) | 0.999 (1.000) |
| exact match depth 3, depth NOT given: six / inner | 0.000 / 0.943 ± 0.104 [0.736 1.000 0.979 1.000 1.000] | 0.000 / 0.972 | 0.992 / 0.992 |
| confident error depth 3 | 0.176 | 0.171 | 0.001 |
| R17 (i) 2% threshold: random / reference / nonce; false rejection | 0.970 / 0.965 / 0.954; 0.318 | 0.992 / 0.943 / 0.954; 0.280 | (R42 fresh row) |

**Confirmatory comparisons, first attempt failed (logs/2026-10-25_1857_confirmatory_cb9ef69.txt, BERT-tiny; _1858_ BERT-mini and
_1859_confirmatory_d613149.txt, fresh seeds, both stopped by hand when the first failed):** TypeError in json.dump (numpy integers in
the result) after the first pair's numbers were printed; no result file kept (a truncated one was deleted). Fix: a JSON default for
numpy scalars (separate commit); the three runs are started again. The printed numbers of the first attempt are not used.


## Deviation R54: a second-identity column for every planted row, and the reader's adjacency rule (declared 2026-10-25 before it runs)

(1) Scoring, every model (R54 item 1): in the 'undecided' set, whether the inserted identity was left in no role (a query of another
kind: the reader's 'unplaced', the MRC rows' untrained 'unsupported' test); the hallucination rate unchanged as defined. The planted
rows are re-scored with it (eval/planted.py, results logs/results_planted2_*; the earlier columns are recomputed identically).
(2) R54's text, verbatim:

2. **Decoder rule, declared now, claim on seeds 5-9 (R39):** an identity left in no role, adjacent (the token before or after) to the
   filler of a role whose kinds admit it, turns that role into 'undecided' with both as candidates and the role's margin set to the gap
   between them at the role's address; this is what a person does with "p3 p5 gives o1 to p2". Expected effect, written before the run:
   'undecided' planted queried rises towards the share of sentences where the second identity is left unplaced (to be read from
   addition 1); needless queries on the other roles unchanged; the 'unsupported' set unchanged except where the appended identity is
   adjacent to a filler of an admitting role (", p7 ." after the recipient: those become 'undecided' instead of 'unsupported', a kind
   change counted as still queried, reported separately). Runs with the R46 decoder; both provenance categories kept.

Implementation: eval/planted.py --model reader_r54 (the R46 decoder, then eval/planted.r54_adjacent: 'adjacent' = the token right before
or after; the role's margin = the absolute gap between the two candidates' in-sentence scores at the role's address; a role so marked
counts as queried). Design set seeds 0-4 (inspected) and the CLAIM on seeds 5-9, which needs frozen planted sets for seeds 5-9:
data/planted_seed5-9.json (data/make_planted_sets.py --seed_start 5, the same generator, additionally disjoint from the growth sets),
committed with sha256 before any model sees them, and the 1a readers of seeds 5-9.

## The reader rows on the CLAIM SET, seeds 5-9 (R39; R37, R42 / R43, R46; MEASURED)

**R37 on seeds 5-9** (vss/run_reader_basic.py --gates_given init_cues --seed_start 5; logs/2026-10-25_1844_run_reader_basic_fda2a9a.txt):
the same gate trajectory on every seed (to / from / . / that 10.1-11.6, by 5.6-5.9, says -8.1 to -8.6); new formulations 0.782 ± 0.021
[0.80 0.79 0.76 0.76 0.81] / 0.787 ± 0.013 [0.78 0.79 0.80 0.80 0.77] against 1a on the same seeds 0.831 / 0.835; depth-3 exact match
0.998. The design-set conclusion holds on fresh seeds: the given set lowers new formulations; it is the set, not the hardness.

**R42 / R43 on seeds 5-9** (eval/residual_explained.py --row 1a_fresh --seed_start 5; logs/2026-10-25_1845_residual_explained_fda2a9a.txt),
the catalog decoder: compiled residual AUROC in vs out 0.883 ± 0.005 (random 0.979, reference 0.787, nonce ph. 0.840), rejected 1.000 /
1.000 / 1.000 at false rejection 0.440; explained residual AUROC 0.724 ± 0.004 (random 0.996, **reference 0.451**, nonce ph. 0.942),
rejected 0.998 / **0.024** / 1.000. The design-set reading is confirmed: with the catalog decoder the explained residual does not see
the reference world (the decoded frame places its identities).

**R46 on seeds 5-9** (eval/sentence_candidates.py --row 1a_fresh --seed_start 5; logs/2026-10-25_1847_sentence_candidates_fda2a9a.txt;
reported against 1a on the same seeds):

| quantity | 1a + R46 (one-to-one), seeds 5-9 | 1a original decoder, seeds 5-9 | design set (seeds 0-4), 1a + R46 |
|---|---|---|---|
| new form. seen trans. | 0.867 [per-role in-sentence 0.855 ± 0.043] | 0.831 ± 0.055 | 0.873 |
| new form. UNSEEN trans. | **0.973** [per-role 0.920 ± 0.011] | 0.835 ± 0.051 | 0.975 |
| seen / ids / role cues / depth 1-2 | 1.000 | 1.000 | 1.000 |
| depth-3 exact match; coverage / confident error | 0.998; 0.999 / 0.000 | 0.992; 1.000 / 0.001 | 0.995; 0.997 / 0.001 |
| compiled residual: AUROC; rejected random / ref. / nonce; false rejection | 0.979 ± 0.003; 1.000 / 1.000 / 0.854; 0.440 | 0.883; 1.000 / 1.000 / 1.000; 0.440 | 0.976; 1.000 / 0.989 / 0.855; 0.435 |
| explained residual: AUROC; rejected random / ref. / nonce; false rejection | **0.979 ± 0.003; 0.988 / 0.930 / 1.000; 0.026 ± 0.003** [0.028 0.020 0.029 0.029 0.026] | 0.724; 0.998 / 0.024 / 1.000; (R42 row) | 0.975; 0.990 / 0.908 / 1.000; 0.026 |

R47's categories on seeds 5-9 (new formulations, UNSEEN trans., per seed): by the address 2,510-2,917 (accuracy 0.977-0.990), by
restriction 65-465 (0.938-1.000), by elimination 212-297 (0.768-0.891); depth 3 0-18 eliminations. **Claim, as measured on fresh seeds:**
the decoder-only change (candidates restricted to the sentence's admissible words + one-to-one) lifts UNSEEN-transition formulations
from 0.835 to 0.973 and, with it, the explained residual separates in- from out-of-world at AUROC 0.979 and rejects 0.99 / 0.93 / 1.00
of the random / reference / nonce sentences at 2.6% false rejection on the in-world report rows. **The demo criterion (>= 0.90 per
source at <= 2% false rejection) is NOT met** (false rejection 2.0-2.9% per seed, mean 2.6%). Seen-transition formulations stay at 0.867
(the T10 cell; the route fix is tool work after the stop).

## Confirmatory comparisons (R30 item 8), reader 1a vs BERT-tiny, seeds 0-4 (eval/confirmatory.py; logs/2026-10-25_1905_confirmatory_a5047ce.txt; MEASURED)

Pre-specified: (i) six-question exact match at depth 3, depth not given; (ii) confident-error rate at depth 3 at the reader's coverage
(here 1.000 for both: tau 0.000); (iii) out-of-world AUROC, reader's compiled residual vs the encoder's Mahalanobis score. Two-level
bootstrap (1,000), Holm per pair at 0.05; the 4-df t and the sign test beside (not relied on; with 5 seeds the smallest two-sided sign
p is 0.0625).

| pair: reader - model | (i) exact match six, diff [95%], p | (ii) confident error, diff [95%], p | (iii) AUROC, diff [95%], p | Holm |
|---|---|---|---|---|
| BERT-tiny pretrained | +0.982 [+0.972, +0.991], p < 0.001 | -0.168 [-0.172, -0.164], p < 0.001 | +0.054 [+0.017, +0.087], p 0.002 (seeds ahead 4 / 1) | all three rejected |
| BERT-tiny from scratch | +0.982, p < 0.001 | -0.355 [-0.376, -0.331], p < 0.001 | +0.125 [+0.104, +0.150], p < 0.001 | all three rejected |
| BERT-tiny kinds given | +0.982, p < 0.001 | -0.211 [-0.255, -0.172], p < 0.001 | +0.070 [+0.060, +0.081], p < 0.001 | all three rejected |

Secondary scores (mean over seeds): reader compiled residual AUROC 0.876, margin 0.887, explained residual 0.730 (catalog decoder);
BERT-tiny pretrained Mahalanobis 0.822, **MSP 0.915**, NULL gap 0.591; from scratch 0.751 / 0.689 / 0.613; kinds 0.806 / 0.907 / 0.607.
AURC at depth 3 (margin as confidence): reader 0.000 (errors 0.3%), BERT-tiny 0.157 / 0.233 / 0.157. Reading, stated plainly:
comparison (i) is decided by construction of the 'zs' training (no MRC row is ever asked the level-3 question; X3 0.000): it measures
that the reader answers a question it was never trained on and the extractive models do not, not a margin of skill on a shared task;
(ii) follows from (i) (the X3 errors are confident); (iii) is the comparison with content, and it favours the reader against the
pre-specified Mahalanobis score, but BERT-tiny pretrained's MSP (0.915) is above the reader's compiled residual (0.876) with the
catalog decoder; with the R46 decoder the reader's residuals reach 0.976-0.979 (design / claim set; not a pre-specified comparison).

**R54 caption addition (R55):** the adjacency rule's reference implementation is vss-demo commit f46b8d4 (tool/sop/infer.py
undecided_adjacent); vss-demo is not a submodule here, so eval/planted.r54_adjacent transcribes R54's definition. The explained
residual's reference is vss-demo a1f4564 (R42). The verb-order route of vss-demo 434fa40 ('vorder') needs retraining and is NOT a row
here (R55; after the stop).


## Deviation R56: the reader 1a + the verb-order route 'vorder' (declared 2026-10-25 before it runs; code vss/v8_vorder.py, vss/run_reader_vorder.py)

The address-level fix of H2 as a declared row (R38 protocol). Reference implementation: vss-demo commit 434fa40 (experiment/v8_gates.py,
the 41-line diff reproduced in responses R56); it is transcribed here as a SUBCLASS of v8_gates.V8 (vss/v8_vorder.py) because
third_party/vsa-semantic-space is pinned at e531283 and read-only. set_vorder / vorder_index / the extra term exactly as R56 lists them;
V8.train replicated with vo added to the optimiser; verb categories: every verb kind and participle of the basic world plus says /
claims (accepts borrows buys claims gets given gives lends lent offered offers receives says sees sells sends sent shown shows sold).
Everything else is 1a: the same seeds, generators, N 30,000, batch 8, lr 0.02, gates learned from the closed start, no selection. Size:
vo adds 2 x 32 x 32 x 4 x 1,024 = 8.4 M phases to the reader's 3.3 M (stated; the route is not small). Check before the row: the subclass
WITHOUT set_vorder trained as 1a must reproduce the stored 1a row EQUAL on the first seed of each run (--check; the run stops otherwise);
set_vorder draws once from the model's rng as the reference does, so the vorder row's training sentences are not 1a's. Design on seeds 0-4
(inspected), CLAIM on seeds 5-9 (tags reader_vorder / reader_vorder_fresh). Evaluated with eval/sentence_candidates.py and
eval/residual_explained.py on the saved models (rows 'vorder', 'vorder_fresh').

R56's evaluation and expectations, verbatim:

Evaluate exactly as the R46 claim rows: per-role rule and one-to-one, with and without the R46 decoder, the explained residual, the per-cell
table (T4:C, T7:A, T10:A+C, T4:B+T11:B) with the three decision categories, so that the row says whether the ADDRESS now holds the filler
(decided by the address) rather than elimination. Expected, written before the run: T4:C and T7:A decided by the address on most items;
T10 improved; nothing else moves outside the seed std; if the in-world rows fall by more than one seed std, the route is rejected and
the row says so.

## R49 legend growth, the reader (vss/reader_growth.py; logs/2026-10-25_1908_reader_growth_7324df2.txt (seeds 0-4), _1934_reader_growth_c7f1d5e.txt (seeds 5-9); MEASURED)

Both runs ALONE on the GPU (the queue held every other job). Check: with k = 0 the replicated training reproduces the saved reader
EQUAL (seed 0 and seed 5; drop 0.000 on every axis). Full retraining, the reader's only regime, 0 hyperparameter choices.
Compositional case (the level-3 question): no run; 0 examples, 0 s, X3 1.000 at depth 3 on every seed of both ranges (exact match
0.982 / 0.992). Non-compositional case (template 4 added with k sentences):

| seeds | k | training s alone | template-4 cell slot: before -> after | exact after | worst drop on an old row (mean [per seed]) | UNSEEN new form. after |
|---|---|---|---|---|---|---|
| 0-4 (design) | 50 | 135.2 ± 0.2 | 0.632 -> 1.000 | 0.999 | 0.008 [0.027 0.000 0.004 0.008 0.000] | 0.939 |
| 0-4 (design) | 500 | 136.8 ± 0.2 | 0.632 -> 1.000 | 1.000 | 0.017 [0.005 0.002 0.002 0.042 0.034] | 0.930 |
| 5-9 (claim) | 50 | 134.9 ± 0.3 | 0.541 -> 0.999 | 0.998 | 0.002 [0.003 0.006 0.000 0.000 0.000] | 0.939 |
| 5-9 (claim) | 500 | 136.6 ± 0.2 | 0.541 -> 1.000 | 1.000 | 0.006 [0.001 0.000 0.006 0.021 0.003] | 0.939 |

The worst drops fall on new formulations with seen transitions or depth 3 and are within those rows' seed std (0.05-0.07 and 0.002-
0.010); seen, new ids, role cues and depths 1-2 move by 0.000. (The UNSEEN-formulation axis rises because template 4 is one of its
templates and is no longer unseen.) For the reader a new form costs a full retraining of 135-137 s alone and k = 50 examples, with no
hyperparameter to choose; a new question composed from known parts costs nothing.

## Confirmatory comparisons (R30 item 8), reader 1a vs BERT-mini (seeds 0-4) and vs BERT-tiny on the CLAIM SET (seeds 5-9) (eval/confirmatory.py; logs/2026-10-25_2003_confirmatory_91932d8.txt, _2004_confirmatory_91932d8.txt; MEASURED)

| pair: reader - model | (i) exact match six, depth 3 not given | (ii) confident error at matched coverage | (iii) AUROC residual vs Mahalanobis | Holm |
|---|---|---|---|---|
| BERT-mini pretrained (0-4) | +0.982 [+0.972, +0.991], p < 0.001 | -0.164 [-0.165, -0.162], p < 0.001 | **+0.026 [-0.024, +0.069], p 0.306** (seeds ahead 4 / 1) | (i), (ii) rejected; **(iii) not rejected** |
| BERT-mini from scratch (0-4) | +0.982, p < 0.001 | -0.362 [-0.392, -0.332], p < 0.001 | +0.055 [+0.029, +0.075], p < 0.001 | all three rejected |
| BERT-tiny pretrained, CLAIM SET (5-9) | +0.992 [+0.982, +1.000], p < 0.001 | -0.175 [-0.193, -0.164], p < 0.001 | +0.046 [+0.017, +0.082], p < 0.001 | all three rejected |

Secondary (means): BERT-mini pretrained Mahalanobis 0.850, MSP 0.895, NULL gap 0.622, AURC 0.082; BERT-mini from scratch 0.821 / 0.661
/ 0.570 / 0.264; BERT-tiny seeds 5-9 0.837 / **0.898** / 0.625 / 0.153; the reader (catalog decoder) residual 0.876 (0-4) / 0.883
(5-9), margin 0.887 / 0.895, explained residual 0.730 / 0.724, AURC 0.000. Withdrawal condition: BERT-mini from scratch inner frame 0.142
< 0.99, the claim stands. Reading: on the out-of-world comparison with content (iii), the reader's compiled residual is ahead of the
Mahalanobis score of every small model except BERT-mini pretrained, where the difference (+0.026) is not significant; the pretrained
models' MSP (0.895-0.915) is above the reader's compiled residual with the catalog decoder (0.876-0.883) on both seed ranges; (i) and
(ii) are decided by the zero-shot level-3 question, as stated with the BERT-tiny pairs.

## R54: the second-identity column and the reader's adjacency rule (eval/planted.py; logs/2026-10-25_20*_planted_*.txt; results logs/results_planted2_*; MEASURED)

Every earlier column is recomputed identically. New column: in the 'undecided' set, the share of sentences whose inserted identity was
left in no role (a query of another kind). The adjacency rule (reference implementation vss-demo f46b8d4) runs with the R46 decoder.

| model (seeds) | undecided: planted queried [per seed] | inserted identity left in no role | other roles needless | halluc. | unsupported: planted queried (as 'undecided') | halluc. |
|---|---|---|---|---|---|---|
| reader 1a (0-4) | 0.513 ± 0.163 | 0.959 | 0.171 | 0.108 | 0.984 (0.000) | 0.003 |
| reader 1a + R46 (0-4) | 0.522 ± 0.180 | 0.951 | 0.154 | 0.106 | 0.983 (0.000) | 0.003 |
| **reader 1a + R46 + R54 rule (0-4, design)** | **0.997 ± 0.005** [1.000 0.987 1.000 1.000 0.997] | 0.951 | 0.154 | 0.001 | 0.983 (0.000) | 0.003 |
| reader 1a + R46 (5-9) | 0.708 ± 0.204 | 0.962 | 0.268 | 0.065 | 0.985 (0.000) | 0.003 |
| **reader 1a + R46 + R54 rule (5-9, CLAIM)** | **0.999 ± 0.003** [1.000 0.993 1.000 1.000 1.000] | 0.962 | 0.268 | 0.000 | 0.985 (0.000) | 0.003 |
| BERT-tiny pretrained (0-4) | 0.991 ± 0.013 | 0.407 ± 0.231 | 0.352 | 0.024 | 0.835 | 0.053 |
| BERT-tiny pretrained (5-9) | 0.993 ± 0.010 | 0.391 ± 0.255 | 0.394 | 0.015 | 0.736 ± 0.210 | 0.062 |
| BERT-tiny from scratch / kinds (0-4) | 0.837 / 0.985 | 0.760 / 0.537 | 0.223 / 0.313 | 0.092 / 0.027 | 0.705 / 0.663 | 0.100 / 0.085 |
| BERT-mini pretrained / from scratch (0-4) | 1.000 / 0.945 | 0.605 / 0.657 | 0.360 / 0.287 | 0.000 / 0.060 | 0.747 / 0.690 | 0.046 / 0.097 |

Against R54's expectations written before the run: 'undecided' planted queried rises towards (here beyond) the share of sentences in
which the second identity is left unplaced (0.95-0.96; the remaining queries come from margins below the threshold): yes, 0.997 /
0.999; needless queries on the other roles unchanged: yes (0.154 / 0.268, identical); the 'unsupported' set unchanged: yes (no appended
identity is adjacent to a filler: it always follows ',', so 0 kind changes). The reader's silent failure on the undecided set was a
misnamed query, not a silent one: in 95-96% of the sentences it left the inserted identity unplaced, against 39-76% for the MRC rows,
which place it in some role in 24-61% of the sentences.

## R56: the reader 1a + the verb-order route (vss/run_reader_vorder.py; logs/2026-10-25_2000_run_reader_vorder_0ea758b.txt (seeds 0-4), logs/2026-10-25_2001_run_reader_vorder_0ea758b.txt (seeds 5-9); evaluations logs/2026-10-25_2038_ / _2039_sentence_candidates_a773e88.txt, _2040_ / _2041_residual_explained_a773e88.txt; MEASURED)

Checks: the subclass without the route reproduces 1a EQUAL on seed 0 and on seed 5. Training 261-458 s per seed (concurrent with a few
evaluation jobs: not a cost number). Learned open gates as 1a's kind of sets ({. by that to} on 6 of 10 seeds, 'from' open on 4).

| axis (slot accuracy) | vorder, per-role (0-4 / 5-9) | vorder + R46 decoder, one-to-one (0-4 / 5-9) | 1a per-role (0-4 / 5-9) | 1a + R46 one-to-one (0-4 / 5-9) |
|---|---|---|---|---|
| seen / ids / role cues / depth 1 / depth 2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| new form. seen trans. | **0.656 ± 0.020 / 0.664 ± 0.005** | 0.667 / 0.667 | 0.864 ± 0.069 / 0.831 ± 0.055 | 0.873 / 0.867 |
| new form. UNSEEN trans. | 0.912 ± 0.021 / 0.912 ± 0.013 | **0.999 / 0.998** | 0.868 / 0.835 | 0.975 / 0.973 |
| depth-3 exact match | 0.996 / 0.997 | 0.999 / 1.000 | 0.982 / 0.992 | 0.995 / 0.998 |
| explained residual (R46 frames): AUROC; rejected random / ref. / nonce; false rejection | — | 0.974 / 0.979; 0.990 / 0.907 / 1.000 (0-4), 0.988 / 0.929 / 1.000 (5-9); 0.025 / 0.026 | — | 0.975 / 0.979; 0.026 / 0.026 |
| catalog decoder: compiled residual AUROC; explained residual AUROC (reference) | 0.873 / 0.876; 0.732 (0.466) / 0.732 (0.466) | — | 0.876 / 0.883; 0.730 (0.464) / 0.724 (0.451) | — |

Per cell (design set, R47 categories with the R46 decoder, 900 decisions each, seeds pooled): **T7:A 899 decided by the address, all
right** (1a: 46 by the address, the rest by restriction / elimination); **T4:C 34 by the address, 845 by elimination** (1a: 259 / 632),
7 wrong; T4:A 7 wrong (1a 198); T4:B, T11:B right (by the address 842 / 859); **T10:A and T10:C wrong on all 900 each, decided by the
address** (1a: 344 wrong each). Per-role rule, wrong decisions on new formulations per seed: 560 / 612 / 824 / 625 / 658 (1a 418 / 563
/ 934 / 431 / 526), of which T10 358-360, T4:C 169-180, T7:A 0-1.
Against R56's expectations written before the run: T7:A decided by the address on most items: **yes** (899 / 900); T4:C decided by
the address on most items: **no** (34 / 900; still by elimination); T10 improved: **no** (worse: wrong on every item, the route now
places A and C confidently in each other's address); nothing else moves outside the seed std: **no**. Rejection rule (R56): "if the
in-world rows fall by more than one seed std, the route is rejected": seen, new ids, role cues and depths 1-3 do not fall; the
new-formulation row with seen transitions, which is made of in-world sentences, falls from 0.864 ± 0.069 to 0.656 (three seed stds;
0.831 -> 0.664 on the claim set). **Read as written, the route is REJECTED.** What it shows: an order-relative address fixes the cleft
(T7) and, with the R46 decoder, every UNSEEN-transition formulation (0.998-0.999), but the fronted 'from' of T10 (the giver before the
verb in second position) is then encoded as the recipient on every item: the order route learns 'the second P before the verb is C'
from the training templates, which T10 contradicts.
