# VSS against pretrained encoders: instructions for the experimental session

_AXIOLOGIC RESEARCH · INSTRUCTIONS FOR THE EXPERIMENTAL SESSION_

A controlled comparison of the VSS reader with BERT on frame reading: accuracy, composition, data efficiency, abstention, size and cost

**Document:** instructions for a separate experiment in its own repository, intended as a separate paper. It fixes the question, the models, the data, the splits, the measures and the criteria before any run; the session executes and reports, and does not change the question.

**Version:** 1.0, 24 October 2026. **Author of the instructions:** the reviewing session, for the experimental session on the DGX. **Main repository (read-only for this work):** ml-circuits-research/vsa-semantic-space at tag paper-2026-10 plus commit e531283.

**Rules carried over:** criteria before runs; one change per run; every number labelled BUILT / MEASURED / PLANNED and pointing to a log whose name carries the commit; thresholds never set on reported rows; a criterion judged on the mean over seeds with the per-seed list printed; the evaluation checklist printed at the end of every script; everything on disk in English; no duration estimates anywhere.

## 1. The question

The VSS paper compares the reader with a small transformer trained from scratch on the same input (RESULTS 7b and 7b-bis), with a bag-of-words classifier and with an associative memory. It does not compare with a pretrained encoder, and that is the first question a reviewer will ask: how does a reader of 1.6 MB with no pretraining stand against BERT, which has read billions of words, on the same task of reading a sentence into a frame of roles? The experiment answers it under controlled conditions, on the same data and the same splits as the paper, and reports what is better, what is worse and by how much, on each axis the paper measures.

The hypotheses, written here so that the result cannot be shaped after the fact. A pretrained BERT is expected to be better on WikiSQL and on new words of real English, because it knows synonyms and the language; the question there is by how much, and at what cost. On the constructed worlds, whose words are invented, pretraining should give nothing, and the comparison isolates the architecture. On composition to a depth not seen in training, the literature on BERT's systematic generalisation (McCoy, Pavlick and Linzen, 2019; Lake and Baroni, 2018) suggests that a fine-tuned encoder reproduces the deepest configuration it has seen rather than composing a new one, as the from-scratch transformer did in 7b; whether a pretrained one does the same is the open question of this experiment. On abstention, a softmax threshold is expected to be poorly calibrated compared with the reader's margins; this is measured, not assumed.

The experiment does not set out to show that VSS beats BERT. Its criterion of success is that the comparison is complete, controlled and reproducible, and that every axis has a number with a confidence interval. A result where BERT wins on most axes is a publishable result if it is exact about where and at what cost; a result where VSS wins on composition and abstention at a thousandth of the size is a different paper. Both are written up.

## 2. Repository and setup

A new repository, vss-vs-bert, with the layout below. The main repository is a dependency pinned at commit e531283 (git submodule or a vendored copy under third_party/, with the commit recorded); nothing in it is modified, and the reader's code is imported, not copied. The rules of the main repository (AGENTS.md, EVALUATION_PRINCIPLES.md rules 1 to 30) apply verbatim and are copied into the new repository's AGENTS.md with a line saying so.

> vss-vs-bert/
>
> AGENTS.md the rules, copied; this document's version
>
> INSTRUCTIONS_2026-10-24.md this document as markdown, verbatim
>
> STATUS.md claims index: one row per axis and model
>
> RESULTS.md numbered sections, one per run
>
> JOURNAL.md one entry per judgment point
>
> third_party/vsa-semantic-space main repository at e531283 (submodule)
>
> data/ splits.json per world and size; worlds; WikiSQL
>
> models/ bert_pretrained.py, bert_scratch.py, heads.py
>
> vss/ wrappers running the reader on the same splits
>
> eval/ metrics.py, checklist.py (rule 3), compare.py
>
> run.sh commit-then-run, as the main repository's
>
> logs/ \<date\>\_\<time\>\_\<script\>\_\<commit\>.txt, memory curve
>
> paper/ the write-up, started after judgment point 1

Environment: the DGX, Python 3.10 or later, torch, the transformers library for the pretrained models (version pinned in requirements.txt), numpy; the main repository's requirements for the reader. Every model is run from a fixed seed list (0 to 4); the data generator seeds are those of the main repository so that the sentences are the same sentences.

## 3. The task: reading a sentence into a frame

The task is the paper's: given a sentence, produce the frame, that is, for each role of the world's legend, the filler or "not given", plus the event (predicate) and, where the world has them, the reporting levels and the negation. Every model is scored by the same function on the same frames, with the main repository's measures: per-role accuracy, FULL frame (every role exact), the three occupancy cases, and calibration (confident-wrong rate and coverage at a threshold set on calibration rows). The output format is the frame as a Python dictionary that the main repository's scorer accepts; models that produce spans or tags are mapped to frames by a fixed, documented rule, and the mapping is tested on the gold data (gold tags must map back to gold frames at 1.000 before any run).

BERT needs a head. Two heads are used, and both are reported. The token-tagging head labels every token with a role or O (BIO over roles), with the event as a sentence-level classification from the \[CLS\] vector and the reporting depth as a second sentence-level classification; the frame is the set of tagged spans. The span head, for WikiSQL, predicts the selected column and the aggregator as classifications over the table's columns and the aggregator set, exactly as the paper's transformer control does, so that the WikiSQL rows are comparable with RESULTS 8c and 8d. A frame that needs a filler the tagger did not tag is "not given" for that role, which is how occupancy is scored for BERT.

## 4. Models

**Table 1. Models compared**

| **Model**                                | **What it is**                                                                                                                                           | **Why it is in the comparison**                                                                                                            |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **VSS reader (reference)**               | The reader of the main repository, reference configuration (path, chainL, bigram; D = 1024), as RESULTS 10b-quinquies and 8c; thresholds as in the paper | the subject of the comparison; run again here on the same splits so that every number in this repository comes from this repository's logs |
| **BERT-base, pretrained, fine-tuned**    | bert-base-uncased (110 M parameters, about 440 MB), fine-tuned end to end with the heads of Section 3                                                    | the standard pretrained encoder; the question of pretraining                                                                               |
| **Small pretrained encoder, fine-tuned** | DistilBERT-base-uncased (66 M) or BERT-small (29 M), one of the two, chosen before the runs and recorded                                                 | a pretrained model closer in budget to what a product could ship on a CPU                                                                  |
| **Same architecture from scratch**       | The small encoder's architecture with random initialisation, trained on the task data only, same heads                                                   | isolates what pretraining brings from what the architecture brings; on the constructed worlds this is the only fair neural comparison      |
| **7b / 7b-bis transformer**              | The paper's control, reproduced from the main repository's scripts, not re-implemented                                                                   | the bridge to the paper's tables; its numbers here must reproduce RESULTS 7b-bis within seed variance before anything else is trusted      |

Tokenisation. The constructed worlds use invented words; BERT's WordPiece splits them into pieces, which is the real condition of any pretrained model on an unknown vocabulary and is kept as is, with the number of pieces per word reported. A second condition adds the world's words to the tokenizer as whole tokens with randomly initialised embeddings; both conditions are run and reported, because the difference is itself a finding about what pretraining can and cannot carry. On WikiSQL the standard tokenizer is used.

Tuning. Each neural model is tuned once per world on a grid written here and not extended: learning rate in {2e-5, 3e-5, 5e-5} for pretrained models and {1e-4, 3e-4, 1e-3} from scratch; epochs {3, 5, 10} for pretrained and {10, 20, 40} from scratch; batch 32; weight decay 0.01; linear warm-up of 10%. Selection on the training-distribution validation split of seed 0 only (rule 25), by FULL frame; the selected configuration runs on five seeds. The reader is not tuned: it takes the paper's configuration.

## 5. Data and splits

Three data sources, all from the main repository, with the same generator seeds. The basic world (v7 nested role world, 30k sentences, reporting to depth 2 in training) for new identities and unseen depth, as RESULTS 7b. The reference world (v16, with the predicate role, occupancy marker and the relation-marker gates given, as RESULTS 10b-quinquies) for full-frame reading at the unseen depth, passives and the fronted forms. WikiSQL (dev and test as the paper uses them, questions read without the table except for the column candidates) at 1k, 5k and 56k training questions, as RESULTS 8c and 8d. The negation and operator worlds are a second round, after the first judgment point, on the same pattern.

**Table 2. Splits (identical for every model; written to data/splits.json before training)**

| **Split**                 | **Content**                                                                                                       | **Paper reference**       |
|---------------------------|-------------------------------------------------------------------------------------------------------------------|---------------------------|
| **seen**                  | training formulations, held-out sentences                                                                         | RESULTS 7b, 10b-quinquies |
| **new identities**        | persons and objects never seen in training, carrying their category                                               | 7b, 7b-bis                |
| **unseen depth**          | depth 3 when training goes to depth 2 (rule 29: an unseen depth whose hops are seen)                              | 7b, 10b-quinquies         |
| **held-out formulations** | template signatures never seen (reference world)                                                                  | 11-septies                |
| **WikiSQL 1k / 5k / 56k** | the paper's training sizes; dev for selection and calibration (even rows) and report (odd rows); test once at 56k | 8c, 8d, 11-octies         |
| **calibration rows**      | training-distribution sentences held out for thresholds; never reported                                           | rule 25                   |

## 6. Axes, measures and criteria

Every axis is reported for every model with the mean over five seeds and the per-seed list; differences between models carry a 95% interval from the seeds. The criteria below are written before the first run; they do not define success of VSS, they define what each outcome will be said to mean.

**Table 3. Axes and the written interpretation**

| **Axis**                                     | **Measure**                                                                                                                         | **Interpretation written in advance**                                                                                                                                                            |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Seen formulations**                        | FULL frame                                                                                                                          | all models expected near 1.0; a model below 0.95 is under-trained and its other rows are not interpreted                                                                                         |
| **New identities (constructed)**             | role accuracy on new fillers                                                                                                        | VSS 1.000 in the paper; BERT pretrained vs from scratch: if pretrained \> 0.9 the pieces carry the role, if \< 0.5 the vocabulary gap is structural; reported as is in between                   |
| **Unseen depth**                             | FULL frame and mean role accuracy at depth 3                                                                                        | the main composition axis; a pretrained BERT \> 0.9 would contradict the paper's claim that same-input neural models do not compose; \< 0.5 confirms it; in between, reported with both readings |
| **WikiSQL selected column at 1k / 5k / 56k** | accuracy on dev-report and on test at 56k                                                                                           | BERT pretrained expected above VSS (0.816) at 56k; the data-efficiency question is the 1k and 5k rows: the size at which each model reaches 0.75 is reported                                     |
| **WikiSQL aggregator**                       | accuracy                                                                                                                            | a lexical decision; the paper's transformer was better (0.89 vs 0.80); BERT expected higher still                                                                                                |
| **Abstention**                               | confident-wrong and coverage at a threshold set on calibration rows; the accuracy-versus-coverage curve; expected calibration error | VSS margins vs BERT softmax (and softmax with temperature scaling fitted on calibration rows); the model with the lower confident-wrong at equal coverage is said to abstain better, per axis    |
| **Occupancy (not given)**                    | the three occupancy cases                                                                                                           | a tagger says not given by not tagging; its rate of false fillers on absent roles is the comparable number                                                                                       |
| **Size and cost**                            | parameters, bytes on disk, CPU milliseconds per sentence (median, p95) on a stated machine, energy if measurable                    | reported; the ratio is part of every claim                                                                                                                                                       |
| **Determinism**                              | frames identical across two runs of the same trained model on the same input                                                        | VSS by construction; BERT in eval mode expected identical, verified                                                                                                                              |

Two further analyses are required in the write-up. First, the error analysis of the paper (RESULTS 11-octies: confident errors classified as the reader's, as gold errors, or as ambiguous questions) is repeated for BERT on the same 50 items, so that the two models' confident errors are compared by kind. Second, on the unseen-depth axis, the per-role table: which roles each model places at chance, and whether BERT, like the 7b transformer, reproduces the deepest configuration it has seen.

## 7. Order of runs

One change per run; each run has its exit check; the session stops at the judgment points and pushes.

**Table 4. Runs**

| **Run**              | **Content**                                                                                                                  | **Exit check**                                                              |
|----------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **0**                | Repository, submodule at e531283, splits written, scorer and the tag-to-frame mapping tested on gold                         | gold tags map to gold frames at 1.000; splits.json committed                |
| **1**                | Reproduce 7b-bis and the reader on the basic world from the main repository's scripts, 5 seeds                               | within seed variance of RESULTS 7b-bis and 7b; otherwise stop and report    |
| **2**                | BERT-base pretrained on the basic world, both tokenizer conditions; tuning on seed 0, then 5 seeds                           | Table 3 rows for seen, new identities, unseen depth                         |
| **3**                | Small pretrained encoder and the same architecture from scratch, basic world                                                 | same rows; the pretraining difference isolated                              |
| **Judgment point 1** | STATUS, RESULTS 1 to 3, JOURNAL; push                                                                                        | the reviewing session decides whether the WikiSQL round proceeds as written |
| **4**                | Reference world: reader and the three encoders, unseen depth full frame, passives, fronted forms                             | Table 3 rows                                                                |
| **5**                | WikiSQL 1k / 5k / 56k, all models, dev-report; test once at 56k                                                              | the data-efficiency table; the 0.75 size per model                          |
| **6**                | Abstention: thresholds on calibration rows for every model; curves; temperature scaling for BERT; the 50-item error analysis | the abstention rows and the error table                                     |
| **7**                | Size and cost on the stated CPU machine; determinism check                                                                   | the cost table                                                              |
| **Judgment point 2** | STATUS, RESULTS, JOURNAL; push; paper outline                                                                                | second round (operators, negation worlds) decided                           |

## 8. Reporting

RESULTS.md follows the main repository's form: one numbered section per run, the exact command, the commit, the log, the table with the mean and the per-seed list, the calibration numbers of rule 18 on every table, the label, and a short reading of what the numbers say, with the alternative reading when the result is in the in-between band of Table 3. STATUS.md has one row per axis and model with its label and log. Every figure is produced by eval/compare.py from the logs, never by hand. The write-up in paper/ starts after judgment point 1 with the question, the models and the protocol, and receives numbers only from RESULTS.md.

What the paper will and will not claim is decided by the numbers, but the form of the claims is fixed now. A claim about VSS is always of the form "at X MB and without pretraining, the reader reaches Y of BERT's accuracy on axis Z, with a confident-wrong rate of W against V", with the interval. A claim about composition names the regime exactly (rule 24: seen at depth 2, tested at depth 3). A claim about abstention names the threshold rule and the calibration rows. No claim uses the word "understands".

## 9. Known risks and what is done about them

Fine-tuning instability on small data: pretrained encoders at 1k examples vary strongly across seeds; five seeds and the per-seed list are the answer, and the variance itself is reported as a finding on data efficiency. WordPiece on invented words: both tokenizer conditions are run, so neither can be blamed afterwards. Under-training of the from-scratch encoder: the grid is fixed, and the paper's rule applies, a from-scratch model that reaches 0.75 on WikiSQL with ten times the budget would mean under-training; this is checked once at 56k with the largest epoch count. Leakage between calibration and report rows on WikiSQL: the even/odd split of the paper is reused unchanged. Compute: BERT-base at 56k for five seeds and a grid is the largest cost; the grid is run on seed 0 only, and the memory curve is logged for every run as the rules require.

## 10. Deliverables of the session

- The repository vss-vs-bert with the layout of Section 2, every run logged, STATUS and RESULTS complete to judgment point 2.

- The comparison tables and figures from eval/compare.py: accuracy per axis and model, data-efficiency curves on WikiSQL, accuracy-versus-coverage curves, the cost table.

- The 50-item confident-error analysis for the reader and for BERT, side by side.

- A paper outline in paper/ with the protocol section written and the result sections pointing to RESULTS.

- current_status.md at the root at every judgment point, as in the main repository.
