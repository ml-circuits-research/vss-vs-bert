# Responses from the author (written by the reviewing session; the experimental session reads, never edits)

Each answer refers to a question of `questions.md` by its number. When an answer is accepted, remove the question from questions.md and
cite the answer (R-number and date) in RESULTS.md or JOURNAL.md where it changes something. Questions about anything below go back into
questions.md.

## R1 (Q1, 2026-10-25): the small encoder is `distilbert-base-uncased`

Accepted as proposed. Record it in RESULTS.md before run 3 with the two reasons you gave (the same WordPiece vocabulary and tokenizer as
BERT-base, so both tokenizer conditions stay comparable; the standard small checkpoint a reviewer expects) and the measured fact that
`prajjwal1/bert-small` does not load with the pinned transformers 5.20.2 (no fast tokenizer, conversion fails). Do NOT add a tokenizer
workaround for bert-small and do not change the transformers pin: the choice is made, and bert-small is not run at all. The from-scratch
twin of run 3b is DistilBERT's architecture (6 layers, same width), randomly initialised, the same tokenizer.

## R2 (Q2, 2026-10-25): commit after every step, push after every run

Accepted: commit after every step, push after every run, and name JP 1 and JP 2 as such in the commit message. One addition: push also
when a run FAILS or stops on its exit check (the failure, its log and its cause in RESULTS.md), and then wait for a response before the
next run. A failed exit check is not repaired by a rerun with other settings (AGENTS rule 4).

## R3 (run 0, 2026-10-25): key order and string hashing, keep it, and make every later run independent of it

Keeping the committed split files after checking that the parsed content is equal is right. But the cause matters for the runs to come:
a mapping built from a Python set whose order depends on the per-process string hash means that any code that builds an index (word ->
row, category -> id) by iterating a set at run time can give different ids in two processes, which changes random draws tied to those ids
and breaks the determinism run (run 7) and the reproduction of the main repository's numbers (run 1). Do this, before run 1:

1. Export `PYTHONHASHSEED=0` for every run. `run.sh` is code: change it in its own commit (one line, `export PYTHONHASHSEED=0` next to
   the other exports) with the reason in the message, then re-run nothing; run 0 stays as recorded (its content was checked equal).
2. Write in RESULTS 0 one sentence: from commit <hash> on, every run has PYTHONHASHSEED=0; run 0 ran without it and its content was
   verified equal to the committed files.
3. Check, once, by reading the code (not by running), whether the reader or the transformer of the main repository builds any index
   from a set at run time; list what you find in JOURNAL.md. Do not change the submodule (it is read-only); if something there depends on
   the hash order, the fixed PYTHONHASHSEED is the whole remedy, and it is stated.

## R4 (standing instructions for runs 1 to 7)

The goal of this experiment is a controlled comparison whose every axis has a number with its interval; it is NOT to make VSS win.
Keep to INSTRUCTIONS_2026-10-24.md and RUNBOOK.md. In particular:

- **Run 1 is the gate.** If the reader or the 7b / 7b-bis transformer does not reproduce the main repository's numbers within seed
  variance, stop, push, and write the difference in questions.md. Do not continue to BERT on a base that does not reproduce.
- **The seen row is the other gate.** A model whose seen-formulation FULL is below 0.95 is under-trained: its other rows are reported
  but not interpreted (INSTRUCTIONS Table 3). Do not extend the tuning grid to rescue it; write it in questions.md.
- **Thresholds and choices only on calibration rows** (rule 25): tau, temperature scaling, the grid selection (seed 0, validation of the
  training distribution). Never look at report rows or the unseen-depth axis to choose anything.
- **Every table**: mean ± std over the five seeds AND the per-seed list; the calibration block of rule 18; the label.
- **Wall clock**: run.sh caps a run (VSA_MAX_HOURS). If a run hits the cap, do not raise it silently: record it in RESULTS.md, and split
  the run by seed (one run per seed, each logged) in a separate, stated commit.
- **Memory**: the GB10 shares memory between CPU and GPU; keep run.sh's guard as it is, and if a BERT run is stopped by the guard, report
  it before changing the batch size (a change of batch size is a deviation, written before the run that uses it).
- **The real-data round** (INSTRUCTIONS_REAL, runs 8 to 12) starts only after JP 2 and after a response here; do not download MASSIVE or
  TOPv2 before that.
- **Interpretation is written after the numbers, in the bands of Table 3**, with both readings when a number is in the in-between band.
  No sentence that says or implies that VSS "understands".

Next: R3 (PYTHONHASHSEED), then run 1a, 1b, 1c, each pushed. I read every push.

## R5 (Q3, 2026-10-25): run 1a's reference is RESULTS 7c "bundle, learned, start closed"; the gate is MET

Accepted as you propose. The RUNBOOK's pointer to 6c was an error of the instructions: 6c is the hand-gate row, and the script runs the
learned gates from the closed start, which is 7c learned-closed. Learned gates are also the right reader for this comparison, because
BERT learns everything from the same data; a hand-gate reader would give VSS knowledge BERT does not get. So:
- RESULTS 1a: the exit check is against 7c learned-closed, MET on every column (keep the 6c comparison sentence as it is, it is honest).
- Correct the RUNBOOK row 1a in the same commit that removes Q3 ("within seed variance of RESULTS 7c, bundle, learned, start closed"),
  citing R5. This is a document change, not a code change.
- Do not run a hand-gate variant: it is not in the instructions and adds nothing to the BERT comparison.
- The PYTHONHASHSEED reading in JOURNAL is accepted: run 1a stands without repetition.

Proceed with 1b and 1c (each pushed). Run 2 may start after 1b and 1c meet their exit checks; if either misses, stop and write it here
as a question (R4).

## R6 (run 1 gate passed; before run 2, 2026-10-25): one metric to add BEFORE the BERT runs, so that depth 3 is not lost by construction

Runs 1a-1c are accepted (1c seed 4 at seen 0.86 is recorded honestly; the model mean is 0.972, the gate is on the mean, keep it so).
Proceed to run 2, after one change written before any BERT number exists:

**The problem.** The token-tagging head of models/bert_basic.py has the labels O, A, B, C, X1, X2, X3, and training covers depths 0-2
only, so the label X3 never occurs in training. At depth 3 the tagger must output X3 for the third speaker; its weights for that label
are untrained, so X3 is at chance BY CONSTRUCTION and the six-role FULL frame at depth 3 is ~0 whatever the encoder can compose. That
would make the composition axis say something about the output space, not about BERT. The reader and the 7b transformer write a state
that the world's addresses decode, so they are not limited in the same way (7b reached X3 0.06, which is its own failure).

**What to do (a deviation of the reporting, written now, before run 2; no change to any model):**
1. In eval/ (report.py or a new function used by every model's report, the reader's and the transformer's included), add for the depth-3
   axis two more numbers beside the existing ones: FULL over the roles whose labels occur in training (A, B, C, X1, X2), and FULL of the
   INNER frame (A, B, C). Keep the six-role FULL and the per-role table exactly as they are.
2. Recompute these two numbers for runs 1a, 1b, 1c from their stored per-item predictions (logs/results_*.json) if the predictions are
   stored there; if they are not, say so in RESULTS and compute them from the saved reader model (1a has --save_model) and leave 1b / 1c
   with the per-role table only. Do not re-run 1a-1c for this.
3. Write in RESULTS (a short section "Depth-3 metrics added before run 2 (R6)") the reason above, and in INSTRUCTIONS Table 3's reading
   for the unseen-depth row that the comparison with the tagger is made on the inner frame and on the trained labels; the six-role FULL
   is reported for every model with the note that a tagger cannot output an untrained label.
4. Commit this as its own commit before run 2.

A head that CAN express an unseen depth (the tagger applied recursively: outer speaker plus the embedded clause, as bert_real.py does for
TOP) is a different model; it is NOT added now. Note it in questions.md as a proposal for after JP 1, with your view of its cost.

Everything else of R4 stands: grid fixed, seed 0 for selection on training-distribution validation, seen row >= 0.95 as the gate,
both tokenizer conditions, 5 seeds, push after every run.

## R7 (Q4, Q5; 2026-10-25)

- Q4: leave INSTRUCTIONS_2026-10-24.md verbatim; the note in RESULTS ("Depth-3 metrics added before run 2 (R6)") is the right place
  (AGENTS: deviations are recorded in RESULTS). Accepted as written, including the bounds for 1b / 1c labelled as bounds.
- Q5: accepted as a proposal, decided at JP 1, with your caveat kept: the recursion hands the tagger the clause structure as an
  architectural prior, so if it is run it is reported beside the flat tagger, never instead of it. Not before JP 1.
- Stopping 2a before any number to put R6 in place was correct. Start run 2a again now, then 2b, as written.

## R8 (2026-10-25): a sanity check of OUR BERT fine-tuning against published numbers, for the real-data round (not now)

We do not pretrain BERT, but the fine-tuning (head, grid, epochs, tokenization, tag-to-frame mapping) is ours, and "the baseline was
under-tuned" is the first objection to any comparison with BERT. When the real-data round starts (after JP 2, INSTRUCTIONS_REAL), its
FIRST run is a reproduction check, before any VSS-vs-BERT number on real data is interpreted:
- models/bert_real.py, bert-base-uncased, on SNIPS and on MASSIVE en-US with their STANDARD splits and the STANDARD metrics of their
  papers (intent accuracy, slot span F1 with the conlleval convention, exact match), next to the published BERT numbers, cited with their
  source in RESULTS (find them in the papers; write the source and the table; do not take numbers from memory).
- Exit check: within the reported variance of the published numbers, or a stated, located reason. If our tagger is clearly below, stop
  and write it in questions.md; nothing on real data is interpreted until it reproduces.
- Write the check into INSTRUCTIONS_REAL as run 8.0 in its own commit, citing R8, when the round starts; nothing to do now.
For the constructed worlds the guard stays as it is (the seen row >= 0.95 gate, the fixed grid, both tokenizer conditions).

## R9 (2026-10-25): rules against bias, in force from now on

The comparison must be one that a reader who favours BERT would accept. In addition to R4 and R6:

1. **Pre-registration.** Before the first BERT number is written, tag the current commit `prereg-2026-10-25` (git tag, pushed) and
   write its hash in RESULTS: INSTRUCTIONS, the Table 3 interpretations, the grid, the R6 metrics and R8 are fixed at that tag. Any later
   change of a criterion, metric or grid is a deviation listed in RESULTS with the date, and the tagged version is reported beside it.
2. **Everything is reported.** Every run, every seed, every failed or stopped run stays in RESULTS with its log; no seed is dropped, no
   run is repeated to obtain a better number. The paper's tables are generated by eval/compare.py from the logs only.
3. **Blind error analysis.** For the 50-item confident-error analysis (run 6), the TSV given to the person who fills the 'kind' column
   has the model names removed and the rows of all models shuffled together (a fixed seed; the key kept in a separate file committed
   after the classification). Write this into eval/error_analysis.py as its own commit before run 6.
4. **Where BERT is expected to win, it is shown first.** The write-up states plainly the axes where BERT is better (expected: new
   formulations, WikiSQL, real data), with the same prominence as the axes where the reader is better. No claim of the form "VSS is
   better than BERT" without the axis, the size and the cost in the same sentence (INSTRUCTIONS section 8).
5. **Robustness of the BERT baseline, as a stated additional analysis after JP 1** (never replacing the pre-registered rows): BERT-base
   with a wider grid on seed 0 (learning rate {1e-5, 2e-5, 3e-5, 5e-5, 1e-4} x epochs {3, 5, 10, 20}, warm-up 10%) on the unseen-depth and
   new-identity axes, to show that the conclusion does not depend on the grid. If the wider grid changes a conclusion, that is written.
6. **An independent review of the BERT code** is being made from the reviewing side (models/bert_*.py, common/, eval/); its findings
   arrive here as a response. Until then, run as planned; a finding that changes a number will be handled as a stated deviation.

## R10 (R9.6, 2026-10-25): findings of the independent review of the BERT baselines, and what to do

An independent reviewer, briefed only to be the advocate of the BERT baseline, read models/bert_*.py, common/, eval/ and ran small
checks on CPU with tests/tiny-bert (mechanisms, not levels). Tokenization and label alignment were checked correct (all 76,940 words of
seed 0, both conditions). The findings below can lower BERT's numbers. The pre-registered rows (tag prereg-2026-10-25) stay exactly as
they are and are reported; every fix below is a DEVIATION declared now, before any BERT number, implemented behind flags whose defaults
reproduce the tagged code, and run as additional rows labelled "reviewed (R10)" beside the pre-registered ones. Let run 2a (pre-registered)
finish; then implement R10 in its own commit(s); then continue with 2b and the R10 rows.

### Must do (changes numbers materially)
1. **One-to-one decoding** (bert_basic.py:82-86, bert_reference.py:62-70, common/decode.py:262-269). Each role takes its own argmax token,
   so one token can fill two roles; in the data the fillers are always distinct (0 repeats on all seeds) and every depth-3 item has
   exactly 5 person tokens for its 5 person roles. Under a one-to-one assignment (maximise the summed log-probability, or the summed
   score for the reader, over roles x candidate tokens; Hungarian or brute force) X3 is the token left once the trained roles are placed,
   so the "X3 by construction" loss of R6 largely disappears. Reviewer's tiny-model check: depth-3 FULL 0.000 -> 0.353. Add it as a
   SECOND decoding rule for EVERY model (reader, 7b, taggers) through the common decoder, reported beside the current rule. It uses a
   property of the world (fillers are distinct) that is equally available to all models; state it. For 1a, recompute it from the saved
   route book (seed 0) as in R6; for 2a and later, from the saved models (--save_model), with no retraining.
2. **Positions never seen in training** (scratch encoder; bert_basic.py:28). The scratch twin learns absolute position embeddings; the
   7b control uses sinusoidal positions precisely so that longer depth-3 sentences are representable (v9_transformer.py:10, 26-27). On
   seed 0, 75% (whole words) / 100% (WordPiece) of depth-3 sentences are longer than any training sentence; 37-50% of A/B/C fillers sit
   at positions never reached. Do: (a) for EVERY model, report the depth-3 accuracy split by "filler position seen in training / not"
   (this separates composition failures from position failures); (b) for the scratch rows (3b), an R10 variant with random position-id
   offsets in training (shift by k in 0..10 per sentence). Optionally the same offsets for BERT-base and DistilBERT as an R10 variant.
3. **Model selection ties go to the least-trained setting** (strict `>` at bert_basic.py:129, bert_reference.py:101, bert_wikisql.py:94,
   bert_real.py:177; the grid starts at lr 2e-5, 3 epochs). On a training-distribution validation every setting is likely 1.000, so the
   "selected" model is simply the first grid point. Do: print the validation token cross-entropy of every grid point; R10 tie-break =
   lowest validation cross-entropy among the points with the best validation FULL. The pre-registered selection stays as reported; if
   R10 selects a different configuration, its 5 seeds are run as the R10 row.

### Should do (may change numbers)
4. **'whole' condition initialisation** (bert_basic.py:29): transformers 5.20.2 resize_token_embeddings uses mean_resizing=True, so new
   rows are not "randomly initialised" as INSTRUCTIONS says, and the held-out identities get no gradient and lose the p/o category that
   WordPiece keeps. State the actual init in RESULTS (it is the pre-registered row); R10 variant: initialise each new token as the mean
   of its WordPiece pieces' embeddings.
5. **Weight decay on biases and LayerNorm** (all bert_*.py AdamW lines): R10 = two parameter groups, no decay for bias and LayerNorm
   (the standard BERT recipe). Apply in every R10 row.
6. **Temperature scaling** exists only in bert_real.py; implement it for the basic, reference and WikiSQL scripts as INSTRUCTIONS asks,
   and state for every model that in-distribution calibration rows (near 100% correct) cannot calibrate out-of-distribution axes. Add the
   depth-3 calibration block over the trained roles beside the six-role one.
7. **Reference world predicate from [CLS]** (bert_reference.py:25-26, 71) while the reader reads it from the verb token; FULL includes
   it. R10: tag the verb token with a 'P' label decoded over the verb candidates; and report FULL with and without the predicate for all.
8. **WikiSQL**: aggregator predicted from the question alone (bert_wikisql.py:33-38) while standard BERT systems condition it on the
   column; max_len 64 truncates some question+header pairs. R10: aggregator from the (question, column) pair; max_len 128. State in RESULTS
   that BERT encodes the header text jointly with the question (the standard cross-encoder) and that this is the same regime as the reader's
   header-bundle candidates.
9. **Few-step training at 1k / 5k** (bert_wikisql.py:44, bert_real.py:57: 96-320 optimizer steps at 1k): the data-efficiency claim rests
   on these rows. R10: at least 1,000 optimizer steps at the small sizes (epochs raised accordingly), as a robustness row.
10. **Cased multilingual models get lower-cased text** (common/real_data.py:11, 56, 144) for run 12, and span F1 is computed on
   normalised tokens: give cased encoders the original-case tokens (alignment unchanged), and compute the R8 check on the dataset's own
   tokens with conlleval. Needed only for the real-data round.

### Presentation (no number changes; write in RESULTS / the cost table)
11. Depth given at decoding to every model and no event / depth head on the basic world: equal for all, but a deviation from INSTRUCTIONS
    section 3; write it. No CRF: standard, state it.
12. Occupancy at depth 3 counts X3 as missed by construction: add the occupancy over the trained labels.
13. Cost: report BERT also at fp16 and int8 dynamic quantization, and the CPU thread count for ms/sentence; note that the scratch twin
    carries a 30,522-row embedding table for a 66-word world.
14. Reproducibility: seed torch per grid configuration (as bert_real.py:173), not once before the grid, so a selected configuration can
    be re-run alone.
15. bert_basic.py:45 fallback `f.append(0)` would label [CLS] for a truncated word; never triggers here; make it None / -100 as in
    bert_real.py.

Order: finish 2a (pre-registered); one commit with items 1-3, 5, 6, 11-15 (common code and the basic world), each behind a flag whose
default is the tagged behaviour, smoke-tested on CPU; the R10 decoding (1) and the position split (2a) applied to 1a and 2a from their
saved models; then 2b, then the R10 rows of run 2 (BERT-base, both tokenizer conditions), then run 3 with both its pre-registered and
R10 rows. Items 4, 7, 8, 9, 10 go in when their runs come (3 / 4 / 5 / real round). List every R10 item in RESULTS under one heading
"Deviations after the independent review (R10)", with this response cited. If any item looks wrong to you, say so in questions.md
before implementing it.

## R11 (Q6, Q7; 2026-10-25)

- **Q6**: accepted as you propose: item 4 applies to the R10 'whole' row of run 2, behind `--whole_init pieces` (default `resize`, the
  tagged behaviour, stated in RESULTS for 2b).
- **Q7 (a)**: do NOT leave 1b / 1c without the one-to-one rule: the transformers may gain from it as much as BERT, and the comparison
  must give every model the same decoder. They are small and cheap to train. Add `--save_model` to models/transformer_7b.py (default off,
  so the tagged behaviour is unchanged), and re-run 1b and 1c with the SAME selected configuration (no grid: the configuration chosen in
  the pre-registered run), 5 seeds, saved. Report these as "1b / 1c re-run for R10 (same configuration, models saved)": first the
  per-role rule numbers, which must reproduce the pre-registered 1b / 1c within seed variance (that is their exit check; if they do
  not, stop and say so), then the one-to-one rule on the same models. The pre-registered 1b / 1c rows stay as they are.
- **Q7 (b)**: accepted: the margin of a role under the one-to-one rule is the assigned candidate's score minus the best score of any
  other candidate for that role (negative = unsure). Report also the assignment margin of the sentence (best total score minus the
  second-best assignment's total), and calibration blocks for both rules, as you propose.

## R12 (Q8, Q9, Q10; 2026-10-25)

- **Q8**: accepted: (a)-(c) from the logs for every row, (d) GPU energy from 2b on, all labelled "this machine (GB10), fp32". Add the
  CPU thread count and the batch size to every cost row.
- **Q9**: accepted: fp32 for every training row (pre-registered, R10, R9.5); fp16 / int8 are inference-cost rows only.
- **Q10**: accepted, with one correction to the wording, so that the asymmetry is not overstated in our favour: the reader has no
  selection in THIS repository, but its architecture and its configuration (codes, gates, D, learning rate, epochs) came out of the main
  repository's development rounds on these same constructed worlds, which is a form of tuning, not zero. Write it as: "selection cost
  of the reader in this repository: 0; its configuration was developed in the main repository on the same worlds (cited rounds), so the
  reader's design is not blind to these worlds, while BERT's architecture is". State the same sentence in the threats-to-validity part of
  the write-up. This is the honest version, and it is the one a reviewer would write if we did not.

## R13 (Q11; 2026-10-25)

Accepted: (a) and (b) after JP 1 as a stated additional analysis; the pre-registered 5-seed rows stay as reported and the 10-seed rows are
shown beside them, same selected configuration, no new grid; splits basic_seed5-9.json made and committed before any model sees them.
(c) is right and stays as you wrote it: no seed chosen or dropped, no gate tuning on report rows; a variance-reduction reader is a new
declared row with the matching BERT treatment, and it is not run unless asked. The power sidecar (eval/power_sampler.sh) is accepted as
written: it reads nvidia-smi only and touches no model code.

## R14 (2026-10-25): a reader with the role-cue gates GIVEN, as a new declared row, with the matching BERT treatment

Run 1a shows that the learned gates differ by seed (seeds 0, 3, 4 open {'.', by, that, to}; seed 1 lacks 'by'; seed 2 lacks '.' and
opens 'from'), and that this is the whole source of the spread on the new-formulation axis. The data do not fix the gates uniquely: on
the training sentences several gate sets score alike. The author's position: the words that open a role's address ("to" for the
recipient, "by" for the agent in a passive, "from" for the source, the sentence end) are a GIVEN of the legend, like the categories
(rule 20): the legend already states which roles exist and which question each answers; stating which cue words those questions rest
on is the same kind of knowledge, not tuning on report rows. The main repository already does this for the reference world
(reader.gates_given: to, by, from open, v16; tool/sop/legend.py), and the basic world here does not, which is why the spread appears.

What to do, after the pre-registered run 2 and before JP 1 (a new model row, never a change of the pre-registered reader):
1. **Reader row "gates given"**: v8_gates.V8 with fix_gates applied to the role-cue categories {to, by, from, '.'} OPEN (logit +8,
   gradient masked, as v16's --rel_gates open) and the level openers {says, that} OPEN; every other gate learned from the closed start as
   before. One flag in vss/run_reader_basic.py (default off = the tagged behaviour), 5 seeds, the same frozen splits. The list of given
   cues is written in RESULTS before the run and is the SAME for every seed. Report all axes beside the learned-gate row; the expected
   effect is on the new-formulation axis; if it also changes depth 3 or new ids, say so.
2. **The matching BERT treatment**, so that the comparison stays fair: the same cue words are given to BERT as an extra input feature in
   a new row "cues given" of run 2 (BERT-base, WordPiece): a learned "cue" embedding added to the token embedding of every token whose
   word is in the same set {to, by, from, '.', says, that} (one extra embedding row, trained; everything else as the pre-registered 2a).
   This is the equivalent of telling BERT which words open roles. Same grid, 5 seeds. If the cue feature changes nothing for BERT, that is
   a finding too.
3. The 7b / 7b-bis controls get the same cue feature as a row only if the budget allows after JP 1; say in RESULTS whether it was run.
4. In the write-up these rows are "cues given" models, reported beside the learned ones; the main claims are made on the pre-registered
   rows, and the cues-given rows show how much of the variance on new formulations is this degeneracy. Write the rationale above (what is
   a given, rule 20, the reference world precedent) in the deviations section before running.

## R15 (run 2a; 2026-10-25): accepted as written; what it means and what comes next

The reading of 2a is accepted exactly as you wrote it, including the observation against R6's premise (X3 right on 4 of 5 seeds under
the free argmax: stated as observed, not explained) and the corrected time ratio. The incident report in JOURNAL is accepted; no run is
repeated. The sentence that matters is written and stays in the paper in that form: for a pretrained encoder fine-tuned on the same
data, the unseen depth 3 over the trained labels and over the inner frame is 1.000, which contradicts, for pretrained encoders, the
paper's same-input claim made against the from-scratch transformer. The paper's claim is therefore narrowed to what runs 1b / 1c and 2a
together support: a same-size, same-input model trained from scratch does not compose the unseen depth; a pretrained encoder does. Do
not soften this and do not look for a metric on which it goes away. The axes where the reader stands after 2a are the six-role FULL
(0.982 ± 0.010 against 0.765 ± 0.376: the reader produces the whole frame of the unseen depth and the tagger does not reliably), the
calibration at depth 3 (confident-wrong 0.003 against 0.039, 0.164 on seed 2), and cost (3.3 M against 109.5 M parameters, 13.3 MB /
1.7 MB against 438 MB, 9-12 ms against 31 ms per sentence, 1.8x training time per seed, 8x per row with selection). State them with the
same prominence as BERT's axes, and no more.

Next, in this order, each pushed:
1. eval/r10_saved.py on the saved 2a seed-0 model and on the saved 1a seed-0 route book (one-to-one rule, position split): the first
   measurement that can say something about the X3 observation. Report, do not explain beyond what it shows.
2. Run 2b (whole tokens, pre-registered), then run 3a / 3b (DistilBERT pretrained and from scratch, pre-registered). 3b is the
   from-scratch twin and the direct test of "pretraining is what composes here": if its inner frame at depth 3 is near 1b / 1c (<= 0.1) the
   conclusion is pretraining; if it is also near 1.0 the conclusion is the tagging head over a BERT-shaped encoder, not pretraining. Write
   both outcomes in RESULTS before 3b runs, as the interpretation to apply.
3. The R10 rows of run 2 and run 3 (one-to-one, position split, tie-break by validation cross-entropy, weight-decay groups, whole_init
   pieces for the whole row), then JP 1.

For JP 1 (not before, and only as a question here, not a run): since every model that composes is at 1.000 at depth 3 on the inner frame,
depth 3 no longer separates them. A declared additional analysis for after JP 1 could extend the same frozen worlds to depths 4 and 5
(training still at depths 0-2) for every model, reader included, to find where each one breaks and how. Write your view of its cost and
of any reason it would be unfair to either side in questions.md at JP 1. It is not part of the pre-registered claims.

## R16 (commit 767cf1f; 2026-10-25): the single-candidate margin fix is accepted, with one condition

The fix (a role with one candidate: margin = score minus the model's floor, as the per-role rule's top1 - 0; log 1e-30 for a tagger,
0 for cosine scores) is a correction of a defect in R10 code that never produced a reported number, found before any one-to-one row
was reported: accepted. Conditions: (1) it is listed in the R10 deviations section with the date and this reason, and the first one-to-one
numbers reported (r10_saved on 2a and 1a) say which version of the margin they use; (2) the floor is written in RESULTS per model
(reader: 0; taggers: log 1e-30), because a tagger's margin for a single candidate becomes very large by this rule and must not be read
as a stronger decision than the reader's 'top1 - 0'; where a calibration block compares the two, say so. The R14 code behind default-off
flags is accepted; the R14 rows run after the pre-registered run 3, as R14 says.

## R17 (2026-10-25): measuring q-formalisation for every model, not only accuracy (declared additional rows, after run 3, before JP 1)

The author defines what the reader does as q-formalisation (vss-demo, vision/ADDENDUM_qformalisation_2026-10-25.md): the answer to each
question of the legend or "not given", a margin per decision, and a residual (the share of the sentence the legend does not explain).
The pre-registered rows measure the first part well and the others only partly, for every model: the depth is given at decoding (so
"not given" is barely tested on the basic world), the margins come from in-distribution calibration rows where every model is near 100%
correct (so tau is 0 and coverage 1 for all: the confident-wrong rates at depth 3 are what appears when no model has anything to
abstain from, which is fair but is not "how well BERT can abstain"), and the residual is not measured at all for BERT because a tagger has
none. Three declared additional rows, for EVERY model, written in RESULTS before they run, never replacing a pre-registered row:

1. **Out-of-world rejection** (the demo specification's criterion: >= 0.90 rejected, <= 2% false rejections on in-world sentences). A
   frozen out-of-world set per seed, committed before any model sees it: (a) random token sequences from the world's vocabulary, (b)
   sentences of the reference world (another legend) tokenised in the basic world's vocabulary where possible, (c) in-world sentences with
   one content word replaced by a word of no kind. Reader: rejected = status no-sense / unknown-event, or residual above its calibration
   threshold. BERT, two variants reported side by side: (i) no training: rejected when the maximum candidate probability of any role is
   below a threshold set on the calibration rows at the 2% false-rejection point; (ii) with training: an "out-of-world" class trained on
   NEGATIVE examples drawn from the same three sources (disjoint from the frozen test set), and the reader then receives the same negatives
   as calibration rows for its residual threshold, so neither side has information the other lacks. Report the rejected share and the false
   rejections for each.
2. **A residual for BERT**, defined once and reported beside the reader's: the share of content tokens (not O in the world's kinds) that
   the tagger assigns to no role with probability >= 0.5. State that it is an analogue, not the same quantity.
3. **Depth not given** on the basic world: the reader estimates the depth from the count of level-opener tokens (as the sop tool does by
   default; one line), BERT from a depth head on [CLS] trained with the tagger (INSTRUCTIONS section 3 asked for it); FULL at depth 3 with
   the estimated depth beside the given-depth row. This tests "not given" on the basic world for both.

These rows measure what the paper claims to measure; they are not meant to find an axis the reader wins. If BERT with the out-of-world
class rejects as well as the reader, that is the result. Write the three rows and their criteria in RESULTS as deviation R17, then run them
after run 3 and before JP 1; the frozen out-of-world sets are a separate commit before any model sees them.

## R18 (Q12 to Q15; 2026-10-25)

- **Q12**: accepted: the sop tool's residual, same definition (||S - a rec|| / ||S||, rec = the world's state of the decoded frame at the
  given or estimated depth, a = best real scale), computed in common/decode.py; thresholds as you state, variant (i) at the 2% false-
  rejection point of the in-world calibration rows, variant (ii) the best separating threshold with the negatives, under <= 2% false
  rejection. No status for the reader on this world: say so in RESULTS, so that "rejected" means "residual above threshold" for the
  reader and nothing else.
- **Q13**: accepted. Keep (b) as you define it (category sequence not that of any basic-world form), report it per source and never
  pooled; add one sentence in RESULTS that (b) is the weakest source because a person reads most of those sentences with C "not given";
  do not drop it. The 320 'neg' reference items are enough; say the pool is exhausted.
- **Q14**: both. The default (rejection by construction, "word outside the legend", reported as such and NOT counted as learned) is the
  primary row; the alternative (a placeholder vector of category 'O', one fixed random vector per seed, so the reader must decide from
  the state) is a second row labelled "nonce as placeholder", because it is the only version of (c) on which the reader's learned
  rejection can be compared with BERT's. For BERT, (c) is a real test in both.
- **Q15**: accepted as proposed, with one addition: item 3 (depth not given) also for the 7b-bis control from its saved R11 re-run models
  (depth from a classifier over the state is not available for it; use the same level-opener count as the reader, since the count is a
  property of the token sequence and not of the model), so that the three model families appear in the depth-not-given row. The
  from-scratch encoder and the trained OOW class for the controls stay out unless asked.
Order unchanged: pre-registered 2b, 3a, 3b first; then the R10 rows; then R14 and R17 rows; then JP 1.

## R19 (2026-10-25): small pretrained encoders at the reader's size, a declared row for after JP 1

The reader's argument is only convincing at equal size. Declared now, run after JP 1 (not before), interpretations written before the
run: BERT-tiny (google/bert_uncased_L-2_H-128_A-2, 4.4 M) and BERT-mini (L-4_H-256_A-4, 11 M), pretrained, fine-tuned on the basic
world, 5 seeds, both tokenizer conditions, the one-to-one rule and the position split as every row, the cue feature in a second row
(R14). Grid for these models, fixed now and not extended: lr {5e-5, 1e-4, 3e-4} x epochs {5, 10, 20}, batch 32, warm-up 10%, selection on
seed 0 as always (with the R10 tie-break); the larger learning rates are the published practice for the small BERTs (Turc et al. 2019),
and a reviewer would call the BERT-base grid under-tuning for them. Check first that both load with transformers 5.20.2 (they are the
official Google uploads); if one does not, say so and do not add a workaround.

Interpretations: (a) inner-frame FULL at depth 3 < 0.9 for the small encoders while the reader stays at 0.98: at equal size, the reader
composes the unseen depth and the pretrained encoders do not; (b) >= 0.9: pretraining composes on this world even at 4 M parameters, and
the reader's case rests on the six-role FULL, the calibration, the residual and the cost; (c) in between, both readings. Report also the
cost rows for both (parameters, bytes at fp32 / fp16 / int8, CPU ms per sentence with the thread count), so that the size comparison is
on the same table as the accuracy. These rows do not change the pre-registered claims; they test whether the reader's advantage survives
when the baseline is as small as the reader.

## R20 (Q16; 2026-10-25)

Accepted, with a third variant so that BERT is not left with only heads that cannot or can barely extrapolate: (a) the classifier over 0-3
with class 3 untrained, reported as such; (b) the regression head (one output, squared error, rounded and clipped to 0-3); (c) a COUNTING
head that is the exact analogue of the reader's rule: a token-level binary tagger "this token opens a level" (trained on the gold level
openers, which occur at every depth 0-2), the depth = the number of tokens tagged as openers. (c) is trained on a label that exists at
every depth and can output 3 as naturally as the reader's count does; it is what a careful BERT engineer would build for this decision.
All three on the same 5 seeds, 2a configuration, stated in RESULTS before the rows run. The reader's level-opener count stays as it is
(a rule on the token kinds, not a learned head); say so in the same table.

## R21 (run 2b; 2026-10-25): accepted as written; one addition to its reading, and the order is unchanged

Accepted, including the whole-token initialisation stated as it is (mean resizing, R10 item 4), the energy figure, and the observation
that the one-to-one rule lowers X2 on this row (write it as a finding against the rule, not for it: the rule is not a free improvement for
any model, and the pre-registered per-role numbers stay the main ones). Add one sentence to the reading: in 2b every depth-3 error is on
the OUTER speakers (X2 0.858, X3 0.495) with the inner frame at 1.000 on every seed, which is the same shape of failure the from-scratch
transformer had in 7b (inner lost, outer kept) in mirror image; stated as observed, not explained, with the per-seed list (seed 0 X2 0.30).
Then the frozen out-of-world sets (a separate commit), 3a, 3b, as planned.

## R22 (2026-10-25): cutting the remaining time without cutting rigour

1. **No grid for derived rows.** The R10, R14, R17 and R20 rows of runs 2 and 3 are variants of 2a / 2b / 3a / 3b; INSTRUCTIONS already
   fixes their configuration as "the one selected on seed 0" for that model and tokenizer condition. Run them with that configuration, no
   grid (write in each row which pre-registered selection it inherits). The grid is kept only for new pre-registered models (3a, 3b) and
   for R19's small encoders, which have their own fixed grid. On 2a / 2b every grid point gave validation FULL 1.000, so the grid never
   chose anything but the tie-break; say so in RESULTS once.
2. **Shared-encoder multi-head training for the add-on heads.** The three depth heads (R20), the out-of-world class (R17 ii) and the cue
   feature (R14) are heads or input features over the same encoder. Train them in ONE run per model and seed with the losses summed
   (equal weights, stated), and decode each head as its own row. A stated deviation: the heads share the encoder, which is the standard
   practice; it favours no model. Keep the pre-registered tagger rows exactly as they are (they ran alone).
3. **Three GPU slots.** The GB10 has 128 GB unified memory; BERT-base at batch 32 in fp32 uses under 12 GB. Change run.sh (its own commit,
   reason in the message) to allow up to 3 concurrent runs: one lock per slot (logs/.run.lock.N), the memory guard checked per launch
   (keep 40 GB available AFTER the launch), separate logs as now. Determinism is per process (fixed seeds, fp32), so concurrency changes no
   number; wall-clock and energy per run are then shared and must be marked "concurrent (N runs)" in the cost rows, and the per-seed
   training time used in the cost table must come from a run that was alone (2a, 2b, 3a, 3b, which are). Queue: 3a and 3b first (alone,
   for their cost rows), then the derived rows three at a time.
4. **Nothing else is cut**: 5 seeds per row, fp32, the pre-registered rows unchanged. R9.5 and R13 (seeds 5-9) stay after JP 1 and may
   run overnight. The no-training measurements (R10 saved, R17 i, residual) are minutes.
Write the new expected order and the derived rows' inherited configurations in RUNBOOK (document change) before starting them.

## R23 (Q17; 2026-10-25): parallelise more, within the memory rule

The measurement settles (1): the GPU is launch- and size-bound (one CPU core per run, ~110 ms per step of batch 32, 2-3 TFLOP/s), so
concurrency is the right lever and the only one that changes no number. Go further than three slots, with the memory rule as the
limit: slots = floor(available memory after the current launches / the measured peak per process), capped at 6 and at the number of
derived rows waiting; VSA_MEM_FRACTION per process = 0.9 / slots (never a sum above 0.9 of the unified memory, so the 2026-10-15 hang
condition cannot occur); the 40 GB post-launch guard stays. Measure the per-process peak once on 3a's log (torch.cuda.max_memory_
reserved is printed by the guard) and write it in RUNBOOK with the slot formula. Then the throughput check as you propose, but at the
chosen slot count: samples per second of N concurrent runs against one alone; keep N if the total is at least 1.3x, else halve N and
measure again. Each run keeps one CPU core; with 20 cores that is not the limit. The sidecar's power samples are per machine, so the
energy of concurrent runs is reported for the batch, not per run (R22 item 3). Cost rows per seed still come only from runs that ran
alone. Do this after 3b, as planned; 3a and 3b run alone.

## R24 (2026-10-25): the BERT-MRC baseline, the one a reviewer from question-based extraction will ask for

The tagger of runs 2 and 3 labels tokens with fixed classes (A, B, C, X1, X2, X3). The field has moved past that for exactly our kind of
task: QA-SRL (He et al. 2015) and MRC-style extraction (Li et al. 2020, "A unified MRC framework for NER"; zero-shot slot filling as
question answering) feed the QUESTION as text and return a span of the sentence or NULL. That model has no untrained label (the level-3
speaker is just another question), anchors every answer in the sentence, learns refusal (NULL on [CLS]) in training, and takes new
questions without retraining. It is the closest neural analogue of the legend's questions, so the paper must compare with it, or be told
to. Declared now; a new row family, not a change of the pre-registered tagger rows; run after 3b, in the derived-row queue (R22 / R23),
before JP 1.

Model: bert-base-uncased (and DistilBERT) with a span head (start / end logits over the sentence tokens) and NULL = the [CLS] position,
input "[CLS] question [SEP] sentence [SEP]"; the question text is the legend's own (basic world: write one natural question per role and
per level exactly once, in RESULTS before training, e.g. "who gives?", "what is given?", "to whom?", "who says it?" with the level named
as in the legend: "who says it at level 2?"); one forward pass per (sentence, question); training on the same 30k sentences with every
question of the legend per sentence, NULL for roles absent at that depth (so refusal is trained on in-distribution absences, which is the
standard practice; the R17 out-of-world negatives are NOT added to this row unless a second row says so). Decoding: the span restricted
to the world's candidate tokens as for every model (the argmax start=end token among candidates, or NULL if NULL scores higher); margin =
best candidate score minus the second best; same thresholds rule; five seeds; the fixed pretrained grid of INSTRUCTIONS section 4
(no R10 tie-break change needed if validation separates; if every point is 1.000 again, the R10 tie-break). Depth: given at decoding as
for every model (the questions asked are those of the depth); the depth-not-given variant asks all six questions and takes NULL as "not
given", which is this model's natural form of R17 item 3; report both.

Cost rows, which matter here more than anywhere: inference per sentence with ALL questions of the legend (six passes at depth 3 against the
reader's one state), CPU ms and energy; training time per seed. Interpretations, written now: inner-frame FULL at depth 3 >= 0.9 means the
question form composes as the tagger did (expected); the six-role FULL and the level-3 question are the new information (a question never
asked in training); confident-wrong under NULL training against the reader's residual is the calibration comparison that a reviewer will
trust most; the cost ratio with the legend's size is the argument that survives whatever the accuracy says. Write the row and the
questions in RESULTS as deviation R24 before it runs; the questions are frozen with the pre-registration tag's successor commit.

## R25 (2026-10-25): the queue before JP 1 is cut to what supports a claim; the rest moves after JP 1

How the baseline choice slipped: the instructions started from the paper's existing comparison (a from-scratch transformer with a
decoded state) and took the tagging head as "BERT on this task", the JointBERT form of intent + slot filling, without asking which form
the literature uses for OUR task, which is question-based (QA-SRL, MRC). The review of R10 was briefed on implementation, not on the
choice of baseline. Recorded in JOURNAL as a decision point, with this reason. Consequence for the queue, rule: before JP 1 only rows
that support a claim of the paper or are pre-registered; a row that existed to defend the tagger against an objection MRC does not have is
deferred.

Deferred to after JP 1 (listed in RUNBOOK as such, not dropped): R20's three depth heads for the tagger (MRC asks all questions and takes
NULL as "not given", which is its natural depth-not-given form; the tagger variants are kept for later); R17 (ii) trained out-of-world
class for the tagger (NULL training on MRC is the standard form of refusal; the tagger keeps only R17 (i), no training); R14's cue feature
for the tagger (for MRC the cue words belong in the question text, which is R24 as written); the R10 rows of 2b (2a's R10 rows show the
effect of the rule on a tagger; on 2b the rule did not help); R14.3 for the controls; seeds 5-9 for the controls.

Before JP 1, in this order: 3a, 3b (pre-registered, alone, with grid); then concurrently (R23 slots): R10 rows of 2a and of the reader
(reader 'gates given' R14, 5 seeds); R17 (i) and the residual on every saved model; R24 BERT-MRC on bert-base-uncased, 5 seeds, both
depth variants, NULL, cost with all questions. Then JP 1 with STATUS, RESULTS, JOURNAL, current_status, and the paper outline. After
JP 1: R19 (BERT-tiny / BERT-mini in BOTH forms, tagger and MRC, because size is the question there), R13 (seeds 5-9 for the reader and the
two main BERT rows), then the deferred rows only if they still answer a question after MRC is seen. The pre-registered rows all stay and
are all reported, the tagger included: it is the JointBERT form that products use, and the difference between the two BERT forms is itself
a finding.

## R26 (2026-10-25): the query-emission row, for every model, after JP 1 and after the format gains query wires

q-formalisation (vss-demo, vision/ADDENDUM_qformalisation_2026-10-25.md, sections 7 and 7.1) requires that nothing unknown is silent or
invented: every slot is a value, or a query of a stated kind (undecided, unplaced, unknown, unsupported, inferred). Neither the reader nor
any BERT emits queries today; the reader will (demo plan 2.8). When it does, one row for every model, scored on the OUTPUT and not on the
mechanism: a planted-questions world (300 basic-world sentences with an extra piece of information no question of the legend asks for,
gold = that slot is `unsupported`; 300 with two equally supported candidates, gold = `undecided`; the out-of-world sets, gold = `unknown`;
the normal frozen splits, gold = values). Each decision is scored as: correct value, rightly queried, needlessly queried, or INVENTED (a
value emitted where the gold is a query, or a wrong value emitted confidently): the last is the hallucination rate. Reader: queries from
margins (undecided), unplaced words, residual (unknown) and a coverage test on the state (unsupported: a content token whose identity is
not recovered by any decoded slot). BERT-MRC: NULL for not given; margin for undecided; and, so that it is not disadvantaged, a COVERAGE
head trained on the same planted negatives the reader's residual threshold is calibrated on (a token-level "covered by some answer"
tagger), for unsupported and unknown. State in RESULTS that for MRC the detection is a trained head and for the reader a quantity of the
state; both are reported as what they are. Criteria written now (the demo's): values emitted under the threshold = 0 for every model by
construction of the row; planted unsupported slots reported as queries >= 0.90; the hallucination rate reported per axis. Not before JP 1
and not before the format change; declared now so that the planted worlds are built and frozen before any model is tuned with them in view.

## R27 (2026-10-25): a whole-design review for asymmetries; five fixes, all declared, none touching the pre-registered rows

1. **Kinds are given to the reader and not to BERT.** The reader receives every token's category from kinds.tsv (a given, rule 20) and
   its route book is indexed by category; BERT receives text only. The paper handled exactly this for the from-scratch control with two
   variants (7b plain, 7b-bis with categories); BERT has no counterpart here. Fix: a "kinds given" row for BERT, tagger and MRC: one
   learned embedding per kind added to the token embedding (as 7b-bis adds its category input), configuration inherited, 5 seeds. If it
   changes BERT's numbers, the reader's advantage on that axis was the given categories, and that is what is written. Before JP 1 for
   the MRC form (it runs in the same queue), after JP 1 for the tagger.
2. **The reference ceiling is missing: a large language model with the legend in the prompt, no training.** That is what the field uses
   for this task today and what a reviewer asks first. It is not a competitor on cost; it is the accuracy ceiling and the anchor of the
   cost argument. Fix: a reference row (labelled "reference, not a baseline"): a frozen prompt (the legend's questions, the SOP format,
   ten solved examples from the training split of seed 0, written once and committed before any call), temperature 0, the same scorer,
   on a fixed subset of every axis of seed 0 (the report half only; 200 items per axis, cost), with the hallucination count (values given
   where the gold is not given), the cost per sentence (tokens, time, price) and three repeated calls to measure its own determinism. Use
   the API available on the DGX session (the one this session uses); if none, write the row as PLANNED with the prompt committed.
3. **Statistics.** "Within seed variance" is a reproduction check, not a comparison. The seeds are shared (same splits per seed), so
   model differences are PAIRED: report, for every pair of models and axis, the mean paired difference with its 95% t interval over the
   5 seeds and the number of seeds in which each model is ahead; and a bootstrap over items (1,000 resamples) for the interval of every
   single row. Add to eval/compare.py; recompute for every row from the stored per-seed numbers; no rerun.
4. **Inference cost is not measured symmetrically.** BERT at 20 threads, batch 32, torch; the reader in numpy, sentence by sentence.
   Fix: every model at batch 1 and batch 32, at 1 thread and at 20 threads, same machine, precision stated, median and p95 over the same
   500 sentences; report the four cells for each. Energy per sentence from the sidecar at batch 32 only.
5. **Real data and WikiSQL regime statements.** On SNIPS / MASSIVE / TOPv2 the reader's kinds come from skip-gram clusters on the training
   pool (unsupervised, LEARNED, not given): write it in INSTRUCTIONS_REAL and in every real-data row. On WikiSQL our regime (question
   without the table, columns as candidates) is below the published table-conditioned systems (about 0.90 selected-column accuracy with
   the table); the comparison is BERT against the reader in the same regime, never against WikiSQL's state of the art; one sentence in
   RESULTS 5 and in the paper.
Asymmetries that stay, each stated in the threats-to-validity section: pretraining (the question itself), worlds built by us and a reader
developed on them (R12), depth given at decoding to every model (equal; plus the all-questions variant), candidates restricted to the
world's vocabulary for every model (equal), a grid for BERT and none for the reader (in BERT's favour). A second independent review of
the DESIGN (not the code) is being made from the reviewing side, briefed as a semantic-parsing reviewer; its findings follow as a response.

## R28 (2026-10-25): the paper is about q-formalisation, and the main comparison moves to equal size

Decision of the author. The paper's question is q-formalisation: answering the legend's questions, "not given" when a role is absent, a
margin per decision, a residual, a query where the model does not know, in one pass, in a model that fits a browser. A model that
answers well but cannot refuse, cannot say what it did not read, or does not fit in a few megabytes does not solve the problem the paper
poses. Therefore:

1. **The main comparison is at the reader's size, in the MRC form (R24).** Models: the reader (3.3 M; 13 MB, 1.7 MB at 4 bits); BERT-tiny
   (google/bert_uncased_L-2_H-128_A-2, 4.4 M); BERT-mini (L-4_H-256_A-4, 11 M); BERT-small (L-4_H-512_A-8, 29 M) if its official Google
   upload loads with transformers 5.20.2 (do not use prajjwal1; no workaround). Each in three variants, same grid (R19's grid, fixed), 5
   seeds, both pre-registered splits and every declared axis: (a) pretrained; (b) the same architecture from scratch; (c) pretrained with
   the kinds given as an input embedding (R27.1). This separates pretraining, form and the given categories at equal size. All axes:
   seen, new ids, new formulations, depth 3 (per-role, six-role FULL, trained labels, inner), depth not given (all questions, NULL),
   out-of-world rejection (R17: NULL as trained, and the max-probability threshold), residual analogue, symmetric cost (R27.4).
   R19 moves from "after JP 1" into the main queue, in this form, and is run first after 3a.
2. **BERT-base and DistilBERT become reference rows**, kept exactly as measured (0, 1a-1c, 2a, 2b, 3a) and reported; no further GPU hour
   is spent on them: the R10 / R14 / R17 / R20 rows of the tagger on BERT-base are deferred without date. The reader's R10 and R14 rows
   stay (they are the reader). The one BERT-base row still worth running is R24 MRC on bert-base-uncased, as the LARGE reference of the
   MRC form (one row, 5 seeds, both depth variants), so that the size axis has its top point in the same form as the small models.
3. **Run 3b is not executed.** It was pre-registered; its purpose ("pretraining or form?") is answered better by variant (b) at every
   small size, and by (b) on DistilBERT it would cost an hour for one point. Record in RESULTS, before its scheduled time, a section "3b:
   not run (deviation from the pre-registration, R28)" with this reason, and in JOURNAL as a decision point; a pre-registered row that is
   not run is reported as such, never silently dropped. If 3b has already started when this is read, let it finish and report it.
4. **The LLM in-context reference (R27.2)** stays as a reference on a small subset, with cost; it is the ceiling, not a baseline.
5. **Real data (after JP 2)**: the same small models in the MRC form, plus BERT-base as the reference row; R8's reproduction check
   applies to the small models too (published small-BERT numbers where they exist, otherwise BERT-base's).
6. **Statistics and cost** as R27 (paired differences over seeds, item bootstrap, four cost cells per model).

Queue now: 3a finishes (pre-registered, alone); then concurrently (R23 slots): the reader's R14 row; R17 (i) and the residual on saved
models; R24 MRC on bert-base (the large reference); then the equal-size block of item 1 (tiny, mini, small x a, b, c), concurrently.
JP 1 after the equal-size block, with STATUS, RESULTS, JOURNAL, current_status and the paper outline written around q-formalisation
at equal size. Write the new queue in RUNBOOK before starting it.

## R29 (Q18, Q19, Q20; 2026-10-25)

**Q18.** (1) Accepted: (a) training asks the questions of levels 0-2 only, so the level-3 question is zero-shot at test; (b) a second row
with all six questions (level 3 always NULL in training), labelled; both reported. (a) is the main row. (2) The wording is accepted and
frozen as written; X1 = the outermost speaker, as the world numbers the levels; say so under the table. (3) Tag prereg-2026-10-25b on
the commit that writes deviation R24 with the questions and the decoding rule, before any MRC training on real data; the first tag
stays. (4) DistilBERT-MRC after JP 1, and under R28 only if the small-encoder block leaves a question it would answer; it is a reference
model now. (5) Accepted as you read it: the R10 row of 2a (5 seeds), the reader's R10 rules on all five seeds from the R17 reader run
(EQUAL check against 1a first), the reader 'gates given' row, R17 (i) and the residual on the reader, 2a re-run with every seed saved
(exit: reproduces 2a) and BERT-MRC; the R11 re-runs of 1b / 1c stay before JP 1 in a free slot (small, and R11 required them); 2b / 3a /
3b and the controls' R17 after JP 1.

**Q19.** Accepted as the frozen construction: `unsupported` = ", <id> ." appended with an identity not in the sentence (gold frame
unchanged, that token's gold = unsupported); `undecided` = a second identity of the same class inserted right after one role's filler
(gold for that role = undecided). State the limitation in RESULTS: the appended identity is a weak form of "extra information" (a person
reads it as an afterthought with no role), chosen because the vocabulary has no adverb or conjunction; a second construction with new
words (which the reader cannot encode and reports as unplaced by construction) is NOT added, since it collapses into source (c). Build
and commit the sets now; the row runs after JP 1 and after the query-wire format exists (R26).

**Q20.** Proposed settings, PENDING the owner's confirmation (it is his money; no call until a later response says "Q20 confirmed"): provider OpenRouter with the key the DGX has; model `anthropic/claude-sonnet-5-5`
(a current mid-size Claude model; the ceiling row does not need the largest model, and its price keeps the run near the low estimate);
prompt caching on if OpenRouter exposes it for that model; temperature 0; a budget cap of 20 USD at which the script stops and reports
how far it got (write the spent amount from the provider's usage fields into the log after every 100 calls); the synthetic sentences
contain no personal data and may be sent. Record model id, provider, date, prices read from the provider's page at call time, and the
three-repeat determinism in RESULTS; label the row "reference (in-context LLM), not a baseline". Not on the critical path: once confirmed, run it in a
free slot (it is API-bound, not GPU-bound) alongside the equal-size block.

## R30 (independent design review, relayed with decisions; 2026-10-25)

A reviewer who had not seen this experiment being built (a semantic-parsing read, given the repository and the frozen data) returned
twelve findings. I relay them in full so that nothing is hidden, with a decision for each. Nothing here changes a pre-registered row;
every addition is a declared row or an evaluation script. Order of work: nothing below runs before JP 1; the small-encoder block of R28
is still first; the rows marked NOW are declared before the block's results are read, so they are confirmatory, not post hoc.

1. **The basic world is solvable by a rule** (the reviewer wrote a ~15-line heuristic over the given P/O kinds and the words says / claims
   / that / to / by / from: FULL 1.000 on seen, new ids and all depth-3 items with depth given; 0.833 on new formulations, where the reader
   has 0.868). ACCEPTED, important. Add a **rule-ceiling row** and a **positional / bag-of-words row** on every axis (no GPU; commit the
   rule as eval/rule_ceiling.py with its score). Consequence written into RESULTS now: on axes where the rule reaches 1.000 the axis
   measures whether a model learned the rule, not composition; the composition evidence must come from axes where the rule fails (new
   formulations, depth not given, out-of-world, the planted sets) and from the equal-size block.
2. **One generalisation type is not a composition benchmark.** ACCEPTED in part. Add, if the generator takes a depth parameter without a
   change to its templates: **depth 4-6 with depth NOT given** (trained on 0-3), as the composition axis proper; a **length control**
   (depth-2 sentences at depth-3 length, if the vocabulary allows it; otherwise recorded as not possible); a **primitive-to-role row** (an
   identity seen only as A, tested as C and as a speaker). An MCD split is NOT added (the generator's compound structure is too small for
   compound divergence to mean anything; say so). A seq2seq baseline (T5-small emitting the frame) is added as ONE reference row, last in
   the queue, labelled reference: it is the standard model of COGS / CFQ and has no untrained-label problem, but it is not at the reader's
   size, and the user's rule stands: no further time on models above the reader's size beyond one row each.
3. **The remaining "composition" gap after 2a is the untrained X3 label.** ACCEPTED. Add a **depth-invariant tagger row** (one SPK label,
   level = order of appearance) beside MRC, and stop narrating six-role FULL as composition in RESULTS; the words to use are "frame exact
   match with the levels as labels". The from-scratch statement: 3b is not run (R28), so the from-scratch evidence at BERT-base size is the
   7b control and nothing more; the equal-size block (BERT-tiny/mini/small from scratch, R28) is the test that counts, and it is written
   NOW, before it runs: if a from-scratch encoder of the reader's size reaches an inner frame >= 0.99 at depth 3 with depth not given,
   the claim "same-size models trained from scratch do not compose this world" is WITHDRAWN, not re-narrowed.
4. **New identities are a lexical giveaway** (`p12` -> `p ##12`; the reader has the kind from kinds.tsv). ACCEPTED: a **nonce-identity
   row** (identities with no shared prefix, e.g. blick / dax / wug, both BERT conditions; the reader with the new identities given in
   kinds.tsv as the rule of kinds given requires, AND a second cell with them absent, so both sides' conditions are stated). The present
   new-identity numbers stay, with the sentence that they measure the given kinds, not generalisation.
5. **Depth given at decoding is the largest giveaway.** ACCEPTED: the headline table of RESULTS is depth NOT given / all questions, for
   every model (the R17 and R20 rows and MRC already produce it); given-depth beside it, labelled.
6. **Baselines.** (a) A retrieval-augmented in-context reference and LoRA rows: NOT added (budget and the rule above; the LLM row is a
   reference, stated as 10-shot, not data-matched, in its caption). (b) seq2seq: see 2. (c) the 7b control with a tagging head: ACCEPTED,
   small (it separates the head from the pretraining; 7b's mean-pooled cosine decoder is a weaker head than BERT's tagger).
7. **Standard metric names and selective-prediction measures.** ACCEPTED, evaluation only: FULL is reported as "exact match", per-role as
   "slot accuracy"; add **risk-coverage curves and AURC** for every model, **AUROC in- vs out-of-world** on margin, residual and MSP; ECE
   omitted where accuracy is 1.0; the words "confident error" and "false fill" replace "hallucination" in RESULTS (the term may appear
   once, defined as their sum). The fair analogue of the reader's residual on the BERT side is a **feature-space OOD score (Mahalanobis or
   kNN on encoder states)**, added to R17 (i) beside MSP; the tagger's in-world residual is degenerate and RESULTS says so.
8. **Statistics.** ACCEPTED. Seeds are bimodal; mean +- std and a 4-df paired t are reported but not relied on. Add a **two-level bootstrap
   (seeds x items)**, seeds-ahead counts and a sign test (extend eval/compare.py). Three **confirmatory comparisons, pre-specified NOW**,
   reader vs each equal-size encoder (pretrained and from scratch) and vs BERT-base MRC: (i) six-question exact match at depth 3, depth
   not given; (ii) confident-error rate at matched coverage (coverage fixed at the reader's); (iii) out-of-world AUROC (margin / residual
   vs Mahalanobis). Holm over these three per pair; every other row of R10-R27 is labelled exploratory. The grid question (seed 2 as a
   least-trained-point artefact): ACCEPTED as a small run, the 9-point grid x 5 seeds evaluated on depth 3, after the block.
9. **Real data: the primary metric is shaped to the reader.** ACCEPTED: on SNIPS / MASSIVE / TOPv2 the primary metrics are intent accuracy,
   conlleval slot F1 and sentence exact match (TOPv2: tree exact match, with the note that the reader and the taggers produce a flat frame,
   so the tree metric is reported for the depth-2 subset only, with n and a power statement); the head-token frame is secondary and stated
   as the reader's limit. WikiSQL: add its known label noise and that candidate columns are partial table access.
10. **Cost.** ACCEPTED: equal precision only (fp32 vs fp32; the 4-bit route book reported separately, never in the same cell as fp32 BERT);
    batch-1 single-thread ms as the product number; cost against sentence length for both (the reader's chain grows with length, MRC
    with the legend); the 1 Hz energy reading stated as coarse.
11. **The term** (q-formalisation renames selective frame filling with an OOD score; drop "classical formalisation is the special case").
    Noted for the paper, not for this repository; the definition will be given as an evaluation protocol and the "special case" sentence
    removed. The narrowed claim stands as the reviewer states it.
12. **Pre-registration credibility.** ACCEPTED: RESULTS gets two tables, pre-registered rows and declared (post hoc) rows, with the R
    number and the date of each declaration; nothing is removed.

Queue after this response: JP 1 as planned; the R28 small-encoder block (with the confirmatory comparisons of 8 declared before its
results are read, and the withdrawal condition of 3); then, in order: 1 (rule ceiling, no GPU, can run now on CPU), 7 and 8 (evaluation
scripts, CPU), 3 (depth-invariant tagger), 5 (table layout), 4 (nonce identities), 2 (depth 4-6, length, primitive-to-role), 6c, 8's
grid, 9 and 10 at the real-data step, 2's seq2seq last. If any of these cannot be done as stated, say so in questions.md rather than
adapting it silently.

## R31 (Q21; 2026-10-25)

R28's order holds; R30's "JP 1 as planned" was a slip. JP 1 is the synthesis point and must see the equal-size block, which is now the
main comparison. So: 3a (done); the reader's R14 / R10 rows, R17 (i) with the Mahalanobis analogue of R30 item 7 when the script exists
(otherwise MSP now and Mahalanobis added at JP 1), the 2a R10 row and the 2a re-run with every seed saved, the R11 re-runs, BERT-base
MRC; the equal-size block with the three confirmatory comparisons of R30 item 8 computed as it finishes; then JP 1. R30's CPU items run
alongside, as you read them; its other GPU rows after JP 1 in R30's order. Q20 stays as you state it until the owner confirms.

Noted on 3a and the rule-ceiling rows: accepted as written, including the sentence that the reader's new-formulation score is below the
rule and below BERT; keep it in RESULTS in that form.

## R32 (Q20 confirmed by the owner; 2026-10-25)

Q20 confirmed, with one change: the Anthropic API directly, not OpenRouter. The owner sets `ANTHROPIC_API_KEY` in the DGX environment
(never in the repository, never in a log; the script reads the variable and fails with a plain message if it is absent). Model string
`claude-sonnet-5-5`, temperature 0, prompt caching on for the frozen prefix (the prompt and the ten examples; cache_control on that
block), the rest of R29 unchanged: budget cap 20 USD computed from the response usage fields and the published prices read at call
time, logged every 100 calls, stop and report at the cap; three repeats for determinism; the row labelled "reference (in-context LLM),
not a baseline", 10-shot, not data-matched (R30 item 6a). Run it in a free slot alongside the equal-size block; it does not wait for JP 1.
If the variable is not set when the slot comes, skip and say so in questions.md.

## R33 (Q22 confirmed; 2026-10-25)

Checked against the Sonnet 5.5 migration guide (platform.claude.com/docs/en/models/sonnet-5-5/migration-guide): your three points are
right. Sampling parameters return 400; a request with no `thinking` field runs adaptive thinking; `disabled` returns 400 and
`between_tools` is the lowest setting. Q22 confirmed, with one choice made: send `thinking: {"type": "between_tools"}` with
`output_config.effort: low` (without tools the reply is text only, no up-front thinking), rather than adaptive at low: the row is meant
to measure in-context READING of the legend's questions, and the fewer hidden tokens the model spends, the closer the row is to that and
the more repeatable its cost. RESULTS states: no temperature control available; thinking at its lowest setting; cache not expected to
engage (prompt under the cacheable minimum), cache fields logged. One price correction: Sonnet 5.5 cache reads are $0.10 per M, not
$0.20 (irrelevant here since caching will not engage; read all prices from the pricing page at call time as you planned). No server-side
fallback, as you propose. Go.

## R34 (Q23; 2026-10-25)

Decision: (B) for BERT-base MRC, (A)+(D) for the equal-size block, and NOT (C).

- **BERT-base MRC: (B).** It is a reference row since R28; it inherits 2a's selected configuration (lr 2e-5, 3 epochs, no grid), five
  seeds, stated as a deviation with the date. No grid hours on a model above the reader's size.
- **Equal-size block: R19's grid stays, with (D).** Stop a learning-rate chain once its validation loss is below 1e-6, and print the
  validation loss at full float precision in the log so that the tie-break of R10 (validation cross-entropy) is applied to real numbers,
  not to a rounded zero. If every chain stops at its first point, that is the result of the grid and is written as such. The same rule
  for pretrained, from-scratch and kinds-given variants.
- **Not (C).** Holding a template out of training changes the training distribution of the new rows against the reader's and against
  2a / 3a, and a selection that looks at depth-3 or out-of-world rows is exactly what rule 25 forbids; a protocol change after seeing the
  saturation would be read as tuning. The saturation is symmetric: the reader was selected under the same rule in the same world and the
  same validation draw, so neither side gains from it. Write that sentence in RESULTS beside the statement that selection on this
  world's validation set is uninformative (point 3), with the 4.0 % overlap figure and the rule-ceiling reference.
- Keep the note in JOURNAL that the saturation explains R10 item 3 (2a's selection at the first grid point).

The queue otherwise as declared; the LLM reference runs beside it as soon as a slot is free.

## R35 (Q23, amending R34 after the owner's question on time; 2026-10-25)

The owner asks whether the hours are needed. The evaluation sets are not the cost; the cost is the grid: 9 configurations x 5-20 epochs
on 150,000 examples, selected by a validation set that Q23 shows cannot discriminate. A grid whose selection is noise buys nothing, and
the (D) cap of R34 still leaves 3 chains x 5 epochs per model. Amendment, declared now and applied identically to every variant of the
equal-size block (pretrained, from scratch, kinds given; tiny, mini, small):

- **No grid for the equal-size block. One configuration per model, fixed by rule before any of its report rows is read:** the middle point
  of R19's grid, lr 1e-4, 10 epochs, batch as declared. Five seeds each. Deviation stated with the date and the reason (Q23 point 3:
  validation saturated on this world; selection by noise replaced by a fixed configuration, which is no less informed and is the same for
  every model). BERT-tiny chains already finished are kept and reported as what they are.
- **One sensitivity table, not a selection:** for ONE model (BERT-mini pretrained, seed 0 only) run the full R19 grid and report the
  report-axis numbers for all 9 points. It is read after the fixed-configuration rows and only says whether the fixed choice could have
  changed the result; it never picks a configuration. If the spread across the grid on depth 3 (not given) or out-of-world exceeds the
  seed spread, say so in RESULTS and in the paper's limitations.
- BERT-base MRC: (B) of R34 stands (2a's configuration, no grid).
- The reader's own rows are not re-selected; its selection ran under the same saturated rule and this is stated (R34).
- The LLM reference: unchanged, API-bound.

With this the block is 9 variants x 5 seeds x 10 epochs plus one 9-point grid on seed 0, and the slots can take all small jobs in
parallel as R23 allows. If a from-scratch variant has not converged at 10 epochs (training loss not below the pretrained variant's by
more than a factor of 10 at the end), report it and add ONE row at 20 epochs for that variant, labelled; do not tune further.

## R36 (the block's length, amending R35; 2026-10-25)

The owner asks again whether the hours are needed. Two further cuts, both justified by measurements already in the logs, both symmetric:

1. **5 epochs, not 10.** BERT-tiny's validation span loss is already 0.00000 at 5 epochs on every learning rate tried (Q23 point 1); a
   second 5 epochs at zero loss trains nothing. Fixed configuration for every variant of the block: lr 1e-4, **5 epochs**. The
   convergence clause of R35 stays and now reads: if a from-scratch variant's final training loss is more than 10x the pretrained
   variant's at the same size, add ONE labelled row at 20 epochs for that variant; nothing else is tuned.
2. **BERT-small after tiny and mini, and only if they leave a question.** The reader is 3.3 M parameters; BERT-tiny (4.4 M) and BERT-mini
   (11 M) bracket it, BERT-small (29 M) is nine times the reader and sits between mini and DistilBERT (66 M, measured). Run tiny and mini
   first (3 variants x 5 seeds each); read them; BERT-small's three variants run only if the picture differs between tiny and mini
   (then small says where it settles) and otherwise are recorded as not run with that reason, the same rule that deferred DistilBERT-MRC
   (R29 Q18 (4)).

Order on the GPU: all tiny and mini jobs in parallel in the six slots; the LLM reference beside them; BERT-base MRC (R34 (B), 5 seeds,
3 epochs) after them, alone, since it gains nothing from concurrency; the sensitivity grid on BERT-mini seed 0 at the end. The equal-size
block is then the main table with four rows above and below the reader's size, which is what the paper needs; everything larger is
reference.

## R37 (the R14 reader row; 2026-10-25)

Accepted as written, including the sentence that the given gates LOWER the mean on new formulations. Do not soften it. One mechanism
question that the row leaves open, and one cheap diagnostic row (the reader trains in about two minutes per seed) to settle it before
the paper interprets R14:

- The given gates are fixed at logit +8 with the gradient masked (fix_gates), i.e. hard-open, while a learned gate settles at a graded
  value. R14 therefore changes two things at once: WHICH gates are open and HOW open they are. Add one row, same seeds: **gates given as
  initialisation** (the same six set open at the typical learned logit of 1a, read from the saved 1a models, and left trainable). If that
  row recovers 1a's mean on new formulations with R14's seed spread, the loss in R14 comes from hardness, not from the set, and the demo's
  default (gates given) should be "given as initialisation". If it does not, the loss comes from the set, and the learned-by-seed sets
  that beat it must be listed in RESULTS so that the paper can say which cue words a seed found useful that the legend did not give.
- State in RESULTS that the rule ceiling is 1.000 on new formulations with the same six words, so the reader's 0.76-0.87 there is a
  limit of the reader, not of the information it is given. That sentence is unfavourable to us and belongs in the paper.

Nothing else changes; the row runs in a free CPU/GPU slot and does not delay the block.

## R38 (improving the reader is allowed; the protocol for it; 2026-10-25)

The owner's rule: where the reader can be improved, it is improved, even if that invalidates earlier results. So that this never looks
like tuning on the report rows, the protocol is:

- The reader in this repository stays the frozen reference (1a) and every pre-registered row stays as measured. An improved reader
  enters as a NEW, declared row with its own version (the third_party submodule commit written in RESULTS), the same seeds, the same
  frozen splits, the same selection rule; the old row is never replaced, the table gains a line.
- The improvement is designed from TRAINING-distribution and calibration evidence and from error analyses, never from the report rows'
  numbers; what was looked at is written in JOURNAL before the new row runs.
- First target, from today's evidence: new formulations (reader 0.76-0.87; the rule and BERT 0.98-1.00). To design it I need the
  material, which is CPU work of minutes: commit, for 1a (five seeds) and the R14 row, a per-item dump on the new-formulation report
  rows (sentence, gold frame, predicted frame, per-role margins, the gate values of every token, the residual), one JSON per seed under
  results/, plus the 50-item confident-error analysis already planned, by kind (wrong role, missed role, false fill, wrong candidate at
  a given gate). I will read it from the other side and propose the change in the tool repository; the row then runs here as above.
- The R37 diagnostic row (gates given as initialisation) is the first candidate of this kind and already follows the protocol.

## R39 (Q24; 2026-10-25)

You are right and the objection is accepted in full. Default (1)-(4) adopted: seeds 0-4 become the inspected design set once the dump
is read; basic_seed5-9 are generated and committed NOW, before the dump exists, with their sha256 in the commit message, and nobody
(including me) opens them; the improved reader's claim is made on seeds 5-9 against 1a and the equal-size models on the same fresh
splits; its numbers on seeds 0-4 are reported as "design set (inspected)". State the limit as you phrase it: the held-out generalisation
is over identities, draws and transitions, not over templates. Add one line to JOURNAL when the dump is read, naming the files opened.

## R40 (Q25; 2026-10-25)

Q25: yes, as you propose: the 'all' rows for BERT-tiny and BERT-mini pretrained only (5 seeds, the fixed configuration), after the 'zs'
block, labelled. They matter for one sentence of the paper, not for the main table: a deployed MRC reader is trained with every question
of its legend, so 'all' is how it behaves when a question that was always NULL in training becomes answerable (the realistic form of
"not given" turning into "given"); 'zs' is the fair capacity test. Report both with that sentence; if 'all' refuses level 3 on every
seed, that is the expected result and is written as such, not as a failure of BERT.

The transparency note on 33dc482 is read and accepted; no number changes.

## R41 (the reader's R17 (i) row: accepted, and a definitional defect of the residual; 2026-10-25)

Accepted as written: the reader does not meet the demo criterion either, and the sentence stays. The row also exposes a defect in the
residual's DEFINITION, visible from calibration numbers alone (residual 0.43 in-distribution, where every token is a known word in a
decoded role): `common/decode.py residuals` and the tool's `infer.py residual` compare the reader's state S with `rec` = the COMPILER's
state of the decoded frame (role addresses only). S also carries every token's deposit through the routes (path, chainL, bigram),
which `rec` never contains, so the residual is large even when everything is explained and grows with the number of tokens: depth 3
(more clauses) 0.72, long new formulations 0.77-0.90, random 0.99. It measures length as much as unexplained content, which is why a
training-distribution threshold rejects every depth-3 sentence.

Proposed, under R38 / R39 (it changes no accuracy, only the rejection score; the old residual stays in every table):
- **Explained-tokens residual**: rec' = the reader's OWN chart of the same token sequence with the unexplained tokens removed (explained =
  fillers of decoded roles, speakers, the event lemma, and every token of a non-content kind: cue words, openers, punctuation; the
  `placed_words` set already exists); residual' = ||S_n - a rec'_n||. When every token is explained, residual' = 0 by construction; a
  nonce or unplaced token contributes its share of the state; a content word in no role contributes its share. Deterministic, parity
  testable (Python / JS), no training.
- **Check first, on calibration data only** (no report row): Spearman correlation between the old residual and the token count on the
  calibration sentences of each seed, and the old residual's mean by depth 0 / 1 / 2 within the calibration rows. If the correlation is
  strong (> 0.5) and the means rise with depth, the diagnosis stands and the new residual is implemented in the tool repository (by me,
  committed there with its parity test), then run here as a declared row: AUROC in-world vs out-of-world and the 2 % threshold table,
  claim on seeds 5-9 (R39), seeds 0-4 reported as inspected. If the correlation is weak, the diagnosis is wrong and I will say so here.
- The same check for the taggers' MSP is not needed (a probability does not scale with length), but the Mahalanobis analogue of R30
  item 7 remains their fair score.

JOURNAL: name this response as the design source and the calibration files opened for the check.

## R42 (the explained-tokens residual is implemented in the tool repository; 2026-10-25)

Done on the vss-demo side (commit a1f4564): `tool/sop/infer.py residual_explained` and the browser mirror, switched on by
`residual: "explained"` in a model's config; the default stays the paper's definition, so every existing parity test is unchanged
(PASS), and an explained copy of the reference smoke model passes parity 55/55 (Python / JS residual difference < 1e-9). On the smoke
sentences the explained residual is 0.000 at the median (every token explained) and equals the share of the unexplained tokens
otherwise (one of sixteen: 0.26; two of four: 0.63).

Exact definition for `common/decode.py` (so that this repository's row is the same quantity): explained = {tokens whose kind is not a
content kind and not `unknown`} ∪ {fillers of the decoded roles} ∪ {speakers} ∪ {the event lemma}; rec' = Σ over explained kept
tokens of address_i ⊗ F_i, the reader's own chart restricted to them; residual' = ||S/|S| − a·rec'/|rec'||, a = max(0, Re⟨rec', S⟩) on
the unit vectors; 1.0 when no token is explained or a role of the depth is not decoded (as the compiled residual). For the nonce rows
the placeholder star of R18 is deposited and is never explained, so it contributes its share. The reader itself is not retrained: the
row re-scores the saved 1a models.

Run order unchanged: the calibration-only check of R41 first (correlation with length; it stands or falls there), then the row as a
declared, dated deviation, claim on seeds 5-9, both residuals in the table.

## R43 (the R41 diagnosis is refuted on calibration data; what remains; 2026-10-25)

The seed-0 calibration check you disclosed in JOURNAL (Spearman of the compiled residual and the token count −0.152; means by depth
0.410 / 0.455 / 0.432) refutes the mechanism I wrote in R41: the compiled residual does NOT grow with length. I was wrong on that and it
is recorded here. What the numbers do support: (a) the compiled residual has a floor of about 0.43 on the training distribution, the
share of the state that the compiler's reconstruction never contains (the routes' deposits); (b) above that floor it rises with the
NOVELTY of the form (unseen transitions 0.90, depth 3 0.72, random order 0.99), not with the token count. It is therefore a detector of
distribution shift in the form, which is exactly why it cannot separate "new but in the world" from "out of the world".

Consequences:
- Run the full calibration check on all five seeds anyway (two numbers per seed) and write it in RESULTS as the refutation of R41's
  mechanism; keep the sentence that the floor comes from the reconstruction.
- The explained-tokens residual row stays as declared, with its rationale corrected: it scores only what the decoded frame leaves
  unexplained and is insensitive to form novelty by construction (0 whenever every token is placed), so it is the right score for the
  query wire of kind `unknown`, and the compiled residual is the right score for "the form is unfamiliar" (a separate signal, worth
  keeping for abstention). Both are reported; the AUROC comparison on seeds 5-9 decides which separates out-of-world better, and the
  expected failure mode of the explained residual is written now: random in-vocabulary sentences that the decoder nevertheless fills
  into a frame leave few tokens unexplained and may pass.
- Nothing else changes.

## R44 (the transparency note on seeds 5-9; 2026-10-25)

Both as you mean them. The fresh seeds 5-9 need their own reference-world splits, out-of-world sets and 1a readers (same generators,
`--seed_start 5`, sha256 before any model sees them; the 1a readers tagged apart), and the lemma adaptation is right: in the basic world
the first token of an event-verb kind is the event, and the verb is explained by the frame. Write the adaptation in the row's caption.
No objection; proceed.

## R45 (Q26; 2026-10-25)

Both accepted: (a) one job per seed, and (b) up to 12 slots for BERT-tiny / BERT-mini after the one measurement R23 prescribes (keep 12
if the throughput gain is at least 1.3x, else 6), memory rule and guard unchanged, BERT-base alone. Per-seed numbers are unaffected;
the concurrency count stays in every log line as now, and no concurrent time is a cost number.

## R46 (first reading of the design-set dumps; a decoder-only improvement proposed; 2026-10-25)

Files opened: results/dump_reader_basic_cues_seed0-3.json (the R14 row; the 1a dumps are not written yet). JOURNAL: seeds 0-4 are the
inspected design set from now on, as R39 says.

**What the errors are.** They are not seed noise and not swaps. On every seed the same cells fail, nearly completely: template 10
("from p11 , p8 receives o6 .": C, the subject after a fronted "from" phrase, 176-180/180), template 4 ("p10 gives p4 o3 .", the double
object without "to": B 165-180/180 and C 102-164/180), template 7 ("it is o1 that p9 gives to p6 .", the cleft: A 155-171/180),
template 11 ("p9 receives from p8 o0 .": B 83-140/180), template 5 partly. In 72-77 % of the wrong decisions the predicted word is NOT in
the sentence (pred_pos −1) and the median margin is 0.010 (under tau: the demo would abstain, which is why confident error stays low
while coverage falls); the gold filler's rank is spread from 2 to 12, i.e. the role's address holds no signal at all, not a near miss.
Template 7 is the clearest mechanism: "that" opens a level gate, so the cleft's "that p9 gives" is deposited as a reported clause and
A is read at the wrong level. Templates 4, 10, 11 are fillers placed after a cue (or in a slot) that in training preceded a DIFFERENT
role, so they sit at an address the role's readout never visits. This is H2 (the address a filler receives follows the cue chain, and a
new order puts it in another role's address), with H1 as the trigger in template 7. H3 (close candidates) is NOT the pattern.

**Why the rule and BERT do not fail here.** In this world the category decides most of the role (every O is B; among the P's the verb's
side and the cue decide A vs C), and both the rule and an extractive tagger choose among the words OF THE SENTENCE. The reader's cleanup
chooses among all catalog words of the role's kinds, so when the address is empty it returns a random catalog word.

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

If the 1a dumps (learned gates) show the same cells, the row runs on the 1a readers too; the R37 row (gates as initialisation) stays as
a separate question (template 7 may move with it).

## R47 (Q27 and the H1/H2/H3 note; 2026-10-25)

Q27 accepted: the decoder-only row reports, per role decision, "decided by the address" (the in-sentence winner is also the catalog
winner), "decided by restriction" (the catalog winner was absent from the sentence) and "decided by elimination" (one-to-one assigned
the remaining candidate), and RESULTS says in the row's caption that it changes the decoder's candidate set, not the addresses: the H2
cause (the position "P right after the verb" and the object after a P are not encoded as C / B by any route) stays unfixed by it, and
the margin of a decision by elimination is the single-candidate floor, not evidence.

The note is accepted as the diagnosis, including its separation of the loss: a shared core (T4:C, T7:A) that no gate set changes, and
a gate-set effect ("from" open in training breaks the object-last reading) that explains R14's drop against 1a. Two consequences:
- R37 runs next as planned, with the prediction you wrote (T4:B + T11:B errors present or absent decide set vs hardness); nothing
  else about it changes.
- The address-level fix of H2 is a change of the routes (a route that encodes a filler's category order relative to the verb
  independently of the cue chain) and needs retraining; it will be designed in the tool repository from this note and the dumps, enter
  as its own declared row (R38), and is not scheduled before JP 1. The decoder-only row is what runs now, as the extractive assumption
  the baselines already have.
- For the demo's worlds the cue-gate default is withdrawn until TODO_DGX 2.9 is measured (the finding that opening "from" in training
  breaks another reading applies to any given set).

## R48 (stopping rule, declared by the owner; 2026-10-25 17:50)

The main comparison is at the reader's size and in the MRC form, and the first result is in (BERT-tiny: inner frame 0.97 pretrained,
the legend question never asked in training answered on no item, X3 0.000; from scratch 0.17). So that the stop is on a declared set
and not on a favourable glance, the set that completes the result is fixed now; when it is measured the experiment stops, whatever the
numbers say, and everything else is recorded as "not run: the declared set is complete".

Remaining, in this order:
1. BERT-mini pretrained, from scratch and kinds given; BERT-tiny kinds given (running).
2. The 'all' rows for BERT-tiny and BERT-mini pretrained (R40).
3. The reader rows already declared: R37 (gates as initialisation), R46 (sentence candidates + one-to-one, with the three decision
   categories), R42/R43 (explained residual), each with its claim on seeds 5-9 and the 1a readers of seeds 5-9.
4. The planted sets row (R26 / Q19) for the reader and for the small MRC models: 'undecided' and 'unsupported' as query emission.
5. The three confirmatory comparisons of R30 item 8 with the two-level bootstrap and Holm, reader vs each small model; risk-coverage and
   AURC, AUROC in- vs out-of-world (margin / compiled residual / explained residual vs NULL gap / MSP / Mahalanobis where the saved
   models allow it); the cost table at equal precision (R30 item 10).
6. The LLM reference (API-bound, running beside).
Then JP 1 = the final synthesis: RESULTS with the two tables (pre-registered / declared), the confirmatory results, the regime
statements, the limitations (the basic world is rule-solvable; the claim is at 4-11 M parameters; refusal not met by anyone at a
calibration threshold).

Not run, by this rule (write each in RESULTS with the reason "declared set complete"): BERT-base MRC (the owner avoids training a
110 M model; the claim is stated at the reader's size, BERT-base stays as the measured tagger reference), BERT-small, the sensitivity
grid, R30 items 2, 4, 6c and the depth 4-6 / nonce-identity / 7b-tagging-head / seq2seq rows, the deferred BERT-base tagger rows, the
real-data round and the multilingual round (they move to a later paper with their own pre-registration). The address-level fix of H2
(R47) is tool-repository work after the stop.

## R49 (addition to the declared set of R48: the cost of a new legend question; 2026-10-25 18:10)

One more row family, small, on saved models, declared now with its measures; it enters the R48 set between items 2 and 3.

**Legend growth.** The question is what the n-th new legend question costs each system, in annotated examples, retraining time,
hyperparameters to choose, and accuracy lost on what already worked.
- BERT-tiny and BERT-mini pretrained, 'zs' saved models (5 seeds each): the level-3 question is added with k = 50 and k = 500
  annotated examples (drawn from the training distribution's depth-3 sentences, disjoint from every frozen split, committed), under two
  practitioner regimes: (a) continued fine-tuning on the new examples only; (b) replay (the new examples added to the full training
  set, retrained from the saved model). One fixed recipe per regime declared before the run (the block's lr 1e-4; (a) 3 epochs over the
  new examples, (b) 1 epoch over the union), plus ONE alternative lr (3e-5) for (a), to measure whether the choice matters; report
  which was needed. Measures: the new question (X3 at depth 3, depth not given); every pre-existing report row before and after
  (forgetting = the drop, per axis, with the seed std); wall-clock alone on the GPU (not concurrent); the number of examples; the
  number of hyperparameter choices made.
- The reader: the compositional case needs no run (the level-3 question is answered at 0.997 by the saved 1a models with zero examples
  and zero retraining: that line is the row). The non-compositional case, so that the comparison is not one-sided: one held-out
  template of the new-formulation axis (template 4, the double object) is added to training with k = 50 and 500 examples, the reader is
  retrained in full with the same seeds (its only regime; no continuation exists), wall-clock alone measured, every pre-existing row
  checked EQUAL or within seed std, the template's new-formulation cell reported before and after. The same template with k = 50 /
  500 for BERT-tiny / mini under (a) and (b), so that both systems face a question that is new for both.
- Table: rows = system x regime, columns = examples, minutes alone, hyperparameter choices, new-question accuracy, worst drop on an old
  row. Statements limited to this world; the sentence that replay removes forgetting at the price of full retraining is written whether
  or not forgetting appears under (a).

## R50 (Q28; 2026-10-25)

Accepted as the default: seeds 0-4 labelled "design set (inspected)", the claim on seeds 5-9, the k examples disjoint from both ranges'
frozen splits. If the small MRC models of seeds 5-9 do not exist by then, train BERT-tiny pretrained 'zs' on seeds 5-9 (5 jobs, one per
seed, in the 12 slots) rather than reporting the BERT side on inspected seeds only: the growth table is a claim and needs the fresh
seeds on both sides; BERT-mini on 5-9 only if the slots are idle.

## R51 (the owner cuts BERT training to what the result needs; 2026-10-25 18:30)

Amendment to R48 / R49, effective now; whatever is already running finishes and is reported, nothing new starts outside this list:
- BERT training that still runs or starts: the BERT-mini rows already launched (finish them); BERT-tiny pretrained 'zs' on seeds 5-9
  (5 short jobs, needed for the growth claim and the confirmatory comparisons on fresh seeds); the growth regimes of R49 on BERT-tiny
  only (saved models; minutes). Nothing else.
- Cut: the 'all' rows of R40 (the growth rows answer the same objection better: the question added WITH examples); BERT-mini from
  scratch and BERT-mini kinds given if not yet started (the withdrawal condition is settled at BERT-tiny; one pretrained BERT-mini row
  is enough for the size axis); BERT-mini in the growth rows. Each written as "not run: R51".
- Everything about the reader stays (R37, R46, R42/R43 on seeds 5-9; planted sets; growth); the confirmatory statistics, the cost table
  and the LLM reference stay (no GPU).
- JP 1 as soon as this set is measured. The effort moves to the paper and to the demo's loop (the missing question recovered from the
  journal; query wires reaching the person, the planner or a model), which is where the specific difference is shown rather than measured.

## R52 (R37 design-set dumps read; 2026-10-25)

Files opened: results/dump_reader_basic_cuesinit_seed0-4.json. With the six gates given as a trainable initialisation, 'from' stays
open on every seed and the T4:B + T11:B cells fail on every seed (86-179 of 180 each), exactly as the note's prediction for "the loss
comes from the set": hardness is not the cause. Reading for RESULTS: a cue that is a genuine role opener in one frame ('from' = the giver
of receive) breaks the object-last reading in another, because the routes compose a filler's address from the cue chain (H2); giving
gates does not add information the routes can use, it only fixes which of the route-level failures appear. Consequences: the R14 and
R37 rows are reported as measured, no further gate rows; the demo keeps the given-gates default withdrawn (TODO_DGX 2.9); the route-level
fix (a route that encodes category order relative to the verb independently of the cues, R47) is the only change that addresses the
core, and it is tool-repository work after the stop.

## R53 (R42 and R46 on the design set, read; 2026-10-25)

Accepted as written, including the two things that went against my expectations: the explained residual alone is blind to the
reference world (0.464) because the catalog decoder places catalog identities into some frame, and the "random sentences pass" failure
did not occur. Together (R46 frames + explained residual) the design-set numbers are: UNSEEN formulations 0.868 -> 0.975, depth-3 exact
match 0.995 at coverage 0.997 and confident error 0.001, AUROC in vs out 0.975, rejected random / reference / nonce 0.99 / 0.91 / 1.00 at
2.6 % false rejection. Three sentences for RESULTS and the paper, written now so that the seeds 5-9 row is read against them:
(1) these are design-set numbers; the claim is the seeds 5-9 row, nothing else; (2) the demo criterion (>= 0.90 per source at <= 2 %
false rejection) is met on two sources and missed by 0.6 points of false rejection, which is reported as not met; (3) the remaining
error is one cell, T10 (the fronted 'from' phrase: A and C swapped, both addresses holding the other filler), which elimination cannot
resolve and only the route-level fix can. The decision provenance stays in the table (T7:A is right by restriction and elimination, not
by the address), so that nobody reads 0.975 as "the reader learned the forms".

## R54 (planted sets, BERT-tiny on seeds 5-9, LLM reference; 2026-10-25)

All three accepted as written. The planted row splits the two query kinds between the systems, and the paper will say so: the reader
reports the unsupported identity in 0.98 of the sentences (criterion met; the MRC rows 0.66-0.84 and they answer some question with it
in 16-34 %), but flags 'undecided' in only 0.51 of the cases where the MRC rows flag 0.99-1.00 (at the price of 0.29-0.43 needless
queries on the other roles). That 0.51 is a real weakness of the margin as the reader's only 'undecided' signal and stays in the table.

Two additions, both scoring or decoder rules, no training:
1. **Scoring, every model:** in the 'undecided' set, count whether the second identity was reported as a query of ANOTHER kind (left in
   no role: for the reader 'unplaced', for MRC the same untrained test as in 'unsupported'). Files opened: logs/results_planted_reader*
   _seed0.json (counts only; n_identity_tokens is 0 for 'undecided', so this is not scored today). The hallucination rate is right as
   defined (the planted role's value WAS invented); the new column says whether the perturbation was silent or merely misnamed, which
   matters for a system whose answer to a query is to ask.
2. **Decoder rule, declared now, claim on seeds 5-9 (R39):** an identity left in no role, adjacent (the token before or after) to the
   filler of a role whose kinds admit it, turns that role into 'undecided' with both as candidates and the role's margin set to the gap
   between them at the role's address; this is what a person does with "p3 p5 gives o1 to p2". Expected effect, written before the run:
   'undecided' planted queried rises towards the share of sentences where the second identity is left unplaced (to be read from
   addition 1); needless queries on the other roles unchanged; the 'unsupported' set unchanged except where the appended identity is
   adjacent to a filler of an admitting role (", p7 ." after the recipient: those become 'undecided' instead of 'unsupported', a kind
   change counted as still queried, reported separately). Runs with the R46 decoder; both provenance categories kept.

BERT-tiny on seeds 5-9 (X3 0.000, inner 0.943, UNSEEN formulations 0.986) confirms the design-set picture on fresh seeds; the LLM
reference (level-3 answered at 0.955 in context, 2.33 USD) is the ceiling row the claim is read against: the compositional question is
answerable from language by a large model in context, not by the 4-11 M extractive models, and by the reader as an address.

## R55 (what changed in the tool repository today, for this session's record; 2026-10-25 19:20)

Three changes in vss-demo (the tool repository; none of them touches this repository's rows unless declared as a row here):
1. **Explained-tokens residual** (a1f4564): `tool/sop/infer.py residual_explained`, config `residual: "explained"`; transcribed here as
   `common/decode.residuals_explained` (R42), measured on the design set and pending on seeds 5-9.
2. **'Undecided' by adjacency** (f46b8d4): `infer.py undecided_adjacent`, config `undecided_adjacent: true`; an identity left in no role
   next to the filler of a role whose kinds admit it makes that role an 'undecided' query with both candidates (`_undecided.<role>`);
   on in the demo's shipped worlds. This is the decoder rule of R54 item 2; transcribe it as declared there and run it on the planted
   sets (seeds 5-9 for the claim), with the scoring column of R54 item 1.
3. **Verb-order route** (434fa40): a fourth route in the bundle, `vorder` in `reader.codes` / `--codes`: every non-verb word is drawn a
   second time at a learned displacement keyed by (side of the nearest verb, that verb's kind, the word's kind, its ordinal among the
   words of its kind since that verb), independent of the cue chain; `experiment/v8_gates.V8.set_vorder` + `vorder_index` (torch),
   `Routebook.vorder_index` (numpy), `routebook.js`, export. It needs retraining, so it is NOT a row here before JP 1 (R47): it is being
   measured in the tool repository on its small reference world (CPU, same seed, with and without the route) and, if it helps, will be
   proposed as its own declared row with its own version, same seeds and splits, claim on seeds 5-9, after the stop.
Nothing in this response changes the R48 / R51 set.

## R56 (the verb-order route as a declared row, now, on the GPU; 2026-10-25 19:50)

The owner asks that the address-level fix run where it is fast. Declared row, R38 protocol: **reader 1a + verb-order route** ("vorder"),
retrained with the same seeds, splits and selection rule as 1a; design on seeds 0-4 (the inspected cells), claim on seeds 5-9.

Code to transcribe (vss-demo commit 434fa40, `experiment/v8_gates.py`; the diff is 41 lines, reproduced here so that nothing is guessed):
- `V8.set_vorder(verb_cats, nmax=4)`: `self.verb_mask[K]` = 1 for the verb categories; `self.vo` = Parameter of shape
  (2, K, K, nmax, D), uniform(-pi, pi) from `np.random.default_rng(int(self.rng.integers(1 << 30)))`; add `m.vo` to the optimizer's params.
- `V8.vorder_index(cats, mask)` -> (side, vcat, ordinal, valid), all B x T: `isv = verb_mask[cats] * mask > 0`; `left_excl[k]` = index of
  the nearest verb strictly left of k (cummax over where(isv, idx, -1), shifted right by one, -1 if none); `right_excl[k]` = nearest verb
  strictly right (flipped cummin, T+1 if none); `has_l`, `has_r`; `side = has_l`; `vpos = left_excl if has_l else right_excl`;
  `vcat = cats[vpos]`; with `cum` = inclusive cumulative one-hot counts (B x T x K): `before_k = cum[k, cats[k]] - 1`,
  `at_v = cum[vpos, cats[k]]`; `ordinal = (before_k - at_v) if has_l else before_k`, clamped to [0, nmax-1]; `valid = (has_l | has_r) & ~isv & mask`.
- In `states`, after the chain term: `terms = terms + E(self.vo[side, vcat, cats, ordinal]) * valid.unsqueeze(-1)`.
- Verb categories for the basic world: every verb kind (gives / sells / lends / sends / offers / shows / receives / buys / gets / borrows /
  accepts and their participles) plus the level openers says / claims. Nothing else changes: the same three routes, the same gates learned
  from the closed start, the same lr / epochs / batch as 1a.
Evaluate exactly as the R46 claim rows: per-role rule and one-to-one, with and without the R46 decoder, the explained residual, the per-cell
table (T4:C, T7:A, T10:A+C, T4:B+T11:B) with the three decision categories, so that the row says whether the ADDRESS now holds the filler
(decided by the address) rather than elimination. Expected, written before the run: T4:C and T7:A decided by the address on most items;
T10 improved; nothing else moves outside the seed std; if the in-world rows fall by more than one seed std, the route is rejected and
the row says so. Run it in the free slots now; the growth and adjacency rows keep their place in the queue.

## R57 (one implementation of the reader's decoder, not two; 2026-10-25 20:50)

The owner's rule: the rules of the reader live in ONE place, the tool repository (vss-demo, `tool/sop/infer.py`), and are reused, not
re-implemented per project. Today this repository transcribed three of them into `common/decode.py` (R42, R46, R54) from my text. That
is a second implementation and it can drift. Change, CPU only, no new training:

1. Add vss-demo as a pinned, read-only submodule (`third_party/vss-demo`, commit c386514 or later), beside the paper repository.
2. Give the basic world a legend in the tool's format (`data/basic_world/legend.json` + `kinds.tsv`: roles A / B / C with their kinds,
   the events (give / sell / lend / send / offer / show and receive / buy / get / borrow / accept with their lemma tables), the level
   openers says / claims, `predicate_role`, `occupancy_marker` as the reader's training used them). Q18 said no legend file exists here;
   this is the moment it should, and it is also what the paper shows as "the legend".
3. Export every saved reader (1a seeds 0-9, R14, R37, later vorder) to a tool model folder (`export_routebook` + catalog + config +
   legend; the tool's `Reader` loads it), and read the report rows through `tool/sop/infer.Reader` with the config flags
   `candidates: "sentence"`, `residual: "explained"`, `undecided_adjacent: true` (the demo's decoder) and with the defaults (the paper's).
4. EQUAL checks, written in RESULTS: the tool's default decoder reproduces the 1a per-role rows; the tool's demo decoder reproduces the
   R46 / R42 / R54 rows of `common/decode.py` to the third decimal. Where they differ, the tool is the reference and the transcription
   is corrected, with the difference recorded.
5. From then on, every reader row in this repository is labelled "decoder: tool @ <commit>" and the transcriptions are deleted or kept
   only as the EQUAL check. A reviewer then reads one decoder, versioned, shared with the demo, not a per-project copy.

If the tool's reader cannot load the saved V8 models without a change on the tool side, say what is missing in questions.md and I make
the change in vss-demo (the submodule stays read-only here). This is CPU work and does not move JP 1.

## R58 (R57 inverted: the reader lives HERE; the `sop/` directory; the verb-order route, variant b; 2026-10-25 20:55)

The owner's decision replaces R57: vss-demo is private, this repository is public, so the reader's Python implementation moves here
and the demo mirrors it (its JS reader, with the parity tests). Done, by me, in commit e872113: a new directory **`sop/`** = the tool
(`legend.py`, `infer.py`, `train.py`, `export_web.py`, `soplang.py`, `world.py`, `check.py`, `diff.py`, `cli.py`), a README with the
decoder configuration keys, `SOURCE` = the vss-demo commit it was taken from, and `v8_gates_vorder_ref.py` = the reference copy of
experiment/v8_gates.py with `set_vorder` / `vorder_index` in variant b. Ownership as for responses.md: I edit `sop/`, you read it and
never edit it; if something in it blocks you, questions.md.

What to do with it (CPU except item 3; nothing moves JP 1):
1. **Read the reader rows through `sop.infer.Reader`** instead of the transcriptions in `common/decode.py` (the plan of R57 items 2-5,
   now with `sop/` in place of the submodule): a legend for the basic world in the tool's format (`data/basic_world/legend.json`,
   `kinds.tsv`), `export_routebook` of every saved reader into a tool model folder, the EQUAL checks (tool default decoder = 1a's rows;
   tool with `candidates: "sentence"`, `residual: "explained"`, `undecided_adjacent: true` = the R46 / R42 / R54 rows of `common/decode.py`
   to the third decimal), and from then on every reader row labelled `decoder: sop @ <SOURCE commit>`. Differences: the tool is the
   reference; record them.
2. **Variant b of the verb-order route**, declared row, same protocol as R56 (seeds 0-4 design, 5-9 claim, the per-cell provenance
   table, the rejection rule unchanged). Variant a was rejected by its own rule and stays reported. Variant b: a word introduced by a
   role cue (the previous word's category in the given cue set {to, by, from}) is CUED: it is drawn by no order term and not counted in
   the ordinals; the order route addresses the bare fillers only, the cue chain keeps the cued ones. Expected, written before the run:
   T7:A still by the address; T10 no longer broken (p11 after 'from' is cued, p8 is the bare first P before the verb, as the subject of
   every receive template); T4:C unchanged (still by elimination: "P right after the verb" is bare P#1 after a give verb, never seen in
   training); seen-transition formulations back within one seed std of 1a. If T10 is still wrong, the route is rejected again and the
   two attempts are reported together. Code: `sop/v8_gates_vorder_ref.py` (`set_vorder(verb_cats, nmax, cue_cats)`; `vorder_index`
   computes `cued` from `cue_mask[cats[:, :-1]]` shifted right by one, excludes cued words from the one-hot counts and from `valid`);
   your `vss/v8_vorder.py` subclass takes `cue_cats = ['to', 'by', 'from']`.
3. The growth rows for BERT-tiny are still the last GPU item; then JP 1.

## R59 (Q29; 2026-10-25 21:15)

Item 2 is done: `sop/world.py Stars` now takes `catalog.json "star_table": {"rng_seed": 100 + seed, "order": ["role:A", "role:B",
"role:C", "op:says.speaker", "op:says.scope"]}` and returns those draws before the hashed star (checked against NestedRoleWorld's draw
order: R_A, R_B, R_C, R_X = says.speaker, R_rep = says.scope, all equal). `sop/SOURCE` updated.
Item 3, confirmed on the tool's side: a legend with `reader.predicate_role: false` and `reader.occupancy_marker: false` compiles
exactly R_A f_A + R_B f_B + R_C f_C (+ R_X f_speaker + R_rep S_inner per level), no predicate term, no marker; the compiled residual is
then the same quantity as `common/decode.residuals`. Items 1 and 4 as you say (identity_table; `filt` per call for the axis's candidate
lists). Item 1 of R58 is unblocked: write the basic-world legend and kinds.tsv, export the saved readers, run the EQUAL checks.

## R60 (variant b accepted; the route enters the reader; 2026-10-25 22:00)

Variant b is accepted as written: T7 by the address, T10 fixed, new formulations 0.989 / 0.956 per-role and 1.000 / 1.000 with the
R46 decoder on both seed ranges, every in-world row at or above 1a, explained residual at 1.8 % (design) / 2.1 % (claim) false
rejection: the demo criterion met on the design set and missed by 0.1 points on the claim set, reported as missed. Report a and b
together, as declared. Consequences:
- The verb-order route (variant b) becomes part of the reader: `reader.codes` of every demo world gains `vorder` (tool repository,
  retraining on the DGX per TODO_DGX); in this repository the reader rows "1a + vorder b + demo decoder" are the reader's current
  version, labelled with the sop commit, beside the pre-registered 1a rows which stay as measured.
- The growth table (R49) gains one line when cheap: the vorder-b reader on template 4 with k = 0 (the route may read the double object
  without examples; if it does not, the 50-example line stands) and, for the minimum, k = 5 / 10 / 20 for the reader only (CPU minutes).
- R58 item 1 (the EQUAL checks through sop/) continues as you have it; the vorder-b models are exported like the others once
  `vorder` is in the tool's Routebook (it is: sop/infer.py reads `vo`, `verb_mask`, `cue_mask`, `vo_nmax`).
Then the BERT-tiny growth rows and JP 1.

## R61 (growth rows and the cost table; 2026-10-25 22:35)

Accepted as written, including the parts that weaken earlier sentences, which the paper will carry as they are:
- **Cost.** At batch 1 on one CPU thread the reader is faster (8.2 ms vs 12.2 / 15.8 / 103.5 ms), but batched (32) BERT-tiny is faster
  than the reader (2.0 vs 6.4 ms) and on CUDA both are under a millisecond. The sentence "faster than BERT" is withdrawn; the sentence
  is "comparable to BERT-tiny at the reader's size, faster only in the single-sentence CPU setting a browser uses". Size: 13.3 vs 17.6 MB
  (fp32), a small difference at this size; the large ratios are against DistilBERT / BERT-base only.
- **Growth.** BERT-tiny learns the level-3 question from 500 examples in 3 s by continued fine-tuning with forgetting (worst drop 0.05
  mean, 0.31 on one seed) or in about a minute by replay without forgetting; 50 examples are not enough on every seed; the alternative
  learning rate mattered (0.17 / 0.92). The reader: 0 examples for the composed question; 50 examples and 2.3 min for the new form,
  0 choices, no drop. The paper's sentence is therefore not "BERT cannot learn it" but: a new question costs MRC annotated examples (50
  are not enough, 500 are), one or two hyperparameter choices that matter, and either forgetting or a full retraining; a composed
  question costs the reader nothing and a new form costs it a deterministic full retraining with no choices. Both regimes' numbers in
  the table, both seed ranges.
- The vorder-b line (k = 0 on template 4) and the reader's k = 5 / 10 / 20 remain the last CPU items (R60); if they are not done when
  JP 1 is written, JP 1 says so and they follow.
This completes the declared set of R48 / R51 except R58 item 1's EQUAL checks (in progress) and the two lines above. Write JP 1.

## R62 (JP 1 read; one correction to make and one count to add; 2026-10-25 22:50)

JP 1 is accepted as the synthesis: the two tables, the confirmatory reading (including that (i) and (ii) are decided by the untrained
question and that BERT-mini pretrained ties the reader on (iii) with the pre-specified score), the regime statements and the
limitations are what the paper will carry. Two things:
1. **The route's parameter count.** "The verb-order route enlarges the reader to 11.7 M parameters" counts the full table vo of shape
   (2, K, K, nmax, D) with K = every category, most of whose entries are never indexed (a verb category as the word's category, a
   non-verb category as the verb's) and never receive a gradient. Report two counts beside each other: the allocated table (11.7 M, as
   now) and the **used keys** (the (side, verb kind, word kind, ordinal) tuples that occur in the training sentences, times D; count them
   from the training set), which is the model an export would ship. The tool's export will store only the used keys (a change on my side,
   sop/); until then the paper states both and compares size at the used count. If the used count is still above BERT-tiny's 4.4 M, the
   size comparison is stated at that count, without softening.
2. **Decoder provenance**: as JP 1 says, the EQUAL checks through sop/ get their own section when done; no reader row is relabelled
   before they pass.
Nothing else. The experiment is at its stop; what follows is the paper, the demo and the tool.

## R63 (housekeeping; 2026-10-25 22:50)

Q29 was answered by R59 (star_table in sop/, 20:59) and you built on it (legend, sop_check.py, 21:54-21:56); its text is still in
questions.md under a header that says no question is open. Remove it, as for the others, so that the file shows the real state.

## R64 (Q30; 2026-10-25 23:30)

The tool was wrong, not the transcription: R42's definition includes the clause, and it is part of the contract (a frame that does not
answer a question the legend always asks has not explained the sentence as a frame). Fixed in the tool and synced here (sop/SOURCE
0b4878b): `infer.Reader.frame_incomplete(f)` is true when a core question of the schema or a speaker of the frame's depth is
unanswered, and the explained residual is then 1.0; an optional role left "not given" does not count. Python and the browser reader
changed together (parity PASS). Re-run the residual part of the EQUAL check with the synced sop/; the expectation is equality with
common/decode.py (reference-world rejection 0.88-0.95 with the sentence-candidates decoder). If it is equal, the paper uses those
numbers labelled "decoder: sop @ 0b4878b"; if not, say where it differs. The rows computed with the clause-less tool are kept and
labelled as superseded by this fix.

## R65 (the one-to-one scope; 2026-10-26 06:40)

Read and accepted: the accuracy rows are equal through the tool, and the remaining difference on the out-of-world sets is the scope of
the one-to-one rule. The scope is decided, and the tool was wrong again: the rule as stated in R46 and in the paper is "a word answers at
most one question", over every question of the frame, speakers included; the tool's implementation covered the inner roles only, so a
speaker could also be read as the giver ("p3 claims that p1 gives o0 ." -> A = p3), which completes a wrong frame and hides it from the
residual. Fixed and synced (sop/SOURCE 6c2c6ee): `one_to_one` now runs over the inner roles and the speakers of every level, and
bounds each word by its number of occurrences in the sentence (a word written twice may answer two questions; the transcription
forbids any repetition, which is the same on this world's sentences, where a repeated identity does not occur; say if it does). A
speaker left without a candidate makes the frame incomplete, so the explained residual is 1.0 (R64). Python and browser together,
parity PASS.

Please re-run the residual part once more with sop @ 6c2c6ee. Expectation: equal to the transcription (AUROC 0.965-0.985, reference
rejected 0.88-0.95, random 0.99, false rejection 0.015-0.033). If equal, the paper's refusal numbers are those, labelled with this sop
commit, and the rows through 0b4878b are superseded; if not, report the first sentence where they differ.

## R66 (Q31; 2026-10-26 06:55)

Your diagnosis is right and the fix is as you propose. sop @ f1af943: each question's lost words accumulate across rounds (the
exclusion set is no longer rebuilt from the original candidates), and a question still colliding after the last round is left
unanswered, so the frame is incomplete and the explained residual is 1.0. Python and browser together, parity PASS. On repeated words:
keep the occurrence bound (a word written twice may answer two questions); it differs from the transcription only on the random
out-of-world sets (34% of their sentences repeat a word), and the paper states it. Re-run the residual part once with this commit;
expectation: equal to the transcription on the world's sets; on the random sets, report both numbers if they differ.

## R68 (two last rows before submission, owner's budget: about one hour of wall-clock in total; 2026-10-26 07:55)

The paper is at its stop; two cheap rows close the two objections a reviewer will raise first. Both declared now, before they run.
Priority order; if the hour is not enough, finish (A) and run (B) with fewer seeds, and say so.

**(A) The depth-invariant tagger (the objection: "MRC cannot answer an unseen question string by construction").** BERT-tiny pretrained,
the tagger form of 2a (models/bert_basic.py), with ONE speaker label: X1 / X2 / X3 collapse to SPK in training; at decoding the level of
each SPK token is its order of appearance (the first SPK is level 1), so level 3 is reachable without an untrained label. Fixed
configuration (lr 1e-4, 5 epochs, batch 32), seeds 5-9, depth NOT given (the level count is the number of SPK tokens). Report: X3 slot
accuracy, six-question exact match at depth 3, inner frame, confident error at depth 3, and the same on new formulations. Written now:
if X3 >= 0.9 the paper states that a dedicated output head answers this particular composed question, and the claim stays the general
one (any question composed of the legend's keys, with no head built for it); if X3 < 0.5, the example also stands against this head.
A small code change (`--speaker_label single`) is allowed for this; default off.

**(B) One real dataset (the objection: "a constructed, rule-solvable world").** SNIPS, the splits already frozen
(data/real/splits_snips.json), the runs already written in RUNBOOK 8b / 8c, reduced to the budget:
1. `./run.sh vss/run_reader_real.py --dataset snips --sizes full --seeds 3 --test --save_model`
2. `./run.sh models/bert_real.py --dataset snips --model google/bert_uncased_L-2_H-128_A-2 --sizes full --seeds 3 --test --save_model`
   (BERT-tiny, the equal-size encoder; its published lr / epochs for small BERTs, lr 1e-4, 10 epochs, no grid, stated)
3. the same two with `--condition ood_intents --test` (the held-out intents as out-of-domain), 3 seeds.
Report with the standard metrics as primary (intent accuracy, slot F1 conlleval-style, sentence exact match) and the head-token frame
as secondary; for ood_intents: AUROC in-domain vs held-out intents of the reader's residual and margin against the tagger's MSP (and
Mahalanobis if the saved model allows it), and the rejection at 2% false rejection on the dev calibration half. Written now: on SNIPS
a pretrained encoder is expected to be above the reader on intent and slot F1 (the reader has no pretraining and reads heads); the
rows report the gap and the cost as they are; the ood_intents row is the q-formalisation row on real data.

Commit each result as it comes (so the paper can use what is done when the hour ends), then the usual RESULTS / STATUS lines. No other
row starts.

## R69 (Q32; 2026-10-26 08:05)

Both points decided, both in the tool (sop/SOURCE updated to the vss-demo commit that does them; Python and browser, parity PASS):
1. **Numerical tolerance, declared now:** an explained residual below 1e-4 is 0 (`residual_explained`). A sentence whose every token is
   placed has residual 0 by construction; float noise is not evidence, and one unexplained token out of 16 is about 0.26, so 1e-4 is
   four orders below any real signal. With it the calibration threshold stays 0 and "false rejection" counts the in-world sentences the
   frame does not fully explain, the same for float32 and float64.
2. **Verbs:** only the event's verb is explained. Without a predicate role the tool now places the FIRST token whose lemma names the
   decoded event, exactly R44's adaptation; a second verb is content the frame does not explain. Drop `--verbs_explained` from
   eval/sop_check.py (the verbs return to being content kinds of the basic-world legend).
Expected: the tool equal to the transcription on the random sets too, apart from the occurrence bound (about 18-20 sentences per seed).
Re-run the residual part once more with this sop, AFTER the R68 rows (they have priority within the hour), and report the paper's refusal
numbers from it. The criterion verdict for route b is whatever these numbers give, stated once.

## R70 (R69 re-check accepted; the ['word', None] note; 2026-10-26 08:40)

Accepted as written, including that the verdict rests on a threshold of 0 and that the nonce source was not re-scored through the tool.
The note on a core role returned as ['word', None] is a tool defect and is fixed (sop/SOURCE 7e10be1: frame_incomplete counts it as
unanswered, Python and browser, parity PASS); as you observed both versions reject those sentences, so no number changes and no re-run is
needed. If the SNIPS reader rows (R68 B) finish within the hour, commit them; otherwise commit what exists and stop. After that, nothing
else runs: thank you, the experiment is closed from my side.

## R71 (reopening one row: the kinds on real data were the handicap; 2026-10-26 09:05)

Diagnosis of R68 (B), from the author's side: the reader takes a word's KIND as a given input (rule 20); on the constructed world the
lexicon gives it, on SNIPS the protocol of INSTRUCTIONS_REAL replaced it by unsupervised skip-gram clusters, so most slot values (names,
playlists, places) reached the reader with a kind unrelated to their role. That was a handicap of the protocol, not a property of the
reader: BERT-tiny was fine-tuned on the same 13,084 annotated utterances, the reader was denied the annotation's types.

New component, in sop/ (sop/SOURCE bb6db1b): `sop/typer.py`, the kinds learned FROM THE TRAINING ANNOTATION (no evaluation label is
used): per token, a small context classifier (hashed character n-grams, shape, the two words on each side; multinomial logistic
regression, 5.4 M weights) and a Viterbi decoding over kind transitions; frequent words stay their own kind; a slot kind is S:<type>;
`kind_run` gives the run of a slot kind around a position (the span). Measured here on CPU with your scorer (eval/metrics.span_f1) on
the frozen SNIPS test, the typer ALONE (spans = runs of S: kinds, no reader): span F1 **0.881** (test), 0.853 (new slot values), 0.871
(unseen templates); BERT-tiny's rows in RESULTS: 0.831 / 0.783 / 0.805. Fit 28 s on CPU.

Declared row, now, before it runs (seeds 0-2, the same frozen splits and scorer as R68 (B)):
1. **typer only** (the ablation above, recomputed by you): intent not given by it; span F1 per set.
2. **typer + reader**: the kinds.tsv of the world written from `Typer.lexicon` over every utterance's text (train + dev + test, text
   only, as build_kinds already does for the vocabulary), the reader trained as in R68 (B) (D 1024, 10 epochs, path + chainL + bigram;
   add `vorder` only if it runs without a change), and the spans of each decoded slot = `kind_run` over the typer's in-context kinds
   of that utterance around the decoded head (replacing spans_by_rule; keep spans_by_rule as a second line). Report intent accuracy, span
   F1, head-frame exact, the new-slot-values and unseen-template sets, cost (reader + typer weights), and on ood_intents the AUROC of the
   reader's residual and intent margin against BERT-tiny's MSP / margin.
3. Written before the run: if (2)'s span F1 is >= (1)'s, the reader adds structure without losing the typer's spans; if below, the
   reader's role readout loses information the kinds carried and the paper reports the typer + reader with the typer's spans. Either
   way the comparison with BERT-tiny is stated as "typer + reader (two small components trained on the same annotation) vs one
   fine-tuned encoder", with both sizes.
Budget: the same order as R68 (B), under an hour. A small change to vss/run_reader_real.py (`--kinds typer`, default unchanged) is yours.

## R72 (addition to R71: one world per intent, as the architecture is meant to be used; 2026-10-26 09:10)

The owner points out a second mismatch: VSS is designed as several small worlds, one per purpose, each with its own legend, and a
dispatcher that picks the clearest reading (status, margin, residual). R68 (B) read SNIPS as ONE world whose legend has 39 slot
questions and 90 kinds (the quadratic pair tables are why the reader reached 24.5 M phases). Add, after R71 item 2 and in the same
budget if it fits, the architecture's own configuration:
4. **One world per SNIPS intent** (7 worlds): each legend has only that intent's slot types as questions (about 4-7), its kinds from the
   same typer (restricted to that world's slot kinds; other slot kinds map to O), its reader trained on that intent's training
   utterances only (D 512 is enough at this size; state it). Reading: all 7 readers read every utterance; the dispatcher keeps the
   reading with the best status, then the smallest explained residual, then the largest minimum margin (sop/infer.py by-fit order).
   The intent is the chosen world (no closed-value intent role); spans as in item 2.
   Report: intent accuracy (= dispatch accuracy), span F1, head-frame exact, the two harder sets, the summed size of the 7 readers +
   the typer, ms per utterance for all 7; on ood_intents (the held-out intents have NO world): the share of held-out-intent utterances
   that every world rejects (residual above its calibration threshold, or no status ok), the AUROC of the best world's residual for
   in-domain vs held-out, and the in-domain false rejection. That last row is the q-formalisation claim on real data: a request of an
   unknown kind is refused because no world explains it.
Written before: the per-world readers are expected to be smaller and their slot readout less crowded than the single world's; the
dispatch may lose a little intent accuracy against the closed-value intent role. Whatever the numbers, both configurations (one world,
7 worlds) are reported side by side. If the budget runs out, item 4 has priority over item 2's vorder variant.

## R73 (drop the verb-order route from the SNIPS rows; 2026-10-26 09:15)

Do not add `vorder` to any SNIPS row (R71 item 2 said "only if it runs without a change"; now: not at all). It addresses roles that
appear without their usual cue word (clefts, the double object); on SNIPS the slots are separated by type, not by position around a
verb, so it would add size and time for no expected gain. The three routes only, in R71 and R72.

## R74 (tool notice: the rules of a world are activated from its nature, not by hand; sop/activate.py; 2026-10-26 09:40)

No new row and no re-run. A change of the shared tool, synced to `sop/` at this commit, which the SNIPS per-intent worlds (R72) can use
at zero cost:
1. **Decoder default.** `sop train` now writes the contract rules into `config.json` itself (`--decoder contract`, the default:
   `candidates: sentence`, one-to-one, `undecided_adjacent`, `residual: explained`), which is exactly the "sop decoder" of R46 / R42 /
   R54 that the reopened rows already use; `--decoder paper` writes none of them (the frozen rows' decoder). Nothing already measured
   changes; a model trained from now on carries its decoder in its config, so a reader no longer depends on who sets the flags.
2. **Routes by need and acceptance.** `sop train --activate auto` selects the structural routes (`vorder`, `pos`) per world: a need signal
   on the training annotation (same-kind competition for `vorder`: two questions answered by words of one kind; rigid order for `pos`),
   then trials trained without the selection rows (the calibration halves of the held-out formulations + a hashed tenth of the training
   rows) and the acceptance rule applied by hand on 2026-10-25 (no selection split drops by more than eps = 0.01; the marked rows rise).
   Trials and decision go to `config.json` (`activation`) and the run folder. `sop profile <world>` prints the need signals without any
   training. The argument (two levels: worlds activated at inference by the dispatcher's vote on the residual; rules and routes activated
   at training by the world's nature; contract rules / structural routes / diagnostics) is in the demo repository's
   vision/ACTIVATION_rules_and_worlds_2026-10-26.md and summarised in `sop/README.md`.
3. **For SNIPS (R72), one cheap line per world:** run `sop profile <world>` on each of the 7 per-intent worlds and put the 7 need-signal
   lines in the report (same-kind competition share, order-dependent share, unknown-kind share). Written before: `vorder` is not
   expected to fire on any of them (the slots of one intent differ by type), which is R73 measured instead of asserted. If it fires
   somewhere, do not act on it in this round; report it.
Checked before this notice on the three worlds of the demo repository (`sop profile`, no training): reference world 78.3% same-kind
competition (0% without a cue), commands world 0%, quantifier world 100%; the first `--activate auto` run (reference world, seed 0, CPU)
is cited in the sync commit's SOURCE.

## R75 (R71 accepted; R72 per-intent worlds: the configuration I asked for in R72 broke the dispatcher, fix and re-run; 2026-10-26 10:20)

**R71 accepted as measured.** Reading: on SNIPS the kind of a word determines its question almost always (one slot type = one question),
so on spans the reader can only lose against the typer (0.798 vs 0.886), and it did; what the reader adds there is the intent, the
margins, the refusal and the residual, not the spans. The paper states it that way (typer's spans; reader's intent 0.977 vs 0.985; the
residual separating held-out intents at 0.933-0.943 in the one-world configuration).

**R72 as run cannot test the dispatcher, and the fault is in my wording.** R72 item 4 said "its kinds from the same typer (restricted to
that world's slot kinds; other slot kinds map to O)", and vss/run_reader_real.py line 330 does exactly that. The effect: in every world the
slot words of the other six intents become O, O is not a content kind, so the explained residual never counts them, and a wrong world
explains an utterance as well as the right one. That is what the numbers say: residual AUROC held-out vs in-domain 0.555 (chance), zero
rejections by the residual alone, dispatch intent 0.47-0.70 against 0.977 in one world, "rejected by every world" 20-27% in-domain
through status alone. The level-W activation (every world reads, the vote is the residual) needs each world to be charged for the words
it has no question for; mapping them to O removes the charge. Fix, declared before the re-run:
1. **Keep every typer kind in every world's lexicon** (all 74 S: kinds; drop the restriction at line 330). A world's legend still has
   only its own slot questions; a word of another intent's kind has no question in that world, is not placed, and raises that world's
   explained residual (content_kinds_of counts a kind with 2+ lexicon words as content, so these kinds count).
2. **Dispatch by the residual, status second.** Order the worlds by explained residual (ties by the largest minimum margin); a world's
   frame status (`abstain` = a slot margin under tau) is a property of the answer inside the chosen world, not a reason to prefer
   another world. Report intent accuracy on every utterance (the chosen world), and separately the share whose chosen frame abstains.
3. **Rejection = every world's residual above its own calibration threshold** (the 98th percentile as now, on that world's own
   training utterances read with the full lexicon). Keep "every status not ok" as a second line only.
4. Same seeds, D 512, 10 epochs, three routes, as before. Both tables (one world R71, per-intent re-run) side by side; the per-intent
   numbers of this round stay in RESULTS marked "configuration error (R72 wording), superseded by R75".
Written before: with item 1 the residual AUROC should move well above 0.9 (the one-world residual already gives 0.933-0.943 and here
each wrong world has whole slot phrases to leave unexplained); dispatch intent should approach the one-world 0.977, and the rejection of
held-out intents should rise far above 0.49 while in-domain false rejection falls under 0.05. If item 1 does not move the AUROC, the
level-W vote by residual is refuted on this data and the paper says so.

**R74 profile lines accepted.** Three of seven worlds show same-kind competition (4-5%, all uncued: `vorder`'s need signal fires there,
e.g. two artist or two playlist phrases in one utterance). Not acted on this round (R73), recorded as the signal that an order route
would be tested there by `--activate auto`.

## R76 (GPU priority to the demo session; 2026-10-26 10:45)

The owner's decision: from now the vss-demo session has the GPU. Let the R75 re-run that is already running finish; do not launch any
new GPU run (seeds, re-runs, extra rows) until the demo session's runs are done (its STATUS.md says when; check it before every
launch). CPU work (reports, RESULTS, `sop profile`, scoring from saved JSONs) continues as usual. If the R75 re-run is not yet started,
start it only with the memory guard at 0.15 or less and a single process, or wait.

## R77 (replaces R76: parallel runs across both sessions are fine; 2026-10-26 11:00)

The owner's decision: the machine (20 cores, 128 GB, one large GPU) is shared by both sessions and our runs are small, so run in
parallel whenever reasonable; R76's "no new GPU launches" is withdrawn. Keep the guards as they are (memory fraction per process, the
minimum available memory, the lock per world or per run slot), keep `nice +10` for long runs, and do not start more concurrent GPU
processes than the guard admits. Timing numbers from concurrent runs are not cost numbers (as already stated in R71).

## R78 (after R75: the reader on SNIPS is to be improved until its numbers are good; the seven worlds will be exported for a demo; 2026-10-26 11:25)

The owner's decision: SNIPS is not a row to close but a reader to improve, and the seven-world configuration is the one to show. After
the R75 re-run, in this order, each as a declared row with expectations written before: (1) if the dispatcher works (R75's expectations
met), the per-intent readout: span F1 of typer + reader per world against the typer alone on the same utterances; where the reader
loses, the first suspect is the role readout with few questions (one slot type -> one question): report per world which questions lose,
and whether `--activate auto` (sop/activate.py; `vorder` fires on three worlds, R74) changes it; (2) the held-out-intent refusal with the
fixed configuration, as the q-formalisation row on real data. When (1) is stable, the demo repository needs the seven worlds and the
typer exported (`sop export-web` per world, plus the typer; the export of the typer's table is mine in sop/ and I will say here when it
is in): keep the per-world model folders of the best seed in the run folder, not deleted, so that the export can be made from them.
## R79 (R75 accepted as measured; the cause is the missing structure anchor, from my R72 wording again; one re-run; 2026-10-26 11:50)

R75's numbers are accepted and stay in RESULTS as the measured failure of that configuration. The cause is visible in your own
diagnosis (four thresholds at 1.0 from incomplete frames on the worlds' OWN training utterances) and in sop/train.py: a world's core
roles are the questions present in >= 95% of its training rows, and when there is none the tool silently took the single MOST FREQUENT
role as the core. The per-intent worlds have no slot present in 95% of their utterances (every slot is optional), so each world's core was
an optional slot; whenever it was absent, the frame was 'incomplete' (residual 1.0, even in the right world), the structure score (the
anchor every presence ratio is divided by) collapsed, the other slots abstained (chosen frame abstains 0.64-0.78), and the 98th
percentile threshold hit 1.0. The one-world configuration never had this problem because the closed-value `intent` role is present in
every utterance (rate 1.0): it WAS the anchor. R72 item 4 said "no closed-value intent role" in the per-intent worlds, which removed it.

Fix, two parts. (a) In sop @ 7ce5c50 (synced here, SOURCE updated): no silent fallback; core = only the >= 95% questions; a world with none
and no predicate gets a loud warning at training and a mean-over-all-roles structure score, and no frame is incomplete by a role.
(b) In the per-intent worlds: keep the closed-value `intent` role with its single value in every world's legend and every example
(rate 1.0 -> the core question and the anchor; it reads by cleanup over one symbol, so it costs nothing and cannot be wrong), exactly
as the one-world configuration has it. Then re-run R75's configuration once (same seeds, D 512, 10 epochs, three routes, --world_kinds
all, --dispatch residual), both conditions. Written before: the warning must NOT appear for any world (the intent role is core);
thresholds below 1.0 in every world; chosen-frame abstention far below 0.6; dispatch intent well above 0.595 (the one-world 0.977 is
the reference); residual AUROC above R75's 0.775; held-out rejection above 0, in-domain false rejection under 0.05. If dispatch intent
stays under 0.9 with the anchor in place, the per-intent vote is reported as below the one-world reader on this data and R78 (1) is
run on the one-world configuration instead. Keep R75's per-world model folders until the re-run's are written.
(The section was written at 11:50 but its commit failed on a push conflict; only the addendum below reached the repository. Sorry.)
On Q33's directions (read after R79 was written): direction 1 is R79 (a) and (b) together, with one addition that matters: making
every slot optional is not enough, because the structure score (the anchor of every presence ratio) then has nothing to stand on; the
anchor is the always-present `intent` role. Run the cheap diagnostic you propose first (share of 1.0 residuals per world and their
source, from the saved models), as the record of the cause. Direction 2: not needed once the clause has a real core question; keep
the second line "graded residual only" in the report anyway. Direction 3 (hybrid): the fallback named at the end of R79, only if the
re-run's dispatch stays under 0.9.

## R80 (answers Q34; the dispatcher: calibrate, and use the anchor's own fit; CPU first; 2026-10-26 12:55)

Option 1 first, as you recommend, with one addition that I think is the actual signal. Why the raw residual cannot carry the dispatch
on SNIPS even when calibrated: the explained residual charges only content kinds (the S: slot kinds); the words that tell the intents
apart ("play", "book", "rate", "add", "find", "is it rainy") are frequent words with their own `w:` kinds or O, never charged, and many
slot types are shared between intents (timeRange, city, object_name). An utterance whose slots are all of shared types has residual 0
in several worlds, and the tie is decided by a margin. The one-world reader reaches 0.977 because its `intent` role reads exactly those
words through trained addresses. The per-intent worlds now have that same role (R79's anchor): in each world the `intent` role's
cleanup score is the fit of the utterance to that world's trained addresses for its single intent symbol, and in a foreign world the
discriminating words have untrained (random) addresses, so the score should be lower. That is a trained, discriminative signal, and it
is already in the saved R79 models.
Declared row (CPU, saved R79 models of seeds 0-2, both conditions, no retraining): per world and utterance, three quantities, each
converted to its percentile in that world's own calibration distribution (200 of its training utterances, as the thresholds): (a) the
explained residual; (b) the `intent` role's cleanup score (the structure score; state both if they differ); (c) the min slot margin.
Dispatch rules reported side by side: residual raw (= R79), residual percentile, anchor percentile (b), anchor then residual
(lexicographic), and the sum of the two percentile ranks. Rejection unchanged (every world's residual above its own threshold), plus a
second line "every world's anchor percentile under 2%". Report dispatch intent, span F1, AUROC held-out vs in-domain for each rule, same
table as R79. Written before: the anchor rule should move dispatch intent well above 0.716; if the anchor alone reaches 0.9 the
per-intent configuration is kept with that rule for the demo and R78 (1) runs on it; if it does not, option 2 (one-vs-rest negatives
in training) is declared next, and option 3 stays the fallback. R79's own three seeds finish and are reported as declared regardless.

## R81 (answers Q35; R80 accepted; option 2 declared: one-vs-rest negatives in training, through the anchor; 2026-10-26 13:20)

R80 is accepted as measured: the anchor's own fit carries the dispatch to 0.863 and the rejection to 0.28 at 3.8% false, and the
calibrated residual is worse than the raw one, which settles that the problem is discrimination, not scale. Option 2, as a declared
row, needs no change of the reader and no new loss: sop @ 82585ea (synced; SOURCE updated) accepts NEGATIVE examples: a wire with
`_negative=true`, the sentence of another world written with the anchor role only (`intent=I`), whose training target is MINUS its
compiled state (the anchor term with the marker), so the cosine objective pushes the state of a foreign utterance away from this
world's anchor; negatives never enter the schema, the calibration rows or the train sample, and the reader at inference is unchanged
(smoke-tested on the commands world, CPU: a negative row is counted and trained, the schema and the calibration ignore it).
Row: R79's configuration (seeds 0-2, D 512, 10 epochs, three routes, --world_kinds all, --intent_anchor), plus, in every per-intent
world, negatives = the other intents' training utterances of that condition, balanced (as many as the world's positives, sampled with
the seed, split "train"), written as above. Read with R80's rules (eval/r80_dispatch.py on the new models): the anchor-percentile
dispatch is the primary, residual raw as the check; both rejection lines. Model folders *_r81. Written before: anchor dispatch at or
above 0.9 (the one-world 0.977 is the reference; if it reaches 0.95 the configuration is kept for the demo and R78 (1) runs on it);
held-out rejection by the anchor line above R80's 0.28 at false rejection under 0.05; in-domain span F1 not below R80's 0.802. If the
anchor dispatch stays under 0.9 with negatives, option 3 (the one-world intent as the gate, the per-world frame, refusal by the chosen
world) is run as the last per-intent row and the paper reports the per-intent vote as below the one-world reader. Also report the
training time per world with and without negatives (they double the rows).

## R82 (tool notice: three contract-decoder rules added at the source from the demo world's diagnosis; no re-run; 2026-10-26 14:55)

sop @ b467e4b (SOURCE updated). From the diagram world's error analysis in the demo repository (vss-demo Q8): (1) `presence_query` (in the
contract decoder): an ABSENT role whose presence margin is under tau is reported as `_presence_query.<role>` and no longer lowers the
frame's margin or status (74% of the diagram world's abstentions were absent optional roles near their threshold, with the frame
right); the paper decoder is unchanged. (2) `candidates: sentence` now also restricts the predicate's candidates to the sentence's verbs.
(3) At training, content kinds = the kinds that fill a question in the training rows plus `reader.content_kinds` of the legend; a kind
that never answers a question (prefix and function words with their own kinds) is not charged by the explained residual. Models already
trained keep their stored schema (3 does not touch them); 1 and 2 change the reading of existing models only through config.json keys,
so no row of this repository changes unless re-scored with `presence_query` set. Not to be applied to the frozen rows. For the SNIPS
per-intent worlds after R81: (1) applies (optional slots near threshold were part of the 0.61 abstention) and (3) may matter (the `w:`
kinds are single-word kinds and were already not content, so expect little change); declare before using them.

## R83 (standing rule from the owner, for every training row from now on; 2026-10-26 15:20)

Before any row that trains a reader on more data of the same kind (more seeds of the same configuration excepted), the row's
declaration states what fails (worst formulations, activation diagnostics, unplaced words, presence queries, per-world losses), where
the cause is (the reader: a contract rule or a route by its acceptance test; the world: a question conflated, a kind missing, an
anchor missing; the examples: a form missing, then that form is added, not everything), what is changed, and which number is expected
to move. The reader learns addresses from kinds and positions, so more instances of the same templates memorise and do not generalise;
a size curve is a check, never the remedy. R81 is already declared this way (the cause: discrimination, the change: negatives through
the anchor); every later row follows the same form, and I will refuse a row without that paragraph.

## R84 (R81 accepted: the per-intent configuration is kept; what follows; 2026-10-26 15:35)

R81 met its main expectation: with one-vs-rest negatives through the anchor, the anchor-percentile dispatch reaches 0.966 ± 0.003
(one-world 0.977), span F1 0.824, residual AUROC of the chosen world 0.941 (above the one-world 0.933), and the raw residual became a
worse dispatcher (0.386), which is what the mechanism predicts: negatives push foreign states away from the anchor, not from the slot
readout. The configuration is kept for the demo (D9 in the demo repository). The rejection expectation was not met at the fixed 2%
anchor line (0.120 at 0.030 false), while the threshold-free separation rose (anchor AUROC 0.622 -> 0.892): the operating point moved,
not the separation. Rows, in order, each with the R83 paragraph (what fails, cause, change, expected number):
1. **Rejection operating points (CPU, saved R81 models, no training):** for each signal (the chosen world's residual; the anchor
   percentile; and their max over worlds), the held-out rejection at in-domain false rejection 1%, 2% and 5%, the thresholds set on
   the calibration half of dev (never on test), reported on test. Expected: at 5% false rejection the anchor signal rejects well above
   0.28 (the AUROC says so); whichever signal is better at 2% becomes the demo's refusal line and is stated as such.
2. **R78 (1) on the per-intent configuration:** per world, span F1 of typer + reader against the typer alone on the same utterances;
   which questions lose; then `--activate auto` per world (the acceptance test; the three worlds where `vorder`'s need signal fires).
   Expected: the loss is concentrated in the roles with same-kind competition; `vorder` accepted on at least one of the three worlds.
3. **Re-score with the current contract decoder (R82: presence_query, sentence verbs) as a declared re-score row**, same models, same
   rules (eval/r80_dispatch.py --sop_dir = SOURCE b467e4b): expected to lower the chosen-frame abstention (optional slots near their
   threshold become presence queries) without changing dispatch or spans; report both lines.
4. **Export for the demo** (after 1-3): the seven base worlds of the best seed (by anchor dispatch on dev, stated) and the typer, through
   `sop export-web` per world into a folder the demo repository can copy (plus the typer's table; its export is mine, I will say in
   this file when sop has it); keep the folders. The demo's page then reads with the rule and the refusal line of item 1.
Training-time note accepted as written (the 5.3x is load, not cost; not to be cited as cost).

## R85 (answers Q37: the refusal line, and the activation test narrowed to where its need signal fires; 2026-10-26 16:25)

Item 1 accepted as measured. The demo's refusal line is the chosen world's residual above its dev-calibrated 2% threshold (held-out
0.183 at 0.008 false on test; 0.467 at 5%); the "every world's residual percentile" line is not taken (0.031 false, above the target).
The anchor's floor is understood and recorded: a percentile cannot order below the calibration minimum, so the anchor dispatches but
does not refuse; the papers say so with the numbers (Papers 6 and 8 of the series). Item 3 runs as declared.
Item 2, the activation part. Cause in the world, as you read it: the per-intent legends have no verb of the utterances, so the
verb-order route has nothing to stand on. Fix in the world: declare each intent's command verbs as the lemmas of its `command`
event in the legend (add / put, play / listen, book / reserve, rate / give, find / show / search, and for GetWeather the words that
open its questions, e.g. "is", "what", "will"), `predicate_role` stays false (the event is not read from a verb; the status rule does
not demand one), so nothing in the readout changes, only the route's verb categories. Then the acceptance test, but only where the need
signal fires with the typer's kinds: AddToPlaylist (2.4% of rows), seed 0 first, three seeds only if seed 0 accepts the route. R83
paragraph: what fails = the same-kind slot pairs (two artists, two playlists) in that world; cause = no address separates them (no cue,
no verb category); change = the verbs declared, `vorder` by the acceptance test; expected to move = span F1 on the competing rows of
AddToPlaylist, at most 2.4% of its rows, so the world's span F1 by at most about 0.02; the other six worlds are not retrained. If seed 0
rejects the route, the activation test on real data is reported as "need signal fired on one world, route rejected", which is a valid
outcome of the method, not a failure of it.

## R86 (R84 item 2 first part accepted; the loss is in presence decisions: cause analysis first, then one decoder rule, no training; 2026-10-26 16:50)

Accepted as measured, and the expectation written in R84 (competition) was wrong: the reader loses to the typer on PRESENCE (215 false
fills, 164 misses per seed), most where there are most questions (BookRestaurant, 14), and dispatch itself costs almost nothing. R83
paragraph before anything runs. What fails: a question filled when the gold has none, or left empty when it has one, in worlds with
many optional questions. Two candidate causes, to be separated on the saved models (CPU):
(a) **Admitted kinds overlap.** A role's admitted kinds are the typer kinds of its training fillers; the typer's lexicon maps a word
    to its majority kind, so a role like `city` may admit S:state or S:country and vice versa. Report, for BookRestaurant and
    SearchScreeningEvent, the admitted kinds per question (schema roles[r].kinds / head_kinds) and, for every false fill, which gold
    question the filled word belongs to (or none) and which kind the typer gave it. If most false fills are words of a sibling
    question's kind, the competition is real but invisible to the profile (different kind names, overlapping admission).
(b) **Presence ratios at D 512 with 14 questions.** From the calibration record of those worlds (config.json thresholds: mean_present,
    mean_absent, threshold per role), the separation per role; if several roles have mean_present and mean_absent within 0.1, it is
    interference at D 512 (the geometric limit of Paper 5), and a single trial at D 1024 for BookRestaurant alone is the test.
Then, whichever (a) or (b) says, one decoder rule in the source (mine), declared here before it is used: **presence by kinds where a
kind names one question.** Under the contract the kinds are given (rule 20); when a kind is admitted by exactly one question of the
legend, the presence of that question is the presence of a word of that kind in the sentence (the typer's decision), the reader
chooses the word when there are several, and the presence threshold is used only for questions whose kinds are shared. Expected: on
questions with a unique kind, false fills and misses fall to the typer's own error rate, so the per-world span F1 rises to about the
typer's where kinds are unique and the gap stays where they are shared; dispatch, margins and refusal unchanged. Re-score only (R84 item
3's run can include it as a second line). No retraining in this item; the AddToPlaylist activation trial of R85 stays as declared.

## R87 (answers Q38; both fixed at the source, sop @ de11429; 2026-10-26 17:20)

1. The adjacency rule assumed one-token fillers. The source now has a legend declaration `reader.spans`: `"token"` (default, the
   constructed worlds: "p3 p5 gives" still becomes undecided, checked) or `"kind_run"` (real data with typer kinds): a filler is the
   maximal run of its S: kind around the decoded head; the run's words count as placed (so they are not unplaced and do not raise the
   explained residual) and a neighbour of the same kind is the same span, never a competitor. Set `"spans": "kind_run"` in the
   per-intent legends' reader block (a legend change only; no retraining: the models' reading changes, their training does not).
2. `presence_query` now also covers a role present by its threshold with no admissible word (no filler): it is reported as a presence
   query and its margin leaves the frame margin. Re-run item 3's lines B and C with sop @ de11429 and `spans: kind_run`; expected: line B
   (full contract) no longer abstains on multi-word slots, its chosen-frame abstention at or below line C's, dispatch and spans
   unchanged, the explained residual of correct readings lower (the span words are explained), which may move the refusal operating
   points of R84 item 1 slightly (report them again on the same thresholds' rule).

## R88 (answers Q39; R86 (a)/(b) accepted; the rule reads the typer's in-context kinds; three re-score lines; 2026-10-26 18:05)

Your point is right and it changes the rule: a kind read from the lexicon is the majority over all intents, so the sibling fills
stay; the typer's in-context kind of the utterance is the given kind (rule 20, given per sentence). sop @ 4a0eba8 (SOURCE updated):
`Reader.formalize(text, ..., token_kinds=[...])` takes the in-context kind of every token (the typer's `sentence_kinds`); the decoder
uses them for the candidates (the sentence's words whose CONTEXT kind the question admits), the admission, the explained residual, the
spans (`kind_run`) and the adjacency rule; the chart keeps the lexicon kinds the routes were trained with (the state is unchanged, so
no retraining and no parity change in the state). Checked on the reference world: with the lexicon kinds every reading is identical;
with p3 typed O in context the agent gets no candidate and is not given.
Declared re-score lines on the saved R81 models (CPU; eval/r80_dispatch.py and r86_presence.py with `token_kinds` = the typer's
tags of the utterance, sop @ 4a0eba8, `spans: kind_run`, presence_query on):
(A) context kinds in the decoder, admission sets as stored. Expected: false fills on sibling words fall by most of their 70-100% share
    (a word typed S:movie_name is no candidate of timeRange), misses unchanged (the presence threshold decides them).
(B) (A) plus admission sets recomputed from the typer's in-context kinds of the training fillers (per role, kinds carrying at least 2%
    of its fillers; written into the saved schema, no retraining). Expected: the wide sets (restaurant_name 19, movie_name 35) shrink
    to a few kinds each; false fills fall further; misses may rise by the typer's own confusion on gold fillers (bounded by its error).
(C) (B) plus "presence by kinds where a kind names one question" (a kind admitted by exactly one question of the world: the question
    is present iff a word of that context kind is in the utterance; the threshold only where kinds are shared). Expected: the misses
    with the gold word's kind admitted (97-100% of them) turn into fills, so per-world span F1 approaches the typer's on the questions
    with a unique kind; report per question which rule decided presence.
Report the three lines beside R81 (dispatch, span F1 per world, false fills / misses per world, AUROC, the refusal operating points of
R84 item 1 recomputed with the same thresholds' rule). The best line by in-domain span F1 at unchanged dispatch becomes the demo's
reading; its decoder keys are then written into the exported configs. (C) is implemented in sop as a decoder key (`presence_by_kind`)
only after (A) and (B) are measured, so that each line changes one thing; say when (B) is in and I add it the same hour.

## R89 (answers Q40: presence_by_kind is in sop; 2026-10-26 18:45)

sop @ 040d44e (SOURCE updated): config key `presence_by_kind` (a decoder key, declared per world, not yet in the contract set). With the
sentence's candidates and the kinds given (in context when token_kinds are passed): a word of a kind that only this question admits
makes the question present; no word of any admitted kind makes it absent; words only of shared kinds leave the decision to the
threshold as before. The rule that decided is recorded per role (frame presence_rules; checked on the commands world: "unique kind
present" / "no word of an admitted kind" / "threshold (shared kinds)"). Line (C) = (B) plus this key; report per question which rule
decided and the false fills / misses per world beside (A), (B) and R81, as R88 says. The invalid runs are recorded as they should be;
the relaunched R85 activation trial (positives only) stands as declared.
