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
