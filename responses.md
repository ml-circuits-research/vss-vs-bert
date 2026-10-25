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
