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
