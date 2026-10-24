# Rules for agents working in this repository (vss-vs-bert)

The rules below are the main repository's AGENTS.md (ml-circuits-research/vsa-semantic-space at e531283), copied verbatim as
INSTRUCTIONS_2026-10-24.md section 2 requires, with these additions for this repository:

- **Instructions.** `INSTRUCTIONS_2026-10-24.md` fixes the question, the models, the data, the splits, the measures and the written
  interpretation of every outcome (its Table 3). The session executes and reports; it does not change the question. A deviation that
  becomes necessary is recorded in RESULTS.md with its reason before the run that uses it.
- **Main repository.** `third_party/vsa-semantic-space` is a submodule pinned at e531283 (`common/mainrepo.py` checks it). Nothing in it
  is modified; its modules are imported. Its `experiment/EVALUATION_PRINCIPLES.md` rules 1-30 apply to every script here.
- **Frozen splits.** Every model is scored on `data/basic_seed<s>.json` (run 0); a model script that draws its own test sentences is not a result.
- **Layout.** `common/` (world, frozen splits, decoding, reporting), `data/`, `models/` (BERT, 7b), `vss/` (the reader), `eval/`
  (metrics, checklist, compare), `logs/` (one log per run, name with the commit), `paper/`.
- **Runs.** `./run.sh <script> [args]` from the repository root (it refuses an unclean tree except logs/ and *.md).

---


1. **Language.** Conversation with the author may be in Romanian. Everything that goes to disk or into a commit — code, comments,
   docstrings, documents, logs, data, commit messages — is in **English**. The Romanian documents have been translated; a
   translation must not change any number, table or conclusion (verify by diffing the extracted numbers before/after).
2. **Source of truth.** `experiment/00_READ_FIRST.md` and `experiment/RESULTS.md`
   rounds 2+ . Anything that contradicts a later round is marked RETRACTED/SUPERSEDED, never deleted.
3. **Evaluation.** Follow `experiment/EVALUATION_PRINCIPLES.md`, rules 1-30, for every new experiment.
   Print the checklist at the end of every evaluation script.
4. **Runs.** Commit the code first, then `./run.sh script.py [args]` (it refuses an unclean tree except `logs/`); every result cited in a
   document points to a log in `experiment/logs/` whose name carries the commit hash. Never re-run to "improve" old numbers.
5. **Workflow.** Code commits are forced by `run.sh`; a **push** marks a judgment point for the reviewing session. At every judgment
   point rewrite `current_status.md` (root) and add an entry to `experiment/JOURNAL.md`.
6. **Synthetic worlds.** From 2026-10-15 on, advanced tests use the English world (`v5_scale.set_lang("en")`); the Romanian world is kept
   for reproducing rounds 1-5.
