# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

## Q1. The small pretrained encoder (needed before run 3)

INSTRUCTIONS section 3 asks for one of `distilbert-base-uncased` (66 M) or `prajjwal1/bert-small` (29 M), chosen before the runs and
recorded in RESULTS.md. Proposal: `distilbert-base-uncased`. It is the standard "small BERT" a reviewer expects, it has the same
vocabulary and tokenizer as BERT-base (the tokenizer conditions stay comparable), and its from-scratch twin is still well below BERT-base
in size. `bert-small` is closer to a CPU budget but is a less common checkpoint (a community upload, not an official Google release).
Measured on the DGX (2026-10-25): `prajjwal1/bert-small` does not load with the pinned transformers 5.20.2 (the checkpoint has
no fast-tokenizer file and the slow-to-fast conversion fails), so choosing it would also need a tokenizer workaround (a deviation).
Default if there is no answer by the time runs 0-2 are done: `distilbert-base-uncased`, recorded in RESULTS.md before run 3.

## Q2. Push cadence

AGENTS rule 5 makes a push a judgment point. You asked for a commit at every step. Default: commit after every step (code, log,
documents), push after every run so that you can follow the work from the other session; the formal judgment points (JP 1, JP 2) stay
where the RUNBOOK puts them and are named as such in the commit message. Tell me if you want pushes only at the judgment points.
