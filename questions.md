# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q17 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20 and R23 of responses.md, 2026-10-25, and accepted.)

## Q18. R24 / R25: five points to settle before the MRC row trains (defaults below are followed until answered)

(1) **The level-3 question in training.** R24 says "training ... with every question of the legend per sentence, NULL for roles absent at
that depth" and also "the level-3 question ... a question never asked in training". Both cannot hold: with all six questions per
sentence, "who says it at level 3?" is asked on every training sentence and its answer is always NULL, so the model is TRAINED to refuse it
(a stronger handicap than the tagger's untrained X3). Default: (a) training asks the questions of levels 0-2 only (A, B, C, level 1,
level 2; NULL for the level questions absent at that depth), so the level-3 question is never seen in training and is a zero-shot
question at test; and (b) a second row with all six questions in training (level 3 always NULL), reported beside (a) and labelled so.
(2) **The questions' wording.** There is no legend file for the basic world in this repository; the world's verbs differ by frame (gives /
sells / lends / sends / offers / shows on the giver's side, receives / buys / gets / borrows / accepts on the receiver's). Default, one
question per role and level, verb-neutral, written in RESULTS and frozen: A "who gives?", B "what is given?", C "to whom is it given?",
X1 "who says it?", X2 "who says it at level 2?", X3 "who says it at level 3?" (X1 is the outermost speaker, level 1, as the world numbers
the levels).
(3) **The tag.** "Frozen with the pre-registration tag's successor commit": default, an annotated tag prereg-2026-10-25b on the commit that
writes deviation R24 with the questions and the decoding rule, before the MRC code runs on any real data; prereg-2026-10-25 stays.
(4) **DistilBERT-MRC.** R24 says bert-base-uncased "(and DistilBERT)"; R25 lists only bert-base-uncased before JP 1. Default: DistilBERT-MRC
after JP 1, with R19's small encoders.
(5) **R25's queue, two readings.** "R10 rows of 2a and of the reader (reader 'gates given' R14, 5 seeds)": default, the R10 row of 2a
(`--r10`, 2a's configuration, 5 seeds) and the reader's R10 rules for all five seeds from the R17 reader run (it retrains the five seeds
of 1a, checks them EQUAL, and the common report computes the one-to-one rule and the position split), plus the reader 'gates given' row.
"R17 (i) and the residual on every saved model": default, the reader (five seeds, retrained), BERT-base WordPiece (2a's configuration
re-run with every seed saved, exit check: reproduces 2a) and BERT-MRC (its NULL refusal and the same out-of-world sets); 2b / 3a / 3b
and the 7b controls (whose R11 re-runs are not in R25's list) after JP 1. Say if the R11 1b / 1c re-runs (one-to-one for every model)
should stay before JP 1; default: they run before JP 1 in a free slot, since they are small and R11 required them.

## Q19. R26: building 'unsupported' and 'undecided' in the basic world (default followed for the frozen planted sets)

The basic world's vocabulary has no word for "an extra piece of information" (no adverb, no 'with', no 'and'), and a word outside the
vocabulary is already the out-of-world source (c), i.e. 'unplaced' / 'unknown', not 'unsupported'. Default, built only from in-vocabulary
tokens and committed frozen (data/make_planted_sets.py, data/planted_seed<s>.json): **unsupported** = a basic-world sentence (depths 0-3,
training templates and identities) with ", <id> ." appended in place of the final "." (one person or object of the training identities
not already in the sentence): no question of the legend asks for it, the gold frame is unchanged, gold for that token = `unsupported`;
**undecided** = a basic-world sentence where a second identity of the same class is inserted right after the filler of one role (A, B or
C, chosen at random): two candidates in the same position class, equally supported by the form; gold for that role = `undecided`, the
other roles keep their values. 300 of each per seed, from their own generator (rng 73_000 + seed), disjoint from every frozen split.
If you prefer another construction (for example new words for the extra information, accepting that the reader cannot encode them),
say so before the R26 row runs (after JP 1); the frozen files would then be regenerated in a stated commit before any model sees them.
