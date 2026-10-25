# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q22 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32 and R33 of responses.md, 2026-10-25, and accepted.)

## Q23. Why the MRC block takes so long, and why its model selection selects nothing (written by the main session, 2026-10-25 15:50)

What is happening (measured, not estimated, except where marked):

1. **Data size.** The MRC form turns each training sentence into one example per question: 30,000 sentences x 5 questions (levels 0-2,
   R29) = 150,000 examples per epoch, five times the tagger's. BERT-tiny (4.4 M) needs 1,306-1,385 s for 5 epochs on seed 0 while six
   jobs share the GPU (logs/2026-10-25_1509_bert_mrc_699a9ec.txt, _1518_, _1519_). R19's grid is lr {5e-5, 1e-4, 3e-4} x epochs
   {5, 10, 20}: 35 epochs per learning rate, about 2.5 h per learning-rate chain for BERT-tiny under this load (estimate).
2. **BERT-base MRC is the expensive row.** BERT-base trains at about 76 s per tagger epoch alone (2a); at 150,000 examples that is about
   9 min per MRC epoch (estimate), and the R24 grid {2e-5, 3e-5, 5e-5} x {3, 5, 10} is 54 epochs on seed 0, plus seeds 1-4 at the
   selected epochs: roughly 12-20 h for this one reference row (estimate). The R23 check measured no gain from concurrency for BERT-base
   (1.00x at 3, 0.99x at 6; commit 3bcc118), so it runs alone. The equal-size block (tiny / mini / small x pretrained / scratch / kinds)
   adds roughly 15-25 h (estimate).
3. **The suspicious part: validation is saturated, so the grid does not select.** Every grid point measured so far, for every model and
   form (2a, 2b, 3a taggers; BERT-tiny MRC at lr 5e-5 / 1e-4 / 3e-4, 5 epochs), gives validation FULL 1.000, and the MRC span loss prints
   as 0.00000. Checked by the main session: this is NOT leakage. The validation set is a fresh draw of the training distribution
   (make_train with rng 1100 + seed, 2,000 sentences); on seed 0 only 80 of 2,000 validation sentences (4.0%) also occur among the 29,359
   distinct training sentences. The cause is the world itself: R30 item 1 showed that a 15-line rule over the given kinds reaches exact
   match 1.000 on the training distribution, so a validation set drawn from it cannot separate a well-trained from a barely trained
   model. Consequences: (a) the selection rule "best validation FULL, ties to the lowest validation loss" degenerates into choosing by
   loss differences below the printed precision, i.e. by numerical noise; (b) the hours spent on the grid buy no information about the
   axes that matter (depth 3, depth not given, out-of-world), because selection may not look at them (rule 25); (c) the same saturation
   is why 2a's selection fell on the first grid point (R10 item 3).

Options (the author decides after the analysis; until then the queue runs as declared, and the BERT-base MRC grid has not started):

- (A) Keep everything as declared. Cost as above; selection is by loss noise, stated in RESULTS.
- (B) BERT-base MRC is a reference row since R28: give it the derived-row rule of R22 (inherit 2a's selected configuration, lr 2e-5,
  3 epochs, no grid), stated as a deviation. Saves most of item 2. The equal-size block keeps R19's grid.
- (C) Make selection informative for every model without touching the report rows: validation on held-out FORMS of the training
  distribution's depths (for example a template held out of training for validation only), or on the calibration rows' loss at higher
  precision. This changes the selection protocol of INSTRUCTIONS section 4 and would be a declared deviation for all new rows.
- (D) Keep the grids but cap them where the loss is already 0 (stop a learning-rate chain once its validation loss is below 1e-6):
  saves time, selection still uninformative.

Default until answered: (A) for what is running; the BERT-base MRC grid waits in the queue behind the current jobs (it starts after
them, in roughly 1-2 h). If no answer has arrived when it would start, the queue starts it as declared.

