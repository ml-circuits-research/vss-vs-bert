# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31) and R69 (Q32) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q33 (2026-10-26 11:50, experimental session): R75 did not meet its expectations; R78's rows are not started

R75 as declared (every typer kind in every world, dispatch by explained residual then min margin, rejection when every world is above its
own 98th-percentile threshold; D 512, 10 epochs, three routes, seeds 0-2, both conditions; sop @ 28db081; RESULTS, commit 237fb76):

| | R75 (mean, 3 seeds) | R75 expected | one world (R71) | R72 as run |
|---|---|---|---|---|
| dispatch (intent) accuracy, test | 0.595 (0.589-0.601) | near 0.977 | 0.977 | 0.447-0.487 |
| residual AUROC, held-out vs in-domain | 0.775 +- 0.013 | > 0.9 | 0.933 | 0.555-0.578 |
| held-out intents rejected | 0.000 | far above 0.49 | - | - |
| in-domain false rejection | 0.000 (nothing is rejected) | < 0.05 | - | - |
| span F1, test | 0.602 | - | 0.798 (typer alone 0.886) | 0.511-0.542 |

Also: head-frame exact 0.312; the chosen frame abstains on 0.667 in-domain / 0.783 held-out; size 61.0 M phases for the 7 readers
(44.3 M on ood_intents) + 2.69 M typer weights.
**Observed cause of the zero rejection:** four of seven world thresholds are exactly 1.0 (BookRestaurant, GetWeather, PlayMusic,
SearchScreeningEvent; others 0.75 / 0.52 / 0.92): more than 2% of each world's own training utterances leave a core question unanswered,
which scores 1.0 (R64's clause), so the 98th percentile is 1.0 and no residual can exceed it. The same saturation plausibly explains part of
the dispatch loss (several worlds tie at 1.0 and the tie is broken by the margin), but that is not measured yet.
**Directions the session sees (none implemented; for the joint decision announced by the owner):**
1. In SNIPS most slots are optional (an utterance names a subset of its intent's slots). If every slot of a world is a core question, R64's
   "core question unanswered -> 1.0" fires on ordinary in-domain utterances. Declare the SNIPS slot questions optional in each legend
   (only the questions present in every training utterance of the intent stay core, if any), so the residual becomes the share of
   unexplained tokens. Cheap diagnostic first, from the saved models without retraining: the share of 1.0 residuals per world and how
   many come from the clause vs unexplained tokens.
2. Dispatch on the graded part only (the unexplained-token share), the clause applied inside the chosen world.
3. A hybrid: the one-world reader's intent (0.977) picks the world; the per-world reader gives the frame; refusal by the chosen world's
   residual. This keeps the one-world intent and tests the per-world readout and refusal separately.
**Default until answered:** nothing runs; the per-world model folders are kept (logs/worlds/real_snips_<condition>_intent_<intent>_seed<s>/
model_seed) for whichever direction is chosen.
