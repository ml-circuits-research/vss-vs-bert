# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1-Q11 were answered by R1, R2, R5, R7, R11, R12 and R13 of responses.md, 2026-10-25, and accepted.)

## Q12. R17: the reader of the basic world has no residual and no status in this repository's path

The basic-world reader (v8_gates.V8 decoded by common/decode.decode_frozen) returns a frame and margins; the residual and the statuses
(no-sense / unknown-event) belong to the sop tool (tool/sop/infer.py), which runs the real-data round, not the basic world. The vss-demo
ADDENDUM is not on this machine; R17's text is taken as the specification.
Default: compute the sop tool's residual for the basic-world reader in common/decode.py, with the same definition (infer.py:12):
residual = ||S - a * rec|| / ||S||, rec = world.state of the decoded event (decoded frame, depth as given or estimated), a = the best real
scale. Rejected = residual above a threshold set on calibration rows: variant (i) at the 2% false-rejection point of the in-world
calibration rows; variant (ii) the same point with the 'neg' negatives added (best separating threshold under <= 2% false rejection).
No "status" for the reader on this world (there is no event classifier to return unknown-event). Needed before the R17 rows.

## Q13. R17 source (b): the reference world shares the basic world's whole vocabulary

All 6,920 reference-world sentences of seed 0 use only basic-world words, and most are basic-world sentences (`p0 gives o9 to p7 .`).
Default (in data/make_oow_sets.py): a reference sentence is out-of-world only if its category sequence is not that of any basic-world
sentence (forms of every template the splits use, depths 0-3): 867 of 6,920 on seed 0, mostly forms without the recipient
('o6 is sent by p0 .', 'p3 claims that p1 gives o0 .'). They are another legend's forms but readable by a person with C "not given",
so (b) is the hardest and least clear-cut source; it is reported per source, never pooled alone. The 'neg' part has 320 reference items on
seed 0 (the pool is exhausted). Needed before the OOW sets are committed (right after run 2b).

## Q14. R17 source (c): a word of no kind cannot be encoded by the reader

The reader's encoder looks every word up in the lexicon; a nonce word has no category and no vector, so the reader cannot produce a
state for (c) at all, while BERT reads its WordPiece pieces. Default: the reader rejects (c) by construction ("word outside the legend",
the sop tool's 'unplaced'), reported as such and NOT counted as a learned rejection; the learned comparison is made on (a) and (b); for
BERT, (c) is a real test. Alternative if you prefer: give the nonce word a random vector of category 'O' (a placeholder) so the reader
must decide from the state. Needed before the R17 rows.

## Q15. R17 scope: which models and which configurations

"Every model" for three new rows would mean new training for every model of runs 1-3. Default: item 1 (i) (no training) and item 2
(residual analogue) for the reader, 7b-bis / 7b (one-to-one cosine scores in place of probabilities) and every BERT / DistilBERT row,
from saved seed-0 models where they exist and from 5-seed reruns with the selected configuration otherwise; item 1 (ii) (trained OOW
class) and item 3 (depth head) as new 5-seed rows for BERT-base WordPiece (2a configuration, lr 2e-5, 3 epochs, no grid) and DistilBERT
pretrained (3a configuration), and the matching reader rows (negatives as residual calibration; depth from the count of level openers).
The from-scratch encoder and the 7b controls get (ii) and item 3 only if you ask. Needed before the R17 rows.
