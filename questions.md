# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(Q1, Q2 and Q3 were answered by R1, R2 and R5 of responses.md, 2026-10-25, and accepted.)

## Q4. R6 item 3: edit INSTRUCTIONS_2026-10-24.md Table 3, or the note in RESULTS only?

AGENTS says INSTRUCTIONS fixes the written interpretation and a deviation is recorded in RESULTS.md. I wrote the Table 3 reading for the
unseen-depth row (tagger compared on the inner frame and the trained labels; six-role FULL reported with the note) in RESULTS, section
"Depth-3 metrics added before run 2 (R6)", and did not edit INSTRUCTIONS_2026-10-24.md. Default: leave INSTRUCTIONS verbatim and keep the
note in RESULTS. If you want the sentence inside INSTRUCTIONS Table 3 as well, say so and I add it marked "(added 2026-10-25, R6)".

## Q5. Proposal for after JP 1: a recursive tagger that can express an unseen depth (R6)

Not added now. The model: the BERT tagger applied recursively, as models/bert_real.py does for TOPv2: at each level the head labels the
outermost speaker (X) and the span of the embedded clause; the clause is fed back into the same encoder and head; at the innermost level
the head labels A, B, C. Every label then occurs in training at depths 0-2, so depth 3 is three applications of a trained step, and the
comparison asks whether BERT composes, not whether it can output an untrained label. Cost: code, moderate (a span head and the recursion
already exist in bert_real.py; training targets from the basic world's gold, which nests by construction); compute, one encoder pass
per level (about 2-4x the current tagger at depth 3), the same grid and 5 seeds for BERT-base and DistilBERT: about the cost of runs 2a
and 3a again. Caveat: the recursion gives the tagger the clause boundary structure as an architectural prior that the reader does not
get from outside, so it is a different model and would be reported as such, beside the flat tagger, not instead of it.
Default: not done; decided at JP 1.
