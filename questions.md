# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35), R84 (Q36), R85 (Q37) and R87 (Q38) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q39 (2026-10-26 17:45, experimental session): R86 measured: cause (a) holds, (b) does not; a point on the decoder rule before you write it

R86 (eval/r86_presence.py, saved R81 models, test; logs 2026-10-26_1519.._1521_r86_presence; RESULTS, 6b24478):
- **(a) admitted-kind overlap: holds.** False fills per seed: BookRestaurant 64 / 69 / 88, of which 70-78% are words of a sibling gold
  question, and in 96-100% of those the filled role also admits that word's kind; SearchScreeningEvent 23 / 28 / 24, 91-100% sibling words,
  100% admitted. Top pairs (filled <- gold): city <- timeRange, city <- restaurant_name, timeRange <- party_size_number, restaurant_type /
  cuisine <- restaurant_name; timeRange <- movie_name, movie_type <- object_type. Admission sets are wide (restaurant_name 19 kinds,
  city 16, movie_name 35). Misses (~58 / ~29 per seed): the gold word's kind is admitted in 97-100%, the presence ratio is under its
  threshold in every miss.
- **(b) interference at D 512: does not hold.** Separation (mean present - mean absent) minimum 0.36 (city), then 0.44, 0.49 in
  BookRestaurant; >= 0.80 in SearchScreeningEvent; no role within 0.1; balanced accuracy of the thresholds >= 0.896. By R86's rule the
  D 1024 trial is not run.
**Mechanism observed:** a role admits the LEXICON kinds of its training fillers, and the lexicon gives each word its majority kind over
all intents; in context the typer says e.g. S:movie_name for "ghost" or "american", while the lexicon kind is S:object_name, which
timeRange admits. **Point for your rule "presence by kinds where a kind names one question":** if the rule reads the kinds from the
lexicon, the sibling fills above stay (the overlap is in the lexicon kinds); if it reads the typer's IN-CONTEXT kinds of the utterance
(Typer.tag, as kind_run already does for spans), most of them should go. Your choice; nothing runs for it on this side.
**Default until answered:** the rule is re-scored as a second line when sop announces it, with whatever kind source it uses, stated.
