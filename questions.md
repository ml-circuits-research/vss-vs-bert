# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35), R84 (Q36) and R85 (Q37) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q38 (2026-10-26 17:05, experimental session): two contract-decoder observations from R84 item 3 on the SNIPS worlds (sop/, yours)

Checked by hand on four dev_rep utterances (AddToPlaylist, BookRestaurant; seed 0 R81 models), each read under sop 82585ea and b467e4b and
four decoder settings (paper, full contract, presence_query alone, undecided_adjacent alone):
1. **`undecided_adjacent` makes every multi-word slot undecided on a head-token readout.** The rule turns a role into an undecided query
   when an unplaced word next to the role's filler has a kind the role admits. In SNIPS the reader places one head token per slot, so the
   other words of the same span are exactly such neighbours ("add slimm cutta calhoun to my this is prince playlist": artist 0.0 and
   playlist 0.0; "book a table at skycity in salado for me and hannah": party_size_description 0.0). With it the chosen frame abstains on
   nearly every utterance. Same under 82585ea, so not a version mismatch. The rule assumes single-token fillers (the synthetic worlds).
   Possible fixes (yours): exempt neighbours inside the typer's span of the same kind (`kind_run`), or keep the rule off for head-token worlds.
2. **`presence_query` leaves a gap: present-without-filler.** A role whose presence decision says present but which ends with no filler
   keeps its presence margin in the frame margin, because the query branch only covers absent roles ("book a table at skycity in salado
   ...": timeRange.presence 0.02 stays in the margin although timeRange is not in the frame). Otherwise presence_query works as designed
   (e.g. spatial_relation 0.025 abstain -> ok).
**Default until answered:** item 3 reports line C (presence_query alone, the line that measures R84's expectation) as the relevant line,
and line B (full contract) with the cause above; no sop change on this side.
