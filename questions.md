# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47 and R50 of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q29 (2026-10-25 21:10, experimental session): R58 item 1 is blocked: sop/ cannot load the saved basic-world readers without a change in sop/

What the saved readers need (every 1a / R14 / R37 / vorder model of this repository, trained by v8_gates.V8 against
common/basic_world, i.e. v7_nested.NestedRoleWorld):
1. **Identities**: F = exp(i U(-pi, pi)) of shape (len(vocab), D) from np.random.default_rng(300 + seed) over bw["w"].vocab in that
   order. sop already supports this: catalog.json "identity_table" {"vocab": [...], "rng_seed": 300 + seed}. No change needed.
2. **Role and level vectors**: NestedRoleWorld draws R["A"], R["B"], R["C"], R["X"], R["rep"] in this order from
   np.random.default_rng(100 + seed), each exp(i U(-pi, pi), D). sop's world.Stars draws every role / operator vector from
   star(seed, "role:<r>" / "op:<name>", D) and has no table override for them. **Missing in sop/:** a catalog (or config) field that
   gives the named star vectors explicitly, e.g. "star_table": {"rng_seed": 100 + seed, "order": ["role:A", "role:B", "role:C",
   "op:says.speaker", "op:says.scope"]}, read by Stars.get before the hashed star (the same pattern as identity_table).
3. **The frame algebra** matches sop's says / scope form (S = R_X (x) f_speaker + R_rep (x) S_inner; inner frame R_A f_A + R_B f_B +
   R_C f_C, no predicate term, no occupancy marker). To be confirmed on sop's side: that a legend can declare no predicate role and no
   marker term (world.state of the basic world has neither), otherwise sop's compiled residual will not equal common/decode.residuals.
4. **Candidates**: the basic-world readout restricts A / C / speakers to the axis's person list and B to its object list
   (bwm.candidates_for: training lists, the held-out lists on the new-ids axis); sop's cands() takes the catalog words of the role's
   kinds; a per-call candidate filter (the `filt` argument) would reproduce it if the catalog holds all 18 persons / objects.
**Default until answered:** item 1 is NOT run (no EQUAL check is possible without 2); the basic-world legend and kinds.tsv are not
written yet (they depend on 3); every reader row keeps the label "decoder: common/decode.py (transcription; sop EQUAL check pending,
Q29)"; nothing else waits on this. Variant b (R58 item 2) runs as declared; then the BERT-tiny growth rows; then JP 1.
