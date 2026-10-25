# sop: the VSS reader (Python), the single implementation shared by the demo and this comparison

Source of truth for the reader's decoder from 2026-10-25 20:50 (owner's decision): this directory. The demo repository (vss-demo,
private) keeps a copy at `tool/sop` that its parity check diffs against this directory; the browser reader (vss-demo `app/src/vss/*.js`)
mirrors it with the Python/JS parity tests. Edited by the coordinator session (the author's side), like `responses.md`; the experimental
session reads it and never edits it.

Files: `legend.py` (the legend format and `sop check`), `infer.py` (the reader: chart, routes, readout, residuals, decoder rules),
`train.py` (training a world's reader with experiment/v8_gates.V8), `export_web.py` (the browser export), `soplang.py`, `world.py`,
`check.py`, `diff.py`, `cli.py`. `v8_gates_vorder_ref.py` is the reference copy of experiment/v8_gates.py with `set_vorder` /
`vorder_index` (variant b), for the row of responses R58; the pinned paper repository's `experiment/v8_gates.py` stays unchanged.

Decoder configuration keys read from a model's `config.json` (all off by default = the paper's decoder):
- `candidates: "sentence"`: a role's filler is a word of the sentence whose kind the role admits; one-to-one across roles (R46).
- `residual: "explained"`: the explained-tokens residual (R42 / R43).
- `undecided_adjacent: true`: an identity left in no role next to a role's filler whose kinds admit it -> that role is 'undecided' (R54).
- `reader.codes` containing `vorder` (legend): the verb-order route, variant b (R58).
Snapshot of vss-demo at the commit named in SOURCE.
