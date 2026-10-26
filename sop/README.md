# sop: the VSS reader (Python), the single implementation shared by the demo and this comparison

Source of truth for the reader's decoder from 2026-10-25 20:50 (owner's decision): this directory. The demo repository (vss-demo,
private) keeps a copy at `tool/sop` that its parity check diffs against this directory; the browser reader (vss-demo `app/src/vss/*.js`)
mirrors it with the Python/JS parity tests. Edited by the coordinator session (the author's side), like `responses.md`; the experimental
session reads it and never edits it.

Files: `legend.py` (the legend format and `sop check`), `infer.py` (the reader: chart, routes, readout, residuals, decoder rules),
`train.py` (training a world's reader with experiment/v8_gates.V8), `export_web.py` (the browser export), `soplang.py`, `world.py`,
`check.py`, `diff.py`, `cli.py`, `activate.py` (activation: contract rules, route selection, diagnostics), `typer.py` (kinds from annotation). `v8_gates_vorder_ref.py` is the reference copy of experiment/v8_gates.py with `set_vorder` /
`vorder_index` (variant b), for the row of responses R58; the pinned paper repository's `experiment/v8_gates.py` stays unchanged.

Activation (2026-10-26, `activate.py`; the argument in vss-demo vision/ACTIVATION_rules_and_worlds_2026-10-26.md): which rules a world's
reader runs is decided from the nature of the world, not by hand. Two levels: worlds are activated at inference by the dispatcher (every
reader reads, the smallest explained residual under its fit threshold wins, none fitting = refusal); rules and routes are activated at
training, per world. Three classes: (1) CONTRACT rules, consequences of q-formalisation, on in every model `sop train` writes
(`--decoder contract`, the default; `--decoder paper` = the frozen rows' decoder, the keys below absent); (2) STRUCTURAL routes
(`vorder`, `pos`) are candidates: `sop train --activate auto` measures a need signal on the training annotation (same-kind competition for
`vorder`: two questions answered by words of one kind; rigid order for `pos`) and accepts a route only when a trial trained without the
selection rows loses no selection split by more than eps and wins the marked rows; the trials and the decision are written to
`config.json` (`activation`) and to the run folder (`activation.json`, `PROFILE.md`); (3) DIAGNOSTICS, reported and never acted on:
tokens without a kind (-> typer.py), competition no route wins (-> a question missing from the legend, or a world to split), levels
without declared openers. `sop profile <world>` prints the need signals without training.

Decoder configuration keys read from a model's `config.json` (absent = the paper's decoder; written by `sop train --decoder contract`):
- `candidates: "sentence"`: a role's filler is a word of the sentence whose kind the role admits; one-to-one across roles (R46).
- `residual: "explained"`: the explained-tokens residual (R42 / R43).
- `undecided_adjacent: true`: an identity left in no role next to a role's filler whose kinds admit it -> that role is 'undecided' (R54).
- `reader.codes` containing `vorder` (legend): the verb-order route, variant b (R58).
Snapshot of vss-demo at the commit named in SOURCE.
