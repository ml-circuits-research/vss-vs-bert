# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34), R81 (Q35) and R84 (Q36) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q37 (2026-10-26 16:10, experimental session): R84 item 1 measured (anchor misses); item 2's activation test cannot fire as the worlds are written

**R84 item 1** (saved R81 models, seeds 0-2, sop @ 82585ea; thresholds on the dev calibration half, reported on test; held-out GetWeather
test 100, in-domain 600; logs 2026-10-26_1501.._1509_r84_oppoints; RESULTS, 74fd4e3). Held-out rejection (measured test false rejection):

| signal | AUROC | target FR 2% | target FR 5% |
|---|---|---|---|
| chosen world's residual | 0.941 | 0.183 +- 0.048 (0.008) | 0.467 (0.031) |
| anchor (1 - percentile) | 0.892 | 0.000 (0.004) | 0.100 (0.028) |
| max of the two | 0.887 | 0.057 | 0.090 |
| second line: every world's residual percentile high | 0.881 | 0.327 (0.031) | 0.447 (0.056) |
| second line: every world's anchor z below calibration (added before the run, after the smoke test) | 0.895 | 0.087 | 0.303 (0.041) |

Expectation "anchor well above 0.28 at 5%" **not met** (0.100): the anchor's AUROC comes from the bulk; the lowest in-domain anchors sit
as low as the held-out ones, and a percentile cannot order below a world's calibration minimum. **Demo refusal line by R84's rule:** the
chosen world's residual above its dev-calibrated 2% threshold (0.183 held-out at 0.008 false on test). The "every world's residual high"
line refuses more at 2% (0.327) but its test false rejection is 0.031, above the target: your choice if wanted.
**R84 item 2, activation part:** with the typer's kinds the `vorder` need signal now fires on AddToPlaylist only (seed 0, 2.4% of rows),
not on R74's three worlds (hand kinds). More decisive: the per-intent legends declare only the event verb `command`, which is no word of
the utterances, so the verb-order route finds no verb category and `--activate auto` would be a near no-op; R84's "vorder accepted on at
least one world" cannot be tested as the worlds are written. Possible world change (R83: cause = the world, an event verb missing):
declare each intent's command verbs (add, play, book, rate, find, show, is it ...) as the legend's event lemmas, so the predicate and the
verb-order route have a verb to stand on. Expected to move: the loss on same-kind slot pairs (e.g. two artists / two playlists), if any.
**Default until answered:** item 2's per-world span readout (CPU, no training) runs; the activation retraining does not (it would cost
shared GPU time for a predicted no-op); item 3 (re-score with presence_query) runs.
