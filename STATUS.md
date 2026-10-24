# STATUS — claims index (one row per axis and model; label BUILT / MEASURED / PLANNED; every number points to a log)

| axis | model | label | log | number | interpretation (INSTRUCTIONS table 3) |
|---|---|---|---|---|---|
| run 0: frozen splits, tag -> frame mapping | — | PLANNED | | | exit check: 1.000 on every seed |
| seen / new ids / unseen depth (basic world) | VSS reader (reference) | PLANNED | | | must reproduce RESULTS 6c / 7b within seed variance |
| seen / new ids / unseen depth (basic world) | transformer 7b-bis, 7b | PLANNED | | | must reproduce RESULTS 7b-bis (0.253 / 0.393) and 7b (0.215 / 0.390) |
| seen / new ids / unseen depth (basic world) | BERT-base pretrained, wordpiece and whole | PLANNED | | | new ids > 0.9 / < 0.5; depth 3 > 0.9 / < 0.5; in between both readings |
| seen / new ids / unseen depth (basic world) | small pretrained encoder; same architecture from scratch | PLANNED | | | the pretraining difference |
| reference world: unseen depth FULL, passives, fronted forms | all | PLANNED | | | |
| WikiSQL 1k / 5k / 56k: selected column, aggregator | all | PLANNED | | | the size at which each model reaches 0.75 |
| abstention: coverage, confident-wrong, curves, ECE | all | PLANNED | | | lower confident-wrong at equal coverage abstains better, per axis |
| occupancy (not given) | taggers | PLANNED | | | false fillers on absent roles |
| size and cost; determinism | all | PLANNED | | | reported; the ratio is part of every claim |
| 50-item confident-error analysis | reader, BERT | PLANNED | | | by kind, side by side |
