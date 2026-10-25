# STATUS — claims index (one row per axis and model; label BUILT / MEASURED / PLANNED; every number points to a log)

| axis | model | label | log | number | interpretation (INSTRUCTIONS table 3) |
|---|---|---|---|---|---|
| run 0: frozen splits, tag -> frame mapping | — | MEASURED | logs/2026-10-25_0748_make_splits_1617108.txt | 1.000 on seeds 0-4 (content identical to the committed splits) | exit check: 1.000 on every seed |
| seen / new ids / unseen depth (basic world) | VSS reader (reference) | PLANNED | | | must reproduce RESULTS 6c / 7b within seed variance |
| seen / new ids / unseen depth (basic world) | transformer 7b-bis, 7b | PLANNED | | | must reproduce RESULTS 7b-bis (0.253 / 0.393) and 7b (0.215 / 0.390) |
| seen / new ids / unseen depth (basic world) | BERT-base pretrained, wordpiece and whole | PLANNED | | | new ids > 0.9 / < 0.5; depth 3 > 0.9 / < 0.5; in between both readings |
| seen / new ids / unseen depth (basic world) | small pretrained encoder; same architecture from scratch | PLANNED | | | the pretraining difference |
| run 4: frozen splits of the reference world, tag -> frame mapping | — | PLANNED | | | exit check: 1.000 on every seed |
| reference world: FULL per axis, roles FULL depth 3, passives, T9, occupancy, predicate | VSS reader (10b-quinquies) | PLANNED | | | must reproduce RESULTS 10b-quinquies (depth 3 roles FULL 0.968 ± 0.020) |
| reference world: the same axes | BERT-base, small encoder, from scratch | PLANNED | | | Table 3 bands on depth 3 and unseen transitions |
| WikiSQL 1k / 5k / 56k: selected column, aggregator (same pool, dev report half) | VSS reader (this repository's run) | PLANNED | | | must reproduce RESULTS 8d within seed variance |
| WikiSQL 1k / 5k / 56k: selected column, aggregator | BERT-base, small encoder, from scratch | PLANNED | | | the size at which each model reaches 0.75 |
| abstention: coverage, confident-wrong, curves, ECE | all | PLANNED | | | lower confident-wrong at equal coverage abstains better, per axis |
| occupancy (not given) | taggers | PLANNED | | | false fillers on absent roles |
| size and cost; determinism | all | PLANNED | | | reported; the ratio is part of every claim |
| 50-item confident-error analysis | reader, BERT | PLANNED | | | by kind, side by side |
| real data: conversion and frozen splits (SNIPS committed; MASSIVE, TOPv2 at runs 9a / 10a) | (none) | BUILT (SNIPS) / PLANNED | | | exit check: annotation round trips exact, gold frames 1.000 (tests/test_real_data.py) |
| real data, SNIPS / MASSIVE base 1k / 5k / full: FULL head frame, intent, slot heads, span F1 (dev report; test once) | VSS reader (sop tool) | PLANNED | | | INSTRUCTIONS_REAL table R7: BERT pretrained expected above; the gap, the size and the cost are the result |
| real data, SNIPS / MASSIVE base: the same | BERT-base, small encoder, from scratch | PLANNED | | | the size at which each model reaches 0.8 FULL |
| real data: new slot values (head accuracy on heads never in training) | all | PLANNED | | | > 0.8 read from context; < 0.3 needs the value seen |
| real data: held-out templates (retrained) and unseen templates (natural) | all | PLANNED | | | the seen-to-held-out drop per model |
| real data: out-of-domain intents (abstained / confidently mapped; AUROC of the intent margin) | all, softmax and temperature for the taggers | PLANNED | | | higher abstention at equal in-domain coverage rejects better |
| real data, TOPv2: depth 2 (hops seen), FULL and nested intents | all | PLANNED | | | reader > 0.8 and encoders < 0.5 carries the constructed-world result to real data |
| real data: size, cost, determinism | all | PLANNED | | | reported; the ratio is part of every claim |
| real data, MASSIVE multilingual (round 2) | reader, mBERT, XLM-R | PLANNED | | | decided at judgment point 3 |
