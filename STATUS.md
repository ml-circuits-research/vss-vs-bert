# STATUS — claims index (one row per axis and model; label BUILT / MEASURED / PLANNED; every number points to a log)

| axis | model | label | log | number | interpretation (INSTRUCTIONS table 3) |
|---|---|---|---|---|---|
| run 0: frozen splits, tag -> frame mapping | — | MEASURED | logs/2026-10-25_0748_make_splits_1617108.txt | 1.000 on seeds 0-4 (content identical to the committed splits) | exit check: 1.000 on every seed |
| seen / new ids / unseen depth (basic world) | VSS reader (reference) | MEASURED | logs/2026-10-25_0750_run_reader_basic_fade9d5.txt | seen 1.000, new ids 1.000, depth 3 0.997 ± 0.002 (FULL 0.982 ± 0.010); reproduces 7c learned-closed (gate MET, R5); R6 seed 0: FULL trained labels 0.982, inner 0.982 | must reproduce RESULTS 6c / 7b within seed variance |
| seen / new ids / unseen depth (basic world) | transformer 7b-bis, 7b | MEASURED | logs/2026-10-25_0809_transformer_7b_c8e4e96.txt; logs/2026-10-25_0817_transformer_7b_b4f0739.txt | 7b-bis: new ids 0.251 ± 0.048, depth 3 0.390 ± 0.021; 7b: 0.215 ± 0.015, 0.392 ± 0.025 (both MET) | must reproduce RESULTS 7b-bis (0.253 / 0.393) and 7b (0.215 / 0.390) |
| seen / new ids / unseen depth (basic world) | BERT-base pretrained, wordpiece (2a, pre-registered) | MEASURED | logs/2026-10-25_0830_bert_basic_4429ea4.txt | seen 1.000, new ids 1.000, new form. 0.999 / 1.000, depth 3 roles 0.961 ± 0.063; FULL depth 3: six roles 0.765 ± 0.376, trained labels 1.000, inner 1.000 |
| seen / new ids / unseen depth (basic world) | BERT-base pretrained, whole (2b, pre-registered) | MEASURED | logs/2026-10-25_1207_bert_basic_f2afe4e.txt | seen 1.000, new ids 1.000, new form. 0.993 / 0.977, depth 3 roles 0.892 ± 0.107; FULL depth 3: six roles 0.495 ± 0.444, trained labels 0.858 ± 0.281, inner 1.000 | new ids > 0.9 / < 0.5; depth 3 > 0.9 / < 0.5; in between both readings |
| seen / new ids / unseen depth (basic world) | small pretrained encoder (3a, DistilBERT, pre-registered) | MEASURED | logs/2026-10-25_1336_bert_basic_7a1500a.txt | seen / new ids 1.000, new form. 1.000 / 0.985, depth 3 slots 0.938 ± 0.063; exact match depth 3: six 0.631 ± 0.379, trained 1.000, inner 1.000 | reference row (R28) |
| seen / new ids / unseen depth (basic world) | same architecture from scratch (3b) | NOT RUN (R28; RESULTS '3b: not run') | | | answered by the (b) rows of the equal-size block |
| equal-size block, MRC form (R28): BERT-tiny / mini / small x pretrained / scratch / kinds given; BERT-base MRC reference | BERT-MRC | PLANNED | | | R19 (a) / (b) / (c) per size |
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
| seen / new ids / new forms / depth 3 (basic world) | VSS reader, gates given (R14, exploratory) | MEASURED | logs/2026-10-25_1506_run_reader_basic_699a9ec.txt | new form. 0.764 ± 0.029 / 0.789 ± 0.011 (1a: 0.864 / 0.868); depth-3 exact match 0.999 ± 0.001 | removes the seed spread; lowers new formulations |
| out-of-world rejection (R17 i), residual, R10 one-to-one (basic world) | VSS reader 1a (exploratory) | MEASURED | logs/2026-10-25_1507_r17_reader_699a9ec.txt | rejected random 1.000, reference 0.989 ± 0.009, nonce 1.000; false rejection pooled 0.435 ± 0.015 (depth 3 and UNSEEN trans. 1.000): demo criterion NOT met; one-to-one depth-3 exact match 0.995 |
| frozen splits seeds 5-9 (R39, held-out claim set; unopened) | — | BUILT | logs/2026-10-25_1606_make_splits_33dc482.txt | tag->frame mapping exact; sha256 in commit 9f71308 |
| BERT-tiny MRC grid chains (pretrained, seed 0) | BERT-tiny MRC | STOPPED, partial (R36) | logs/2026-10-25_1509_bert_mrc_699a9ec.txt, _1518_, _1519_bert_mrc_bb13fee.txt | 5-epoch points: validation exact match 1.000, span loss 0.00000 at lr 5e-5 / 1e-4 / 3e-4; 10/20-epoch points stopped |
| LLM reference (R27.2, R32, R33) | claude-sonnet-5-5, 10-shot | STOPPED (1,300 / 4,800 calls, 0.62 USD; killed with its parent session), RESTARTED 16:12 | logs/2026-10-25_1514_llm_reference_bb13fee.txt | cap of the restart 19.35 USD |
