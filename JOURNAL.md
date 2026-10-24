# JOURNAL — one entry per judgment point

## 2026-10-24 — repository created (reviewing session)
Instructions written and fixed before any run (INSTRUCTIONS_2026-10-24.md); main repository pinned at e531283 as a submodule; the
common world, frozen splits, decoding, scorer and reporting written and smoke-tested on CPU at small size (D 256, few hundred sentences;
the numbers of a smoke test are not results); the reader, the 7b / 7b-bis control, the BERT tagger (both tokenizer conditions, scratch
variant) and the WikiSQL head written; eval/compare.py produces the tables from logs/results_*.json. Decisions left to the first
judgment point: the small pretrained encoder (DistilBERT-base or BERT-small) and whether the second round (negation and operator worlds)
follows judgment point 1 or 2. Next: run 0 on the DGX (frozen splits at D 1024, five seeds), then run 1.
