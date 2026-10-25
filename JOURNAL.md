# JOURNAL — one entry per judgment point

## 2026-10-24 — repository created (reviewing session)
Instructions written and fixed before any run (INSTRUCTIONS_2026-10-24.md); main repository pinned at e531283 as a submodule; the
common world, frozen splits, decoding, scorer and reporting written and smoke-tested on CPU at small size (D 256, few hundred sentences;
the numbers of a smoke test are not results); the reader, the 7b / 7b-bis control, the BERT tagger (both tokenizer conditions, scratch
variant) and the WikiSQL head written; eval/compare.py produces the tables from logs/results_*.json. Decisions left to the first
judgment point: the small pretrained encoder (DistilBERT-base or BERT-small) and whether the second round (negation and operator worlds)
follows judgment point 1 or 2. Next: run 0 on the DGX (frozen splits at D 1024, five seeds), then run 1.

## 2026-10-25 — set and hash order in the reader's and the transformer's path (responses R3; read, not run)

run.sh exports PYTHONHASHSEED=0 from commit c8e4e96. The code of runs 1a-1c was read for indices or random draws built by iterating a
set (or a dict built from a set) at run time: vss/run_reader_basic.py, models/transformer_7b.py, common/basic_world.py, decode.py,
report.py, data/make_splits.py, and in the submodule (unmodified) v5_scale, v7_nested, v8_gates, v6_left, v2, v4, v4_torch, camp,
v9_transformer, v22_transformer_categories. Findings:
- v5_scale.py:53-57 BigWorld: `words` is a set; `vocab` (and so `idx`, the row of every word in F) is `sorted(...)`, independent of the
  hash; `cat` is a dict whose key ORDER follows the set (the byte difference of run 0). Every use of `cat` is a lookup `w.cat[x]` or
  `sorted(set(w.cat.values()))` (v2.py:24/131, v5_scale.py:178, v7_nested.py:81, v8_gates.py:33/154, v6_left.py:82/94,
  v22_transformer_categories.py:45 cat_id from the sorted list, camp.py:63): no id depends on the key order.
- camp.py:23-28 the same pattern: `function_words` and `vocab` sorted; `cat` key order follows the set, used only by lookup.
- v5_scale.py:120-160 split_templates / split_templates_rolecues: sets of transitions and cues used only for membership and difference
  (the printed lists are sorted); the template partition does not depend on order.
- v6_left.py:24 `releft` set: membership only.
- World draws (v5_scale.py BigWorld.event: `rng.choice(list(self.frames))`) iterate a dict built from the VERBS dict literal, in insertion
  order: hash-independent.
Conclusion: nothing in the reader's or the transformer's path assigns an id or draws at random from a hash-ordered iteration; the fixed
PYTHONHASHSEED is a safeguard, not a correction, and run 1a (which ran without it) needs no repetition on this account.
