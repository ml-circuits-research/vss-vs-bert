# data/real: the real-data sources of runs 8-12 (INSTRUCTIONS_REAL_2026-10-24.md)

**No dataset file is in this repository.** This is the author's rule, whatever the licence. What is committed: `download.py` (fetch
and check sha256), `checksums.json` (the pinned hashes), `make_real_splits.py` (the frozen split conditions), and
`splits_<dataset>.json` (ids, seeds, rules and counts, no text). The loaders and the conversion are in `common/real_data.py`. Downloads
go to `data/real/raw/<dataset>/`, which is untracked and listed in .gitignore. `REAL_DATA=<dir>` overrides the folder. The fake files of
the tests are written by `tests/make_fake_real.py` into `tests/real_fake/`, also untracked.

## Sources and licences (verified 2026-10-24 from the primary source where it was reachable)

| dataset | what | licence | primary source | download |
|---|---|---|---|---|
| SNIPS 2017-06 custom intent engines | 7 intents, crowd-sourced queries; train_<intent>_full.json (about 2,000 per intent) and validate_<intent>.json (100 per intent) | **CC0 1.0 Universal**: repository LICENSE, read on 2026-10-24. The README asks every publication to cite Coucke et al. 2018, arXiv 1805.10190 | https://github.com/sonos/nlu-benchmark (LICENSE; folder 2017-06-custom-intent-engines) | `python3 data/real/download.py snips` (raw.githubusercontent.com; 15 files pinned by sha256 in checksums.json) |
| MASSIVE 1.1 | 51 locales (52 with Catalan in 1.1), 60 intents, 55 slot types; en-US 11,514 train / 2,033 dev / 2,974 test as published | **CC BY 4.0**: NOTICE.md of github.com/alexa/massive, read on 2026-10-24 ("The MASSIVE dataset is licensed under CC BY 4.0"). The English seed data is SLURP, also CC BY 4.0. Cite FitzGerald et al. 2022 (arXiv 2204.08582) and SLURP (Bastianelli et al. 2020) | https://github.com/alexa/massive (README, NOTICE.md) | `python3 data/real/download.py massive` on the DGX (Amazon S3; not reachable from the preparation machine, so the sha256 is pinned on first download: commit checksums.json afterwards) |
| TOPv2 | 8 domains of hierarchical intent / slot trees; navigation and event come from TOP (Gupta et al. 2018), with intents nested inside slots | **CC BY-SA**, as listed by Papers with Code for TOPv2 (https://paperswithcode.com/dataset/topv2). The paper (Chen et al. 2020, arXiv 2010.03546) states no licence. **Confirm from the LICENSE the archive carries** (download.py prints it) and record it here at run 10a | https://fb.me/TOPv2Dataset (paper footnote) -> dl.fbaipublicfiles.com/TOPv2/TOPv2_Dataset.zip | `python3 data/real/download.py topv2` on the DGX (sha256 pinned on first download) |
| MTOP | an alternative to TOPv2 (6 languages, flat and nested parses) | CC BY-SA 4.0 as commonly listed; primary source not reached | https://fb.me/mtop_dataset | not implemented (TOPv2 is used) |
| ATIS | real users (airline) | LDC licence (not free) | LDC93S4B and related | **not included**: no freely licensed copy is known |
| Spider | cross-domain text-to-SQL | CC BY-SA 4.0 | https://yale-lily.github.io/spider | **not implemented**: a later and harder SQL round |

## Counts

SNIPS (MEASURED here by `make_real_splits.py snips`; `splits_snips.json`):

| partition | utterances | intents | slot types | slots | items with a repeated slot type | spans dropped by the first-span rule | mean tokens |
|---|---|---|---|---|---|---|---|
| train | 13,084 | 7 | 39 | 33,922 | 58 | 59 | 9.28 |
| dev (100 per intent from train_full) | 700 | 7 | 39 | 1,826 | 3 | 3 | 9.19 |
| test (validate files) | 700 | 7 | 39 | 1,794 | 0 | 0 | 9.59 |

Loader notes for SNIPS: 83 entity boundaries fall inside a token (for example "playlist's"), and the whole token is taken. 3 pairs of
overlapping entities exist. 1 file is not valid UTF-8 and is decoded as Latin-1. 11 test utterances occur verbatim in train; the
benchmark is kept as published.

SNIPS conditions: base (train 13,084; dev calibration 350, report 350; new slot values: 96 dev-report and 207 test items; unseen
template: 174 and 402); heldout_templates (125 of 776 eligible templates held out; 1,065 held-out items; train 12,126); ood_intents
(GetWeather held out; train 11,184; out-of-domain 50 dev-report and 100 test items).

MASSIVE and TOPv2: counted at runs 9a and 10a by the same script, then written here. For TOPv2 the per-depth counts by partition are
in the split file (`conditions.depth.depth_counts_by_partition`). If depth 2 has fewer than 30 items the depth condition is marked
unsupported with its counts, and no unseen-depth split is made.

## Conversion rules (common/real_data.py)

- **Tokens.** The dataset's whitespace tokens, lowercased. Each is split by the main repository's tool tokenizer
  (tool/sop/legend.tokenize: a trailing . , ; : ! ? is split off) applied to a fixed point, so the reader tokenizes ' '.join(tokens)
  back to the same tokens; this is checked on every training row. A span covers the normalised tokens of the raw tokens it covered,
  minus trailing punctuation-only tokens, so 'p.m.' keeps the head 'p.m'.
- **SNIPS.** The 'data' chunks are joined into the text. An entity chunk covers the normalised pieces that overlap its characters. Ids:
  `snips:<train|validate>:<intent>:<index in the file>`.
- **MASSIVE.** `annot_utt` is parsed (`[type : value]`) into tokens and spans; the partitions are the dataset's. Ids: `massive:<id>`.
- **TOPv2.** The parse string (`[IN:X ... [SL:Y [IN:Z ...] ] ]`) gives the tokens and a tree; a slot holding an intent has a child
  tree. The low-resource `*_25spis` files are excluded. Default domains: navigation, event and reminder, the domains with nested
  intents (`--domains`). Ids: `topv2:<domain>:<train|eval|test>:<row>`; eval becomes dev.
- **Frame.** The head-token frame: {intent, <slot type>: the LAST token of the FIRST span of that type, or the nested frame}. The
  depth is the number of intents nested under the top one.
- **Round trips** (tests/test_real_data.py, before any run): MASSIVE annot_utt and TOP parse strings are rebuilt exactly from tokens
  and spans; SNIPS entity chunks match their spans except the boundaries inside a token, which are counted; spans -> BIO -> spans is
  exact; gold frames score 1.000.

## Splits

`make_real_splits.py <dataset>` writes `splits_<dataset>.json`. Its header docstring states every condition and its rule. Committing
the file freezes the splits: every model reads its rows through `common/real_data.load_condition`, which refuses ids missing from the
loaded data. The file contains ids and slot-type / intent names (the schema), never utterance text.
