# vss-vs-bert

A controlled comparison of the VSS reader (ml-circuits-research/vsa-semantic-space) with pretrained encoders (BERT) on frame reading:
accuracy, composition to an unseen depth, data efficiency, abstention, size and cost. The question, the models, the data, the splits,
the measures and the written interpretation of every outcome are fixed in `INSTRUCTIONS_2026-10-24.md` before any run. The rules of the
main repository apply (`AGENTS.md`). Intended as a separate paper.

## Setup

```
git clone --recurse-submodules https://github.com/ml-circuits-research/vss-vs-bert
cd vss-vs-bert && pip install -r requirements.txt            # plus the main repository's torch / numpy
python3 -c "from common import mainrepo; print(mainrepo.check_pinned())"   # must print e531283
```
The pretrained models are downloaded by `transformers` on first use (bert-base-uncased, distilbert-base-uncased or prajjwal1/bert-small;
the choice between the last two is made once and recorded in RESULTS.md). WikiSQL is read from the main repository's
`experiment/corp/wikisql/data/` on the DGX (not tracked); `WIKISQL_DATA=<dir>` overrides the path.

## Order of runs

The exact commands, the exit check of each run and what to write after it are in **RUNBOOK.md** (environment setup, pre-download of
the encoders, runs 0-7 with the two judgment points, smoke tests; section 4: the real-data round, runs 8-12, with the download step). In short (INSTRUCTIONS section 7; one change per run; `./run.sh` logs
each run under logs/ with the commit in the name):

| run | what | scripts |
|---|---|---|
| 0 | frozen splits of the basic world (committed) | `data/make_splits.py` |
| 1 | the reader and the paper's 7b / 7b-bis control, reproduced on the frozen splits | `vss/run_reader_basic.py`, `models/transformer_7b.py` |
| 2 | BERT-base pretrained, both tokenizer conditions | `models/bert_basic.py` |
| 3 | the small pretrained encoder and the same architecture from scratch | `models/bert_basic.py --scratch` |
| JP 1 | judgment point 1 | `eval/compare.py` |
| 4 | the reference world (10b-quinquies): frozen splits (committed), the reader, the taggers with a predicate head | `data/make_reference_splits.py`, `vss/run_reader_reference.py`, `models/bert_reference.py` |
| 5 | WikiSQL 1k / 5k / 56k on the same training pool and dev halves: the reader's row and the encoders' rows | `vss/run_reader_wikisql.py`, `models/bert_wikisql.py` |
| 6 | tables, abstention curves, the 50-item confident-error analysis | `eval/compare.py`, `eval/error_analysis.py` |
| 7 | determinism and cost | `eval/determinism.py` (cost is in every model's JSON) |
| JP 2 | judgment point 2 | |
| 8 | real data, SNIPS: frozen splits (committed, ids only), the reader through the sop tool, the encoders with BIO + intent heads | `data/real/make_real_splits.py`, `vss/run_reader_real.py`, `models/bert_real.py` |
| 9 | real data, MASSIVE en-US (downloaded on the DGX): the same models and conditions | the same |
| 10 | real data, TOPv2: nested intents, the depth split (train at depth <= 1, test at depth 2) | the same |
| 11 | determinism and cost of the real-data models | `eval/determinism.py --real_dataset` |
| JP 3 / 12 | judgment point 3; MASSIVE multilingual (mBERT, XLM-R) | `eval/compare.py` |

## What is where

- `common/mainrepo.py`: import path to the pinned main repository; `common/basic_world.py`: the basic world per seed exactly as the main
  scripts build it, the frozen splits, gold frames and tags; `common/decode.py`: decoding with the world's addresses on the frozen
  splits for any model with the main repository's interface; `common/report.py`: the common evaluation and printing (RESULTS 7b columns,
  calibration of rule 18, JSON per seed under logs/).
- `models/bert_basic.py`: pretrained or from-scratch encoder with the token-tagging head, both tokenizer conditions, the fixed grid,
  occupancy by free tags, cost. `models/transformer_7b.py`: the paper's control from the main repository's classes. `models/bert_wikisql.py`:
  selected column and aggregator with the table's columns as candidates, 1k / 5k / 56k, calibration and report halves of dev, test once.
- `vss/run_reader_basic.py`: the reader (v8_gates.V8, unmodified) on the frozen splits, with its cost. `common/reference_world.py`,
  `common/report_ref.py`, `vss/run_reader_reference.py`, `models/bert_reference.py`: run 4. `common/wikisql_data.py`: the training pool
  and dev halves every WikiSQL script uses (v12's). `vss/run_reader_wikisql.py`: the reader's WikiSQL row (main repository code, v12's
  setup reproduced). `eval/error_analysis.py`, `eval/determinism.py`: runs 6 and 7. `tests/`: a local tiny BERT config and a fake
  WikiSQL generator for CPU smoke tests.
- Real-data round (INSTRUCTIONS_REAL_2026-10-24.md): `data/real/` (download with sha256, checksums, the split maker and the frozen
  split files of ids; README with licences, counts and conversion rules; **no dataset file is committed**), `common/real_data.py`
  (loaders for SNIPS, MASSIVE, TOPv2; tokens, spans, trees, head-token frames, BIO, templates, conditions), `common/report_real.py`
  (the common evaluation of the round), `vss/run_reader_real.py` (the reader through the main repository's sop tool: a world folder per
  condition, sop.train.train_one, Reader.formalize; the intent as a closed-value role; the nested readout wrapper),
  `models/bert_real.py` (BIO + intent + nested heads, recursive), `tests/make_fake_real.py` and `tests/test_real_data.py` (fake MASSIVE /
  TOPv2, conversion round trips on gold).
- `eval/metrics.py`: the one scorer (per role, FULL, calibration with the written threshold rule, coverage curve, rule of three, ECE,
  checklist); `eval/compare.py`: tables and figures from logs/results_*.json only.
