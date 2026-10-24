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

## Order of runs (INSTRUCTIONS section 7; one change per run; `./run.sh` logs each run under logs/ with the commit in the name)

| run | command | exit check |
|---|---|---|
| 0 | `./run.sh data/make_splits.py --seeds 5` | tag -> frame mapping exact on every seed (printed); `data/basic_seed*.json` committed |
| 1 | `./run.sh vss/run_reader_basic.py --seeds 5` and `./run.sh models/transformer_7b.py --variant cat --seeds 5` (and `--variant plain`) | within seed variance of RESULTS 6c / 7b / 7b-bis; otherwise stop and report |
| 2 | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --seeds 5` and `--tokenizer whole` | Table 3 rows: seen, new identities, unseen depth |
| 3 | `./run.sh models/bert_basic.py --model <small> --seeds 5` and `--scratch` | the pretraining difference isolated |
| judgment point 1 | STATUS, RESULTS, JOURNAL, current_status; push | |
| 4 | reference world (to be added on the same pattern: `common/reference_world.py`, frozen splits, the three encoders) | Table 3 rows |
| 5 | `./run.sh models/bert_wikisql.py --model bert-base-uncased --sizes 1000,5000,56355 --seeds 5 --test` (and the small encoder, and `--scratch`) | the data-efficiency table |
| 6 | abstention curves and the 50-item error analysis: `python3 eval/compare.py`, plus the per-item dump of the reader's and BERT's confident errors | the abstention rows and the error table |
| 7 | cost (printed by `models/bert_basic.py`; the reader's from the main repository's tool/README) and determinism (two runs, identical frames) | the cost table |
| judgment point 2 | STATUS, RESULTS, JOURNAL; paper outline in paper/ | |

Smoke tests on CPU (no download; a local tiny BERT config under tests/):
```
VSA_MIN_AVAILABLE_GB=1 python3 data/make_splits.py --seeds 1 --D 256
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_basic.py --seeds 1 --N 600 --D 256 --device cpu
VSA_MIN_AVAILABLE_GB=1 python3 models/transformer_7b.py --seeds 1 --N 400 --epochs 1 --D 256 --device cpu --lrs 1e-3 --dropouts 0.0
VSA_MIN_AVAILABLE_GB=1 python3 models/bert_basic.py --model tests/tiny-bert --scratch --seeds 1 --N 300 --D 256 --device cpu --lrs 1e-3 --epochs 2
```
(`--D 256` smoke splits must be regenerated with `--D 1024` before the real runs; the scripts assert the D of the splits.)

## What is where

- `common/mainrepo.py`: import path to the pinned main repository; `common/basic_world.py`: the basic world per seed exactly as the main
  scripts build it, the frozen splits, gold frames and tags; `common/decode.py`: decoding with the world's addresses on the frozen
  splits for any model with the main repository's interface; `common/report.py`: the common evaluation and printing (RESULTS 7b columns,
  calibration of rule 18, JSON per seed under logs/).
- `models/bert_basic.py`: pretrained or from-scratch encoder with the token-tagging head, both tokenizer conditions, the fixed grid,
  occupancy by free tags, cost. `models/transformer_7b.py`: the paper's control from the main repository's classes. `models/bert_wikisql.py`:
  selected column and aggregator with the table's columns as candidates, 1k / 5k / 56k, calibration and report halves of dev, test once.
- `vss/run_reader_basic.py`: the reader (v8_gates.V8, unmodified) on the frozen splits.
- `eval/metrics.py`: the one scorer (per role, FULL, calibration with the written threshold rule, coverage curve, rule of three, ECE,
  checklist); `eval/compare.py`: tables and figures from logs/results_*.json only.
