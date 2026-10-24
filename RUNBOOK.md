# RUNBOOK — the DGX session, step by step

This file turns INSTRUCTIONS_2026-10-24.md section 7 into commands. It states no durations. Every run goes through `./run.sh` from the
repository root (it refuses an unclean tree except logs/ and *.md, takes the lock, checks free memory, applies the main repository's
guard, and writes `logs/<date>_<time>_<script>_<commit>.txt`). After every run: the exit check below, then RESULTS / STATUS / JOURNAL as
described at the end of this file. A run whose exit check fails stops the sequence; it is reported, not repeated to "improve" it (AGENTS
rule 4).

## 0. Environment (once)

```
git clone --recurse-submodules https://github.com/ml-circuits-research/vss-vs-bert && cd vss-vs-bert
git submodule update --init                     # third_party/vsa-semantic-space at e531283
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # torch must be the DGX's CUDA build: if pip pulls a CPU wheel, install torch first from the
                                                # index the main repository's README_DGX.md uses, then `pip install -r requirements.txt`
python3 -c "from common import mainrepo; print(mainrepo.check_pinned())"          # must print e531283
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi                                                                         # the GPU and its free memory
export HF_HOME=$PWD/.hf_cache                   # the Hugging Face cache inside the repository folder (ignored by git)
python3 - <<'EOF'                               # pre-download the three encoders and their tokenizers once
from transformers import AutoModel, AutoTokenizer
for m in ("bert-base-uncased", "distilbert-base-uncased", "prajjwal1/bert-small"):
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); print("cached", m)
EOF
ls third_party/vsa-semantic-space/experiment/corp/wikisql/data/    # train/dev/test .jsonl and .tables.jsonl must be there (untracked);
                                                                   # otherwise copy them from the main repository's DGX checkout, or
                                                                   # export WIKISQL_DATA=<dir with the six files>
```
`.hf_cache/` and `logs/models/` are in .gitignore. Keep `export HF_HOME=...` in every shell that runs a BERT script.

Decision before run 3 (record it in RESULTS.md before the run, with the reason): the small pretrained encoder, `distilbert-base-uncased`
or `prajjwal1/bert-small`.

## 1. The runs, in order

| run | command(s) | exit check |
|---|---|---|
| 0 | `./run.sh data/make_splits.py --seeds 5` | the log prints "tag->frame mapping exact" for seeds 0-4; `git diff --stat data/` shows no change (the committed splits were made by the same code) |
| 1a | `./run.sh vss/run_reader_basic.py --seeds 5 --save_model` | new ids and depth 3 within seed variance of RESULTS 6c (1.000 / 1.000) |
| 1b | `./run.sh models/transformer_7b.py --variant cat --seeds 5` | within seed variance of RESULTS 7b-bis (new ids 0.253 ± 0.034, depth 3 0.393 ± 0.023) |
| 1c | `./run.sh models/transformer_7b.py --variant plain --seeds 5` | within seed variance of RESULTS 7b (0.215 ± 0.012, 0.390 ± 0.024) |
| 2a | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --seeds 5 --save_model` | the seen row ≥ 0.95 (otherwise under-trained: its other rows are not interpreted, INSTRUCTIONS table 3) |
| 2b | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer whole --seeds 5` | same |
| 3a | `./run.sh models/bert_basic.py --model <small> --tokenizer wordpiece --seeds 5` | same |
| 3b | `./run.sh models/bert_basic.py --model <small> --tokenizer wordpiece --scratch --seeds 5` | same |
| JP 1 | `python3 eval/compare.py`; RESULTS sections 0-3; STATUS rows; JOURNAL entry; current_status.md; commit; push | the reviewing session decides whether run 4 onward proceeds as written |
| 4a | `./run.sh data/make_reference_splits.py --seeds 5` | "tag->frame mapping exact" for seeds 0-4; no diff in data/ |
| 4b | `./run.sh vss/run_reader_reference.py --seeds 5` | depth 3 UNSEEN roles FULL within seed variance of RESULTS 10b-quinquies (0.968 ± 0.020); seen and passives near 1.000 |
| 4c | `./run.sh models/bert_reference.py --model bert-base-uncased --tokenizer wordpiece --seeds 5` | seen FULL ≥ 0.95 |
| 4d | `./run.sh models/bert_reference.py --model <small> --tokenizer wordpiece --seeds 5` and the same with `--scratch` | seen FULL ≥ 0.95 |
| 5a | `./run.sh vss/run_reader_wikisql.py --sizes 1000,5000,56355 --seeds 5 --test` | selected column on the dev report half within seed variance of RESULTS 8d (0.632 / 0.760 / 0.816, whole dev) |
| 5b | `./run.sh models/bert_wikisql.py --model bert-base-uncased --sizes 1000,5000,56355 --seeds 5 --test` | the table prints for every size |
| 5c | `./run.sh models/bert_wikisql.py --model <small> --sizes 1000,5000,56355 --seeds 5 --test` and the same with `--scratch` | the table prints for every size; the from-scratch row at 56k is the under-training check of INSTRUCTIONS section 9 |
| 6a | `python3 eval/compare.py` | paper/tables/basic_world.md, reference_world.md, wikisql.md and the coverage figure are written |
| 6b | `python3 eval/error_analysis.py dump --size 56355 --tags reader,bert-base-uncased_pre` | two TSV files under paper/error_analysis/ with up to 50 rows each; a person fills the 'kind' column, then `python3 eval/error_analysis.py summarize paper/error_analysis/*.tsv` |
| 7 | `./run.sh eval/determinism.py --reader logs/models/reader_basic_seed0.pt --tagger logs/models/bert_bert-base-uncased_pre_wordpiece_seed0` | "DETERMINISTIC" for both; the cost rows are in the JSON of runs 1a, 2a-3b, 4b-4d, 5a-5c and in paper/tables/ |
| JP 2 | RESULTS, STATUS, JOURNAL, current_status; paper outline in paper/; commit; push | the second round (negation and operator worlds) is decided |

`<small>` is the encoder recorded before run 3. The grids are fixed in the scripts (INSTRUCTIONS section 4) and are not extended; a
script that needs another setting is a deviation and is recorded in RESULTS.md before the run that uses it.

## 2. After every run

- **RESULTS.md**: a numbered section (`## <run>. <title> (<script>; logs/<log name>)`) with the exact command, the commit, the table the
  script printed (mean ± std over seeds with the per-seed list), the calibration block of rule 18, the label (MEASURED), and two or three
  sentences of reading. When a number falls in the in-between band of INSTRUCTIONS table 3, both readings are written.
- **STATUS.md**: the row of the axis and model gets the label, the log and the number; nothing else changes.
- **JOURNAL.md** and **current_status.md**: only at the judgment points (JP 1, JP 2), as in the main repository.
- **Commit** the code first (run.sh requires it), the log and the documents after the run; push at the judgment points.
- If a run fails (exception, out of memory, the guard stops it): the log stays, the failure is written in RESULTS.md with its cause,
  and the fix is a separate commit before the run is started again.

## 3. Smoke tests (CPU, no download)

```
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_basic.py --seeds 1 --N 300 --device cpu --cost_n 20 --save_model
VSA_MIN_AVAILABLE_GB=1 python3 models/bert_basic.py --model tests/tiny-bert --scratch --seeds 1 --N 200 --device cpu --lrs 1e-3 --epochs 1 --cost_n 10 --save_model
VSA_MIN_AVAILABLE_GB=1 python3 eval/determinism.py --reader logs/models/reader_basic_seed0.pt --tagger logs/models/bert_tests_tiny-bert_scratch_wordpiece_seed0 --device cpu
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_reference.py --seeds 1 --N 600 --device cpu
VSA_MIN_AVAILABLE_GB=1 python3 models/bert_reference.py --model tests/tiny-bert --scratch --seeds 1 --N 300 --device cpu --lrs 1e-3 --epochs 2 --cost_n 20
python3 tests/make_fake_wikisql.py         # writes tests/wikisql_fake/data (untracked)
WIKISQL_DATA=tests/wikisql_fake/data VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_wikisql.py --sizes 50,100 --epochs 3,2 --seeds 1 --D 128 --n_val 10 --k_clusters 4 --n_freq 5 --device cpu --test
WIKISQL_DATA=tests/wikisql_fake/data VSA_MIN_AVAILABLE_GB=1 python3 models/bert_wikisql.py --model tests/tiny-bert --scratch --sizes 50 --seeds 1 --device cpu --lrs 1e-3 --epochs 2 --n_val 10 --n_dev 40 --test
WIKISQL_DATA=tests/wikisql_fake/data python3 eval/error_analysis.py dump --size 50 --tags reader,tests_tiny-bert_scratch --n_dev 40 --out /tmp/ea
```
Smoke numbers are not results: delete `logs/results_*.json` and `logs/models/` after a smoke test, before the real runs, so that
eval/compare.py reads only real runs.
