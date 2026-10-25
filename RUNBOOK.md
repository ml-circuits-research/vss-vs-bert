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
| 1a | `./run.sh vss/run_reader_basic.py --seeds 5 --save_model` | new ids and depth 3 within seed variance of RESULTS 7c, bundle, learned, start closed (1.000 / 0.996; A 0.986, C 0.991) (corrected 2026-10-25, responses R5: 6c is the hand-gate row) |
| 1b | `./run.sh models/transformer_7b.py --variant cat --seeds 5` | within seed variance of RESULTS 7b-bis (new ids 0.253 ± 0.034, depth 3 0.393 ± 0.023) |
| 1c | `./run.sh models/transformer_7b.py --variant plain --seeds 5` | within seed variance of RESULTS 7b (0.215 ± 0.012, 0.390 ± 0.024) |
| 2a | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --seeds 5 --save_model` | the seen row ≥ 0.95 (otherwise under-trained: its other rows are not interpreted, INSTRUCTIONS table 3) |
| 2b | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer whole --seeds 5` | same |
| 3a | `./run.sh models/bert_basic.py --model <small> --tokenizer wordpiece --seeds 5` | same |
| 3b | `./run.sh models/bert_basic.py --model <small> --tokenizer wordpiece --scratch --seeds 5` | same |
| 3c | the derived rows below (section 1b), then JP 1 | each row's exit check in section 1b |
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

## 1b. Derived rows before JP 1 (responses R10, R11, R14, R17, R18, R20, R22; written 2026-10-25 before the first of them)

No grid for a derived row (R22 item 1): it inherits the configuration selected on seed 0 by the pre-registered row of the same model and
tokenizer condition, passed explicitly with `--lrs` / `--epochs` (taggers) or `--lrs` / `--dropouts` (7b). Inherited configurations:
BERT-base WordPiece (2a) lr 2e-5, 3 epochs; BERT-base whole (2b) lr 2e-5, 3 epochs; DistilBERT pretrained (3a) and from scratch (3b): the
configurations their logs print as chosen; 7b-bis (1b) lr 3e-4, dropout 0; 7b (1c) lr 1e-3, dropout 0; the reader has no grid. R10 item 3
(tie-break by validation cross-entropy) is therefore reported, not run: the pre-registered logs from 2b on print every grid point's
cross-entropy, and RESULTS says which point the tie-break would have chosen. Add-on heads share the encoder (R22 item 2): one run per model
and seed trains the tagger with the cue feature (BERT-base only, R14), the out-of-world class (R17 ii) and the three depth heads (R20 a /
b / c), losses summed with equal weights; each head is decoded as its own row. Concurrency (R22 item 3, Q17 default): `VSA_SLOTS=2` only
after one measurement of 1 run alone against 2 concurrent runs; below 1.3x the throughput, one slot. Cost rows of concurrent runs are
marked "concurrent (N runs)"; per-seed training times in the cost table come from runs that ran alone (2a, 2b, 3a, 3b).

Slots (responses R23, superseding the two-slot default of Q17). Per-process peak memory, measured once (2026-10-25): the main
repository's guard does NOT print torch.cuda.max_memory_reserved (R23 assumed it did), so the peak comes from the system: run 3a
(DistilBERT, batch 32) 2.2 GB GPU (nvidia-smi compute apps) + 2.0 GB host RSS peak (VmHWM); BERT-base (2a / 2b) 7.3-8.6 GB of system
memory in use during the runs against 2.4 GB idle (sysstat); the largest system-wide reading during any single run of the day is 12.7 GB.
Peak per process used for the formula: **13 GB** (the conservative upper bound). Formula: slots = min(6, rows waiting, floor(MemAvailable
after the current launches / 13 GB)), VSA_MEM_FRACTION = 0.9 / slots for every process, the 40 GB post-launch guard kept. With about
112 GB available that gives 6 slots and a fraction of 0.15 (18 GB of allocator per process, against at most a few GB used). Throughput
check at the chosen N: samples per second of N concurrent runs against one alone (the same script and configuration); keep N if the total
is >= 1.3x, else halve N and measure again. The energy of concurrent runs is reported per batch of runs (the sidecar samples the machine),
the per-seed cost rows only from runs that ran alone. From the first derived row on, models/bert_basic.py also prints its
torch.cuda.max_memory_reserved per seed (an additive line), and the peak above is checked against it.

| row | command | exit check |
|---|---|---|
| D1 (R11) | `./run.sh models/transformer_7b.py --variant cat --lrs 3e-4 --dropouts 0.0 --seeds 5 --save_model --tag_suffix _rerun` | per-role numbers within seed variance of 1b; then the one-to-one rule on the same models |
| D2 (R11) | the same with `--variant plain --lrs 1e-3` | the same against 1c |
| D3 (R17 i) | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --lrs 2e-5 --epochs 3 --seeds 5 --save_all --tag_suffix _saved` | reproduces 2a (per-role, per seed) |
| D4 (R17 i) | the same with `--tokenizer whole` | reproduces 2b |
| D5, D6 (R17 i) | the same for DistilBERT pretrained and `--scratch`, with the 3a / 3b configurations | reproduce 3a / 3b |
| D7-D10 (R10) | `--r10` with the inherited configuration: BERT-base WordPiece; BERT-base whole with `--whole_init pieces`; DistilBERT pretrained; DistilBERT `--scratch`, and `--scratch --pos_offset 10` | the seen row >= 0.95 |
| D11 (R14) | `./run.sh vss/run_reader_basic.py --gates_given cues --seeds 5 --save_model` | the seen row 1.000 |
| D12 (R14 + R17 ii + R20) | `./run.sh models/bert_basic.py --model bert-base-uncased --tokenizer wordpiece --lrs 2e-5 --epochs 3 --seeds 5 --save_all --cues given --oow_class --depth_head cls,reg,count` | the seen row >= 0.95 |
| D13 (R17 ii + R20) | the same for DistilBERT pretrained with the 3a configuration, without `--cues` | the seen row >= 0.95 |
| D14 (no training) | `./run.sh eval/r17_reader.py --seeds 5`; `--gates_given cues`; `--load transformer7b_cat_rerun`; `--load transformer7b_plain_rerun` | the retrained / loaded per-role rule equals the stored row |
| D15 (no training) | `./run.sh eval/r17_tagger.py --tag <row>` for D3-D6, D12, D13 | prints for every seed |

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

## 4. The real-data round (runs 8-12; INSTRUCTIONS_REAL_2026-10-24.md)

When: after judgment point 2, unless the reviewing session moves it earlier (recorded in JOURNAL.md). Nothing in this round needs a
constructed world. **No dataset file is ever committed**: downloads go to `data/real/raw/` (untracked). Only `data/real/checksums.json`
(pins written on first download) and `data/real/splits_<dataset>.json` (ids, seeds, rules) are committed.

Download step (DGX, once per source). `download.py` checks every file against `data/real/checksums.json` and stops on a mismatch.
MASSIVE and TOPv2 could not be fetched from the preparation machine, so their pins are written on the first download: commit
checksums.json right after it. The LICENSE the archive carries is printed: record it in data/real/README.md.
```
python3 data/real/download.py snips        # pinned already; must print 'ok' for 15 files
python3 data/real/download.py massive      # Amazon S3 tarball (all locales); first download pins the sha256
python3 data/real/download.py topv2        # TOPv2_Dataset.zip; first download pins the sha256; read the printed LICENSE
git add data/real/checksums.json && git commit -m "real data: sha256 pins of MASSIVE and TOPv2 (first download)"
```

| run | command(s) | exit check |
|---|---|---|
| 8a | `./run.sh data/real/make_real_splits.py snips` then `python3 tests/test_real_data.py --real snips` | `git diff --stat data/real/` shows no change (the committed split file was made by the same code on the same pinned files); test PASS |
| 8b | `./run.sh vss/run_reader_real.py --dataset snips --sizes 1000,5000,full --seeds 5 --test --save_model` | the table prints for every size; FULL on dev_rep at full ≥ 0.5, otherwise the reader is under-trained on this data and its other rows are not interpreted (INSTRUCTIONS_REAL R7) |
| 8b | `./run.sh models/bert_real.py --dataset snips --model bert-base-uncased --sizes 1000,5000,full --seeds 5 --test --save_model` | the table prints for every size, softmax and temperature rows |
| 8b | the same with `--model <small>` and with `--model <small> --scratch` | the same; the from-scratch row at full is the under-training check |
| 8c | `./run.sh vss/run_reader_real.py --dataset snips --condition heldout_templates --seeds 5`, the same with `--condition ood_intents --test`; and `models/bert_real.py` with the same two conditions for the three encoders | the held-out and out-of-domain rows (abstained / confidently mapped / AUROC) print |
| 9a | `./run.sh data/real/make_real_splits.py massive`; `python3 tests/test_real_data.py --real massive`; commit `data/real/splits_massive.json` | test PASS (annot_utt round trip exact on every line); counts written into data/real/README.md |
| 9b | `./run.sh vss/run_reader_real.py --dataset massive --sizes 1000,5000,full --seeds 5 --test --save_model`; `./run.sh models/bert_real.py --dataset massive --model bert-base-uncased --sizes 1000,5000,full --seeds 5 --test --save_model`; the small encoder, pretrained and `--scratch` | as 8b |
| 9c | conditions `heldout_templates` and `ood_intents` (with `--test`) for the four models on MASSIVE | as 8c |
| 10a | `./run.sh data/real/make_real_splits.py topv2 --domains navigation,event,reminder`; `python3 tests/test_real_data.py --real topv2`; commit `data/real/splits_topv2.json` | test PASS (parse round trip exact); the log prints the per-depth counts and whether the depth condition is supported; if not, it is recorded as such in RESULTS and run 10b is the base condition only |
| 10b | `./run.sh vss/run_reader_real.py --dataset topv2 --condition depth --seeds 5 --test` (the log must print both `[check]` lines: the wrapper with nesting off equal to the tool's readout on every item, and the wrapper exact on the compiled gold target states); `./run.sh models/bert_real.py --dataset topv2 --condition depth --model bert-base-uncased --seeds 5 --test`; the small encoder and `--scratch`; and the base condition (`--sizes full`) for the four models | the depth rows: FULL and nested intents on `depth2_unseen:hops_seen` |
| 11 | `./run.sh eval/determinism.py --real_dataset massive --real_reader logs/models/reader_real_massive_base_full_seed0 --real_tagger logs/models/bert_real_massive_base_bert-base-uncased_pre_full_seed0` | "DETERMINISTIC" for both; the cost rows are in the JSONs of 8b / 9b and in paper/tables/real_<dataset>.md |
| JP 3 | `python3 eval/compare.py` (writes paper/tables/real_snips.md, real_massive.md, real_topv2.md); RESULTS sections 8-11; STATUS rows; JOURNAL; current_status; commit; push | the multilingual round (run 12) is decided |
| 12 | per locale L (whitespace-segmented, for example de-DE fr-FR es-ES ro-RO ru-RU): `./run.sh data/real/make_real_splits.py massive --locale L`; `./run.sh vss/run_reader_real.py --dataset massive --locale L --sizes 1000,full --seeds 5 --test`; `./run.sh models/bert_real.py --dataset massive --locale L --model bert-base-multilingual-cased ...` and `--model xlm-roberta-base ...` | the multilingual rows; pre-download the two encoders as in section 0 |

`<small>` is the encoder recorded before run 3. Pre-download the two multilingual encoders before run 12, as in section 0:
`for m in ("bert-base-multilingual-cased", "xlm-roberta-base")`. The reader writes one world folder per condition and size under
`logs/worlds/` (untracked). It contains the converted examples, so keep it on the DGX only.

Smoke tests of the round (CPU; SNIPS is downloadable here; no result):
```
python3 data/real/download.py snips && python3 data/real/make_real_splits.py snips && python3 tests/test_real_data.py
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_real.py --dataset snips --sizes 500 --epochs 3 --seeds 1 --D 128 --n_cal 100 --max_eval 60 --k_clusters 16 --n_freq 20 --cost_n 10 --device cpu --save_model
VSA_MIN_AVAILABLE_GB=1 python3 models/bert_real.py --dataset snips --model tests/tiny-bert --scratch --sizes 300 --seeds 1 --lrs 1e-3 --epochs 2 --max_eval 60 --cost_n 10 --device cpu --save_model
VSA_MIN_AVAILABLE_GB=1 python3 eval/determinism.py --real_dataset snips --real_reader logs/models/reader_real_snips_base_500_seed0 --real_tagger logs/models/bert_real_snips_base_tests_tiny-bert_scratch_300_seed0 --device cpu
export REAL_DATA=$PWD/tests/real_fake REAL_SPLITS=$PWD/tests/real_fake/splits     # the fake MASSIVE / TOPv2 written by tests/test_real_data.py
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_real.py --dataset topv2 --condition depth --seeds 1 --epochs 15 --D 256 --n_cal 100 --k_clusters 8 --n_freq 20 --cost_n 5 --device cpu
VSA_MIN_AVAILABLE_GB=1 python3 models/bert_real.py --dataset topv2 --condition depth --model tests/tiny-bert --scratch --seeds 1 --lrs 1e-3 --epochs 15 --cost_n 5 --device cpu
VSA_MIN_AVAILABLE_GB=1 python3 vss/run_reader_real.py --dataset massive --condition ood_intents --seeds 2 --epochs 10 --D 128 --n_cal 80 --k_clusters 8 --n_freq 20 --cost_n 5 --device cpu --test
unset REAL_DATA REAL_SPLITS; python3 eval/compare.py --out /tmp/real_tables
```
Smoke numbers are not results: afterwards delete `logs/results_real_*.json`, `logs/results_determinism_real_*.json`, `logs/models/`
and `logs/worlds/`.
