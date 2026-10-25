# The real-data round: real user commands annotated with frames

_AXIOLOGIC RESEARCH · INSTRUCTIONS FOR THE EXPERIMENTAL SESSION · ADDENDUM TO INSTRUCTIONS_2026-10-24.md_

**Version:** 1.0, 24 October 2026. Written before any run of this round; it fixes the question, the data, the splits, the measures and
the written interpretation of every outcome. The rules of INSTRUCTIONS_2026-10-24.md and of AGENTS.md apply unchanged: criteria before
runs; one change per run; every number labelled BUILT / MEASURED / PLANNED and pointing to a log whose name carries the commit;
thresholds never set on reported rows (rule 25); five seeds, mean ± std with the per-seed list (rule 27); the checklist printed at the
end of every script; everything on disk in English; no duration estimates anywhere. A further rule for this round, from the author: no
third-party data is committed to the repository, whatever its licence. Only download scripts with sha256 checks, loaders, conversion
code, split definitions made of ids, seeds and rules, and small fake files written for the tests are committed.

**When.** Runs 8 onward, after judgment point 2 of the main instructions. The reviewing session may move the round earlier, for example
right after run 5, because the round needs no constructed world. The decision is recorded in JOURNAL.md at the judgment point that takes it.

## R1. The question

The comparison of the main instructions is made on constructed worlds and on WikiSQL. The author asks whether it can go beyond WikiSQL to
something closer to the product. The product is a voice or text command interpreter: a sentence comes in and a frame goes out, an event
with its roles. The closest public analogue is spoken-language-understanding data: real user utterances annotated with an intent (the
event) and slots (the roles). This round runs the same comparison there, on three sources:

1. **MASSIVE** (FitzGerald et al. 2022; Amazon; CC BY 4.0). en-US: 11.5k train, 2k dev and 3k test utterances, 60 intents, 55 slot
   types, crowd-sourced Alexa-style commands. 51 languages give a second round on the multilingual axis: the reader, which has no
   pretraining, against mBERT and XLM-R.
2. **SNIPS** (2017-06 custom intent engines, sonos/nlu-benchmark; CC0 1.0). 7 intents and 39 slot types, 13,084 train (after the dev
   draw), 700 dev and 700 test. It is downloadable on the preparation machine and is the end-to-end smoke test of the round.
3. **TOPv2** (Chen et al. 2020; Facebook; CC BY-SA as listed, to be confirmed from the archive's own LICENSE). Hierarchical
   intent / slot trees with intents nested inside slots ("directions to [the event I have tonight]"). This is the real-data version of
   the paper's composition axis.

ATIS (real users, LDC licence) is not included: no freely licensed copy is known. Spider (cross-domain text-to-SQL) is a later and
harder SQL round, and is not implemented here.

## R2. Hypotheses, written before the runs

- **Pretraining helps on real English.** BERT-base pretrained is expected above the reader on every in-distribution axis of MASSIVE
  and SNIPS: real utterances have synonyms, paraphrases and open-vocabulary slot values that a model that has read the language
  handles and a 1-10 MB reader trained on the task data only does not. The questions are **by how much**, **at what training size**
  (1k / 5k / full) and **at what cost** (parameters, bytes, CPU milliseconds per sentence).
- **Abstention on what was never trained.** Two kinds of input are never in training: **out-of-domain intents**, held out entirely,
  and **new slot values**, head tokens never seen in training. On these, the experiment measures how often each model **abstains**
  (a margin below the threshold set on in-domain calibration rows) and how often it **maps the input confidently to a wrong frame**.
  A softmax tagger is expected to map out-of-domain commands confidently to a known intent more often than its in-domain calibration
  suggests. Whether the reader's margins do better is the open question; it is measured, not assumed.
- **Nested depth on real data.** On TOPv2, trained at depth ≤ 1 and tested at depth 2, the constructed-world result (the reader
  composes an unseen depth whose hops are seen; same-input neural models reproduce the deepest configuration they have seen) may or may
  not carry over. Real trees are fewer, noisier and lexically varied. The question is whether the depth-2 FULL of each model behaves as
  in the constructed worlds; it is answered on the subset whose every intent and slot type occurs in training (rule 29).
- **The success criterion is the same as in the main instructions.** The comparison must be complete, controlled and reproducible,
  with a number and an interval on every axis. A round where BERT wins every in-distribution axis is a publishable round if it is exact
  about where, by how much and at what cost.

## R3. The task, the frame and the scorer

The frame is {event: intent, roles: {slot type: value}}. Slot values are multi-word. The reader deposits one token identity per role:
the tool reads a multi-word value by its LAST token. So the **primary metric** is the **head-token frame**: the intent, plus, for each
slot type, the last token of its first span. Every model is mapped to it by the same rule; BERT's tagged spans are reduced to heads in
the same way. Scored, by one function (eval/metrics.score_real) for every model:

- FULL frame (the intent and every head exact; absent roles absent; nested frames exact on TOPv2);
- intent accuracy; slot-head accuracy on the gold slots; precision of the filled slots; nested-intent accuracy (TOPv2);
- the occupancy cases: present read present, absent read absent, false fills per utterance;
- per-role accuracy on the utterances where the role is in the gold or in the reading;
- calibration: confident-wrong and coverage at the written threshold rule (the smallest tau with confident-wrong ≤ 0.05 on the dev
  CALIBRATION half, separately for the intent, the slot and the frame decisions), the accuracy-versus-coverage curve, the rank-mapped
  ECE of the intent.

**Secondary:** exact span F1, the standard slot F1 of these benchmarks. For the BERT tagger it uses the spans it tags. For the reader
it uses a stated, deterministic span rule fitted only on training data, and is labelled as a rule, not learned by the reader: from the
head token, extend left while the token's training rate of being a non-final token of a span of that role exceeds 0.5. The rule applied
to the GOLD heads is printed beside it as its ceiling. On TOPv2 the standard metric is the exact match of the tree, which is FULL here;
span F1 is not reported there.

**Repeated slot types** in one utterance: the first span by position is kept. The tool's set roles take their candidates from a
sentence table and do not apply. The share of repeats is reported by data/real/make_real_splits.py; on SNIPS it is 58 of 13,084
training utterances.

**Conversion tested on gold before any run** (tests/test_real_data.py): annotation -> (tokens, spans) -> annotation round trip
(MASSIVE annot_utt and TOP parse strings exact; SNIPS entity chunks exact except the boundaries inside a token, counted); spans ->
BIO -> spans exact; gold frames score 1.000.

## R4. Models

The models of the main instructions, Table 1, on the same rows:

- **VSS reader** through the main repository's tool (tool/sop), imported unmodified. vss/run_reader_real.py writes each condition as a
  world folder and trains it with sop.train.train_one; the readout is sop.infer.Reader.formalize. Not tuned: D 1024, batch 32, lr 0.02,
  length buckets, epochs 60 / 20 / 10 at 1k / 5k / full, stated here before the runs. What the tool cannot express, and what was done
  about it, is in R6.
- **BERT-base pretrained** (bert-base-uncased), **the small pretrained encoder** chosen before run 3, and **the same architecture from
  scratch**. They use models/bert_real.py: BIO tags over the slot types, the intent from [CLS], and a per-word nested head, applied
  recursively to nested spans with the same weights at every depth. Grid and selection are those of section 4 of the main instructions:
  seed 0 only, FULL head frame on the dev calibration half. Margins come from softmax, and from softmax with temperature scaling fitted on
  the calibration half. Both rows are reported.
- **Multilingual, second round (MASSIVE):** bert-base-multilingual-cased and xlm-roberta-base against the reader on locales with
  whitespace word boundaries (`--locale`, for example de-DE, fr-FR, es-ES, ro-RO, ru-RU). Locales written without spaces between words
  (ja-JP, zh-CN, zh-TW, th-TH, km-KH, my-MM) need a segmenter the reader does not have, and are not run.

## R5. Splits (data/real/splits_<dataset>.json; ids, seeds and rules only; written before training)

| condition | content | what it measures |
|---|---|---|
| base | the dataset's train / dev / test (SNIPS: test = validate files, dev = 100 per intent from train_full by rng(0)); dev even / odd = calibration / report; test once | in-distribution reading |
| base, sizes | 1k / 5k / full: prefixes of the training ids permuted by rng(0) | data efficiency |
| base, new_slot_values | dev-report / test items with at least one head token never in the training utterances, scored on those heads | new identities on real data |
| base, unseen_template | dev-report / test items whose delexicalised template (top-level slots replaced by their type) never occurs in train | natural held-out formulations |
| heldout_templates | templates with ≥ 3 items held out by hash (share 0.15), removed from train and calibration; scored on all their items | held-out formulations, retrained |
| ood_intents | 6 (MASSIVE), 1 (SNIPS) or 3 (TOPv2) intents with ≥ 50 training items, chosen by hash, removed entirely from train and calibration | out-of-domain rejection |
| depth (TOPv2) | train and calibration at depth ≤ 1; depth 2 of every partition never trained; the hops-seen subset (rule 29); unsupported, and said so, if depth 2 has fewer than 30 items | composition on real data |

## R6. What the tool cannot express, and the smallest wrapper (vss/run_reader_real.py)

- **The intent as the event.** The legend requires verbs for every event. The tool's event decoder reads the event either from a verb
  token whose lemma the legend lists (predicate_role) or by a lexical lookup. An intent has no verb token: "wake me up at five" is
  alarm_set. So the intent is a role with a CLOSED value set (legend reader.closed_values, as WikiSQL's aggregate in the main
  repository), read from the state by cleanup over the intent symbols. The legend has one event, 'command', whose required verb is
  never used.
- **Candidates.** The tool's default candidates for a role are every catalog word of the role's kinds. With distributional kinds over a
  real vocabulary that is thousands of words per kind, and a role could be filled by a word the utterance does not contain. The
  candidates are therefore the utterance's own tokens (formalize's `filt`), the output space of a tagger.
- **Nested intents.** The decoder does not read nested event wires (tool README, limits of v1), although they are compiled into the
  targets. A readout wrapper uses the Reader's own primitives: after unbinding a slot role, the nested intent is read like the top one.
  The slot is nested iff that cleanup's ratio to the structure score reaches a threshold FITTED on the dev calibration half. Nested
  roles reuse the top-level presence thresholds. The wrapper with nesting disabled must reproduce the tool's readout; this is checked
  and printed.
- **A role whose training fillers are all nested** would make the tool's schema treat it as an entity role without head kinds. The
  schema entry is set to a word role before training (printed).
- **Kinds** are given by the WikiSQL recipe of the main repository: the 40 most frequent training tokens as their own kinds, and 48
  skip-gram clusters on the training utterances only. Words never seen in training keep their untrained vector, which gives a cluster
  independent of any label (v12's behaviour; `--unseen_kind unknown` is the alternative). No gold slot label defines a kind.

## R7. Axes and the written interpretation

| axis | measure | interpretation written in advance |
|---|---|---|
| in-distribution (dev report, test once) | FULL head frame, intent, slot heads | BERT pretrained expected above the reader; the gap and its interval are the result. A model below 0.5 FULL on SNIPS full is under-trained and its other rows are not interpreted |
| data efficiency | FULL at 1k / 5k / full | the size at which each model reaches 0.8 FULL on dev report is reported; if the reader reaches it at a smaller size than the from-scratch encoder, data efficiency is said to favour the reader's architecture; if only the pretrained encoder reaches it, the advantage is said to come from pretraining |
| new slot values | head accuracy on new heads | above 0.8 means the role is read from the context; below 0.3 means the model needs the value seen; reported as is in between, with both readings |
| held-out formulations | FULL on heldout_templates and unseen_template | the drop from seen to held-out templates is the measure of formulation generalisation, per model |
| out-of-domain intents | share abstained at the in-domain tau; share confidently mapped to a known intent; AUROC of the intent margin, in-domain against out-of-domain | the model with the higher abstention at equal in-domain coverage is said to reject better; an AUROC near 0.5 means its margin does not see the domain shift |
| abstention in-domain | frame coverage and confident-wrong at the written rule; softmax with and without temperature | the model with the lower confident-wrong at equal coverage abstains better |
| nested depth (TOPv2) | FULL and nested-intent accuracy at depth 2 (hops seen) | above 0.8 for the reader and below 0.5 for the encoders would carry the constructed-world result to real data; both above 0.8 would mean real depth-2 trees are within reach of a recursive tagger; reported with both readings in between |
| multilingual (round 2) | the same axes per locale | pretraining on 100 languages expected to help more where training data is small |
| size and cost | parameters, MB, CPU ms per sentence | reported; the ratio is part of every claim |
| determinism | frames identical across fresh loads | expected for both, verified (run 11) |

## R8. Runs (continuing the numbering of the main instructions)

| run | content | exit check |
|---|---|---|
| 8a | SNIPS: download (pinned sha256), splits (committed), conversion tests | tests/test_real_data.py PASS; the split file unchanged by the rerun |
| 8b | SNIPS base 1k / 5k / full: reader; BERT-base; small encoder; from scratch | the in-distribution table of R7; the under-training check |
| 8c | SNIPS heldout_templates and ood_intents: the four models | the held-out and out-of-domain rows |
| 9a | MASSIVE en-US: download (sha256 pinned on first download, checksums.json committed), splits (committed), conversion tests | as 8a |
| 9b-9c | MASSIVE base (sizes, test once), heldout_templates, ood_intents: the four models | the R7 rows |
| 10a | TOPv2 (navigation, event, reminder): download, splits with the per-depth counts, conversion tests | as 8a; the depth condition supported or marked unsupported with its counts |
| 10b | TOPv2 base and depth: reader and encoders | the depth rows of R7 |
| 11 | determinism and cost of the real-data models (saved with --save_model) | DETERMINISTIC for both |
| JP 3 | STATUS, RESULTS, JOURNAL, current_status; tables (eval/compare.py -> paper/tables/real_<dataset>.md); push | the multilingual round (12) decided |
| 12 | MASSIVE multilingual: the reader, mBERT, XLM-R on the chosen locales | the multilingual rows |

The exact commands and exit checks are in RUNBOOK.md section 4.

## R9. Known risks

- **Annotation noise.** MASSIVE and SNIPS contain annotation errors. They hit every model equally; the confident-error analysis of
  the main instructions can be repeated on 50 items per model.
- **Train / test overlap in text.** 11 SNIPS test utterances occur verbatim in train. It is the benchmark as published and is not
  altered; the count is reported.
- **The reader's kinds** are distributional and coarse (88 kinds); a real tagger of word kinds is not part of the tool. This is a
  limit of the reader as a product. It is measured here and not repaired here.
- **Compute.** The tool builds targets for every training step (steps × D complex numbers). MASSIVE full at 10 epochs and D 1024 is
  about 115k × 1024 complex targets, which fits the DGX. The memory curve is in every log (rule 21).

---

**Note added 2026-10-25 (responses R27 item 5; the text above is unchanged).** On SNIPS, MASSIVE and TOPv2 the reader's word kinds come
from skip-gram clusters on the training pool: they are LEARNED, unsupervised, not given as on the constructed worlds (where kinds.tsv is a
given, rule 20). Every real-data row of RESULTS states this regime.
