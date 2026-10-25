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

## 2026-10-25 — incident: loss of network at 09:57, power cycle at 11:52 (no run affected)

Facts from the system journal and sysstat of the boot 07:35-11:51. At 09:57:31 NetworkManager removed the Wi-Fi profile in use
(fwv-46172; device wlP9s9, state change 'connection-removed') in the same second as a password SSH session from the LAN
(192.168.1.130, opened 07:36:01) closed. The interface stayed disconnected; it did not fall back to the other saved profile (axlg), so
the machine had no network from 09:57 on (Tailscale relay errors every minute until 11:51). The experimental session's worker lost the
API (DNS EAI_AGAIN) and stopped. Run 2a was not affected: it finished at 10:02:40 with its checklist, and its log and JSONs were committed
after the restart (RESULTS 2a). Memory was normal throughout (sysstat: about 2.4 GB used and 121 GB available from 10:10 to 11:50; at most
12.7 GB used during the runs). The journal has no OOM, no GPU Xid and no thermal event; the GPU never entered thermal slowdown. The
journal ends at 11:51:36 with no shutdown sequence: the machine was power-cycled at about 11:52 and came back at 11:53 on the axlg
profile, later fwv-46172. Cause of the profile removal: not established from the logs (it coincides with the close of that SSH
session). Work resumed from the committed state; no run was repeated.

## 2026-10-25 — decision point: the baseline form (responses R24, R25)

How the baseline choice slipped. The instructions started from the paper's existing comparison (a from-scratch transformer whose state is
decoded by the world's addresses) and took the token-tagging head as "BERT on this task", the JointBERT form of intent + slot filling,
without asking which form the literature uses for this task, which is question-based extraction (QA-SRL, He et al. 2015; MRC-style
extraction, Li et al. 2020). The independent review of R10 was briefed on the implementation of the baselines, not on the choice of
baseline, so it did not catch it. Decision (R24, R25): a BERT-MRC row family is added (the legend's questions as text, a span head, NULL
on [CLS]), bert-base-uncased before JP 1; the queue before JP 1 is cut to the pre-registered rows and the rows that support a claim;
the rows that existed to defend the tagger against objections MRC does not have are deferred to after JP 1, not dropped. The
pre-registered tagger rows stay and are all reported: the tagger is the JointBERT form that products use, and the difference between the
two BERT forms is itself a finding.

## 2026-10-25 — decision point: q-formalisation at equal size (responses R28)

The author decided that the paper is about q-formalisation (answers, "not given", margins, a residual, queries, in one pass, in a
model that fits a browser), so the main comparison moves to the reader's size in the MRC form: BERT-tiny / BERT-mini / BERT-small, each
pretrained, from scratch and with the kinds given. BERT-base and DistilBERT are kept as measured as reference rows; the only further
BERT-base row is MRC as the large reference. Run 3b, pre-registered, is not run (RESULTS '3b: not run'), because its question
(pretraining or form) is answered by the from-scratch variant at every small size. Tagger add-on rows on BERT-base (R10 / R14 / R17 /
R20) are deferred without date.

## 2026-10-25 — validation saturation (questions Q23, responses R34 / R35)

The validation set drawn from the training distribution cannot discriminate between configurations on this world (every grid point at
FULL 1.000, the loss below printed precision; 4.0% overlap with training, not leakage; a rule reaches 1.000 on it). This explains R10 item
3 (2a's selection fell on the first grid point) and makes every grid on this world a selection by noise. Decision: no grid for BERT-base
MRC and the equal-size block (fixed configuration by rule, the same for every model), one sensitivity table on seed 0 for BERT-mini; the
reader's own selection ran under the same saturated rule and is stated as such.

## 2026-10-25 16:20 — design-set dumps and the residual check (responses R38, R39, R41, R42; experimental session)

Seeds 5-9 (data/basic_seed5-9.json) were generated and committed at 16:06 (9f71308, sha256 in the message) before any dump existed; no
item of them has been opened. Design sources named as R38 / R41 ask: R38 / R39 and the coordinator's refinement (H1 gates / H2 routes /
H3 decoding) for the dumps; R41 for the residual check; R42 for the conditional explained-tokens row. Files opened so far: none of the
dumps (they are produced by queued runs). Disclosure: the CPU smoke tests of eval/dump_reader.py and eval/residual_length_check.py (on
the saved 1a seed-0 model, outputs deleted) printed two seed-0 lines that were read: the dump's count line (1,800 new-formulation items,
418 wrong role decisions, open gates . by that to) and the calibration check of seed 0 (Spearman of residual and token count -0.152;
mean residual at depth 0 / 1 / 2 0.410 / 0.455 / 0.432). The diagnosis patterns (eval/diagnose_reader.py, 175b473) were written after
that count line and before any per-item content was read; the R41 criterion is R41's own. The files opened for the analysis will be
listed here when it is written.

## 2026-10-25 17:15 — the design-set dumps were read (R38 / R39; experimental session)

Files opened (programmatically, aggregated; no item of seeds 5-9 was opened): results/dump_reader_basic_seed0-4.json,
results/dump_reader_basic_cues_seed0-4.json (by eval/diagnose_reader.py and by a cell breakdown of the same files: per-cell error counts,
swap / absent kinds, gold ranks, one example sentence per cell), results/diagnosis_reader_basic.json,
results/diagnosis_reader_basic_cues.json, logs/2026-10-25_1705_diagnose_reader_eb89dbb.txt; results/confident_errors_50_*.tsv written,
counted by kind only. From now on seeds 0-4 are the inspected design set. Calibration files of the R41 check:
logs/results_r41check_seed0-4.json (calibration rows only). The diagnosis is in questions.md (note of 17:15).

## 2026-10-25 22:40 — JP 1 (the declared set of R48, amended by R49-R58, measured; experimental session)

Measured since the handover (16:05): the equal-size MRC block (BERT-tiny / mini x pretrained / scratch / kinds; mini kinds 3 seeds, R51),
BERT-tiny on fresh seeds, the reader rows R37, R41 check (refutes R41), R42, R46, R54, R56 (route a, rejected by its own rule) and R58
(route b, not rejected), the planted sets, the confirmatory comparisons, the legend-growth rows, the cost table and the LLM reference.
Decision points of the day, each from a response: R48 / R51 cut the set (BERT-base MRC, BERT-small, the sensitivity grid, the 'all' rows,
BERT-mini growth, real data: not run); R46 / R54 / R56 / R58 changed the reader's decoder and routes as declared rows, designed on the
inspected seeds 0-4 (files opened: results/dump_reader_basic*_seed0-4.json, results/dump_reader_basic_cues*_seed0-4.json,
results/dump_reader_basic_cuesinit_seed0-4.json, results/diagnosis_*.json; the R41 check read logs/results_r41check_seed0-4.json,
calibration rows only), claimed on seeds 5-9. Design sources: R38-R43 (dumps, residual), R46 / R47 (decoder), R52 (R37's reading), R54
(adjacency), R55 / R56 / R58 (routes; reference implementations vss-demo a1f4564, f46b8d4, 434fa40 and sop/ at 561cbcb / 4a92afd). Tool
repository work recorded here, not run here: vss-demo's 'vorder' variant a was transcribed and run (R56), its variant b transcribed from
sop/v8_gates_vorder_ref.py and run (R58). Exposures, stated: the smoke tests printed seed-0 lines (JOURNAL 16:20); make_planted_sets
printed one example sentence per seed of the seeds 5-9 planted sets (read in the log). Run-infrastructure lessons: run.sh names logs by
the minute (two launches in one minute share a file: the 16:08 / 16:09 block logs; per-run stdout copies kept); a 30-min log-stall
watchdog killed a grid chain that printed nothing per epoch (progress lines added); untracked result files block run.sh launches.
JP 1 is written in RESULTS ('JP 1: synthesis'); the experiment stops here (R48). Next, outside this session: the paper; sop EQUAL
checks are reported in RESULTS as they finish.
