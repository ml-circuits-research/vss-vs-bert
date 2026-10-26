# Questions for the author

Open questions of the experimental session. The author answers in `responses.md` (written from another session; this session never
edits it). A question is removed from here once its answer is accepted; otherwise it is clarified here. Each question states the default
this session follows until an answer arrives, and the run before which the answer is needed.

(No open question. Q1-Q28 were answered by R1, R2, R5, R7, R11, R12, R13, R18, R20, R23, R29, R31, R32, R33, R34, R35, R39, R40, R45, R47, R50, R59 (Q29), R64 (Q30), R66 (Q31), R69 (Q32), R79 (Q33), R80 (Q34) and R81 (Q35) of responses.md, 2026-10-25, and accepted; the transparency note on 33dc482 was read and accepted in R40; the note on the seeds 5-9 data was accepted in R44; the H1 / H2 / H3 diagnosis note was accepted in R47.)

## Q36 (2026-10-26 15:40, experimental session): R81 met the dispatch bar (0.966); R78 (1)/(2) run on it as R81 says; two further proposals need your declaration

R81 (training logs 2026-10-26_1311.._1316_run_reader_real_a9855d4.txt, sop @ 82585ea; readings _1450.._1455_r80_dispatch_ba2a6a4.txt; RESULTS,
commit 4e98701). Test; R80 on the R79 models in brackets:
- anchor-percentile dispatch **0.966 +- 0.003** [0.963 0.964 0.970] (0.863); new slot values 0.957, unseen templates 0.949; references:
  one world 0.977, BERT-tiny 0.985.
- span F1 0.824 (0.802; typer alone 0.886); head-frame exact 0.565.
- AUROC held-out vs in-domain: residual 0.941 (0.849; one world 0.933); anchor percentile 0.892 (0.622).
- raw-residual dispatch fell to 0.386 (0.731): the negatives move foreign utterances away from the anchor, not from the slot readout.
- rejection on ood_intents test: residual line 0.030 held-out at 0.001 false; **anchor line 0.120 +- 0.037 at 0.030 false** (R80 0.280 at 0.038).
- against R81: dispatch >= 0.95 met; span F1 >= 0.802 met; anchor-line rejection > 0.28 **not met** (fell to 0.12 although both AUROCs rose).
- training time 5.3x for 2x rows on a shared machine (load ~120): not a cost number.
By R81 ("if it reaches 0.95 the configuration is kept for the demo and R78 (1) runs on it") R78 (1) now runs on the *_r81 worlds, then
R78 (2), each with R83's paragraph in its RESULTS declaration. Two proposals beyond that, not started:
1. **presence_query (R82 rule 1) on the R81 worlds.** Fails: chosen-frame abstention ~0.61 (R79). Cause: a contract rule of the reader
   (absent optional roles under their presence tau lower the frame margin). Change: the decoder only (config key), no retraining. Expected
   to move: chosen-frame abstention and frame coverage; dispatch and span F1 should not drop. R82 rule 3 as a separate line.
2. **The anchor-line rejection.** Fails: 0.12 held-out at 0.030 false, while the anchor AUROC is 0.892. Cause (hypothesis): the fixed
   "every world's anchor percentile < 2%" line is a fixed quantile, not a threshold chosen on the dev calibration half; with negatives the
   in-domain anchor distribution moved. Change: the threshold on the anchor percentile chosen on the dev calibration half at 2% false
   rejection (as every other refusal threshold of the paper), no retraining. Expected to move: held-out rejection toward what an AUROC of
   0.89 allows at 2% false rejection.
**Default until answered:** R78 (1) and (2) run; proposals 1 and 2 do not.
