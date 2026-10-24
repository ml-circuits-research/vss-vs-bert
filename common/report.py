"""report.py: the common evaluation of one trained model on the frozen splits of one seed, and the common printing over seeds.
Every model script calls evaluate(model_name, seed, preds_by_axis, split) and, at the end, summarize(rows). The output has the columns
of RESULTS 7b (so that the numbers line up with the paper's tables) plus the calibration block of rule 18 and a JSON dump per seed in
logs/results_<script>_<seed>.json for eval/compare.py."""
import json, os
import numpy as np
from . import basic_world as bwm, mainrepo
from eval import metrics

AXES = ["seen", "new_ids", "new_form_seen_trans", "new_form_unseen_trans", "new_form_role_cues_seen", "depth1", "depth2_seen", "depth3_unseen"]
COLS = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "new form., ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]

def evaluate(name, seed, preds, split, tag, max_cw=0.02):
    """preds: axis -> list of predictions (incl. 'calibration'); returns the row dict and prints it"""
    row = {"model": name, "seed": seed, "axes": {}, "per_role_depth3": {}, "calibration": {}}
    for axis in AXES:
        items = split["axes"][axis]
        if not items: row["axes"][axis] = None; continue
        d = items[0]["depth"]; row["axes"][axis] = metrics.score_axis(items, preds[axis], bwm.roles_at(d))
    row["per_role_depth3"] = row["axes"]["depth3_unseen"]["per_role"]
    # calibration: tau on the calibration rows by the written rule; reported on new_ids and depth3_unseen (the paper's abstention axes)
    cal_dec = []
    for it, p in zip(split["axes"]["calibration"], preds["calibration"]):
        cal_dec += metrics.role_decisions([it], [p], bwm.roles_at(it["depth"]))
    tau = metrics.set_tau(cal_dec, max_cw); row["calibration"]["tau"] = tau; row["calibration"]["rule"] = f"smallest tau with confident-wrong <= {max_cw} on calibration rows"
    for axis in ("new_ids", "depth3_unseen", "seen"):
        items = split["axes"][axis]; dec = metrics.role_decisions(items, preds[axis], bwm.roles_at(items[0]["depth"]))
        row["calibration"][axis] = metrics.calibration(dec, tau); row["calibration"][axis]["curve"] = metrics.coverage_curve(dec); row["calibration"][axis]["ece_rank"] = metrics.expected_calibration_error(dec)
    a = row["axes"]
    print(f"  seed {seed} {name:34s} " + " | ".join(f"{a[x]['mean_role']:.2f} (m {a[x]['margin'] and np.mean(list(a[x]['margin'].values())):.2f})" if a[x] else " nan " for x in AXES)
          + "  || depth 3 per role: " + " ".join(f"{k} {v:.2f}" for k, v in row["per_role_depth3"].items())
          + f"  || FULL depth3 {a['depth3_unseen']['full']:.3f} | tau {tau:.3f}: new ids cov {row['calibration']['new_ids']['coverage']:.2f} cw {row['calibration']['new_ids']['confident_wrong']:.3f}; depth3 cov {row['calibration']['depth3_unseen']['coverage']:.2f} cw {row['calibration']['depth3_unseen']['confident_wrong']:.3f}", flush=True)
    os.makedirs(os.path.join(mainrepo.ROOT, "logs"), exist_ok=True)
    with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
    return row

def summarize(rows, name):
    """mean ± std over seeds with the per-seed list (rule 27), columns of RESULTS 7b"""
    n = len(rows); print(f"\n== {name}: MEANS over {n} seeds (mean role accuracy; per-seed list) ==  columns: " + " | ".join(COLS))
    cells = []
    for axis in AXES:
        v = [r["axes"][axis]["mean_role"] for r in rows if r["axes"][axis]]
        cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.2f}' for x in v)}]" if v else "nan")
    print("  " + " | ".join(cells))
    v = [r["axes"]["depth3_unseen"]["full"] for r in rows]; print(f"  FULL frame at depth 3 (UNSEEN): {np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]")
    print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3")
    print("  " + " | ".join(f"{np.mean([r['per_role_depth3'][k] for r in rows]):.3f}±{np.std([r['per_role_depth3'][k] for r in rows]):.3f}" for k in bwm.ROLES))
    print("== calibration (rule 18; tau set on calibration rows, reported on the axes): coverage / confident-wrong / accuracy among answered")
    for axis in ("new_ids", "depth3_unseen"):
        c = [r["calibration"][axis] for r in rows]
        print(f"  {axis:14s} tau {np.mean([r['calibration']['tau'] for r in rows]):.3f}: coverage {np.mean([x['coverage'] for x in c]):.3f} / confident-wrong {np.mean([x['confident_wrong'] for x in c]):.3f} / acc. answered {np.nanmean([x['accuracy_answered'] for x in c]):.3f}"
              + "  (per seed cw: " + " ".join(f"{x['confident_wrong']:.3f}" for x in c) + ")")
