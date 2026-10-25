"""report.py: the common evaluation of one trained model on the frozen splits of one seed, and the common printing over seeds.
Every model script calls evaluate(model_name, seed, preds_by_axis, split) and, at the end, summarize(rows). The output has the columns
of RESULTS 7b (so that the numbers line up with the paper's tables) plus the calibration block of rule 18 and a JSON dump per seed in
logs/results_<script>_<seed>.json for eval/compare.py."""
import json, os
import numpy as np
from . import basic_world as bwm, mainrepo
from eval import metrics

AXES = ["seen", "new_ids", "new_form_seen_trans", "new_form_unseen_trans", "new_form_role_cues_seen", "depth1", "depth2_seen", "depth3_unseen"]
TRAINED_D3 = ["A", "B", "C", "X1", "X2"]   # the depth-3 roles whose labels occur in training (depths 0-2); X3 never does
INNER = ["A", "B", "C"]
COLS = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "new form., ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]

def _calibration(split, preds, tau_rows, max_cw, axes=("new_ids", "depth3_unseen", "seen")):
    """rule 18 block: tau on the calibration rows by the written rule, then coverage / confident-wrong on the reported axes; R10 item 6
    adds the depth-3 block over the trained roles (A B C X1 X2) beside the six-role one"""
    cal_dec = []
    for it, p in zip(split["axes"]["calibration"], tau_rows):
        cal_dec += metrics.role_decisions([it], [p], bwm.roles_at(it["depth"]))
    tau = metrics.set_tau(cal_dec, max_cw); out = {"tau": tau, "rule": f"smallest tau with confident-wrong <= {max_cw} on calibration rows"}
    for axis in axes:
        items = split["axes"][axis]; dec = metrics.role_decisions(items, preds[axis], bwm.roles_at(items[0]["depth"]))
        out[axis] = metrics.calibration(dec, tau); out[axis]["curve"] = metrics.coverage_curve(dec); out[axis]["ece_rank"] = metrics.expected_calibration_error(dec)
    items = split["axes"]["depth3_unseen"]; dec = metrics.role_decisions(items, preds["depth3_unseen"], TRAINED_D3)
    out["depth3_unseen_trained"] = metrics.calibration(dec, tau); out["depth3_unseen_trained"]["ece_rank"] = metrics.expected_calibration_error(dec)
    return out

def _depth3_extra(split, preds, positions, train_max):
    d3 = split["axes"]["depth3_unseen"]; p3 = preds["depth3_unseen"]
    out = {"full_trained": metrics.full_over(d3, p3, TRAINED_D3), "full_inner": metrics.full_over(d3, p3, INNER)}
    if positions is not None:
        out["position_split"] = {"all_roles": metrics.position_split(d3, p3, positions, train_max, bwm.roles_at(3)),
                                 "inner": metrics.position_split(d3, p3, positions, train_max, INNER)}
    return out

def evaluate(name, seed, preds, split, tag, max_cw=0.02, positions=None, train_max=None, unit="word", extra_rules=None):
    """preds: axis -> list of predictions (incl. 'calibration'); returns the row dict and prints it.
    R10 (2026-10-25), additive only (the existing fields are computed exactly as before): when the predictions carry per-candidate scores,
    the one-to-one rule (metrics.one_to_one) is scored as a second decoding rule under row["one_to_one"]; the depth-3 accuracy is split by
    filler position seen / unseen in training when `positions` (one {role: position} per depth-3 item, in the model's `unit`) and
    `train_max` are given, or in words by default; the depth-3 per-item predictions are stored (frame and margins). `extra_rules` maps a
    name to another full set of predictions (for example the temperature-scaled tagger) scored with its own calibration block."""
    row = {"model": name, "seed": seed, "axes": {}, "per_role_depth3": {}, "calibration": {}}
    for axis in AXES:
        items = split["axes"][axis]
        if not items: row["axes"][axis] = None; continue
        d = items[0]["depth"]; row["axes"][axis] = metrics.score_axis(items, preds[axis], bwm.roles_at(d))
    row["per_role_depth3"] = row["axes"]["depth3_unseen"]["per_role"]
    d3 = split["axes"]["depth3_unseen"]
    if positions is None:
        positions = [bwm.word_positions(it) for it in d3]; train_max = bwm.train_max_words(bwm.build(seed, split["D"])["w"]) if train_max is None else train_max; unit = "word"
    # R6 (2026-10-25, before run 2): depth-3 FULL over the trained labels and over the inner frame, beside the six-role FULL; R10 item 2a
    row["axes"]["depth3_unseen"].update(_depth3_extra(split, preds, positions, train_max)); row["axes"]["depth3_unseen"]["position_unit"] = unit
    row["calibration"] = _calibration(split, preds, preds["calibration"], max_cw)
    row["depth3_items"] = [{"frame": p["frame"], "margins": p["margins"]} for p in preds["depth3_unseen"]]
    rules = dict(extra_rules or {})
    if all("cand_scores" in p for ax in AXES + ["calibration"] if split["axes"][ax] for p in preds[ax]):
        rules["one_to_one"] = {ax: [metrics.one_to_one(p, bwm.roles_at(it["depth"])) for it, p in zip(split["axes"][ax], preds[ax])] for ax in AXES + ["calibration"] if split["axes"][ax]}
    for rn, rp in rules.items():
        rr = {"axes": {ax: (metrics.score_axis(split["axes"][ax], rp[ax], bwm.roles_at(split["axes"][ax][0]["depth"])) if split["axes"][ax] else None) for ax in AXES}}
        rr["per_role_depth3"] = rr["axes"]["depth3_unseen"]["per_role"]; rr["axes"]["depth3_unseen"].update(_depth3_extra(split, rp, positions, train_max))
        rr["calibration"] = _calibration(split, rp, rp["calibration"], max_cw)
        rr["depth3_items"] = [{"frame": p["frame"], "margins": p["margins"]} for p in rp["depth3_unseen"]]
        if all("assignment_margin" in p for p in rp["depth3_unseen"]): rr["assignment_margin_depth3_mean"] = float(np.mean([p["assignment_margin"] for p in rp["depth3_unseen"]]))
        row[rn] = rr
    a = row["axes"]
    print(f"  seed {seed} {name:34s} " + " | ".join(f"{a[x]['mean_role']:.2f} (m {a[x]['margin'] and np.mean(list(a[x]['margin'].values())):.2f})" if a[x] else " nan " for x in AXES)
          + "  || depth 3 per role: " + " ".join(f"{k} {v:.2f}" for k, v in row["per_role_depth3"].items())
          + f"  || FULL depth3 {a['depth3_unseen']['full']:.3f} (trained labels {a['depth3_unseen']['full_trained']:.3f}, inner {a['depth3_unseen']['full_inner']:.3f}) | tau {row['calibration']['tau']:.3f}: new ids cov {row['calibration']['new_ids']['coverage']:.2f} cw {row['calibration']['new_ids']['confident_wrong']:.3f}; depth3 cov {row['calibration']['depth3_unseen']['coverage']:.2f} cw {row['calibration']['depth3_unseen']['confident_wrong']:.3f}", flush=True)
    for rn in rules:
        b = row[rn]["axes"]; ps = b["depth3_unseen"].get("position_split", {}).get("all_roles", {})
        print(f"  seed {seed}   rule '{rn}': " + " | ".join(f"{b[x]['mean_role']:.2f}" if b[x] else " nan " for x in AXES) + f"  || depth 3 per role: " + " ".join(f"{k} {v:.2f}" for k, v in row[rn]["per_role_depth3"].items())
              + f"  || FULL depth3 {b['depth3_unseen']['full']:.3f} (trained {b['depth3_unseen']['full_trained']:.3f}, inner {b['depth3_unseen']['full_inner']:.3f}); depth3 cw {row[rn]['calibration']['depth3_unseen']['confident_wrong']:.3f}", flush=True)
    ps = a["depth3_unseen"]["position_split"]["all_roles"]
    print(f"  seed {seed}   depth-3 role accuracy by filler position ({unit}s; train max {ps['train_max']}): seen {ps['seen_position']['accuracy']} (n {ps['seen_position']['n']}), unseen {ps['unseen_position']['accuracy']} (n {ps['unseen_position']['n']})", flush=True)
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
    for k, lab in (("full_trained", "A B C X1 X2, the labels seen in training"), ("full_inner", "A B C, the inner frame")):
        v = [r["axes"]["depth3_unseen"][k] for r in rows]; print(f"  FULL at depth 3 over {lab} (R6): {np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]")
    print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3")
    print("  " + " | ".join(f"{np.mean([r['per_role_depth3'][k] for r in rows]):.3f}±{np.std([r['per_role_depth3'][k] for r in rows]):.3f}" for k in bwm.ROLES))
    def _ps(rs, k, key):
        v = [r["axes"]["depth3_unseen"]["position_split"][k][key]["accuracy"] for r in rs if "position_split" in r["axes"]["depth3_unseen"]]
        v = [x for x in v if x is not None]; return f"{np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.2f}' for x in v)}]" if v else "n/a"
    print(f"== depth 3 by filler position (R10 item 2a; seen in training / never reached): all roles {_ps(rows, 'all_roles', 'seen_position')} / {_ps(rows, 'all_roles', 'unseen_position')}; inner A B C {_ps(rows, 'inner', 'seen_position')} / {_ps(rows, 'inner', 'unseen_position')}")
    for rn in [k for k in ("one_to_one", "temperature") if all(k in r for r in rows)]:
        R = [r[rn] for r in rows]; print(f"== second rule '{rn}' (R10): " + " | ".join(f"{np.mean([x['axes'][ax]['mean_role'] for x in R if x['axes'][ax]]):.3f}" if any(x['axes'][ax] for x in R) else "nan" for ax in AXES))
        for k in ("full", "full_trained", "full_inner"):
            v = [x["axes"]["depth3_unseen"][k] for x in R]; print(f"  '{rn}' depth 3 {k}: {np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]")
        print(f"  '{rn}' depth 3 per role: " + " | ".join(f"{np.mean([x['per_role_depth3'][k] for x in R]):.3f}" for k in bwm.ROLES))
        print(f"  '{rn}' depth 3 by filler position: all roles {_ps(R, 'all_roles', 'seen_position')} / {_ps(R, 'all_roles', 'unseen_position')}")
        for axis in ("new_ids", "depth3_unseen", "depth3_unseen_trained"):
            c = [x["calibration"][axis] for x in R]
            print(f"  '{rn}' calibration {axis:22s} tau {np.mean([x['calibration']['tau'] for x in R]):.3f}: coverage {np.mean([y['coverage'] for y in c]):.3f} / confident-wrong {np.mean([y['confident_wrong'] for y in c]):.3f} (per seed cw: " + " ".join(f"{y['confident_wrong']:.3f}" for y in c) + ")")
    print("== calibration (rule 18; tau set on calibration rows, reported on the axes): coverage / confident-wrong / accuracy among answered")
    for axis in ("new_ids", "depth3_unseen", "depth3_unseen_trained"):
        c = [r["calibration"][axis] for r in rows]
        print(f"  {axis:14s} tau {np.mean([r['calibration']['tau'] for r in rows]):.3f}: coverage {np.mean([x['coverage'] for x in c]):.3f} / confident-wrong {np.mean([x['confident_wrong'] for x in c]):.3f} / acc. answered {np.nanmean([x['accuracy_answered'] for x in c]):.3f}"
              + "  (per seed cw: " + " ".join(f"{x['confident_wrong']:.3f}" for x in c) + ")")
