"""report_ref.py: the common evaluation of one trained model on the frozen splits of the REFERENCE world (run 4), and the summary over
seeds. Same scorer and threshold rule as the basic world (eval/metrics.py); the frame includes the predicate ('pred') and the recipient's
absence, so FULL = A, B, C (or its absence), every speaker and the predicate exact. Also: 'roles FULL' (A, B, C only, the v16 column),
the predicate as an EVENT type (through the synonym table), the per-template breakdown at depth 3 (passives T2, T3, T5, T14; the
fronted-recipient comma form T9), and the three occupancy cases of the recipient (known filler on 'seen', unknown filler on
'unknown_recipient', not given on 'omit'). JSON per seed under logs/results_ref_<tag>_seed<s>.json."""
import collections, json, os
import numpy as np
from . import reference_world as rw, mainrepo
from eval import metrics

COLS = ["seen", "new ids", "role cues", "unseen trans.", "depth 2 (seen)", "depth 3 (UNSEEN)", "omit (C not given)", "unknown recipient"]

def evaluate(name, seed, preds, split, tag, max_cw=0.02):
    row = {"model": name, "seed": seed, "world": "reference", "axes": {}, "calibration": {}}
    ev_of = split["event_table"]
    for axis in rw.AXES:
        items = split["axes"][axis]
        if not items: row["axes"][axis] = None; continue
        d = items[0]["depth"]; roles = rw.roles_at(d); sc = metrics.score_axis(items, preds[axis], roles)
        sc["roles_full"] = float(np.mean([all(p["frame"].get(r) == it["frame"].get(r) for r in ("A", "B", "C")) for it, p in zip(items, preds[axis])]))
        sc["event"] = float(np.mean([ev_of.get(p["frame"].get("pred") or "", "?") == ev_of[it["frame"]["pred"]] for it, p in zip(items, preds[axis])]))
        sc["recipient_present_read_present"] = float(np.mean([(p["frame"].get("C") is not None) == it["has_C"] for it, p in zip(items, preds[axis])]))
        row["axes"][axis] = sc
    bd = collections.defaultdict(list)
    for it, p in zip(split["axes"]["depth3_unseen"], preds["depth3_unseen"]):
        bd[f"T{it['template']}"].append(all(p["frame"].get(r) == it["frame"].get(r) for r in rw.roles_at(3)))
    row["depth3_full_by_template"] = {k: float(np.mean(v)) for k, v in sorted(bd.items(), key=lambda kv: int(kv[0][1:]))}
    row["occupancy"] = {"known (seen)": row["axes"]["seen"]["recipient_present_read_present"],
                        "unknown filler": row["axes"]["unknown_recipient"]["recipient_present_read_present"],
                        "not given (omit)": row["axes"]["omit"]["recipient_present_read_present"]}
    cal_dec = []
    for it, p in zip(split["axes"]["calibration"], preds["calibration"]): cal_dec += metrics.role_decisions([it], [p], rw.roles_at(it["depth"]))
    tau = metrics.set_tau(cal_dec, max_cw); row["calibration"]["tau"] = tau; row["calibration"]["rule"] = f"smallest tau with confident-wrong <= {max_cw} on calibration rows"
    for axis in ("new_ids", "depth3_unseen", "unseen_trans", "seen"):
        items = split["axes"][axis]; dec = metrics.role_decisions(items, preds[axis], rw.roles_at(items[0]["depth"]))
        row["calibration"][axis] = metrics.calibration(dec, tau); row["calibration"][axis]["curve"] = metrics.coverage_curve(dec); row["calibration"][axis]["ece_rank"] = metrics.expected_calibration_error(dec)
    a = row["axes"]
    print(f"  seed {seed} {name:34s} FULL: " + " | ".join(f"{a[x]['full']:.2f}" if a[x] else "nan" for x in rw.AXES)
          + f"  || roles FULL depth3 {a['depth3_unseen']['roles_full']:.3f}; pred seen {a['seen']['per_role']['pred']:.2f}, event seen {a['seen']['event']:.2f}"
          + "  || occupancy " + " ".join(f"{k} {v:.2f}" for k, v in row["occupancy"].items())
          + f"  || tau {tau:.3f}: depth3 cov {row['calibration']['depth3_unseen']['coverage']:.2f} cw {row['calibration']['depth3_unseen']['confident_wrong']:.3f}", flush=True)
    print("     depth 3 FULL by template: " + " ".join(f"{k} {v:.2f}" for k, v in row["depth3_full_by_template"].items()), flush=True)
    os.makedirs(os.path.join(mainrepo.ROOT, "logs"), exist_ok=True)
    with open(os.path.join(mainrepo.ROOT, "logs", f"results_ref_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
    return row

def summarize(rows, name):
    n = len(rows); f = lambda v: f"{np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.2f}' for x in v)}]"
    print(f"\n== {name} (reference world): MEANS over {n} seeds, FULL frame (A, B, C or absent, speakers, predicate) ==  columns: " + " | ".join(COLS))
    print("  " + " | ".join(f([r["axes"][x]["full"] for r in rows]) if all(r["axes"][x] for r in rows) else "nan" for x in rw.AXES))
    print(f"  roles FULL (A, B, C; the v16 column) depth 3 UNSEEN: {f([r['axes']['depth3_unseen']['roles_full'] for r in rows])}")
    print(f"  predicate (surface lemma) seen: {f([r['axes']['seen']['per_role']['pred'] for r in rows])}; event type seen: {f([r['axes']['seen']['event'] for r in rows])}")
    print("  occupancy of the recipient: " + " | ".join(f"{k} {f([r['occupancy'][k] for r in rows])}" for k in rows[0]["occupancy"]))
    keys = rows[0]["depth3_full_by_template"].keys()
    print("  depth 3 FULL by template: " + " ".join(f"{k} {np.mean([r['depth3_full_by_template'][k] for r in rows]):.3f}" for k in keys))
    for axis in ("depth3_unseen", "unseen_trans", "new_ids"):
        c = [r["calibration"][axis] for r in rows]
        print(f"  calibration {axis:14s} tau {np.mean([r['calibration']['tau'] for r in rows]):.3f}: coverage {np.mean([x['coverage'] for x in c]):.3f} / confident-wrong {np.mean([x['confident_wrong'] for x in c]):.3f}"
              + "  (per seed cw: " + " ".join(f"{x['confident_wrong']:.3f}" for x in c) + ")")
