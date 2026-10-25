"""mrc_table.py (R45, 2026-10-25): the per-row table of the MRC rows from their per-seed JSONs (logs/results_<tag>_seed<s>.json), since
R45 runs one job per (model, variant, seed). No model is run; prints mean ± std and the per-seed list for every axis (slot accuracy),
the depth-3 exact matches (depth given and NOT given), confident error at depth 3, occupancy with the depth not given, R17 (i), the
residual analogue, the last-epoch training loss (R35 / R36's convergence clause) and the cost fields (concurrent runs: not cost numbers).
Run: python3 eval/mrc_table.py <tag> [<tag> ...] [--seeds 0-4]"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
from common.report import AXES
import numpy as np

def ms(xs, f=".3f"):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):{f}} ± {np.std(xs):{f}} [{' '.join(format(x, f) for x in xs)}]" if xs else "-"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--seeds", default="0-4"); a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    for tag in a.tags:
        rows = [json.load(open(p)) for s in range(lo, hi + 1) for p in [os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{s}.json")] if os.path.exists(p)]
        print(f"\n== {tag}: {len(rows)} seeds {[r['seed'] for r in rows]}; {rows[0]['model'] if rows else ''}")
        if not rows: continue
        for ax in AXES: print(f"  slot accuracy {ax:24s} {ms([r['axes'][ax]['mean_role'] for r in rows])}")
        d3 = [r["axes"]["depth3_unseen"] for r in rows]
        print(f"  depth 3 per role: " + " | ".join(f"{k} {np.mean([x['per_role'][k] for x in d3]):.3f}" for k in d3[0]["per_role"]))
        print(f"  exact match depth 3 (depth given): six {ms([x['full'] for x in d3])}; trained {ms([x['full_trained'] for x in d3])}; inner {ms([x['full_inner'] for x in d3])}")
        print(f"  exact match depth 3 (depth NOT given, all six questions): six {ms([r['depth_not_given']['full_six'] for r in rows])}; trained {ms([r['depth_not_given']['full_trained'] for r in rows])}; inner {ms([r['depth_not_given']['full_inner'] for r in rows])}; depth accuracy {ms([r['depth_not_given']['depth_accuracy_depth3'] for r in rows])}")
        print(f"  confident error depth 3 (tau {ms([r['calibration']['tau'] for r in rows])}): {ms([r['calibration']['depth3_unseen']['confident_wrong'] for r in rows])}; coverage {ms([r['calibration']['depth3_unseen']['coverage'] for r in rows])}; over trained labels {ms([r['calibration']['depth3_unseen_trained']['confident_wrong'] for r in rows])}")
        for ax in ("seen", "new_ids", "depth3_unseen"):
            o = [r["occupancy_depth_not_given"][ax] for r in rows]
            print(f"  occupancy (depth not given) {ax:14s} false fill on absent {ms([x['false_fill_on_absent'] for x in o])}; missed present {ms([x['missed_present'] for x in o])}")
        for v in ("null", "threshold"):
            print(f"  R17 (i) rejected ({v:9s}) random {ms([r['r17'][f'rejected_{v}']['random'] for r in rows])}; reference {ms([r['r17'][f'rejected_{v}']['reference'] for r in rows])}; nonce {ms([r['r17'][f'rejected_{v}']['nonce'] for r in rows])}; false rejection pooled {ms([r['r17']['false_rejection_pooled'][v] for r in rows])}")
        print(f"  residual analogue seen / new ids / depth 3: {ms([r['residual_analogue']['seen'] for r in rows])} / {ms([r['residual_analogue']['new_ids'] for r in rows])} / {ms([r['residual_analogue']['depth3_unseen'] for r in rows])}")
        tl = [r.get("train_loss_last_epoch", float("nan")) for r in rows]; print("  last-epoch mean training loss: " + " ".join(f"{x:.3e}" for x in tl))
        c = rows[0]["cost"]; print(f"  cost: {c['params']/1e6:.2f} M parameters, {c['bytes']/1e6:.1f} MB fp32; training {ms([r['cost']['train_s'] for r in rows], '.0f')} s per seed (concurrent: not a cost number); CPU ms per sentence with six passes {ms([r['cost']['cpu_ms_per_sentence_all_questions'] for r in rows], '.1f')} (concurrent)")
