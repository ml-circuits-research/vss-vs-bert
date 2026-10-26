"""r90_hybrid.py (responses R90 item 1): the demo's reading as three parts, CPU only, on the SAVED R81 models (no training):
dispatch by the anchor percentile (R80's rule); the FRAME by line C of R88 (presence_by_kind + presence_query, reader.spans kind_run,
the typer's in-context kinds, admission sets from the in-context kinds >= 2%); the REFUSAL by line A's explained residual of the chosen
world (presence_query, kind_run, in-context kinds, admission as stored) above its dev-calibrated 2% threshold (the (1 - 0.02) quantile of
that residual on the dev calibration half, as eval/r84_oppoints.py). One sop version (--sop_dir; 040d44e: line A there = 4a0eba8's
reading, the only change being the presence_by_kind key, off for line A), two reader sets per world (A and C). Checks: the anchor's
choice is the same in both sets; line A's threshold equals eval/r84_oppoints.py's (same rule, same rows).
Writes logs/results_r90_hybrid_<condition>_seed<s>.json. Run: ./run.sh eval/r90_hybrid.py --sop_dir DIR --condition base|ood_intents --seed S"""
import argparse, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd, report_real
from eval import metrics
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--condition", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only"); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir)
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", a.condition); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    WA = wr.Worlds(SR, cond, a.condition, a.seed, "r81", kinds, "presence_query", "kind_run", ty, "stored")       # line A: the refusal
    WC = wr.Worlds(SR, cond, a.condition, a.seed, "r81", kinds, "presence_by_kind", "kind_run", ty, "ctx")       # line C: the frame
    ev = report_real.eval_sets(cond, True, a.max_eval); cal_ids = cond["dev_cal"][:a.max_eval] if a.max_eval else cond["dev_cal"]
    print(f"r90_hybrid: sop @ {src}; condition {a.condition}; seed {a.seed}; worlds {WA.intents}; dispatch anchor percentile; frame line C; refusal line A's residual at the dev_cal 2% threshold")
    mism = [0]
    def read(i):
        toks = items[i]["tokens"]; oa = WA.read_all(toks); oc = WC.read_all(toks)
        I = WA.dispatch_anchor(oa); Ic = WC.dispatch_anchor(oc); mism[0] += I != Ic
        fr = dict(oc[I]["frame"]); fr["intent"] = I
        return {"intent": I, "frame": fr, "res_A": oa[I]["residual"], "res_C": oc[I]["residual"], "status_C": oc[I]["status"], "spans": rr.spans_by_typer(toks, fr, ty, T)}
    cal = [read(i) for i in cal_ids]; thr = float(np.quantile([x["res_A"] for x in cal], 0.98)); thr_c = float(np.quantile([x["res_C"] for x in cal], 0.98))
    out = {"sop": src, "condition": a.condition, "seed": a.seed, "threshold_A_2pct": thr, "threshold_C_2pct": thr_c, "fr_dev_cal": float(np.mean([x["res_A"] > thr for x in cal])), "sets": {}}
    for k, ids in ev.items():
        P = [read(i) for i in ids]; g = [report_real.gold_flat(items[i]) for i in ids]
        s = {"n": len(P), "intent": float(np.mean([p["intent"] == gg.get("intent") for p, gg in zip(P, g)])) if not k.startswith("ood:") else None,
             "span_f1": metrics.span_f1([report_real.gold_spans(items[i]) for i in ids], [p["spans"] for p in P])["f1"],
             "refused_by_A_residual": float(np.mean([p["res_A"] > thr for p in P])), "refused_by_C_residual": float(np.mean([p["res_C"] > thr_c for p in P])),
             "chosen_frame_abstains": float(np.mean([p["status_C"] == "abstain" for p in P])), "res_A": [p["res_A"] for p in P], "res_C": [p["res_C"] for p in P]}
        out["sets"][k] = s
        print(f"  {k} (n {s['n']}): dispatch intent {s['intent'] if s['intent'] is not None else float('nan'):.3f} | span F1 (frame C) {s['span_f1']:.3f} | refused by line A's residual {s['refused_by_A_residual']:.3f} "
              f"(second line, line C's own residual at its 2% threshold: {s['refused_by_C_residual']:.3f}) | chosen frame (C) abstains {s['chosen_frame_abstains']:.3f}", flush=True)
    for k in [x for x in out["sets"] if x.startswith("ood:")]:
        ind = k.split(":", 1)[1] + ":in_domain"
        if ind in out["sets"]:
            out["sets"][k]["auroc_A"] = report_real.auroc(out["sets"][k]["res_A"], out["sets"][ind]["res_A"]); out["sets"][k]["auroc_C"] = report_real.auroc(out["sets"][k]["res_C"], out["sets"][ind]["res_C"])
            print(f"  AUROC held-out vs in-domain, {k}: line A's residual {out['sets'][k]['auroc_A']:.3f}, line C's residual {out['sets'][k]['auroc_C']:.3f}")
    print(f"  threshold (line A's residual, dev_cal 2%): {thr:.4f} (dev_cal FR {out['fr_dev_cal']:.3f}); anchor choice differs between the A and C reader sets on {mism[0]} utterances")
    out["anchor_mismatches"] = mism[0]
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_r90_hybrid_{a.condition}_seed{a.seed}.json"), "w"), indent=1)
    metrics.checklist(1, "[x] saved R81 models, no training [x] threshold on the dev calibration half only (rule 25) [x] the anchor's choice checked equal in both reader sets [x] sop version named")
