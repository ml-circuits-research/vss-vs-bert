"""r84_oppoints.py (responses R84 item 1): rejection operating points of the per-intent configuration, CPU only, on the SAVED R81 models
(no training), read through a named sop version (--sop_dir; the R81 models were trained with SOURCE 82585ea). The dispatch is R80's
anchor-percentile rule (R81's primary). Signals, each oriented so that LARGER = refuse:
  residual   : the chosen world's explained residual;
  anchor     : 1 - the chosen world's anchor percentile (under the anchor rule the chosen world holds the MAX anchor percentile over worlds,
               so this is also "every world's anchor percentile is low", the R80 line made continuous);
  max        : max(the chosen world's residual percentile, 1 - its anchor percentile): the max of the two signals in percentile units;
  residual_all: min over worlds of the residual percentile (the R79 line "every world's residual is high", continuous; second line);
  anchor_z_all: min over worlds of (mean - anchor) / std of the world's calibration anchors (second line, declared before the run: the
               percentile has 201 levels and saturates below the calibration minimum, where the held-out utterances sit).
For each signal and target in-domain false rejection 1% / 2% / 5%: threshold = the (1 - target) quantile of the signal on the dev
CALIBRATION half (cond['dev_cal'], in-domain only, never scored; rule 25); refuse iff signal > threshold; reported: the realised false
rejection on dev_cal, and on test the held-out-intent rejection (ood:test) and the in-domain false rejection (test:in_domain); dev_rep
sets as second lines; AUROC held-out vs in-domain per signal. Writes logs/results_r84_oppoints_<condition>_seed<s>.json.
Run: ./run.sh eval/r84_oppoints.py --sop_dir DIR --condition ood_intents|base --seed S"""
import argparse, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd, report_real
from eval import metrics
import numpy as np

SIGNALS = ["residual", "anchor", "max", "residual_all", "anchor_z_all"]
TARGETS = [0.01, 0.02, 0.05]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--condition", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--models", default="r81"); ap.add_argument("--decoder", default="paper", choices=["paper", "contract", "presence_query"]); ap.add_argument("--spans", default=None, choices=["token", "kind_run"]); ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only"); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir)
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", a.condition); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    W = wr.Worlds(SR, cond, a.condition, a.seed, a.models, kinds, a.decoder, a.spans)
    ev = report_real.eval_sets(cond, True, a.max_eval); cal_ids = cond["dev_cal"][:a.max_eval] if a.max_eval else cond["dev_cal"]
    print(f"r84_oppoints: models {a.models}; sop @ {src}; decoder {a.decoder}; spans {a.spans}; condition {a.condition}; seed {a.seed}; worlds {W.intents}; dev_cal {len(cal_ids)}; sets " + ", ".join(f"{k} {len(v)}" for k, v in ev.items()))
    def sig(i):
        outs = W.read_all(items[i]["tokens"]); I = W.dispatch_anchor(outs); o = outs[I]
        return {"chosen": I, "residual": o["residual"], "anchor": 1.0 - o["p_anc"], "max": max(o["p_res"], 1.0 - o["p_anc"]), "residual_all": min(x["p_res"] for x in outs.values()),
                "anchor_z_all": min((W.cal[J]["anchor"].mean() - x["anchor"]) / max(1e-9, W.cal[J]["anchor"].std()) for J, x in outs.items())}
    cal = [sig(i) for i in cal_ids]; sets = {k: [sig(i) for i in ids] for k, ids in ev.items()}
    out = {"sop": src, "decoder": a.decoder, "spans": a.spans, "models": a.models, "condition": a.condition, "seed": a.seed, "signals": {}}
    for s_ in SIGNALS:
        cs = np.array([x[s_] for x in cal]); res = {"auroc": {}, "points": {}}
        for k in [x for x in sets if x.startswith("ood:")]:
            ind = k.split(":", 1)[1] + ":in_domain"
            res["auroc"][k] = report_real.auroc([x[s_] for x in sets[k]], [x[s_] for x in sets[ind]])   # held-out should have the LARGER signal
        for t in TARGETS:
            thr = float(np.quantile(cs, 1.0 - t)); pt = {"threshold": thr, "fr_dev_cal": float(np.mean(cs > thr))}
            for k, v in sets.items(): pt[k] = float(np.mean([x[s_] > thr for x in v]))
            res["points"][str(t)] = pt
            print(f"  [{s_}] target FR {t:.0%}: threshold {thr:.4f} (dev_cal FR {pt['fr_dev_cal']:.3f}) | " + "; ".join(f"{k} refused {pt[k]:.3f}" for k in sets)
                  + "".join(f" | AUROC {k} {v:.3f}" for k, v in res["auroc"].items()), flush=True)
        out["signals"][s_] = res
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_r84_oppoints_{a.condition}{'' if a.decoder == 'paper' else '_' + a.decoder}{'_' + a.spans if a.spans else ''}_seed{a.seed}.json"), "w"), indent=1)
    metrics.checklist(1, "[x] saved R81 models, no training [x] thresholds on the dev calibration half only (rule 25) [x] reported on test [x] sop version named")
