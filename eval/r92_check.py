"""r92_check.py (responses R92 step 1): the demo reading as ONE config per world, checked on the saved R81 models against R90 item 1.
For every base / ood_intents world of a seed, a READING COPY of the saved model (logs/worlds/..._r81/model_seed, not modified) is written to
--stage/<condition>_seed<s>/<intent>/: config.json gets the frame keys of R92 (candidates sentence, undecided_adjacent, residual explained,
presence_query, presence_by_kind) and "refusal": {"presence_by_kind": false}; legend.json's reader block gets "spans": "kind_run" (the
config's legend_sha256 is updated to the new legend, the training one kept as legend_sha256_at_training); catalog.json's schema gets the
admission sets of R88 (B) (per question, the typer's in-context kinds >= 2% of the training fillers' heads). Each copy is read by sop
@ --sop_dir (60f861b) with token_kinds = the typer's in-context kinds. Dispatch: the anchor percentile (R80) over these readers; frame =
the frame pass; refusal = the refusal pass's explained residual (info residual_refusal) of the chosen world above the dev_cal 2% threshold
(pooled, as R90 measured; the per-world 98th percentiles written as well). Compared with R90 item 1 (results_r90_hybrid_*): dispatch,
span F1, refusal held-out / false rejection. Writes logs/results_r92_check_<condition>_seed<s>.json.
Run: ./run.sh eval/r92_check.py --sop_dir DIR --stage DIR --condition base|ood_intents --seed S"""
import argparse, json, os, shutil, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd, report_real
from eval import metrics
import numpy as np

FRAME_KEYS = {"candidates": "sentence", "undecided_adjacent": True, "residual": "explained", "presence_query": True, "presence_by_kind": True}
REFUSAL = {"presence_by_kind": False}

def stage_world(SR, cond, condition, seed, I, ty, stage_dir):
    src = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_{condition}_intent_{I}_seed{seed}_r81", "model_seed")
    wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_{condition}_intent_{I}_seed{seed}_r81")
    dst = os.path.join(stage_dir, f"{condition}_seed{seed}", I)
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    leg = json.load(open(os.path.join(dst, "legend.json"))); leg.setdefault("reader", {})["spans"] = "kind_run"
    json.dump(leg, open(os.path.join(dst, "legend.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(dst, "config.json"))); cfg["legend_sha256_at_training"] = cfg["legend_sha256"]
    cfg["legend_sha256"] = SR.legend.Legend(os.path.join(dst, "legend.json")).sha; cfg.update(FRAME_KEYS); cfg["refusal"] = dict(REFUSAL)
    cfg["reading_config"] = "vss-vs-bert responses R92: frame = R88 line C, refusal = presence_by_kind off; written by eval/r92_check.py"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    R = SR.infer.Reader(dst); rinv = {r["id"]: r["question"][5:] for r in leg["roles"] if r["question"].startswith("slot ")}
    adm = wr.ctx_admission(R, rinv, cond, I, ty)          # R88 (B) sets, into the schema of the copy
    cat = json.load(open(os.path.join(dst, "catalog.json"))); cat["schema"] = R.schema; json.dump(cat, open(os.path.join(dst, "catalog.json"), "w"))
    return dst, rinv, {q: n for q, (o, n) in adm.items()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--max_eval", type=int, default=None); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir)
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", a.condition); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    intents = sorted({items[i]["intent"] for i in cond["train"]}); W = {}
    for I in intents:
        d, rinv, adm = stage_world(SR, cond, a.condition, a.seed, I, ty, a.stage)
        R = SR.infer.Reader(d); W[I] = {"R": R, "rinv": rinv, "dir": d}
        assert R.cfg.get("refusal") == REFUSAL and R.legend.reader.get("spans") == "kind_run"
    print(f"r92_check: sop @ {src}; condition {a.condition}; seed {a.seed}; reading copies in {os.path.join(a.stage, f'{a.condition}_seed{a.seed}')}; frame keys {FRAME_KEYS}; refusal {REFUSAL}")
    def read_one(I, toks):
        R = W[I]["R"]; rinv = W[I]["rinv"]
        wires, f, info = R.formalize(" ".join(toks), 0, SR.infer.idgen("e"), filt=set(toks), token_kinds=dict(zip(toks, ty.sentence_kinds(toks))))
        S = info["S"]; nS = np.linalg.norm(S); anchor = 0.0
        if nS > 1e-9 and info["kept"]: anchor = R.best(S / nS * np.conj(R.st.role("intent")), R.role_cands("intent", set(toks)))[2]
        fr = {rinv[r]: fl[1] for r, fl in f["roles"].items() if r in rinv and fl is not None and fl[1] is not None}
        return {"frame": fr, "status": info["status"], "res_frame": float(info["residual"]), "res_refusal": float(info.get("residual_refusal", np.nan)), "anchor": float(anchor)}
    cal_anchor = {I: np.array([read_one(I, items[i]["tokens"])["anchor"] for i in [j for j in cond["train"] if items[j]["intent"] == I][:200]]) for I in intents}
    def read(i):
        toks = items[i]["tokens"]; outs = {I: read_one(I, toks) for I in intents}
        I = sorted(intents, key=lambda J: (-wr.pct(cal_anchor[J], outs[J]["anchor"]), -outs[J]["anchor"]))[0]
        fr = dict(outs[I]["frame"]); fr["intent"] = I
        return {"intent": I, "frame": fr, "res": outs[I]["res_refusal"], "res_frame": outs[I]["res_frame"], "status": outs[I]["status"], "spans": rr.spans_by_typer(toks, fr, ty, T)}
    cal_ids = cond["dev_cal"][:a.max_eval] if a.max_eval else cond["dev_cal"]
    cal = [read(i) for i in cal_ids]; thr = float(np.quantile([x["res"] for x in cal], 0.98))
    per_world = {I: (float(np.quantile([x["res"] for x in cal if x["intent"] == I], 0.98)) if any(x["intent"] == I for x in cal) else None) for I in intents}
    ref = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_r90_hybrid_{a.condition}_seed{a.seed}.json")))
    out = {"sop": src, "condition": a.condition, "seed": a.seed, "threshold_pooled": thr, "threshold_per_world": per_world, "r90_threshold": ref["threshold_A_2pct"], "sets": {}}
    print(f"  refusal threshold (pooled dev_cal 2%): {thr:.4f} (R90: line A's {ref['threshold_A_2pct']:.4f}); per world " + ", ".join(f"{I} {v:.3f}" for I, v in per_world.items() if v is not None))
    for k, ids in report_real.eval_sets(cond, True, a.max_eval).items():
        P = [read(i) for i in ids]; g = [report_real.gold_flat(items[i]) for i in ids]; r90 = ref["sets"].get(k, {})
        s = {"n": len(P), "intent": float(np.mean([p["intent"] == gg.get("intent") for p, gg in zip(P, g)])) if not k.startswith("ood:") else None,
             "span_f1": metrics.span_f1([report_real.gold_spans(items[i]) for i in ids], [p["spans"] for p in P])["f1"],
             "refused": float(np.mean([p["res"] > thr for p in P])), "refused_per_world_thr": float(np.mean([p["res"] > (per_world.get(p["intent"]) or thr) for p in P])),
             "abstains": float(np.mean([p["status"] == "abstain" for p in P])), "res": [p["res"] for p in P]}
        out["sets"][k] = s
        print(f"  {k}: dispatch {s['intent'] if s['intent'] is not None else float('nan'):.3f} (R90 {r90.get('intent') if r90.get('intent') is not None else float('nan'):.3f}) | span F1 {s['span_f1']:.3f} (R90 {r90.get('span_f1', float('nan')):.3f}) | "
              f"refused {s['refused']:.3f} (R90 {r90.get('refused_by_A_residual', float('nan')):.3f}; per-world thresholds {s['refused_per_world_thr']:.3f}) | abstains {s['abstains']:.3f} (R90 {r90.get('chosen_frame_abstains', float('nan')):.3f})", flush=True)
    for k in [x for x in out["sets"] if x.startswith("ood:")]:
        ind = k.split(":", 1)[1] + ":in_domain"
        if ind in out["sets"]: out["sets"][k]["auroc"] = report_real.auroc(out["sets"][k]["res"], out["sets"][ind]["res"]); print(f"  AUROC {k}: {out['sets'][k]['auroc']:.3f}")
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_r92_check_{a.condition}_seed{a.seed}.json"), "w"), indent=1)
    metrics.checklist(1, "[x] saved R81 models, reading copies only [x] thresholds on the dev calibration half (rule 25) [x] compared with R90 item 1 [x] sop version named")
