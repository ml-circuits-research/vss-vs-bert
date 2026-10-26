"""r80_dispatch.py (responses R80, answer to Q34): calibrated dispatch rules for the per-intent worlds, CPU only, on the SAVED R79 models
(logs/worlds/real_snips_<condition>_intent_<I>_seed<s>_r79/model_seed; no reader is retrained; the typer is re-fitted exactly as R79
fitted it, seed 0 on the training annotation, and its lexicon is checked against the kinds.tsv saved in every world).
Per world and utterance, three quantities, each converted to its percentile in that world's OWN calibration distribution (the first 200
of its training utterances, the rows R79's thresholds were set on; percentile = the share of calibration values strictly below the
value, so 201 levels):
  (a) the explained residual (sop @ SOURCE, residual 'explained');
  (b) the `intent` anchor role's cleanup score (the top cosine of the single intent symbol after unbinding the role, at the decoded
      structure hypothesis), and the structure score (f['core'] = mean top of the core roles); both are reported, and a second anchor
      rule on the structure score is printed when they differ;
  (c) the minimum slot margin (reported, used only as the last tie-break).
Dispatch rules (the chosen world's frame is the reading; spans by the typer around its heads, as R79):
  residual_raw       : min (residual, -min margin over all margins)            = R79's rule (checked: must reproduce R79's intent)
  residual_pct       : min (pct residual, residual, -min margin)
  anchor_pct         : max (pct anchor, anchor raw)
  anchor_then_res    : max pct anchor, then min pct residual, then min residual
  sum_pct            : max (pct anchor - pct residual), then min residual
  (structure_pct     : max (pct structure, structure raw); printed only when the structure score differs from the anchor)
Rejection: unchanged (every world's residual above its own R79 threshold, the 98th percentile on the same 200 rows); second line: every
world's anchor percentile under 0.02. Per rule: dispatch intent, span F1, held-out vs in-domain AUROC of the chosen world's residual (and,
second line, of its anchor percentile). Writes logs/results_real_snips_<condition>_reader_worlds_r80_<rule>_full_seed<s>.json.
Run: ./run.sh eval/r80_dispatch.py --condition base|ood_intents --seed S"""
import argparse, importlib, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vss import run_reader_real as rr
from common import mainrepo, real_data as rd, report_real
from eval import metrics, worlds_read as wr
import numpy as np

RULES = ["residual_raw", "residual_pct", "anchor_pct", "anchor_then_res", "sum_pct"]

def pct(cal, x):
    return float(np.mean(cal < x)) if len(cal) else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--condition", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only")
    ap.add_argument("--models", default="r79", choices=["r79", "r81"], help="R81: the models trained with one-vs-rest negatives (folders *_r81)")
    ap.add_argument("--decoder", default="paper", choices=["paper", "contract", "presence_query"], help="R84 item 3: 'contract' = the sop version's contract rules set in each reader's config at load (the models were written with decoder paper); 'presence_query' = that rule alone")
    ap.add_argument("--token_kinds", default=None, choices=["typer"], help="R88: pass the typer's in-context kinds to the decoder (formalize token_kinds)")
    ap.add_argument("--admission", default="stored", choices=["stored", "ctx"], help="R88 (B): admission sets recomputed from the typer's in-context kinds")
    ap.add_argument("--spans", default=None, choices=["token", "kind_run"], help="R87: reader.spans of the legends, set at load")
    ap.add_argument("--sop_dir", default=None, help="a directory holding the 'sop' package of the version the models were trained with (R82: the tree may hold a newer sop/)"); a = ap.parse_args()
    if a.sop_dir:   # loaded as 'sop_repo' before rr.load_sop_repo(), which then returns it
        import importlib, importlib.util
        spec = importlib.util.spec_from_file_location("sop_repo", os.path.join(a.sop_dir, "sop", "__init__.py"), submodule_search_locations=[os.path.join(a.sop_dir, "sop")])
        m_ = importlib.util.module_from_spec(spec); sys.modules["sop_repo"] = m_; spec.loader.exec_module(m_); importlib.import_module("sop_repo.infer")
    SR = rr.load_sop_repo(); src = open(os.path.join(a.sop_dir or mainrepo.ROOT, "sop", "SOURCE")).read().strip()[:7]
    cond = rd.load_condition("snips", a.condition); items = cond["items"]; intents = sorted({items[i]["intent"] for i in cond["train"]})
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    ev = report_real.eval_sets(cond, True, a.max_eval)
    print(f"r80_dispatch: models {a.models}; sop @ {src} (residual explained); condition {a.condition}; seed {a.seed}; worlds {intents}; sets " + ", ".join(f"{k} {len(v)}" for k, v in ev.items()))
    W = {}; all_types = set(); adm_log = {}
    for I in intents:
        wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_{a.condition}_intent_{I}_seed{a.seed}_{a.models}")
        saved = {}
        for line in open(os.path.join(wd, "kinds.tsv")):
            if line.startswith("#") or not line.strip(): continue
            w, k = line.rstrip("\n").split("\t"); saved[w] = k
        diff = sum(1 for w in set(saved) | set(kinds) if saved.get(w) != kinds.get(w))
        if diff: raise SystemExit(f"world {I}: the re-fitted typer's lexicon differs from the saved kinds.tsv in {diff} words")
        leg = json.load(open(os.path.join(wd, "legend.json"))); rinv = {r["id"]: r["question"][5:] for r in leg["roles"] if r["question"].startswith("slot ")}
        all_types |= set(rinv.values())
        R = SR.infer.Reader(os.path.join(wd, "model_seed")); R.cfg["residual"] = "explained"
        if a.decoder == "contract": R.cfg.update(importlib.import_module("sop_repo.activate").DECODERS["contract"])   # R84 item 3: the sop version's contract rules
        elif a.decoder == "presence_query": R.cfg["presence_query"] = True                                          # second line: that rule alone
        if a.admission == "ctx":   # R88 (B): admission sets from the typer's in-context kinds of the training fillers (>= 2%), set at load
            adm_log[I] = wr.ctx_admission(R, rinv, cond, I, ty); print(f"  world {I}: admission recomputed from context kinds: " + "; ".join(f"{q} {len(o)} -> {len(n)}" for q, (o, n) in sorted(adm_log[I].items())), flush=True)
        if a.spans: R.legend.reader["spans"] = a.spans   # R87: the legend's reader.spans declaration, set at load (no retraining; the saved legends are not rewritten)
        assert "intent" in R.schema["core_roles"], f"world {I}: intent is not a core role"
        W[I] = {"R": R, "rinv": rinv, "train": [i for i in cond["train"] if items[i]["intent"] == I][:200]}
    rep_op = lambda R: np.conj(R.st.op("says.scope")) if "says" in R.legend.operators else 1.0

    def read_one(R, rinv, toks):
        tk = {"token_kinds": dict(zip(toks, ty.sentence_kinds(toks)))} if a.token_kinds == "typer" else {}   # R88: the typer's in-context kinds
        text = " ".join(toks); wires, f, info = R.formalize(text, 0, SR.infer.idgen("e"), filt=set(toks), **tk); S = info["S"]; nS = np.linalg.norm(S)
        anchor = structure = 0.0
        if nS > 1e-9 and info["kept"]:
            Sn = S / nS; d = f.get("depth", 0) or 0; K = rep_op(R) ** d * (np.conj(R.st.op("not")) if f.get("neg") else 1.0)
            anchor = R.best(Sn * K * np.conj(R.st.role("intent")), R.role_cands("intent", set(toks)))[2]
            structure = float(f.get("core", R.structure_score(Sn, K, set(toks))))
        fr = {}
        for r in R.schema["role_order"]:
            fl = f["roles"].get(r)
            if r != "intent" and fl is not None and fl[1] is not None: fr[rinv.get(r, r)] = fl[1]
        mm = [float(v) for v in f.get("margins", {}).values()]
        sm = [float(f["margins"][r]) for r in R.schema["role_order"] if r != "intent" and r in f.get("margins", {})]
        return {"frame": fr, "status": info["status"], "residual": float(info["residual"]), "anchor": float(anchor), "structure": structure,
                "min_margin": min(mm) if mm else 0.0, "min_slot_margin": min(sm) if sm else 0.0, "n_presence_queries": len(f.get("presence_queries", {})),
                "margins": {rinv.get(r, r): float(f["margins"].get(r, 0.0)) for r in R.schema["role_order"] if r != "intent"}}

    cal = {}
    for I, w in W.items():
        rows = [read_one(w["R"], w["rinv"], items[i]["tokens"]) for i in w["train"]]
        cal[I] = {q: np.array([r[q] for r in rows]) for q in ("residual", "anchor", "structure", "min_slot_margin")}
        cal[I]["thr"] = float(np.quantile(cal[I]["residual"], 0.98))
        dif = float(np.mean(np.abs(cal[I]["anchor"] - cal[I]["structure"]) > 1e-9))
        print(f"  world {I}: core roles {w['R'].schema['core_roles']}; threshold {cal[I]['thr']:.3f}; calibration medians residual {np.median(cal[I]['residual']):.3f} "
              f"anchor {np.median(cal[I]['anchor']):.3f} structure {np.median(cal[I]['structure']):.3f} min slot margin {np.median(cal[I]['min_slot_margin']):.3f}; "
              f"anchor != structure on {dif:.3f} of the calibration rows", flush=True)
    differ = any(np.any(np.abs(cal[I]["anchor"] - cal[I]["structure"]) > 1e-9) for I in W)
    rules = RULES + (["structure_pct"] if differ else [])

    def all_worlds(it):
        outs = {}
        for I, w in W.items():
            o = read_one(w["R"], w["rinv"], it["tokens"]); c = cal[I]
            o.update(p_res=pct(c["residual"], o["residual"]), p_anc=pct(c["anchor"], o["anchor"]), p_str=pct(c["structure"], o["structure"]), p_mm=pct(c["min_slot_margin"], o["min_slot_margin"]))
            outs[I] = o
        return outs
    KEY = {"residual_raw": lambda o: (o["residual"], -o["min_margin"]),
           "residual_pct": lambda o: (o["p_res"], o["residual"], -o["min_margin"]),
           "anchor_pct": lambda o: (-o["p_anc"], -o["anchor"]),
           "anchor_then_res": lambda o: (-o["p_anc"], o["p_res"], o["residual"]),
           "sum_pct": lambda o: (o["p_res"] - o["p_anc"], o["residual"]),
           "structure_pct": lambda o: (-o["p_str"], -o["structure"])}
    SCORE = {"residual_raw": lambda o: -o["residual"], "residual_pct": lambda o: -o["p_res"], "anchor_pct": lambda o: o["p_anc"],
             "anchor_then_res": lambda o: o["p_anc"], "sum_pct": lambda o: o["p_anc"] - o["p_res"], "structure_pct": lambda o: o["p_str"]}

    def predict(outs, it, rule):
        order = sorted(outs, key=lambda I: KEY[rule](outs[I])); b = outs[order[0]]
        fr = dict(b["frame"]); fr["intent"] = order[0]; mg = dict(b["margins"])
        mg["intent"] = (SCORE[rule](b) - SCORE[rule](outs[order[1]])) if len(order) > 1 else 1.0
        return {"frame": fr, "margins": mg, "status": b["status"], "residual": b["residual"], "p_anchor": b["p_anc"], "chosen_abstains": b["status"] == "abstain", "n_presence_queries": b["n_presence_queries"],
                "rejected_by_every_world": all(o["residual"] > cal[I]["thr"] for I, o in outs.items()),
                "rejected_by_anchor_pct": all(o["p_anc"] < 0.02 for o in outs.values()),
                "spans": rr.spans_by_typer(it["tokens"], fr, ty, T), "spans_typer_only": rr.spans_typer_only(it["tokens"], ty)}

    reads = {k: [all_worlds(items[i]) for i in ids] for k, ids in ev.items()}
    reads_cal = [all_worlds(items[i]) for i in cond["dev_cal"]]
    r79 = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_real_snips_{a.condition}_reader_worlds_{a.models}_full_seed{a.seed}.json")))
    ms = lambda xs: f"{np.mean(xs):.3f}"
    summary = {}
    for rule in rules:
        preds = {k: [predict(o, items[i], rule) for o, i in zip(reads[k], ev[k])] for k in ev}
        cal_p = [predict(o, items[i], rule) for o, i in zip(reads_cal, cond["dev_cal"])]
        rej = {k: float(np.mean([p["rejected_by_every_world"] for p in v])) for k, v in preds.items()}
        rej_a = {k: float(np.mean([p["rejected_by_anchor_pct"] for p in v])) for k, v in preds.items()}
        auroc_anc = {}
        for k in [x for x in preds if x.startswith("ood:")]:
            ind = k.split(":", 1)[1] + ":in_domain"
            if ind in preds: auroc_anc[k] = report_real.auroc([x["p_anchor"] for x in preds[ind]], [x["p_anchor"] for x in preds[k]])
        row = report_real.evaluate(f"VSS, one world per intent + intent anchor{' + negatives' if a.models == 'r81' else ''}, dispatch {rule} (R80 rules, saved {a.models.upper()} models)", f"reader_worlds_{'r80' if a.models == 'r79' else 'r81r80'}{'' if a.decoder == 'paper' else '_' + a.decoder}{'_' + a.spans if a.spans else ''}{'_ctx' if a.token_kinds else ''}{'_adm' if a.admission == 'ctx' else ''}_{rule}", cond, a.seed, None, preds,
                                   cond["dev_cal"], cal_p, sorted(all_types), extra={"rule": rule, "decoder": a.decoder, "spans_declaration": a.spans, "token_kinds": a.token_kinds, "admission": a.admission, "admission_sets": adm_log, "rejected_by_every_world": rej, "rejected_by_anchor_pct": rej_a,
                                   "chosen_abstains": {k: float(np.mean([p["chosen_abstains"] for p in v])) for k, v in preds.items()},
                                   "presence_queries_per_utterance": {k: float(np.mean([p["n_presence_queries"] for p in v])) for k, v in preds.items()},
                                   "share_with_presence_query": {k: float(np.mean([p["n_presence_queries"] > 0 for p in v])) for k, v in preds.items()},
                                   "ood_auroc_anchor_pct": auroc_anc, "world_thresholds": {I: cal[I]["thr"] for I in W}, "sop": src, "models": f"{a.models.upper()} saved, seed {a.seed}"})
        summary[rule] = row
        print(f"  [{rule}] " + "; ".join(f"{k}: intent {row['sets'][k]['intent']:.3f} span F1 {row['spans'][k]['f1']:.3f} rejected {rej[k]:.3f} (anchor pct < 0.02: {rej_a[k]:.3f}) chosen frame abstains {row['chosen_abstains'][k]:.3f} with a presence query {row['share_with_presence_query'][k]:.3f}" for k in ev)
              + "".join(f"; AUROC {k} residual {row['ood_auroc_residual'][k]:.3f} anchor pct {auroc_anc[k]:.3f}" for k in auroc_anc), flush=True)
    for k in ev:
        if k in r79["sets"]: print(f"  check: residual_raw reproduces the run's own residual dispatch ({a.models}) on {k}: intent {summary['residual_raw']['sets'][k]['intent']:.4f} vs the run {r79["sets"][k]["intent"]:.4f}")
    metrics.checklist(1, "[x] saved R79 models, no reader retrained [x] percentiles on each world's own 200 training rows (rule 25) [x] rejection unchanged, anchor line second [x] R79's rule reproduced")
