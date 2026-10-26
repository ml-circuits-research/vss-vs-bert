"""q33_diag.py (questions Q33 / responses R78 addendum, R79): the record of the cause of R75's thresholds at 1.0, from the SAVED R75
per-world models (logs/worlds/real_snips_base_intent_<I>_seed<s>/model_seed; no retraining). Every world reads (a) 200 of its own
training utterances (the rows its threshold was set on) and (b) every utterance of the SNIPS test set, through a given version of sop/
(--sop_dir: a directory holding a 'sop' package; the R75 runs imported 28db081). Per world and seed: the share of residuals equal to 1.0
and their source: the frame-incomplete clause (a core question unanswered), nothing explained / shape not ok, or the graded value
itself; the world's core roles; and, as a second line, the graded residual alone (the clause ignored): its share at 1.0 and its 98th
percentile on (a). Writes logs/results_q33diag_<label>_seed<s>.json.
Run: ./run.sh eval/q33_diag.py --sop_dir DIR --label 28db081 --seed S"""
import argparse, importlib, importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo, real_data as rd
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--label", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="base"); a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("sop_diag", os.path.join(a.sop_dir, "sop", "__init__.py"), submodule_search_locations=[os.path.join(a.sop_dir, "sop")])
    m = importlib.util.module_from_spec(spec); sys.modules["sop_diag"] = m; spec.loader.exec_module(m); inf = importlib.import_module("sop_diag.infer")
    src = open(os.path.join(a.sop_dir, "sop", "SOURCE")).read().strip()[:7]
    cond = rd.load_condition("snips", a.condition); items = cond["items"]; test = cond["test"] if "test" in cond else []
    test = [i for i in items if items[i]["partition"] == "test"] if not test else test
    intents = sorted({items[i]["intent"] for i in cond["train"]})
    print(f"q33_diag: sop @ {src} (label {a.label}); seed {a.seed}; condition {a.condition}; worlds {intents}; test utterances {len(test)}")
    out = {"sop": src, "label": a.label, "seed": a.seed, "worlds": {}}
    for I in intents:
        md = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_{a.condition}_intent_{I}_seed{a.seed}", "model_seed")
        R = inf.Reader(md); R.cfg["residual"] = "explained"; core = list(R.schema.get("core_roles", []))
        own = [i for i in cond["train"] if items[i]["intent"] == I][:200]
        def read(ids):
            rows = []
            for i in ids:
                toks = items[i]["tokens"]; wires, f, info = R.formalize(" ".join(toks), 0, inf.idgen("e"), filt=set(toks))
                inc = bool(R.frame_incomplete(f)) if hasattr(R, "frame_incomplete") else False
                placed = R.placed_words(f) if f.get("shape") == "ok" else set()
                if placed and f.get("lemma") is None and f.get("event"):
                    ev = next((t for t in toks if R.legend.event_of_lemma(R.lemma(t)) == f["event"]), None)
                    if ev is not None: placed = placed | {ev}
                graded = float(R.residual_explained(info["S"], info["addr"], info["kept"], toks, placed)) if info["kept"] else 1.0
                rows.append({"res": float(info["residual"]), "inc": inc, "graded": graded, "shape_ok": f.get("shape") == "ok", "status": info["status"]})
            return rows
        w = {"core_roles": core, "n_questions": len(R.schema.get("role_order", []))}
        for name, ids in (("own_training_200", own), ("test_all", test)):
            rr = read(ids); one = [r for r in rr if r["res"] >= 1.0 - 1e-9]; n = max(1, len(rr))
            w[name] = {"n": len(rr), "share_res_1": len(one) / n, "of_which_incomplete_clause": sum(r["inc"] for r in one) / max(1, len(one)),
                       "of_which_shape_not_ok": sum(not r["shape_ok"] for r in one) / max(1, len(one)), "of_which_graded_1": sum(r["graded"] >= 1.0 - 1e-9 for r in one) / max(1, len(one)),
                       "graded_only_share_1": sum(r["graded"] >= 1.0 - 1e-9 for r in rr) / n, "p98_clause": float(np.quantile([r["res"] for r in rr], 0.98)),
                       "p98_graded_only": float(np.quantile([r["graded"] for r in rr], 0.98))}
            x = w[name]; print(f"  seed {a.seed} {I:22s} {name:16s} n {x['n']}: residual 1.0 in {x['share_res_1']:.3f} (incomplete clause {x['of_which_incomplete_clause']:.3f}, shape not ok {x['of_which_shape_not_ok']:.3f}, graded itself 1.0 {x['of_which_graded_1']:.3f}); 98th pct {x['p98_clause']:.3f} | graded only: share 1.0 {x['graded_only_share_1']:.3f}, 98th pct {x['p98_graded_only']:.3f}; core roles {core}", flush=True)
        out["worlds"][I] = w
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_q33diag_{a.label}_seed{a.seed}.json"), "w"), indent=1)
    print("checklist: [x] saved R75 models, no retraining [x] the sop version named [x] the clause and the graded residual reported apart")
