"""sop_check.py (responses R58 item 1, R59; Q29): the saved readers read through the TOOL, sop.infer.Reader (sop/ = the reader's single
implementation, SOURCE commit in sop/SOURCE; the author's directory, read only here), and EQUAL checks against this repository's
transcriptions (common/decode.py, eval/sentence_candidates.py). The tool is the reference: a difference is recorded, never hidden.
  - data/basic_world/legend.json + kinds.tsv: the basic world in the tool's legend format (roles A, B, C; operator says; no predicate
    term, no occupancy marker; level openers says / claims), written by --write_legend (seed-independent: the vocabulary and kinds are
    the same on every seed).
  - per (row, seed): a tool model folder logs/models/sop/<tag>_seed<s>/ (untracked): routebook.npz (sop.train.export_routebook of the
    saved V8), legend.json, kinds.tsv, catalog.json (words and kinds, the schema, identity_table = the seed's identity draw rng 300 + seed,
    star_table = NestedRoleWorld's role / level draw rng 100 + seed), config.json (thresholds tau 0).
  - check 0: the tool's chart S against V8.states (relative difference) on 50 calibration sentences;
  - check 1, default decoder: per axis, per-role slot accuracy with the depth given (force) and the axis's candidate lists (filt), and the
    depth-3 exact match, against the stored row of the same reader (results_<tag>_seed<s>.json);
  - check 2, the demo decoder (config candidates "sentence", residual "explained", undecided_adjacent true): slot accuracy with filt =
    the sentence's words (the tool's one-to-one), against the R46 one-to-one rows (logs/results_r46_<tag>_seed<s>.json); the explained
    residual's AUROC in vs out (random + reference) and the rejection / false rejection at the 2% calibration threshold, against the R46
    rows' explained-residual numbers (seeds with an R46 row only).
Writes logs/results_sopcheck_<tag>_seed<s>.json and prints the comparisons; label every reader row "decoder: sop @ <SOURCE>" afterwards.
Run: ./run.sh eval/sop_check.py --write_legend | --row 1a|1a_fresh|r14|r37|vorder|vorderb|... --seed_start S --seeds N [--demo]"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from scipy.stats import rankdata
from common import basic_world as bwm, decode, report
from eval.reader_models import get_reader, ROWS
from sop import infer as sop_infer, train as sop_train, legend as sop_legend

SOURCE = open(os.path.join(mainrepo.ROOT, "sop", "SOURCE")).read().strip()[:7]
LEG_DIR = os.path.join(mainrepo.ROOT, "data", "basic_world")
NONCONTENT = {"COP", "from", ",", "what", "it", "to", "who", "by", "says", "claims", ".", "that"}

def write_legend():
    split = bwm.load(0); cat = split["cat"]; os.makedirs(LEG_DIR, exist_ok=True)
    verbs = sorted(w for w, k in cat.items() if k not in NONCONTENT and k not in ("P", "O"))
    leg = {"version": "basic-world 1 (vss-vs-bert, R58 item 1)", "roles": [{"id": "A", "question": "who gives?"}, {"id": "B", "question": "what is given?"},
           {"id": "C", "question": "to whom is it given?"}], "events": [{"id": "transfer", "verbs": verbs}], "operators": ["says"],
           "reader": {"predicate_role": False, "occupancy_marker": False, "level_openers": ["says", "claims"]}, "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}}
    json.dump(leg, open(os.path.join(LEG_DIR, "legend.json"), "w"), indent=1)
    with open(os.path.join(LEG_DIR, "kinds.tsv"), "w") as f:
        f.write("# the basic world's lexicon: word<TAB>kind (the kinds of common/basic_world, the same on every seed)\n")
        for w in sorted(cat): f.write(f"{w}\t{cat[w]}\n")
    L = sop_legend.Legend(os.path.join(LEG_DIR, "legend.json")); print(f"legend written: {L.sha}; errors {L.errors}; warnings {L.warnings}; {len(L.lexicon)} words; verbs {len(verbs)}")

def model_folder(row, seed, m, bw, split, demo, verbs_explained=False):
    tag = ROWS[row]; d = os.path.join(mainrepo.ROOT, "logs", "models", "sop", f"{tag}_seed{seed}" + ("_demo" if demo else "") + ("_vx" if verbs_explained else "")); os.makedirs(d, exist_ok=True)
    if getattr(m, "vo", None) is not None: m.vo_nmax = m.nmax
    sop_train.export_routebook(m, os.path.join(d, "routebook.npz"))
    for n in ("legend.json", "kinds.tsv"): shutil.copyfile(os.path.join(LEG_DIR, n), os.path.join(d, n))
    cat = split["cat"]; content = ["O", "P"] if verbs_explained else sorted({"P", "O"} | {k for k in cat.values() if k not in NONCONTENT})
    # verbs_explained (2026-10-25, the R42-equivalent configuration): the verb kinds are not content kinds, so the event verb counts as
    # explained as R42's definition says (the legend has no predicate role, so the tool never places a lemma); default: verbs are content
    schema = {"roles": {r: {"rate": 1.0, "main": "word", "kinds": ["O" if r == "B" else "P"]} for r in ("A", "B", "C")}, "role_order": ["A", "B", "C"],
              "core_roles": ["A", "B", "C"], "speaker_kinds": ["P"], "max_depth": 2, "neg_event": False, "single_event": "transfer", "content_kinds": content, "unsupported_shapes": []}
    catalog = {"seed": seed, "words": {w: {"kind": k} for w, k in sorted(cat.items())}, "schema": schema,
               "identity_table": {"vocab": list(bw["w"].vocab), "rng_seed": 300 + seed}, "star_table": {"rng_seed": 100 + seed, "order": ["role:A", "role:B", "role:C", "op:says.speaker", "op:says.scope"]}}
    json.dump(catalog, open(os.path.join(d, "catalog.json"), "w"), indent=1)
    cfg = {"world": "basic", "D": int(m.F.shape[1]), "seed": seed, "legend_sha256": sop_legend.sha256_file(os.path.join(d, "legend.json")), "codes": list(m.codes),
           "depth_estimator": "kinds", "presence_readout": "decided", "thresholds": {"tau": 0.0}}
    if demo: cfg.update(candidates="sentence", residual="explained", undecided_adjacent=True)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    return sop_infer.Reader(d)

def auroc(inw, out):
    x = np.concatenate([inw, out]); r = rankdata(x); n1, n2 = len(inw), len(out); return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))

def tool_frame(rd, tokens, depth, filt):
    S, kept, unknown, addr = rd.chart(tokens); f = rd.read(S, tokens, filt=filt, force=(depth, False))
    fr = {r: None for r in bwm.ROLES}
    for r in ("A", "B", "C"):
        fl = f["roles"].get(r); fr[r] = fl[1] if fl else None
    for k, sp in enumerate(f["speakers"]): fr[f"X{k + 1}"] = sp
    return fr, f, S, kept, addr

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write_legend", action="store_true"); ap.add_argument("--row", default="1a"); ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--demo", action="store_true"); ap.add_argument("--verbs_explained", action="store_true"); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    print(f"sop_check (R58 item 1): main repository {mainrepo.check_pinned()}; decoder: sop @ {SOURCE} (sop/SOURCE); row {a.row}; seeds {a.seed_start}-{a.seed_start + a.seeds - 1}; {'demo decoder (candidates sentence, residual explained, undecided_adjacent)' if a.demo else 'default decoder'}")
    if a.write_legend: write_legend(); sys.exit(0)
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        m, bw, split, preds, how, chk = get_reader(a.row, seed, a.device); rd = model_folder(a.row, seed, m, bw, split, a.demo, a.verbs_explained); res = {"seed": seed, "row": a.row, "decoder": f"sop @ {SOURCE}", "demo": a.demo}
        cal = split["axes"]["calibration"][:50]; worst = 0.0
        with torch.no_grad():
            for it in cal:
                St = m.states(*m.encode([it["tokens"]]))[0].cpu().numpy(); Sn = rd.chart(it["tokens"])[0]; worst = max(worst, float(np.linalg.norm(St - Sn) / np.linalg.norm(St)))
        res["chart_rel_diff"] = worst; print(f"  seed {seed}: {how}; {chk}; tool chart vs V8.states: max relative difference {worst:.2e}", flush=True)
        stored = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_{ROWS[a.row]}_seed{seed}.json")))
        r46p = os.path.join(mainrepo.ROOT, "logs", f"results_r46_{ROWS[a.row]}_seed{seed}.json"); r46 = json.load(open(r46p)) if os.path.exists(r46p) else None
        res["axes"] = {}
        for ax in report.AXES:
            items = split["axes"][ax]
            if not items: continue
            filt0 = set(bwm.candidates_for(split, ax, "A")) | set(bwm.candidates_for(split, ax, "B"))
            ok = n = ex = 0
            for it in items:
                filt = (set(it["tokens"]) if a.demo else filt0)
                fr = tool_frame(rd, it["tokens"], it["depth"], filt)[0]; roles = bwm.roles_at(it["depth"])
                ok += sum(fr[r] == it["frame"][r] for r in roles); n += len(roles); ex += all(fr[r] == it["frame"][r] for r in bwm.ROLES)
            ref = (r46["one_to_one"]["axes"][ax]["mean_role"] if (a.demo and r46) else None) if a.demo else stored["axes"][ax]["mean_role"]
            res["axes"][ax] = {"slot": ok / n, "exact": ex / len(items), "reference": ref}
            print(f"  seed {seed} {ax:24s} tool slot accuracy {ok / n:.4f} (exact {ex / len(items):.4f}); {'R46 one-to-one row' if a.demo else 'stored row'} {('%.4f' % ref) if ref is not None else '-'}; {'EQUAL to 3 decimals' if ref is not None and round(ok / n, 3) == round(ref, 3) else ('DIFFERENT' if ref is not None else '')}", flush=True)
        if a.demo:
            est = lambda items: [dict(x, depth=decode.depth_from_openers(x["tokens"])) for x in items]
            def resid(items):
                out = []
                for it in est(items):
                    fr, f, S, kept, addr = tool_frame(rd, it["tokens"], it["depth"], set(it["tokens"]))
                    out.append(rd.residual_explained(S, addr, kept, it["tokens"], rd.placed_words(f)) if kept else 1.0)
                return np.array(out)
            c = resid(split["axes"]["calibration"]); t = float(np.quantile(c, 0.98)); inw = np.concatenate([resid(split["axes"][ax]) for ax in report.AXES if split["axes"][ax]])
            oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"))); o = {s: resid(oow["test"][s]) for s in ("random", "reference")}
            res["explained"] = {"auroc_pooled": auroc(inw, np.concatenate([o["random"], o["reference"]])), "rejected": {s: float(np.mean(v > t)) for s, v in o.items()}, "false_rejection_pooled": float(np.mean(inw > t)), "threshold": t}
            refx = (r46 or {}).get("oow", {}).get("explained", {})
            print(f"  seed {seed} explained residual (tool): AUROC {res['explained']['auroc_pooled']:.4f} (R46 row {refx.get('auroc_pooled_random_reference', float('nan')):.4f}); rejected {res['explained']['rejected']} (R46 {refx.get('rejected')}); false rejection {res['explained']['false_rejection_pooled']:.4f} (R46 {refx.get('false_rejection_pooled', float('nan')):.4f})", flush=True)
        json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_sopcheck_{ROWS[a.row]}{'_demo' if a.demo else ''}{'_vx' if a.verbs_explained else ''}_seed{seed}.json"), "w"))
    print("checklist: [x] the tool (sop/) is the reference [x] the saved readers, no retraining [x] differences recorded")
