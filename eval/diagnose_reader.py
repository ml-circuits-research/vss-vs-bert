"""diagnose_reader.py (R38, coordinator's refinement of 2026-10-25): reads ONLY the design-set dumps results/dump_<tag>_seed<s>.json
(eval/dump_reader.py; new-formulation report rows of seeds 0-4) and counts the wrong role decisions (per-role rule) under three
hypothesis-consistent patterns, per row and seed. The patterns are written here before the dumps are read and are not exclusive:
  H3 (decoding)  : the gold candidate is ranked 2nd AND the one-to-one rule (R10 item 1) recovers it ('near miss fixed by decoding');
                   also reported: rank 2 overall, rank >= 3, and the median margin of wrong vs right decisions.
  H2 (routes)    : the decision's cell (template, role) has an error rate >= 0.5 on EVERY seed of the row ('systematic, seed-independent':
                   the gate sets differ by seed in 1a, so a cell wrong on every seed is not explained by one seed's gates); for R14 / R37
                   the same within the row, and the overlap with 1a's systematic cells.
  H1 (gates)     : the cell's error rate is >= 0.5 on some seed and <= 0.1 on another ('seed-dependent'), AND some cue category present in
                   the template is open on one of those seeds and closed on the other (gate value > 0.5 vs < 0.5). Per category: the
                   cells' error rate on seeds with the gate open vs closed.
  other          : wrong decisions in none of the three.
Also: error rate by the gold filler's position (before / after the event verb), by template, per axis; the 50-item confident-error
sample per row (fixed rng 0; confident = margin >= the row's calibration tau, 0.000 for 1a and R14 per RESULTS, so every error is
confident) classified by kind: 'wrong role' (the prediction is another role's gold filler), 'wrong candidate in sentence', 'candidate
not in sentence'; missed role and false fill cannot occur with the depth given (stated, counted 0); with the gate value of the token right
before the gold filler. Writes results/diagnosis_<tag>.json and results/confident_errors_50_<tag>.tsv.
Run: ./run.sh eval/diagnose_reader.py [--rows 1a,r14,r37]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
import numpy as np

TAGS = {"1a": "reader_basic", "r14": "reader_basic_cues", "r37": "reader_basic_cuesinit"}
CUES = ["to", "by", "from", ".", "says", "that", "claims", ",", "COP", "what", "it", "who"]

def load(tag):
    out = []
    for s in range(5):
        p = os.path.join(mainrepo.ROOT, "results", f"dump_{tag}_seed{s}.json")
        if os.path.exists(p): out.append(json.load(open(p)))
    return out

def cells(dump):
    c = collections.defaultdict(lambda: [0, 0])
    for it in dump["items"]:
        for r, v in it["roles"].items(): c[(it["template"], r)][0] += not v["correct"]; c[(it["template"], r)][1] += 1
    return {k: e / n for k, (e, n) in c.items()}

def analyse(row, dumps, sys1a=None):
    seeds = [d["seed"] for d in dumps]; rate = {d["seed"]: cells(d) for d in dumps}; keys = sorted(set().union(*[set(r) for r in rate.values()]))
    tmpl_cats = {}
    for d in dumps:
        for it in d["items"]: tmpl_cats.setdefault(it["template"], set(it["cats"]))
    systematic = {k for k in keys if all(rate[s].get(k, 0) >= 0.5 for s in seeds)}
    seeddep = {}
    for k in keys:
        hi = [s for s in seeds if rate[s].get(k, 0) >= 0.5]; lo = [s for s in seeds if rate[s].get(k, 1) <= 0.1]
        if hi and lo:
            gated = sorted({c for c in tmpl_cats[k[0]] if c in CUES for h in hi for l in lo
                            if (dumps[seeds.index(h)]["gates"].get(c, 0) > 0.5) != (dumps[seeds.index(l)]["gates"].get(c, 0) > 0.5)})
            if gated: seeddep[k] = gated
    per_seed = []
    for d in dumps:
        cnt = collections.Counter(); wm, rm = [], []; pos = collections.defaultdict(lambda: [0, 0]); n_dec = 0
        for it in d["items"]:
            for r, v in it["roles"].items():
                n_dec += 1; k = (it["template"], r); bv = v["before_verb"]; pos[bv][0] += not v["correct"]; pos[bv][1] += 1
                if v["correct"]: rm.append(v["margin"]); continue
                wm.append(v["margin"]); pats = []
                if v["rank_gold"] == 2 and v["o2o_correct"]: pats.append("H3")
                if k in systematic: pats.append("H2")
                if k in seeddep: pats.append("H1")
                cnt["wrong"] += 1; cnt["rank2"] += v["rank_gold"] == 2; cnt["rank3plus"] += (v["rank_gold"] or 99) >= 3; cnt["o2o_fixed"] += v["o2o_correct"]
                for p_ in pats: cnt[p_] += 1
                cnt["only_" + "+".join(pats) if pats else "other"] += 1
        per_seed.append({"seed": d["seed"], "open": d["open"], "decisions": n_dec, "counts": dict(cnt), "median_margin_wrong": float(np.median(wm)) if wm else None,
                         "median_margin_right": float(np.median(rm)) if rm else None,
                         "error_rate_before_verb": pos[True][0] / max(1, pos[True][1]), "error_rate_after_verb": pos[False][0] / max(1, pos[False][1]),
                         "n_before_verb": pos[True][1], "n_after_verb": pos[False][1]})
    gate_effect = {}
    for c in CUES:
        op, cl = [], []
        for k in keys:
            if c not in tmpl_cats[k[0]]: continue
            for d in dumps: (op if d["gates"].get(c, 0) > 0.5 else cl).append(rate[d["seed"]].get(k, 0))
        if op and cl: gate_effect[c] = {"cell_error_open": float(np.mean(op)), "cell_error_closed": float(np.mean(cl)), "n_open": len(op), "n_closed": len(cl)}
    by_t = collections.defaultdict(lambda: [0, 0])
    for d in dumps:
        for it in d["items"]:
            for v in it["roles"].values(): by_t[it["template"]][0] += not v["correct"]; by_t[it["template"]][1] += 1
    return {"row": row, "seeds": seeds, "per_seed": per_seed, "systematic_cells": sorted(f"T{t}:{r}" for t, r in systematic),
            "systematic_cells_shared_with_1a": sorted(f"T{t}:{r}" for t, r in systematic & sys1a) if sys1a is not None else None,
            "seed_dependent_cells": {f"T{t}:{r}": g for (t, r), g in sorted(seeddep.items())}, "gate_effect": gate_effect,
            "cell_error_rate": {f"T{t}:{r}": [round(rate[s].get((t, r), float('nan')), 3) for s in seeds] for t, r in keys},
            "error_rate_by_template": {f"T{t}": e / n for t, (e, n) in sorted(by_t.items())}}, systematic

def sample50(tag, dumps):
    errs = [(d["seed"], it, r, v) for d in dumps for it in d["items"] for r, v in it["roles"].items() if not v["correct"] and v["margin"] >= 0.0]
    rng = np.random.default_rng(0); pick = [errs[i] for i in sorted(rng.choice(len(errs), size=min(50, len(errs)), replace=False))] if errs else []
    lines = ["seed\taxis\ti\ttemplate\trole\tgold\tpred\tkind\trank_gold\tmargin\tgate_before_gold\tbefore_verb\tsentence"]; kinds = collections.Counter()
    for s, it, r, v in pick:
        kind = "wrong role" if v["pred_is_other_role_filler"] else ("wrong candidate in sentence" if v["pred_in_sentence"] else "candidate not in sentence")
        kinds[kind] += 1; gp = v["gold_pos"]; gb = f"{it['tokens'][gp - 1]}:{it['gate'][gp - 1]:.3f}" if gp else "-"
        lines.append(f"{s}\t{it['axis']}\t{it['i']}\tT{it['template']}\t{r}\t{v['gold']}\t{v['pred']}\t{kind}\t{v['rank_gold']}\t{v['margin']:.4f}\t{gb}\t{v['before_verb']}\t{' '.join(it['tokens'])}")
    kinds["missed role"] = 0; kinds["false fill"] = 0
    open(os.path.join(mainrepo.ROOT, "results", f"confident_errors_50_{tag}.tsv"), "w").write("\n".join(lines) + "\n")
    return dict(kinds), len(errs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default="1a,r14,r37"); a = ap.parse_args()
    print(f"diagnose_reader: reads results/dump_<tag>_seed<s>.json only (design set, seeds 0-4; R39)")
    sys1a = None
    for row in a.rows.split(","):
        tag = TAGS[row]; dumps = load(tag)
        if not dumps: print(f"\n== {row} ({tag}): no dump found; skipped"); continue
        res, sysc = analyse(row, dumps, sys1a)
        if row == "1a": sys1a = sysc
        res["confident_error_kinds_50"], res["n_errors_pooled"] = sample50(tag, dumps)
        json.dump(res, open(os.path.join(mainrepo.ROOT, "results", f"diagnosis_{tag}.json"), "w"), indent=1)
        print(f"\n== {row} ({tag}), seeds {res['seeds']}: wrong role decisions on the new-formulation rows, by pattern (not exclusive) ==")
        print("seed | open gates | decisions | wrong | H3 rank2+o2o | H2 systematic | H1 seed-dep.+gate | other | rank2 | rank>=3 | o2o fixes | med. margin wrong / right | err before / after verb")
        for p in res["per_seed"]:
            c = p["counts"]
            print(f"  {p['seed']} | {' '.join(p['open'])} | {p['decisions']} | {c.get('wrong', 0)} | {c.get('H3', 0)} | {c.get('H2', 0)} | {c.get('H1', 0)} | {c.get('other', 0)} | {c.get('rank2', 0)} | {c.get('rank3plus', 0)} | {c.get('o2o_fixed', 0)} | {p['median_margin_wrong']:.4f} / {p['median_margin_right']:.4f} | {p['error_rate_before_verb']:.3f} ({p['n_before_verb']}) / {p['error_rate_after_verb']:.3f} ({p['n_after_verb']})")
        print(f"  exclusive combinations per seed: {[{k: v for k, v in p['counts'].items() if k.startswith('only_') or k == 'other'} for p in res['per_seed']]}")
        print(f"  systematic cells (error >= 0.5 on every seed): {res['systematic_cells']}")
        if res["systematic_cells_shared_with_1a"] is not None: print(f"  of which also systematic in 1a: {res['systematic_cells_shared_with_1a']}")
        print(f"  seed-dependent cells with a differing cue gate: {res['seed_dependent_cells']}")
        print(f"  cell error rate by cue gate open / closed (cells whose template has the cue): " + "; ".join(f"{c} {v['cell_error_open']:.3f} (n {v['n_open']}) / {v['cell_error_closed']:.3f} (n {v['n_closed']})" for c, v in res["gate_effect"].items()))
        print(f"  error rate by template (pooled seeds): {', '.join(f'{k} {v:.3f}' for k, v in res['error_rate_by_template'].items())}")
        print(f"  cell error rate per seed: {res['cell_error_rate']}")
        print(f"  50-item confident-error sample ({res['n_errors_pooled']} errors pooled): {res['confident_error_kinds_50']}; results/confident_errors_50_{tag}.tsv")
    print("\nchecklist: [x] design set only (seeds 0-4) [x] patterns written before the dumps were read [x] per-seed counts [x] no threshold set on report rows")
