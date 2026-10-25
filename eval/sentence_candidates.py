"""sentence_candidates.py (responses R46, declared in RESULTS 'Deviation R46'): the reader row "sentence candidates + one-to-one", decoder
only, on saved readers (eval/reader_models.py, EQUAL-checked against the stored row with the ORIGINAL decoder first; no retraining).
Decoding: common/decode.decode_frozen(..., sentence_candidates=True): each role's cleanup over the words of the sentence whose kind the
role admits; then the one-to-one rule of R10 (metrics.one_to_one) over those candidates (its margin: assigned score minus the best other
candidate of the role; a single candidate against R10's floor 0). report.evaluate scores both the per-role in-sentence top-1 and the
one-to-one rule (the declared row) on every axis, with the rule-18 calibration block (tau on the calibration rows). Beside it: the
out-of-world AUROC (in-world report rows pooled vs random + reference pooled; per source; nonce as placeholder apart) and the 2%
calibration-threshold table, for the compiled and the explained residual computed on the one-to-one frames. Every role decision in R47's three
categories (address / restriction / elimination) with its accuracy, per axis, and on the design set per (template, role) cell. Writes logs/results_r46_<tag>_seed<s>.json.
Run: ./run.sh eval/sentence_candidates.py --row 1a|r14|1a_fresh [--seed_start 0] [--seeds 5]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np
from scipy.stats import rankdata
from common import basic_world as bwm, decode, report
from eval import metrics
from eval.reader_models import get_reader, ROWS

FR = 0.02; NF = ["new_form_seen_trans", "new_form_unseen_trans", "new_form_role_cues_seen"]
def auroc(inw, out):
    x = np.concatenate([inw, out]); r = rankdata(x); n1, n2 = len(inw), len(out); return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))
def est(items): return [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]

def one_to_one(pred, roles):
    """metrics.one_to_one (R10 item 1) over the in-sentence candidates, with one guard that only matters when a sentence has fewer
    admissible words than roles (out-of-world sentences): a role the assignment leaves without one of ITS candidates is decoded None
    ("not given") with margin 0. Identical to metrics.one_to_one whenever every role receives one of its own candidates."""
    from scipy.optimize import linear_sum_assignment
    cs = pred["cand_scores"]; rs = [r for r in roles if r in cs and len(cs[r][0])]
    out = {"frame": {r: None for r in pred["frame"]}, "margins": {r: 0.0 for r in roles}}
    if not rs: return out
    U = sorted({f for r in rs for f in cs[r][0]}); col = {f: j for j, f in enumerate(U)}; M = np.full((len(rs), len(U)), -1e9)
    for i, r in enumerate(rs):
        for f, v in zip(*cs[r]): M[i, col[f]] = v
    ri, ci = linear_sum_assignment(-M)
    for i, j in zip(ri, ci):
        r = rs[i]; f = U[j]
        if M[i, j] <= -1e8: continue
        sc = dict(zip(*cs[r])); others = [v for g, v in sc.items() if g != f]
        out["frame"][r] = f; out["margins"][r] = float(sc[f] - (max(others) if others else 0.0))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--row", required=True, choices=list(ROWS)); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args(); seeds = list(range(a.seed_start, a.seed_start + a.seeds)); tag = ROWS[a.row]
    label = "CLAIM SET (fresh splits, R39)" if a.seed_start >= 5 else "DESIGN SET (inspected, R39)"
    name = f"VSS reader {a.row} + sentence candidates (R46)"
    print(f"sentence_candidates (R46): main repository {mainrepo.check_pinned()}; row {a.row} ({tag}); seeds {seeds}: {label}; decoder only, no retraining")
    print("expected (R46, before the run): T11 B and T4 B right by construction; T4 C, T7 A, T10 C right through one-to-one whenever the other role is right; in-world rows cannot fall; cost: random in-vocabulary sentences filled into frames more often")
    rows = []
    for seed in seeds:
        m, bw, split, preds0, how, chk = get_reader(a.row, seed, a.device); world = bw["world"]
        print(f"  seed {seed}: {how}; original decoder vs stored row: {chk}", flush=True)
        P = {ax: decode.decode_frozen(m, world, split, ax, sentence_candidates=True) for ax in report.AXES + ["calibration"]}
        row = report.evaluate(name, seed, P, split, f"r46_{tag}")
        def o2o_items(items, axis="x"):
            e = est(items); p = decode.decode_frozen(m, world, {"axes": {axis: e}, "candidates": split["candidates"]}, axis, sentence_candidates=True)
            return e, [one_to_one(pp, bwm.roles_at(it["depth"])) for pp, it in zip(p, e)]
        def both(items, axis="x"):
            e, q = o2o_items(items, axis); return np.array(decode.residuals(m, world, e, q)), np.array(decode.residuals_explained(m, world, e, q))
        cal = both(split["axes"]["calibration"]); inw = {ax: both(split["axes"][ax], ax) for ax in report.AXES if split["axes"][ax]}
        oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"))); out = {s: both(oow["test"][s]) for s in ("random", "reference")}
        with decode.Placeholders(m, bw["w"], seed) as ph: out["nonce_placeholder"] = both([dict(x, tokens=ph.map(x["tokens"])) for x in oow["test"]["nonce"]])
        row["oow"] = {}
        for k, nm in ((0, "compiled"), (1, "explained")):
            t = float(np.quantile(cal[k], 1 - FR)); pin = np.concatenate([v[k] for v in inw.values()]); pout = np.concatenate([out["random"][k], out["reference"][k]])
            row["oow"][nm] = {"threshold": t, "auroc_pooled_random_reference": auroc(pin, pout), "auroc": {s: auroc(pin, v[k]) for s, v in out.items()},
                              "rejected": {s: float(np.mean(v[k] > t)) for s, v in out.items()}, "false_rejection_pooled": float(np.mean(pin > t)),
                              "false_rejection": {ax: float(np.mean(v[k] > t)) for ax, v in inw.items()}}
            r_ = row["oow"][nm]; print(f"  seed {seed} {nm:9s} residual on the one-to-one frames: AUROC pooled {r_['auroc_pooled_random_reference']:.3f} {r_['auroc']}; rejected {r_['rejected']}; false rejection pooled {r_['false_rejection_pooled']:.3f}", flush=True)
        # R47: every role decision of the one-to-one row in three categories: decided by the ADDRESS (the in-sentence winner is the
        # catalog winner of the original decoder), by RESTRICTION (the catalog winner is absent from the sentence), by ELIMINATION
        # (one-to-one assigned another candidate than the in-sentence winner; its margin is the floor, not evidence); with accuracy.
        cats_ax = {}; cells = collections.defaultdict(lambda: collections.Counter())
        for ax in report.AXES:
            if not split["axes"][ax]: continue
            c = collections.Counter()
            for it, p, p0 in zip(split["axes"][ax], P[ax], preds0[ax]):
                o = one_to_one(p, bwm.roles_at(it["depth"]))
                for r in bwm.roles_at(it["depth"]):
                    k = "elimination" if o["frame"][r] != p["frame"][r] else ("address" if p["frame"][r] == p0["frame"][r] else "restriction")
                    ok = o["frame"][r] == it["frame"][r]; c[k] += 1; c[k + "_correct"] += ok
                    if ax in NF and a.seed_start < 5: cells[f"T{it['template']}:{r}"][k] += 1; cells[f"T{it['template']}:{r}"][k + "_correct"] += ok
            cats_ax[ax] = dict(c)
        row["r47_categories"] = cats_ax
        print(f"  seed {seed} R47 decisions (address / restriction / elimination, correct in each): " + "; ".join(f"{ax} {v.get('address', 0)}/{v.get('restriction', 0)}/{v.get('elimination', 0)} ({v.get('address_correct', 0)}/{v.get('restriction_correct', 0)}/{v.get('elimination_correct', 0)})" for ax, v in cats_ax.items()), flush=True)
        if cells:   # design set only (Q27 / R47): per (template, role) cell of the new-formulation rows
            row["r47_cells"] = {k: dict(v) for k, v in sorted(cells.items())}
            print(f"  seed {seed} R47 cells, new formulations (address / restriction / elimination, correct): " + "; ".join(f"{k} {v.get('address', 0)}/{v.get('restriction', 0)}/{v.get('elimination', 0)} ({v.get('address_correct', 0)}/{v.get('restriction_correct', 0)}/{v.get('elimination_correct', 0)})" for k, v in sorted(cells.items())), flush=True)
        row["set"] = label; row["base_model"] = how
        json.dump(row, open(os.path.join(mainrepo.ROOT, "logs", f"results_r46_{tag}_seed{seed}.json"), "w")); rows.append(row)
    report.summarize(rows, name)
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    for nm in ("compiled", "explained"):
        print(f"  {nm} residual (one-to-one frames): AUROC pooled {ms([r['oow'][nm]['auroc_pooled_random_reference'] for r in rows])}; " + "; ".join(f"vs {s} {ms([r['oow'][nm]['auroc'][s] for r in rows])}" for s in ("random", "reference", "nonce_placeholder")))
        print(f"  {nm} residual: rejected at 2% calibration FR " + "; ".join(f"{s} {ms([r['oow'][nm]['rejected'][s] for r in rows])}" for s in ("random", "reference", "nonce_placeholder")) + f"; false rejection pooled {ms([r['oow'][nm]['false_rejection_pooled'] for r in rows])}")
    metrics.checklist(len(rows), "[x] decoder only, no retraining [x] expected effects written before the run (R46) [x] thresholds on calibration rows only (25) [x] " + label)
