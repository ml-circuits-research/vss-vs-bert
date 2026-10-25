"""compare.py: the comparison tables and figures, produced from logs/results_*.json only (INSTRUCTIONS section 8: never by hand).
Basic-world rows (results_reader_basic_*, results_transformer7b_*, results_bert_*): one table of mean role accuracy per axis and model,
mean ± std over seeds with the per-seed list and a 95% interval of the difference to the reader (t over seeds); the depth-3 per-role
table; the calibration table; the cost table. WikiSQL rows (results_wikisql_*): the data-efficiency table and figure. Real-data rows
(results_real_*, runs 8-10): paper/tables/real_<dataset>.md per condition (FULL, intent, slot heads, new-value heads, span F1, abstention),
the difference to the reader with its interval, the out-of-domain abstention table, FULL by depth (TOPv2), the cost table.
Run: python3 eval/compare.py [--out paper/tables]"""
import argparse, glob, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import mainrepo, basic_world as bwm
from common.report import AXES, COLS

def load(pattern):
    rows = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(mainrepo.ROOT, "logs", pattern))):
        r = json.load(open(p)); rows[r["model"]].append(r)
    return rows

def ms(v): return f"{np.mean(v):.3f} ± {np.std(v):.3f} [{' '.join(f'{x:.2f}' for x in v)}]"

def ci_diff(a, b):
    """95% interval of mean(a) - mean(b) over paired seeds (same seed list), t with n-1 degrees of freedom"""
    a, b = np.array(a), np.array(b); n = min(len(a), len(b)); d = a[:n] - b[:n]
    if n < 2: return f"{d.mean():+.3f} (n={n})"
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n); return f"{d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]"

def basic_tables(out):
    rows = load("results_reader_basic_seed*.json"); rows.update(load("results_transformer7b_*_seed*.json")); rows.update(load("results_bert_*_seed*.json"))
    if not rows: return
    ref = next((m for m in rows if m.startswith("VSS reader")), None); lines = ["# Basic world: mean role accuracy per axis (mean ± std over seeds [per seed])", "", "| model | " + " | ".join(COLS) + " | FULL depth 3 | FULL depth 3, trained labels (A B C X1 X2) | FULL depth 3, inner (A B C) |", "|---|" + "---|" * (len(COLS) + 3)]
    for m, rs in rows.items():
        cells = [ms([r["axes"][a]["mean_role"] for r in rs if r["axes"][a]]) if any(r["axes"][a] for r in rs) else "nan" for a in AXES]
        d3 = lambda k: ms([r["axes"]["depth3_unseen"][k] for r in rs]) if all(k in r["axes"]["depth3_unseen"] for r in rs) else "not stored (R6)"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {d3('full')} | {d3('full_trained')} | {d3('full_inner')} |")
    if ref:
        lines += ["", f"## Difference to {ref} (model minus reader), 95% interval over paired seeds", "", "| model | new ids | depth 3 (UNSEEN) | new form., UNSEEN trans. |", "|---|---|---|---|"]
        rr = rows[ref]
        for m, rs in rows.items():
            if m == ref: continue
            lines.append(f"| {m} | " + " | ".join(ci_diff([r["axes"][a]["mean_role"] for r in rs], [r["axes"][a]["mean_role"] for r in rr]) for a in ("new_ids", "depth3_unseen", "new_form_unseen_trans")) + " |")
    lines += ["", "## Depth 3 (UNSEEN), per role", "", "| model | " + " | ".join(bwm.ROLES) + " |", "|---|" + "---|" * 6]
    for m, rs in rows.items(): lines.append(f"| {m} | " + " | ".join(f"{np.mean([r['per_role_depth3'][k] for r in rs]):.3f}" for k in bwm.ROLES) + " |")
    lines += ["", "## Calibration (tau on calibration rows; coverage / confident-wrong / accuracy among answered; ECE rank-mapped)", "", "| model | axis | tau | coverage | confident-wrong | acc. answered | ECE |", "|---|---|---|---|---|---|---|"]
    for m, rs in rows.items():
        for ax in ("new_ids", "depth3_unseen"):
            c = [r["calibration"][ax] for r in rs]
            lines.append(f"| {m} | {ax} | {np.mean([r['calibration']['tau'] for r in rs]):.3f} | {np.mean([x['coverage'] for x in c]):.3f} | {np.mean([x['confident_wrong'] for x in c]):.3f} | {np.nanmean([x['accuracy_answered'] for x in c]):.3f} | {np.mean([x['ece_rank'] for x in c]):.3f} |")
    cost = [(m, rs) for m, rs in rows.items() if "cost" in rs[0]]
    if cost:
        lines += ["", "## Size and cost", "", "| model | parameters | MB on disk | CPU ms / sentence |", "|---|---|---|---|"]
        for m, rs in cost: lines.append(f"| {m} | {rs[0]['cost']['params']/1e6:.1f} M | {np.mean([r['cost']['bytes'] for r in rs])/1e6:.1f} | {np.mean([r['cost']['cpu_ms_per_sentence'] for r in rs]):.1f} |")
    open(os.path.join(out, "basic_world.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for m, rs in rows.items():
            c = rs[0]["calibration"]["depth3_unseen"]["curve"]; ax.plot([x["coverage"] for x in c], [x["accuracy_answered"] for x in c], marker=".", label=m)
        ax.set_xlabel("coverage"); ax.set_ylabel("accuracy among answered"); ax.set_title("depth 3 (UNSEEN): accuracy vs coverage, seed 0"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out, "coverage_depth3.png"), dpi=200)
    except Exception as e: print("figure skipped:", e)

def ref_tables(out):
    """reference world (run 4): FULL frame per axis, roles FULL at depth 3, predicate and event, occupancy, depth-3 FULL by template, calibration"""
    rows = load("results_ref_*_seed*.json")
    if not rows: return
    from common import reference_world as rw
    from common.report_ref import COLS as RCOLS
    lines = ["# Reference world (10b-quinquies): FULL frame (A, B, C or absent, speakers, predicate), mean ± std over seeds [per seed]", "",
             "| model | " + " | ".join(RCOLS) + " | roles FULL depth 3 | predicate seen | event seen |", "|---|" + "---|" * (len(RCOLS) + 3)]
    for m, rs in rows.items():
        lines.append(f"| {m} | " + " | ".join(ms([r["axes"][a]["full"] for r in rs]) if all(r["axes"][a] for r in rs) else "nan" for a in rw.AXES)
                     + f" | {ms([r['axes']['depth3_unseen']['roles_full'] for r in rs])} | {ms([r['axes']['seen']['per_role']['pred'] for r in rs])} | {ms([r['axes']['seen']['event'] for r in rs])} |")
    lines += ["", "## Occupancy of the recipient (share read correctly as present / absent)", "", "| model | known filler | unknown filler | not given |", "|---|---|---|---|"]
    for m, rs in rows.items(): lines.append(f"| {m} | " + " | ".join(f"{np.mean([r['occupancy'][k] for r in rs]):.3f}" for k in rs[0]["occupancy"]) + " |")
    keys = list(next(iter(rows.values()))[0]["depth3_full_by_template"])
    lines += ["", "## Depth 3 (UNSEEN) FULL by template (passives T2, T3, T5, T14; fronted recipient T9)", "", "| model | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for m, rs in rows.items(): lines.append(f"| {m} | " + " | ".join(f"{np.mean([r['depth3_full_by_template'].get(k, np.nan) for r in rs]):.3f}" for k in keys) + " |")
    lines += ["", "## Calibration (tau on calibration rows)", "", "| model | axis | tau | coverage | confident-wrong |", "|---|---|---|---|---|"]
    for m, rs in rows.items():
        for ax in ("depth3_unseen", "unseen_trans", "new_ids"):
            c = [r["calibration"][ax] for r in rs]
            lines.append(f"| {m} | {ax} | {np.mean([r['calibration']['tau'] for r in rs]):.3f} | {np.mean([x['coverage'] for x in c]):.3f} | {np.mean([x['confident_wrong'] for x in c]):.3f} |")
    open(os.path.join(out, "reference_world.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

def wikisql_tables(out):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(glob.glob(os.path.join(mainrepo.ROOT, "logs", "results_wikisql_*_seed*.json"))):
        r = json.load(open(p)); rows[r["model"]][r["size"]].append(r)
    if not rows: return
    lines = ["# WikiSQL (dev-report): selected column / aggregator by training size", "", "| model | size | sel | agg | both | sel coverage | sel confident-wrong |", "|---|---|---|---|---|---|---|"]
    for m, by in rows.items():
        for size, rs in sorted(by.items()):
            d = [r["dev_report"] for r in rs]
            lines.append(f"| {m} | {size} | {ms([x['sel'] for x in d])} | {ms([x['agg'] for x in d])} | {ms([x['both'] for x in d])} | {np.mean([x['cal_sel']['coverage'] for x in d]):.3f} | {np.mean([x['cal_sel']['confident_wrong'] for x in d]):.3f} |")
    lines += ["", "Reference (main repository, dev): reader sel 0.632 / 0.760 / 0.816 at 1k / 5k / 56k; transformer 0.415 / 0.510 / 0.543; BoW+LR 0.475 / 0.502 / 0.499."]
    open(os.path.join(out, "wikisql.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

def real_tables(out):
    """the real-data round (INSTRUCTIONS_REAL_2026-10-24.md): one file per dataset, paper/tables/real_<dataset>.md, from
    logs/results_real_<dataset>_<condition>_<tag>[_temperature]_<size>_seed<k>.json only"""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))         # dataset -> (condition, model, variant, size) -> rows
    for p in sorted(glob.glob(os.path.join(mainrepo.ROOT, "logs", "results_real_*_seed*.json"))):
        r = json.load(open(p)); rows[r["dataset"]][(r["condition"], r["model"], r.get("variant") or ("own margins" if r["model"].startswith("VSS") else "softmax"), r["size"] or "full")].append(r)
    for ds, by in rows.items():
        key = lambda k: (k[0], 0 if k[1].startswith("VSS") else 1, k[1], k[2], str(k[3]))
        L = [f"# Real data: {ds} (head-token frame: intent + the last token of the first span of each slot type; mean ± std over seeds [per seed])", "",
             "Thresholds on the dev calibration half by the written rule (confident-wrong <= 0.05 per decision kind). 'softmax' / 'temperature' = the "
             "tagger's margins without / with temperature scaling fitted on the calibration half; the reader's rows carry its own margins (top1 - top2 cosine, presence margins).", ""]
        for cond in sorted({k[0] for k in by}):
            ks = sorted([k for k in by if k[0] == cond], key=key); sets = list(by[ks[0]][0]["sets"])
            L += [f"## Condition `{cond}`", "", "| model | margins | size | set | n | FULL | intent | slot heads | new-value heads | nested intents | span F1 | frame coverage | frame confident-wrong | intent abstain |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for k in ks:
                rs = by[k]
                for st in sets:
                    S = [r["sets"].get(st) for r in rs if r["sets"].get(st)]
                    if not S: continue
                    g = lambda f: ms([f(x) for x in S if f(x) is not None]) if any(f(x) is not None for x in S) else "-"
                    sp = [r["spans"][st]["f1"] for r in rs if st in r.get("spans", {})]
                    L.append(f"| {k[1]} | {k[2]} | {k[3]} | {st} | {S[0]['n']} | {g(lambda x: x['full'])} | {g(lambda x: x['intent'])} | {g(lambda x: x['slot_heads_on_gold'])} | {g(lambda x: x.get('new_value_heads'))} | "
                             f"{g(lambda x: x.get('nested_intents'))} | {ms(sp) if sp else '-'} | {np.mean([x['calibration']['frame']['coverage'] for x in S]):.3f} | {np.mean([x['calibration']['frame']['confident_wrong'] for x in S]):.3f} | {np.mean([x['abstain']['intent_margin_below_tau'] for x in S]):.3f} |")
            ref = next((k for k in ks if k[1].startswith("VSS")), None)
            if ref:
                L += ["", f"Difference of FULL to the reader (model minus reader), 95% interval over paired seeds, per set and size", "", "| model | margins | size | set | difference |", "|---|---|---|---|---|"]
                for k in ks:
                    if k[1].startswith("VSS"): continue
                    rr = by.get((cond, ref[1], ref[2], k[3]))
                    if not rr: continue
                    for st in sets:
                        a_ = [r["sets"][st]["full"] for r in by[k] if r["sets"].get(st)]; b_ = [r["sets"][st]["full"] for r in rr if r["sets"].get(st)]
                        if a_ and b_: L.append(f"| {k[1]} | {k[2]} | {k[3]} | {st} | {ci_diff(a_, b_)} |")
            if cond == "ood_intents":
                L += ["", "Out-of-domain intents (every reading of a held-out intent is wrong): the share ABSTAINED (intent margin below the intent tau set on in-domain calibration rows) "
                      "and the share mapped CONFIDENTLY to a known intent (1 - abstained); the reader's tool status counts and mean residual beside it", "",
                      "| model | margins | set | abstained (intent) | confidently mapped | abstained (frame) | AUROC of the intent margin, in-domain vs this set | tool status | residual (mean) |", "|---|---|---|---|---|---|---|---|---|"]
                for k in ks:
                    for st in [x for x in sets if x.startswith("ood") or x.endswith("in_domain")]:
                        S = [r["sets"][st] for r in by[k] if r["sets"].get(st)]
                        if not S: continue
                        ab = [x["abstain"]["intent_margin_below_tau"] for x in S]; tl = [r["tool"].get(st) for r in by[k] if r.get("tool", {}).get(st)]
                        au = [r.get("ood_auroc_intent_margin", {}).get(st) for r in by[k]]; au = [x for x in au if x is not None]
                        L.append(f"| {k[1]} | {k[2]} | {st} | {ms(ab)} | {ms([1 - x for x in ab])} | {ms([x['abstain']['frame_margin_below_tau'] for x in S])} | {ms(au) if au else '-'} | "
                                 f"{tl[0]['status'] if tl else '-'} | {np.mean([t['residual_mean'] for t in tl]) if tl else float('nan'):.3f} |")
            if cond == "depth":
                L += ["", "FULL by depth (depth = intents nested under the top one; training at depth <= 1)", "", "| model | margins | set | FULL by depth (mean over seeds; n) |", "|---|---|---|---|"]
                for k in ks:
                    for st in sets:
                        S = [r["sets"][st] for r in by[k] if r["sets"].get(st)]
                        if S: L.append(f"| {k[1]} | {k[2]} | {st} | " + ", ".join(f"depth {d}: {np.mean([x['full_by_depth'][d][0] for x in S if d in x['full_by_depth']]):.3f} (n {S[0]['full_by_depth'][d][1]})" for d in S[0]["full_by_depth"]) + " |")
            L.append("")
        cost = {(k[1], k[3]): v[0]["cost"] for k, v in by.items() if k[0] == "base" and v and "cost" in v[0]}
        if cost:
            L += ["## Size and cost (condition base)", "", "| model | size | parameters | MB on disk | CPU ms / sentence |", "|---|---|---|---|---|"]
            for (m, sz), c in sorted(cost.items(), key=lambda x: (not x[0][0].startswith("VSS"), x[0][0], str(x[0][1]))):
                L.append(f"| {m} | {sz} | {c['params']/1e6:.2f} M | {c['bytes']/1e6:.1f} | {c['cpu_ms_per_sentence']:.1f} |")
        open(os.path.join(out, f"real_{ds}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(mainrepo.ROOT, "paper", "tables")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); basic_tables(a.out); ref_tables(a.out); wikisql_tables(a.out); real_tables(a.out)
