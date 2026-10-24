"""compare.py: the comparison tables and figures, produced from logs/results_*.json only (INSTRUCTIONS section 8: never by hand).
Basic-world rows (results_reader_basic_*, results_transformer7b_*, results_bert_*): one table of mean role accuracy per axis and model,
mean ± std over seeds with the per-seed list and a 95% interval of the difference to the reader (t over seeds); the depth-3 per-role
table; the calibration table; the cost table. WikiSQL rows (results_wikisql_*): the data-efficiency table and figure.
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
    ref = next((m for m in rows if m.startswith("VSS reader")), None); lines = ["# Basic world: mean role accuracy per axis (mean ± std over seeds [per seed])", "", "| model | " + " | ".join(COLS) + " | FULL depth 3 |", "|---|" + "---|" * (len(COLS) + 1)]
    for m, rs in rows.items():
        cells = [ms([r["axes"][a]["mean_role"] for r in rs if r["axes"][a]]) if any(r["axes"][a] for r in rs) else "nan" for a in AXES]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {ms([r['axes']['depth3_unseen']['full'] for r in rs])} |")
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(mainrepo.ROOT, "paper", "tables")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); basic_tables(a.out); ref_tables(a.out); wikisql_tables(a.out)
