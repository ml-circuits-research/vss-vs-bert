"""error_analysis.py (run 6): the 50-item confident-error analysis of the main repository's RESULTS 11-octies, repeated for the reader and
for BERT on the same items. For each model, from logs/results_wikisql_<tag>_<size>_seed0.json (written by vss/run_reader_wikisql.py and
models/bert_wikisql.py): the dev REPORT-half items whose selected column was answered with margin >= the model's tau (set on the
calibration half) and is wrong; the first 50 in dev order (fixed); one TSV row each with the question, the table's columns, the gold and
the predicted column, the margin, whether the gold column's name occurs in the question, and an EMPTY 'kind' column for a person to fill
with one of: reader-error | gold-error | ambiguous | synonym | other. Both models' files list the same columns, side by side in paper/.

  python3 eval/error_analysis.py dump --size 56355 --tags reader,bert-base-uncased_pre [--out paper/error_analysis]
  python3 eval/error_analysis.py summarize paper/error_analysis/*.tsv      # after the 'kind' column is filled

The summary prints, per file, the counts per kind and the confident-wrong rate when the gold column's name is in the question vs not
(the paper: 0.023 vs 0.169 for the reader). Unfilled rows are counted as 'unfilled' and the summary says so."""
import argparse, collections, csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo, wikisql_data

KINDS = ["reader-error", "gold-error", "ambiguous", "synonym", "other"]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

def dump(a):
    wd = wikisql_data.load(n_dev=a.n_dev, with_test=False); dev, dtabs = wd["dev"], wd["dtabs"]; os.makedirs(a.out, exist_ok=True)
    for tag in a.tags.split(","):
        p = os.path.join(mainrepo.ROOT, "logs", f"results_wikisql_{tag}_{a.size}_seed0.json")
        if not os.path.exists(p): print(f"missing {p}: run the model first"); continue
        row = json.load(open(p)); tau = row["tau"]["sel"]; items = sorted(row["report_items"], key=lambda x: x["i"])
        conf_wrong, n_ans, in_q = [], 0, collections.Counter()
        for it in items:
            r = dev[it["i"]]; cols = dtabs[r["table_id"]]; pred = it["pred"]["frame"]["sel"]; m = it["pred"]["margins"]["sel"]; gold = r["sql"]["sel"]
            name_in_q = set(tok(cols[gold])) <= set(tok(r["question"]))
            if m >= tau:
                n_ans += 1; in_q[(name_in_q, pred == gold)] += 1
                if pred != gold: conf_wrong.append((it["i"], r, cols, gold, pred, m, name_in_q))
        out = os.path.join(a.out, f"confident_errors_{tag}_{a.size}_seed0.tsv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["dev_index", "question", "columns", "gold", "predicted", "margin", "gold_name_in_question", "kind"])
            for i, r, cols, gold, pred, m, nq in conf_wrong[:a.n]: w.writerow([i, r["question"], " | ".join(cols), cols[gold], cols[pred], f"{m:.4f}", int(nq), ""])
        def rate(flag):
            tot = in_q[(flag, True)] + in_q[(flag, False)]; return f"{in_q[(flag, False)] / tot:.3f} of {tot}" if tot else "n/a"
        print(f"{tag}: tau {tau:.3f}; answered {n_ans} of {len(items)}; confident-wrong {len(conf_wrong)}; written the first {min(a.n, len(conf_wrong))} to {os.path.relpath(out, mainrepo.ROOT)}"
              f"; wrong among answered, gold name in question {rate(True)}, not in question {rate(False)}")

def summarize(a):
    for p in a.files:
        rows = list(csv.DictReader(open(p), delimiter="\t")); c = collections.Counter((r["kind"].strip() or "unfilled") for r in rows)
        bad = [k for k in c if k not in KINDS + ["unfilled"]]
        print(f"{os.path.basename(p)}: {len(rows)} items; " + ", ".join(f"{k} {c[k]}" for k in KINDS + ["unfilled"] if c[k])
              + (f"; UNKNOWN kinds {bad} (allowed: {KINDS})" if bad else "") + ("; NOTE: unfilled rows remain" if c["unfilled"] else ""))
        nq = collections.Counter((r["gold_name_in_question"], r["kind"].strip() or "unfilled") for r in rows)
        print("   by gold name in question: " + "; ".join(f"{'in' if k[0] == '1' else 'not in'} question / {k[1]}: {v}" for k, v in sorted(nq.items())))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--size", type=int, default=56355); d.add_argument("--tags", default="reader,bert-base-uncased_pre")
    d.add_argument("--n", type=int, default=50); d.add_argument("--n_dev", type=int, default=None); d.add_argument("--out", default=os.path.join(mainrepo.ROOT, "paper", "error_analysis"))
    s = sub.add_parser("summarize"); s.add_argument("files", nargs="+")
    a = ap.parse_args(); dump(a) if a.cmd == "dump" else summarize(a)
