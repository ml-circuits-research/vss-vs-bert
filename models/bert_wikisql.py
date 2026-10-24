"""bert_wikisql.py (run 5): a pretrained encoder (or the same architecture from scratch) fine-tuned on WikiSQL for the two decisions the
paper reports with the table's columns as candidates (RESULTS 8c / 8d): the SELECTED COLUMN (a score per (question, column) pair from
the encoder's [CLS] on '[CLS] question [SEP] column header [SEP]', softmax over the table's columns) and the AGGREGATOR (6-way from
'[CLS] question [SEP]'). Training sizes 1k / 5k / 56k (the paper's curve); dev split into calibration (even rows: thresholds, rule 25)
and report (odd rows); test once at 56k. Margins: top1 - top2 probability over the table's columns (sel) and over the 6 aggregators.
Data: common/wikisql_data.py, the training pool of the main repository's v12_wikisql_baselines.py (the same questions at every size for
every model) and the paper's dev / test files. Reference numbers (main repository): reader sel 0.816 / 0.760 / 0.632 at 56k / 5k / 1k
(dev), transformer 0.543 / 0.510 / 0.415, BoW+LR 0.499 / 0.502 / 0.475; aggregator reader 0.80, transformer 0.89.
Run: python3 models/bert_wikisql.py --model bert-base-uncased [--scratch] [--sizes 1000,5000,56355] [--seeds 5] [--device cuda]"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import v11_wikisql as v11
from eval import metrics

from common import wikisql_data

class WikiModel(nn.Module):
    def __init__(self, name, scratch):
        super().__init__(); cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_config(cfg) if scratch else AutoModel.from_pretrained(name)
        self.sel = nn.Linear(cfg.hidden_size, 1); self.agg = nn.Linear(cfg.hidden_size, 6); self.drop = nn.Dropout(0.1)
    def cls(self, ids, att): return self.drop(self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0])

def batch_pairs(tok, rows, tabs, device, max_len=64):
    """one question with all its columns: returns (sel_ids, sel_att, group index per pair, agg_ids, agg_att)"""
    q = [r["question"] for r in rows]; cols = [tabs[r["table_id"]] for r in rows]
    pq = [r["question"] for r, cs in zip(rows, cols) for _ in cs]; pc = [c for cs in cols for c in cs]; grp = [i for i, cs in enumerate(cols) for _ in cs]
    e = tok(pq, pc, return_tensors="pt", padding=True, truncation=True, max_length=max_len); ea = tok(q, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return e["input_ids"].to(device), e["attention_mask"].to(device), grp, ea["input_ids"].to(device), ea["attention_mask"].to(device), cols

def forward(model, tok, rows, tabs, device):
    ids, att, grp, aid, aatt, cols = batch_pairs(tok, rows, tabs, device)
    s = model.sel(model.cls(ids, att)).squeeze(-1); a = model.agg(model.cls(aid, aatt))
    out = []; k = 0
    for i, cs in enumerate(cols): out.append(s[k:k + len(cs)]); k += len(cs)
    return out, a

def train(model, tok, rows, tabs, lr, epochs, batch, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(rows) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(rows), generator=g).tolist()
        for i in range(0, len(rows), batch):
            chunk = [rows[j] for j in perm[i:i + batch]]; sel_sc, agg_lg = forward(model, tok, chunk, tabs, device)
            loss = ce(agg_lg, torch.tensor([r["sql"]["agg"] for r in chunk], device=device))
            loss = loss + sum(ce(s.unsqueeze(0), torch.tensor([r["sql"]["sel"]], device=device)) for s, r in zip(sel_sc, chunk)) / len(chunk)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def predict(model, tok, rows, tabs, device, batch=64):
    model.eval(); out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]; sel_sc, agg_lg = forward(model, tok, chunk, tabs, device); pa = torch.softmax(agg_lg, -1).cpu().numpy()
        for s, p, r in zip(sel_sc, pa, chunk):
            ps = torch.softmax(s, -1).cpu().numpy(); ts = np.argsort(-ps); ta = np.argsort(-p)
            out.append({"frame": {"sel": int(ts[0]), "agg": int(ta[0])}, "margins": {"sel": float(ps[ts[0]] - (ps[ts[1]] if len(ps) > 1 else 0.0)), "agg": float(p[ta[0]] - p[ta[1]])}})
    return out

def score(rows, preds, tau_sel, tau_agg):
    items = [{"frame": {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"]}} for r in rows]
    sc = metrics.score_axis(items, preds, ["sel", "agg"])
    dsel = metrics.role_decisions(items, preds, ["sel"]); dagg = metrics.role_decisions(items, preds, ["agg"])
    return {"sel": sc["per_role"]["sel"], "agg": sc["per_role"]["agg"], "both": sc["full"], "n": len(rows),
            "cal_sel": metrics.calibration(dsel, tau_sel), "cal_agg": metrics.calibration(dagg, tau_agg), "curve_sel": metrics.coverage_curve(dsel)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--sizes", default="1000,5000,56355")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--device", default="cuda")
    ap.add_argument("--lrs", default=None); ap.add_argument("--epochs", default=None); ap.add_argument("--n_dev", type=int, default=8421); ap.add_argument("--test", action="store_true"); ap.add_argument("--n_val", type=int, default=2000)
    a = ap.parse_args()
    lrs = [float(x) for x in (a.lrs or ("1e-4,3e-4,1e-3" if a.scratch else "2e-5,3e-5,5e-5")).split(",")]
    eps = [int(x) for x in (a.epochs or ("10,20,40" if a.scratch else "3,5,10")).split(",")]
    name = f"{a.model}{' from scratch' if a.scratch else ' pretrained'}"; tag = f"wikisql_{a.model.replace('/', '_')}_{'scratch' if a.scratch else 'pre'}"
    wd = wikisql_data.load(n_val=a.n_val, n_dev=a.n_dev, with_test=a.test); train_rows, ttabs, dtabs, cal, rep = wd["pool"], wd["ttabs"], wd["dtabs"], wd["cal"], wd["rep"]
    test, xtabs = wd["test"], wd["tetabs"]       # the training pool of v12 (train permuted by rng(0), 2,000 validation questions held out), as the reader
    print(f"bert_wikisql: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; sizes {a.sizes}; seeds {a.seeds}; grid lr {lrs} x epochs {eps}; dev calibration {len(cal)} / report {len(rep)}; test {len(test)}")
    print("regime: questions read with the table's columns as candidates (as RESULTS 8c); chance sel ~0.15 (mean 1/columns), majority aggregator 0.72")
    tok = AutoTokenizer.from_pretrained(a.model); results = {}
    for size in [int(x) for x in a.sizes.split(",")]:
        rows_n = train_rows[:size]; chosen = None; per_seed = []
        for seed in range(a.seeds):
            torch.manual_seed(seed); best = None
            for lr in (lrs if seed == 0 else [chosen[0]]):
                for ep in (eps if seed == 0 else [chosen[1]]):
                    model = WikiModel(a.model, a.scratch).to(a.device); t0 = time.time(); train(model, tok, rows_n, ttabs, lr, ep, a.batch, a.device, seed)
                    pc = predict(model, tok, cal, dtabs, a.device); vsel = np.mean([p["frame"]["sel"] == r["sql"]["sel"] for p, r in zip(pc, cal)])
                    print(f"  size {size} seed {seed} lr={lr:g} epochs={ep} ({time.time() - t0:.0f}s) calibration-half sel {vsel:.3f}", flush=True)
                    if best is None or vsel > best[0]: best = (vsel, lr, ep, model, pc)
                    else: del model
            if seed == 0: chosen = (best[1], best[2]); print(f"  => size {size} chosen config: lr={chosen[0]:g}, epochs={chosen[1]} (by selected column on the calibration half, seed 0)")
            model, pc = best[3], best[4]
            tau_sel = metrics.set_tau(metrics.role_decisions([{"frame": {"sel": r["sql"]["sel"]}} for r in cal], pc, ["sel"]), 0.05)
            tau_agg = metrics.set_tau(metrics.role_decisions([{"frame": {"agg": r["sql"]["agg"]}} for r in cal], pc, ["agg"]), 0.05)
            pr = predict(model, tok, rep, dtabs, a.device); row = {"model": name, "size": size, "seed": seed, "tau": {"sel": tau_sel, "agg": tau_agg}, "dev_report": score(rep, pr, tau_sel, tau_agg), "report_items": [{"i": 2 * j + 1, "pred": p} for j, p in enumerate(pr)]}
            if a.test and size == max(int(x) for x in a.sizes.split(",")): row["test"] = score(test, predict(model, tok, test, xtabs, a.device), tau_sel, tau_agg)
            d = row["dev_report"]; print(f"  size {size} seed {seed} dev-report sel {d['sel']:.3f} agg {d['agg']:.3f} both {d['both']:.3f} | tau_sel {tau_sel:.3f}: cov {d['cal_sel']['coverage']:.2f} cw {d['cal_sel']['confident_wrong']:.3f}" + (f" | test sel {row['test']['sel']:.3f} agg {row['test']['agg']:.3f}" if "test" in row else ""), flush=True)
            with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_{size}_seed{seed}.json"), "w") as f: json.dump(row, f)
            per_seed.append(row); del model
        results[size] = per_seed
    print(f"\n== {name}: WikiSQL by training size (dev-report; mean ± std over {a.seeds} seeds, per-seed list) ==")
    for size, rs in results.items():
        for k in ("sel", "agg", "both"):
            v = [r["dev_report"][k] for r in rs]; print(f"  {size:6d} {k:5s} {np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]")
        cw = [r["dev_report"]["cal_sel"]["confident_wrong"] for r in rs]; cov = [r["dev_report"]["cal_sel"]["coverage"] for r in rs]
        print(f"  {size:6d} sel abstention (tau on calibration half, cw <= 0.05 rule): coverage {np.mean(cov):.3f} confident-wrong {np.mean(cw):.3f}")
    print("reference (main repository, dev): reader sel 0.632 / 0.760 / 0.816 at 1k / 5k / 56k; transformer 0.415 / 0.510 / 0.543; BoW+LR 0.475 / 0.502 / 0.499; the size at which each model reaches 0.75 is the data-efficiency number")
    metrics.checklist(a.seeds, "[x] dev calibration/report halves as the paper (even/odd) [x] test only once at the largest size")
