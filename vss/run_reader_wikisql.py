"""run_reader_wikisql.py (run 5, reader): the reader's WikiSQL row computed IN THIS REPOSITORY, on the same rows and with the same scorer
and threshold rule as models/bert_wikisql.py. The model, the world and the training are the main repository's, imported unmodified:
v11_wikisql.World (fixed identities, the SQL state), v11_wikisql.RealModel (path + bigram + soft chainL, gates learned), v10_controls.skipgram
(the distributional categories). What is NOT importable is v12_wikisql_baselines.py's setup, which lives inside its `__main__`: the
vocabulary over train + dev + test, build_categories (40 frequent tokens + 48 skip-gram clusters on the WHOLE training pool only),
targets() and train_ours(). Those lines are reproduced here verbatim in substance (same calls, same seeds, same defaults: D 512, batch 32,
lr 0.02, epochs 60 / 20 / 3 at 1k / 5k / 56k) -- the smallest wrapper; any difference to v12 is a bug of this file.

Per-item predictions: the selected column = argmax cosine over the table's column bundles after unbinding R_SEL, the aggregator = argmax
over the six aggregator identities after unbinding R_AGG (v12's full_eval), each with its MARGIN = top1 - top2 cosine. Thresholds: the
written rule (eval/metrics.set_tau, confident-wrong <= 0.05 on the dev calibration half), reported on the report half; test once at 56k.
Exit check: the selected column on dev within seed variance of RESULTS 8d (0.632 / 0.760 / 0.816 at 1k / 5k / 56k on the whole dev set).
Run: python3 vss/run_reader_wikisql.py [--sizes 1000,5000,56355] [--seeds 5] [--D 512] [--device cuda] [--test]"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo, wikisql_data
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v11_wikisql as v11, v10_controls as v10
from eval import metrics

def build(a, wd, dev_):
    tok = v11.tok; vocab = {}; vid = lambda w: vocab.setdefault(w, len(vocab))
    for rows, tabs in ((wd["train_all"], wd["ttabs"]), (wd["dev"], wd["dtabs"]), (wd["test"], wd["tetabs"])):
        for r in rows:
            for w in tok(r["question"]): vid(w)
            for h in tabs[r["table_id"]]:
                for w in tok(h): vid(w)
            for c in r["sql"]["conds"]:
                for w in tok(str(c[2])): vid(w)
    V = len(vocab); pool, ttabs = wd["pool"], wd["ttabs"]
    freq = collections.Counter(w for r in pool for w in tok(r["question"])); freq.update(w for r in pool for h in ttabs[r["table_id"]] for w in tok(h))
    singles = [w for w, _ in freq.most_common(a.n_freq)]
    sents = [tok(r["question"]) for r in pool] + [tok(h) for r in pool[:20000] for h in ttabs[r["table_id"]]]
    print(f"clustering data: {len(sents)} sequences from {len(pool)} training-pool questions only; dev/test/val not used.")
    emb = v10.skipgram([s for s in sents if len(s) > 1], vocab, dim=64, window=2, epochs=3, device=dev_, seed=0)
    X = torch.tensor(emb, device=dev_); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); gk = torch.Generator().manual_seed(0)
    k = min(a.k_clusters, V); C = X[torch.randperm(V, generator=gk)[:k].to(dev_)].clone()
    for it in range(25):
        asg = (X @ C.T).argmax(1)
        for c in range(k):
            mem = X[asg == c]
            if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
    asg = (X @ C.T).argmax(1).cpu().numpy()
    cat_of = torch.tensor(np.array([a.k_clusters + singles.index(w) if w in singles else asg[i] for w, i in sorted(vocab.items(), key=lambda x: x[1])]))
    K = a.k_clusters + len(singles); world = v11.World(vocab, a.D, dev_, 0)
    hid = lambda tabs, tid: [[vid(w) for w in tok(h)] for h in tabs[tid]]
    def targets(rows, tabs):
        return torch.stack([world.sql_state({"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [[c, o, [vid(w) for w in tok(str(v))]] for c, o, v in r["sql"]["conds"]]}, hid(tabs, r["table_id"])) for r in rows])
    qids = lambda r: [vid(w) for w in tok(r["question"])]
    return dict(vocab=vocab, cat_of=cat_of, K=K, world=world, hid=hid, targets=targets, qids=qids)

def train_ours(b, rows, ttabs, seed, n_ep, a, dev_):
    m = v11.RealModel(b["K"], a.D, dev_, seed); opt = torch.optim.Adam(m.params, lr=a.lr); tgt = b["targets"](rows, ttabs)
    seqs = [b["qids"](r) for r in rows]; order = np.argsort([len(s) for s in seqs]); g = np.random.default_rng(seed)
    for ep in range(n_ep):
        buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]; g.shuffle(buckets)
        for bb in buckets:
            ids, mask = v11.pad([seqs[i] for i in bb], dev_); cats = b["cat_of"][ids.cpu()].to(dev_)
            loss = -v11.cosb(m.states(b["world"].F, cats, ids, mask), tgt[torch.tensor(bb, device=dev_)]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def predict(m, b, rows, tabs, dev_):
    world = b["world"]; out = []; aggF = world.F[torch.tensor(world.AGG, device=dev_)]
    for i in range(0, len(rows), 256):
        chunk = rows[i:i + 256]; ids, mask = v11.pad([b["qids"](r) for r in chunk], dev_); S_all = m.states(world.F, b["cat_of"][ids.cpu()].to(dev_), ids, mask)
        for S, r in zip(S_all, chunk):
            cols = b["hid"](tabs, r["table_id"]); cb = torch.stack([world.bundle(c) for c in cols])
            ss = v11.cosb((S * torch.conj(world.R["SEL"])).unsqueeze(0).expand(len(cols), -1), cb).cpu().numpy()
            sa = v11.cosb((S * torch.conj(world.R["AGG"])).unsqueeze(0).expand(6, -1), aggF).cpu().numpy()
            ts, ta = np.argsort(-ss), np.argsort(-sa)
            out.append({"frame": {"sel": int(ts[0]), "agg": int(ta[0])}, "margins": {"sel": float(ss[ts[0]] - (ss[ts[1]] if len(ss) > 1 else 0.0)), "agg": float(sa[ta[0]] - sa[ta[1]])}})
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,5000,56355"); ap.add_argument("--epochs", default="60,20,3"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--D", type=int, default=512); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--n_dev", type=int, default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--test", action="store_true"); ap.add_argument("--n_val", type=int, default=2000)
    a = ap.parse_args(); dev_ = a.device; sizes = [int(x) for x in a.sizes.split(",")]; epochs = dict(zip(sizes, [int(x) for x in a.epochs.split(",")]))
    import models.bert_wikisql as bw_         # the same scorer function as the BERT rows
    wd = wikisql_data.load(n_val=a.n_val, n_dev=a.n_dev, with_test=a.test)
    print(f"run_reader_wikisql: main repository {mainrepo.check_pinned()}; device {dev_}; D {a.D}; sizes {sizes}, epochs {epochs}; seeds {a.seeds}; pool {len(wd['pool'])}, dev calibration {len(wd['cal'])} / report {len(wd['rep'])}; test {len(wd['test'])}")
    print("regime: question only (no table at the input), the table's columns only as decoding candidates (RESULTS 8c / 8d); chance sel ~0.15, majority aggregator 0.72")
    b = build(a, wd, dev_); name = "VSS reader (v11 RealModel)"; results = {}
    for n in sizes:
        rows = wd["pool"][:n]; per = []
        for seed in range(a.seeds):
            t0 = time.time(); m = train_ours(b, rows, wd["ttabs"], seed, epochs[n], a, dev_)
            pc = predict(m, b, wd["cal"], wd["dtabs"], dev_)
            tau_sel = metrics.set_tau(metrics.role_decisions([{"frame": {"sel": r["sql"]["sel"]}} for r in wd["cal"]], pc, ["sel"]), 0.05)
            tau_agg = metrics.set_tau(metrics.role_decisions([{"frame": {"agg": r["sql"]["agg"]}} for r in wd["cal"]], pc, ["agg"]), 0.05)
            pr = predict(m, b, wd["rep"], wd["dtabs"], dev_)
            row = {"model": name, "size": n, "seed": seed, "tau": {"sel": tau_sel, "agg": tau_agg}, "dev_report": bw_.score(wd["rep"], pr, tau_sel, tau_agg),
                   "report_items": [{"i": 2 * j + 1, "pred": p} for j, p in enumerate(pr)]}
            if a.test and n == max(sizes): row["test"] = bw_.score(wd["test"], predict(m, b, wd["test"], wd["tetabs"], dev_), tau_sel, tau_agg)
            row["cost"] = {"params": sum(p.numel() for p in m.params), "bytes": int(sum(p.numel() for p in m.params) * 4)}
            d = row["dev_report"]; print(f"  size {n} seed {seed} ({time.time() - t0:.0f}s) dev-report sel {d['sel']:.3f} agg {d['agg']:.3f} both {d['both']:.3f} | tau_sel {tau_sel:.3f}: cov {d['cal_sel']['coverage']:.2f} cw {d['cal_sel']['confident_wrong']:.3f}" + (f" | test sel {row['test']['sel']:.3f}" if "test" in row else ""), flush=True)
            with open(os.path.join(mainrepo.ROOT, "logs", f"results_wikisql_reader_{n}_seed{seed}.json"), "w") as f: json.dump(row, f)
            per.append(row)
        results[n] = per
    print(f"\n== {name}: WikiSQL by training size (dev-report; mean ± std over {a.seeds} seeds, per-seed list) ==")
    for n, rs in results.items():
        for k in ("sel", "agg", "both"):
            v = [r["dev_report"][k] for r in rs]; print(f"  {n:6d} {k:5s} {np.mean(v):.3f}±{np.std(v):.3f} [{' '.join(f'{x:.3f}' for x in v)}]")
    print("reference (main repository, RESULTS 8d, whole dev): reader sel 0.632 / 0.760 / 0.816 at 1k / 5k / 56k")
    metrics.checklist(a.seeds, "[x] the same training pool and dev halves as the BERT rows [x] test only once at the largest size")
