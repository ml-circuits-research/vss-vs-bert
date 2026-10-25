"""reader_growth.py (responses R49 / R50, declared in RESULTS 'Deviation R49'): the reader's non-compositional legend-growth case. The
reader is retrained IN FULL (its only regime) exactly as 1a (v8_gates.V8, the same seed, generators, N 30,000, batch 8, lr 0.02, gates
learned from closed), with k = 50 or 500 annotated template-4 sentences of data/growth_seed<s>.json ADDED to its training sentences at
fixed random positions (rng 75_000 + seed). The training loop is V8.train's, replicated here line for line so that the k sentences can be
added (the main repository is not modified); with k = 0 it must reproduce the stored row EQUAL (--check). Reported: wall-clock of the
training (cuda synchronised; run ALONE on the GPU), every report axis before (the saved reader of the row) and after, the template-4 cell
of the new-formulation rows before and after, the number of examples and of hyperparameter choices (0). Writes
logs/results_growth_reader_<tag>_k<k>_seed<s>.json.
Run: ./run.sh vss/reader_growth.py --seed_start 0|5 --seeds 5 [--ks 50,500] [--check]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode, report
from eval import metrics
from eval.reader_models import get_reader

def train_plus(m, world, n_sent, extra, seed, lr=0.02, templates=None, persons=None, objects=None, batch=8, depth_train=2):
    """V8.train (v8_gates.py lines 104-117 at e531283) with the extra (tokens, ev) sentences inserted at fixed random positions"""
    w = m.w; sents, targets = [], []
    for n in range(n_sent):
        d = int(m.rng.integers(depth_train + 1)); ev = w.event(persons, objects, depth=d)
        sents.append(w.render(ev, m.rng.choice(list(templates)))); targets.append(world.state(ev))
    if extra:
        pos = np.sort(np.random.default_rng(75_000 + seed).choice(n_sent + len(extra), size=len(extra), replace=False)); S2, T2, e = [], [], 0; j = 0
        for i in range(n_sent + len(extra)):
            if e < len(extra) and i == pos[e]: S2.append(extra[e][0]); T2.append(world.state(tuple(extra[e][1]))); e += 1
            else: S2.append(sents[j]); T2.append(targets[j]); j += 1
        sents, targets = S2, T2
    N = len(sents); cats, ids, mask = m.encode(sents)
    tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
    params = [m.theta] + ([m.logits] if m.logits.requires_grad else [])
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for i in range(0, N, batch):
        sl = slice(i, i + batch); S = m.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
        cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
        opt.zero_grad(); (-cos.mean()).backward(); opt.step()
        with torch.no_grad(): m.theta[0] = 0

def t4_cell(split, preds):
    its = [(it, p) for it, p in zip(split["axes"]["new_form_unseen_trans"], preds["new_form_unseen_trans"]) if it["template"] == 4]
    return {"n": len(its), "slot": float(np.mean([p["frame"][r] == it["frame"][r] for it, p in its for r in ("A", "B", "C")])),
            "exact": float(np.mean([all(p["frame"][r] == it["frame"][r] for r in ("A", "B", "C")) for it, p in its])),
            "per_role": {r: float(np.mean([p["frame"][r] == it["frame"][r] for it, p in its])) for r in ("A", "B", "C")}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--ks", default="50,500")
    ap.add_argument("--check", action="store_true", help="k = 0 on the first seed: must reproduce the stored row EQUAL"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    row_name = "1a_fresh" if a.seed_start >= 5 else "1a"; label = "CLAIM SET (fresh splits)" if a.seed_start >= 5 else "DESIGN SET (inspected)"
    ks = ([0] if a.check else []) + [int(x) for x in a.ks.split(",")]
    print(f"reader_growth (R49): main repository {mainrepo.check_pinned()}; row {row_name}; seeds {a.seed_start}-{a.seed_start + a.seeds - 1} ({label}); k {ks}; full retraining, 0 hyperparameter choices")
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        m0, bw, split, preds0, how, chk = get_reader(row_name, seed, a.device); world = bw["world"]
        before = report.evaluate(f"before ({row_name})", seed, preds0, split, "growthbefore"); cell0 = t4_cell(split, preds0); del m0
        g = json.load(open(os.path.join(mainrepo.ROOT, "data", f"growth_seed{seed}.json")))["template4"]
        print(f"  seed {seed}: before = {how} ({chk}); template-4 cell before {cell0}", flush=True)
        for k in (ks if seed == a.seed_start else [x for x in ks if x > 0]):
            m = v8_gates.V8(bw["w"], split["D"], np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(bw["F"])
            if a.device == "cuda": torch.cuda.synchronize()
            t0 = time.time(); train_plus(m, world, 30000, [(it["tokens"], it["ev"]) for it in g[:k]], seed, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], depth_train=bwm.DEPTH_TRAIN)
            if a.device == "cuda": torch.cuda.synchronize()
            tt = time.time() - t0; preds = {ax: decode.decode_frozen(m, world, split, ax) for ax in report.AXES + ["calibration"]}
            after = report.evaluate(f"reader + {k} template-4 sentences", seed, preds, split, "growthafter"); cell = t4_cell(split, preds)
            drop = {ax: before["axes"][ax]["mean_role"] - after["axes"][ax]["mean_role"] for ax in report.AXES if before["axes"][ax]}
            res = {"seed": seed, "row": row_name, "set": label, "k": k, "train_s": tt, "examples": k, "hyperparameter_choices": 0, "t4_before": cell0, "t4_after": cell,
                   "before": {ax: before["axes"][ax]["mean_role"] for ax in drop}, "after": {ax: after["axes"][ax]["mean_role"] for ax in drop}, "drop": drop,
                   "exact_depth3_before": before["axes"]["depth3_unseen"]["full"], "exact_depth3_after": after["axes"]["depth3_unseen"]["full"], "gates_open": sorted(c for c, v in m.gates().items() if v > 0.5)}
            if k == 0: print(f"  seed {seed} k 0 (check): per-role rule vs the saved reader: {'EQUAL' if max(abs(x) for x in drop.values()) < 1e-9 else 'largest difference ' + format(max(abs(x) for x in drop.values()), '.4f')}", flush=True)
            print(f"  seed {seed} k {k}: trained in {tt:.0f}s (alone); template-4 cell slot {cell0['slot']:.3f} -> {cell['slot']:.3f}, exact {cell0['exact']:.3f} -> {cell['exact']:.3f}; worst drop on a report axis {max(drop.values()):+.3f} ({max(drop, key=drop.get)}); gates open {res['gates_open']}", flush=True)
            json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_growth_reader_{row_name}_k{k}_seed{seed}.json"), "w")); del m; torch.cuda.empty_cache() if a.device == "cuda" else None
    metrics.checklist(a.seeds, "[x] k examples frozen before any model (data/growth_seed<s>.json) [x] full retraining, the same seed [x] wall-clock alone [x] " + label)
