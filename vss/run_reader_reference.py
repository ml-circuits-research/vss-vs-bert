"""run_reader_reference.py (run 4, reader): the VSS reader on the REFERENCE world as RESULTS 10b-quinquies: experiment/v16_reference.py's
world and training, UNMODIFIED code (v8_gates.V8 path + chainL + bigram, gates learned from closed with to / by / from GIVEN open
(`--rel_gates open`), trainer.train_loop, N 30000, recipient-less share 0.2, batch 8, lr 0.02, D 1024), decoded on the FROZEN splits of
data/reference_seed<s>.json with the world's addresses: A, B, X_k and the predicate by cleanup; the recipient's presence by the occupancy
marker (cosine after unbinding, as v16's readout), with the threshold set on the calibration rows (depth-0 rows with and without a
recipient, midpoint of the two means, as v16 sets it on its training rows; rule 25). Depth is given, as in v16's evaluation.
Exit check: within seed variance of RESULTS 10b-quinquies (depth 3 UNSEEN roles FULL 0.968 ± 0.020; seen and passives 1.000).
Run: python3 vss/run_reader_reference.py [--seeds 5] [--N 30000] [--D 1024] [--device cuda]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v8_gates as v8, trainer
from common import reference_world as rw, report_ref
from eval import metrics

def train_reader(bw, seed, a):
    """v16_reference.py's __main__ training block for one seed, verbatim in substance (same generators, same calls)"""
    v16 = rw.import_v16()[2]; w, world, F = bw["w"], bw["world"], bw["F"]
    m = v8.V8(w, a.D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
    print(f"  relation-marker gates GIVEN open (rule 20, not learned; E1b): {m.fix_gates(['to', 'by', 'from'], True)}")
    trng = np.random.default_rng(200 + seed); sents, targets = [], []
    for n in range(a.N):
        d = int(trng.integers(3)); ev = w.event(bw["trP"], bw["trO"], depth=d)
        if trng.random() < a.p_omit: toks = w.render_omit(ev, int(trng.integers(len(v16.OMIT_C)))); targets.append(world.state(ev, toks, has_C=False))
        else: toks = w.render(ev, int(trng.choice(v5.TRAIN_T))); targets.append(world.state(ev, toks))
        sents.append(toks)
    cats, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
    trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, tag=f" seed {seed}")
    return m

@torch.no_grad()
def states(m, items, batch=256):
    out = []
    for i in range(0, len(items), batch):
        c, i_, k = m.encode([it["tokens"] for it in items[i:i + batch]]); out.append(m.states(c, i_, k))
    return torch.cat(out) if out else None

@torch.no_grad()
def read(m, world, split, axis, thr):
    items = split["axes"][axis]
    if not items: return []
    w = world.w; R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; mk = torch.tensor(world.m, dtype=m.F.dtype, device=m.device)
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); S = states(m, items); Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
    def best(sn, key, cands):
        cid = torch.tensor([w.idx[x] for x in cands], device=m.device); sc = torch.real((sn * key) @ torch.conj(Fn[cid]).T); top = torch.topk(sc, 2)
        return cands[int(top.indices[0])], float(top.values[0] - top.values[1])
    preds = []
    for j, it in enumerate(items):
        d = it["depth"]; rep = torch.conj(R["rep"]) ** d; sn = Sn[j]; fr = {r: None for r in rw.ROLES}; mg = {}; pres = None
        for r in ("A", "B", "C"):
            key = torch.conj(R[r]) * rep; ident, margin = best(sn, key, rw.candidates_for(split, axis, r))
            if r == "C":
                pres = float(torch.real((sn * key * torch.conj(mk)).sum()) / mk.abs().pow(2).sum().sqrt())
                if pres >= thr: fr["C"] = ident; mg["C"] = min(margin, abs(pres - thr))
                else: mg["C"] = abs(pres - thr)
            else: fr[r] = ident; mg[r] = margin
        for k in range(d):
            ident, margin = best(sn, torch.conj(R["X"]) * torch.conj(R["rep"]) ** k, rw.candidates_for(split, axis, "X"))
            fr[f"X{k+1}"] = ident; mg[f"X{k+1}"] = margin
        fr["pred"], mg["pred"] = best(sn, torch.conj(R["pred"]) * rep, rw.candidates_for(split, axis, "pred"))
        preds.append({"frame": fr, "margins": mg, "presence_C": pres})
    return preds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--p_omit", type=float, default=0.2)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args()
    print(f"run_reader_reference: main repository {mainrepo.check_pinned()}; device {a.device}; D {a.D}; N {a.N}; batch {a.batch}; lr {a.lr}; p_omit {a.p_omit}; seeds {a.seeds}")
    print("regime (rule 24): v16 reference world (10b-quinquies: predicate role, occupancy marker, to/by/from gates given open), training depths 0-2 and templates TRAIN_T, depth 3 UNSEEN; frozen splits")
    rows = []
    for seed in range(a.seeds):
        split = rw.load(seed); assert split["D"] == a.D, f"splits were made with D={split['D']}; rerun data/make_reference_splits.py --D {a.D}"
        bw = rw.build(seed, a.D); t0 = time.time(); m = train_reader(bw, seed, a)
        if a.device == "cuda": torch.cuda.synchronize()
        print(f"  seed {seed}: trained in {time.time() - t0:.0f}s")
        # presence threshold on calibration rows only: depth-0 calibration rows with and without a recipient (midpoint of the means, v16's rule)
        cal0 = [it for it in split["axes"]["calibration"] if it["depth"] == 0]
        cp = read(m, bw["world"], {"axes": {"c": cal0}, "candidates": split["candidates"]}, "c", thr=-1e9)
        known = [p["presence_C"] for p, it in zip(cp, cal0) if it["has_C"]]; absent = [p["presence_C"] for p, it in zip(cp, cal0) if not it["has_C"]]
        thr = 0.5 * (np.mean(known) + np.mean(absent)); print(f"  seed {seed}: recipient presence threshold {thr:.3f} (calibration rows: known {np.mean(known):.3f}, absent {np.mean(absent):.3f})")
        preds = {axis: read(m, bw["world"], split, axis, thr) for axis in rw.AXES + ["calibration"]}
        row = report_ref.evaluate("VSS reader (reference, 10b-quinquies)", seed, preds, split, "reader"); row["presence_threshold"] = thr; rows.append(row)
    report_ref.summarize(rows, "VSS reader (reference, 10b-quinquies)")
    print("reference (main repository, RESULTS 10b-quinquies, fresh test draws): depth 3 UNSEEN roles FULL 0.968 ± 0.020; passives 1.000; T9 recipient 0.84; unseen transitions 0.627")
    metrics.checklist(a.seeds, "[x] predicate role present (23) [x] presence threshold on calibration rows (25)")
