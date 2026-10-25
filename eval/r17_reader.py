"""r17_reader.py (deviation R17, settled by R18; RESULTS): the reader's rows of R17 on the basic world, 5 seeds.
  item 1: out-of-world rejection on the frozen sets data/oow_seed<s>.json: rejected = residual above a threshold (the reader has no status
          on this world); threshold (i) the 2% false-rejection point of the in-world calibration rows, (ii) the threshold that rejects
          the most 'neg' negatives under <= 2% false rejection on the calibration rows (for one scalar score it can only coincide with or
          lie above (i): stated, not hidden). Sources reported apart: random, reference, nonce by construction (a word outside the legend
          cannot be encoded: rejected, NOT a learned rejection), nonce as placeholder (one fixed 'O' vector per seed, R18 Q14).
          False rejections on the in-world report axes. The depth of every sentence is the level-opener count (no depth is given).
  item 2: the residual itself, per set.
  item 3: depth not given: the depth accuracy of the opener count and FULL at depth 3 with it, beside the given-depth FULL.
The reader is retrained per seed exactly as run 1a (vss/run_reader_basic.py, same generators); the per-role rule is compared with the
stored 1a results (EQUAL / the largest difference printed). Writes logs/results_r17_reader_seed<s>.json.
Run: ./run.sh eval/r17_reader.py [--seeds 5] [--gates_given none|cues]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode, report
from eval import metrics
from vss.run_reader_basic import GIVEN_OPEN

FR_MAX = 0.02

def oow_path(seed): return os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")

def with_depth(items):
    return [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]

def read(m, world, split, items, axis="x"):
    """decode and residual of items at their (estimated) depth; axis selects the candidate lists (new_ids: held-out identities)"""
    if not items: return [], []
    sp = {"axes": {axis: items}, "candidates": split["candidates"]}; p = decode.decode_frozen(m, world, sp, axis)
    return p, decode.residuals(m, world, items, p)

def thresholds(cal, neg):
    cal, neg = np.asarray(cal), np.asarray(neg)
    t1 = float(np.quantile(cal, 1 - FR_MAX))                     # (i): reject res > t1; at most 2% of calibration rows rejected
    cands = np.unique(np.concatenate([cal, neg])); ok = [t for t in cands if (cal > t).mean() <= FR_MAX]
    t2 = float(max(ok, key=lambda t: ((neg > t).mean(), -t))) if ok else t1   # (ii): most negatives rejected, then the lowest threshold
    return t1, t2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda")
    ap.add_argument("--gates_given", default="none", choices=["none", "cues"]); a = ap.parse_args()
    stored_tag = "reader_basic" if a.gates_given == "none" else "reader_basic_cues"; tag = "r17_" + stored_tag
    print(f"r17_reader: main repository {mainrepo.check_pinned()}; device {a.device}; seeds {a.seeds}; reader {stored_tag}; false-rejection cap {FR_MAX}")
    print("regime (rule 24): basic world; depth of every sentence estimated from the level openers (no depth given); thresholds on calibration rows only (rule 25); out-of-world sets frozen before this run")
    rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, a.D); w, world = bw["w"], bw["world"]; oow = json.load(open(oow_path(seed)))
        m = v8_gates.V8(w, a.D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(bw["F"])
        if a.gates_given == "cues": m.fix_gates(GIVEN_OPEN, True)
        t0 = time.time(); m.train(world, a.N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=a.batch, lr=a.lr, depth_train=bwm.DEPTH_TRAIN)
        preds = {ax: decode.decode_frozen(m, world, split, ax) for ax in report.AXES + ["calibration"]}
        st = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_{stored_tag}_seed{seed}.json")))
        row0 = report.evaluate("check", seed, preds, split, "none")
        diff = max(abs(row0["axes"][ax]["mean_role"] - st["axes"][ax]["mean_role"]) for ax in report.AXES if st["axes"][ax])
        print(f"  seed {seed}: retrained in {time.time() - t0:.0f}s; per-role rule vs stored {stored_tag} seed {seed}: {'EQUAL' if diff < 1e-9 else f'largest difference {diff:.4f}'}")
        res = {"seed": seed, "model": stored_tag, "check_max_diff": diff}
        # item 3: depth not given (opener count) on the in-world axes
        dacc, full_est = {}, None
        for ax in report.AXES:
            items = split["axes"][ax]
            if not items: continue
            est = with_depth(items); dacc[ax] = float(np.mean([e["depth"] == it["depth"] for e, it in zip(est, items)]))
            if ax == "depth3_unseen":
                p, _ = read(m, world, split, est, ax)
                full_est = float(np.mean([all(pp["frame"].get(r) == it["frame"][r] for r in bwm.ROLES) for pp, it in zip(p, items)]))
        res["depth_accuracy"] = dacc; res["full_depth3_estimated"] = full_est; res["full_depth3_given"] = row0["axes"]["depth3_unseen"]["full"]
        # items 1-2: residuals; in-world calibration and report rows, then the out-of-world sets
        _, rcal = read(m, world, split, with_depth(split["axes"]["calibration"]))
        rin = {ax: read(m, world, split, with_depth(split["axes"][ax]), ax)[1] for ax in report.AXES if split["axes"][ax]}
        rneg = []
        for src in ("random", "reference"): rneg += read(m, world, split, with_depth(oow["neg"][src]))[1]
        with decode.Placeholders(m, w, seed) as ph:
            negn = [dict(x, tokens=ph.map(x["tokens"])) for x in oow["neg"]["nonce"]]; rneg += read(m, world, split, with_depth(negn))[1]
            testn = [dict(x, tokens=ph.map(x["tokens"])) for x in oow["test"]["nonce"]]; r_nonce_ph = read(m, world, split, with_depth(testn))[1]
        rt = {src: read(m, world, split, with_depth(oow["test"][src]))[1] for src in ("random", "reference")}; rt["nonce_placeholder"] = r_nonce_ph
        t1, t2 = thresholds(rcal, rneg); res["thresholds"] = {"i": t1, "ii": t2}
        res["rejected"] = {v: {src: float(np.mean(np.array(r) > t)) for src, r in rt.items()} for v, t in (("i", t1), ("ii", t2))}
        for v in ("i", "ii"): res["rejected"][v]["nonce_by_construction"] = 1.0
        res["false_rejection"] = {v: {ax: float(np.mean(np.array(r) > t)) for ax, r in rin.items()} for v, t in (("i", t1), ("ii", t2))}
        res["false_rejection_pooled"] = {v: float(np.mean(np.concatenate([np.array(r) for r in rin.values()]) > t)) for v, t in (("i", t1), ("ii", t2))}
        res["residual_mean"] = {"calibration": float(np.mean(rcal)), **{ax: float(np.mean(r)) for ax, r in rin.items()}, **{f"oow_{s}": float(np.mean(r)) for s, r in rt.items()}, "neg": float(np.mean(rneg))}
        print(f"  seed {seed}: thresholds (i) {t1:.4f} (ii) {t2:.4f}; rejected (i) {res['rejected']['i']}; false rejection pooled (i) {res['false_rejection_pooled']['i']:.4f} (ii) {res['false_rejection_pooled']['ii']:.4f}")
        print(f"  seed {seed}: residual means {res['residual_mean']}")
        print(f"  seed {seed}: depth accuracy of the opener count {dacc}; FULL depth 3 estimated depth {full_est:.3f} / given {res['full_depth3_given']:.3f}")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(res, f)
        rows.append(res); del m; torch.cuda.empty_cache() if a.device == "cuda" else None
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== R17, {stored_tag}: MEANS over {len(rows)} seeds (per-seed list) ==")
    for v in ("i", "ii"):
        for src in ("random", "reference", "nonce_placeholder", "nonce_by_construction"): print(f"  rejected ({v}) {src:22s} {ms([r['rejected'][v][src] for r in rows])}")
        print(f"  false rejection ({v}) pooled in-world report axes {ms([r['false_rejection_pooled'][v] for r in rows])}")
    print(f"  criterion (demo specification): rejected >= 0.90 per source with false rejections <= {FR_MAX}")
    print(f"  FULL depth 3, depth estimated from openers {ms([r['full_depth3_estimated'] for r in rows])}; given {ms([r['full_depth3_given'] for r in rows])}")
    print(f"  depth accuracy of the opener count, depth3_unseen {ms([r['depth_accuracy']['depth3_unseen'] for r in rows])}")
    metrics.checklist(len(rows), "[x] out-of-world sets frozen before the run [x] thresholds on calibration rows (and the 'neg' part for ii) only [x] sources reported apart")
