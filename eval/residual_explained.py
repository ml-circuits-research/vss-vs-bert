"""residual_explained.py (responses R42 / R43, declared in RESULTS 'Deviation R42 / R43'): the reader's out-of-world rejection with BOTH
residuals, the compiled one (common/decode.residuals, the paper's) and the explained-tokens one (common/decode.residuals_explained), on
saved 1a readers (no retraining when the model is saved; eval/reader_models.py, EQUAL-checked). Per seed: the threshold of each residual
at the 2% false-rejection point of the in-world CALIBRATION rows (rule 25); rejected per out-of-world test source (random, reference,
nonce as placeholder; nonce by construction is 1.000 for the reader and not a score); false rejection per in-world report axis and
pooled; AUROC in-world report rows (pooled) vs out-of-world (random + reference pooled, the confirmatory pooling of R30 item 8 (iii)) and
per source. The depth of every sentence is its level-opener count (no depth given). Seeds 0-4 are the design set (inspected); the claim
is made on seeds 5-9 (R39; row '1a_fresh': 1a trained on the fresh splits). Writes logs/results_r42_<row>_seed<s>.json.
Run: ./run.sh eval/residual_explained.py --row 1a --seed_start 0 | --row 1a_fresh --seed_start 5  [--seeds 5]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np
from scipy.stats import rankdata
from common import decode, report
from eval import metrics
from eval.reader_models import get_reader

FR = 0.02
def auroc(inw, out):
    x = np.concatenate([inw, out]); r = rankdata(x); n1, n2 = len(inw), len(out); return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))
def est(items): return [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--row", default="1a", choices=["1a", "1a_fresh"]); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args(); seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    label = "CLAIM SET (fresh splits, R39)" if a.seed_start >= 5 else "DESIGN SET (inspected, R39)"
    print(f"residual_explained (R42 / R43): main repository {mainrepo.check_pinned()}; row {a.row}; seeds {seeds}: {label}; thresholds at {FR:.0%} false rejection on the calibration rows")
    print("expected failure mode, written before the run (R43): random in-vocabulary sentences that the decoder still fills into a frame leave few tokens unexplained and may pass the explained residual")
    rows = []
    for seed in seeds:
        m, bw, split, _, how, chk = get_reader(a.row, seed, a.device); world = bw["world"]
        def both(items, axis="x"):
            e = est(items); p = decode.decode_frozen(m, world, {"axes": {axis: e}, "candidates": split["candidates"]}, axis)
            return np.array(decode.residuals(m, world, e, p)), np.array(decode.residuals_explained(m, world, e, p))
        cal = both(split["axes"]["calibration"]); inw = {ax: both(split["axes"][ax], ax) for ax in report.AXES if split["axes"][ax]}
        oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"))); out = {src: both(oow["test"][src]) for src in ("random", "reference")}
        with decode.Placeholders(m, bw["w"], seed) as ph: out["nonce_placeholder"] = both([dict(x, tokens=ph.map(x["tokens"])) for x in oow["test"]["nonce"]])
        res = {"seed": seed, "row": a.row, "set": label, "model": how, "check": chk}
        for k, name in ((0, "compiled"), (1, "explained")):
            t = float(np.quantile(cal[k], 1 - FR)); pooled_in = np.concatenate([v[k] for v in inw.values()]); pooled_out = np.concatenate([out["random"][k], out["reference"][k]])
            res[name] = {"threshold": t, "rejected": {s: float(np.mean(v[k] > t)) for s, v in out.items()}, "false_rejection": {ax: float(np.mean(v[k] > t)) for ax, v in inw.items()},
                         "false_rejection_pooled": float(np.mean(pooled_in > t)), "auroc_pooled_random_reference": auroc(pooled_in, pooled_out),
                         "auroc": {s: auroc(pooled_in, v[k]) for s, v in out.items()}, "mean": {"calibration": float(cal[k].mean()), **{ax: float(v[k].mean()) for ax, v in inw.items()}, **{f"oow_{s}": float(v[k].mean()) for s, v in out.items()}},
                         "share_zero": {"calibration": float(np.mean(cal[k] < 1e-6)), **{f"oow_{s}": float(np.mean(v[k] < 1e-6)) for s, v in out.items()}}}
            r = res[name]
            print(f"  seed {seed} {name:9s}: {how}; {chk}; threshold {t:.4f}; AUROC pooled {r['auroc_pooled_random_reference']:.3f} (random {r['auroc']['random']:.3f}, reference {r['auroc']['reference']:.3f}, nonce ph. {r['auroc']['nonce_placeholder']:.3f}); rejected {r['rejected']}; false rejection pooled {r['false_rejection_pooled']:.3f} {r['false_rejection']}", flush=True)
            print(f"  seed {seed} {name:9s}: means {{{', '.join(f'{kk}: {vv:.3f}' for kk, vv in r['mean'].items())}}}; share exactly 0 {r['share_zero']}", flush=True)
        json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_r42_{a.row}_seed{seed}.json"), "w")); rows.append(res)
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== R42 / R43, row {a.row}, {label}: MEANS over {len(rows)} seeds (per-seed list) ==")
    for name in ("compiled", "explained"):
        print(f"  {name}: AUROC in-world vs out-of-world (random + reference) {ms([r[name]['auroc_pooled_random_reference'] for r in rows])}")
        for s in ("random", "reference", "nonce_placeholder"): print(f"  {name}: AUROC vs {s:18s} {ms([r[name]['auroc'][s] for r in rows])}; rejected at 2% calibration FR {ms([r[name]['rejected'][s] for r in rows])}")
        print(f"  {name}: false rejection pooled in-world report axes {ms([r[name]['false_rejection_pooled'] for r in rows])}")
        for ax in rows[0][name]["false_rejection"]: print(f"  {name}: false rejection {ax:24s} {ms([r[name]['false_rejection'][ax] for r in rows])}")
    metrics.checklist(len(rows), "[x] thresholds on calibration rows only (25) [x] the reported model (EQUAL check) [x] expected failure mode written before the run [x] sources reported apart")
