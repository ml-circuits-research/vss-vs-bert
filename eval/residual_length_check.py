"""residual_length_check.py (responses R41, 2026-10-25): a check on CALIBRATION rows only (training distribution; no report row is
read). R41's diagnosis: the residual (common/decode.residuals, the sop tool's) compares the reader's state S with the compiler's state of
the decoded frame, while S also carries every token's route deposits, so the residual grows with sentence length. Per seed of the reader
1a (loaded or retrained exactly as 1a, EQUAL-checked; eval/reader_models.py): Spearman correlation between the residual and the token
count on the 2,000 calibration sentences, and the residual's mean by depth 0 / 1 / 2 within them. R41's criterion, written there: the
diagnosis stands if the correlation is > 0.5 and the means rise with depth. Writes logs/results_r41check_seed<s>.json.
Run: ./run.sh eval/residual_length_check.py [--seeds 5] [--row 1a]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np
from scipy.stats import spearmanr
from common import decode
from eval import metrics
from eval.reader_models import get_reader

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--row", default="1a"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    print(f"residual_length_check (R41): main repository {mainrepo.check_pinned()}; row {a.row}; seeds {a.seeds}; CALIBRATION rows only (rule 25), no report row read")
    rows = []
    for seed in range(a.seeds):
        m, bw, split, preds, how, chk = get_reader(a.row, seed, a.device)
        cal = split["axes"]["calibration"]; est = [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in cal]
        pc = decode.decode_frozen(m, bw["world"], {"axes": {"x": est}, "candidates": split["candidates"]}, "x"); res = np.array(decode.residuals(m, bw["world"], est, pc))
        n = np.array([len(it["tokens"]) for it in cal]); dep = np.array([it["depth"] for it in cal])
        rho, p = spearmanr(res, n); by_d = {int(d): float(res[dep == d].mean()) for d in sorted(set(dep))}; cnt = {int(d): int((dep == d).sum()) for d in sorted(set(dep))}
        # within-depth correlation as well (separates length from depth)
        within = {int(d): float(spearmanr(res[dep == d], n[dep == d])[0]) for d in sorted(set(dep)) if len(set(n[dep == d])) > 1}
        by_len = {int(k): float(res[n == k].mean()) for k in sorted(set(n))}
        r = {"seed": seed, "row": a.row, "model": how, "check": chk, "spearman_residual_tokens": float(rho), "p": float(p), "mean_by_depth": by_d, "n_by_depth": cnt,
             "spearman_within_depth": within, "mean_by_token_count": by_len, "mean": float(res.mean())}
        print(f"  seed {seed}: {how}; per-role rule vs stored row: {chk}; Spearman(residual, tokens) {rho:.3f} (p {p:.1e}); mean by depth {by_d} (n {cnt}); within-depth Spearman {within}")
        print(f"  seed {seed}: mean residual by token count {{{', '.join(f'{k}: {v:.3f}' for k, v in by_len.items())}}}")
        json.dump(r, open(os.path.join(mainrepo.ROOT, "logs", f"results_r41check_seed{seed}.json"), "w")); rows.append(r)
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== R41 check, row {a.row}, calibration rows: MEANS over {len(rows)} seeds (per-seed list) ==")
    print(f"  Spearman(residual, token count) {ms([r['spearman_residual_tokens'] for r in rows])}")
    for d in sorted(rows[0]["mean_by_depth"]): print(f"  mean residual at depth {d} {ms([r['mean_by_depth'][d] for r in rows])}")
    rise = all(all(r["mean_by_depth"][d] < r["mean_by_depth"][d + 1] for d in sorted(r["mean_by_depth"])[:-1]) for r in rows)
    strong = all(r["spearman_residual_tokens"] > 0.5 for r in rows)
    print(f"  R41 criterion: correlation > 0.5 on every seed: {strong}; means rise with depth on every seed: {rise} => diagnosis {'STANDS' if strong and rise else 'NOT supported as stated'}")
    metrics.checklist(len(rows), "[x] calibration rows only (25) [x] the reported model (EQUAL check)")
