"""confirmatory.py (responses R30 item 8, declared in RESULTS 'Declared NOW, before any result of the equal-size block is read'): the three
pre-specified comparisons of the reader against an MRC model, on the five shared seeds 0-4 (frozen splits; saved models, no training):
  (i)   six-question exact match at depth 3 with the depth NOT given (reader: depth from its level-opener count; MRC: all six questions,
        NULL = not given);
  (ii)  confident-error rate at matched coverage on depth 3 (depth not given): the reader's coverage under its threshold set on the
        calibration rows by the written rule (eval/metrics.set_tau); the MRC threshold set on the SAME calibration rows to give that
        coverage (the (1 - coverage) quantile of its calibration margins); confident error = a wrong answered decision, over all decisions;
  (iii) out-of-world AUROC: in-world report rows (every report axis pooled) against the out-of-world test sources (a) random and
        (b) reference POOLED (source (c) nonce reported apart: the reader rejects it by construction); the reader's residual (and, beside
        it, its smallest role margin), the MRC encoder's Mahalanobis distance (one Gaussian, mean and covariance fitted on the
        calibration rows' features = the final-layer [CLS] vector averaged over the six question passes; covariance shrunk by 1e-3 x
        identity) and, beside it, MSP (1 - the smallest over the six questions of the maximum softmax probability over candidates + NULL).
Test: difference reader - model per seed; p-value from the two-level bootstrap (1,000 resamples: seeds with replacement, then items with
replacement within each seed, the same indices for both models), two-sided; Holm over the three comparisons of the pair at 0.05; beside
it mean ± std, the 4-df paired t, seeds ahead and a sign test (reported, not relied on). Also prints R30 item 3's withdrawal condition
(an encoder FROM SCRATCH with inner-frame exact match >= 0.99 at depth 3, depth not given). Secondary (R48 item 5): AURC of the
risk-coverage curve on depth 3 (depth not given; margin as confidence) for both models; AUROC of the reader's explained residual (R42)
and of the MRC NULL gap (minus the smallest best-candidate-minus-NULL score over the six questions). Writes logs/results_confirmatory_<mrc tag>.json.
Run: ./run.sh eval/confirmatory.py [--seed_start 0|5] --mrc <tag>[,<tag>...] (tags as logs/models/<tag>_seed<s>/, e.g. mrc_google_bert_uncased_L-2_H-128_A-2_zs)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from scipy.stats import rankdata, ttest_rel, binomtest
from transformers import AutoTokenizer
from common import basic_world as bwm, decode, report
from eval import metrics
from eval.reader_models import get_reader
import models.bert_mrc as bm

SEEDS = range(5); B = 1000; OOW_POOL = ("random", "reference")

def auroc(inw, out):
    """P(score_out > score_in) with ties at 1/2 (Mann-Whitney)"""
    x = np.concatenate([inw, out]); r = rankdata(x); n1, n2 = len(inw), len(out)
    return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))

def aurc(correct, conf):
    """area under the risk-coverage curve: decisions sorted by confidence (descending), risk = error rate among the answered, averaged
    over every coverage level 1/n .. 1 (lower is better)"""
    o = np.argsort(-np.asarray(conf), kind="stable"); err = 1.0 - np.asarray(correct, float)[o]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))

def est(items): return [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]

# ---------- reader ----------
def reader_seed(seed, device, row="1a"):
    m, bw, split, _, how, chk = get_reader(row, seed, device); world = bw["world"]
    def run(items, axis="x"):
        e = est(items); p = decode.decode_frozen(m, world, {"axes": {axis: e}, "candidates": split["candidates"]}, axis); return e, p
    cal_e, cal_p = run(split["axes"]["calibration"])
    tau = metrics.set_tau([(None, p["frame"][r] == it["frame"][r], p["margins"][r]) for it, p in zip(cal_e, cal_p) for r in bwm.roles_at(it["depth"])])
    g3 = split["axes"]["depth3_unseen"]; e3, p3 = run(g3, "depth3_unseen")
    em = np.array([all(p["frame"].get(r) == it["frame"][r] for r in bwm.ROLES) for p, it in zip(p3, g3)], float)
    inner = float(np.mean([all(p["frame"].get(r) == it["frame"][r] for r in ("A", "B", "C")) for p, it in zip(p3, g3)]))
    dec = [[(p["frame"].get(r) == it["frame"][r], p["margins"].get(r, -np.inf)) for r in bwm.ROLES] for p, it in zip(p3, g3)]
    cov = float(np.mean([mg >= tau for d in dec for _, mg in d]))
    cw = np.array([np.mean([(not c) and mg >= tau for c, mg in d]) for d in dec])
    au = aurc([c for d in dec for c, _ in d], [mg for d in dec for _, mg in d])
    sc_in, mg_in, ex_in, ex_out = [], [], [], []
    for ax in report.AXES:
        if not split["axes"][ax]: continue
        e, p = run(split["axes"][ax], ax); sc_in += decode.residuals(m, world, e, p); mg_in += [min(pp["margins"].values()) for pp in p]; ex_in += decode.residuals_explained(m, world, e, p)
    oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"))); sc_out, mg_out = [], []
    for src in OOW_POOL:
        e, p = run(oow["test"][src]); sc_out += decode.residuals(m, world, e, p); mg_out += [min(pp["margins"].values()) for pp in p]; ex_out += decode.residuals_explained(m, world, e, p)
    print(f"  reader seed {seed}: {how}; {chk}; tau {tau:.4f}, coverage depth 3 (not given) {cov:.3f}", flush=True)
    return {"em": em, "inner": inner, "cov": cov, "tau": tau, "cw": cw, "in": np.array(sc_in), "out": np.array(sc_out),
            "in_margin": -np.array(mg_in), "out_margin": -np.array(mg_out), "in_expl": np.array(ex_in), "out_expl": np.array(ex_out), "aurc": au}

# ---------- MRC ----------
def mrc_load(tag, seed, device):
    md = os.path.join(mainrepo.ROOT, "logs", "models", f"{tag}_seed{seed}"); cfg = json.load(open(os.path.join(md, "config.json")))
    kinds = cfg["kinds"]; model = bm.MRC(cfg["model"], kinds, scratch=True)          # architecture only; the weights are loaded below
    model.load_state_dict(torch.load(os.path.join(md, "mrc.pt"), map_location="cpu")); model.to(device).eval()
    if kinds: bm.KIND_OF = {w_: kinds.index(c) + 1 for w_, c in bwm.load(seed)["cat"].items()}
    return model, AutoTokenizer.from_pretrained(md), cfg

@torch.no_grad()
def mrc_pass(model, tok, split, axis, items, device, batch=256):
    """all six questions per item -> per item: frame, margins, min max-prob, mean [CLS] feature"""
    meta = [(bm.QUESTIONS[r], it["tokens"], set(bwm.candidates_for(split, axis, r)), n, r) for n, it in enumerate(items) for r in bwm.ROLES]
    cap = {}; h = model.enc.register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o.last_hidden_state))
    out = [{"frame": {}, "margins": {}, "maxp": [], "cls": [], "gap": []} for _ in items]
    for i in range(0, len(meta), batch):
        chunk = meta[i:i + batch]; pairs = [(q, s) for q, s, _, _, _ in chunk]; ids, att, tt, spans = bm.encode(tok, pairs, device)
        s, e = bm.fwd(model, ids, att, tt, spans, pairs); s, e = s.float().cpu().numpy(), e.float().cpu().numpy(); cls = cap["h"][:, 0].float().cpu().numpy()
        for b, (_, toks, cands, n, r) in enumerate(chunk):
            null = s[b, 0] + e[b, 0]; sc = [(s[b, spans[b][k][0]] + e[b, spans[b][k][1]], k) for k, t in enumerate(toks) if t in cands and spans[b][k] is not None]
            allv = np.array(sorted([v for v, _ in sc] + [null], reverse=True)); best = max(sc) if sc else (-np.inf, None)
            out[n]["frame"][r] = toks[best[1]] if best[0] > null else None; out[n]["margins"][r] = float(allv[0] - allv[1]) if len(allv) > 1 else float(allv[0])
            pz = np.exp(allv - allv[0]); out[n]["maxp"].append(float(1 / pz.sum())); out[n]["cls"].append(cls[b]); out[n]["gap"].append(float(best[0] - null))
    h.remove()
    for o in out: o["cls"] = np.mean(o["cls"], 0); o["minmaxp"] = min(o["maxp"]); o["mingap"] = min(o["gap"])
    return out

def mrc_seed(tag, seed, cov_target, device):
    model, tok, cfg = mrc_load(tag, seed, device); split = bwm.load(seed)
    cal = mrc_pass(model, tok, split, "calibration", split["axes"]["calibration"], device)
    cal_m = np.array([mg for o in cal for mg in o["margins"].values()])
    thr = -np.inf if cov_target >= 1.0 else float(np.quantile(cal_m, 1 - cov_target))
    g3 = split["axes"]["depth3_unseen"]; o3 = mrc_pass(model, tok, split, "depth3_unseen", g3, device)
    em = np.array([all(o["frame"][r] == it["frame"][r] for r in bwm.ROLES) for o, it in zip(o3, g3)], float)
    inner = float(np.mean([all(o["frame"][r] == it["frame"][r] for r in ("A", "B", "C")) for o, it in zip(o3, g3)]))
    cw = np.array([np.mean([(o["frame"][r] != it["frame"][r]) and o["margins"][r] >= thr for r in bwm.ROLES]) for o, it in zip(o3, g3)])
    cov = float(np.mean([o["margins"][r] >= thr for o in o3 for r in bwm.ROLES]))
    au = aurc([o["frame"][r] == it["frame"][r] for o, it in zip(o3, g3) for r in bwm.ROLES], [o["margins"][r] for o in o3 for r in bwm.ROLES])
    X = np.stack([o["cls"] for o in cal]); mu = X.mean(0); C = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1]); P = np.linalg.inv(C)
    maha = lambda os_: np.array([float(np.sqrt(max(0.0, (o["cls"] - mu) @ P @ (o["cls"] - mu)))) for o in os_])
    ins = [o for ax in report.AXES if split["axes"][ax] for o in mrc_pass(model, tok, split, ax, split["axes"][ax], device)]
    oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")))
    outs = [o for src in OOW_POOL for o in mrc_pass(model, tok, split, "x", oow["test"][src], device)]
    print(f"  {tag} seed {seed}: threshold {thr:.4f} for coverage {cov_target:.3f} (achieved {cov:.3f}); exact match six {em.mean():.3f}, inner {inner:.3f}", flush=True)
    return {"em": em, "inner": inner, "cov": cov, "thr": thr, "cw": cw, "in": maha(ins), "out": maha(outs),
            "in_msp": 1 - np.array([o["minmaxp"] for o in ins]), "out_msp": 1 - np.array([o["minmaxp"] for o in outs]), "scratch": cfg.get("scratch", False),
            "in_gap": -np.array([o["mingap"] for o in ins]), "out_gap": -np.array([o["mingap"] for o in outs]), "aurc": au}

# ---------- statistics ----------
def boot(R, M, stat, rng):
    """two-level bootstrap of mean over seeds of stat(R_s) - stat(M_s); stat takes (dict, item index arrays)"""
    S = len(R); obs = np.mean([stat(R[s], None) - stat(M[s], None) for s in range(S)]); d = []
    for _ in range(B):
        ss = rng.integers(0, S, S); v = []
        for s in ss: v.append(stat(R[s], rng) - stat(M[s], rng, R[s].get("_last")))
        d.append(np.mean(v))
    d = np.array(d); p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))); return float(obs), p, (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))

def paired_item_stat(key):
    def f(x, rng, idx=None):
        a = x[key]
        if rng is None: return float(a.mean())
        if idx is None: idx = rng.integers(0, len(a), len(a)); x["_last"] = idx
        return float(a[idx].mean())
    return f

def auroc_stat(kin, kout):
    def f(x, rng, idx=None):
        if rng is None: return auroc(x[kin], x[kout])
        if idx is None: idx = (rng.integers(0, len(x[kin]), len(x[kin])), rng.integers(0, len(x[kout]), len(x[kout]))); x["_last"] = idx
        return auroc(x[kin][idx[0]], x[kout][idx[1]])
    return f

def holm(ps, alpha=0.05):
    order = np.argsort(ps); rej = [False] * len(ps); k = len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (k - j): rej[i] = True
        else: break
    return rej

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mrc", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed_start", type=int, default=0); a = ap.parse_args()
    SEEDS = range(a.seed_start, a.seed_start + 5); row = "1a_fresh" if a.seed_start >= 5 else "1a"; label = "CLAIM SET (fresh splits, R39 / R51)" if a.seed_start >= 5 else "seeds 0-4 (the pre-specified seeds; design set since R39)"
    print(f"confirmatory (R30 item 8): main repository {mainrepo.check_pinned()}; reader {row} against {a.mrc}; seeds {list(SEEDS)}: {label}; two-level bootstrap {B} resamples; Holm at 0.05 over the three comparisons of each pair")
    R = [reader_seed(s, a.device, row) for s in SEEDS]; ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"  reader: exact match six (depth 3, not given) {ms([r['em'].mean() for r in R])}; coverage {ms([r['cov'] for r in R])}; confident error {ms([r['cw'].mean() for r in R])}; AUROC residual {ms([auroc(r['in'], r['out']) for r in R])}; AUROC margin {ms([auroc(r['in_margin'], r['out_margin']) for r in R])}")
    for tag in a.mrc.split(","):
        M = [mrc_seed(tag, s, R[i]["cov"], a.device) for i, s in enumerate(SEEDS)]; res = {"mrc": tag, "comparisons": {}}
        print(f"\n== reader 1a vs {tag} ==")
        print(f"  {tag}: exact match six {ms([m['em'].mean() for m in M])}; inner {ms([m['inner'] for m in M])}; confident error at matched coverage {ms([m['cw'].mean() for m in M])}; AUROC Mahalanobis {ms([auroc(m['in'], m['out']) for m in M])}; AUROC MSP {ms([auroc(m['in_msp'], m['out_msp']) for m in M])}")
        comps = [("i_exact_match_six_depth3_not_given", paired_item_stat("em"), lambda x: x["em"].mean()),
                 ("ii_confident_error_matched_coverage", paired_item_stat("cw"), lambda x: x["cw"].mean()),
                 ("iii_oow_auroc_residual_vs_mahalanobis", auroc_stat("in", "out"), lambda x: auroc(x["in"], x["out"]))]
        ps = []
        for name, stat, per in comps:
            rng = np.random.default_rng(0); obs, p, ci = boot(R, M, stat, rng); dr = [per(r) - per(m) for r, m in zip(R, M)]
            t = ttest_rel([per(r) for r in R], [per(m) for m in M]); ahead = sum(x > 0 for x in dr); behind = sum(x < 0 for x in dr)
            sign = binomtest(ahead, ahead + behind).pvalue if ahead + behind else 1.0
            res["comparisons"][name] = {"reader": [per(r) for r in R], "model": [per(m) for m in M], "diff_mean": obs, "boot_p": p, "boot_ci95": ci,
                                        "t_p": float(t.pvalue) if np.isfinite(t.pvalue) else None, "reader_ahead": ahead, "model_ahead": behind, "sign_p": float(sign)}; ps.append(p)
            print(f"  {name}: reader - model {obs:+.4f} (95% {ci[0]:+.4f} .. {ci[1]:+.4f}), bootstrap p {p:.4f}; per seed diff {' '.join(f'{x:+.3f}' for x in dr)}; 4-df t p {t.pvalue:.3g}; seeds ahead reader {ahead} / model {behind}, sign p {sign:.3g}")
        for (name, _, _), rj in zip(comps, holm(ps)): res["comparisons"][name]["holm_reject"] = bool(rj)
        print(f"  Holm (alpha 0.05, three comparisons): " + "; ".join(f"{n} {'REJECT H0' if res['comparisons'][n]['holm_reject'] else 'not rejected'}" for n, _, _ in comps))
        inner = np.mean([m["inner"] for m in M]); res["inner_mean"] = float(inner)
        if any(m["scratch"] for m in M): print(f"  withdrawal condition (R30 item 3): from-scratch inner-frame exact match at depth 3, depth not given {inner:.3f} {'>= 0.99: the claim is WITHDRAWN' if inner >= 0.99 else '< 0.99: the claim stands'}")
        res["secondary"] = {"auroc_reader_residual": [auroc(r["in"], r["out"]) for r in R], "auroc_reader_margin": [auroc(r["in_margin"], r["out_margin"]) for r in R],
                            "auroc_reader_explained_residual": [auroc(r["in_expl"], r["out_expl"]) for r in R], "auroc_mrc_mahalanobis": [auroc(m["in"], m["out"]) for m in M],
                            "auroc_mrc_msp": [auroc(m["in_msp"], m["out_msp"]) for m in M], "auroc_mrc_null_gap": [auroc(m["in_gap"], m["out_gap"]) for m in M],
                            "aurc_reader": [r["aurc"] for r in R], "aurc_mrc": [m["aurc"] for m in M], "coverage_reader": [r["cov"] for r in R], "coverage_mrc": [m["cov"] for m in M]}
        for k, v in res["secondary"].items(): print(f"  {k}: {ms(v)}")
        res["seeds"] = list(SEEDS); res["reader_row"] = row
        json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_confirmatory_{tag}_s{a.seed_start}.json"), "w"), indent=1)
    metrics.checklist(5, "[x] comparisons pre-specified (R30 item 8) [x] thresholds on calibration rows only [x] two-level bootstrap [x] Holm")
