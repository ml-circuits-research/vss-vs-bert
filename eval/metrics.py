"""metrics.py: ONE scorer for every model (INSTRUCTIONS section 3 and 6). A model's output on an axis is a list of predictions, one per
sentence: {"frame": {role: filler or None}, "margins": {role: float}}; the gold is the split's items. The scorer computes:
  - per-role accuracy and FULL frame (every role of the depth exact, absent roles must be absent),
  - mean margin per role (the margin is the model's own: top1 - top2 cosine for the reader and the 7b transformer; top1 - top2 probability
    gap for a tagger; always 'larger = surer'),
  - calibration (rule 18 / 25): a threshold tau is set on CALIBRATION rows (never on reported rows) by a stated rule and then, on the
    reported rows, coverage (share of role decisions with margin >= tau), confident-wrong (wrong AND margin >= tau, over all decisions)
    and accuracy among the answered; the full accuracy-versus-coverage curve is returned for the figures,
  - the rule of three for a zero count: a bound on the true error rate below 3/n with 95% confidence.
Means over seeds with the per-seed list are the job of eval/compare.py; this module scores one seed."""
import numpy as np

def role_decisions(items, preds, roles):
    """flat list of (role, correct, margin, gold_present) over sentences and roles"""
    out = []
    for it, p in zip(items, preds):
        gold = it["frame"]
        for r in roles:
            g = gold.get(r); f = p["frame"].get(r); m = float(p["margins"].get(r, 0.0))
            out.append((r, (f == g), m, g is not None))
    return out

def score_axis(items, preds, roles):
    dec = role_decisions(items, preds, roles)
    per_role = {r: float(np.mean([c for rr, c, _, _ in dec if rr == r])) for r in roles}
    per_margin = {r: float(np.mean([m for rr, _, m, _ in dec if rr == r])) for r in roles}
    full = float(np.mean([all(p["frame"].get(r) == it["frame"].get(r) for r in roles) for it, p in zip(items, preds)]))
    mean_role = float(np.mean([c for _, c, _, _ in dec]))
    return {"n": len(items), "mean_role": mean_role, "full": full, "per_role": per_role, "margin": per_margin}

def set_tau(cal_dec, max_confident_wrong=0.02, grid=None):
    """the threshold rule (written before the runs): the SMALLEST tau at which the confident-wrong rate on the calibration rows is below
    max_confident_wrong (as the paper's verifier threshold, RESULTS 11-octies); tau = +inf when no threshold reaches it"""
    m = np.array([x[2] for x in cal_dec]); c = np.array([x[1] for x in cal_dec])
    grid = np.unique(np.concatenate([[0.0], m])) if grid is None else grid
    for tau in grid:
        ans = m >= tau
        if (ans & ~c).mean() <= max_confident_wrong: return float(tau)
    return float("inf")

def calibration(dec, tau):
    m = np.array([x[2] for x in dec]); c = np.array([x[1] for x in dec]); ans = m >= tau
    cov = float(ans.mean()); cw = float((ans & ~c).mean()); acc_ans = float(c[ans].mean()) if ans.any() else float("nan")
    n_wrong = int((ans & ~c).sum()); n_ans = int(ans.sum())
    return {"tau": tau, "coverage": cov, "confident_wrong": cw, "accuracy_answered": acc_ans, "n_answered": n_ans, "n_wrong_answered": n_wrong,
            "rule_of_three_bound": (3.0 / n_ans if n_wrong == 0 and n_ans > 0 else None)}

def coverage_curve(dec, k=20):
    """accuracy among answered vs coverage, at k thresholds between the min and max margin"""
    m = np.array([x[2] for x in dec]); c = np.array([x[1] for x in dec]); out = []
    for tau in np.linspace(m.min(), m.max(), k):
        ans = m >= tau
        if ans.any(): out.append({"tau": float(tau), "coverage": float(ans.mean()), "accuracy_answered": float(c[ans].mean()), "confident_wrong": float((ans & ~c).mean())})
    return out

def expected_calibration_error(dec, bins=10):
    """ECE over the role decisions, with the margin mapped to [0, 1] by rank (margins are not probabilities for every model, so the ECE
    is reported for the models whose confidence IS a probability gap and marked 'rank-mapped' for the others)"""
    m = np.array([x[2] for x in dec]); c = np.array([x[1] for x in dec]).astype(float)
    r = (np.argsort(np.argsort(m)) + 0.5) / len(m); e = 0.0
    for b in range(bins):
        sel = (r >= b / bins) & (r < (b + 1) / bins)
        if sel.any(): e += sel.mean() * abs(c[sel].mean() - r[sel].mean())
    return float(e)

def checklist(a_seeds, extra=""):
    """rule 3 of the main repository: the evaluation checklist printed at the end of every script"""
    print(f"\nchecklist: [x] world addresses or the same frozen gold [x] same frozen splits for every model [x] chance printed [x] separate axes "
          f"[x] margin [{'x' if a_seeds >= 3 else ' '}] >=3 seeds [x] thresholds on calibration rows only (25) [x] regime named (24) "
          f"[x] mean over seeds with the per-seed list (27) [x] memory curve in the log (21) {extra}")
