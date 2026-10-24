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

# ------------------------------------------------------------------ real data (INSTRUCTIONS_REAL_2026-10-24.md): head-token frames
# A gold or predicted frame here is FLAT: {path: value} from common/real_data.flatten ("intent", "<slot>", "<slot>/intent", ...; a slot
# holding a nested intent has the value "@frame"). A prediction is {"frame": flat frame, "margins": {path: margin}}, where the margins
# also cover the roles the model read as ABSENT (its confidence that they are absent), so that a missed role is scored with the
# model's own margin. Decisions are made on the paths in gold OR prediction (a role absent in both is not a decision, but it is
# counted in the occupancy case 'absent read absent').

def real_decisions(golds, preds, kind="all"):
    """(path, correct, margin, gold_present) over items; kind: 'all' | 'intent' (paths ending in 'intent') | 'slot' (the others)"""
    out = []
    for g, p in zip(golds, preds):
        f, m = p["frame"], p["margins"]
        for k in sorted(set(g) | set(f)):
            isint = k == "intent" or k.endswith("/intent")
            if (kind == "intent" and not isint) or (kind == "slot" and isint): continue
            out.append((k, g.get(k) == f.get(k), float(m.get(k, 0.0)), k in g))
    return out

def frame_margin(p):
    """the frame's own confidence: the smallest margin over what it SAYS (the intent and the filled roles); the presence margins of the
    roles it leaves empty are not included (stated)"""
    return float(min([p["margins"].get(k, 0.0) for k in p["frame"]] or [0.0]))

def frame_decisions(golds, preds):
    return [("frame", g == p["frame"], frame_margin(p), True) for g, p in zip(golds, preds)]

def set_taus(cal_golds, cal_preds, max_cw=0.05):
    """the written threshold rule (set_tau) applied on the CALIBRATION rows, once per decision kind"""
    return {"intent": set_tau(real_decisions(cal_golds, cal_preds, "intent"), max_cw), "slot": set_tau(real_decisions(cal_golds, cal_preds, "slot"), max_cw),
            "frame": set_tau(frame_decisions(cal_golds, cal_preds), max_cw), "rule": f"smallest tau with confident-wrong <= {max_cw} on the calibration rows, per decision kind"}

def score_real(golds, preds, roles, taus, new_paths=None, depths=None):
    """golds: flat gold frames; preds: predictions; roles: the top-level slot types of the schema (for 'absent read absent');
    new_paths: per item the paths whose head token is new (the 'new slot values' axis) or None; depths: per item, for FULL by depth"""
    n = len(golds)
    if n == 0: return None
    full = [g == p["frame"] for g, p in zip(golds, preds)]
    intent = [g["intent"] == p["frame"].get("intent") for g, p in zip(golds, preds)]
    sp = [(g[k] == p["frame"].get(k)) for g, p in zip(golds, preds) for k in g if k != "intent"]
    pp = [(g.get(k) == p["frame"][k]) for g, p in zip(golds, preds) for k in p["frame"] if k != "intent"]
    lev = [(g[k] == p["frame"].get(k)) for g, p in zip(golds, preds) for k in g if k.endswith("/intent")]
    pres = [(k in p["frame"]) for g, p in zip(golds, preds) for k in g if k != "intent"]
    absn = [(r not in p["frame"]) for g, p in zip(golds, preds) for r in roles if r not in g]
    per_role = {}
    for r in roles:
        v = [g.get(r) == p["frame"].get(r) for g, p in zip(golds, preds) if r in g or r in p["frame"]]
        if v: per_role[r] = (float(np.mean(v)), len(v))
    out = {"n": n, "full": float(np.mean(full)), "intent": float(np.mean(intent)), "slot_heads_on_gold": float(np.mean(sp)) if sp else None,
           "slot_heads_precision": float(np.mean(pp)) if pp else None, "nested_intents": float(np.mean(lev)) if lev else None,
           "occupancy": {"present_read_present": float(np.mean(pres)) if pres else None, "absent_read_absent": float(np.mean(absn)) if absn else None,
                         "false_fills_per_item": float(sum(len([k for k in p["frame"] if k not in g]) for g, p in zip(golds, preds)) / n)},
           "per_role": per_role}
    if new_paths is not None:
        v = [(g.get(k) == p["frame"].get(k)) for g, p, ks in zip(golds, preds, new_paths) for k in (ks or [])]
        out["new_value_heads"] = float(np.mean(v)) if v else None; out["n_new_value_heads"] = len(v)
    if depths is not None:
        by = {}
        for d, f in zip(depths, full): by.setdefault(int(d), []).append(f)
        out["full_by_depth"] = {str(d): (float(np.mean(v)), len(v)) for d, v in sorted(by.items())}
    di, ds, dfm = real_decisions(golds, preds, "intent"), real_decisions(golds, preds, "slot"), frame_decisions(golds, preds)
    out["calibration"] = {"intent": calibration(di, taus["intent"]), "slot": calibration(ds, taus["slot"]) if ds else None, "frame": calibration(dfm, taus["frame"]),
                          "frame_curve": coverage_curve(dfm), "ece_rank_intent": expected_calibration_error(di)}
    out["abstain"] = {"intent_margin_below_tau": float(np.mean([p["margins"].get("intent", 0.0) < taus["intent"] for p in preds])),
                      "frame_margin_below_tau": float(np.mean([frame_margin(p) < taus["frame"] for p in preds]))}
    return out

def span_f1(gold_spans, pred_spans):
    """exact span micro precision / recall / F1 over (type, start, end) (the standard slot F1 of these benchmarks)"""
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold_spans, pred_spans)); ng = sum(len(set(g)) for g in gold_spans); npred = sum(len(set(p)) for p in pred_spans)
    P = tp / npred if npred else 0.0; R = tp / ng if ng else 0.0
    return {"precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0, "n_gold": ng, "n_pred": npred}
