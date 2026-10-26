"""
activate.py: which rules and routes the reader of a world runs, decided from the NATURE of the world (its annotated examples), not by hand.

Two levels of activation (vision/ACTIVATION.md has the argument; this docstring has the contract):

  W. WORLDS, at inference. Every world's reader reads the sentence; the dispatcher keeps the reading whose residual is under that world's
     fit threshold with the largest margin, and refuses when no world fits (infer.formalize_text with several readers). Columns voting on an
     object, except that the vote is a measured quantity (the residual) and "no column fits" is itself a result. Nothing here changes it.

  R. RULES and ROUTES, at training: the grammar of ONE world. Three classes, treated differently.

     1. CONTRACT rules. Consequences of q-formalisation that hold in every world: the answer to a question is a word of the sentence
        (candidates = sentence); a word answers one question (one-to-one over roles and speakers); two candidates that the state cannot
        separate are declared undecided (undecided_adjacent); the residual counts the words no answer explains (residual = explained).
        They do not depend on the data, cost nothing to train, and are ON in every model written with decoder "contract" (the default).
        Decoder "paper" (catalog candidates, compiled residual) is kept so that the frozen rows of the comparison can be reproduced.

     2. STRUCTURAL routes. Hypotheses about the FORM of the language of this world; each adds addresses (parameters) and is trained.
        `path`, `chainL`, `bigram` are the reading itself (the reference bundle) and stay on. The others (`vorder`, `pos`) are CANDIDATES:
        each has a NEED SIGNAL measured on the training annotation (does the form of this world ever require it?) and an ACCEPTANCE TEST
        measured on SELECTION rows the trial models did not train on. The rule:

            a route is on in a world  iff  its need signal fires
                                       and the trial with it loses no selection split by more than eps
                                       and it wins the rows its need signal marked by more than eps (or every selection row).

        The trials, the numbers and the decision are written into model/config.json under "activation" and into run_dir/activation.json;
        the final model is then trained on the full training split with the accepted codes. With `--activate off` the legend's reader.codes
        (or --codes) are taken as declared, as before.

     3. DIAGNOSTICS. What no rule or route can fix, reported and never acted on: tokens without a kind (-> a typer, typer.py, or a lexicon
        entry); same-kind competition that the accepted bundle still loses on the selection rows (-> a question missing from the legend,
        or a world to split in two); training rows that the reader cannot explain (residual > 0 with the gold structure: examples outside
        the legend).

The need signals are computed by `profile(rows, legend, schema)` from the training rows only (never the reported splits) and can be
printed without any training: `sop profile <world>`.
"""
import collections, copy, json, os, shutil, tempfile

import numpy as np

from . import world

BASE_CODES = ("path", "chainL", "bigram")
CANDIDATE_CODES = ("vorder", "pos")
CONTRACT = {"candidates": "sentence", "undecided_adjacent": True, "residual": "explained"}      # the contract rules, as config.json keys
DECODERS = {"contract": CONTRACT, "paper": {}}


# ------------------------------------------------------------------ the nature of a world
def _head_positions(row, legend):
    """[(role, token index of the role's head, kind of the head, cued?)] for the word/entity fillers of a training row; a word used
    twice is resolved left to right; cued = the token before the head is a role cue (legend reader.gates_given.open) """
    cues = set((legend.reader.get("gates_given") or {}).get("open") or [])
    toks = row["tokens"]; used = set(); out = []
    for role, fl in sorted(row["frame"]["roles"].items(), key=lambda kv: kv[0]):
        if fl is None or fl[0] not in ("word", "negword", "entity"): continue
        h = fl[1]
        js = [j for j, t in enumerate(toks) if (t == h or t.lower() == h.lower()) and j not in used]
        if not js: continue
        j = js[0]; used.add(j)
        cued = j > 0 and (legend.kind(toks[j - 1]) in cues or toks[j - 1] in cues)
        out.append((role, j, legend.kind(h), cued))
    return out


def profile(rows, legend, schema):
    """the measured nature of a world from its TRAINING rows: what its form requires of a reader. Pure numbers; no training.
       same_kind_pairs: rows where two roles are filled by words of the SAME kind (competition): the need signal of `vorder`, because the
                        order relative to the verb is then a second, independent address for the pair, next to the cue the chain reads
                        (a new formulation may drop or move the cue; the rows the acceptance test must win);
       order_dependent: the subset where neither word is introduced by a cue, so that only the order separates them (then the chain
                        alone cannot read the pair: the strict form of the signal, and the diagnostic when no route wins them);
       rigid_order:     the share of rows in which each core role's head sits at its modal absolute position: the need signal of `pos`;
       depth:           rows with reported speech (levels): the chain with level openers is required (declared by the legend, checked);
       unknown_share:   tokens with no kind in the lexicon (-> typer / lexicon);
       same_kind_pairs: how often any two roles share a kind at all (competition, cued or not)."""
    train = [r for r in rows if r.get("split") == "train" and r["frame"]["shape"] == "ok"]
    n = len(train); od_ids, pair_ids = [], []
    pos_counts = collections.defaultdict(collections.Counter); role_rows = collections.Counter()
    n_tok = n_unk = 0; depth_rows = 0; verbs = set(legend.all_verbs())
    verb_seen = 0
    for r in train:
        hp = _head_positions(r, legend)
        kinds = collections.defaultdict(list)
        for role, j, k, cued in hp:
            kinds[k].append((role, cued)); pos_counts[role][j] += 1; role_rows[role] += 1
        if any(len(v) >= 2 for v in kinds.values()): pair_ids.append(r["id"])
        if any(len(v) >= 2 and sum(1 for _, c in v if not c) >= 2 for v in kinds.values()): od_ids.append(r["id"])
        if r["depth"] > 0: depth_rows += 1
        for t in r["tokens"]:
            n_tok += 1
            if legend.kind(t) is None: n_unk += 1
            if legend.lemma_of(t) in verbs or t in verbs: verb_seen += 1
    rigid = {}
    for role in schema.get("core_roles") or schema.get("role_order") or []:
        c = pos_counts.get(role)
        rigid[role] = (c.most_common(1)[0][1] / role_rows[role]) if c and role_rows[role] else None
    rigid_min = min((v for v in rigid.values() if v is not None), default=None)
    prof = {"n_train_rows": n,
            "order_dependent": {"n": len(od_ids), "share": len(od_ids) / max(1, n), "rows": od_ids},
            "same_kind_pairs": {"n": len(pair_ids), "share": len(pair_ids) / max(1, n), "rows": pair_ids},
            "rigid_order": {"per_core_role": rigid, "min": rigid_min},
            "depth": {"n": depth_rows, "share": depth_rows / max(1, n), "level_openers_declared": bool(legend.reader.get("level_openers"))},
            "cues_declared": sorted((legend.reader.get("gates_given") or {}).get("open") or []),
            "unknown_share": n_unk / max(1, n_tok), "kinds_source": legend.kinds_source,
            "verb_tokens_per_row": verb_seen / max(1, n)}
    return prof


def need_signals(prof, min_rows=10, min_share=0.02, rigid_at=0.9):
    """{code: (fires, reason)} for the candidate routes, from a profile"""
    od = prof["order_dependent"]; sk = prof["same_kind_pairs"]; rg = prof["rigid_order"]["min"]
    out = {"vorder": (sk["n"] >= min_rows and sk["share"] >= min_share,
                      f"{sk['n']} training rows ({sk['share']:.1%}) fill two roles with the same kind; in {od['n']} of them neither word is cued, so only the "
                      f"order relative to the verb separates them" if sk["n"] else "no training row fills two roles with the same kind: the kinds alone separate the roles"),
           "pos": (rg is not None and rg >= rigid_at and prof["n_train_rows"] >= min_rows,
                   (f"every core role's head sits at one absolute position in at least {rg:.0%} of the rows (rigid order)" if rg >= rigid_at else
                    f"the most frequent absolute position of a core role's head covers only {rg:.0%} of the rows: the order is not rigid") if rg is not None else "no core roles")}
    return out


def diagnostics(prof, accepted, sel_rows_fail=None):
    d = []
    if prof["kinds_source"] != "lexicon" or prof["unknown_share"] > 0.05:
        d.append({"signal": "kinds", "value": prof["unknown_share"], "message": f"{prof['unknown_share']:.1%} of the training tokens have no kind: no route "
                  "reads a word without a kind; learn the kinds from the annotation (sop/typer.py) or complete the lexicon"})
    if prof["depth"]["n"] and not prof["depth"]["level_openers_declared"]:
        d.append({"signal": "levels", "value": prof["depth"]["share"], "message": "reported speech in the examples but no reader.level_openers in the legend: the depth "
                  "cannot be estimated from kinds; declare the opener kinds"})
    if sel_rows_fail:
        d.append({"signal": "competition", "value": sel_rows_fail, "message": f"{sel_rows_fail} selection rows with same-kind competition are still misread with the accepted "
                  "bundle: no route separates them; a question is missing from the legend or this world should be split"})
    return d


# ------------------------------------------------------------------ the selection
def select(a, legend, rows, splits, schema, train_one, evaluate, lexical, bag, seen_words, log=print, eps=0.01, sel_share=0.1):
    """returns (codes, report). Trials: the base bundle and base + each firing candidate (and base + all of them when two or more fire),
    trained on the training split minus the SELECTION rows, scored on the selection rows (FULL, catalog candidates, primary readout).
    The selection rows = the calibration_* halves of the held-out formulations (never reported, rule 25) + a hashed tenth of the training
    rows (removed from the trials' training, restored for the final model)."""
    from .train import h01
    prof = profile(rows, legend, schema); needs = need_signals(prof)
    declared = list(a.code_list)
    cands = [c for c in CANDIDATE_CODES if needs[c][0]]
    rep = {"rule": "a candidate route is on iff its need signal fires and, on selection rows its trials did not train on, it loses no split by "
                   f"more than tol and wins the rows its need signal marked (or all rows) by more than tol; tol = max(eps={eps}, one row of the split)",
           "profile": {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) and "rows" in v else v) for k, v in prof.items()},
           "need_signals": {c: {"fires": f, "reason": r} for c, (f, r) in needs.items()}, "declared_codes": declared, "candidates": cands, "trials": {}}
    base = [c for c in BASE_CODES]
    if not cands:
        rep["accepted"] = []; rep["codes"] = base; rep["diagnostics"] = diagnostics(prof, [])
        log(f"[activate] no candidate route's need signal fires: codes {base}; " + "; ".join(f"{c}: {needs[c][1]}" for c in CANDIDATE_CODES))
        return base, rep
    # selection rows
    saved = {r["id"]: r.get("split") for r in rows}
    od = set(prof["same_kind_pairs"]["rows"])         # the rows the vorder trial must win: same-kind competition (cued or not)
    sel = []
    for r in rows:
        s = r.get("split")
        if s and s.startswith("calibration_"): sel.append(r)
        elif s == "train" and h01(a.seed, "sel", r["text"]) < sel_share: r["split"] = "selection"; sel.append(r)
    sel_od = [r for r in sel if r["id"] in od]
    log(f"[activate] candidates {cands} ({'; '.join(needs[c][1] for c in cands)}); selection rows {len(sel)} (of which with same-kind competition {len(sel_od)}); "
        f"trials trained on {sum(1 for r in rows if r.get('split') == 'train')} rows")
    bundles = [base] + [base + [c] for c in cands] + ([base + cands] if len(cands) > 1 else [])
    scores = {}
    tmp = tempfile.mkdtemp(prefix="sop_activate_")
    try:
        for codes in bundles:
            a2 = copy.copy(a); a2.code_list = list(codes); a2.no_ckpt = True; a2.store_presence_ratios = False
            md = os.path.join(tmp, "_".join(codes), "model")
            rd, _ = train_one(a2, legend, rows, splits, schema, a.seed, md, os.path.join(tmp, "_".join(codes)))
            ev = evaluate(rd, sel, legend, schema, a.tau, None, lexical, 10 ** 9, bag=bag, seen_words=seen_words, modes="primary")
            ev_od = evaluate(rd, sel_od, legend, schema, a.tau, None, lexical, 10 ** 9, bag=bag, seen_words=seen_words, modes="primary") if sel_od else None
            by_split, n_split = {}, {}
            for s in sorted({saved[r["id"]] for r in sel}):
                sub = [r for r in sel if saved[r["id"]] == s]; n_split[s] = len(sub)
                by_split[s] = evaluate(rd, sub, legend, schema, a.tau, None, lexical, 10 ** 9, bag=bag, seen_words=seen_words, modes="primary")["acc"]["FULL"]
            scores[tuple(codes)] = {"FULL": ev["acc"]["FULL"], "FULL_marked": ev_od["acc"]["FULL"] if ev_od else None, "n": len(sel), "n_marked": len(sel_od),
                                    "by_split": by_split, "n_split": n_split, "residual_mean": ev["residual_mean"]}
            log(f"[activate] trial {'+'.join(codes)}: FULL {ev['acc']['FULL']:.3f} on {len(sel)} selection rows" +
                (f", {ev_od['acc']['FULL']:.3f} on the {len(sel_od)} with same-kind competition" if ev_od else "") + " | " + " ".join(f"{s} {v:.3f}" for s, v in by_split.items()))
    finally:
        for r in rows: r["split"] = saved[r["id"]]
        shutil.rmtree(tmp, ignore_errors=True)
    b = scores[tuple(base)]
    tol = lambda n: max(eps, 1.0 / max(1, n))          # a split of n rows cannot move by less than one row: the tolerance is at least one row
    accepted = []
    for c in cands:
        t = scores[tuple(base + [c])]
        loses = [s for s, v in t["by_split"].items() if v < b["by_split"][s] - tol(b["n_split"][s])]
        wins_marked = (t["FULL_marked"] is not None and b["FULL_marked"] is not None and t["FULL_marked"] > b["FULL_marked"] + tol(t["n_marked"])) if c == "vorder" else False
        wins_all = t["FULL"] > b["FULL"] + tol(t["n"])
        ok = not loses and (wins_marked or wins_all)
        rep["trials"]["+".join(base + [c])] = dict(t, loses=loses, wins_marked_rows=wins_marked, wins_all=wins_all, accepted=ok)
        if ok: accepted.append(c)
    rep["trials"]["+".join(base)] = b
    codes = base + accepted
    if len(accepted) > 1 and tuple(base + cands) in scores:
        t = scores[tuple(base + cands)]; rep["trials"]["+".join(base + cands)] = t
        if any(t["by_split"][s] < scores[tuple(base + [accepted[0]])]["by_split"][s] - tol(t["n_split"][s]) for s in t["by_split"]):
            log(f"[activate] the routes together lose a split that one alone keeps: only {accepted[0]} kept"); codes = base + accepted[:1]; accepted = accepted[:1]
    fail = None
    if sel_od and "vorder" in cands:
        best = scores[tuple(codes)] if tuple(codes) in scores else b
        fail = int(round((1 - (best["FULL_marked"] or 0)) * len(sel_od)))
    rep["accepted"] = accepted; rep["codes"] = codes; rep["diagnostics"] = diagnostics(prof, accepted, fail if fail else None)
    if declared and sorted(declared) != sorted(codes): rep["note"] = f"the declared codes {declared} are replaced by the selected {codes} (--activate auto)"
    log(f"[activate] accepted {accepted or 'none'}: codes {codes}" + (f"; diagnostics: " + "; ".join(d["message"] for d in rep["diagnostics"]) if rep["diagnostics"] else ""))
    return codes, rep


def profile_markdown(prof, needs, diags):
    rg = prof["rigid_order"]["min"]; rg_txt = "n/a" if rg is None else f"{rg:.0%}"
    L = ["# World profile (the nature of the world, from its training rows)", "",
         f"- training rows: {prof['n_train_rows']}; kinds source: {prof['kinds_source']}; tokens without a kind: {prof['unknown_share']:.1%}",
         f"- rows with same-kind competition (two roles, one kind): {prof['same_kind_pairs']['n']} ({prof['same_kind_pairs']['share']:.1%}); of them order-dependent (no cue): {prof['order_dependent']['n']} ({prof['order_dependent']['share']:.1%})",
         f"- rigid absolute order (min over core roles): {rg_txt}",
         f"- rows with levels: {prof['depth']['n']} ({prof['depth']['share']:.1%}); level openers declared: {prof['depth']['level_openers_declared']}; cues declared: {prof['cues_declared']}",
         "", "## Need signals (candidate routes)", ""]
    for c, (f, r) in needs.items(): L.append(f"- `{c}`: {'FIRES' if f else 'off'}: {r}")
    L += ["", "## Contract rules (always on with decoder `contract`)", "", "- " + ", ".join(f"{k}={v}" for k, v in CONTRACT.items())]
    if diags: L += ["", "## Diagnostics", ""] + [f"- {d['signal']}: {d['message']}" for d in diags]
    return "\n".join(L) + "\n"
