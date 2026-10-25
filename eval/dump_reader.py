"""dump_reader.py (responses R38 / R39, coordinator's refinement of 2026-10-25): per-item dumps of the reader on the NEW-FORMULATION
report rows of the DESIGN SET (seeds 0-4 only; seeds 5-9 are never opened here) for the rows 1a, R14 and R37, so that three hypotheses
about the reader's loss on new formulations can be separated without a new run:
  H1 gates: errors follow which cue gates are open (gate value per token, the seed's learned gate set);
  H2 routes: errors follow the role's position / transition whatever the gates (template, transitions, filler position vs the verb);
  H3 decoding: errors are near misses (gold candidate ranked 2nd, small margin; the one-to-one rule's effect).
The reader of each seed is the reported model (loaded from logs/models/ or retrained exactly as its row; per-role rule EQUAL-checked
against the stored row, eval/reader_models.py). Per item: axis, index, template, kind, depth, tokens, the category and the GATE VALUE of
every token, the category bigrams (transitions) with those absent from the training distribution marked (the training distribution =
this seed's calibration rows, rule 25), the residual (common/decode.residuals, at the gold depth), and per role: gold, predicted (per-role
rule) and one-to-one prediction, rank of the gold candidate, top-1 / top-2 / gold scores, margin, the gold filler's word position and
whether it stands before the event verb. Per seed: every category's gate logit and value, the open set. Also writes the calibration rows'
residual and token count (R41's check reads them; no report quantity). Writes results/dump_<tag>_seed<s>.json.
Run: ./run.sh eval/dump_reader.py --row 1a|r14|r37 [--seeds 5]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode
from eval import metrics
from eval.reader_models import get_reader, ROWS

AXES_NF = ["new_form_seen_trans", "new_form_unseen_trans", "new_form_role_cues_seen"]
NONVERB = {"P", "O", "COP", "from", ",", "what", "it", "to", "who", "by", "says", "claims", ".", "that"}

def bigrams(cats): return [f"{a}>{b}" for a, b in zip(["<s>"] + cats, cats + ["</s>"])]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--row", required=True, choices=list(ROWS)); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    assert a.seeds <= 5, "the design set is seeds 0-4 (R39); seeds 5-9 are not opened"
    tag = ROWS[a.row]; os.makedirs(os.path.join(mainrepo.ROOT, "results"), exist_ok=True)
    print(f"dump_reader: main repository {mainrepo.check_pinned()}; row {a.row} ({tag}); design set seeds 0-{a.seeds - 1}; axes {AXES_NF} + calibration residuals")
    for seed in range(a.seeds):
        m, bw, split, preds, how, chk = get_reader(a.row, seed, a.device); w, world = bw["w"], bw["world"]
        gl = m.logits.detach().float().cpu(); gate = {c: float(torch.sigmoid(gl[i])) for i, c in enumerate(m.cats)}
        cat_of = lambda t: w.cat[t]
        train_bg = {b for it in split["axes"]["calibration"] for b in bigrams([cat_of(t) for t in it["tokens"]])}
        out = {"seed": seed, "row": a.row, "tag": tag, "model": how, "check": chk, "gate_logits": {c: float(gl[i]) for i, c in enumerate(m.cats)}, "gates": gate,
               "open": sorted(c for c, v in gate.items() if v > 0.5), "items": []}
        for ax in AXES_NF:
            items = split["axes"][ax]; P = preds[ax]; res = decode.residuals(m, world, items, P)
            for j, (it, p, rr) in enumerate(zip(items, P, res)):
                cats = [cat_of(t) for t in it["tokens"]]; roles = bwm.roles_at(it["depth"]); o2o = metrics.one_to_one(p, roles)
                vpos = next((k for k, c in enumerate(cats) if c not in NONVERB), None); bg = bigrams(cats)
                rec = {"axis": ax, "i": j, "template": it["template"], "kind": it["kind"], "depth": it["depth"], "tokens": it["tokens"], "cats": cats,
                       "gate": [gate[c] for c in cats], "transitions": bg, "new_transitions": [b for b in bg if b not in train_bg], "verb_pos": vpos,
                       "residual": rr, "roles": {}}
                for r in roles:
                    cands, sc = p["cand_scores"][r]; sc = np.array(sc); order = np.argsort(-sc); g = it["frame"][r]
                    gi = cands.index(g) if g in cands else None; rank = int(np.where(order == gi)[0][0]) + 1 if gi is not None else None
                    gpos = it["tokens"].index(g) if g in it["tokens"] else None
                    rec["roles"][r] = {"gold": g, "pred": p["frame"][r], "o2o": o2o["frame"][r], "correct": p["frame"][r] == g, "o2o_correct": o2o["frame"][r] == g,
                                       "rank_gold": rank, "top1": float(sc[order[0]]), "top2": float(sc[order[1]]) if len(sc) > 1 else None,
                                       "gold_score": float(sc[gi]) if gi is not None else None, "margin": p["margins"][r], "gold_pos": gpos,
                                       "before_verb": (gpos < vpos) if (gpos is not None and vpos is not None) else None,
                                       "pred_is_other_role_filler": p["frame"][r] in [it["frame"][q] for q in roles if q != r], "pred_in_sentence": p["frame"][r] in it["tokens"]}
                out["items"].append(rec)
        cal = split["axes"]["calibration"]; rc = decode.residuals(m, world, cal, preds["calibration"])
        out["calibration"] = {"residual": rc, "n_tokens": [len(it["tokens"]) for it in cal], "depth": [it["depth"] for it in cal]}
        p = os.path.join(mainrepo.ROOT, "results", f"dump_{tag}_seed{seed}.json"); json.dump(out, open(p, "w"))
        nerr = sum(not v["correct"] for it in out["items"] for v in it["roles"].values())
        print(f"  seed {seed}: {how}; per-role rule vs stored row: {chk}; open gates {out['open']}; {len(out['items'])} items, {nerr} wrong role decisions; written {os.path.relpath(p, mainrepo.ROOT)}", flush=True)
        del m; torch.cuda.empty_cache() if a.device == "cuda" else None
    metrics.checklist(a.seeds, "[x] design set only (seeds 0-4, R39) [x] the reported model (EQUAL check) [x] thresholds not used")
