"""planted.py (deviation R26 / Q19, run under the stopping rule R48 item 4; declared in RESULTS 'The planted-sets row' before it runs):
query emission on the FROZEN planted sets data/planted_seed<s>.json (seeds 0-4, committed at e6c6f68 before any model saw them), for the
reader 1a (original decoder, and R46's sentence candidates + one-to-one) and the saved small MRC models. Scored on the OUTPUT.
  depth: the level-opener count (no depth given), for every model;
  'undecided' (300 per seed: a second identity of the same class right after one role's filler): the planted role is QUERIED when its
      margin is below the model's threshold t = the 2% quantile of its role margins on the calibration rows (the R17 cap: at most 2% of
      in-distribution decisions queried; rule 18's tau is 0.000 for every model on this world and would query nothing); otherwise its
      value is INVENTED (the gold is a query);
  'unsupported' (300 per seed: ', <id> .' appended, an identity no question asks for): the model emits 'unsupported' for every identity
      token (kind P or O) that it places in no role (the same untrained coverage test for every model: no coverage head is trained, the
      declared set has no new training); the planted token is RIGHTLY QUERIED if it is placed in no role, INVENTED if some role took it;
      an original identity left unplaced is a NEEDLESS 'unsupported' query;
  the other roles of both sets: correct value, needlessly queried (margin below t on a right value), confident error (wrong value at or
      above t), or queried-and-wrong.
Hallucination rate (R30 terminology: used once, defined) = invented planted decisions + confident errors on the other roles, over all
decisions. Writes logs/results_planted_<model>_seed<s>.json.
R54 (2026-10-25): every row also counts, in the 'undecided' set, whether the inserted identity was left in no role (a query of another
kind); model 'reader_r54' = the R46 decoder plus R54's adjacency rule (r54_adjacent). Results of these runs: logs/results_planted2_*.
Run: ./run.sh eval/planted.py --model reader | reader_r46 | reader_r54 | <mrc tag> [--seeds 5] [--seed_start 5 for reader_r54's claim]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np
from common import basic_world as bwm, decode
from eval import metrics

def est(items): return [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]

def r54_adjacent(items, preds, cat):
    """responses R54 item 2 (declared in RESULTS): an identity left in no role, adjacent (the token before or after) to the filler of a
    role whose kinds admit it, turns that role into 'undecided' with both as candidates; the role's margin = the gap between their
    scores at the role's address (from the R46 decoder's in-sentence candidate scores)"""
    for it, p in zip(items, preds):
        placed = {v for v in p["frame"].values() if v is not None}; und = {}; toks = it["tokens"]
        for k, w in enumerate(toks):
            if cat.get(w) not in ("P", "O") or w in placed: continue
            for j in (k - 1, k + 1):
                if 0 <= j < len(toks):
                    for r, f in p["frame"].items():
                        if f is not None and f == toks[j] and (("O" if r == "B" else "P") == cat.get(w)) and r not in und:
                            cs, sc = p.get("cand_scores", {}).get(r, ([], [])); d = dict(zip(cs, sc))
                            und[r] = (f, w); p["margins"][r] = float(abs(d.get(f, 0.0) - d.get(w, 0.0)))
        p["undecided_roles"] = und
    return preds

def score(items, preds, t, kind, cat):
    c = {"planted_queried": 0, "planted_invented": 0, "n_planted": 0, "other_correct": 0, "other_needless_query": 0, "other_confident_error": 0,
         "other_queried_wrong": 0, "n_other": 0, "needless_unsupported": 0, "n_identity_tokens": 0,
         "second_identity_unplaced": 0, "planted_queried_as_undecided": 0}
    for it, p in zip(items, preds):
        roles = bwm.roles_at(it["depth"]); placed = {v for v in p["frame"].values() if v is not None}
        if kind == "undecided":
            r0 = it["role"]; c["n_planted"] += 1; und = p.get("undecided_roles", {})
            if r0 in und or p["frame"].get(r0) is None or p["margins"].get(r0, 0.0) < t: c["planted_queried"] += 1
            else: c["planted_invented"] += 1
            second = [x for x in it["candidates"] if x != it["frame"][r0]]          # R54 (1): the inserted identity left in no role
            c["second_identity_unplaced"] += bool(second) and second[0] not in placed
        else:
            tok = it["tokens"][it["planted_index"]]; c["n_planted"] += 1; und = p.get("undecided_roles", {})
            if tok in placed: c["planted_invented"] += 1
            else: c["planted_queried"] += 1; c["planted_queried_as_undecided"] += any(tok in v for v in und.values())
            for k, w in enumerate(it["tokens"]):
                if k != it["planted_index"] and cat.get(w) in ("P", "O"): c["n_identity_tokens"] += 1; c["needless_unsupported"] += w not in placed
        for r in roles:
            if kind == "undecided" and r == it["role"]: continue
            c["n_other"] += 1; v = p["frame"].get(r); ok = v == it["frame"][r]; q = v is None or p["margins"].get(r, 0.0) < t or r in p.get("undecided_roles", {})
            c["other_correct" if ok and not q else "other_needless_query" if ok and q else "other_confident_error" if not q else "other_queried_wrong"] += 1
    n = c["n_planted"] + c["n_other"]; c["hallucination_rate"] = (c["planted_invented"] + c["other_confident_error"]) / max(1, n)
    c["planted_queried_share"] = c["planted_queried"] / max(1, c["n_planted"]); return c

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    print(f"planted (R26 / Q19 / R48 item 4): main repository {mainrepo.check_pinned()}; model {a.model}; seeds 0-{a.seeds - 1}; frozen planted sets; depth from the level openers; threshold = 2% quantile of the calibration margins")
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        split = bwm.load(seed); pl = json.load(open(os.path.join(mainrepo.ROOT, "data", f"planted_seed{seed}.json"))); cat = split["cat"]
        if a.model.startswith("reader"):
            from eval.reader_models import get_reader
            m, bw, _, _, how, chk = get_reader("1a_fresh" if seed >= 5 else "1a", seed, a.device); sc = a.model in ("reader_r46", "reader_r54")
            def dec(items):
                p = decode.decode_frozen(m, bw["world"], {"axes": {"x": items}, "candidates": split["candidates"]}, "x", sentence_candidates=sc)
                if sc:
                    from eval.sentence_candidates import one_to_one
                    p = [dict(one_to_one(pp, bwm.roles_at(it["depth"])), cand_scores=pp["cand_scores"]) for pp, it in zip(p, items)]
                if a.model == "reader_r54": p = r54_adjacent(items, p, cat)
                return p
            info = f"{how}; {chk}"
        else:
            import torch
            from eval.confirmatory import mrc_load
            import models.bert_mrc as bm
            model, tok, cfg = mrc_load(a.model, seed, a.device)
            def dec(items): return bm.read(model, tok, split, "x", items, a.device, given=True)
            info = f"loaded logs/models/{a.model}_seed{seed}"
        cal = est(split["axes"]["calibration"]); pc = dec(cal)
        t = float(np.quantile([pp["margins"][r] for pp, it in zip(pc, cal) for r in bwm.roles_at(it["depth"])], 0.02))
        res = {"seed": seed, "model": a.model, "threshold": t, "info": info}
        for kind in ("undecided", "unsupported"):
            items = est(pl[kind]); res[kind] = score(items, dec(items), t, kind, cat)
            r_ = res[kind]; print(f"  seed {seed} {kind:11s}: planted queried {r_['planted_queried_share']:.3f} ({r_['planted_queried']}/{r_['n_planted']}), invented {r_['planted_invented']}; other roles correct {r_['other_correct']}, needless query {r_['other_needless_query']}, confident error {r_['other_confident_error']}, queried wrong {r_['other_queried_wrong']} of {r_['n_other']}; needless 'unsupported' {r_['needless_unsupported']} of {r_['n_identity_tokens']}; hallucination rate {r_['hallucination_rate']:.3f}", flush=True)
        print(f"  seed {seed}: {info}; threshold {t:.4f}", flush=True)
        json.dump(res, open(os.path.join(mainrepo.ROOT, "logs", f"results_planted2_{a.model}_seed{seed}.json"), "w"))   # planted2: with R54's column; rows.append(res)
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== planted, {a.model}: MEANS over {len(rows)} seeds (per-seed list) ==")
    for kind in ("undecided", "unsupported"):
        print(f"  {kind}: planted queried {ms([r[kind]['planted_queried_share'] for r in rows])}; hallucination rate {ms([r[kind]['hallucination_rate'] for r in rows])}; other roles needless query {ms([r[kind]['other_needless_query'] / r[kind]['n_other'] for r in rows])}; confident error {ms([r[kind]['other_confident_error'] / r[kind]['n_other'] for r in rows])}")
    print(f"  undecided set: the inserted identity left in no role (R54 (1)) {ms([r['undecided']['second_identity_unplaced'] / r['undecided']['n_planted'] for r in rows])}; unsupported set: planted queried as 'undecided' (R54 (2)) {ms([r['unsupported']['planted_queried_as_undecided'] / r['unsupported']['n_planted'] for r in rows])}")
    print(f"  needless 'unsupported' on original identities {ms([r['unsupported']['needless_unsupported'] / max(1, r['unsupported']['n_identity_tokens']) for r in rows])}")
    print("  criterion (R26, the demo's): planted unsupported reported as queries >= 0.90")
    metrics.checklist(len(rows), "[x] planted sets frozen before any model (e6c6f68) [x] thresholds on calibration rows only (25) [x] depth not given [x] the same coverage test for every model")
