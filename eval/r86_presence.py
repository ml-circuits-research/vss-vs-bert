"""r86_presence.py (responses R86 (a) and (b)): the cause of the presence losses of the per-intent readout, CPU only, on the SAVED R81 base
models (no training), read through a named sop version (--sop_dir; R81: 82585ea), with the gold intent's world (oracle: the readout alone).
(a) Admitted kinds: per question of each world, the kinds the schema admits (roles[r].kinds, head_kinds); for every FALSE FILL on the
    test utterances of the world (a question read present whose gold is absent): the filled word, the gold question whose span holds that
    word (or none), the typer's in-context kind of the word and its lexicon kind, and whether that kind is also admitted by the gold
    sibling question. Summary: the share of false fills that are words of a sibling question (the competition invisible to the profile).
    MISSES (gold present, read absent) are listed with the gold word's typer kind and the read presence ratio.
(b) Presence ratios: per question, the presence ratio (f['presence'][r], the ratio to the structure score that the presence threshold is
    applied to) on the world's calibration utterances (the first 200 of its own training utterances, as R79-R81's thresholds) split by
    gold present / absent: mean present, mean absent, separation, the stored threshold (config.json thresholds.presence, fitted at
    training), and the balanced accuracy of that threshold on these rows. Questions with separation < 0.1 are flagged (R86: interference).
Writes logs/results_r86_presence_seed<s>.json. Run: ./run.sh eval/r86_presence.py --sop_dir DIR --seed S [--worlds BookRestaurant,SearchScreeningEvent]"""
import argparse, collections, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd, report_real
from eval import metrics
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--worlds", default="BookRestaurant,SearchScreeningEvent"); ap.add_argument("--models", default="r81"); ap.add_argument("--n_freq", type=int, default=40); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir)
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", "base"); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    print(f"r86_presence: models {a.models}; sop @ {src}; seed {a.seed}; worlds {a.worlds}")
    out = {"sop": src, "seed": a.seed, "worlds": {}}
    for I in a.worlds.split(","):
        wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_base_intent_{I}_seed{a.seed}_{a.models}")
        leg = json.load(open(os.path.join(wd, "legend.json"))); rinv = {r["id"]: r["question"][5:] for r in leg["roles"] if r["question"].startswith("slot ")}; rmap = {v: k for k, v in rinv.items()}
        R = SR.infer.Reader(os.path.join(wd, "model_seed")); R.cfg["residual"] = "explained"; sch = R.schema
        thr = json.load(open(os.path.join(wd, "model_seed", "config.json")))["thresholds"]["presence"]
        adm = {rinv[r]: {"kinds": sorted(sch["roles"][r].get("kinds") or []), "head_kinds": sorted(sch["roles"][r].get("head_kinds") or [])} for r in sch["role_order"] if r in rinv}
        kind_owners = collections.defaultdict(set)
        for q, d in adm.items():
            for k in set(d["kinds"]) | set(d["head_kinds"]): kind_owners[k].add(q)
        print(f"== {I}: admitted kinds per question (kinds | head_kinds); a kind admitted by several questions is shared")
        for q, d in sorted(adm.items()):
            sh = sorted({k for k in set(d['kinds']) | set(d['head_kinds']) if len(kind_owners[k]) > 1})
            print(f"   {q:24s} kinds {d['kinds']} | head {d['head_kinds']} | shared with another question: {sh}")
        def read(toks):
            w, f, info = R.formalize(" ".join(toks), 0, SR.infer.idgen("e"), filt=set(toks)); return f
        # (b) presence ratios on the calibration utterances
        own = [i for i in cond["train"] if items[i]["intent"] == I][:200]; st = collections.defaultdict(lambda: {"present": [], "absent": []})
        for i in own:
            f = read(items[i]["tokens"]); g = report_real.gold_flat(items[i])
            for r in sch["role_order"]:
                if r not in rinv: continue
                st[rinv[r]]["present" if g.get(rinv[r]) is not None else "absent"].append(float(f["presence"].get(r, 0.0)))
        sep = {}
        for q, s in sorted(st.items()):
            mp = float(np.mean(s["present"])) if s["present"] else None; ma = float(np.mean(s["absent"])) if s["absent"] else None; t = thr.get(rmap[q])
            ba = None
            if s["present"] and s["absent"] and t is not None: ba = 0.5 * (np.mean(np.array(s["present"]) >= t) + np.mean(np.array(s["absent"]) < t))
            sep[q] = {"n_present": len(s["present"]), "n_absent": len(s["absent"]), "mean_present": mp, "mean_absent": ma, "separation": (mp - ma) if mp is not None and ma is not None else None, "threshold": t, "balanced_acc": ba}
            x = sep[q]
            print(f"   presence {q:24s} present {x['n_present']:3d} mean {mp if mp is not None else float('nan'):.3f} | absent {x['n_absent']:3d} mean {ma if ma is not None else float('nan'):.3f} | separation {x['separation'] if x['separation'] is not None else float('nan'):+.3f}"
                  f"{'  < 0.1' if x['separation'] is not None and x['separation'] < 0.1 else ''} | threshold {t if t is not None else float('nan'):.3f} balanced acc {ba if ba is not None else float('nan'):.3f}", flush=True)
        # (a) false fills and misses on the test utterances of this world (oracle world)
        ids = [i for i in cond["eval"]["test"] if items[i]["intent"] == I]; ff, miss = [], []
        for i in ids:
            it = items[i]; toks = it["tokens"]; f = read(toks); g = report_real.gold_flat(it); ks = ty.sentence_kinds(toks)
            owner = {}
            for (t_, s_, e_, _) in it["slots"]:
                for j in range(s_, e_): owner.setdefault(j, t_)
            for r in sch["role_order"]:
                if r not in rinv: continue
                q = rinv[r]; fl = f["roles"].get(r); word = fl[1] if fl is not None else None
                if word is not None and g.get(q) is None:
                    js = [j for j, t in enumerate(toks) if t == word]; j = js[0] if js else None
                    sib = owner.get(j) if j is not None else None; k = ks[j] if j is not None else None
                    ff.append({"q": q, "word": word, "sibling": sib, "kind": k, "lex_kind": kinds.get(word), "kind_admitted_by_sibling": (sib in kind_owners.get(k, set())) if sib else None, "ratio": float(f["presence"].get(r, 0.0)), "thr": thr.get(r)})
                if word is None and g.get(q) is not None:
                    js = [j for j, t in enumerate(toks) if t == g[q]]; k = ks[js[0]] if js else None
                    miss.append({"q": q, "gold_word": g[q], "kind": k, "kind_admitted": k in (set(adm[q]["kinds"]) | set(adm[q]["head_kinds"])), "ratio": float(f["presence"].get(r, 0.0)), "thr": thr.get(r)})
        n = max(1, len(ff)); sibs = [x for x in ff if x["sibling"]]
        summ = {"n_false_fills": len(ff), "share_sibling_word": len(sibs) / n, "share_sibling_kind_admitted_by_sibling": sum(1 for x in sibs if x["kind_admitted_by_sibling"]) / max(1, len(sibs)),
                "share_no_gold_question": sum(1 for x in ff if not x["sibling"]) / n, "n_misses": len(miss), "misses_kind_admitted": sum(1 for x in miss if x["kind_admitted"]) / max(1, len(miss)),
                "false_fill_pairs": collections.Counter(f"{x['q']} <- {x['sibling']}" for x in ff).most_common(15), "miss_by_question": collections.Counter(x["q"] for x in miss).most_common()}
        print(f"   (a) test {len(ids)} utterances: false fills {len(ff)}; the filled word belongs to a sibling gold question in {summ['share_sibling_word']:.3f} "
              f"(its typer kind admitted by that sibling too: {summ['share_sibling_kind_admitted_by_sibling']:.3f}); to no gold question {summ['share_no_gold_question']:.3f}; "
              f"misses {len(miss)} (gold word's kind admitted by the question: {summ['misses_kind_admitted']:.3f})", flush=True)
        print(f"   false fill (question <- the word's gold question): {summ['false_fill_pairs']}"); print(f"   misses by question: {summ['miss_by_question']}")
        for x in ff[:12]: print(f"     false fill {x['q']}: '{x['word']}' (gold question {x['sibling']}, typer kind {x['kind']}, lexicon {x['lex_kind']}; ratio {x['ratio']:.3f} vs threshold {x['thr'] if x['thr'] is not None else float('nan'):.3f})")
        out["worlds"][I] = {"admitted": adm, "presence": sep, "summary": summ, "false_fills": ff, "misses": miss}
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_r86_presence_seed{a.seed}.json"), "w"), indent=1, default=str)
    metrics.checklist(1, "[x] saved R81 models, no training [x] oracle world (the readout alone) [x] calibration rows = the world's own training rows [x] sop version named")
