"""r78_readout.py (responses R78 item 1, R84 item 2, first part): the per-intent readout of the R81 configuration, CPU only, on the SAVED
models (no training), read through a named sop version (--sop_dir). Per world (= the gold intent of the utterance), on the same utterances:
span F1 of the typer alone (R71 item 1) against typer + reader with (a) the frame of the dispatched world (R80's anchor-percentile rule)
and (b) the frame of the gold intent's world (oracle dispatch: the readout alone). Per question (slot type) of the world: span F1 typer
alone vs typer + reader (oracle), the reader's head decisions against the gold head frame (missed = gold present / read absent; false
fill = read present / gold absent; wrong head), and the question's same-kind competition on the world's training rows (the share of its
training occurrences in which another question of the same row has a head of the same kind: the need signal of `vorder`, per question).
Per world: the activation profile and need signals (sop activate.profile / need_signals on the world's positive training rows), unplaced
words of the oracle reading (most frequent), and the presence queries when the sop version writes them.
Writes logs/results_r78_readout_<condition>_seed<s>.json.
Run: ./run.sh eval/r78_readout.py --sop_dir DIR --seed S [--condition base] [--sets test]"""
import argparse, collections, importlib, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd, report_real
from eval import metrics
import numpy as np


def f1(g, p):
    tp = len(g & p); P = tp / len(p) if p else 0.0; R = tp / len(g) if g else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sop_dir", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="base"); ap.add_argument("--models", default="r81"); ap.add_argument("--sets", default="test,dev_rep")
    ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only"); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir); TR = importlib.import_module("sop_repo.train"); ACT = importlib.import_module("sop_repo.activate"); LG = importlib.import_module("sop_repo.legend")
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", a.condition); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    W = wr.Worlds(SR, cond, a.condition, a.seed, a.models, kinds)
    print(f"r78_readout: models {a.models}; sop @ {src}; condition {a.condition}; seed {a.seed}; worlds {W.intents}")
    out = {"sop": src, "models": a.models, "condition": a.condition, "seed": a.seed, "worlds": {}, "sets": {}}
    for I in W.intents:   # the world's nature, from its positive training rows
        w = W.W[I]; L = LG.Legend(os.path.join(w["dir"], "legend.json")); rows, errors, _ = TR.load_rows(w["dir"], L)
        tset = set(i for i in cond["train"] if items[i]["intent"] == I)
        for r in rows: r["split"] = "train" if (r["wire"].value("_item") in tset and not r.get("negative")) else "other"
        prof = ACT.profile(rows, L, w["R"].schema); needs = ACT.need_signals(prof)
        comp = collections.Counter(); occ = collections.Counter()
        for r in rows:
            if r["split"] != "train" or r["frame"]["shape"] != "ok": continue
            hp = ACT._head_positions(r, L); bykind = collections.Counter(k for _, _, k, _ in hp)
            for role, j, k, cued in hp:
                q = w["rinv"].get(role, role); occ[q] += 1
                if bykind[k] >= 2: comp[q] += 1
        out["worlds"][I] = {"need_signals": {c: {"fires": bool(f), "reason": rs} for c, (f, rs) in needs.items()}, "same_kind_pairs_share": prof["same_kind_pairs"]["share"],
                            "order_dependent_share": prof["order_dependent"]["share"], "rigid_order_min": prof["rigid_order"]["min"],
                            "competition_by_question": {q: comp[q] / occ[q] for q in occ}, "train_occurrences": dict(occ)}
        print(f"  world {I}: vorder need {'FIRES' if needs['vorder'][0] else 'no'} ({prof['same_kind_pairs']['share']:.1%} rows with same-kind pairs, {prof['order_dependent']['share']:.1%} order-dependent); "
              f"pos need {'FIRES' if needs['pos'][0] else 'no'} (rigid min {prof['rigid_order']['min']})", flush=True)
    for sname in a.sets.split(","):
        ids = cond["eval"][sname][:a.max_eval] if a.max_eval else cond["eval"][sname]
        per = collections.defaultdict(lambda: {"n": 0, "gold": [], "typer": [], "disp": [], "orac": [], "dispatch_ok": 0, "unplaced": collections.Counter(), "pq": collections.Counter(),
                                               "q": collections.defaultdict(lambda: collections.Counter())})
        for i in ids:
            it = items[i]; I = it["intent"]
            if I not in W.W: continue
            outs = W.read_all(it["tokens"]); D = W.dispatch_anchor(outs); g = report_real.gold_flat(it)
            gs = set(report_real.gold_spans(it)); ts = set(rr.spans_typer_only(it["tokens"], ty))
            ds = set(rr.spans_by_typer(it["tokens"], dict(outs[D]["frame"], intent=D), ty, T)); os_ = set(rr.spans_by_typer(it["tokens"], dict(outs[I]["frame"], intent=I), ty, T))
            p = per[I]; p["n"] += 1; p["dispatch_ok"] += D == I; p["gold"].append(gs); p["typer"].append(ts); p["disp"].append(ds); p["orac"].append(os_)
            p["unplaced"].update(outs[I]["unplaced"]); p["pq"].update(outs[I]["presence_queries"])
            fr = outs[I]["frame"]
            for q in set(W.W[I]["rinv"].values()):
                gq, rq = g.get(q), fr.get(q); c = p["q"][q]
                c["gold"] += gq is not None; c["read"] += rq is not None
                if gq is not None and rq is None: c["missed"] += 1
                elif gq is None and rq is not None: c["false_fill"] += 1
                elif gq is not None and gq != rq: c["wrong_head"] += 1
                elif gq is not None: c["right_head"] += 1
                for nm, src_ in (("typer", ts), ("orac", os_)):
                    gg = {x for x in gs if x[0] == q}; pp = {x for x in src_ if x[0] == q}
                    c[nm + "_tp"] += len(gg & pp); c[nm + "_np"] += len(pp); c["n_gold_spans"] += len(gg) if nm == "typer" else 0
        res = {}
        for I, p in sorted(per.items()):
            sf = lambda key: metrics.span_f1([list(x) for x in p["gold"]], [list(x) for x in p[key]])["f1"]
            qs = {}
            for q, c in p["q"].items():
                ng = c["n_gold_spans"]
                fq = lambda nm: (2 * c[nm + "_tp"] / (ng + c[nm + "_np"])) if ng + c[nm + "_np"] else None
                qs[q] = dict(c, f1_typer=fq("typer"), f1_reader_oracle=fq("orac"), competition=out["worlds"][I]["competition_by_question"].get(q, 0.0))
            res[I] = {"n": p["n"], "dispatch": p["dispatch_ok"] / max(1, p["n"]), "f1_typer": sf("typer"), "f1_reader_dispatched": sf("disp"), "f1_reader_oracle": sf("orac"),
                      "questions": qs, "unplaced_top": p["unplaced"].most_common(10), "presence_queries": dict(p["pq"])}
            r_ = res[I]
            print(f"  [{sname}] {I:22s} n {p['n']:3d}: dispatch {r_['dispatch']:.3f} | span F1 typer {r_['f1_typer']:.3f}, typer + reader (dispatched) {r_['f1_reader_dispatched']:.3f}, (oracle world) {r_['f1_reader_oracle']:.3f}"
                  f" | unplaced {r_['unplaced_top'][:5]}" + (f" | presence queries {dict(p['pq'].most_common(5))}" if p["pq"] else ""), flush=True)
            for q, c in sorted(qs.items(), key=lambda kv: -(kv[1]["n_gold_spans"])):
                if not c["n_gold_spans"] and not c["read"]: continue
                d = (c["f1_reader_oracle"] or 0.0) - (c["f1_typer"] or 0.0)
                print(f"      {q:28s} gold {c['n_gold_spans']:3d}: F1 typer {c['f1_typer'] if c['f1_typer'] is not None else float('nan'):.3f} reader {c['f1_reader_oracle'] if c['f1_reader_oracle'] is not None else float('nan'):.3f} ({d:+.3f}) | "
                      f"right head {c.get('right_head', 0)}, wrong head {c.get('wrong_head', 0)}, missed {c.get('missed', 0)}, false fill {c.get('false_fill', 0)} | same-kind competition {c['competition']:.2f}", flush=True)
        out["sets"][sname] = res
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_r78_readout_{a.condition}_seed{a.seed}.json"), "w"), indent=1, default=list)
    metrics.checklist(1, "[x] saved models, no training [x] the same utterances for the typer and the reader [x] oracle and dispatched worlds apart [x] sop version named")
