"""bert_mrc_growth.py (responses R49 / R50, declared in RESULTS 'Deviation R49'): the legend-growth rows of the small MRC models. From the
SAVED 'zs' model of one seed (logs/models/<tag>_seed<s>/, the block's row), each condition starts again from that model:
  cases:   'level3'    the new legend question, the level-3 speaker: k depth-3 training-distribution sentences of data/growth_seed<s>.json,
                       each annotated with all six legend questions (the k examples are sentences);
           'template4' the new form: k template-4 sentences ('{A} V {C} {B} .'), each annotated with the 'zs' questions (A, B, C; levels
                       1-2 NULL), the same sentences the reader's growth row adds;
  k:       50, 500 (the first 50 of the 500);
  regimes: (a) continued fine-tuning on the new examples only, lr 1e-4, 3 epochs (the declared recipe) and the ONE alternative lr 3e-5;
           (b) replay: the new examples added to the full training set of the block (the same 30,000 sentences, 'zs' questions), 1 epoch
               from the saved model, lr 1e-4;
  the optimiser and schedule of models/bert_mrc.train (AdamW, weight decay 0.01, 10% linear warm-up, batch 32); torch seed = the seed.
Measured per condition: wall-clock of the training (cuda synchronised; the run is ALONE on the GPU), the new question (level3: X3 slot
accuracy at depth 3 and the six-question exact match with the depth NOT given; template4: the template-4 cell of the new-formulation rows,
slot accuracy and exact match), every report axis before (the saved model, evaluated here) and after, the drop (forgetting), the number
of examples and of hyperparameter choices (a: lr, epochs; b: lr, epochs). Writes logs/results_growth_<tag>_seed<s>.json.
Run: ./run.sh models/bert_mrc_growth.py --tag mrc_google_bert_uncased_L-2_H-128_A-2_zs --seed S"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, report
from eval import metrics
import models.bert_mrc as bm
from models.bert_basic import make_train
from eval.confirmatory import mrc_load

ZS = lambda d: ["A", "B", "C", "X1", "X2"]

def evaluate(model, tok, split, device):
    P = {ax: bm.read(model, tok, split, ax, split["axes"][ax], device) for ax in report.AXES if split["axes"][ax]}
    slot = {ax: float(np.mean([p["frame"][r] == it["frame"][r] for it, p in zip(split["axes"][ax], P[ax]) for r in bwm.roles_at(it["depth"])])) for ax in P}
    g3 = split["axes"]["depth3_unseen"]; ng = bm.read(model, tok, split, "depth3_unseen", g3, device, given=False)
    x3 = float(np.mean([p["frame"]["X3"] == it["frame"]["X3"] for p, it in zip(ng, g3)])); six = float(np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.ROLES) for p, it in zip(ng, g3)]))
    its = [(it, p) for it, p in zip(split["axes"]["new_form_unseen_trans"], P["new_form_unseen_trans"]) if it["template"] == 4]
    t4 = {"slot": float(np.mean([p["frame"][r] == it["frame"][r] for it, p in its for r in ("A", "B", "C")])), "exact": float(np.mean([all(p["frame"][r] == it["frame"][r] for r in ("A", "B", "C")) for it, p in its])), "n": len(its)}
    return {"slot": slot, "x3_depth3_not_given": x3, "exact_six_depth3_not_given": six, "t4": t4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", default="50,500"); a = ap.parse_args(); seed = a.seed; label = "CLAIM SET (fresh splits)" if seed >= 5 else "DESIGN SET (inspected)"
    print(f"bert_mrc_growth (R49): main repository {mainrepo.check_pinned()}; {a.tag} seed {seed} ({label}); cases level3 / template4; k {a.ks}; regimes a (lr 1e-4, 3 epochs; alternative lr 3e-5) and b (replay, 1 epoch, lr 1e-4)")
    split = bwm.load(seed); bw = bwm.build(seed, split["D"]); g = json.load(open(os.path.join(mainrepo.ROOT, "data", f"growth_seed{seed}.json")))
    model, tok, cfg = mrc_load(a.tag, seed, a.device); before = evaluate(model, tok, split, a.device); del model
    print(f"  before: X3 (depth 3, not given) {before['x3_depth3_not_given']:.3f}; six-question exact match {before['exact_six_depth3_not_given']:.3f}; template-4 cell slot {before['t4']['slot']:.3f} exact {before['t4']['exact']:.3f}; slots {{{', '.join(f'{k}: {v:.3f}' for k, v in before['slot'].items())}}}", flush=True)
    base_ex = bm.examples(make_train(bw, seed, 30000), ZS); out = {"tag": a.tag, "seed": seed, "set": label, "before": before, "conditions": []}
    for case in ("level3", "template4"):
        for k in (int(x) for x in a.ks.split(",")):
            data = [(it["tokens"], it["tags"]) for it in g[case][:k]]; new = bm.examples(data, (lambda d: bwm.ROLES) if case == "level3" else ZS)
            for regime, lr, ep, ex in (("a", 1e-4, 3, new), ("a_alt_lr", 3e-5, 3, new), ("b_replay", 1e-4, 1, base_ex + new)):
                model, tok, _ = mrc_load(a.tag, seed, a.device); torch.manual_seed(seed)
                if a.device == "cuda": torch.cuda.synchronize()
                t0 = time.time(); tl = bm.train(model, tok, ex, lr, ep, 32, a.device, seed)
                if a.device == "cuda": torch.cuda.synchronize()
                tt = time.time() - t0; after = evaluate(model, tok, split, a.device); del model; torch.cuda.empty_cache() if a.device == "cuda" else None
                drop = {ax: before["slot"][ax] - after["slot"][ax] for ax in before["slot"]}
                new_q = after["x3_depth3_not_given"] if case == "level3" else after["t4"]["slot"]
                c = {"case": case, "k": k, "regime": regime, "lr": lr, "epochs": ep, "examples": k, "train_examples_seen": len(ex) * ep, "train_s": tt, "hyperparameter_choices": 2,
                     "after": after, "drop": drop, "worst_drop": max(drop.values()), "worst_axis": max(drop, key=drop.get), "new_question": new_q, "train_loss_last_epoch": tl}
                out["conditions"].append(c)
                print(f"  {case:9s} k {k:3d} {regime:9s}: trained {tt:6.1f}s alone on {len(ex) * ep} examples; new question {new_q:.3f}; X3 {after['x3_depth3_not_given']:.3f}, six exact {after['exact_six_depth3_not_given']:.3f}, T4 cell {after['t4']['slot']:.3f}; worst drop {c['worst_drop']:+.3f} ({c['worst_axis']})", flush=True)
    json.dump(out, open(os.path.join(mainrepo.ROOT, "logs", f"results_growth_{a.tag}_seed{seed}.json"), "w"))
    metrics.checklist(1, "[x] k examples frozen before any model [x] one declared recipe per regime plus one alternative lr [x] wall-clock alone [x] " + label)
