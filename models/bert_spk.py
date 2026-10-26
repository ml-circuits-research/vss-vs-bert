"""bert_spk.py (responses R68 (A), declared there 2026-10-26 07:55; RESULTS 'R68 (A)'): the depth-invariant tagger. The tagger form of
2a (models/bert_basic.py: its Tagger, word-level first-sub-token labels, make_train's 30,000 sentences, AdamW with a 10% linear warm-up,
gradient clipping 1.0) with ONE speaker label (`--speaker_label single`, the only setting of this script; models/bert_basic.py itself is
unchanged): X1 / X2 / X3 collapse to SPK in training (labels O, A, B, C, SPK). Decoding with the depth NOT given: A, B, C = the
candidate of the axis's lists with the highest probability of that label (margin = top-1 minus top-2 probability); the speakers = the
words whose argmax label is SPK, in their order of appearance (the first is level 1, the second level 2, ...), restricted to the axis's
person candidates, so the level count is the number of SPK words and level 3 is reachable without an untrained label; a speaker's
margin = its SPK probability minus its next-best label probability. Fixed configuration (R68): BERT-tiny pretrained
(google/bert_uncased_L-2_H-128_A-2), lr 1e-4, 5 epochs, batch 32, seeds 5-9 (the fresh splits). Reported per axis: slot accuracy over
the gold roles of the item's depth (a missing level counts wrong), six-question exact match (all six roles, absent ones must be None),
inner frame; at depth 3: X3 slot accuracy, exact match, inner frame, the level count's accuracy, confident error (a wrong decision at or
above tau, over all decisions; tau by eval/metrics.set_tau on the calibration rows, rule 18); the same on new formulations.
Written by R68 before the run: X3 >= 0.9 -> a dedicated output head answers this particular composed question (the claim stays the
general one); X3 < 0.5 -> the example also stands against this head. Writes logs/results_bertspk_<tag>_seed<s>.json.
Run: ./run.sh models/bert_spk.py --speaker_label single [--seed_start 5] [--seeds 5]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from common import basic_world as bwm, report
from eval import metrics
import models.bert_basic as bb

LABELS = ["O", "A", "B", "C", "SPK"]; LAB = {l: i for i, l in enumerate(LABELS)}
def spk(t): return "SPK" if t.startswith("X") else t

def train(model, tok, data, lr, epochs, batch, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(data) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss(ignore_index=-100); last = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(data), batch):
            chunk = [data[j] for j in perm[i:i + batch]]; ids, att, firsts = bb.encode_batch(tok, [c[0] for c in chunk], device)
            lab = torch.full(ids.shape, -100, dtype=torch.long)
            for b, (toks, tags) in enumerate(chunk):
                for k, t in enumerate(tags):
                    if firsts[b][k] is not None: lab[b, firsts[b][k]] = LAB[spk(t)]
            logits = model(ids, att); loss = ce(logits.view(-1, len(LABELS)), lab.view(-1).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if ep == epochs - 1: last.append(float(loss))
        print(f"    [progress] epoch {ep + 1}/{epochs}", flush=True)
    return float(np.mean(last))

@torch.no_grad()
def predict(model, tok, split, axis, items, device, batch=128):
    model.eval(); preds = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = bb.encode_batch(tok, [it["tokens"] for it in chunk], device)
        prob = torch.softmax(model(ids, att), -1).cpu().numpy()
        for b, it in enumerate(chunk):
            p = np.stack([prob[b][j] if j is not None else np.zeros(len(LABELS)) for j in firsts[b]]); f = {r: None for r in bwm.ROLES}; mg = {}
            for r in ("A", "B", "C"):
                cands = bwm.candidates_for(split, axis, r); pos = [k for k, t in enumerate(it["tokens"]) if t in cands]
                if not pos: mg[r] = 0.0; continue
                sc = np.array([p[k, LAB[r]] for k in pos]); o = np.argsort(-sc); f[r] = it["tokens"][pos[o[0]]]; mg[r] = float(sc[o[0]] - (sc[o[1]] if len(sc) > 1 else 0.0))
            pc = set(bwm.candidates_for(split, axis, "X1")); sp = [k for k, t in enumerate(it["tokens"]) if t in pc and p[k].argmax() == LAB["SPK"]]
            for lvl, k in enumerate(sp[:3]):
                srt = np.sort(p[k])[::-1]; f[f"X{lvl + 1}"] = it["tokens"][k]; mg[f"X{lvl + 1}"] = float(srt[0] - srt[1])
            preds.append({"frame": f, "margins": mg, "n_spk": len(sp)})
    return preds

def decisions(items, preds):
    """(correct, margin) for every role of the gold depth AND every role the model emitted beyond it (a false speaker is a wrong decision)"""
    out = []
    for it, p in zip(items, preds):
        for r in bwm.ROLES:
            g = it["frame"][r]; v = p["frame"][r]
            if g is None and v is None: continue
            out.append((None, v == g, p["margins"].get(r, 0.0)))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--speaker_label", required=True, choices=["single"]); ap.add_argument("--model", default="google/bert_uncased_L-2_H-128_A-2")
    ap.add_argument("--seed_start", type=int, default=5); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    tag = f"{a.model.replace('/', '_')}_spk"
    print(f"bert_spk (R68 A): main repository {mainrepo.check_pinned()}; {a.model} pretrained, tagger with ONE speaker label (labels {LABELS}); lr {a.lr}, {a.epochs} epochs, batch {a.batch}; seeds {a.seed_start}-{a.seed_start + a.seeds - 1}; depth NOT given (levels = SPK words in order)")
    print("written before the run (R68): X3 >= 0.9 -> a dedicated head answers this composed question; X3 < 0.5 -> the example stands against this head too")
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, split["D"]); tok = AutoTokenizer.from_pretrained(a.model)
        torch.manual_seed(seed); model = bb.Tagger(a.model, False, len(tok)); model.head = nn.Linear(model.head.in_features, len(LABELS)); model.to(a.device)
        data = bb.make_train(bw, seed, a.N); t0 = time.time(); tl = train(model, tok, data, a.lr, a.epochs, a.batch, a.device, seed); tt = time.time() - t0
        cal = split["axes"]["calibration"]; tau = metrics.set_tau(decisions(cal, predict(model, tok, split, "calibration", cal, a.device)))
        row = {"seed": seed, "model": a.model, "tau": tau, "train_s": tt, "train_loss_last_epoch": tl, "axes": {}}
        for ax in report.AXES:
            items = split["axes"][ax]
            if not items: continue
            P = predict(model, tok, split, ax, items, a.device)
            slot = float(np.mean([p["frame"][r] == it["frame"][r] for it, p in zip(items, P) for r in bwm.roles_at(it["depth"])]))
            six = float(np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.ROLES) for it, p in zip(items, P)]))
            inner = float(np.mean([all(p["frame"][r] == it["frame"][r] for r in ("A", "B", "C")) for it, p in zip(items, P)]))
            lvl = float(np.mean([min(p["n_spk"], 3) == it["depth"] for it, p in zip(items, P)]))
            dec = decisions(items, P); cw = float(np.mean([(not c) and m >= tau for _, c, m in dec])) if dec else 0.0
            r_ = {"slot": slot, "exact_six": six, "inner": inner, "level_count_acc": lvl, "confident_error": cw, "per_role": {r: float(np.mean([p["frame"][r] == it["frame"][r] for it, p in zip(items, P)])) for r in bwm.roles_at(items[0]["depth"])}}
            row["axes"][ax] = r_
            print(f"  seed {seed} {ax:24s} slot {slot:.3f}; exact six {six:.3f}; inner {inner:.3f}; level count {lvl:.3f}; confident error {cw:.3f}; per role {{{', '.join(f'{k}: {v:.3f}' for k, v in r_['per_role'].items())}}}", flush=True)
        print(f"  seed {seed}: trained in {tt:.0f}s (concurrent: not a cost number); tau {tau:.4f}; last-epoch loss {tl:.2e}", flush=True)
        json.dump(row, open(os.path.join(mainrepo.ROOT, "logs", f"results_bertspk_{tag}_seed{seed}.json"), "w")); rows.append(row); del model; torch.cuda.empty_cache()
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== bert_spk, {a.model}, seeds {[r['seed'] for r in rows]}: MEANS (per-seed list) ==")
    for ax in report.AXES:
        if ax not in rows[0]["axes"]: continue
        print(f"  {ax:24s} slot {ms([r['axes'][ax]['slot'] for r in rows])}; exact six {ms([r['axes'][ax]['exact_six'] for r in rows])}; inner {ms([r['axes'][ax]['inner'] for r in rows])}; confident error {ms([r['axes'][ax]['confident_error'] for r in rows])}")
    x3 = [r["axes"]["depth3_unseen"]["per_role"]["X3"] for r in rows]
    print(f"  depth 3: X3 slot accuracy {ms(x3)}; level count {ms([r['axes']['depth3_unseen']['level_count_acc'] for r in rows])}")
    print(f"  R68's reading: X3 mean {np.mean(x3):.3f} -> " + ("a dedicated head answers this composed question (>= 0.9)" if np.mean(x3) >= 0.9 else "the example also stands against this head (< 0.5)" if np.mean(x3) < 0.5 else "between 0.5 and 0.9: both readings written"))
    metrics.checklist(len(rows), "[x] fixed configuration (R68) [x] depth not given [x] tau on calibration rows (rule 18) [x] fresh seeds 5-9")
