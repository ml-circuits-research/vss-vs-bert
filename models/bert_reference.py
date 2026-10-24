"""bert_reference.py (run 4, taggers): the encoders of models/bert_basic.py on the REFERENCE world (common/reference_world.py): the same
token-tagging head for A, B, C, X1..X3 plus a sentence-level classification head on the [CLS] vector for the predicate (the surface
verb's lemma, 12 classes; the event type follows by the synonym table). Trained on 30k sentences from v16's training distribution (rng
200+seed, depths 0-2, training templates and identities, recipient-less forms at share 0.2), the same fixed grid as bert_basic, tuned on
seed 0 only (rule 25), decoded on the FROZEN splits with the world's candidates.

Recipient presence (INSTRUCTIONS section 3: 'a tagger says not given by not tagging'): C is present iff the largest probability of tag C
over the candidate tokens is >= 0.5; its margin is min(identity gap, |p - 0.5|) when present, |p - 0.5| when absent.
Run: python3 models/bert_reference.py --model bert-base-uncased [--scratch] [--tokenizer wordpiece|whole] [--seeds 5] [--N 30000] [--device cuda]"""
import argparse, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from common import reference_world as rw, report_ref
from models.bert_basic import Tagger, encode_batch, model_bytes, LABELS, LAB
from eval import metrics

class RefTagger(Tagger):
    def __init__(self, name, scratch, n_tokens, n_pred):
        super().__init__(name, scratch, n_tokens); self.pred = nn.Linear(self.head.in_features, n_pred)
    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state; h = self.drop(h)
        return self.head(h), self.pred(h[:, 0])

def make_train(bw, seed, N, p_omit=rw.P_OMIT):
    v16 = rw.import_v16()[2]; w = bw["w"]; rng = np.random.default_rng(200 + seed); saved = w.rng; w.rng = rng; out = []
    try:
        for _ in range(N):
            d = int(rng.integers(rw.DEPTH_TRAIN + 1)); ev = w.event(bw["trP"], bw["trO"], depth=d)
            if rng.random() < p_omit: toks = w.render_omit(ev, int(rng.integers(len(v16.OMIT_C)))); has_C = False
            else: toks = w.render(ev, int(rng.choice(list(v5.TRAIN_T)))); has_C = True
            fr = rw.frame_of(ev, toks, has_C, bw); out.append((toks, rw.tags_of(toks, fr), fr["pred"]))
    finally: w.rng = saved
    return out

def train(model, tok, data, verbs, lr, epochs, batch, device, seed):
    vid = {v: i for i, v in enumerate(verbs)}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(data) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(data), batch):
            chunk = [data[j] for j in perm[i:i + batch]]; ids, att, firsts = encode_batch(tok, [c[0] for c in chunk], device)
            lab = torch.full(ids.shape, -100, dtype=torch.long)
            for b, (toks, tags, _) in enumerate(chunk):
                for k, t in enumerate(tags): lab[b, firsts[b][k]] = LAB[t]
            logits, pl = model(ids, att)
            loss = ce(logits.view(-1, len(LABELS)), lab.view(-1).to(device)) + ce(pl, torch.tensor([vid[c[2]] for c in chunk], device=device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def predict(model, tok, split, axis, verbs, device, batch=128):
    model.eval(); items = split["axes"][axis]; preds = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = encode_batch(tok, [it["tokens"] for it in chunk], device)
        logits, pl = model(ids, att); prob = torch.softmax(logits, -1).cpu().numpy(); pp = torch.softmax(pl, -1).cpu().numpy()
        for b, it in enumerate(chunk):
            p = prob[b][firsts[b]]; fr = {r: None for r in rw.ROLES}; mg = {}
            for r in [x for x in rw.roles_at(it["depth"]) if x != "pred"]:
                cands = rw.candidates_for(split, axis, r); pos = [k for k, t in enumerate(it["tokens"]) if t in cands]
                sc = np.array([p[k, LAB[r]] for k in pos]) if pos else np.zeros(0)
                if len(sc) == 0: mg[r] = 0.0; continue
                top = np.argsort(-sc); gap = float(sc[top[0]] - (sc[top[1]] if len(sc) > 1 else 0.0))
                if r == "C":
                    if sc[top[0]] >= 0.5: fr["C"] = it["tokens"][pos[top[0]]]; mg["C"] = min(gap, abs(float(sc[top[0]]) - 0.5))
                    else: mg["C"] = abs(float(sc[top[0]]) - 0.5)
                else: fr[r] = it["tokens"][pos[top[0]]]; mg[r] = gap
            tp = np.argsort(-pp[b]); fr["pred"] = verbs[tp[0]]; mg["pred"] = float(pp[b][tp[0]] - pp[b][tp[1]])
            preds.append({"frame": fr, "margins": mg})
    return preds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--tokenizer", default="wordpiece", choices=["wordpiece", "whole"])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lrs", default=None); ap.add_argument("--epochs", default=None); ap.add_argument("--cost_n", type=int, default=200)
    a = ap.parse_args()
    lrs = [float(x) for x in (a.lrs or ("1e-4,3e-4,1e-3" if a.scratch else "2e-5,3e-5,5e-5")).split(",")]
    eps = [int(x) for x in (a.epochs or ("10,20,40" if a.scratch else "3,5,10")).split(",")]
    name = f"{a.model}{' from scratch' if a.scratch else ' pretrained'} [{a.tokenizer}]"; tag = f"bert_{a.model.replace('/', '_')}_{'scratch' if a.scratch else 'pre'}_{a.tokenizer}"
    print(f"bert_reference: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; N {a.N}; seeds {a.seeds}; grid lr {lrs} x epochs {eps}; batch {a.batch}")
    print("regime (rule 24): reference world (v16, 10b-quinquies), training depths 0-2, templates TRAIN_T, recipient-less share 0.2; depth 3 UNSEEN; token tags + predicate head; frozen splits")
    chosen = None; rows = []
    for seed in range(a.seeds):
        split = rw.load(seed); bw = rw.build(seed, a.D); verbs = split["candidates"]["verbs"]; torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(a.model)
        if a.tokenizer == "whole": tok.add_tokens(list(split["vocab"]))
        print(f"  seed {seed}: tokenizer '{a.tokenizer}', pieces per world word {np.mean([len(tok.tokenize(x)) for x in split['vocab']]):.2f}")
        data = make_train(bw, seed, a.N); val = make_train(bw, 900 + seed, 2000)
        val_split = {"axes": {"val": [{"tokens": t, "depth": sum(x.startswith("X") for x in g), "frame": rw.frame_from_tags(t, g, p)} for t, g, p in val]}, "candidates": split["candidates"]}
        best = None
        for lr in (lrs if seed == 0 else [chosen[0]]):
            for ep in (eps if seed == 0 else [chosen[1]]):
                model = RefTagger(a.model, a.scratch, len(tok), len(verbs)).to(a.device); t0 = time.time(); train(model, tok, data, verbs, lr, ep, a.batch, a.device, seed)
                vp = predict(model, tok, val_split, "val", verbs, a.device)
                vfull = np.mean([all(p["frame"].get(r) == it["frame"].get(r) for r in rw.roles_at(it["depth"])) for it, p in zip(val_split["axes"]["val"], vp)])
                print(f"  seed {seed} lr={lr:g} epochs={ep} ({time.time() - t0:.0f}s) validation FULL {vfull:.3f}", flush=True)
                if best is None or vfull > best[0]: best = (vfull, lr, ep, model)
                else: del model
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, epochs={chosen[1]} (by validation FULL frame, seed 0)")
        model = best[3]; preds = {axis: predict(model, tok, split, axis, verbs, a.device) for axis in rw.AXES + ["calibration"]}
        row = report_ref.evaluate(name, seed, preds, split, tag)
        cpu = model.to("cpu"); sents = split["axes"]["seen"][:a.cost_n]; t0 = time.time()
        for it in sents: predict(cpu, tok, {"axes": {"x": [it]}, "candidates": split["candidates"]}, "x", verbs, "cpu", batch=1)
        row["cost"] = {"params": sum(p.numel() for p in model.parameters()), "bytes": model_bytes(model), "cpu_ms_per_sentence": 1000 * (time.time() - t0) / len(sents)}
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_ref_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        rows.append(row); del model
    report_ref.summarize(rows, name)
    metrics.checklist(a.seeds, "[x] tokenizer condition stated [x] grid fixed in advance [x] predicate role present (23)")
