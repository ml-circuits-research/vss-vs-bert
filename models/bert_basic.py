"""bert_basic.py (runs 2 and 3): a pretrained encoder (or the same architecture from scratch) fine-tuned to read the basic world's frames
with a token-tagging head (INSTRUCTIONS section 3), trained on the same 30k sentences as the reader and the 7b transformer (generator
rng 200+seed, depths 0-2, training templates and identities), tuned on the written grid on seed 0 only (rule 25), 5 seeds, decoded on
the FROZEN splits with the same restricted candidates as the world decoder: for a role r, the filler is the candidate token with the
highest probability of tag r; the margin is the top1 - top2 probability gap over the candidates.

Tokenizer conditions (section 4): 'wordpiece' keeps the pretrained tokenizer (invented words are split into pieces; pieces per word
reported); 'whole' adds the world's vocabulary as whole tokens with randomly initialised embeddings. The depth of a sentence is given at
decoding (the roles read are those of its depth), as it is for the world decoder.
Run: python3 models/bert_basic.py --model bert-base-uncased [--scratch] [--tokenizer wordpiece|whole] [--seeds 5] [--N 30000] [--device cuda]
Grid (fixed, section 4): pretrained lr {2e-5,3e-5,5e-5} x epochs {3,5,10}; scratch lr {1e-4,3e-4,1e-3} x epochs {10,20,40}; batch 32; wd 0.01; warm-up 10%."""
import argparse, os, sys, time, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from common import basic_world as bwm, report
from eval import metrics

LABELS = ["O"] + bwm.ROLES; LAB = {l: i for i, l in enumerate(LABELS)}

class Tagger(nn.Module):
    def __init__(self, name, scratch, n_tokens):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_config(cfg) if scratch else AutoModel.from_pretrained(name)
        if n_tokens > self.enc.get_input_embeddings().num_embeddings: self.enc.resize_token_embeddings(n_tokens)
        self.head = nn.Linear(cfg.hidden_size, len(LABELS)); self.drop = nn.Dropout(0.1)
    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(self.drop(h))

def encode_batch(tok, sents, device, max_len=96):
    """word-level inputs: first sub-token of every word carries the label; returns ids, attention, first-subtoken positions per word"""
    enc = tok([" ".join(s) for s in sents], is_split_into_words=False, return_tensors="pt", padding=True, truncation=True, max_length=max_len, return_offsets_mapping=True)
    firsts = []
    for b, s in enumerate(sents):
        offs = enc["offset_mapping"][b].tolist(); text = " ".join(s); starts = []; pos = 0
        for w_ in s: starts.append(pos); pos += len(w_) + 1
        f = []; j = 0
        for st in starts:
            while j < len(offs) and not (offs[j][0] == st and offs[j][1] > offs[j][0]): j += 1
            f.append(j if j < len(offs) else 0)
        firsts.append(f)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), firsts

def make_train(bw, seed, N):
    """the 30k training sentences, as make_data of v9 / v22 draws them (rng 200+seed): (tokens, tags)"""
    w = bw["w"]; rng = np.random.default_rng(200 + seed); saved = w.rng; w.rng = rng; out = []
    try:
        for _ in range(N):
            d = int(rng.integers(bwm.DEPTH_TRAIN + 1)); ev = w.event(bw["trP"], bw["trO"], depth=d); toks = w.render(ev, rng.choice(list(v5.TRAIN_T)))
            out.append((toks, bwm.tags_of(toks, ev)))
    finally: w.rng = saved
    return out

def train(model, tok, data, lr, epochs, batch, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(data) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(data), batch):
            chunk = [data[j] for j in perm[i:i + batch]]; ids, att, firsts = encode_batch(tok, [c[0] for c in chunk], device)
            lab = torch.full(ids.shape, -100, dtype=torch.long)
            for b, (toks, tags) in enumerate(chunk):
                for k, t in enumerate(tags): lab[b, firsts[b][k]] = LAB[t]
            logits = model(ids, att); loss = ce(logits.view(-1, len(LABELS)), lab.view(-1).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def predict(model, tok, split, axis, device, batch=128):
    """frames with restricted candidates and probability-gap margins; also the 'free' frame (argmax tag over all tokens) for the occupancy row"""
    model.eval(); items = split["axes"][axis]; preds = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = encode_batch(tok, [it["tokens"] for it in chunk], device)
        prob = torch.softmax(model(ids, att), -1).cpu().numpy()
        for b, it in enumerate(chunk):
            p = prob[b][firsts[b]]                        # words x labels
            out = {"frame": {r: None for r in bwm.ROLES}, "margins": {}, "free_frame": {r: None for r in bwm.ROLES}}
            for r in bwm.roles_at(it["depth"]):
                cands = bwm.candidates_for(split, axis, r); pos = [k for k, t in enumerate(it["tokens"]) if t in cands]
                sc = np.array([p[k, LAB[r]] for k in pos]) if pos else np.zeros(0)
                if len(sc) == 0: out["margins"][r] = 0.0; continue
                top = np.argsort(-sc); out["frame"][r] = it["tokens"][pos[top[0]]]; out["margins"][r] = float(sc[top[0]] - (sc[top[1]] if len(sc) > 1 else 0.0))
            tags = [LABELS[j] for j in p.argmax(1)]; out["free_frame"] = bwm.frame_from_tags(it["tokens"], tags)
            preds.append(out)
    return preds

def occupancy(items, preds):
    """the tagger's own decision on which roles are present (free argmax tags, no candidates): absent roles falsely filled, present roles left untagged"""
    fa = fm = na = nm = 0
    for it, p in zip(items, preds):
        for r in bwm.ROLES:
            g = it["frame"].get(r); f = p["free_frame"].get(r)
            if g is None: na += 1; fa += (f is not None)
            else: nm += 1; fm += (f is None)
    return {"false_fill_on_absent": fa / max(na, 1), "missed_present": fm / max(nm, 1)}

def model_bytes(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--tokenizer", default="wordpiece", choices=["wordpiece", "whole"])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lrs", default=None); ap.add_argument("--epochs", default=None); ap.add_argument("--cost_n", type=int, default=200)
    a = ap.parse_args()
    lrs = [float(x) for x in (a.lrs or ("1e-4,3e-4,1e-3" if a.scratch else "2e-5,3e-5,5e-5")).split(",")]
    eps = [int(x) for x in (a.epochs or ("10,20,40" if a.scratch else "3,5,10")).split(",")]
    name = f"{a.model}{' from scratch' if a.scratch else ' pretrained'} [{a.tokenizer}]"; tag = f"bert_{a.model.replace('/', '_')}_{'scratch' if a.scratch else 'pre'}_{a.tokenizer}"
    print(f"bert_basic: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; N {a.N}; seeds {a.seeds}; grid lr {lrs} x epochs {eps}; batch {a.batch}")
    print("regime (rule 24): basic world, prefixed forms, training depths 0-2, tested at depth 3 UNSEEN; token-tagging head; decoding restricted to the world's candidates; frozen splits")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167"); chosen = None; rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, a.D); torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(a.model)
        if a.tokenizer == "whole": tok.add_tokens(list(split["vocab"]))
        pieces = np.mean([len(tok.tokenize(w_)) for w_ in split["vocab"]]); print(f"  seed {seed}: tokenizer '{a.tokenizer}', pieces per world word {pieces:.2f}")
        data = make_train(bw, seed, a.N); val = [(t, g) for t, g in make_train(bw, 900 + seed, 2000)]
        val_items = [{"tokens": t, "frame": bwm.frame_from_tags(t, g), "depth": sum(x.startswith("X") for x in g)} for t, g in val]
        val_split = {"axes": {"val": val_items}, "candidates": split["candidates"]}; best = None
        for lr in (lrs if seed == 0 else [chosen[0]]):
            for ep in (eps if seed == 0 else [chosen[1]]):
                model = Tagger(a.model, a.scratch, len(tok)).to(a.device); t0 = time.time(); train(model, tok, data, lr, ep, a.batch, a.device, seed)
                vp = predict(model, tok, val_split, "val", a.device); vfull = np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.roles_at(it["depth"])) for it, p in zip(val_items, vp)])
                print(f"  seed {seed} lr={lr:g} epochs={ep} ({time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params) validation FULL {vfull:.3f}", flush=True)
                if best is None or vfull > best[0]: best = (vfull, lr, ep, model)
                else: del model; torch.cuda.empty_cache() if a.device == "cuda" else None
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, epochs={chosen[1]} (by validation FULL frame, seed 0)")
        model = best[3]; preds = {axis: predict(model, tok, split, axis, a.device) for axis in report.AXES + ["calibration"]}
        row = report.evaluate(name, seed, preds, split, tag); row["occupancy"] = {ax: occupancy(split["axes"][ax], preds[ax]) for ax in ("seen", "new_ids", "depth3_unseen")}
        # cost: bytes and CPU milliseconds per sentence (the stated machine is the one printed by run.sh)
        cpu = model.to("cpu"); sents = split["axes"]["seen"][:a.cost_n]; t0 = time.time()
        for it in sents: predict(cpu, tok, {"axes": {"x": [it]}, "candidates": split["candidates"]}, "x", "cpu", batch=1)
        ms = 1000 * (time.time() - t0) / len(sents); row["cost"] = {"params": sum(p.numel() for p in model.parameters()), "bytes": model_bytes(model), "cpu_ms_per_sentence": ms}
        print(f"  seed {seed} occupancy (free tags) {row['occupancy']}; cost {row['cost']['params']/1e6:.1f}M params, {row['cost']['bytes']/1e6:.0f} MB, {ms:.1f} ms/sentence on CPU")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        rows.append(row); model.to(a.device); del model
    report.summarize(rows, name)
    print("written interpretation (INSTRUCTIONS table 3): new ids > 0.9 => the pieces carry the role; < 0.5 => the vocabulary gap is structural; depth 3 > 0.9 would contradict the paper's same-input claim, < 0.5 confirms it; in between reported with both readings")
    metrics.checklist(a.seeds, "[x] tokenizer condition stated [x] grid fixed in advance")
