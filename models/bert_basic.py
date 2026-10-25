"""bert_basic.py (runs 2 and 3): a pretrained encoder (or the same architecture from scratch) fine-tuned to read the basic world's frames
with a token-tagging head (INSTRUCTIONS section 3), trained on the same 30k sentences as the reader and the 7b transformer (generator
rng 200+seed, depths 0-2, training templates and identities), tuned on the written grid on seed 0 only (rule 25), 5 seeds, decoded on
the FROZEN splits with the same restricted candidates as the world decoder: for a role r, the filler is the candidate token with the
highest probability of tag r; the margin is the top1 - top2 probability gap over the candidates.

Tokenizer conditions (section 4): 'wordpiece' keeps the pretrained tokenizer (invented words are split into pieces; pieces per word
reported); 'whole' adds the world's vocabulary as whole tokens with randomly initialised embeddings. The depth of a sentence is given at
decoding (the roles read are those of its depth), as it is for the world decoder.
Run: python3 models/bert_basic.py --model bert-base-uncased [--scratch] [--tokenizer wordpiece|whole] [--seeds 5] [--N 30000] [--device cuda]
Grid (fixed, section 4): pretrained lr {2e-5,3e-5,5e-5} x epochs {3,5,10}; scratch lr {1e-4,3e-4,1e-3} x epochs {10,20,40}; batch 32; wd 0.01; warm-up 10%.
R10 (responses R10, 2026-10-25; deviations after the independent review, RESULTS): flags whose defaults reproduce the pre-registered code
(tag prereg-2026-10-25): --r10 (= --select r10 --wd_groups --seed_per_config --trunc_none --temperature), --pos_offset K (random position
offsets 0..K in training), --whole_init pieces. Always on and additive (no pre-registered number changes): per-candidate scores for the
one-to-one rule, the validation cross-entropy of every grid point, the depth-3 split by filler position (first-sub-token index), the
occupancy over the trained labels, the cost at fp16 and int8 dynamic quantization and the CPU thread count.
R14 (responses R14, 2026-10-25): --cues given adds ONE learned 'cue' embedding (hidden size, init N(0, 0.02)) to the input embedding of
every token that is a whole cue word of CUE_WORDS (the same set the reader's gates are given for); default 'none' = the tagged code.
R17 (responses R17 / R18, 2026-10-25), default off: --save_all saves every seed's model (for eval/r17_tagger.py); --oow_class trains a
[CLS] out-of-world head on the 'neg' part of data/oow_seed<s>.json (their tokens tagged O); --depth_head cls|reg trains a [CLS] depth
head (classes 0-3, class 3 never in training; or one regression output, rounded and clipped to 0-3 at decoding)."""
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
CUE_WORDS = ["to", "by", "from", ".", "says", "that"]   # R14: the role-cue and level-opener words, given from the legend (rule 20)

class Tagger(nn.Module):
    def __init__(self, name, scratch, n_tokens, piece_init=None, cue_tok=None, oow=False, depth_head="none"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_config(cfg) if scratch else AutoModel.from_pretrained(name)
        if n_tokens > self.enc.get_input_embeddings().num_embeddings:
            n0 = self.enc.get_input_embeddings().num_embeddings; self.enc.resize_token_embeddings(n_tokens)   # transformers 5.20.2: mean_resizing=True
            if piece_init is not None:                     # R10 item 4: each new whole-word token = the mean of its WordPiece pieces' embeddings
                E = self.enc.get_input_embeddings().weight
                with torch.no_grad():
                    for new_id, pieces in piece_init.items(): E[new_id] = E[pieces].mean(0)
        self.head = nn.Linear(cfg.hidden_size, len(LABELS)); self.drop = nn.Dropout(0.1); self.cue = None; self.last_cls = None
        self.oow_head = nn.Linear(cfg.hidden_size, 2) if oow else None                     # R17 item 1 (ii)
        self.depth_kind = depth_head; self.depth_head = {"cls": lambda: nn.Linear(cfg.hidden_size, 4), "reg": lambda: nn.Linear(cfg.hidden_size, 1)}.get(depth_head, lambda: None)()
        if cue_tok is not None:                        # R14: a token is a cue iff it is a cue word's own (single) token and the next token does not continue it
            vocab = cue_tok.get_vocab(); cont = torch.zeros(max(n_tokens, len(vocab)), dtype=torch.bool)
            for t, i in vocab.items():
                if t.startswith("##"): cont[i] = True
            ids_ = [vocab[w_] for w_ in CUE_WORDS if w_ in vocab]; assert len(ids_) == len(CUE_WORDS), "every cue word must be one token"
            self.register_buffer("cue_ids", torch.tensor(ids_)); self.register_buffer("cont", cont)
            self.cue = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
    def forward(self, ids, att, position_ids=None):
        kw = {"position_ids": position_ids} if position_ids is not None else {}
        if self.cue is None: h = self.enc(input_ids=ids, attention_mask=att, **kw).last_hidden_state
        else:
            nxt = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], 1)
            m = torch.isin(ids, self.cue_ids) & ~self.cont[nxt] & att.bool()
            emb = self.enc.get_input_embeddings()(ids) + m.unsqueeze(-1).to(self.cue.dtype) * self.cue
            h = self.enc(inputs_embeds=emb, attention_mask=att, **kw).last_hidden_state
        self.last_cls = h[:, 0]
        return self.head(self.drop(h))
    def depth_of(self, cls):
        """R17 item 3: the depth decided by the [CLS] head (argmax of 0-3, or the regression rounded and clipped)"""
        o = self.depth_head(cls)
        return o.argmax(-1) if self.depth_kind == "cls" else o.squeeze(-1).round().clamp(0, 3).long()

def encode_batch(tok, sents, device, max_len=96, trunc_none=False):
    """word-level inputs: first sub-token of every word carries the label; returns ids, attention, first-subtoken positions per word"""
    enc = tok([" ".join(s) for s in sents], is_split_into_words=False, return_tensors="pt", padding=True, truncation=True, max_length=max_len, return_offsets_mapping=True)
    firsts = []
    for b, s in enumerate(sents):
        offs = enc["offset_mapping"][b].tolist(); text = " ".join(s); starts = []; pos = 0
        for w_ in s: starts.append(pos); pos += len(w_) + 1
        f = []; j = 0
        for st in starts:
            while j < len(offs) and not (offs[j][0] == st and offs[j][1] > offs[j][0]): j += 1
            f.append(j if j < len(offs) else (None if trunc_none else 0))   # R10 item 15: None (no label, no prediction) for a truncated word
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

def param_groups(model, wd):
    """R10 item 5: no weight decay on biases and LayerNorm weights (the standard BERT recipe)"""
    dec, nodec = [], []
    for n, p in model.named_parameters(): (nodec if n.endswith("bias") or "norm" in n.lower() else dec).append(p)
    return [{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}]

def train(model, tok, data, lr, epochs, batch, device, seed, wd_groups=False, pos_offset=0, trunc_none=False, neg=None):
    """data: (tokens, tags); R17: neg = out-of-world sentences (tagged O, out-of-world label 1) mixed into the epochs when the model
    has an out-of-world head; the depth head is trained on the in-world sentences (depth = the number of X tags)"""
    n_in = len(data)
    if neg: data = list(data) + [(t, ["O"] * len(t)) for t in neg]
    opt = torch.optim.AdamW(param_groups(model, 0.01) if wd_groups else model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(data) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(data), batch):
            chunk = [data[j] for j in perm[i:i + batch]]; ids, att, firsts = encode_batch(tok, [c[0] for c in chunk], device, trunc_none=trunc_none)
            lab = torch.full(ids.shape, -100, dtype=torch.long)
            for b, (toks, tags) in enumerate(chunk):
                for k, t in enumerate(tags):
                    if firsts[b][k] is not None: lab[b, firsts[b][k]] = LAB[t]
            pid = None
            if pos_offset:                                  # R10 item 2b: a random position offset 0..K per sentence in training only
                off = torch.randint(0, pos_offset + 1, (ids.shape[0], 1), generator=g); pid = (torch.arange(ids.shape[1]).unsqueeze(0) + off).to(device)
            logits = model(ids, att, pid); loss = ce(logits.view(-1, len(LABELS)), lab.view(-1).to(device))
            if model.oow_head is not None:
                y = torch.tensor([int(j >= n_in) for j in perm[i:i + batch]], device=device); loss = loss + ce(model.oow_head(model.drop(model.last_cls)), y)
            if model.depth_head is not None:
                keep = torch.tensor([j < n_in for j in perm[i:i + batch]], device=device)
                d = torch.tensor([sum(x.startswith("X") for x in c[1]) for c in chunk], device=device)
                if keep.any():
                    o = model.depth_head(model.drop(model.last_cls[keep]))
                    loss = loss + (ce(o, d[keep]) if model.depth_kind == "cls" else nn.functional.mse_loss(o.squeeze(-1), d[keep].float()))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def predict(model, tok, split, axis, device, batch=128, T=1.0, trunc_none=False):
    """frames with restricted candidates and probability-gap margins; also the 'free' frame (argmax tag over all tokens) for the occupancy row.
    R10: per-candidate log-probabilities (cand_scores) for the one-to-one rule; T = temperature (R10 item 6; 1.0 = the pre-registered softmax)"""
    model.eval(); items = split["axes"][axis]; preds = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = encode_batch(tok, [it["tokens"] for it in chunk], device, trunc_none=trunc_none)
        prob = torch.softmax(model(ids, att) / T, -1).cpu().numpy()
        for b, it in enumerate(chunk):
            p = np.stack([prob[b][j] if j is not None else np.zeros(len(LABELS)) for j in firsts[b]])   # words x labels
            out = {"frame": {r: None for r in bwm.ROLES}, "margins": {}, "free_frame": {r: None for r in bwm.ROLES}, "score_floor": float(np.log(1e-30))}
            for r in bwm.roles_at(it["depth"]):
                cands = bwm.candidates_for(split, axis, r); pos = [k for k, t in enumerate(it["tokens"]) if t in cands]
                sc = np.array([p[k, LAB[r]] for k in pos]) if pos else np.zeros(0)
                if len(sc) == 0: out["margins"][r] = 0.0; continue
                top = np.argsort(-sc); out["frame"][r] = it["tokens"][pos[top[0]]]; out["margins"][r] = float(sc[top[0]] - (sc[top[1]] if len(sc) > 1 else 0.0))
                out.setdefault("cand_scores", {})[r] = ([it["tokens"][k] for k in pos], np.log(np.maximum(sc, 1e-30)).tolist())
            tags = [LABELS[j] for j in p.argmax(1)]; out["free_frame"] = bwm.frame_from_tags(it["tokens"], tags)
            preds.append(out)
    return preds

def occupancy(items, preds):
    """the tagger's own decision on which roles are present (free argmax tags, no candidates): absent roles falsely filled, present roles left
    untagged; R10 item 12: also the missed rate over the trained labels only (X3 never occurs in training, so it is missed by construction)"""
    fa = fm = na = nm = fmt = nmt = 0
    for it, p in zip(items, preds):
        for r in bwm.ROLES:
            g = it["frame"].get(r); f = p["free_frame"].get(r)
            if g is None: na += 1; fa += (f is not None)
            else:
                nm += 1; fm += (f is None)
                if r in report.TRAINED_D3: nmt += 1; fmt += (f is None)
    return {"false_fill_on_absent": fa / max(na, 1), "missed_present": fm / max(nm, 1), "missed_present_trained_labels": fmt / max(nmt, 1)}

def gold_labels(it):
    inv = {v: r for r, v in it["frame"].items() if v is not None}; return [inv.get(t, "O") for t in it["tokens"]]

@torch.no_grad()
def logits_and_labels(model, tok, items, device, batch=128, trunc_none=False):
    """first-sub-token logits and gold label ids of a list of items (validation cross-entropy, temperature fitting)"""
    model.eval(); L, Y = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = encode_batch(tok, [it["tokens"] for it in chunk], device, trunc_none=trunc_none); lg = model(ids, att).float().cpu()
        for b, it in enumerate(chunk):
            for k, t in enumerate(gold_labels(it)):
                if firsts[b][k] is not None: L.append(lg[b, firsts[b][k]]); Y.append(LAB[t])
    return torch.stack(L), torch.tensor(Y)

def fit_temperature(model, tok, items, device, trunc_none=False):
    """R10 item 6: one temperature for the tagger's softmax, fitted on the CALIBRATION rows (token cross-entropy, grid of 200 values on a
    log scale in [0.05, 20]); calibration rows are the training distribution, so this cannot calibrate the out-of-distribution axes"""
    L, Y = logits_and_labels(model, tok, items, device, trunc_none=trunc_none); ce = nn.CrossEntropyLoss()
    Ts = np.exp(np.linspace(np.log(0.05), np.log(20), 200)); v = [float(ce(L / t, Y)) for t in Ts]; return float(Ts[int(np.argmin(v))])

def subword_positions(tok, items, trunc_none=False):
    """R10 item 2a: the first-sub-token index of every gold filler (the unit of the tagger's absolute positions)"""
    _, _, firsts = encode_batch(tok, [it["tokens"] for it in items], "cpu", trunc_none=trunc_none)
    return [{r: (firsts[b][it["tokens"].index(f)] if f is not None and f in it["tokens"] else None) for r, f in it["frame"].items()} for b, it in enumerate(items)]

def train_max_subwords(tok, data, max_len=96):
    """the longest training sentence in sub-tokens, [CLS] and [SEP] included: a sub-token index >= this is never reached in training"""
    best = 0
    for i in range(0, len(data), 1024):
        enc = tok([" ".join(t) for t, _ in data[i:i + 1024]], truncation=True, max_length=max_len); best = max(best, max(len(x) for x in enc["input_ids"]))
    return best

def model_bytes(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--tokenizer", default="wordpiece", choices=["wordpiece", "whole"])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--r10", action="store_true", help="R10 row: --select r10 --wd_groups --seed_per_config --trunc_none --temperature")
    ap.add_argument("--select", default="prereg", choices=["prereg", "r10"], help="r10: ties of validation FULL go to the lowest validation cross-entropy (R10 item 3)")
    ap.add_argument("--wd_groups", action="store_true"); ap.add_argument("--seed_per_config", action="store_true"); ap.add_argument("--trunc_none", action="store_true")
    ap.add_argument("--temperature", action="store_true"); ap.add_argument("--pos_offset", type=int, default=0); ap.add_argument("--whole_init", default="resize", choices=["resize", "pieces"])
    ap.add_argument("--tag_suffix", default="", help="appended to the results / model names (R17: a re-run of a row with its selected configuration and saved models)")
    ap.add_argument("--save_all", action="store_true", help="R17: save every seed's model to logs/models/<tag>_seed<s>")
    ap.add_argument("--oow_class", action="store_true", help="R17 item 1 (ii): a [CLS] out-of-world head trained on the 'neg' part of data/oow_seed<s>.json")
    ap.add_argument("--depth_head", default="none", choices=["none", "cls", "reg"], help="R17 item 3: a [CLS] depth head")
    ap.add_argument("--cues", default="none", choices=["none", "given"], help="R14: 'given' adds the learned cue embedding on the CUE_WORDS tokens (row 'cues given')")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lrs", default=None); ap.add_argument("--epochs", default=None); ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--save_model", action="store_true", help="save the seed-0 model and tokenizer to logs/models/ for eval/determinism.py")
    a = ap.parse_args()
    if a.r10: a.select, a.wd_groups, a.seed_per_config, a.trunc_none, a.temperature = "r10", True, True, True, True
    r10 = a.r10 or a.select == "r10" or a.wd_groups or a.seed_per_config or a.trunc_none or a.temperature or a.pos_offset or a.whole_init != "resize"
    lrs = [float(x) for x in (a.lrs or ("1e-4,3e-4,1e-3" if a.scratch else "2e-5,3e-5,5e-5")).split(",")]
    eps = [int(x) for x in (a.epochs or ("10,20,40" if a.scratch else "3,5,10")).split(",")]
    name = f"{a.model}{' from scratch' if a.scratch else ' pretrained'} [{a.tokenizer}]"; tag = f"bert_{a.model.replace('/', '_')}_{'scratch' if a.scratch else 'pre'}_{a.tokenizer}"
    if r10:
        name += " reviewed (R10" + (f", position offsets 0-{a.pos_offset}" if a.pos_offset else "") + (", piece-mean init" if a.whole_init == "pieces" else "") + ")"
        tag += "_r10" + (f"_off{a.pos_offset}" if a.pos_offset else "") + ("_pieceinit" if a.whole_init == "pieces" else "")
        print(f"R10 flags: select {a.select}, wd groups {a.wd_groups}, seed per config {a.seed_per_config}, truncated word -> None {a.trunc_none}, temperature {a.temperature}, position offsets 0-{a.pos_offset}, whole init {a.whole_init}")
    if a.cues == "given":
        name += " cues given (R14)"; tag += "_cues"; print(f"R14: cue embedding added on the tokens of {CUE_WORDS}")
    if a.oow_class or a.depth_head != "none":
        name += " R17 (" + ", ".join(x for x in (("out-of-world class" if a.oow_class else ""), (f"depth head {a.depth_head}" if a.depth_head != "none" else "")) if x) + ")"
        tag += "_r17" + ("_oow" if a.oow_class else "") + (f"_depth{a.depth_head}" if a.depth_head != "none" else "")
    if a.tag_suffix: name += f" [{a.tag_suffix.strip('_')}]"; tag += a.tag_suffix
    print(f"bert_basic: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; N {a.N}; seeds {a.seeds}; grid lr {lrs} x epochs {eps}; batch {a.batch}")
    print("regime (rule 24): basic world, prefixed forms, training depths 0-2, tested at depth 3 UNSEEN; token-tagging head; decoding restricted to the world's candidates; frozen splits")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167"); chosen = None; rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, a.D); torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(a.model); piece_init = None
        if a.tokenizer == "whole":
            base = AutoTokenizer.from_pretrained(a.model); n0 = len(tok); tok.add_tokens(list(split["vocab"]))
            if a.whole_init == "pieces": piece_init = {tok.convert_tokens_to_ids(w_): base.convert_tokens_to_ids(base.tokenize(w_)) for w_ in split["vocab"] if tok.convert_tokens_to_ids(w_) >= n0}
        pieces = np.mean([len(tok.tokenize(w_)) for w_ in split["vocab"]]); print(f"  seed {seed}: tokenizer '{a.tokenizer}', pieces per world word {pieces:.2f}")
        neg = [x["tokens"] for src in ("random", "reference", "nonce") for x in json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")))["neg"][src]] if a.oow_class else None
        data = make_train(bw, seed, a.N); val = [(t, g) for t, g in make_train(bw, 900 + seed, 2000)]
        val_items = [{"tokens": t, "frame": bwm.frame_from_tags(t, g), "depth": sum(x.startswith("X") for x in g)} for t, g in val]
        val_split = {"axes": {"val": val_items}, "candidates": split["candidates"]}; best = None
        for lr in (lrs if seed == 0 else [chosen[0]]):
            for ep in (eps if seed == 0 else [chosen[1]]):
                if a.seed_per_config: torch.manual_seed(seed)   # R10 item 14: a selected configuration can be re-run alone
                model = Tagger(a.model, a.scratch, len(tok), piece_init, tok if a.cues == "given" else None, a.oow_class, a.depth_head).to(a.device); t0 = time.time(); train(model, tok, data, lr, ep, a.batch, a.device, seed, a.wd_groups, a.pos_offset, a.trunc_none, neg)
                vp = predict(model, tok, val_split, "val", a.device, trunc_none=a.trunc_none); vfull = np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.roles_at(it["depth"])) for it, p in zip(val_items, vp)])
                L_, Y_ = logits_and_labels(model, tok, val_items, a.device, trunc_none=a.trunc_none); vce = float(nn.CrossEntropyLoss()(L_, Y_))
                print(f"  seed {seed} lr={lr:g} epochs={ep} ({time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params) validation FULL {vfull:.3f}, token cross-entropy {vce:.5f}", flush=True)
                better = best is None or vfull > best[0] or (a.select == "r10" and vfull == best[0] and vce < best[4])
                if better:
                    if best is not None: del best[3]
                    best = [vfull, lr, ep, model, vce]
                else: del model
                torch.cuda.empty_cache() if a.device == "cuda" else None
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, epochs={chosen[1]} (by validation FULL frame, seed 0{'; ties to the lowest validation cross-entropy, R10' if a.select == 'r10' else ''})")
        model = best[3]; preds = {axis: predict(model, tok, split, axis, a.device, trunc_none=a.trunc_none) for axis in report.AXES + ["calibration"]}
        extra = {}
        if a.temperature:
            T = fit_temperature(model, tok, split["axes"]["calibration"], a.device, a.trunc_none); print(f"  seed {seed} temperature (calibration rows) {T:.3f}")
            extra["temperature"] = {axis: predict(model, tok, split, axis, a.device, T=T, trunc_none=a.trunc_none) for axis in report.AXES + ["calibration"]}
        pos = subword_positions(tok, split["axes"]["depth3_unseen"], a.trunc_none); tmax = train_max_subwords(tok, data)
        row = report.evaluate(name, seed, preds, split, tag, positions=pos, train_max=tmax, unit="sub-token", extra_rules=extra)
        if a.temperature: row["temperature"]["T"] = T
        row["occupancy"] = {ax: occupancy(split["axes"][ax], preds[ax]) for ax in ("seen", "new_ids", "depth3_unseen")}
        # cost: bytes and CPU milliseconds per sentence (the stated machine is the one printed by run.sh)
        cpu = model.to("cpu"); sents = split["axes"]["seen"][:a.cost_n]; t0 = time.time()
        for it in sents: predict(cpu, tok, {"axes": {"x": [it]}, "candidates": split["candidates"]}, "x", "cpu", batch=1)
        ms = 1000 * (time.time() - t0) / len(sents); row["cost"] = {"params": sum(p.numel() for p in model.parameters()), "bytes": model_bytes(model), "cpu_ms_per_sentence": ms}
        # R10 item 13: fp16 size, int8 dynamic quantization (size and CPU ms), the CPU thread count; the embedding table's share
        q = torch.ao.quantization.quantize_dynamic(cpu, {nn.Linear}, dtype=torch.qint8); t0 = time.time()
        for it in sents: predict(q, tok, {"axes": {"x": [it]}, "candidates": split["candidates"]}, "x", "cpu", batch=1)
        emb = cpu.enc.get_input_embeddings().weight
        row["cost"].update({"bytes_fp16": 2 * row["cost"]["params"], "bytes_int8_dynamic": model_bytes(q), "cpu_ms_per_sentence_int8": 1000 * (time.time() - t0) / len(sents),
                            "cpu_threads": torch.get_num_threads(), "embedding_rows": int(emb.shape[0]), "embedding_params": int(emb.numel())}); del q
        print(f"  seed {seed} occupancy (free tags) {row['occupancy']}; cost {row['cost']['params']/1e6:.1f}M params, {row['cost']['bytes']/1e6:.0f} MB, {ms:.1f} ms/sentence on CPU")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        if (a.save_model and seed == 0) or a.save_all:
            md = os.path.join(mainrepo.ROOT, "logs", "models", f"{tag}_seed{seed}"); os.makedirs(md, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(md, "tagger.pt")); tok.save_pretrained(md)
            json.dump({"model": a.model, "scratch": a.scratch, "n_tokens": len(tok), "cues": a.cues, "oow_class": a.oow_class, "depth_head": a.depth_head, "seed": seed}, open(os.path.join(md, "config.json"), "w"))
        rows.append(row); model.to(a.device); del model
    report.summarize(rows, name)
    print("written interpretation (INSTRUCTIONS table 3): new ids > 0.9 => the pieces carry the role; < 0.5 => the vocabulary gap is structural; depth 3 > 0.9 would contradict the paper's same-input claim, < 0.5 confirms it; in between reported with both readings")
    metrics.checklist(a.seeds, "[x] tokenizer condition stated [x] grid fixed in advance")
