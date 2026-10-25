"""bert_mrc.py (deviation R24, responses R24 / R25, 2026-10-25; RESULTS): BERT-MRC on the basic world, the question-based form of the
task (QA-SRL, He et al. 2015; MRC extraction, Li et al. 2020). Input "[CLS] question [SEP] sentence [SEP]", a span head (start / end
logits) over the sentence's sub-tokens, NULL = the [CLS] position; one example per (sentence, question); the questions are the frozen
QUESTIONS below (one per role and level, verb-neutral, questions Q18 (2)). Training on the same 30k sentences as every model (rng
200+seed, depths 0-2), every question of `--train_questions` per sentence, NULL for the level questions absent at that depth:
'zs' (default, Q18 (1a)) asks A, B, C, level 1, level 2, so the level-3 question is never seen in training; 'all' (Q18 (1b)) asks all six
(level 3 always NULL in training).
Decoding: a role's answer = the candidate word (the world's candidate list, as for every model) with the highest start + end logit
(first and last sub-token of the word), or NULL if the [CLS] score is higher; margin = best minus second best over candidates and NULL.
Depth given (the questions of the sentence's depth, as every model) and depth NOT given (all six questions; NULL = not given; the depth is
the number of answered level questions). R17 (i) in the run: on the frozen out-of-world sets, rejected = some question of the
opener-count depth answered NULL ('NULL refusal'), and the threshold rule of R17 (smallest NULL gap at the 2% false-rejection point of
the calibration rows); the residual analogue = the share of content words that answer no question.
Grid (INSTRUCTIONS section 4, the pretrained grid): lr {2e-5, 3e-5, 5e-5} x epochs {3, 5, 10} on seed 0, selected by validation FULL,
ties to the lowest validation span loss (R24: the R10 tie-break if every point is 1.000). Because the seed-0 grid exceeds run.sh's 6 h
cap, it runs as one run per learning rate (`--grid_only --lrs <lr>`, writing logs/mrc_grid_<tag>_lr<lr>.json); `--select` reads them and
prints the chosen configuration; the 5-seed run then uses it (`--lrs <lr> --epochs <ep>`).
Run: ./run.sh models/bert_mrc.py --model bert-base-uncased [--train_questions zs|all] [--grid_only --lrs 2e-5] | --select | --lrs L --epochs E --seeds 5"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from common import basic_world as bwm, decode, report
from eval import metrics
from models.bert_basic import make_train, model_bytes

QUESTIONS = {"A": "who gives?", "B": "what is given?", "C": "to whom is it given?",
             "X1": "who says it?", "X2": "who says it at level 2?", "X3": "who says it at level 3?"}   # frozen (R24; Q18 (2))
FR_MAX = 0.02; VERBS = ("gives", "sells", "lends", "sends", "offers", "shows", "buys", "gets", "receives", "borrows", "accepts", "sees")

class MRC(nn.Module):
    def __init__(self, name):
        super().__init__(); self.enc = AutoModel.from_pretrained(name); self.span = nn.Linear(AutoConfig.from_pretrained(name).hidden_size, 2); self.drop = nn.Dropout(0.1)
    def forward(self, ids, att, tt=None):
        h = self.enc(input_ids=ids, attention_mask=att, **({"token_type_ids": tt} if tt is not None else {})).last_hidden_state
        s, e = self.span(self.drop(h)).unbind(-1); m = att == 0
        return s.masked_fill(m, -1e4), e.masked_fill(m, -1e4)

def encode(tok, pairs, device, max_len=128):
    """pairs: (question, sentence tokens) -> ids, attention, token types, and per pair the (first, last) sub-token of every sentence word"""
    enc = tok([q for q, _ in pairs], [" ".join(s) for _, s in pairs], return_tensors="pt", padding=True, truncation="only_second", max_length=max_len, return_offsets_mapping=True)
    spans = []
    for b, (_, s) in enumerate(pairs):
        offs = enc["offset_mapping"][b].tolist(); sid = enc.sequence_ids(b); st = []; pos = 0
        for w_ in s: st.append((pos, pos + len(w_))); pos += len(w_) + 1
        sp = []
        for a0, a1 in st:
            js = [j for j, (o, q) in enumerate(offs) if sid[j] == 1 and o >= a0 and q <= a1 and q > o]
            sp.append((js[0], js[-1]) if js else None)
        spans.append(sp)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), enc.get("token_type_ids", None).to(device) if "token_type_ids" in enc else None, spans

def examples(data, roles_fn):
    """(question, tokens, gold word index or None) for every training sentence and every asked question"""
    out = []
    for toks, tags in data:
        d = sum(t.startswith("X") for t in tags)
        for r in roles_fn(d):
            k = tags.index(r) if r in tags else None; out.append((QUESTIONS[r], toks, k))
    return out

def train(model, tok, ex, lr, epochs, batch, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(ex) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ex), generator=g).tolist()
        for i in range(0, len(ex), batch):
            chunk = [ex[j] for j in perm[i:i + batch]]; ids, att, tt, spans = encode(tok, [(q, s) for q, s, _ in chunk], device)
            ys = torch.tensor([0 if k is None else spans[b][k][0] for b, (_, _, k) in enumerate(chunk)], device=device)
            ye = torch.tensor([0 if k is None else spans[b][k][1] for b, (_, _, k) in enumerate(chunk)], device=device)
            s, e = model(ids, att, tt); loss = ce(s, ys) + ce(e, ye)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def answer(model, tok, pairs_meta, device, batch=256):
    """pairs_meta: (question, tokens, candidate set) -> (word index or None, margin, null gap = best candidate - NULL)"""
    model.eval(); out = []
    for i in range(0, len(pairs_meta), batch):
        chunk = pairs_meta[i:i + batch]; ids, att, tt, spans = encode(tok, [(q, s) for q, s, _ in chunk], device)
        s, e = model(ids, att, tt); s, e = s.float().cpu().numpy(), e.float().cpu().numpy()
        for b, (_, toks, cands) in enumerate(chunk):
            null = s[b, 0] + e[b, 0]; sc = [(s[b, spans[b][k][0]] + e[b, spans[b][k][1]], k) for k, t in enumerate(toks) if t in cands and spans[b][k] is not None]
            allsc = sorted([v for v, _ in sc] + [null], reverse=True); best = max(sc) if sc else (-np.inf, None)
            k = best[1] if best[0] > null else None
            out.append((k, float(allsc[0] - allsc[1]) if len(allsc) > 1 else float(allsc[0]), float(best[0] - null)))
    return out

def read(model, tok, split, axis, items, device, given=True):
    """predictions in the scorer's format; depth given: the questions of the item's depth; not given: all six, NULL = not given"""
    roles = (lambda it: bwm.roles_at(it["depth"])) if given else (lambda it: bwm.ROLES)
    meta, idx = [], []
    for n, it in enumerate(items):
        for r in roles(it):
            meta.append((QUESTIONS[r], it["tokens"], set(bwm.candidates_for(split, axis, r)))); idx.append((n, r))
    ans = answer(model, tok, meta, device); preds = [{"frame": {r: None for r in bwm.ROLES}, "margins": {}, "null_gap": {}} for _ in items]
    for (n, r), (k, m, gap) in zip(idx, ans):
        preds[n]["frame"][r] = None if k is None else items[n]["tokens"][k]; preds[n]["margins"][r] = m; preds[n]["null_gap"][r] = gap
    return preds

def r17(model, tok, split, oow, device):
    """R17 (i) for MRC: rejection on the frozen out-of-world sets (depth from the level openers), NULL refusal and the threshold rule"""
    est = lambda items: [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]
    gap = lambda items, ax="x": [min(p["null_gap"].values()) for p in read(model, tok, split, ax, est(items), device)]
    cal = np.array(gap(split["axes"]["calibration"])); t = float(np.quantile(cal, FR_MAX))
    inw = {ax: np.array(gap(split["axes"][ax], ax)) for ax in report.AXES if split["axes"][ax]}
    res = {"threshold": t, "rejected_null": {}, "rejected_threshold": {}, "false_rejection_null": {}, "false_rejection_threshold": {}}
    for src in ("random", "reference", "nonce"):
        g = np.array(gap(oow["test"][src])); res["rejected_null"][src] = float(np.mean(g < 0)); res["rejected_threshold"][src] = float(np.mean(g < t))
    for ax, g in inw.items(): res["false_rejection_null"][ax] = float(np.mean(g < 0)); res["false_rejection_threshold"][ax] = float(np.mean(g < t))
    allg = np.concatenate(list(inw.values())); res["false_rejection_pooled"] = {"null": float(np.mean(allg < 0)), "threshold": float(np.mean(allg < t))}
    return res

def residual_analogue(items, preds, split):
    cat = split["cat"]; out = []
    for it, p in zip(items, preds):
        content = [k for k, t in enumerate(it["tokens"]) if t not in cat or cat[t] in ("P", "O") or t in VERBS]
        used = {v for v in p["frame"].values() if v is not None}
        out.append(float(np.mean([it["tokens"][k] not in used for k in content])) if content else 0.0)
    return float(np.mean(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--train_questions", default="zs", choices=["zs", "all"])
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lrs", default="2e-5,3e-5,5e-5"); ap.add_argument("--epochs", default="3,5,10"); ap.add_argument("--grid_only", action="store_true"); ap.add_argument("--select", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--n_val", type=int, default=2000); ap.add_argument("--save_model", action="store_true")
    a = ap.parse_args()
    tag = f"mrc_{a.model.replace('/', '_')}_{a.train_questions}"; name = f"{a.model} MRC [train questions {a.train_questions}]"
    if a.select:
        pts = [p for f in sorted(glob.glob(os.path.join(mainrepo.ROOT, "logs", f"mrc_grid_{tag}_lr*.json"))) for p in json.load(open(f))]
        best = max(pts, key=lambda p: (p["val_full"], -p["val_loss"])); print(f"bert_mrc --select ({tag}): {len(pts)} grid points: " + "; ".join(f"lr {p['lr']:g} ep {p['epochs']} FULL {p['val_full']:.3f} loss {p['val_loss']:.5f}" for p in pts))
        print(f"  => chosen config: lr={best['lr']:g}, epochs={best['epochs']} (by validation FULL, ties to the lowest validation span loss)"); sys.exit(0)
    lrs = [float(x) for x in a.lrs.split(",")]; eps = [int(x) for x in a.epochs.split(",")]
    tq = (lambda d: ["A", "B", "C", "X1", "X2"]) if a.train_questions == "zs" else (lambda d: bwm.ROLES)
    print(f"bert_mrc: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; N {a.N}; seeds {1 if a.grid_only else a.seeds}; lr {lrs} x epochs {eps}; batch {a.batch}; questions {QUESTIONS}")
    print("regime (rule 24): basic world, prefixed forms, training depths 0-2, tested at depth 3 UNSEEN; span head over the sentence, NULL on [CLS]; decoding restricted to the world's candidates; frozen splits")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167"); rows = []; grid = []
    for seed in (range(1) if a.grid_only else range(a.seeds)):
        split = bwm.load(seed); bw = bwm.build(seed, a.D); tok = AutoTokenizer.from_pretrained(a.model)
        data = make_train(bw, seed, a.N); ex = examples(data, tq)
        val = make_train(bw, 900 + seed, a.n_val); vitems = [{"tokens": t, "frame": bwm.frame_from_tags(t, g), "depth": sum(x.startswith("X") for x in g)} for t, g in val]
        vex = examples(val, tq); best = None
        for lr in lrs:
            for ep in eps:
                torch.manual_seed(seed); model = MRC(a.model).to(a.device); t0 = time.time(); train(model, tok, ex, lr, ep, a.batch, a.device, seed); tt = time.time() - t0
                vp = read(model, tok, {"axes": {"val": vitems}, "candidates": split["candidates"]}, "val", vitems, a.device)
                vfull = float(np.mean([all(p["frame"][r] == it["frame"][r] for r in bwm.roles_at(it["depth"])) for it, p in zip(vitems, vp)]))
                with torch.no_grad():
                    model.eval(); ce = nn.CrossEntropyLoss(reduction="sum"); L = 0.0
                    for i in range(0, len(vex), 256):
                        chunk = vex[i:i + 256]; ids, att, ttk, spans = encode(tok, [(q, s) for q, s, _ in chunk], a.device); s, e = model(ids, att, ttk)
                        ys = torch.tensor([0 if k is None else spans[b][k][0] for b, (_, _, k) in enumerate(chunk)], device=a.device); ye = torch.tensor([0 if k is None else spans[b][k][1] for b, (_, _, k) in enumerate(chunk)], device=a.device)
                        L += float(ce(s, ys) + ce(e, ye))
                vloss = L / len(vex)
                print(f"  seed {seed} lr={lr:g} epochs={ep} ({tt:.0f}s, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {len(ex)} examples) validation FULL {vfull:.3f}, span loss {vloss:.5f}; peak cuda reserved {torch.cuda.max_memory_reserved()/1e9 if a.device == 'cuda' else 0:.1f} GB", flush=True)
                grid.append({"lr": lr, "epochs": ep, "val_full": vfull, "val_loss": vloss, "train_s": tt})
                if best is None or (vfull, -vloss) > (best[0], -best[1]):
                    if best is not None: del best[2]
                    best = [vfull, vloss, model, tt]
                else: del model
                torch.cuda.empty_cache() if a.device == "cuda" else None
        if a.grid_only:
            p = os.path.join(mainrepo.ROOT, "logs", f"mrc_grid_{tag}_lr{'_'.join(f'{x:g}' for x in lrs)}.json"); json.dump(grid, open(p, "w")); print(f"grid points written: {os.path.relpath(p, mainrepo.ROOT)}"); break
        model = best[2]; preds = {ax: read(model, tok, split, ax, split["axes"][ax], a.device) for ax in report.AXES + ["calibration"]}
        row = report.evaluate(name + " (depth given)", seed, preds, split, tag)
        # depth not given: all six questions, NULL = not given (R24's natural form of R17 item 3)
        ng = read(model, tok, split, "depth3_unseen", split["axes"]["depth3_unseen"], a.device, given=False); g3 = split["axes"]["depth3_unseen"]
        full = lambda roles: float(np.mean([all(p["frame"].get(r) == it["frame"][r] for r in roles) for p, it in zip(ng, g3)]))
        dep = lambda p: sum(p["frame"][f"X{k}"] is not None for k in (1, 2, 3))
        row["depth_not_given"] = {"full_six": full(bwm.ROLES), "full_trained": full(["A", "B", "C", "X1", "X2"]), "full_inner": full(["A", "B", "C"]),
                                  "depth_accuracy_depth3": float(np.mean([dep(p) == 3 for p in ng]))}
        occ = {}
        for ax in ("seen", "new_ids", "depth3_unseen"):
            pn = read(model, tok, split, ax, split["axes"][ax], a.device, given=False); fa = na = fm = nm = 0
            for it, p in zip(split["axes"][ax], pn):
                for r in bwm.ROLES:
                    gtr = it["frame"].get(r); f = p["frame"].get(r)
                    if gtr is None: na += 1; fa += f is not None
                    else: nm += 1; fm += f is None
            occ[ax] = {"false_fill_on_absent": fa / max(1, na), "missed_present": fm / max(1, nm)}
        row["occupancy_depth_not_given"] = occ
        row["r17"] = r17(model, tok, split, json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"))), a.device)
        row["residual_analogue"] = {ax: residual_analogue(split["axes"][ax], preds[ax], split) for ax in ("seen", "new_ids", "depth3_unseen")}
        # cost: inference with ALL six questions per sentence (six passes), CPU; and with the questions of the depth
        cpu = model.to("cpu"); sents = split["axes"]["depth3_unseen"][:a.cost_n]; t0 = time.time()
        for it in sents: read(cpu, tok, split, "depth3_unseen", [it], "cpu", given=False)
        ms6 = 1000 * (time.time() - t0) / len(sents)
        row["cost"] = {"params": sum(p.numel() for p in cpu.parameters()), "bytes": model_bytes(cpu), "bytes_fp16": 2 * sum(p.numel() for p in cpu.parameters()), "cpu_ms_per_sentence_all_questions": ms6,
                       "passes_per_sentence_all_questions": 6, "cpu_threads": torch.get_num_threads(), "batch": a.batch, "train_s": best[3], "examples_per_epoch": len(ex)}
        print(f"  seed {seed} depth not given: {row['depth_not_given']}; occupancy (not given) {occ}")
        print(f"  seed {seed} R17 (i): {row['r17']}; residual analogue {row['residual_analogue']}")
        print(f"  seed {seed} cost: {row['cost']['params']/1e6:.1f}M params, {row['cost']['bytes']/1e6:.0f} MB, {ms6:.1f} ms per sentence on CPU with all six questions (6 passes), {torch.get_num_threads()} threads")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        if a.save_model and seed == 0:
            md = os.path.join(mainrepo.ROOT, "logs", "models", f"{tag}_seed0"); os.makedirs(md, exist_ok=True); torch.save(cpu.state_dict(), os.path.join(md, "mrc.pt")); tok.save_pretrained(md)
        rows.append(row); del model, cpu
    if not a.grid_only:
        report.summarize(rows, name + " (depth given)")
        ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
        for k in ("full_six", "full_trained", "full_inner", "depth_accuracy_depth3"): print(f"  depth NOT given, depth 3: {k} {ms([r['depth_not_given'][k] for r in rows])}")
        for v in ("null", "threshold"):
            for src in ("random", "reference", "nonce"): print(f"  R17 (i) rejected ({v}) {src:10s} {ms([r['r17'][f'rejected_{v}'][src] for r in rows])}")
            print(f"  R17 (i) false rejection ({v}) pooled {ms([r['r17']['false_rejection_pooled'][v] for r in rows])}")
        print("written interpretation (R24): inner-frame FULL at depth 3 >= 0.9 => the question form composes as the tagger did; the six-role FULL and the level-3 question are the new information; confident-wrong under NULL training against the reader's residual is the calibration comparison; the cost ratio is the argument that survives")
    metrics.checklist(1 if a.grid_only else a.seeds, "[x] questions frozen before training (R24) [x] grid fixed in advance [x] candidates restricted as every model")
