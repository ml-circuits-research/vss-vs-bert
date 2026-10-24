"""bert_real.py (runs 8-10, encoders): a pretrained encoder (or the same architecture from scratch) fine-tuned on REAL utterances (SNIPS,
MASSIVE, TOPv2) with a token-tagging BIO head over the slot types, an intent head on the [CLS] vector, and a per-word NESTED head (does
the slot this word belongs to hold an intent?). One example per intent NODE (common/real_data.node_examples: the node's own span, its
intent, its direct slots); at reading the model is applied to the utterance, then RECURSIVELY to the span of every slot read as nested
(up to depth 4), with the same weights at every level: nothing in the architecture ties it to the depths seen in training (for the flat
datasets the nested head is trained on zeros and never fires). Scored by common/report_real.evaluate (eval/metrics.score_real), the
reader's scorer, on the same condition rows.

Frame (the PRIMARY metric, the same rule as the reader's): spans decoded from the BIO tags (an I- without its B- starts a span), the first
span of each type by position, its head = its last token; a nested slot -> the frame read on its span. Margins (larger = surer): intent =
top1 - top2 probability; a filled slot = the smallest top1 - top2 tag-probability gap over its words (and, on nested data, the nested
decision |2 p - 1|, the smaller); an absent slot type r = max(0, 1 - 2 max_w p(B-r or I-r)) (its confidence that r is absent). The same
threshold rule as the reader, on the dev calibration half: plain softmax, and softmax with TEMPERATURE scaling (one temperature for the
intent logits and one for the tag logits, fitted by NLL on the dev calibration half; argmax unchanged, margins rescaled): both rows are
written (variant 'temperature'). SECONDARY: exact span F1 of all top-level spans (the benchmarks' slot F1).
Grid (fixed, INSTRUCTIONS section 4): pretrained lr {2e-5, 3e-5, 5e-5} x epochs {3, 5, 10}; scratch lr {1e-4, 3e-4, 1e-3} x epochs {10, 20,
40}; batch 32; weight decay 0.01; warm-up 10%; selection on seed 0 only by FULL head frame on the dev CALIBRATION half (rule 25), per size.
Models: bert-base-uncased, the small encoder of run 3 (RESULTS), --scratch; for the multilingual round (MASSIVE, --locale):
bert-base-multilingual-cased, xlm-roberta-base. Tokens are the loader's (lowercased: stated, for every model).
Run: python3 models/bert_real.py --dataset snips|massive|topv2 --model bert-base-uncased [--scratch] [--condition base] [--sizes 1000,5000,full] [--seeds 5] [--test]"""
import argparse, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from common import real_data as rd, report_real
from eval import metrics

class NodeTagger(nn.Module):
    def __init__(self, name, scratch, n_labels, n_intents):
        super().__init__(); cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_config(cfg) if scratch else AutoModel.from_pretrained(name)
        h = cfg.hidden_size; self.tag = nn.Linear(h, n_labels); self.intent = nn.Linear(h, n_intents); self.nested = nn.Linear(h, 1); self.drop = nn.Dropout(0.1)
    def forward(self, ids, att):
        x = self.drop(self.enc(input_ids=ids, attention_mask=att).last_hidden_state)
        return self.tag(x), self.intent(x[:, 0]), self.nested(x).squeeze(-1)

def encode(tok, toks_list, device, max_len=96):
    """word-level inputs: the first sub-token of every word carries the word's label; words cut by truncation get None"""
    e = tok([t if t else ["."] for t in toks_list], is_split_into_words=True, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    firsts = []
    for b, t in enumerate(toks_list):
        wid = e.word_ids(b); f = [None] * len(t)
        for j, w in enumerate(wid):
            if w is not None and w < len(t) and f[w] is None: f[w] = j
        firsts.append(f)
    return e["input_ids"].to(device), e["attention_mask"].to(device), firsts

class Labels:
    def __init__(self, types, intents):
        self.types = sorted(types); self.L = ["O"] + [p + t for t in self.types for p in ("B-", "I-")]; self.id = {l: i for i, l in enumerate(self.L)}
        self.intents = sorted(intents); self.iid = {x: i for i, x in enumerate(self.intents)}

def train(model, tok, nodes, lab, lr, epochs, batch, device, seed, nested):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * ((len(nodes) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); g = torch.Generator().manual_seed(seed)
    ce = nn.CrossEntropyLoss(ignore_index=-100); bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(nodes), generator=g).tolist()
        for i in range(0, len(nodes), batch):
            ch = [nodes[j] for j in perm[i:i + batch]]; ids, att, firsts = encode(tok, [c["tokens"] for c in ch], device)
            y = torch.full(ids.shape, -100, dtype=torch.long); yn = torch.zeros(ids.shape); mn = torch.zeros(ids.shape)
            for b, c in enumerate(ch):
                tags = rd.bio(len(c["tokens"]), [(t, s, e) for t, s, e, _ in c["slots"]]); nf = [0.0] * len(c["tokens"])
                for t, s, e, isn in c["slots"]:
                    for k in range(s, e): nf[k] = 1.0 if isn else 0.0
                for k, f in enumerate(firsts[b]):
                    if f is None: continue
                    y[b, f] = lab.id.get(tags[k], 0); yn[b, f] = nf[k]; mn[b, f] = 1.0
            tl, il, nl = model(ids, att)
            loss = ce(tl.reshape(-1, len(lab.L)), y.view(-1).to(device)) + ce(il, torch.tensor([lab.iid[c["intent"]] for c in ch], device=device))
            if nested: loss = loss + (bce(nl, yn.to(device)) * mn.to(device)).sum() / mn.sum().clamp(min=1).to(device)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

@torch.no_grad()
def logits_of(model, tok, toks_list, device, batch=64):
    model.eval(); out = []
    for i in range(0, len(toks_list), batch):
        ch = toks_list[i:i + batch]; ids, att, firsts = encode(tok, ch, device); tl, il, nl = model(ids, att)
        tl, il, nl = tl.float().cpu().numpy(), il.float().cpu().numpy(), nl.float().cpu().numpy()
        for b, t in enumerate(ch):
            f = firsts[b]; W = np.zeros((len(f), tl.shape[-1]))
            for k, j in enumerate(f):
                if j is None: W[k] = -1e4; W[k, 0] = 0.0          # a truncated word is read as O
                else: W[k] = tl[b, j]
            out.append((W, il[b], np.array([nl[b, j] if j is not None else -1e4 for j in f])))
    return out

def softmax(x, T=1.0):
    z = x / T; z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def read_level(W, il, nl, lab, T, nested):
    pt = softmax(W, T[1]) if len(W) else W; pi = softmax(il, T[0]); tags = [lab.L[j] for j in pt.argmax(1)] if len(W) else []
    sp = rd.spans_of_bio(tags); o = np.argsort(-pi); out = {"intent": (lab.intents[o[0]], float(pi[o[0]] - pi[o[1]])), "spans": sp, "slots": []}
    gap = (np.sort(pt, 1)[:, -1] - np.sort(pt, 1)[:, -2]) if len(W) else np.zeros(0); seen = set()
    pn = 1 / (1 + np.exp(-np.clip(nl, -50, 50))) if len(nl) else nl
    for t, s, e in sorted(sp, key=lambda x: x[1]):
        if t in seen: continue
        seen.add(t); m = float(gap[s:e].min()); isn = False
        if nested: q = float(pn[s:e].mean()); isn = q >= 0.5; m = min(m, abs(2 * q - 1))
        out["slots"].append((t, s, e, isn, m))
    out["absent"] = {}
    for r in lab.types:
        if r in seen: continue
        jb, ji = lab.id["B-" + r], lab.id["I-" + r]; sr = (pt[:, jb] + pt[:, ji]).max() if len(W) else 0.0
        out["absent"][r] = float(max(0.0, 1 - 2 * sr))
    return out

def predict(model, tok, items, lab, device, T=(1.0, 1.0), nested=False, max_depth=4):
    """frames for items (recursive on nested spans, level by level in batches)"""
    preds = [{"frame": {}, "margins": {}, "spans": []} for _ in items]
    queue = [(k, "", it["tokens"], 0) for k, it in enumerate(items)]
    while queue:
        L = logits_of(model, tok, [q[2] for q in queue], device); nxt = []
        for (k, prefix, toks, d), (W, il, nl) in zip(queue, L):
            r = read_level(W, il, nl, lab, T, nested); p = preds[k]
            p["frame"][prefix + "intent"], p["margins"][prefix + "intent"] = r["intent"]
            if prefix == "": p["spans"] = r["spans"]
            for t, s, e, isn, m in r["slots"]:
                if isn and d < max_depth: p["frame"][prefix + t] = rd.NESTED; p["margins"][prefix + t] = m; nxt.append((k, prefix + t + "/", toks[s:e], d + 1))
                else: p["frame"][prefix + t] = toks[e - 1]; p["margins"][prefix + t] = m
            for t, m in r["absent"].items(): p["margins"][prefix + t] = m
        queue = nxt
    return preds

def fit_temperatures(model, tok, items, lab, device):
    """one temperature for the intent logits and one for the tag logits, minimising the NLL on the CALIBRATION rows (top-level nodes)"""
    nodes = [rd.node_examples(it)[0] for it in items]; L = logits_of(model, tok, [n["tokens"] for n in nodes], device)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
    yi = np.array([lab.iid.get(n["intent"], -1) for n in nodes]); IL = np.array([x[1] for x in L]); ok = yi >= 0
    nll_i = lambda T: -np.mean(np.log(softmax(IL[ok], T)[np.arange(ok.sum()), yi[ok]] + 1e-12))
    W = np.concatenate([x[0] for x in L if len(x[0])]); y = np.concatenate([[lab.id.get(t, 0) for t in rd.bio(len(n["tokens"]), [(a, s, e) for a, s, e, _ in n["slots"]])] for n, x in zip(nodes, L) if len(x[0])])
    nll_t = lambda T: -np.mean(np.log(softmax(W, T)[np.arange(len(y)), y] + 1e-12))
    return float(min(grid, key=nll_i)), float(min(grid, key=nll_t))

def full_on(golds, preds): return float(np.mean([g == p["frame"] for g, p in zip(golds, preds)]))

def model_bytes(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getbuffer().nbytes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=rd.DATASETS); ap.add_argument("--condition", default="base"); ap.add_argument("--locale", default="en-US")
    ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--scratch", action="store_true"); ap.add_argument("--sizes", default="1000,5000,full")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--device", default="cuda")
    ap.add_argument("--lrs", default=None); ap.add_argument("--epochs", default=None); ap.add_argument("--test", action="store_true")
    ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only"); ap.add_argument("--max_train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--save_model", action="store_true")
    a = ap.parse_args()
    lrs = [float(x) for x in (a.lrs or ("1e-4,3e-4,1e-3" if a.scratch else "2e-5,3e-5,5e-5")).split(",")]
    eps = [int(x) for x in (a.epochs or ("10,20,40" if a.scratch else "3,5,10")).split(",")]
    cond = rd.load_condition(a.dataset, a.condition, a.locale); items = cond["items"]
    if a.max_train: cond["train_order"] = cond["train_order"][:a.max_train]; keep = set(cond["train_order"]); cond["train"] = [i for i in cond["train"] if i in keep]
    sizes = a.sizes.split(",") if a.condition == "base" else ["full"]; nested = any(it["depth"] > 0 for it in items.values())
    name = f"{a.model}{' from scratch' if a.scratch else ' pretrained'}"; tag = f"{a.model.replace('/', '_')}_{'scratch' if a.scratch else 'pre'}"
    ev = report_real.eval_sets(cond, a.test, a.max_eval)
    print(f"bert_real: {name}; main repository {mainrepo.check_pinned()}; dataset {a.dataset} / condition {a.condition}; device {a.device}; sizes {sizes}; seeds {a.seeds}; grid lr {lrs} x epochs {eps}; "
          f"train {len(cond['train'])}, dev calibration {len(cond['dev_cal'])}; eval " + ", ".join(f"{k} {len(v)}" for k, v in ev.items()))
    print("regime (rule 24): real utterances, the dataset's own annotation; head-token frame; BIO + intent + nested heads, applied recursively; selection and thresholds on the dev calibration half; test only with --test")
    tok = AutoTokenizer.from_pretrained(a.model); cal_items = [items[i] for i in cond["dev_cal"]]; cal_g = [report_real.gold_flat(it) for it in cal_items]
    for size in sizes:
        n = len(cond["train"]) if size == "full" else int(size); train_ids = cond["train_order"][:n] if a.condition == "base" else cond["train"]
        nodes = [x for i in train_ids for x in rd.node_examples(items[i])]
        lab = Labels({t for x in nodes for t, *_ in x["slots"]}, {x["intent"] for x in nodes}); chosen = None; rows = {None: [], "temperature": []}
        ci = [items[i]["intent"] for i in train_ids]; maj = max(ci.count(x) for x in set(ci)) / len(ci)
        print(f"  size {size}: {len(train_ids)} utterances, {len(nodes)} intent nodes; {len(lab.types)} slot types, {len(lab.intents)} intents; chance: intent 1/{len(set(ci))} = {1 / len(set(ci)):.3f}, majority {maj:.3f}")
        for seed in range(a.seeds):
            best = None
            for lr in (lrs if seed == 0 else [chosen[0]]):
                for ep in (eps if seed == 0 else [chosen[1]]):
                    torch.manual_seed(seed); model = NodeTagger(a.model, a.scratch, len(lab.L), len(lab.intents)).to(a.device); t0 = time.time()
                    train(model, tok, nodes, lab, lr, ep, a.batch, a.device, seed, nested)
                    v = full_on(cal_g, predict(model, tok, cal_items, lab, a.device, nested=nested))
                    print(f"  size {size} seed {seed} lr={lr:g} epochs={ep} ({time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params) calibration-half FULL {v:.3f}", flush=True)
                    if best is None or v > best[0]: best = (v, lr, ep, model)
                    else: del model
            if seed == 0: chosen = (best[1], best[2]); print(f"  => size {size} chosen config: lr={chosen[0]:g}, epochs={chosen[1]} (FULL head frame on the dev calibration half, seed 0)")
            model = best[3]; Ts = fit_temperatures(model, tok, cal_items, lab, a.device)
            cpu = model.to("cpu"); sub = [items[i] for i in (ev.get("dev_rep") or next(iter(ev.values())))[:a.cost_n]]; t1 = time.time()
            for it in sub: predict(cpu, tok, [it], lab, "cpu", nested=nested)
            cost = {"params": sum(p.numel() for p in model.parameters()), "bytes": model_bytes(model), "cpu_ms_per_sentence": 1000 * (time.time() - t1) / max(1, len(sub))}
            model.to(a.device)
            for variant, T in ((None, (1.0, 1.0)), ("temperature", Ts)):
                cal_p = predict(model, tok, cal_items, lab, a.device, T, nested)
                preds = {k: predict(model, tok, [items[i] for i in ids], lab, a.device, T, nested) for k, ids in ev.items()}
                rows[variant].append(report_real.evaluate(name, tag, cond, seed, None if size == "full" else n, preds, cond["dev_cal"], cal_p, sorted(lab.types), variant,
                                                          extra={"cost": cost, "config": {"lr": chosen[0], "epochs": chosen[1]}, "temperatures": {"intent": Ts[0], "tags": Ts[1]}}))
            print(f"  seed {seed} size {size}: temperatures intent {Ts[0]:.2f} tags {Ts[1]:.2f}; cost {cost['params']/1e6:.1f}M params, {cost['bytes']/1e6:.0f} MB, {cost['cpu_ms_per_sentence']:.1f} ms/sentence on CPU", flush=True)
            if a.save_model and seed == 0:
                md = os.path.join(mainrepo.ROOT, "logs", "models", f"bert_real_{a.dataset}_{a.condition}_{tag}_{size}_seed0"); os.makedirs(md, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(md, "tagger.pt")); tok.save_pretrained(md)
                json.dump({"model": a.model, "scratch": a.scratch, "labels": lab.L, "types": lab.types, "intents": lab.intents, "nested": nested, "temperatures": Ts}, open(os.path.join(md, "config.json"), "w"))
            del model
        report_real.summarize(rows[None], name); report_real.summarize(rows["temperature"], name)
    metrics.checklist(a.seeds, "[x] the reader's scorer and rows [x] grid fixed in advance, chosen on seed 0 on the calibration half [x] test only with --test")
