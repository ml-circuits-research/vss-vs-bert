"""r17_tagger.py (deviation R17, settled by R18; RESULTS): the taggers' rows of R17 on the basic world, from models saved for every seed
(models/bert_basic.py --save_all), no training.
  item 1 (i), every tagger: rejected when the smallest, over the roles of the sentence's depth, of the largest candidate probability is
          below a threshold set at the 2% false-rejection point of the in-world calibration rows (a role with no candidate token: 0);
  item 1 (ii), a tagger trained with the out-of-world class (--oow_class): rejected when p(out-of-world | [CLS]) is above its 2%
          false-rejection point on the calibration rows (the 'neg' part was its training data, so it is not used again here);
  item 2: the residual analogue (R17): the share of content tokens (persons, objects, verbs, words outside the vocabulary) whose largest
          role probability (any label but O) is below 0.5; an analogue of the reader's residual, not the same quantity;
  item 3 (--depth_head): the depth accuracy of the [CLS] head and FULL at depth 3 with the depth it decides.
The depth used for items 1-2 is the level-opener count, as for the reader (common/decode.depth_from_openers). Sources reported apart.
Run: ./run.sh eval/r17_tagger.py --tag <results tag of the saved row, e.g. bert_bert-base-uncased_pre_wordpiece_r17rerun> [--seeds 5]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from transformers import AutoTokenizer
from common import basic_world as bwm, decode, report
from eval import metrics
from models import bert_basic as bb

FR_MAX = 0.02; VERBS = ("gives", "sells", "lends", "sends", "offers", "shows", "buys", "gets", "receives", "borrows", "accepts", "sees")

@torch.no_grad()
def forward(model, tok, items, device, batch=128):
    """per item: word x label probabilities (first sub-token) and the [CLS] outputs"""
    model.eval(); out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; ids, att, firsts = bb.encode_batch(tok, [it["tokens"] for it in chunk], device)
        prob = torch.softmax(model(ids, att), -1).cpu().numpy(); cls = model.last_cls
        poow = torch.softmax(model.oow_head(cls), -1)[:, 1].cpu().numpy() if model.oow_head is not None else None
        dep = model.depth_of(cls).cpu().numpy() if model.depth_head is not None else None
        for b, it in enumerate(chunk):
            p = np.stack([prob[b][j] if j is not None else np.zeros(len(bb.LABELS)) for j in firsts[b]])
            out.append({"p": p, "oow": None if poow is None else float(poow[b]), "depth": None if dep is None else int(dep[b])})
    return out

def accept_score(it, f, split, axis):
    """item 1 (i): min over the roles of the depth of the largest candidate probability"""
    s = []
    for r in bwm.roles_at(it["depth"]):
        cands = set(bwm.candidates_for(split, axis, r)); pos = [k for k, t in enumerate(it["tokens"]) if t in cands]
        s.append(max(f["p"][k, bb.LAB[r]] for k in pos) if pos else 0.0)
    return min(s)

def residual_analogue(it, f, split):
    cat = split["cat"]; content = [k for k, t in enumerate(it["tokens"]) if t not in cat or cat[t] in ("P", "O") or t in VERBS]
    if not content: return 0.0
    return float(np.mean([f["p"][k, 1:].max() < 0.5 for k in content]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    print(f"r17_tagger: main repository {mainrepo.check_pinned()}; device {a.device}; saved row {a.tag}; seeds {a.seeds}; false-rejection cap {FR_MAX}")
    print("regime (rule 24): basic world; depth of every sentence from the level openers for items 1-2; thresholds on calibration rows only (rule 25); out-of-world sets frozen before this run")
    rows = []
    for seed in range(a.seeds):
        md = os.path.join(mainrepo.ROOT, "logs", "models", f"{a.tag}_seed{seed}"); cfg = json.load(open(os.path.join(md, "config.json")))
        tok = AutoTokenizer.from_pretrained(md)
        model = bb.Tagger(cfg["model"], cfg["scratch"], cfg["n_tokens"], cue_tok=tok if cfg.get("cues") == "given" else None, oow=cfg.get("oow_class", False), depth_head=cfg.get("depth_head", "none")).to(a.device)
        model.load_state_dict(torch.load(os.path.join(md, "tagger.pt"), map_location=a.device))
        split = bwm.load(seed); oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")))
        est = lambda items: [dict(it, depth=decode.depth_from_openers(it["tokens"])) for it in items]
        sets = {"calibration": ("x", est(split["axes"]["calibration"]))}
        for ax in report.AXES:
            if split["axes"][ax]: sets[ax] = (ax, est(split["axes"][ax]))
        for src in ("random", "reference", "nonce"): sets[f"oow_{src}"] = ("x", est(oow["test"][src]))
        F = {k: forward(model, tok, items, a.device) for k, (_, items) in sets.items()}
        sc = {k: np.array([accept_score(it, f, split, ax) for it, f in zip(sets[k][1], F[k])]) for k in sets}
        t1 = float(np.quantile(sc["calibration"], FR_MAX))
        res = {"seed": seed, "tag": a.tag, "threshold_i": t1, "rejected": {"i": {}}, "false_rejection": {"i": {}}}
        inw = [k for k in sets if k != "calibration" and not k.startswith("oow_")]
        for k in sets:
            if k.startswith("oow_"): res["rejected"]["i"][k[4:]] = float(np.mean(sc[k] < t1))
            elif k != "calibration": res["false_rejection"]["i"][k] = float(np.mean(sc[k] < t1))
        res["false_rejection_pooled"] = {"i": float(np.mean(np.concatenate([sc[k] for k in inw]) < t1))}
        if model.oow_head is not None:
            po = {k: np.array([f["oow"] for f in F[k]]) for k in sets}; t2 = float(np.quantile(po["calibration"], 1 - FR_MAX)); res["threshold_ii"] = t2
            res["rejected"]["ii"] = {k[4:]: float(np.mean(po[k] > t2)) for k in sets if k.startswith("oow_")}
            res["false_rejection"]["ii"] = {k: float(np.mean(po[k] > t2)) for k in inw}
            res["false_rejection_pooled"]["ii"] = float(np.mean(np.concatenate([po[k] for k in inw]) > t2))
        res["residual_analogue_mean"] = {k: float(np.mean([residual_analogue(it, f, split) for it, f in zip(sets[k][1], F[k])])) for k in sets}
        if model.depth_head is not None:
            gold = {ax: split["axes"][ax] for ax in inw}
            res["depth_head_accuracy"] = {ax: float(np.mean([f["depth"] == it["depth"] for it, f in zip(gold[ax], F[ax])])) for ax in inw}
            items3 = [dict(it, depth=f["depth"]) for it, f in zip(split["axes"]["depth3_unseen"], F["depth3_unseen"])]
            p3 = bb.predict(model, tok, {"axes": {"depth3_unseen": items3}, "candidates": split["candidates"]}, "depth3_unseen", a.device)
            g3 = split["axes"]["depth3_unseen"]; full = lambda roles: float(np.mean([all(p["frame"].get(r) == it["frame"][r] for r in roles) for p, it in zip(p3, g3)]))
            res["full_depth3_head_depth"] = {"six": full(bwm.ROLES), "trained": full(["A", "B", "C", "X1", "X2"]), "inner": full(["A", "B", "C"])}
        print(f"  seed {seed}: threshold (i) {t1:.4f}; rejected (i) {res['rejected']['i']}; false rejection pooled {res['false_rejection_pooled']}")
        if "ii" in res["rejected"]: print(f"  seed {seed}: rejected (ii) {res['rejected']['ii']}")
        print(f"  seed {seed}: residual analogue means {res['residual_analogue_mean']}")
        if "depth_head_accuracy" in res: print(f"  seed {seed}: depth head accuracy {res['depth_head_accuracy']}; FULL depth 3 with the head's depth {res['full_depth3_head_depth']}")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_r17_{a.tag}_seed{seed}.json"), "w") as f: json.dump(res, f)
        rows.append(res); del model; torch.cuda.empty_cache() if a.device == "cuda" else None
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== R17, {a.tag}: MEANS over {len(rows)} seeds (per-seed list) ==")
    for v in [x for x in ("i", "ii") if x in rows[0]["rejected"]]:
        for src in ("random", "reference", "nonce"): print(f"  rejected ({v}) {src:10s} {ms([r['rejected'][v][src] for r in rows])}")
        print(f"  false rejection ({v}) pooled in-world report axes {ms([r['false_rejection_pooled'][v] for r in rows])}")
    print(f"  criterion (demo specification): rejected >= 0.90 per source with false rejections <= {FR_MAX}")
    if "depth_head_accuracy" in rows[0]:
        print(f"  depth head accuracy, depth3_unseen {ms([r['depth_head_accuracy']['depth3_unseen'] for r in rows])}")
        for k in ("six", "trained", "inner"): print(f"  FULL depth 3 with the head's depth, {k} {ms([r['full_depth3_head_depth'][k] for r in rows])}")
    metrics.checklist(len(rows), "[x] out-of-world sets frozen before the run [x] thresholds on calibration rows only [x] sources reported apart [x] no training")
