"""vorder_keys.py (responses R62 item 1, R60): two counts for the verb-order route.
(1) The size of the route at its USED keys: the distinct (side, verb kind, word kind, ordinal) tuples that occur, with valid = 1, in the
    TRAINING sentences of each seed's reader (the 30,000 sentences V8Vorder.train draws, reproduced draw for draw from the model's rng:
    the same constructor, the same set_vorder draw, then train()'s sentence loop), times D; beside it the allocated table vo
    (2 x K x K x nmax x D). No training, no GPU.
(2) The template-4 line of the growth table with k = 0 (R60): the template-4 cell of the new-formulation rows (slot accuracy and exact
    match over A, B, C) of the saved route-b readers, per-role rule and R46 decoder + one-to-one; template 4 was never in their training.
Writes logs/results_vorder_keys_<tag>_seed<s>.json.
Run: ./run.sh eval/vorder_keys.py --variant b --seed_start 0|5 --seeds 5"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode
from eval import metrics
from eval.reader_models import get_reader
from vss.v8_vorder import V8Vorder, verb_categories

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="b", choices=["a", "b"]); ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); a = ap.parse_args(); D = 1024
    cues = ["to", "by", "from"] if a.variant == "b" else []; fresh = a.seed_start >= 5
    row = ("vorderb" if a.variant == "b" else "vorder") + ("_fresh" if fresh else ""); tag = "reader_" + row
    print(f"vorder_keys (R62): main repository {mainrepo.check_pinned()}; variant {a.variant}; seeds {a.seed_start}-{a.seed_start + a.seeds - 1}; N {a.N}")
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        bw = bwm.build(seed, D)
        m = V8Vorder(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cpu", gate_init="closed"); m.set_F(bw["F"])
        m.set_vorder(verb_categories(m.cats), cue_cats=cues); w = m.w; sents = []
        for n in range(a.N):                                                   # V8Vorder.train's draw, verbatim
            d = int(m.rng.integers(bwm.DEPTH_TRAIN + 1)); ev = w.event(bw["trP"], bw["trO"], depth=d); sents.append(w.render(ev, m.rng.choice(list(v5.TRAIN_T))))
        keys = set()
        for i in range(0, len(sents), 512):
            cats, ids, mask = m.encode(sents[i:i + 512]); side, vcat, ordn, valid = m.vorder_index(cats, mask)
            for s_, v_, c_, o_ in zip(side[valid].tolist(), vcat[valid].tolist(), cats[valid].tolist(), ordn[valid].tolist()): keys.add((s_, v_, c_, o_))
        K = len(m.cats); alloc = int(np.prod(m.vo.shape)); used = len(keys) * D
        base = int(m.theta.numel() + m.logits.numel())
        # (2) template 4 with k = 0, the saved route reader
        mm, bw2, split, preds, how, chk = get_reader(row, seed, "cuda" if torch.cuda.is_available() else "cpu")
        its = [(j, it) for j, it in enumerate(split["axes"]["new_form_unseen_trans"]) if it["template"] == 4]
        P0 = preds["new_form_unseen_trans"]; P1 = decode.decode_frozen(mm, bw2["world"], {"axes": {"x": [it for _, it in its]}, "candidates": split["candidates"]}, "x", sentence_candidates=True)
        from eval.sentence_candidates import one_to_one
        P1 = [one_to_one(p, bwm.roles_at(it["depth"])) for p, (_, it) in zip(P1, its)]
        cell = lambda P: {"slot": float(np.mean([P[k]["frame"][r] == it["frame"][r] for k, (_, it) in enumerate(its) for r in "ABC"])), "exact": float(np.mean([all(P[k]["frame"][r] == it["frame"][r] for r in "ABC") for k, (_, it) in enumerate(its)]))}
        c0 = cell([P0[j] for j, _ in its]); c1 = cell(P1)
        r = {"seed": seed, "variant": a.variant, "K": K, "nmax": m.nmax, "D": D, "used_keys": len(keys), "used_params": used, "allocated_params": alloc, "base_params": base,
             "reader_params_used": base + used, "reader_params_allocated": base + alloc, "t4_k0_per_role": c0, "t4_k0_r46": c1, "n_t4": len(its), "model": how, "check": chk}
        print(f"  seed {seed}: {len(keys)} used keys of {2 * K * K * m.nmax} allocated; route {used / 1e6:.2f} M used / {alloc / 1e6:.2f} M allocated; reader {r['reader_params_used'] / 1e6:.2f} M (used) / {r['reader_params_allocated'] / 1e6:.2f} M (allocated); template 4, k = 0: per-role slot {c0['slot']:.3f} exact {c0['exact']:.3f}; R46 slot {c1['slot']:.3f} exact {c1['exact']:.3f} ({how}; {chk})", flush=True)
        json.dump(r, open(os.path.join(mainrepo.ROOT, "logs", f"results_vorder_keys_{tag}_seed{seed}.json"), "w")); rows.append(r)
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    print(f"\n== {tag}: used keys {[r['used_keys'] for r in rows]}; reader size {ms([r['reader_params_used'] / 1e6 for r in rows])} M at the used keys, {rows[0]['reader_params_allocated'] / 1e6:.2f} M allocated; BERT-tiny 4.39 M")
    print(f"  template 4, k = 0: per-role slot {ms([r['t4_k0_per_role']['slot'] for r in rows])}, exact {ms([r['t4_k0_per_role']['exact'] for r in rows])}; R46 slot {ms([r['t4_k0_r46']['slot'] for r in rows])}, exact {ms([r['t4_k0_r46']['exact'] for r in rows])}")
    metrics.checklist(len(rows), "[x] training sentences reproduced draw for draw [x] saved models, no training")
