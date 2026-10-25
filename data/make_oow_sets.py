"""make_oow_sets.py (deviation R17 item 1, responses R17, 2026-10-25): the FROZEN out-of-world (OOW) sets of the basic world, one file per
seed, data/oow_seed<s>.json, written and committed before any model sees them. Three sources, as R17 lists them:
  (a) random:    token sequences drawn uniformly from the world's vocabulary, lengths drawn from the in-world test sentences;
  (b) reference: sentences of the reference world (data/reference_seed<s>.json, another legend) whose category sequence is NOT the
                 category sequence of any basic-world sentence (the two worlds share their whole vocabulary, so a reference sentence of
                 an in-world form is an in-world sentence and is excluded; questions.md Q13);
  (c) nonce:     in-world sentences (the basic split's 'seen' and 'depth3_unseen' items) with ONE content word (a filler or the verb)
                 replaced by a nonce word of no kind (letters only, not in the vocabulary).
In-world forms: the category sequences (ids -> P / O, every other word its category) of the basic world rendered with every template
index the splits use (training, held-out seen / unseen transitions, role cues) at depths 0-3; sources (a) and (b) drop any sequence whose
form is in that set, so every OOW item is out of the world by construction.
Two disjoint parts per seed: 'test' (frozen, reported) and 'neg' (negatives for R17 item 1 (ii): training of BERT's out-of-world class
and calibration of the reader's residual threshold), drawn from separate generators; no token sequence is in both.
Run: python3 data/make_oow_sets.py [--seeds 5] [--n 500]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import basic_world as bwm, mainrepo

LETTERS = "bcdfghjklmnprstvz"; VOWELS = "aeiou"

def form(tokens, cat):
    return tuple(cat.get(t, "?") for t in tokens)

def in_world_forms(bw, split, rng):
    T = sorted(set(split["train_templates"]) | set(bw["held_seen"]) | set(bw["held_new"]) | set(bw["cues_ok"] or []))
    F = set()
    for d in range(4):
        for it in bwm.draw(bw, rng, T, bw["trP"], bw["trO"], d, 3): F.add(form(it["tokens"], split["cat"]))
    for items in split["axes"].values():
        for it in items: F.add(form(it["tokens"], split["cat"]))
    return F

def nonce(rng, vocab):
    while True:
        w = "".join(rng.choice(list(LETTERS)) + rng.choice(list(VOWELS)) for _ in range(int(rng.integers(2, 4)))) + rng.choice(list(LETTERS))
        if w not in vocab: return w

def make_part(split, ref, F, rng, n, exclude):
    vocab = list(split["vocab"]); V = set(vocab); cat = split["cat"]; out = {"random": [], "reference": [], "nonce": []}; seen = set(exclude)
    lens = [len(it["tokens"]) for ax in ("seen", "new_ids", "depth1", "depth2_seen", "depth3_unseen") for it in split["axes"][ax]]
    while len(out["random"]) < n:
        toks = [vocab[i] for i in rng.integers(len(vocab), size=int(rng.choice(lens)))]
        if form(toks, cat) in F or tuple(toks) in seen: continue
        seen.add(tuple(toks)); out["random"].append({"tokens": toks, "source": "random"})
    pool = [it for items in ref["axes"].values() for it in items if form(it["tokens"], cat) not in F]
    for k in rng.permutation(len(pool)):
        if len(out["reference"]) >= n: break
        toks = pool[k]["tokens"]
        if tuple(toks) in seen: continue
        seen.add(tuple(toks)); out["reference"].append({"tokens": toks, "source": "reference", "ref_kind": pool[k].get("kind")})
    base = split["axes"]["seen"] + split["axes"]["depth3_unseen"]
    while len(out["nonce"]) < n:
        it = base[int(rng.integers(len(base)))]; toks = list(it["tokens"])
        content = [i for i, t in enumerate(toks) if cat[t] in ("P", "O") or t in ("gives", "sells", "lends", "sends", "offers", "shows", "buys", "gets", "receives", "borrows", "accepts", "sees")]
        i = int(rng.choice(content)); toks[i] = nonce(rng, V)
        if tuple(toks) in seen: continue
        seen.add(tuple(toks)); out["nonce"].append({"tokens": toks, "source": "nonce", "replaced_index": i, "depth": it["depth"]})
    return out, seen

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--n", type=int, default=500); a = ap.parse_args()
    print(f"make_oow_sets: main repository {mainrepo.check_pinned()}; seeds {a.seeds}; {a.n} items per source and part")
    for seed in range(a.seed_start, a.seed_start + a.seeds):   # --seed_start: R39 / R43, seeds 5-9 without rewriting 0-4
        split = bwm.load(seed); bw = bwm.build(seed, split["D"]); ref = json.load(open(os.path.join(mainrepo.ROOT, "data", f"reference_seed{seed}.json")))
        F = in_world_forms(bw, split, np.random.default_rng(70_000 + seed))
        n_ref_ok = sum(form(it["tokens"], split["cat"]) not in F for items in ref["axes"].values() for it in items)
        test, seen = make_part(split, ref, F, np.random.default_rng(71_000 + seed), a.n, set())
        neg, _ = make_part(split, ref, F, np.random.default_rng(72_000 + seed), a.n, seen)
        assert not ({tuple(x["tokens"]) for v in test.values() for x in v} & {tuple(x["tokens"]) for v in neg.values() for x in v})
        out = {"seed": seed, "n_in_world_forms": len(F), "reference_out_of_form": n_ref_ok, "test": test, "neg": neg, "main_repo_commit": mainrepo.check_pinned()}
        p = os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json"); json.dump(out, open(p, "w"))
        print(f"  seed {seed}: in-world forms {len(F)}; reference sentences out of the basic forms {n_ref_ok} of {sum(len(v) for v in ref['axes'].values())}; "
              f"test {[len(v) for v in test.values()]} neg {[len(v) for v in neg.values()]} (random, reference, nonce); disjoint; written {os.path.relpath(p, mainrepo.ROOT)}")
    print("exit check: every random / reference item has a form outside the basic world's; test and neg disjoint: MET")
