"""make_planted_sets.py (deviation R26, responses R26, 2026-10-25; questions Q19 default): the FROZEN planted-questions sets of the
basic world for the query-emission row (after JP 1), one file per seed, data/planted_seed<s>.json, committed before any model sees them.
  unsupported: a basic-world sentence (depths 0-3, training templates and identities) whose final '.' is replaced by ', <id> .', <id> a
               training person or object not in the sentence: no question of the legend asks for it; the gold frame is unchanged and
               the gold of that token is 'unsupported';
  undecided:   a basic-world sentence with a second identity of the same class inserted right after the filler of one role (A, B or C,
               at random): two candidates in the same position class; the gold of that role is 'undecided', the others keep their values.
The out-of-world sets (data/oow_seed<s>.json) serve 'unknown'; the frozen splits serve the values. Every planted sentence is drawn from
its own generator (rng 73_000 + seed) and differs from every sentence of the frozen splits and of the out-of-world sets.
Run: python3 data/make_planted_sets.py [--seeds 5] [--seed_start 0] [--n 300]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import basic_world as bwm, mainrepo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--n", type=int, default=300); a = ap.parse_args()
    print(f"make_planted_sets: main repository {mainrepo.check_pinned()}; seeds {a.seeds}; {a.n} per kind")
    for seed in range(a.seed_start, a.seed_start + a.seeds):   # --seed_start: seeds 5-9 for R54's claim (R39)
        split = bwm.load(seed); bw = bwm.build(seed, split["D"]); rng = np.random.default_rng(73_000 + seed)
        oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")))
        taken = {tuple(it["tokens"]) for v in split["axes"].values() for it in v} | {tuple(x["tokens"]) for part in ("test", "neg") for v in oow[part].values() for x in v}
        gp = os.path.join(mainrepo.ROOT, "data", f"growth_seed{seed}.json")   # 2026-10-25 (R54, seeds 5-9): also disjoint from the growth sets
        if seed >= 5 and os.path.exists(gp): taken |= {tuple(x["tokens"]) for k, v in json.load(open(gp)).items() if isinstance(v, list) for x in v}
        P, O = list(bw["trP"]), list(bw["trO"]); out = {"unsupported": [], "undecided": []}
        while len(out["unsupported"]) < a.n or len(out["undecided"]) < a.n:
            d = int(rng.integers(4)); it = bwm.draw(bw, rng, [int(rng.choice(list(bw["TRAIN_T"])))], bw["trP"], bw["trO"], d, 1)[0]
            it["frame"] = bwm.frame_of(it["ev"]); toks = list(it["tokens"])
            if len(out["unsupported"]) < a.n and toks[-1] == "." and rng.random() < 0.5:
                pool = [x for x in P + O if x not in toks]; x = pool[int(rng.integers(len(pool)))]; t2 = toks[:-1] + [",", x, "."]
                if tuple(t2) not in taken:
                    taken.add(tuple(t2)); out["unsupported"].append({"tokens": t2, "frame": it["frame"], "depth": d, "planted_index": len(t2) - 2, "gold_query": "unsupported"})
            elif len(out["undecided"]) < a.n:
                r = ["A", "B", "C"][int(rng.integers(3))]; f = it["frame"][r]; k = toks.index(f)
                pool = [x for x in (O if r == "B" else P) if x not in toks]; x = pool[int(rng.integers(len(pool)))]; t2 = toks[:k + 1] + [x] + toks[k + 1:]
                if tuple(t2) not in taken:
                    taken.add(tuple(t2)); out["undecided"].append({"tokens": t2, "frame": it["frame"], "depth": d, "role": r, "candidates": [f, x], "gold_query": "undecided"})
        p = os.path.join(mainrepo.ROOT, "data", f"planted_seed{seed}.json"); json.dump({"seed": seed, **out, "main_repo_commit": mainrepo.check_pinned()}, open(p, "w"))
        print(f"  seed {seed}: unsupported {len(out['unsupported'])}, undecided {len(out['undecided'])}; e.g. {' '.join(out['unsupported'][0]['tokens'])} | {' '.join(out['undecided'][0]['tokens'])}; written {os.path.relpath(p, mainrepo.ROOT)}")
    print("exit check: every planted sentence differs from the frozen splits and the out-of-world sets: MET")
