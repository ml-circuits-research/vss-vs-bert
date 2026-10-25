"""make_growth_sets.py (responses R49 / R50, 2026-10-25): the FROZEN example sets of the legend-growth rows, one file per seed,
data/growth_seed<s>.json, written and committed before any model sees them. Two sets of 500 annotated sentences each (the k = 50 set
is the first 50 of the k = 500 set), drawn from the TRAINING identities with their own generator (rng 74_000 + seed), every sentence
different from every sentence of the seed's frozen split, out-of-world sets and planted sets:
  level3:    depth-3 sentences of the training templates (the training distribution's depth-3 sentences): the new legend question is the
             level-3 speaker (X3); the compositional case;
  template4: depth-0 sentences of template 4 ('{A} V {C} {B} .', the double object of the new-formulation axis): the non-compositional
             case, a form never in training.
Each item: tokens, ev, frame, tags, depth, template, kind (as the frozen splits).
Run: python3 data/make_growth_sets.py [--seeds 10] [--seed_start 0] [--n 500]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import basic_world as bwm, mainrepo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--n", type=int, default=500); a = ap.parse_args()
    print(f"make_growth_sets: main repository {mainrepo.check_pinned()}; seeds {a.seed_start}-{a.seed_start + a.seeds - 1}; {a.n} per set")
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, split["D"]); rng = np.random.default_rng(74_000 + seed)
        taken = {tuple(it["tokens"]) for v in split["axes"].values() for it in v}
        oow = json.load(open(os.path.join(mainrepo.ROOT, "data", f"oow_seed{seed}.json")))
        taken |= {tuple(x["tokens"]) for part in ("test", "neg") for v in oow[part].values() for x in v}
        pp = os.path.join(mainrepo.ROOT, "data", f"planted_seed{seed}.json")
        if os.path.exists(pp): taken |= {tuple(x["tokens"]) for v in json.load(open(pp)).values() if isinstance(v, list) for x in v}
        out = {"seed": seed, "main_repo_commit": mainrepo.check_pinned()}
        for name, templates, depth in (("level3", list(bw["TRAIN_T"]), 3), ("template4", [4], 0)):
            items, seen = [], set(taken)
            while len(items) < a.n:
                for it in bwm.draw(bw, rng, templates, bw["trP"], bw["trO"], depth, 1):
                    t = tuple(it["tokens"])
                    if t in seen: continue
                    seen.add(t); it["frame"] = bwm.frame_of(it["ev"]); it["tags"] = bwm.tags_of(it["tokens"], it["ev"])
                    assert bwm.frame_from_tags(it["tokens"], it["tags"]) == it["frame"]; items.append(it)
            order = rng.permutation(len(items))[:a.n]; out[name] = [items[i] for i in order]
        p = os.path.join(mainrepo.ROOT, "data", f"growth_seed{seed}.json"); json.dump(out, open(p, "w"))
        print(f"  seed {seed}: level3 {len(out['level3'])} (templates {sorted({it['template'] for it in out['level3']})}), template4 {len(out['template4'])}; disjoint from the frozen split, out-of-world and planted sets; tag->frame exact; written {os.path.relpath(p, mainrepo.ROOT)}")
