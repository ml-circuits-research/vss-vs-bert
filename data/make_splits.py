"""make_splits.py (run 0): write the frozen evaluation splits of the basic world for seeds 0..4 to data/basic_seed<s>.json, and test the
tag -> frame mapping on the gold data (INSTRUCTIONS run 0 exit check: gold tags must map back to gold frames at 1.000).
Run: python3 data/make_splits.py [--seeds 5] [--D 1024]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import basic_world as bwm, mainrepo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024)
    a = ap.parse_args(); print(f"make_splits: main repository at {mainrepo.check_pinned()}; seeds {a.seeds}; D {a.D}")
    for seed in range(a.seeds):
        sp = bwm.make_splits(seed, a.D); n_ok = n = 0
        for axis, items in sp["axes"].items():
            for it in items:
                it["frame"] = bwm.frame_of(it["ev"]); it["tags"] = bwm.tags_of(it["tokens"], it["ev"])
                back = bwm.frame_from_tags(it["tokens"], it["tags"]); n += 1; n_ok += (back == it["frame"])
        assert n_ok == n, f"seed {seed}: tags -> frame mapping failed on {n - n_ok} of {n} sentences"
        with open(bwm.path(seed), "w") as f: json.dump(sp, f)
        sizes = {k: len(v) for k, v in sp["axes"].items()}
        print(f"  seed {seed}: {n} sentences, tag->frame mapping exact on all; axes {sizes}; vocabulary {len(sp['vocab'])}; written {os.path.relpath(bwm.path(seed), mainrepo.ROOT)}")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167 (restricted candidates, as decode_nested)")
    print("exit check (run 0): gold tags map to gold frames at 1.000 on every seed: MET")
