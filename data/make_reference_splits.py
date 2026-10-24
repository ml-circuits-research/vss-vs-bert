"""make_reference_splits.py (run 4, part 0): frozen evaluation splits of the reference world (common/reference_world.py) for seeds 0..4,
written to data/reference_seed<s>.json; tests the tag -> frame mapping on the gold data (must be exact).
Run: python3 data/make_reference_splits.py [--seeds 5] [--D 1024]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import reference_world as rw, mainrepo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024)
    a = ap.parse_args(); print(f"make_reference_splits: main repository at {mainrepo.check_pinned()}; seeds {a.seeds}; D {a.D}")
    for seed in range(a.seeds):
        sp = rw.make_splits(seed, a.D); n = n_ok = 0
        for axis, items in sp["axes"].items():
            for it in items:
                n += 1; n_ok += rw.frame_from_tags(it["tokens"], it["tags"], it["frame"]["pred"]) == it["frame"]
        assert n == n_ok, f"seed {seed}: tags -> frame mapping failed on {n - n_ok} of {n}"
        with open(rw.path(seed), "w") as f: json.dump(sp, f)
        print(f"  seed {seed}: {n} sentences, tag->frame mapping exact; axes {({k: len(v) for k, v in sp['axes'].items()})}; held-out new-transition templates {sp['held_new']}; role-cue templates {sp['cues_ok']}")
    print("chance: roles 1/12 (new ids 1/6); predicate 1/12")
    print("exit check (run 4 part 0): gold tags map to gold frames at 1.000 on every seed: MET")
