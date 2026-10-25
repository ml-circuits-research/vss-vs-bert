"""basic_world.py: the basic world of RESULTS 7b / 7b-bis (v5_scale.BigWorld + v7_nested.NestedRoleWorld), built per seed EXACTLY as the
main repository's scripts build it (same generator seeds: world rng(seed), identities rng(300+seed), role vectors rng(100+seed), training
data rng(200+seed)), and the FROZEN evaluation splits every model is scored on (INSTRUCTIONS section 5).

Frozen splits: the main repository's decode_nested draws fresh test sentences at decode time; here the test sentences of every axis are
drawn ONCE per seed from their own generator, written to data/basic_seed<s>.json and read back by every model, so that the reader, the
7b transformer and BERT are scored on the same sentences. The gold frame of a sentence is the event: A (agent, a person), B (object),
C (recipient, a person), X1..Xd (speakers, outermost first). Gold tags for the tagging head: a token is tagged with its role when it is
that role's filler, O otherwise; a filler is one token in this world, so tags map back to frames exactly (tested in data/make_splits.py).

Axes (the columns of RESULTS 7b): seen | new ids | new form., seen trans. | new form., UNSEEN trans. | new form., ROLE CUES seen |
depth 1 | depth 2 (seen) | depth 3 (UNSEEN). Plus calibration rows (training distribution, never reported) for thresholds (rule 25)."""
import json, os
import numpy as np
from . import mainrepo

ROLES = ["A", "B", "C", "X1", "X2", "X3"]
N_ID = 18          # persons and objects per class; 12 train + 6 held out, as v9 / v22
DEPTH_TRAIN = 2    # training depths 0-2, test depth 3 (rule 29: the hop seen at depth 2)

def build(seed, D=1024):
    """the world of one seed: (w, world, F, trP, teP, trO, teO, held_seen, held_new, cues_ok)"""
    v5, v7 = mainrepo.import_experiment()
    w = v5.BigWorld(N_ID, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
    world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
    w0 = v5.BigWorld(N_ID, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    return dict(w=w, world=world, F=F, trP=trP, teP=teP, trO=trO, teO=teO, held_seen=held_seen, held_new=held_new, cues_ok=cues_ok, TRAIN_T=v5.TRAIN_T)

def frame_of(ev):
    """gold frame of an event: role -> filler (None when the role is absent at this depth)"""
    f = {"A": ev[1], "B": ev[2], "C": ev[3]}
    for k in range(3): f[f"X{k+1}"] = ev[4 + k] if 4 + k < len(ev) else None
    return f

def tags_of(tokens, ev):
    """gold tags: the role of each token that is a filler, O otherwise"""
    inv = {v: r for r, v in frame_of(ev).items() if v is not None}
    return [inv.get(t, "O") for t in tokens]

def frame_from_tags(tokens, tags):
    """the fixed mapping tags -> frame (INSTRUCTIONS section 3): the first token tagged r is r's filler; untagged roles are not given"""
    f = {r: None for r in ROLES}
    for t, g in zip(tokens, tags):
        if g != "O" and f.get(g) is None: f[g] = t
    return f

def draw(bw, rng, templates, persons, objects, depth, n):
    """n sentences per (frame kind, template) at a depth, as decode_nested draws them, but from a given rng"""
    w = bw["w"]; items = []
    saved = w.rng; w.rng = rng
    try:
        for kind in w.frames:
            for t in templates:
                for _ in range(n):
                    ev = (kind,) + w.event(persons, objects, depth=depth)[1:]
                    toks = w.render(ev, t); items.append({"tokens": toks, "ev": list(ev), "depth": depth, "template": int(t), "kind": kind})
    finally: w.rng = saved
    return items

def make_splits(seed, D=1024, n_cal=2000):
    """the frozen evaluation splits of one seed; the candidate lists per role are part of the file (decoding is restricted to them, as
    decode_nested restricts it: persons for A, C, X; objects for B; held-out ones on the new-identities axis)"""
    bw = build(seed, D); T = bw["TRAIN_T"]; r = lambda k: np.random.default_rng(10_000 * (seed + 1) + k)
    axes = {
        "seen": draw(bw, r(1), T, bw["trP"], bw["trO"], 0, 20),
        "new_ids": draw(bw, r(2), T, bw["teP"], bw["teO"], 0, 20),
        "new_form_seen_trans": draw(bw, r(3), bw["held_seen"], bw["trP"], bw["trO"], 0, 30),
        "new_form_unseen_trans": draw(bw, r(4), bw["held_new"], bw["trP"], bw["trO"], 0, 30),
        "new_form_role_cues_seen": draw(bw, r(5), bw["cues_ok"], bw["trP"], bw["trO"], 0, 30) if bw["cues_ok"] else [],
        "depth1": draw(bw, r(6), T, bw["trP"], bw["trO"], 1, 15),
        "depth2_seen": draw(bw, r(7), T, bw["trP"], bw["trO"], 2, 15),
        "depth3_unseen": draw(bw, r(8), T, bw["trP"], bw["trO"], 3, 15),
    }
    # calibration rows: training distribution (depths 0-2, training templates and identities), never reported (rule 25)
    cal = []; rc = r(9)
    for _ in range(n_cal):
        d = int(rc.integers(DEPTH_TRAIN + 1)); cal.extend(draw(bw, rc, [int(rc.choice(list(T)))], bw["trP"], bw["trO"], d, 1)[:1])
    axes["calibration"] = cal
    cands = {"persons_train": bw["trP"], "persons_test": bw["teP"], "objects_train": bw["trO"], "objects_test": bw["teO"]}
    return {"seed": seed, "D": D, "n_id": N_ID, "depth_train": DEPTH_TRAIN, "vocab": bw["w"].vocab, "cat": bw["w"].cat, "train_templates": [int(t) for t in T],
            "candidates": cands, "axes": axes, "main_repo_commit": mainrepo.check_pinned()}

def candidates_for(split, axis, role):
    """the candidate fillers of a role on an axis: held-out identities on new_ids, training identities elsewhere"""
    c = split["candidates"]; new = axis == "new_ids"
    if role == "B": return c["objects_test"] if new else c["objects_train"]
    return c["persons_test"] if new else c["persons_train"]

def train_max_words(w):
    """R10 item 2a: the length in words of the longest sentence the training distribution can produce (training templates, depths 0-2,
    every combination of outer reporting forms); word lengths do not depend on the identities, so this is exact for every model's training
    draws. A word position >= this value is never reached in training."""
    v5, v7 = mainrepo.import_experiment(); import itertools
    ev = ("GIVE" if "GIVE" in w.frames else next(iter(w.frames)), w.persons[0], w.objects[0], w.persons[1]); best = 0
    for d in range(DEPTH_TRAIN + 1):
        e = ev + tuple(w.persons[2:2 + d]) + (None,) * (3 - d)
        for t in v5.TRAIN_T:
            for outer in itertools.product(range(len(v5.OUTER)), repeat=d): best = max(best, len(w.render(e, t, outer=list(outer))))
    return best

def word_positions(item):
    """the word index of each role's gold filler (fillers are distinct and occur once)"""
    return {r: (item["tokens"].index(f) if f is not None and f in item["tokens"] else None) for r, f in item["frame"].items()}

def roles_at(depth): return ["A", "B", "C"] + [f"X{k+1}" for k in range(depth)]

def path(seed): return os.path.join(mainrepo.ROOT, "data", f"basic_seed{seed}.json")
def load(seed):
    with open(path(seed)) as f: return json.load(f)
