"""reference_world.py (run 4): the reference world of RESULTS 10b-quinquies, built per seed EXACTLY as the main repository's
experiment/v16_reference.py builds it (RefWorld rng(seed), identities rng(300+seed), RefRoleWorld rng(100+seed): predicate role = surface
verb lemma, occupancy marker m, recipient-less forms OMIT_C at share p_omit = 0.2 in training; relation-marker gates to / by / from GIVEN
open, `--rel_gates open`), and the FROZEN evaluation splits every model is scored on (INSTRUCTIONS section 5).

Gold frame of a sentence: A (agent), B (object), C (recipient; None when the form omits it), X1..Xd (speakers, outermost first) and
'pred' (the surface verb's lemma, give / receive / sell ...; the event type follows by the synonym table EVENT). FULL frame (INSTRUCTIONS
section 3, as the paper states it for 10b-quinquies): A, B, C (or its absence), every speaker and the predicate exact. Also reported:
'roles FULL' = A, B, C only, which is the column v16_reference.py prints, for the bridge to the main repository's numbers.

Axes: seen (training templates, depth 0) | new_ids (held-out persons / objects) | role_cues (new formulations with all role cues seen) |
unseen_trans (new formulations with unseen transitions) | depth2_seen | depth3_unseen (training templates; per-template breakdown in the
items: passives T2, T3, T5, T14; the fronted-recipient comma form T9) | omit (recipient not given: the occupancy 'not given' case) |
unknown_recipient (C a held-out person, A and B training identities: the occupancy 'unknown filler' case) | calibration (training
distribution incl. the omit share; thresholds only, rule 25)."""
import json, os
import numpy as np
from . import mainrepo

ROLES = ["A", "B", "C", "X1", "X2", "X3"]
N_ID = 18; P_OMIT = 0.2; DEPTH_TRAIN = 2
AXES = ["seen", "new_ids", "role_cues", "unseen_trans", "depth2_seen", "depth3_unseen", "omit", "unknown_recipient"]

def import_v16():
    v5, v7 = mainrepo.import_experiment()
    import v16_reference as v16
    return v5, v7, v16

def build(seed, D=1024):
    v5, v7, v16 = import_v16()
    w = v16.RefWorld(N_ID, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
    world = v16.RefRoleWorld(F, w, np.random.default_rng(100 + seed))
    w0 = v16.RefWorld(N_ID, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    return dict(w=w, world=world, F=F, trP=trP, teP=teP, trO=trO, teO=teO, held_seen=held_seen, held_new=held_new, cues_ok=cues_ok,
                TRAIN_T=v5.TRAIN_T, verbs=sorted(set(v16.LEMMA.values())), EVENT=v16.EVENT, LEMMA=v16.LEMMA, n_omit=len(v16.OMIT_C))

def frame_of(ev, toks, has_C, bw):
    f = {"A": ev[1], "B": ev[2], "C": ev[3] if has_C else None}
    for k in range(3): f[f"X{k+1}"] = ev[4 + k] if 4 + k < len(ev) else None
    f["pred"] = bw["w"].surface_verb(toks)
    return f

def tags_of(tokens, frame):
    inv = {v: r for r, v in frame.items() if v is not None and r != "pred"}
    return [inv.get(t, "O") for t in tokens]

def frame_from_tags(tokens, tags, pred):
    f = {r: None for r in ROLES}
    for t, g in zip(tokens, tags):
        if g != "O" and f.get(g) is None: f[g] = t
    f["pred"] = pred; return f

def roles_at(depth): return ["A", "B", "C"] + [f"X{k+1}" for k in range(depth)] + ["pred"]

def _item(bw, ev, toks, has_C, depth, template):
    fr = frame_of(ev, toks, has_C, bw)
    return {"tokens": toks, "ev": list(ev), "depth": depth, "template": template, "has_C": bool(has_C), "frame": fr, "tags": tags_of(toks, fr)}

def draw(bw, rng, templates, persons, objects, depth, n, omit=False, recipients=None):
    w = bw["w"]; items = []; saved = w.rng; w.rng = rng
    try:
        for kind in w.frames:
            for t in templates:
                for _ in range(n):
                    ev = (kind,) + w.event(persons, objects, depth=depth)[1:]
                    if recipients is not None:
                        c = recipients[int(rng.integers(len(recipients)))]
                        if c in ev[1:2] + ev[4:]: continue
                        ev = ev[:3] + (c,) + ev[4:]
                    if omit: items.append(_item(bw, ev, w.render_omit(ev, t), False, depth, f"omit{t}"))
                    else: items.append(_item(bw, ev, w.render(ev, t), True, depth, int(t)))
    finally: w.rng = saved
    return items

def make_splits(seed, D=1024, n_cal=2000):
    bw = build(seed, D); T = bw["TRAIN_T"]; r = lambda k: np.random.default_rng(20_000 * (seed + 1) + k)
    axes = {
        "seen": draw(bw, r(1), T, bw["trP"], bw["trO"], 0, 20),
        "new_ids": draw(bw, r(2), T, bw["teP"], bw["teO"], 0, 20),
        "role_cues": draw(bw, r(3), bw["cues_ok"], bw["trP"], bw["trO"], 0, 30) if bw["cues_ok"] else [],
        "unseen_trans": draw(bw, r(4), bw["held_new"], bw["trP"], bw["trO"], 0, 20),
        "depth2_seen": draw(bw, r(5), T, bw["trP"], bw["trO"], 2, 15),
        "depth3_unseen": draw(bw, r(6), T, bw["trP"], bw["trO"], 3, 15),
        "omit": draw(bw, r(7), list(range(bw["n_omit"])), bw["trP"], bw["trO"], 0, 20, omit=True),
        "unknown_recipient": draw(bw, r(8), [0], bw["trP"], bw["trO"], 0, 20, recipients=bw["teP"]),
    }
    cal = []; rc = r(9); w = bw["w"]; saved = w.rng; w.rng = rc
    try:
        for _ in range(n_cal):           # the training distribution of v16 (depths 0-2, omit share 0.2), drawn from its own generator
            d = int(rc.integers(DEPTH_TRAIN + 1)); ev = w.event(bw["trP"], bw["trO"], depth=d)
            if rc.random() < P_OMIT: k = int(rc.integers(bw["n_omit"])); cal.append(_item(bw, ev, w.render_omit(ev, k), False, d, f"omit{k}"))
            else: t = int(rc.choice(list(T))); cal.append(_item(bw, ev, w.render(ev, t), True, d, t))
    finally: w.rng = saved
    axes["calibration"] = cal
    cands = {"persons_train": bw["trP"], "persons_test": bw["teP"], "objects_train": bw["trO"], "objects_test": bw["teO"], "verbs": bw["verbs"]}
    return {"seed": seed, "D": D, "n_id": N_ID, "depth_train": DEPTH_TRAIN, "p_omit": P_OMIT, "vocab": bw["w"].vocab, "cat": bw["w"].cat,
            "train_templates": [int(t) for t in T], "held_new": [int(t) for t in bw["held_new"]], "cues_ok": [int(t) for t in bw["cues_ok"]],
            "event_table": bw["EVENT"], "candidates": cands, "axes": axes, "main_repo_commit": mainrepo.check_pinned()}

def candidates_for(split, axis, role):
    c = split["candidates"]
    if role == "pred": return c["verbs"]
    if role == "B": return c["objects_test"] if axis == "new_ids" else c["objects_train"]
    if axis == "new_ids": return c["persons_test"]
    if axis == "unknown_recipient" and role == "C": return c["persons_train"] + c["persons_test"]
    return c["persons_train"]

def path(seed): return os.path.join(mainrepo.ROOT, "data", f"reference_seed{seed}.json")
def load(seed):
    with open(path(seed)) as f: return json.load(f)
