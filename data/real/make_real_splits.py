"""make_real_splits.py (run 8a / 9a / 10a): the frozen split CONDITIONS of a real-data source, written to data/real/splits_<dataset>.json
BEFORE any training. The file holds ids only, the seed and the rule of every split, and the counts; never text (the author's rule: no
third-party data in the repository). Every model reads its rows through common/real_data.load_condition.

Partitions. MASSIVE and TOPv2: the dataset's own train / dev (TOPv2 'eval') / test. SNIPS: test = the validate_<intent>.json files
(100 per intent, the benchmark's held-out queries); dev = 100 per intent drawn from train_<intent>_full.json by
np.random.default_rng(0).permutation within each intent (file order); train = the rest of train_full. Dev is cut into the CALIBRATION
half (even positions in id order: thresholds, model selection; rule 25) and the REPORT half (odd positions). The test is read once.

Conditions (each: train, dev_cal, eval sets; an eval set whose name starts with 'test' is scored only with --test):
  base               train / dev halves / test; train_order = train permuted by rng(0): a training size n takes its first n (as
                     common/wikisql_data.py); eval: dev_rep, dev_rep:new_slot_values, dev_rep:unseen_template, and the same three on test.
                     new_slot_values = the items with at least one head token never present in the training utterances (new_roles lists,
                     per item, the frame paths concerned; scored on those paths only). unseen_template = items whose delexicalised template
                     (top-level slot spans replaced by their type) never occurs in train (the natural held-out formulations; no retraining).
  heldout_templates  templates with at least --min_template_items items over all partitions are held out when h01(0, 'template', t) <
                     --template_share; their items leave train and dev_cal; eval 'heldout_templates' = their items from every partition
                     (never trained, never used for a threshold), 'dev_rep:seen_templates' = the rest of dev_rep.
  ood_intents        --n_ood intents (top-level intents with at least 50 training items, the --n_ood lowest h01(0, 'ood', intent)) leave
                     train and dev_cal entirely (TOP: every item in which the intent occurs at any level); eval 'ood:dev_rep' and
                     'ood:test' = their items, 'dev_rep:in_domain' and 'test:in_domain' = the rest.
  depth (TOP only)   train and dev halves restricted to depth <= --train_depth (1 by default); eval 'depth<=1:dev_rep', 'depth2_unseen' =
                     every depth-2 item of every partition (none is trained), 'depth2_unseen:hops_seen' = those whose every intent and slot type occurs
                     in the depth-split training rows (main repository rule 29: an unseen depth whose hops are seen), 'depth3plus_unseen' when any. Marked unsupported (with the
                     counts) when depth 2 has fewer than --min_depth_items items: no unseen-depth split is faked.
Run: python3 data/real/make_real_splits.py snips|massive|topv2 [--locale en-US] [--domains navigation,event,reminder]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from common import real_data as rd

def stats_of(items):
    n = len(items); slots = sum(len(i["slots"]) for i in items)
    return {"items": n, "intents": len({i["intent"] for i in items}), "slot_types": len({t for i in items for t, *_ in i["slots"]}),
            "slots": slots, "items_with_a_repeated_slot_type": sum(rd.repeated_types(i["tree"]) > 0 for i in items),
            "spans_dropped_by_first_span_rule": sum(rd.repeated_types(i["tree"]) for i in items),
            "mean_tokens": float(np.mean([len(i["tokens"]) for i in items])) if n else 0.0,
            "items_without_slots": sum(not i["slots"] for i in items), "depth_counts": dict(sorted(collections.Counter(i["depth"] for i in items).items()))}

def all_types(tree): return {sl["type"] for sl in tree["slots"]} | {x for sl in tree["slots"] if sl["child"] for x in all_types(sl["child"])}

def new_roles_of(ids, by, vocab):
    out = {}
    for i in ids:
        r = rd.new_value_roles(by[i], vocab)
        if r: out[i] = r
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=rd.DATASETS); ap.add_argument("--locale", default="en-US")
    ap.add_argument("--domains", default="navigation,event,reminder", help="TOPv2 domains (stated default: the domains with nested intents)")
    ap.add_argument("--template_share", type=float, default=0.15); ap.add_argument("--min_template_items", type=int, default=3)
    ap.add_argument("--n_ood", type=int, default=None, help="default: 6 for MASSIVE, 1 for SNIPS, 3 for TOPv2"); ap.add_argument("--train_depth", type=int, default=1)
    ap.add_argument("--min_depth_items", type=int, default=30); ap.add_argument("--snips_dev_per_intent", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); doms = a.domains.split(",") if a.dataset == "topv2" else None
    items, load_stats = rd.load(a.dataset, a.locale, doms); by = {i["id"]: i for i in items}
    if a.dataset == "snips":
        tr_full = [i for i in items if i["partition"] == "train_full"]; dev = []
        for I in rd.SNIPS_INTENTS:
            xs = [i["id"] for i in tr_full if i["intent"] == I]; perm = np.random.default_rng(0).permutation(len(xs))
            dev += [xs[k] for k in sorted(perm[:a.snips_dev_per_intent])]
        dset = set(dev); train = [i["id"] for i in tr_full if i["id"] not in dset]
        for i in tr_full: i["partition"] = "dev" if i["id"] in dset else "train"
    else:
        train = [i["id"] for i in items if i["partition"] == "train"]; dev = [i["id"] for i in items if i["partition"] == "dev"]
    test = [i["id"] for i in items if i["partition"] == "test"]
    dev_cal, dev_rep = dev[0::2], dev[1::2]
    vocab = {t for i in train for t in by[i]["tokens"]}; ttrain = {rd.template(by[i]) for i in train}
    unseen_t = lambda ids: [i for i in ids if rd.template(by[i]) not in ttrain]
    train_order = [train[k] for k in np.random.default_rng(0).permutation(len(train))]
    C = {}
    nr_rep, nr_test = new_roles_of(dev_rep, by, vocab), new_roles_of(test, by, vocab)
    C["base"] = {"rule": "the dataset's partitions; dev even / odd = calibration / report; train_order = train permuted by np.random.default_rng(0)",
                 "train": train, "dev_cal": dev_cal, "train_order": train_order,
                 "eval": {"dev_rep": dev_rep, "dev_rep:new_slot_values": sorted(nr_rep, key=dev_rep.index), "dev_rep:unseen_template": unseen_t(dev_rep),
                          "test": test, "test:new_slot_values": sorted(nr_test, key=test.index), "test:unseen_template": unseen_t(test)},
                 "new_roles": {**nr_rep, **nr_test}}
    tc = collections.Counter(rd.template(i) for i in items)
    held = {t for t, n in tc.items() if n >= a.min_template_items and rd.h01(0, "template", t) < a.template_share}
    isheld = lambda i: rd.template(by[i]) in held
    C["heldout_templates"] = {"rule": f"templates with >= {a.min_template_items} items over all partitions, held out when h01(0, 'template', template) < {a.template_share}",
                              "n_templates_held": len(held), "n_templates_eligible": sum(1 for n in tc.values() if n >= a.min_template_items),
                              "train": [i for i in train if not isheld(i)], "dev_cal": [i for i in dev_cal if not isheld(i)],
                              "eval": {"heldout_templates": [i["id"] for i in items if isheld(i["id"])], "dev_rep:seen_templates": [i for i in dev_rep if not isheld(i)]}}
    n_ood = a.n_ood or {"massive": 6, "snips": 1, "topv2": 3}[a.dataset]
    tri = collections.Counter(by[i]["intent"] for i in train)
    H = sorted(sorted((I for I, n in tri.items() if n >= 50), key=lambda I: rd.h01(0, "ood", I))[:n_ood])
    isood = lambda i: bool(rd.all_intents(by[i]["tree"]) & set(H))
    C["ood_intents"] = {"rule": f"the {n_ood} intents with >= 50 training items and the lowest h01(0, 'ood', intent) leave train and dev_cal (every item in which they occur at any level)",
                        "held_intents": H, "train": [i for i in train if not isood(i)], "dev_cal": [i for i in dev_cal if not isood(i)],
                        "eval": {"dev_rep:in_domain": [i for i in dev_rep if not isood(i)], "ood:dev_rep": [i for i in dev_rep if isood(i)],
                                 "test:in_domain": [i for i in test if not isood(i)], "ood:test": [i for i in test if isood(i)]}}
    if a.dataset == "topv2":
        D = lambda ids, f: [i for i in ids if f(by[i]["depth"])]
        d2 = [i["id"] for i in items if i["depth"] == a.train_depth + 1]; d3 = [i["id"] for i in items if i["depth"] > a.train_depth + 1]
        counts = {p: dict(sorted(collections.Counter(i["depth"] for i in items if i["partition"] == p).items())) for p in ("train", "dev", "test")}
        c = {"rule": f"train and dev halves at depth <= {a.train_depth}; depth {a.train_depth + 1} items of every partition never trained", "depth_counts_by_partition": counts,
             "train": D(train, lambda d: d <= a.train_depth), "dev_cal": D(dev_cal, lambda d: d <= a.train_depth),
             "eval": {f"depth<={a.train_depth}:dev_rep": D(dev_rep, lambda d: d <= a.train_depth), f"depth{a.train_depth + 1}_unseen": d2,
                      f"test:depth<={a.train_depth}": D(test, lambda d: d <= a.train_depth)}}
        if d3: c["eval"][f"depth{a.train_depth + 2}plus_unseen"] = d3
        tr_int = {x for i in c["train"] for x in rd.all_intents(by[i]["tree"])}; tr_typ = {x for i in c["train"] for x in all_types(by[i]["tree"])}
        c["eval"][f"depth{a.train_depth + 1}_unseen:hops_seen"] = [i for i in d2 if rd.all_intents(by[i]["tree"]) <= tr_int and all_types(by[i]["tree"]) <= tr_typ]
        if len(d2) < a.min_depth_items:
            c = {"supported": False, "reason": f"only {len(d2)} items at depth {a.train_depth + 1} (< {a.min_depth_items}): the data does not support an unseen-depth split", "depth_counts_by_partition": counts,
                 "train": [], "dev_cal": [], "eval": {}}
        C["depth"] = c
    out = {"dataset": a.dataset, "locale": a.locale if a.dataset == "massive" else None, "domains": doms,
           "data_sha256": json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksums.json")))[a.dataset]["files"],
           "seed": 0, "load_stats": load_stats,
           "stats": {p: stats_of([by[i] for i in ids]) for p, ids in (("train", train), ("dev", dev), ("test", test))},
           "conditions": C}
    path = a.out or rd.split_path(a.dataset, a.locale)
    with open(path, "w") as f: json.dump(out, f, separators=(",", ":"))
    print(f"make_real_splits {a.dataset}: {path}\n  load: {load_stats}")
    for p, s in out["stats"].items(): print(f"  {p}: {s}")
    for n, c in C.items():
        if not c.get("supported", True): print(f"  condition {n}: NOT SUPPORTED: {c['reason']}"); continue
        print(f"  condition {n}: train {len(c['train'])}, dev_cal {len(c['dev_cal'])}, eval " + ", ".join(f"{k} {len(v)}" for k, v in c["eval"].items())
              + (f"; held intents {c['held_intents']}" if "held_intents" in c else "") + (f"; templates held {c['n_templates_held']} of {c['n_templates_eligible']} eligible" if "n_templates_held" in c else "")
              + (f"; depth counts {c['depth_counts_by_partition']}" if "depth_counts_by_partition" in c else ""))
    # the conversion checks (gold): BIO round trip and head frames from BIO equal the head frames from the tree (flat level)
    bad = 0
    for i in items:
        sp = [(t, s, e) for t, s, e, _ in i["slots"]]
        if not any(x[1] < y[2] and y[1] < x[2] and x != y for x in sp for y in sp) and rd.spans_of_bio(rd.bio(len(i["tokens"]), sp)) != sorted(sp, key=lambda x: x[1]): bad += 1
    print(f"  gold BIO -> spans round trip: {len(items) - bad} / {len(items)} exact (overlapping spans excluded)")

if __name__ == "__main__":
    main()
