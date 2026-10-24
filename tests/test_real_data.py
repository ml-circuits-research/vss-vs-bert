"""test_real_data.py: the conversion checks of the real-data round on GOLD (INSTRUCTIONS_REAL section 3), on the fake MASSIVE / TOPv2
files of tests/make_fake_real.py and, when downloaded, on the real files (SNIPS here; MASSIVE and TOPv2 on the DGX):
  1. annotation -> (tokens, spans) -> annotation round trip: MASSIVE annot_utt and TOP parse strings rebuilt exactly (whitespace normalised);
  2. spans -> BIO -> spans round trip (non-overlapping spans);
  3. gold frames scored by eval/metrics.score_real against themselves give 1.000 on FULL, intent and slot heads, with any thresholds;
  4. the split file of a dataset (when present) resolves every id in the loaded data, and its conditions are disjoint where they must be
     (train vs dev_cal vs every eval set; held-out templates / intents / depths never in train);
  5. TOP depth: the fake tree depths are those generated (0, 1, 2) and the depth split holds depth-2 items only outside training.
Run: python3 tests/make_fake_real.py && python3 tests/test_real_data.py [--real snips,massive,topv2]"""
import argparse, json, os, sys, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)

def check_conversion(items, dataset, raw_check):
    from common import real_data as rd, report_real
    from eval import metrics
    bad_bio = 0
    for it in items:
        sp = [(t, s, e) for t, s, e, _ in it["slots"]]
        if any(x[1] < y[2] and y[1] < x[2] and x != y for x in sp for y in sp): continue
        if rd.spans_of_bio(rd.bio(len(it["tokens"]), sp)) != sorted(sp, key=lambda x: x[1]): bad_bio += 1
    golds = [report_real.gold_flat(it) for it in items]; preds = [{"frame": dict(g), "margins": {k: 1.0 for k in g}} for g in golds]
    roles = sorted({t for it in items for t, *_ in it["slots"]}); taus = metrics.set_taus(golds, preds)
    sc = metrics.score_real(golds, preds, roles, taus)
    assert bad_bio == 0, f"{dataset}: BIO round trip failed on {bad_bio} items"
    assert sc["full"] == 1.0 and sc["intent"] == 1.0 and sc["slot_heads_on_gold"] in (1.0, None), f"{dataset}: gold frames do not score 1.000: {sc['full']}"
    nraw = raw_check()
    print(f"  {dataset}: {len(items)} items; annotation round trip exact on {nraw[0]} / {nraw[1]} (SNIPS: entities, the loader's misaligned ones counted as such); BIO round trip exact; gold frames FULL {sc['full']:.3f}, intent {sc['intent']:.3f}, slot heads {sc['slot_heads_on_gold']}")
    assert nraw[0] == nraw[1], f"{dataset}: annotation round trip failed on {nraw[1] - nraw[0]} items"

def massive_raw(path):
    from common import real_data as rd
    def f():
        ok = n = 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); toks, spans = rd.parse_massive_annot(r["annot_utt"]); n += 1
            canon = " ".join(rd.ANNOT.sub(lambda m: f" [{m.group(1).strip()} : {' '.join(m.group(2).split())}] ", r["annot_utt"]).split())
            ok += rd.massive_annot_of(toks, spans) == canon
        return ok, n
    return f

def top_raw(files):
    from common import real_data as rd
    def f():
        ok = n = 0
        for p in files:
            for k, line in enumerate(open(p, encoding="utf-8")):
                cols = line.rstrip("\n").split("\t")
                if k == 0 and cols[:1] == ["domain"]: continue
                try: toks, tree = rd.parse_top(cols[-1])
                except ValueError: continue
                n += 1; ok += rd.top_string(toks, tree) == " ".join(cols[-1].split())
        return ok, n
    return f

def snips_raw(st):
    """SNIPS: every entity chunk's own normalised tokens (minus trailing punctuation) equal the tokens of its converted span, except the
    spans counted as misaligned by the loader (a boundary inside a token); returns (exact, entities - misaligned)"""
    from common import real_data as rd
    def f():
        ok = n = 0
        for I, fs in rd.snips_files().items():
            for p in fs.values():
                for q in rd.read_json_any(p, __import__("collections").Counter()).get(I, []):
                    toks, spans, mis = rd.snips_chunks_to_tokens(q["data"]); it = rd.make_item("x", "x", toks, I, {"intent": I, "start": 0, "end": len(toks), "slots": [{"type": t, "start": a, "end": b, "child": None} for t, a, b in spans]})
                    want = []
                    for c in q["data"]:
                        if c.get("entity") and c["text"].strip():
                            w, _ = rd.normalise(c["text"].split())
                            while len(w) > 1 and w[-1] in rd.PUNCT_TOKENS: w = w[:-1]
                            want.append(w)
                    got = [it["tokens"][a:b] for _, a, b, _ in it["slots"]]
                    n += len(want); ok += sum(x == y for x, y in zip(want, got)) if len(want) == len(got) else 0
        return ok + st.get("misaligned_spans", 0), n
    return f

def check_splits(dataset, items):
    from common import real_data as rd
    sp = json.load(open(rd.split_path(dataset))); by = {i["id"] for i in items}
    for name, c in sp["conditions"].items():
        if not c.get("supported", True): print(f"  {dataset} condition {name}: unsupported ({c['reason']})"); continue
        tr, cal = set(c["train"]), set(c["dev_cal"])
        assert not (tr & cal), f"{name}: train and dev_cal overlap"
        for k, v in c["eval"].items():
            assert not (set(v) & tr) and not (set(v) & cal), f"{dataset} {name}: eval set {k} overlaps train or dev_cal"
        miss = (tr | cal | {i for v in c["eval"].values() for i in v}) - by
        assert not miss, f"{dataset} {name}: {len(miss)} ids not in the data"
    c = sp["conditions"]["ood_intents"]; it = {i["id"]: i for i in items}
    assert not any(rd.all_intents(it[i]["tree"]) & set(c["held_intents"]) for i in c["train"]), "a held-out intent occurs in training"
    if "depth" in sp["conditions"] and sp["conditions"]["depth"].get("supported", True):
        assert all(it[i]["depth"] <= 1 for i in sp["conditions"]["depth"]["train"]), "depth > 1 in the depth split's training rows"
    print(f"  {dataset}: split file resolves every id; train / dev_cal / eval sets disjoint in every condition; held-out intents and depths never trained")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", default="snips"); a = ap.parse_args()
    fake = os.path.join(HERE, "real_fake")
    if not os.path.exists(fake): subprocess.run([sys.executable, os.path.join(HERE, "make_fake_real.py")], check=True)
    print("fake MASSIVE and TOPv2 (tests/real_fake):")
    env = dict(os.environ, REAL_DATA=fake, REAL_SPLITS=os.path.join(fake, "splits")); os.makedirs(env["REAL_SPLITS"], exist_ok=True)
    for ds in ("massive", "topv2"):
        subprocess.run([sys.executable, os.path.join(ROOT, "data", "real", "make_real_splits.py"), ds, "--domains", "navigation", "--min_depth_items", "10"], check=True, env=env, stdout=subprocess.DEVNULL)
    os.environ.update(REAL_DATA=fake, REAL_SPLITS=env["REAL_SPLITS"])
    from common import real_data as rd
    import importlib; importlib.reload(rd)
    items, _ = rd.load_massive(path=os.path.join(fake, "massive", "1.1", "data", "en-US.jsonl")); check_conversion(items, "massive (fake)", massive_raw(os.path.join(fake, "massive", "1.1", "data", "en-US.jsonl"))); check_splits("massive", items)
    items, _ = rd.load_topv2(("navigation",)); check_conversion(items, "topv2 (fake)", top_raw(list(rd.topv2_files({"navigation"}).values()))); check_splits("topv2", items)
    import collections
    dc = collections.Counter(i["depth"] for i in items); assert set(dc) == {0, 1, 2}, dc; print(f"  topv2 (fake): depth counts {dict(sorted(dc.items()))}")
    deep = next(i for i in items if i["depth"] == 2); print(f"  topv2 (fake) depth-2 head frame: {rd.head_frame(deep['tree'], deep['tokens'])}")
    for ds in [x for x in a.real.split(",") if x]:
        os.environ["REAL_DATA"] = os.path.join(ROOT, "data", "real", "raw"); os.environ["REAL_SPLITS"] = os.path.join(ROOT, "data", "real"); importlib.reload(rd)
        try: items, st = rd.load(ds)
        except FileNotFoundError as e: print(f"  {ds}: not downloaded here ({e}); skipped"); continue
        print(f"real {ds}: load stats {st}")
        if ds == "massive": raw = massive_raw(rd.massive_path("en-US"))
        elif ds == "topv2": raw = top_raw(list(rd.topv2_files({"navigation", "event", "reminder"}).values()))
        else: raw = snips_raw(st)
        check_conversion(items, ds, raw)
        if os.path.exists(rd.split_path(ds)): check_splits(ds, items)
    print("test_real_data: PASS")
