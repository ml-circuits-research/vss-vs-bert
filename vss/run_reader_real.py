"""run_reader_real.py (runs 8-10, reader): the VSS reader on REAL utterances annotated with frames (SNIPS, MASSIVE, TOPv2), through the
main repository's reader-as-a-tool (tool/sop), imported UNMODIFIED: each condition is written as a WORLD FOLDER (legend.json, kinds.tsv,
world.json, examples/*.sop) under logs/worlds/ (untracked), its rows are loaded by sop.train.load_rows, the reader is trained by
sop.train.train_one (experiment/v8_gates.V8 + experiment/trainer.train_loop, path + chainL + bigram, gates learned from closed) and
read by sop.infer.Reader.formalize (numpy). Scored by common/report_real.evaluate (eval/metrics.score_real), the scorer of the BERT rows.

How the frames are expressed in the tool (each a stated choice; INSTRUCTIONS_REAL section 3):
  - The INTENT is a role 'intent' with a CLOSED value set (legend reader.closed_values, as WikiSQL's aggregate in tool/export_wikisql.py):
    it is read from the state by cleanup over the intent symbols, not from a verb token. The tool's event decoder cannot read the intent
    as the legend's EVENT: with predicate_role it needs a verb token whose lemma the legend lists for that event; without it, the event
    is the single event or a lexical lookup. The legend therefore has ONE event, 'command' (its verbs list ['command'] is required by the
    legend validation and is never used: predicate_role is false).
  - Each SLOT TYPE is a role; its value is the slot's HEAD token (the last token of the span; the tool reads a multi-word value by its last
    token, so the target is the same as with the full value). The first span of a type is kept when a type repeats (set roles of the tool
    are table roles: their candidates come from a sentence table; they are not used). Presence: the occupancy marker (default reader
    block), thresholds FITTED by the tool on calibration rows = a sample of the training rows + the dev CALIBRATION half (split
    'calibration_dev'; never reported, rule 25).
  - TOPv2 nested intents: a slot holding an intent is a $ref to a nested 'command' wire (compiled into the target by the tool). The tool's
    decoder does not read nested event wires (tool README, limits of v1), so this file adds the SMALLEST READOUT WRAPPER (read_nested
    below, using the Reader's own primitives: stars, role vectors, candidates, cleanup): after unbinding a slot's role, the nested intent
    is read like the top one; 'nested' iff the nested intent's cleanup ratio to the structure score >= a threshold FITTED on the dev
    calibration half (sop.train.fit_threshold, balanced accuracy). The tool's schema is built from the top-level wires only, so a nested
    level reads EVERY slot role, with the kinds of the role's heads at any level of the training trees and presence thresholds fitted on
    the dev calibration half at the nested levels (fit_nested; the tool's top-level threshold when a role has too few nested cases, then
    the threshold pooled over every role at nested levels).
    The same wrapper with nesting disabled must reproduce the tool's own readout (checked and printed on the first eval set).
  - CANDIDATES of a slot role: the utterance's own tokens of the role's kinds (formalize's `filt`), the output space of a tagger; the
    tool's default (every catalog word of the role's kinds, here thousands of words per cluster) would let a role be filled by a word
    the utterance does not contain.
  - A slot type whose training fillers are ALL nested intents would make the tool's schema treat it as an entity role without head kinds;
    such a role is set to a word role with its (possibly empty) word kinds before training (schema edit in this file, stated, printed).
Kinds (GIVEN, rule 20), the WikiSQL recipe of the main repository (v12_wikisql_baselines / run_reader_wikisql.py): the --n_freq (40) most
frequent tokens of the condition's TRAINING utterances are their own kinds, every other word of the vocabulary (train + dev + test, as
v12) is one of --k_clusters (48) k-means (cosine) clusters of skip-gram vectors (experiment/v10_controls.skipgram) trained on the training
utterances only; a word never seen in training keeps its untrained vector and so a cluster drawn independently of any label
(--unseen_kind cluster, v12's behaviour) or gets the tool's 'unknown' kind (--unseen_kind unknown: never a filler). No gold slot label is
used to define any kind. The kinds depend on the condition's full training set, not on the training size (as v12).
Spans (SECONDARY, a RULE, not learned by the reader): from a head token, extend left while the token's training rate of being a
non-final token of a span of that role exceeds 0.5; the head is the leftmost occurrence of the token not used by another role. Applied
to GOLD heads too, as the rule's ceiling.
Training (not tuned, INSTRUCTIONS section 4: the reader takes the paper's configuration): D 1024, batch 32, lr 0.02, length buckets (as
v11), epochs 60 / 20 / 10 at 1k / 5k / full (stated before the runs, sized as v12's 60 / 20 / 3 at 1k / 5k / 56k); stars per seed
(rule 6). Thresholds of the scorer on the dev calibration half.
Run: python3 vss/run_reader_real.py --dataset snips|massive|topv2 [--condition base] [--sizes 1000,5000,full] [--seeds 5] [--test] [--save_model]"""
import argparse, collections, copy, json, os, shutil, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v10_controls as v10
mainrepo.import_sop()
from sop import soplang, world as sworld, train as strain
from sop.legend import Legend, tokenize, sha256_file
from sop.infer import Reader, idgen
from common import real_data as rd, report_real
from eval import metrics

def role_id(t):
    """a slot type as a legend role id (^[A-Za-z][A-Za-z0-9_]*$): other characters -> '_', a leading non-letter gets 'r_'"""
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in t)
    return s if s[:1].isalpha() else "r_" + s

def build_kinds(cond, a, dev_):
    """the kinds of every word (see the docstring): returns {word: kind}"""
    items = cond["items"]; train = [items[i]["tokens"] for i in cond["train"]]
    vocab = {}
    for it in items.values():
        for w in it["tokens"]: vocab.setdefault(w, len(vocab))
    freq = collections.Counter(w for s in train for w in s); singles = [w for w, _ in freq.most_common(a.n_freq)]
    print(f"kinds: {len(train)} training utterances only for the frequencies and the skip-gram; vocabulary {len(vocab)} (train + dev + test, as v12); "
          f"{sum(1 for w in vocab if w not in freq)} words never in training ({a.unseen_kind})")
    emb = v10.skipgram([s for s in train if len(s) > 1], vocab, dim=64, window=2, epochs=3, device=dev_, seed=0)
    X = torch.tensor(emb, device=dev_); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); V = len(vocab); gk = torch.Generator().manual_seed(0)
    k = min(a.k_clusters, V); C = X[torch.randperm(V, generator=gk)[:k].to(dev_)].clone()
    for _ in range(25):
        asg = (X @ C.T).argmax(1)
        for c in range(k):
            mem = X[asg == c]
            if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
    asg = (X @ C.T).argmax(1).cpu().numpy(); out = {}
    for w, i in vocab.items():
        if w in singles: out[w] = "w:" + w
        elif w not in freq and a.unseen_kind == "unknown": continue
        else: out[w] = f"C{int(asg[i])}"
    return out

def wires_of(item, ids, rmap):
    """an item -> SOP wires (top wire first): event 'command', intent (closed value), slot roles = head token or $ref to a nested wire"""
    out = []
    def node(n, top):
        w = soplang.Wire(next(ids), "command"); out.append(w)
        if top: w.set("_text", " ".join(item["tokens"])); w.set("_item", item["id"]); w.set("_source", f"{item['id'].split(':')[0]} gold (converted by common/real_data.py)")
        w.set("intent", n["intent"]); seen = set()
        for sl in sorted(n["slots"], key=lambda s: s["start"]):
            r = rmap[sl["type"]]
            if r in seen: continue
            seen.add(r)
            if sl["child"]: w.set(r, ("ref", node(sl["child"], False).id))
            else: w.set(r, item["tokens"][sl["end"] - 1])
        return w
    node(item["tree"], True)
    return out

def write_world(wd, cond, train_ids, kinds, a):
    items = cond["items"]; types_ = sorted({sl["type"] for i in train_ids + cond["dev_cal"] for sl in walk_slots(items[i]["tree"])})
    rmap = {t: role_id(t) for t in types_}; assert len(set(rmap.values())) == len(rmap) and "intent" not in rmap.values(), "slot type ids collide"
    intents = sorted({x for i in train_ids + cond["dev_cal"] for x in rd.all_intents(items[i]["tree"])})
    legend = {"version": 1, "roles": [{"id": "intent", "question": "what does the user want done?"}] + [{"id": rmap[t], "question": f"slot {t}"} for t in types_],
              "events": [{"id": "command", "verbs": ["command"]}], "operators": [], "moods": ["statement"], "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}, "values": [],
              "reader": {"predicate_role": False, "occupancy_marker": True, "definite_dets": [], "closed_values": {"intent": intents}, "no_sense_share": 1.0}}
    if os.path.exists(wd): shutil.rmtree(wd)
    for d in ("examples", "model", "runs"): os.makedirs(os.path.join(wd, d))
    json.dump(legend, open(os.path.join(wd, "legend.json"), "w"), indent=1); sha = sha256_file(os.path.join(wd, "legend.json"))
    with open(os.path.join(wd, "kinds.tsv"), "w") as f:
        f.write("# word\tkind: the n most frequent training tokens as their own kinds (w:<word>), skip-gram clusters C<i> (GIVEN, rule 20; vss/run_reader_real.py)\n")
        for w in sorted(kinds): f.write(f"{w}\t{kinds[w]}\n")
    json.dump({"name": os.path.basename(wd), "language": "en", "description": f"{cond['dataset']} / {cond['condition']}: real utterances -> intent + slot heads", "version": 1},
              open(os.path.join(wd, "world.json"), "w"), indent=1)
    for fname, ids_ in (("train.sop", train_ids), ("calibration.sop", cond["dev_cal"])):
        doc = soplang.Doc(directives=[("world", os.path.basename(wd), None), ("legend", sha, None), ("generated", "vss/run_reader_real.py from the dataset's gold annotation", None)])
        g = idgen()
        for i in ids_: doc.wires += wires_of(items[i], g, rmap)
        with open(os.path.join(wd, "examples", fname), "w") as f: f.write(soplang.write(doc))
    return rmap, {v: k for k, v in rmap.items()}

def walk_slots(tree):
    for sl in tree["slots"]:
        yield sl
        if sl["child"]: yield from walk_slots(sl["child"])

def fix_schema(schema):
    fixed = []
    for r, rs in schema["roles"].items():
        if rs["main"] == "entity" and not rs.get("head_kinds"): rs["main"] = "word"; fixed.append(r)
    if fixed: print(f"[schema] roles whose training fillers are all nested intents set to word roles (kinds {[schema['roles'][r]['kinds'] for r in fixed]}): {fixed}")
    return schema

def span_rates(cond, train_ids):
    """rate[type][w] = share of the training occurrences of w that are a NON-FINAL token of a span of that type (the span rule)"""
    occ = collections.Counter(); inside = collections.defaultdict(collections.Counter)
    for i in train_ids:
        it = cond["items"][i]; occ.update(it["tokens"])
        for t, s, e, _ in it["slots"]:
            for k in range(s, e - 1): inside[t][it["tokens"][k]] += 1
    return {t: {w: c / occ[w] for w, c in ws.items()} for t, ws in inside.items()}

def spans_by_rule(tokens, flat, rates):
    used, out = set(), []
    for r, h in flat.items():
        if r == "intent" or "/" in r or h == rd.NESTED: continue
        js = [j for j, t in enumerate(tokens) if t == h and j not in used]
        if not js: continue
        e = js[0] + 1; s = js[0]
        while s > 0 and (s - 1) not in used and rates.get(r, {}).get(tokens[s - 1], 0.0) > 0.5: s -= 1
        used |= set(range(s, e)); out.append((r, s, e))
    return out

def nested_kinds(cond, train_ids, legend, rmap):
    """the kinds of every slot role's head tokens at ANY level of the training trees (the tool's schema sees the top level only)"""
    ks = collections.defaultdict(set)
    def walk(n, toks):
        for sl in n["slots"]:
            if sl["child"]: walk(sl["child"], toks)
            else:
                k = legend.kind(toks[sl["end"] - 1])
                if k: ks[rmap[sl["type"]]].add(k)
    for i in train_ids: walk(cond["items"][i]["tree"], cond["items"][i]["tokens"])
    return {r: sorted(ks.get(r, ())) for r in rmap.values()}

def presence_ratio(R, Sn, key, core):
    mk = R.st.marker() / np.sqrt(R.D); return float(np.real(np.sum(Sn * key * np.conj(mk)))) / core if core > 1e-9 else 0.0

def read_nested(R, S, tokens, th_nested, rinv, nest=True, max_depth=4, nk=None, nth=None):
    """the readout wrapper (docstring): returns (flat frame, margins) in the dataset's slot-type names. The TOP level is the tool's own
    readout (its schema roles, its presence thresholds, the utterance's tokens as candidates); a NESTED level reads every slot role,
    with the kinds of nk (heads at any level of training) and the presence thresholds nth fitted on the dev calibration half at nested
    levels (fit_nested); a role without a nested threshold falls back to the tool's top-level one, then to the threshold pooled over every
    role at nested levels (stated)"""
    nS = np.linalg.norm(S)
    if nS < 1e-9: return {}, {}
    Sn = S / nS; core = R.structure_score(Sn, 1.0); sch = R.schema; nk = nk or {}; nth = nth or {}
    frame, marg = {}, {}
    def level(K, prefix, depth):
        w, mg, top = R.best(Sn * K * np.conj(R.st.role("intent")), R.role_cands("intent")); frame[prefix + "intent"] = w; marg[prefix + "intent"] = mg
        roles = [r for r in sch["role_order"] if r != "intent"] if depth == 0 else sorted(r for r in nk if nk[r])
        for r in roles:
            key = K * np.conj(R.st.role(r)); name = prefix + rinv.get(r, r); ratio = presence_ratio(R, Sn, key, core)
            thr = R.th.get("presence", {}).get(r) if depth == 0 else nth.get(r, R.th.get("presence", {}).get(r, nth.get("_pooled")))
            present = (ratio >= thr) if thr is not None else (sch["roles"][r]["rate"] >= 0.5 if r in sch["roles"] else False)
            pm = abs(ratio - thr) if thr is not None else 1.0
            if not present: marg[name] = pm; continue
            nested_m = 1.0; nested = False
            if nest and th_nested is not None and depth < max_depth:
                _, _, ntop = R.best(Sn * key * np.conj(R.st.role("intent")), R.role_cands("intent")); nr = ntop / core if core > 1e-9 else 0.0
                nested = nr >= th_nested; nested_m = abs(nr - th_nested)
            if nested:
                frame[name] = rd.NESTED; marg[name] = min(pm, nested_m); level(key, name + "/", depth + 1)
            else:
                kinds = sch["roles"][r]["kinds"] if depth == 0 and r in sch["roles"] else nk.get(r, [])
                h, mg_, _ = R.best(Sn * key, R.cands(kinds, set(tokens)))
                if h is None: marg[name] = 0.0; continue
                frame[name] = h; marg[name] = min(mg_, pm, nested_m)
    level(1.0, "", 0)
    return frame, marg

def fit_nested(R, items, rmap, nk):
    """presence thresholds of the slot roles at NESTED levels, FITTED on the dev calibration half (sop.train.fit_threshold, balanced
    accuracy), with the gold structure: for every nested gold node, the presence ratio of every role after unbinding the path's roles"""
    st = collections.defaultdict(lambda: ([], []))
    for it in items:
        S, _, _, _ = R.chart(it["tokens"]); nS = np.linalg.norm(S)
        if nS < 1e-9: continue
        Sn = S / nS; core = R.structure_score(Sn, 1.0)
        def walk(n, K, depth):
            if depth > 0:
                have = {rmap.get(sl["type"]) for sl in n["slots"]}
                for r in nk:
                    if nk[r]: st[r][0 if r in have else 1].append(presence_ratio(R, Sn, K * np.conj(R.st.role(r)), core))
            for sl in n["slots"]:
                if sl["child"] and sl["type"] in rmap: walk(sl["child"], K * np.conj(R.st.role(rmap[sl["type"]])), depth + 1)
        walk(it["tree"], 1.0, 0)
    th = {r: strain.fit_threshold(p, q) for r, (p, q) in st.items() if len(p) >= 5 and len(q) >= 5}
    P = [x for p, _ in st.values() for x in p]; Q = [x for _, q in st.values() for x in q]
    if len(P) >= 5 and len(Q) >= 5: th["_pooled"] = strain.fit_threshold(P, Q)
    print(f"[nested] presence thresholds at nested levels fitted on the dev calibration half for {sorted(k for k in th if k != '_pooled')}; pooled over every role "
          f"{th.get('_pooled', float('nan')):.3f} ({len(P)} present / {len(Q)} absent cases), used for a role without its own (nor a top-level one)")
    return th

def predict(R, items, rinv, rates, th_nested=None, nested=False, nk=None, nth=None):
    out = []; g = idgen("e")
    for it in items:
        text = " ".join(it["tokens"]); wires, f, info = R.formalize(text, 0, g, filt=set(it["tokens"]))
        if nested:
            fr, mg = read_nested(R, info["S"], it["tokens"], th_nested, rinv, nk=nk, nth=nth)
        else:
            fr, mg = {}, {}
            for r in R.schema["role_order"]:
                name = rinv.get(r, r); fl = f["roles"].get(r)
                if fl is not None and fl[1] is not None: fr[name] = fl[1]; mg[name] = float(f["margins"].get(r, 0.0))
                else: mg[name] = float(f["margins"].get(r + ".presence", 0.0))
        p = {"frame": fr, "margins": mg, "status": info["status"], "residual": float(info["residual"])}
        if not nested:
            p["spans"] = spans_by_rule(it["tokens"], fr, rates); p["spans_rule_on_gold"] = spans_by_rule(it["tokens"], report_real.gold_flat(it), rates)
        out.append(p)
    return out

def fit_nested_threshold(R, items, rmap):
    """the nested / word decision threshold, FITTED on the dev calibration half (sop.train.fit_threshold): the ratio of the best nested-intent
    cleanup to the structure score after unbinding each gold top-level slot role, for slots holding an intent vs a word"""
    pos, neg = [], []
    for it in items:
        S, _, _, _ = R.chart(it["tokens"]); nS = np.linalg.norm(S)
        if nS < 1e-9: continue
        Sn = S / nS; core = R.structure_score(Sn, 1.0); seen = set()
        for sl in sorted(it["tree"]["slots"], key=lambda s: s["start"]):
            r = rmap.get(sl["type"])
            if r is None or r in seen or r not in R.schema["roles"]: continue
            seen.add(r); _, _, top = R.best(Sn * np.conj(R.st.role(r)) * np.conj(R.st.role("intent")), R.role_cands("intent"))
            (pos if sl["child"] else neg).append(top / core if core > 1e-9 else 0.0)
    if len(pos) < 5 or len(neg) < 5: print(f"[nested] too few calibration slots (nested {len(pos)}, word {len(neg)}): nesting not read"); return None
    t = strain.fit_threshold(pos, neg); print(f"[nested] threshold {t:.3f} fitted on {len(pos)} nested / {len(neg)} word slots of the dev calibration half (mean {np.mean(pos):.3f} / {np.mean(neg):.3f})")
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=rd.DATASETS); ap.add_argument("--condition", default="base"); ap.add_argument("--locale", default="en-US")
    ap.add_argument("--sizes", default="1000,5000,full"); ap.add_argument("--epochs", default="60,20,10", help="per size, in the order of --sizes")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--unseen_kind", default="cluster", choices=["cluster", "unknown"])
    ap.add_argument("--n_cal", type=int, default=400); ap.add_argument("--device", default="cuda"); ap.add_argument("--test", action="store_true")
    ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only: the first n items of each eval set"); ap.add_argument("--max_train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--save_model", action="store_true", help="copy the seed-0 model/ folder of each size to logs/models/")
    a = ap.parse_args(); dev_ = a.device
    cond = rd.load_condition(a.dataset, a.condition, a.locale); items = cond["items"]
    if a.max_train: cond["train_order"] = cond["train_order"][:a.max_train]; keep = set(cond["train_order"]); cond["train"] = [i for i in cond["train"] if i in keep]
    sizes = a.sizes.split(",") if a.condition == "base" else ["full"]; eps = dict(zip(a.sizes.split(","), [int(x) for x in a.epochs.split(",")]))
    if a.condition != "base": eps = {"full": int(a.epochs.split(",")[-1])}
    nested = any(it["depth"] > 0 for it in items.values())
    print(f"run_reader_real: main repository {mainrepo.check_pinned()}; dataset {a.dataset} / condition {a.condition}; device {dev_}; D {a.D}; batch {a.batch}; lr {a.lr}; "
          f"sizes {sizes}, epochs {eps}; seeds {a.seeds}; train {len(cond['train'])}, dev calibration {len(cond['dev_cal'])}; eval " + ", ".join(f"{k} {len(v)}" for k, v in report_real.eval_sets(cond, a.test).items()))
    print("regime (rule 24): real utterances, the dataset's own annotation; head-token frame (intent + the LAST token of the first span of each slot type); "
          "the intent as a closed-value role, slots as roles with the occupancy marker; thresholds on the dev calibration half; test only with --test" + ("; nested intents read by the wrapper" if nested else ""))
    kinds = build_kinds(cond, a, dev_); ev = report_real.eval_sets(cond, a.test, a.max_eval)
    allrows = []
    for size in sizes:
        n = len(cond["train"]) if size == "full" else int(size); train_ids = cond["train_order"][:n] if a.condition == "base" else cond["train"]
        wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_{a.dataset}_{a.condition}_{size}"); rmap, rinv = write_world(wd, cond, train_ids, kinds, a)
        legend = Legend(os.path.join(wd, "legend.json")); ci = collections.Counter(items[i]["intent"] for i in train_ids)
        print(f"chance (size {size}): intent 1/{len(ci)} = {1 / len(ci):.3f}, majority intent {ci.most_common(1)[0][1] / len(train_ids):.3f}; a slot head: 1 / (the utterance's tokens of the role's kinds)")
        if legend.errors: raise SystemExit("legend errors: " + "; ".join(legend.errors))
        rows, errors, _ = strain.load_rows(wd, legend)
        if errors: raise SystemExit("examples rejected by sop check:\n" + "\n".join(errors[:20]))
        tset, cset = set(train_ids), set(cond["dev_cal"]); splits = collections.defaultdict(list)
        for r in rows:
            iid = r["wire"].value("_item"); assert r["tokens"] == items[iid]["tokens"], f"tokenisation differs for {iid}"
            r["split"] = "train" if iid in tset else ("calibration_dev" if iid in cset else "unused"); splits[r["split"]].append(r["id"])
        tr_rows = [r for r in rows if r["split"] == "train"]; schema = fix_schema(strain.build_schema(tr_rows, legend)); schema["content_kinds"] = sorted(strain.content_kinds_of(tr_rows, legend))
        rates = span_rates(cond, train_ids); per = []; nk = nested_kinds(cond, train_ids, legend, rmap) if nested else None
        print(f"[world] {wd}: legend {legend.sha[:19]}..., roles {len(legend.roles)} (intent + {len(legend.roles) - 1} slot types), intents {len(legend.reader['closed_values']['intent'])}, "
              f"kinds {len(set(kinds.values()))}; rows train {len(splits['train'])}, calibration_dev {len(splits['calibration_dev'])}")
        for seed in range(a.seeds):
            ns = types.SimpleNamespace(D=a.D, star_seed=f"seed{seed}", identities="stars", device=dev_, length_buckets=True, epochs=eps[size], batch=a.batch, shuffle=True,
                                       max_train=0, code_list=["path", "chainL", "bigram"], no_ckpt=True, resume=False, log_every=500, lr=a.lr, n_cal=a.n_cal, tau=0.05,
                                       world_name=os.path.basename(wd), language="en", commit=mainrepo.check_pinned(), depth_estimator="kinds", threshold_rule="fit", store_presence_ratios=False)
            md = os.path.join(wd, f"model_seed{seed}"); t0 = time.time()
            R, ex = strain.train_one(ns, legend, rows, splits, schema, seed, md, os.path.join(wd, "runs"))
            t_train = time.time() - t0
            th_n = fit_nested_threshold(R, [items[i] for i in cond["dev_cal"]], rmap) if nested else None
            nth = fit_nested(R, [items[i] for i in cond["dev_cal"]], rmap, nk) if nested else None
            cal_p = predict(R, [items[i] for i in cond["dev_cal"]], rinv, rates, th_n, nested, nk, nth)
            preds = {k: predict(R, [items[i] for i in ids], rinv, rates, th_n, nested, nk, nth) for k, ids in ev.items()}
            if nested:     # the wrapper with nesting disabled must reproduce the tool's own readout on the top level
                k0 = next(iter(ev)); sub = [items[i] for i in ev[k0][:100]]; A = predict(R, sub, rinv, rates, None, False); B = [read_nested(R, R.chart(it["tokens"])[0], it["tokens"], None, rinv, nest=False)[0] for it in sub]
                print(f"[check] wrapper (nesting off) = tool readout on {sum(x['frame'] == y for x, y in zip(A, B))} / {len(sub)} items of {k0}")
                ok = nn_ = 0          # the wrapper on the compiled GOLD target states (decoding independent of what was learned)
                for k1, ids in ev.items():
                    for i in ids[:100]:
                        ws = wires_of(items[i], idgen(), rmap); St = R.comp.wire(ws[0], {w.id: w for w in ws}, items[i]["tokens"])
                        ok += read_nested(R, St, items[i]["tokens"], th_n, rinv, nk=nk, nth=nth)[0] == report_real.gold_flat(items[i]); nn_ += 1
                print(f"[check] wrapper on the compiled GOLD target states (first 100 items of each eval set): {ok} / {nn_} frames exact")
            rb = os.path.join(md, "routebook.npz"); z = np.load(rb); nparam = int(sum(z[k].size for k in ("path", "bg0", "bg", "A", "L", "logits")))
            sub = [items[i] for i in (ev.get("dev_rep") or next(iter(ev.values())))[:a.cost_n]]; t1 = time.time(); predict(R, sub, rinv, rates, th_n, nested, nk, nth)
            cost = {"params": nparam, "bytes": os.path.getsize(rb), "bytes_float32": nparam * 4, "bytes_4bit": nparam // 2, "cpu_ms_per_sentence": 1000 * (time.time() - t1) / max(1, len(sub)), "train_s": round(t_train, 1)}
            row = report_real.evaluate("VSS reader (sop tool)", "reader", cond, seed, None if size == "full" else n, preds, cond["dev_cal"], cal_p, sorted(rmap),
                                       extra={"cost": cost, "kinds": len(set(kinds.values())), "th_nested": th_n, "th_presence_nested": nth, "tool_calibration": ex.get("calibration"), "epochs": eps[size], "D": a.D})
            print(f"  seed {seed} size {size}: cost {nparam/1e6:.2f}M phases, routebook.npz {cost['bytes']/1e6:.1f} MB ({cost['bytes_4bit']/1e6:.1f} MB at 4 bits), {cost['cpu_ms_per_sentence']:.2f} ms/sentence (numpy, CPU, read + scoring rule)", flush=True)
            if a.save_model and seed == 0:
                dst = os.path.join(mainrepo.ROOT, "logs", "models", f"reader_real_{a.dataset}_{a.condition}_{size}_seed0")
                if os.path.exists(dst): shutil.rmtree(dst)
                shutil.copytree(md, dst)
            if seed > 0 or not a.save_model: shutil.rmtree(md, ignore_errors=True)
            per.append(row)
        report_real.summarize(per, "VSS reader (sop tool)"); allrows += per
    print("reference: none in the main repository for these data (a new round); the written interpretation is INSTRUCTIONS_REAL_2026-10-24.md table R7")
    metrics.checklist(a.seeds, "[x] the same condition rows and scorer as the BERT rows [x] kinds from training text only, no gold label [x] test only with --test")
