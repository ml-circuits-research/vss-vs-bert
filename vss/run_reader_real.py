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
import argparse, collections, copy, importlib, json, os, shutil, sys, time, types
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

def load_typer():
    """this repository's sop/typer.py (R71; the author's file, read only), loaded by path: the pinned tool is also a package named 'sop'"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sop_typer_r71", os.path.join(mainrepo.ROOT, "sop", "typer.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

R85_VERBS = {"AddToPlaylist": ["add", "put"], "PlayMusic": ["play", "listen"], "BookRestaurant": ["book", "reserve"], "RateBook": ["rate", "give"],
             "SearchCreativeWork": ["find", "show", "search"], "SearchScreeningEvent": ["find", "show", "search"], "GetWeather": ["is", "what", "will"]}   # responses R85
SOP_DIR = None   # --sop_dir: a directory holding the 'sop' package of a named version (R82: the tree's sop/ may be newer than a row's)

def sop_source():
    return open(os.path.join(SOP_DIR or mainrepo.ROOT, "sop", "SOURCE")).read().strip()[:7]

def load_sop_repo():
    """this repository's sop/ (the reader's single implementation since R58) as the package 'sop_repo', beside the pinned tool"""
    import importlib.util
    if "sop_repo" in sys.modules: return sys.modules["sop_repo"]
    base = SOP_DIR or mainrepo.ROOT
    spec = importlib.util.spec_from_file_location("sop_repo", os.path.join(base, "sop", "__init__.py"), submodule_search_locations=[os.path.join(base, "sop")])
    m = importlib.util.module_from_spec(spec); sys.modules["sop_repo"] = m; spec.loader.exec_module(m)
    import importlib; importlib.import_module("sop_repo.infer"); return m

def build_kinds_typer(cond, a):
    """R71: the kinds learned from the TRAINING annotation by sop/typer.py (no evaluation label; text only for the lexicon): returns
    ({word: kind}, typer)"""
    T = load_typer(); items = cond["items"]
    sents = [(items[i]["tokens"], [(t, s_, e) for t, s_, e, _ in items[i]["slots"]]) for i in cond["train"]]
    ty = T.Typer(n_freq=a.n_freq, seed=0).fit(sents); lex = ty.lexicon([it["tokens"] for it in items.values()])
    print(f"kinds (typer, R71): fitted on {len(sents)} training utterances (train accuracy {ty.train_acc:.3f}); {len(set(lex.values()))} kinds over {len(lex)} words; {ty.n_params()} typer weights")
    return lex, ty, T

def spans_by_typer(tokens, flat, ty, T):
    """R71: the span of each decoded slot = kind_run over the typer's in-context kinds around the decoded head (its leftmost unused occurrence)"""
    ks = ty.sentence_kinds(tokens); used, out = set(), []
    for r, h in flat.items():
        if r == "intent" or "/" in r or h == rd.NESTED: continue
        js = [j for j, t in enumerate(tokens) if t == h and j not in used]
        if not js: continue
        s_, e = T.kind_run(ks, js[0])
        if any(k in used for k in range(s_, e)): s_, e = js[0], js[0] + 1
        used |= set(range(s_, e)); out.append((r, s_, e))
    return out

def spans_typer_only(tokens, ty):
    """R71 item 1: spans = the maximal runs of one S: kind in the typer's in-context kinds (no reader)"""
    ks = ty.sentence_kinds(tokens); out = []; j = 0
    while j < len(ks):
        if str(ks[j]).startswith("S:"):
            e = j + 1
            while e < len(ks) and ks[e] == ks[j]: e += 1
            out.append((ks[j][2:], j, e)); j = e
        else: j += 1
    return out

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

TYPER = None   # (typer, module) when --kinds typer (R71)

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
        if not nested and TYPER is not None:     # R71: spans from the typer's kinds around the decoded head; the rule kept as a second line
            p["spans"] = spans_by_typer(it["tokens"], fr, *TYPER); p["spans_by_rule"] = spans_by_rule(it["tokens"], fr, rates); p["spans_typer_only"] = spans_typer_only(it["tokens"], TYPER[0])
        elif not nested:
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

def per_intent(cond, a, kinds, ty, T, ev):
    """R72 item 4: one world per intent (its slot types only as questions; the typer's kinds with every other slot kind mapped to O; its
    reader trained on that intent's training utterances only, D = --D, the three routes path + chainL + bigram, R73), read through this
    repository's sop/ (the reference since R58; residual 'explained'); all worlds read every utterance; the DISPATCHER keeps the reading
    with the best status (ok, then abstain, then the rest), then the smallest explained residual, then the largest minimum margin; the
    intent is the chosen world. Rejection: every world rejects when each has a status other than ok or a residual above its own
    threshold (the 98% quantile of its residual on its own intent's dev calibration utterances, rule 25)."""
    SR = load_sop_repo(); items = cond["items"]; intents = sorted({items[i]["intent"] for i in cond["train"]})
    print(f"[R72] {len(intents)} worlds (one per intent): {intents}; D {a.D}; routes path + chainL + bigram; decoder: sop @ {sop_source()} (residual explained)")
    rank = lambda st: {"ok": 0, "abstain": 1}.get(st, 2); allrows = []; all_types = set()
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        worlds = {}; nparam = 0; t0 = time.time()
        for I in intents:
            tr = [i for i in cond["train"] if items[i]["intent"] == I]; cal = [i for i in cond["dev_cal"] if items[i]["intent"] == I]
            types_ = sorted({sl["type"] for i in tr + cal for sl in items[i]["tree"]["slots"]}); all_types |= set(types_)
            rmap = {t: role_id(t) for t in types_}; rinv = {v: k for k, v in rmap.items()}
            kI = dict(kinds) if a.world_kinds == "all" else {w: (k if not str(k).startswith("S:") or k[2:] in types_ else "O") for w, k in kinds.items()}   # R75: 'all' keeps every typer kind
            wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_{a.dataset}_{a.condition}_intent_{I}_seed{seed}" + ("_r85act" if a.activate == "auto" else "_r81" if a.negatives else "_r79" if a.intent_anchor else ""))   # R79 / R81: own folders; earlier ones kept
            retrain = a.retrain_worlds is None or I in a.retrain_worlds.split(",")
            # R85: the intent's command verbs as lemmas of the event 'command' (predicate_role stays false), only those whose typer kind is their own
            # (w:<verb>): the route marks verb CATEGORIES, and a verb of kind O would make every O word a verb; only in the retrained worlds
            vb = [w for w in R85_VERBS.get(I, []) if kI.get(w) == "w:" + w] if (a.declare_verbs and retrain) else []
            if a.declare_verbs and retrain: print(f"[R85] seed {seed} world {I}: command verbs declared {vb}; left out (kind not their own): {[(w, kI.get(w)) for w in R85_VERBS.get(I, []) if w not in vb]}", flush=True)
            anchor = a.intent_anchor and "intent" not in rmap.values()   # R79 (b): the closed-value intent role, one value, in every example (rate 1.0: the core question)
            legend = {"version": 1, "roles": ([{"id": "intent", "question": "what does the user want done?"}] if anchor else []) + [{"id": rmap[t], "question": f"slot {t}"} for t in types_], "events": [{"id": "command", "verbs": ["command"] + vb}], "operators": [],
                      "moods": ["statement"], "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}, "values": [],
                      "reader": dict({"predicate_role": False, "occupancy_marker": True, "definite_dets": [], "no_sense_share": 1.0}, **({"closed_values": {"intent": [I]}} if anchor else {}))}
            if os.path.exists(wd): shutil.rmtree(wd)
            for d in ("examples", "model", "runs"): os.makedirs(os.path.join(wd, d))
            json.dump(legend, open(os.path.join(wd, "legend.json"), "w"), indent=1); sha = sha256_file(os.path.join(wd, "legend.json"))
            with open(os.path.join(wd, "kinds.tsv"), "w") as f:
                f.write(f"# word\tkind: the typer's kinds (R71), slot kinds of other intents mapped to O (R72 world {I})\n")
                for w in sorted(kI): f.write(f"{w}\t{kI[w]}\n")
            json.dump({"name": os.path.basename(wd), "language": "en", "description": f"{a.dataset}: the intent {I}", "version": 1}, open(os.path.join(wd, "world.json"), "w"), indent=1)
            neg = []
            if a.negatives:   # R81: the OTHER intents' training utterances, as many as the positives, sampled with the seed; anchor role only, _negative=true
                others = [i for i in cond["train"] if items[i]["intent"] != I]
                neg = [others[j] for j in sorted(np.random.default_rng(seed).choice(len(others), min(len(tr), len(others)), replace=False))]
            for fname, ids_ in (("train.sop", tr), ("calibration.sop", cal)) + ((("negatives.sop", neg),) if neg else ()):
                doc = soplang.Doc(directives=[("world", os.path.basename(wd), None), ("legend", sha, None), ("generated", "vss/run_reader_real.py --worlds per_intent (R72)", None)]); g = idgen()
                for i in ids_:
                    it = items[i]; w = soplang.Wire(next(g), "command"); w.set("_text", " ".join(it["tokens"])); w.set("_item", it["id"]); seen = set()
                    if anchor: w.set("intent", I)
                    if fname == "negatives.sop": w.set("_negative", "true"); doc.wires.append(w); continue
                    for sl in sorted(it["tree"]["slots"], key=lambda s_: s_["start"]):
                        r = rmap[sl["type"]]
                        if r in seen: continue
                        seen.add(r); w.set(r, it["tokens"][sl["end"] - 1])
                    doc.wires.append(w)
                with open(os.path.join(wd, "examples", fname), "w") as f: f.write(soplang.write(doc))
            L = importlib.import_module("sop_repo.legend").Legend(os.path.join(wd, "legend.json")) if a.negatives else Legend(os.path.join(wd, "legend.json"))   # R81: the tool of sop @ SOURCE end to end
            if L.errors: raise SystemExit(f"legend errors ({I}): " + "; ".join(L.errors))
            TR = importlib.import_module("sop_repo.train") if a.negatives else strain
            rows, errors, _ = TR.load_rows(wd, L)
            if errors: raise SystemExit(f"examples rejected ({I}):\n" + "\n".join(errors[:10]))
            tset, cset = set(tr), set(cal); splits = collections.defaultdict(list)
            for r in rows:
                iid = r["wire"].value("_item"); r["split"] = "train" if iid in tset or r.get("negative") else ("calibration_dev" if iid in cset else "unused"); splits[r["split"]].append(r["id"])
            trr = [r for r in rows if r["split"] == "train" and not r.get("negative")]   # the schema from the positive rows only (R81)
            bs = importlib.import_module("sop_repo.train").build_schema if a.intent_anchor else strain.build_schema   # R79 (a): the schema of sop @ SOURCE (no silent anchor; it warns)
            schema = fix_schema(bs(trr, L)); schema["content_kinds"] = sorted(strain.content_kinds_of(trr, L))
            if a.intent_anchor: print(f"[R79] seed {seed} world {I}: core roles {schema['core_roles']}, structure anchor {schema.get('structure_anchor')}", flush=True)
            ns = types.SimpleNamespace(D=a.D, star_seed=f"seed{seed}", identities="stars", device=a.device, length_buckets=True, epochs=int(a.epochs.split(",")[-1]), batch=a.batch, shuffle=True,
                                       max_train=0, code_list=["path", "chainL", "bigram"], no_ckpt=True, resume=False, log_every=500, lr=a.lr, n_cal=a.n_cal, tau=0.05,
                                       world_name=os.path.basename(wd), language="en", commit=mainrepo.check_pinned(), depth_estimator="kinds", threshold_rule="fit", store_presence_ratios=False)
            if a.negatives: ns.decoder = "paper"   # R81: inference unchanged from R79 (no contract keys in config.json)
            md = os.path.join(wd, "model_seed"); tw = time.time(); R = None
            if a.activate == "auto":   # R84 item 2: the acceptance test of sop activate.py per world (need signal on the positive training rows)
                ACT = importlib.import_module("sop_repo.activate"); ns.seed = seed; ns.tau = 0.05
                if not getattr(ACT, "_r85_pos_only", False):   # the profile (need signals) from the POSITIVE rows: negatives never enter the schema or the profile (R81)
                    _prof = ACT.profile; ACT.profile = lambda rows_, *x, **k: _prof([r for r in rows_ if not r.get("negative")], *x, **k); ACT._r85_pos_only = True
                needs = ACT.need_signals(ACT.profile(rows, L, schema)); fires = [c for c in ACT.CANDIDATE_CODES if needs[c][0]]
                print(f"[R84] seed {seed} world {I}: need signals " + "; ".join(f"{c} {'FIRES' if needs[c][0] else 'no'} ({needs[c][1]})" for c in ACT.CANDIDATE_CODES), flush=True)
                if not fires or not retrain:   # no candidate (or a world not retrained, R85): the R81 configuration exactly; its saved model is copied, not retrained (declared)
                    src_md = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_{a.dataset}_{a.condition}_intent_{I}_seed{seed}_r81", "model_seed")
                    shutil.copytree(src_md, md); R = SR.infer.Reader(md); print(f"[R84] seed {seed} world {I}: no candidate fires: the R81 model copied ({src_md})", flush=True)
                else:
                    if "vorder" in fires:   # the route's model: the reference V8 with set_vorder (sop/v8_gates_vorder_ref.py), in place of the pinned v8_gates
                        spec = importlib.util.spec_from_file_location("v8_gates", os.path.join(SOP_DIR or mainrepo.ROOT, "sop", "v8_gates_vorder_ref.py"))
                        v8r = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8r); sys.modules["v8_gates"] = v8r
                    trs = [r for r in rows if r["split"] == "train" and not r.get("negative")]
                    lexical, _ = TR.lexical_baseline(trs, L, schema); bag = TR.bag_baseline(trs, L, schema)
                    seen = {w for r in trs for fl in r["frame"]["roles"].values() for w in sworld.filler_words(fl)}
                    ev_pos = lambda rd_, rows_, *x, **k: TR.evaluate(rd_, [r for r in rows_ if not r.get("negative")], *x, **k)   # negatives are never scored
                    codes, act = ACT.select(ns, L, rows, splits, schema, TR.train_one, ev_pos, lexical, bag, seen, eps=0.01)
                    ns.code_list = list(codes); ns.activation_report = act
                    json.dump(act, open(os.path.join(wd, "activation.json"), "w"), indent=1, default=str)
                    print(f"[R84] seed {seed} world {I}: accepted {act.get('accepted')}; codes {codes}", flush=True)
            if R is None: R, ex = TR.train_one(ns, L, rows, splits, schema, seed, md, os.path.join(wd, "runs"))
            tw = time.time() - tw
            print(f"[R81] seed {seed} world {I}: {len(tr)} positives, {sum(1 for r in rows if r.get('negative'))} negatives; training {tw:.1f} s (concurrent)", flush=True)
            z = np.load(os.path.join(md, "routebook.npz")); np_ = int(sum(z[k].size for k in ("path", "bg0", "bg", "A", "L", "logits"))); nparam += np_
            RR = SR.infer.Reader(md); RR.cfg["residual"] = "explained"
            s0 = items[tr[0]]["tokens"]; d0 = float(np.linalg.norm(R.chart(s0)[0] - RR.chart(s0)[0]) / max(1e-9, np.linalg.norm(R.chart(s0)[0])))
            worlds[I] = {"R": RR, "rinv": rinv, "rates": span_rates(cond, tr), "types": types_, "cal": cal, "train": tr, "params": np_}
            print(f"[R72] seed {seed} world {I}: {len(tr)} training utterances, {len(types_)} slot questions, {len(set(kI.values()))} kinds, {np_ / 1e6:.2f} M phases; chart of sop/ vs the pinned tool: relative difference {d0:.1e}", flush=True)
        t_train = time.time() - t0
        def read_all(it):
            text = " ".join(it["tokens"]); outs = {}
            for I, W in worlds.items():
                wires, f, info = W["R"].formalize(text, 0, SR.infer.idgen("e"), filt=set(it["tokens"]))
                fr = {}
                for r in W["R"].schema["role_order"]:
                    fl = f["roles"].get(r)
                    if fl is not None and fl[1] is not None: fr[W["rinv"].get(r, r)] = fl[1]
                mm = [float(v) for v in f.get("margins", {}).values()]
                outs[I] = {"frame": fr, "status": info["status"], "residual": float(info["residual"]), "min_margin": min(mm) if mm else 0.0,
                           "margins": {W["rinv"].get(r, r): float(f["margins"].get(r, 0.0)) for r in W["R"].schema["role_order"]}}
            return outs
        thr = {}
        for I, W in worlds.items():
            if a.dispatch == "residual":   # R75: the 98th percentile of the world's residual on ITS OWN training utterances (200, fixed order), read with the full lexicon
                res = [float(W["R"].formalize(" ".join(items[i]["tokens"]), 0, SR.infer.idgen("e"), filt=set(items[i]["tokens"]))[2]["residual"]) for i in W["train"][:200]]
            else: res = [read_all(items[i])[I]["residual"] for i in W["cal"][:200]]
            thr[I] = float(np.quantile(res, 0.98)) if res else 1.0
        def predict_w(ids):
            out = []
            for i in ids:
                it = items[i]; outs = read_all(it); order = sorted(outs, key=(lambda I: (outs[I]["residual"], -outs[I]["min_margin"])) if a.dispatch == "residual" else (lambda I: (rank(outs[I]["status"]), outs[I]["residual"], -outs[I]["min_margin"])))
                b = outs[order[0]]; fr = dict(b["frame"]); fr["intent"] = order[0]; mg = dict(b["margins"])
                mg["intent"] = (outs[order[1]]["residual"] - b["residual"]) if len(order) > 1 else 1.0
                rej = all(o["status"] != "ok" or o["residual"] > thr[I] for I, o in outs.items())
                rej_res = all(o["residual"] > thr[I] for I, o in outs.items())          # second line: the residual alone, status ignored
                if a.dispatch == "residual": rej, rej_status = rej_res, all(o["status"] != "ok" for o in outs.values())   # R75: the residual first, status a second line
                else: rej_status = rej
                p = {"frame": fr, "margins": mg, "status": b["status"], "residual": b["residual"], "rejected_by_every_world": rej, "rejected_by_residual_alone": rej_res, "rejected_by_status": rej_status, "chosen_abstains": b["status"] == "abstain",
                     "spans": spans_by_typer(it["tokens"], fr, ty, T), "spans_by_rule": spans_by_rule(it["tokens"], fr, worlds[order[0]]["rates"]), "spans_typer_only": spans_typer_only(it["tokens"], ty)}
                out.append(p)
            return out
        cal_p = predict_w(cond["dev_cal"]); preds = {k: predict_w(ids) for k, ids in ev.items()}
        sub = (ev.get("dev_rep") or next(iter(ev.values())))[:a.cost_n]; t1 = time.time(); predict_w(sub); ms_ = 1000 * (time.time() - t1) / max(1, len(sub))
        rej = {k: float(np.mean([p["rejected_by_every_world"] for p in v])) for k, v in preds.items()}
        rej_r = {k: float(np.mean([p["rejected_by_residual_alone"] for p in v])) for k, v in preds.items()}
        rej_s = {k: float(np.mean([p["rejected_by_status"] for p in v])) for k, v in preds.items()}; abst = {k: float(np.mean([p["chosen_abstains"] for p in v])) for k, v in preds.items()}
        cost = {"params_7_readers": nparam, "params_typer": ty.n_params(), "bytes_4bit_readers": nparam // 2, "cpu_ms_per_utterance_all_worlds": ms_, "train_s": round(t_train, 1), "D": a.D}
        row = report_real.evaluate("VSS, one world per intent + dispatcher (R72)" if a.dispatch == "status" else ("VSS, one world per intent + intent anchor + negatives, residual dispatch (R81)" if a.negatives else "VSS, one world per intent + intent anchor, residual dispatch (R79)" if a.intent_anchor else "VSS, one world per intent, residual dispatch (R75)"), "reader_worlds" if a.dispatch == "status" else ("reader_worlds_r85act" if a.activate == "auto" else "reader_worlds_r81" if a.negatives else "reader_worlds_r79" if a.intent_anchor else "reader_worlds_r75"), cond, seed, None, preds, cond["dev_cal"], cal_p, sorted(all_types),
                                   extra={"cost": cost, "rejected_by_every_world": rej, "rejected_by_residual_alone": rej_r, "rejected_by_status": rej_s, "chosen_abstains": abst, "world_thresholds": thr, "worlds": intents, "world_kinds": a.world_kinds, "dispatch": a.dispatch})
        print(f"  seed {seed}: rejected by every world {rej}; by the residual alone {rej_r}; by status {rej_s}; chosen frame abstains {abst}; cost {nparam / 1e6:.2f} M phases (7 readers, D {a.D}) + {ty.n_params() / 1e6:.2f} M typer weights; {ms_:.1f} ms per utterance for all worlds", flush=True)
        allrows.append(row)
    report_real.summarize(allrows, "VSS, one world per intent + dispatcher (R72)")
    for key in ("rejected_by_status", "chosen_abstains"):
        for k in allrows[0].get(key, {}): print(f"  {key}, {k}: {np.mean([r[key][k] for r in allrows]):.3f} [{' '.join(f'{r[key][k]:.3f}' for r in allrows)}]")
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    for k in allrows[0]["rejected_by_every_world"]: print(f"  rejected by every world, {k}: {ms([r['rejected_by_every_world'][k] for r in allrows])}")
    for k in allrows[0]["rejected_by_residual_alone"]: print(f"  rejected by the residual of every world (status ignored), {k}: {ms([r['rejected_by_residual_alone'][k] for r in allrows])}")
    metrics.checklist(a.seeds, "[x] worlds from the training annotation only [x] thresholds on the dev calibration half [x] three routes only (R73)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=rd.DATASETS); ap.add_argument("--condition", default="base"); ap.add_argument("--locale", default="en-US")
    ap.add_argument("--sizes", default="1000,5000,full"); ap.add_argument("--epochs", default="60,20,10", help="per size, in the order of --sizes")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--unseen_kind", default="cluster", choices=["cluster", "unknown"])
    ap.add_argument("--n_cal", type=int, default=400); ap.add_argument("--device", default="cuda"); ap.add_argument("--test", action="store_true")
    ap.add_argument("--max_eval", type=int, default=None, help="smoke tests only: the first n items of each eval set"); ap.add_argument("--max_train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--save_model", action="store_true", help="copy the seed-0 model/ folder of each size to logs/models/")
    ap.add_argument("--kinds", default="skipgram", choices=["skipgram", "typer"], help="R71: 'typer' = the kinds learned from the training annotation by sop/typer.py (default: the skip-gram clusters)")
    ap.add_argument("--worlds", default="one", choices=["one", "per_intent"], help="R72 item 4: one world per intent with the dispatcher (needs --kinds typer)")
    ap.add_argument("--world_kinds", default="own", choices=["own", "all"], help="R75: 'all' keeps every typer kind in every per-intent world (default 'own' = R72 as run)")
    ap.add_argument("--dispatch", default="status", choices=["status", "residual"], help="R75: 'residual' = dispatch by the explained residual, ties by the min margin; rejection by the residual (default = R72 as run)")
    ap.add_argument("--intent_anchor", action="store_true", help="R79: keep the closed-value intent role (one value) in every per-intent world and example; schema by sop @ SOURCE")
    ap.add_argument("--negatives", action="store_true", help="R81: one-vs-rest negatives (other intents' training utterances, balanced, anchor role only, _negative=true); trained by sop @ SOURCE; needs --intent_anchor")
    ap.add_argument("--activate", default="off", choices=["off", "auto"], help="R84 item 2: sop activate.py's acceptance test per world (with --negatives); folders *_r84act")
    ap.add_argument("--declare_verbs", action="store_true", help="R85: the intents' command verbs as lemmas of the event 'command' in the retrained worlds' legends")
    ap.add_argument("--retrain_worlds", default=None, help="R85: comma list of the worlds trained (acceptance test); the others copy their R81 model")
    ap.add_argument("--sop_dir", default=None, help="a directory holding the 'sop' package of the version to train / read with (R82)")
    ap.add_argument("--seed_start", type=int, default=0, help="first seed (seeds seed_start .. seed_start + seeds - 1), to split seeds across processes")
    a = ap.parse_args(); dev_ = a.device
    assert not a.negatives or a.intent_anchor, "--negatives needs --intent_anchor (R81)"
    assert a.activate == "off" or a.negatives, "--activate auto is declared on the R81 configuration (--negatives)"
    SOP_DIR = a.sop_dir
    cond = rd.load_condition(a.dataset, a.condition, a.locale); items = cond["items"]
    if a.max_train: cond["train_order"] = cond["train_order"][:a.max_train]; keep = set(cond["train_order"]); cond["train"] = [i for i in cond["train"] if i in keep]
    sizes = a.sizes.split(",") if a.condition == "base" else ["full"]; eps = dict(zip(a.sizes.split(","), [int(x) for x in a.epochs.split(",")]))
    if a.condition != "base": eps = {"full": int(a.epochs.split(",")[-1])}
    nested = any(it["depth"] > 0 for it in items.values())
    print(f"run_reader_real: main repository {mainrepo.check_pinned()}; dataset {a.dataset} / condition {a.condition}; device {dev_}; D {a.D}; batch {a.batch}; lr {a.lr}; "
          f"sizes {sizes}, epochs {eps}; seeds {a.seeds}; train {len(cond['train'])}, dev calibration {len(cond['dev_cal'])}; eval " + ", ".join(f"{k} {len(v)}" for k, v in report_real.eval_sets(cond, a.test).items()))
    print("regime (rule 24): real utterances, the dataset's own annotation; head-token frame (intent + the LAST token of the first span of each slot type); "
          "the intent as a closed-value role, slots as roles with the occupancy marker; thresholds on the dev calibration half; test only with --test" + ("; nested intents read by the wrapper" if nested else ""))
    if a.kinds == "typer": kinds, ty_, T_ = build_kinds_typer(cond, a); TYPER = (ty_, T_)
    else: kinds = build_kinds(cond, a, dev_)
    ev = report_real.eval_sets(cond, a.test, a.max_eval)
    if a.worlds == "per_intent":
        assert a.kinds == "typer", "--worlds per_intent needs --kinds typer (R72)"
        per_intent(cond, a, kinds, ty_, T_, ev); sys.exit(0)
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
        for seed in range(a.seed_start, a.seed_start + a.seeds):
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
