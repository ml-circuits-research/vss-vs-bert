"""real_data.py: the real-data round (INSTRUCTIONS_REAL_2026-10-24.md). Loaders for SNIPS (2017-06 custom intent engines), MASSIVE
(en-US or another locale) and TOPv2 (or the original TOP), the conversion of their annotations into one item format, the head-token
frame, BIO tags, delexicalised templates, and the frozen split CONDITIONS (data/real/splits_<dataset>.json: ids only, plus the seed and
the rule; no text of the datasets is ever written to the repository).

Item (one utterance):
  {id, partition: train | dev | test, tokens: [str], intent, slots: [(type, start, end, value)] (top level, every span, repeats kept),
   tree: {intent, start, end, slots: [{type, start, end, child: tree | None}]}, depth: number of intents nested under the top one}
For the flat datasets (SNIPS, MASSIVE) the tree has one level and every child is None.

Tokens (stated rule): the dataset's whitespace tokens, lowercased, each split further by the main repository's tool tokenizer
(tool/sop/legend.tokenize: trailing . , ; : ! ? split off) applied to a fixed point, so that ' '.join(tokens) tokenizes back to the same tokens in the reader.
A span covers the normalised tokens of the raw tokens it covered, minus trailing tokens that are only punctuation (stated rule: 'p.m.'
-> 'p.m' '.' keeps the head 'p.m').

Frame (what every model is scored on, eval/metrics.score_real):
  head frame = {"intent": intent, <slot type>: HEAD token | nested head frame}, one entry per slot type and level: the FIRST span of a
  type (by position) is kept when a type repeats (the share of repeats is reported by data/real/make_real_splits.py); the head of a span
  is its LAST token (the token identity the reader deposits: tool README 'multi-word values are read by their last token').
  flatten(frame) -> {path: value}: "intent", "<slot>", "<slot>/intent", "<slot>/<sub slot>", ...; a nested slot has the value "@frame".
"""
import collections, glob, hashlib, json, os, re
import numpy as np
from . import mainrepo

RAW = os.environ.get("REAL_DATA", os.path.join(mainrepo.ROOT, "data", "real", "raw"))
SPLIT_DIR = os.environ.get("REAL_SPLITS", os.path.join(mainrepo.ROOT, "data", "real"))
DATASETS = ("snips", "massive", "topv2")
SNIPS_INTENTS = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
NESTED = "@frame"
PUNCT_TOKENS = {".", ",", ";", ":", "!", "?"}

_tok = None
def tool_tokenize(s):
    global _tok
    if _tok is None: mainrepo.import_sop(); from sop.legend import tokenize as t; _tok = t
    return _tok(s)

def tokenize_fixed(w):
    """the tool tokenizer applied to a word until nothing changes ('wife...' -> 'wife' '.' '.' '.'): one pass is not idempotent on
    repeated trailing punctuation, the fixed point is, so ' '.join(tokens) tokenizes back to the same tokens in the reader"""
    cur = [w]
    while True:
        nxt = [p for x in cur for p in (tool_tokenize(x) or [x])]
        if nxt == cur: return cur
        cur = nxt

def h01(*parts):
    """a deterministic number in [0, 1) from the parts (the same function as tool/sop/train.h01)"""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") / 2 ** 64

def normalise(raw_tokens):
    """raw whitespace tokens -> (tokens, spans of each raw token in the normalised tokens)"""
    toks, where = [], []
    for w in raw_tokens:
        p = tokenize_fixed(w.lower())
        where.append((len(toks), len(toks) + len(p))); toks += p
    return toks, where

def make_item(iid, part, raw_tokens, intent, raw_tree):
    """raw_tree spans are in RAW token indices; converted to normalised indices"""
    toks, where = normalise(raw_tokens)
    def conv(node):
        s = where[node["start"]][0] if node["start"] < len(where) else len(toks)
        e = where[node["end"] - 1][1] if node["end"] > node["start"] else s
        out = {"intent": node["intent"], "start": s, "end": e, "slots": []}
        for sl in node["slots"]:
            if sl["end"] <= sl["start"]: continue                      # an empty slot carries no token: dropped (counted by the loaders)
            ss, se = where[sl["start"]][0], where[sl["end"] - 1][1]
            while se > ss + 1 and toks[se - 1] in PUNCT_TOKENS: se -= 1    # trailing punctuation split off by the tokenizer is not part of a value
            out["slots"].append({"type": sl["type"], "start": ss, "end": se, "child": conv(sl["child"]) if sl["child"] else None})
        return out
    tree = conv(raw_tree)
    slots = [(sl["type"], sl["start"], sl["end"], " ".join(toks[sl["start"]:sl["end"]])) for sl in tree["slots"]]
    return {"id": iid, "partition": part, "tokens": toks, "intent": intent, "slots": slots, "tree": tree, "depth": depth(tree)}

def depth(tree):
    ch = [depth(sl["child"]) + 1 for sl in tree["slots"] if sl["child"]]
    return max(ch) if ch else 0

# ------------------------------------------------------------------ MASSIVE
ANNOT = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")

def parse_massive_annot(annot):
    """'wake me up at [time : five am] [date : this week]' -> (raw tokens, [(type, start, end)]) in raw token indices"""
    toks, spans, pos = [], [], 0
    for m in ANNOT.finditer(annot):
        toks += annot[pos:m.start()].split(); v = m.group(2).split()
        spans.append((m.group(1).strip(), len(toks), len(toks) + len(v))); toks += v; pos = m.end()
    toks += annot[pos:].split()
    return toks, spans

def massive_annot_of(toks, spans):
    """the inverse (for the round-trip test): raw tokens + spans -> the annot_utt string with single spaces"""
    out, i, by = [], 0, {s: (t, e) for t, s, e in spans}
    while i < len(toks):
        if i in by: t, e = by[i]; out.append(f"[{t} : {' '.join(toks[i:e])}]"); i = e
        else: out.append(toks[i]); i += 1
    return " ".join(out)

def massive_path(locale):
    env = os.environ.get("MASSIVE_DATA")
    cands = ([os.path.join(env, f"{locale}.jsonl")] if env else []) + sorted(glob.glob(os.path.join(RAW, "massive", "**", f"{locale}.jsonl"), recursive=True))
    for c in cands:
        if os.path.exists(c): return c
    raise FileNotFoundError(f"MASSIVE {locale}.jsonl not found under {os.path.join(RAW, 'massive')} (run: python3 data/real/download.py massive) or $MASSIVE_DATA")

def load_massive(locale="en-US", path=None):
    items, stats = [], collections.Counter()
    for line in open(path or massive_path(locale), encoding="utf-8"):
        r = json.loads(line); toks, spans = parse_massive_annot(r["annot_utt"])
        if toks != r["utt"].split(): stats["utt_tokens_differ_from_annot_tokens"] += 1
        part = {"train": "train", "dev": "dev", "test": "test"}.get(r["partition"])
        if part is None: stats[f"partition_{r['partition']}_skipped"] += 1; continue
        tree = {"intent": r["intent"], "start": 0, "end": len(toks), "slots": [{"type": t, "start": s, "end": e, "child": None} for t, s, e in spans if e > s]}
        stats["empty_span"] += sum(1 for _, s, e in spans if e <= s)
        items.append(make_item(f"massive:{r['id']}", part, toks, r["intent"], tree))
    return items, dict(stats)

# ------------------------------------------------------------------ SNIPS
def snips_files():
    d = os.path.join(RAW, "snips")
    return {I: {"train": os.path.join(d, I, f"train_{I}_full.json"), "validate": os.path.join(d, I, f"validate_{I}.json")} for I in SNIPS_INTENTS}

def read_json_any(path, stats):
    b = open(path, "rb").read()
    try: return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError: stats["files_decoded_latin1"] += 1; return json.loads(b.decode("latin-1"))

def snips_chunks_to_tokens(chunks):
    """SNIPS 'data' chunks [{text, entity?}] -> (tokens, [(type, start, end)], misaligned count). The text is cut into whitespace tokens,
    each lowercased and split by the tool tokenizer, every piece with its character offsets; a slot covers the pieces that overlap its
    characters; a slot boundary inside a piece is counted as misaligned (the whole piece is taken). The tokens returned are already
    normalised (normalise() leaves them unchanged)."""
    text, ents, pos = "", [], 0
    for c in chunks:
        t = c["text"]
        if c.get("entity"):
            lead = len(t) - len(t.lstrip()); core = t.strip()
            if core: ents.append((c.get("entity"), pos + lead, pos + lead + len(core)))
        text += t; pos += len(t)
    pieces = []
    for m in re.finditer(r"\S+", text):
        w = m.group(0).lower(); q = 0
        for p in tokenize_fixed(w):
            k = w.find(p, q); k = q if k < 0 else k; pieces.append((p, m.start() + k, m.start() + k + len(p))); q = k + len(p)
    spans, mis = [], 0
    for typ, cs, ce in ents:
        idx = [k for k, (_, s, e) in enumerate(pieces) if s < ce and e > cs]
        if not idx: mis += 1; continue
        if pieces[idx[0]][1] != cs or pieces[idx[-1]][2] != ce: mis += 1
        spans.append((typ, idx[0], idx[-1] + 1))
    return [w for w, _, _ in pieces], spans, mis

def load_snips():
    """train_<I>_full.json -> partition 'train_full' (train and the dev carved from it, see make_real_splits.py); validate_<I>.json ->
    'test' (the benchmark's held-out queries). Ids: snips:<file kind>:<intent>:<index in the file>."""
    items, stats = [], collections.Counter()
    for I, fs in snips_files().items():
        for kind, p in fs.items():
            if not os.path.exists(p): raise FileNotFoundError(f"{p} missing (run: python3 data/real/download.py snips)")
            d = read_json_any(p, stats)
            for k, q in enumerate(d[I]):
                toks, spans, mis = snips_chunks_to_tokens(q["data"]); stats["misaligned_spans"] += mis
                ov = [(s, e) for _, s, e in spans]; stats["overlapping_spans"] += sum(1 for a in ov for b in ov if a < b and a[1] > b[0])
                tree = {"intent": I, "start": 0, "end": len(toks), "slots": [{"type": t, "start": s, "end": e, "child": None} for t, s, e in spans]}
                items.append(make_item(f"snips:{kind}:{I}:{k}", "train_full" if kind == "train" else "test", toks, I, tree))
    return items, dict(stats)

# ------------------------------------------------------------------ TOPv2 / TOP
def parse_top(parse):
    """'[IN:A w w [SL:B [IN:C w ] ] ]' -> (raw tokens, tree in raw indices). Raises ValueError on a malformed string."""
    parts = parse.split(); toks = []; stack = []; root = None
    for p in parts:
        if p.startswith("[IN:") or p.startswith("[SL:"):
            node = {"kind": p[1:3], "label": p[4:], "start": len(toks), "children": []}
            if stack: stack[-1]["children"].append(node)
            elif root is None: root = node
            else: raise ValueError("two roots")
            stack.append(node)
        elif p == "]":
            if not stack: raise ValueError("unbalanced ]")
            stack.pop()["end"] = len(toks)
        else: toks.append(p)
    if stack or root is None or root["kind"] != "IN": raise ValueError("unbalanced or no intent root")
    def conv(n):
        out = {"intent": n["label"], "start": n["start"], "end": n["end"], "slots": []}
        for c in n["children"]:
            if c["kind"] != "SL": raise ValueError("intent directly inside an intent")
            ins = [x for x in c["children"] if x["kind"] == "IN"]
            if len(ins) > 1: raise ValueError("two intents in one slot")
            out["slots"].append({"type": c["label"], "start": c["start"], "end": c["end"], "child": conv(ins[0]) if ins else None})
        return out
    return toks, conv(root)

def top_string(toks, tree):
    """the inverse of parse_top on raw tokens (for the round-trip test); words a slot holds besides its nested intent stay in place"""
    def emit(kind, label, s, e, children):
        out, i = [f"[{kind}:{label}"], s
        for cs, ce, f in sorted(children, key=lambda c: c[0]): out += toks[i:cs]; out.append(f()); i = ce
        return " ".join(out + toks[i:e] + ["]"])
    def intent(n): return emit("IN", n["intent"], n["start"], n["end"], [(sl["start"], sl["end"], (lambda sl=sl: slot(sl))) for sl in n["slots"]])
    def slot(sl):
        c = sl["child"]; return emit("SL", sl["type"], sl["start"], sl["end"], [(c["start"], c["end"], (lambda c=c: intent(c)))] if c else [])
    return intent(tree)

def topv2_files(domains):
    d = os.environ.get("TOPV2_DATA") or next(iter(sorted(glob.glob(os.path.join(RAW, "topv2", "**", "TOPv2_Dataset"), recursive=True))), os.path.join(RAW, "topv2"))
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.tsv"))):
        m = re.match(r"^([a-z]+)_(train|eval|test)\.tsv$", os.path.basename(p))      # the low-resource *_25spis files are excluded
        if m and (domains is None or m.group(1) in domains): out[(m.group(1), m.group(2))] = p
    if not out: raise FileNotFoundError(f"no TOPv2 <domain>_<train|eval|test>.tsv under {d} (run: python3 data/real/download.py topv2) or $TOPV2_DATA")
    return out

def load_topv2(domains=("navigation", "event", "reminder")):
    """TOPv2 tsv (header 'domain utterance semantic_parse'); the tokens are the parse's words (stated: TOPv2's parse is tokenised).
    Ids: topv2:<domain>:<train|eval|test>:<row index>; partition eval -> dev."""
    items, stats = [], collections.Counter()
    for (dom, part), p in topv2_files(set(domains) if domains else None).items():
        with open(p, encoding="utf-8") as f:
            rows = [l.rstrip("\n").split("\t") for l in f]
        if rows and rows[0][:1] == ["domain"]: rows = rows[1:]
        for k, r in enumerate(rows):
            try: toks, tree = parse_top(r[-1])
            except (ValueError, IndexError): stats["malformed_parse"] += 1; continue
            items.append(make_item(f"topv2:{dom}:{part}:{k}", {"eval": "dev"}.get(part, part), toks, tree["intent"], tree))
    return items, dict(stats)

def load(dataset, locale="en-US", domains=None):
    if dataset == "snips": return load_snips()
    if dataset == "massive": return load_massive(locale)
    if dataset == "topv2": return load_topv2(tuple(domains) if domains else ("navigation", "event", "reminder"))
    raise ValueError(dataset)

# ------------------------------------------------------------------ frames, tags, templates
def head_frame(tree, tokens):
    """the head-token frame of a tree (first span per type; head = last token; nested -> the child's frame)"""
    f = {"intent": tree["intent"]}
    for sl in sorted(tree["slots"], key=lambda s: s["start"]):
        if sl["type"] in f: continue
        f[sl["type"]] = head_frame(sl["child"], tokens) if sl["child"] else tokens[sl["end"] - 1]
    return f

def flatten(frame, prefix=""):
    out = {}
    for k, v in frame.items():
        if isinstance(v, dict): out[prefix + k] = NESTED; out.update(flatten(v, prefix + k + "/"))
        elif v is not None: out[prefix + k] = v
    return out

def repeated_types(tree):
    """number of slot spans dropped by the first-span rule, over the whole tree"""
    c = collections.Counter(sl["type"] for sl in tree["slots"])
    return sum(n - 1 for n in c.values()) + sum(repeated_types(sl["child"]) for sl in tree["slots"] if sl["child"])

def bio(n_tokens, spans):
    lab = ["O"] * n_tokens
    for t, s, e in spans:
        for k in range(s, e): lab[k] = ("B-" if k == s else "I-") + t
    return lab

def spans_of_bio(lab):
    """BIO -> [(type, start, end)]; an I- without its B- starts a span (the conlleval convention)"""
    out, cur = [], None
    for k, l in enumerate(list(lab) + ["O"]):
        if cur and (l == "O" or l.startswith("B-") or l[2:] != cur[0]): out.append((cur[0], cur[1], k)); cur = None
        if l != "O" and cur is None: cur = (l[2:], k)
    return out

def template(item):
    """the delexicalised formulation: top-level slot spans replaced by [type]"""
    toks, out, i = item["tokens"], [], 0
    for sl in sorted(item["tree"]["slots"], key=lambda s: s["start"]):
        if sl["start"] < i: continue
        out += toks[i:sl["start"]] + [f"[{sl['type']}]"]; i = sl["end"]
    return " ".join(out + toks[i:])

def all_intents(tree):
    return {tree["intent"]} | {x for sl in tree["slots"] if sl["child"] for x in all_intents(sl["child"])}

def node_examples(item):
    """one example per intent node (the recursive tagger of models/bert_real.py): (tokens of the node span, intent, slot spans relative
    to the node, nested flag per slot)"""
    out = []
    def walk(n):
        toks = item["tokens"][n["start"]:n["end"]]
        out.append({"tokens": toks, "intent": n["intent"], "slots": [(sl["type"], sl["start"] - n["start"], sl["end"] - n["start"], sl["child"] is not None) for sl in n["slots"]]})
        for sl in n["slots"]:
            if sl["child"]: walk(sl["child"])
    walk(item["tree"]); return out

# ------------------------------------------------------------------ splits
def split_path(dataset, locale="en-US"):
    return os.path.join(SPLIT_DIR, f"splits_{dataset}" + ("" if dataset != "massive" or locale == "en-US" else f"_{locale}") + ".json")

def load_condition(dataset, condition="base", locale="en-US", items=None):
    """-> {items: id -> item, train, dev_cal, eval: {name: ids}, train_order (base: the seed-0 permutation), meta}; the split file
    was written by data/real/make_real_splits.py; ids missing from the loaded data are an error (the data changed)"""
    sp = json.load(open(split_path(dataset, locale)))
    if items is None: items, _ = load(dataset, locale, sp.get("domains"))
    by = {it["id"]: it for it in items}
    c = sp["conditions"][condition]
    if not c.get("supported", True): raise SystemExit(f"condition {condition} of {dataset} is not supported by the data: {c.get('reason')}")
    need = set(c["train"]) | set(c["dev_cal"]) | {i for v in c["eval"].values() for i in v}
    miss = [i for i in need if i not in by]
    if miss: raise RuntimeError(f"{len(miss)} ids of {split_path(dataset, locale)} are not in the loaded data (e.g. {miss[:3]}): the data differs from the one the splits were made on")
    return {"items": by, "train": c["train"], "dev_cal": c["dev_cal"], "eval": c["eval"], "train_order": c.get("train_order", c["train"]),
            "new_roles": c.get("new_roles", {}), "meta": {k: v for k, v in c.items() if k not in ("train", "dev_cal", "eval", "train_order", "new_roles")},
            "dataset": dataset, "condition": condition, "sha256": sp.get("sha256")}

def new_value_roles(item, train_vocab):
    """paths of the head frame whose head token never occurs in the training utterances (the 'new slot values' axis)"""
    return sorted(p for p, v in flatten(head_frame(item["tree"], item["tokens"])).items() if p != "intent" and not p.endswith("/intent") and v != NESTED and v not in train_vocab)
