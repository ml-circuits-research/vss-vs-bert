"""
legend.py — the world's closed vocabulary (spec 1): roles, events with their synonym table, operators, moods, kinds (word categories)
from a lexicon, value patterns; plus the `reader` block (an extension of the informal schema, see README): how a frame is written as a
target state (predicate role on/off, occupancy marker on/off, determiners that add no term).

Lexicon (kinds.tsv, next to legend.json): one word per line, `word<TAB>kind[<TAB>lemma]`; '#' comments. The kind is the word's category
(rule 20: categories are GIVEN, never learned); the lemma maps a surface form to the legend's verb (gives -> give) or a negative affix form
to its positive (unsaved -> saved).
"""
import hashlib, json, os, re

OPERATOR_KEYS = {"not": {"scope"}, "must": {"scope"}, "may": {"scope"}, "if": {"condition", "consequence"}, "says": {"speaker", "scope"},
                 "and": {"items"},
                 # discourse operators of the SOP circuit (VSS Diagram Editor spec 15.2; app/src/segment/circuit.js); vss-demo addition
                 "then": {"after", "scope"}, "but": {"after", "scope"}, "so": {"cause", "scope"}, "because": {"scope", "cause"}}
ENTITY_KEYS = {"head", "attr", "quantity", "unit", "det"}
VERDICT_KEYS = {"left", "right"}
KNOWN_OPERATORS = set(OPERATOR_KEYS)
READER_DEFAULTS = {"predicate_role": True, "occupancy_marker": True, "definite_dets": ["the"], "scope_terms": None, "scope_rule": None,
                   "rel_emitters": None, "full_excludes": [], "gates_given": None, "level_openers": None,
                   "value_identity": "last_token", "closed_values": None, "table_roles": None, "set_roles": None, "value_terms": False,
                   "no_sense_share": None, "codes": None, "content_kinds": None,
                   # "kind_run": a filler is the maximal run of its (typer) kind around the decoded head (real data, multi-word slots); the
                   # run's words are explained by the filler and are never 'adjacent competitors' of the adjacency rule (Q38 / R87)
                   "spans": "token",
                   # "lemma": every token is deposited and read by the identity of its lexicon lemma (kinds.tsv third column), so that the
                   # surface forms of several languages map to the one canonical value of the legend (roșie, red -> red); vss-demo addition
                   "identity": "token"}
PUNCT = re.compile(r"([.,;:!?](?=\s|$))")


class LegendError(Exception): pass


def sha256_file(path):
    with open(path, "rb") as f: return "sha256:" + hashlib.sha256(f.read()).hexdigest()


class Legend:
    def __init__(self, path):
        self.path = os.path.abspath(path); self.dir = os.path.dirname(self.path)
        with open(path, encoding="utf-8") as f: d = json.load(f)
        self.raw = d; self.sha = sha256_file(path); self.version = d.get("version")
        self.roles = [r["id"] for r in d.get("roles", [])]; self.questions = {r["id"]: r.get("question", "") for r in d.get("roles", [])}
        self.events = {e["id"]: list(e.get("verbs", [])) for e in d.get("events", [])}
        self.operators = list(d.get("operators", [])); self.moods = list(d.get("moods", ["statement"]))
        self.reader = {**READER_DEFAULTS, **d.get("reader", {})}
        self.values = [(v["kind"], re.compile("^(?:" + v["pattern"] + ")$")) for v in d.get("values", [])]
        self.errors = self._validate(); self.warnings = []
        self.verb_event = {v: e for e, vs in self.events.items() for v in vs}
        kinds = d.get("kinds", {}); self.kinds_source = kinds.get("source", "lexicon")
        self.lexicon, self.lemma = {}, {}
        if self.kinds_source == "lexicon":
            lp = os.path.join(self.dir, kinds.get("lexicon", "kinds.tsv"))
            if os.path.exists(lp): self._load_lexicon(lp)
            else: self.warnings.append(f"lexicon {lp} not found (kinds unknown; sop check does not need them, training does)")
        else:
            self.errors.append("kinds.source 'tagger' is not implemented in v1 (use a lexicon)")   # limit of v1 (spec 6.3 allows a tagger; none is installed)
        kinds_known = set(self.lexicon.values())
        if kinds_known:      # kinds named by the reader block must exist (sop check: a legend error)
            gg = self.reader.get("gates_given") or {}
            for side in ("open", "closed"):
                for k in gg.get(side, []):
                    if k not in kinds_known: self.errors.append(f"reader.gates_given.{side}: kind {k!r} is not a kind of the lexicon")
            for k in set(gg.get("open", [])) & set(gg.get("closed", [])): self.errors.append(f"reader.gates_given: kind {k!r} both open and closed")
            for k in self.reader.get("level_openers") or []:
                if k not in kinds_known: self.errors.append(f"reader.level_openers: kind {k!r} is not a kind of the lexicon")
            re_ = self.reader.get("rel_emitters") or {}
            for k in list(re_.get("emitters", [])) + list(re_.get("takers", [])):
                if k not in kinds_known: self.errors.append(f"reader.rel_emitters: kind {k!r} is not a kind of the lexicon")

    def _validate(self):
        e = []
        for r in self.roles:
            if r.startswith("_"): e.append(f"role {r!r}: a legend may not define roles starting with '_'")
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", r): e.append(f"role {r!r}: bad id")
        if len(set(self.roles)) != len(self.roles): e.append("duplicate role ids")
        for ev in self.events:
            if ev in KNOWN_OPERATORS or ev in ("entity", "unknown", "verdict"): e.append(f"event id {ev!r} collides with a reserved wire type")
        for ev in self.raw.get("events", []):
            if not ev.get("verbs"): e.append(f"event {ev.get('id')!r} has no verbs (spec 6.1: an event without verbs is a legend inconsistency)")
        for op in self.operators:
            if op not in KNOWN_OPERATORS: e.append(f"operator {op!r} is not one of {sorted(KNOWN_OPERATORS)}")
        return e

    def _load_lexicon(self, path):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"): continue
                parts = line.split("\t")
                if len(parts) < 2: self.errors.append(f"{path}:{n}: expected word<TAB>kind[<TAB>lemma]"); continue
                self.lexicon[parts[0]] = parts[1]
                if len(parts) > 2 and parts[2]: self.lemma[parts[0]] = parts[2]

    # -------- words --------
    def kind(self, tok):
        if tok in self.lexicon: return self.lexicon[tok]
        if tok.lower() in self.lexicon: return self.lexicon[tok.lower()]
        for k, pat in self.values:
            if pat.match(tok): return k
        return None
    def lemma_of(self, tok): return self.lemma.get(tok, self.lemma.get(tok.lower(), tok))
    def event_of_lemma(self, lemma): return self.verb_event.get(lemma)
    def all_verbs(self): return sorted(self.verb_event)
    def wire_types(self):
        return set(self.events) | {"entity", "unknown"} | set(self.operators)
    def keys_for(self, wtype):
        if wtype in self.events: return set(self.roles)
        if wtype == "entity": return ENTITY_KEYS
        if wtype in OPERATOR_KEYS: return OPERATOR_KEYS[wtype]
        if wtype == "verdict": return VERDICT_KEYS
        return set()


def tokenize(text):
    """whitespace tokens with punctuation split off (the constructed worlds write ' .' and ' ,' already)"""
    return PUNCT.sub(r" \1 ", text).split()
