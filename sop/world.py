"""
world.py — what the legend's roles mean as vectors (numpy only), and the canonical frame of a top-level wire.

Stars (identities and role vectors) are unit-modulus complex vectors regenerated from a seed: star(seed, name, D) draws D phases from a
numpy generator seeded by sha256(f"{seed}:{name}"). Identities use names "id:<word>" with the GLOBAL star seed (a word's star is
the same in every world, spec v2 section 0); role vectors "role:<role id>", operator parts "op:<name>" and the marker use the world's seed. Nothing here is learned (rule 20: roles, identities, kinds are GIVEN; the reader learns the route book only).

Target state of a wire (the constructed worlds of the repository, generalized; RESULTS 10b / 10d / 9c):
  event e:   [R_pred (x) f_lemma] + sum_r R_r (x) (val(filler_r) [+ m])          m = occupancy marker (legend reader.occupancy_marker)
  entity:    f_head + R_attr (x) val(attr) [+ R_det (x) f_det unless det is definite] (quantity/unit: values are invisible to the reader)
  not:       R_not (x) val(scope)            must / may: R_must / R_may (x) val(scope)
  if:        R_cond (x) val(condition) + R_then (x) val(consequence)
  says:      R_speaker (x) (f_speaker [+ m]) + R_rep (x) val(scope)
  and:       sum_i R_next^i (x) val(item_i)
  values:    word -> f_word (a multi-word value uses the identity of its LAST token: the token the sentence carries; stated convention);
             $w -> state(w); a & b -> sum; a | b -> sum / sqrt(n) (superposition); ~ -> m (filled, unrecognized); ? -> nothing.
"""
import hashlib, json, re
import numpy as np

ALNUM = re.compile(r"[a-z0-9]+")
def alnum(s): return ALNUM.findall(str(s).lower())

OPERATORS = {"not", "must", "may", "if", "says", "and"}


def star(seed, name, D):
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return np.exp(1j * rng.uniform(-np.pi, np.pi, D))


GLOBAL_STAR_SEED = "sop-stars-v1"      # spec v2 section 0: a word's star is the same in every world (catalog of word stars, shared)


def _xoshiro128ss_phases(key16, D):
    """portable stars (star_version 2, VSS Diagram Editor spec 5.1): xoshiro128** seeded with 4 little-endian uint32 words, each 32-bit output u
    gives the phase 2*pi*(u + 0.5)/2**32 - pi. Reproduced exactly by app/src/vss.js (no NumPy generator in the browser)."""
    M = 0xFFFFFFFF
    s = [int.from_bytes(key16[4 * i:4 * i + 4], "little") for i in range(4)]
    if not any(s): s = [0x9E3779B9, 0x243F6A88, 0xB7E15162, 0x6A09E667]
    rotl = lambda x, k: ((x << k) | (x >> (32 - k))) & M
    out = np.empty(D, dtype=np.float64)
    for i in range(D):
        r = (rotl((s[1] * 5) & M, 7) * 9) & M
        t = (s[1] << 9) & M
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11)
        out[i] = 2 * np.pi * (r + 0.5) / 4294967296.0 - np.pi
    return out


def star_v2(seed, name, D):
    """star_version 2: SHA-256(f"{seed}:{name}") first 16 bytes -> xoshiro128** -> D phases"""
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return np.exp(1j * _xoshiro128ss_phases(h[:16], D))


class Stars:
    """cache of stars: word identities from the GLOBAL star seed (shared across worlds), role / operator vectors and the occupancy marker
    from the world's training seed (the world's named stars; not shared)"""
    def __init__(self, seed, D, star_seed=GLOBAL_STAR_SEED, identity_table=None, star_version=1, star_table=None):
        self.seed, self.D, self.star_seed, self.cache = seed, D, star_seed, {}
        self.star_version = int(star_version or 1)          # 1 = NumPy generator (every existing world), 2 = portable (browser)
        self.table = None; self.named = {}
        if star_table:          # named stars GIVEN by a per-seed draw (vss-vs-bert Q29: the basic world's role / level vectors R_A, R_B, R_C, R_X,
            # R_rep from np.random.default_rng(100 + seed) in that order): catalog.json "star_table": {"rng_seed": int, "order": ["role:A", ...]};
            # read by get() before the hashed star, the same pattern as identity_table
            rng = np.random.default_rng(star_table["rng_seed"])
            for name in star_table["order"]: self.named[name] = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        if identity_table:      # a per-seed identity DRAW over a world vocabulary, as experiment/v16_reference (judgment e9274a9 C2, run C)
            voc = identity_table["vocab"]; rng = np.random.default_rng(identity_table["rng_seed"])
            F = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(voc), D))); self.table = {w: F[i] for i, w in enumerate(voc)}
            self.marker_word = identity_table.get("marker_word")
    def get(self, name, seed=None):
        if name in self.named: return self.named[name]
        seed = self.seed if seed is None else seed; v = self.cache.get((seed, name))
        if v is None: v = self.cache[(seed, name)] = (star_v2 if self.star_version == 2 else star)(seed, name, self.D)
        return v
    def id(self, word):
        t = value_token(word)
        if self.table is not None and t in self.table: return self.table[t]
        return self.get("id:" + t, self.star_seed)
    def val(self, text, key=None, reader=None):
        """the identity of a VALUE: a closed-set symbol (legend reader.closed_values[key]: its own star, 'sym:<value>'); a BUNDLE of the
        lowercase alphanumeric tokens' stars (reader.value_identity = 'bundle_alnum', the WikiSQL world: a column name or a cell value is
        the sum of its words, as experiment/v11_wikisql); else the identity of the last token (the default)"""
        reader = reader or {}
        if key is not None and key in (reader.get("closed_values") or {}): return self.get("sym:" + str(text), self.star_seed)
        if reader.get("value_identity") == "bundle_alnum":
            ts = alnum(text)
            return sum(self.id(t) for t in ts) if ts else np.zeros(self.D, dtype=complex)
        return self.id(text)
    def role(self, r): return self.get("role:" + r)
    def op(self, name): return self.get("op:" + name)
    def marker(self):
        if self.table is not None and getattr(self, "marker_word", None) in self.table: return self.table[self.marker_word]
        return self.get("op:marker")


def value_token(word):
    """the identity a value is read with: the last token of a multi-word value"""
    parts = str(word).split()
    return parts[-1] if parts else str(word)


# ---------------------------------------------------------------- canonical frames
def _atom_filler(atom, by_id, legend, seen):
    if atom.kind == "word": return ["word", atom.text]
    if atom.kind == "unknown": return ["unknown"]
    if atom.kind == "asked": return ["asked"]
    w = by_id.get(atom.text)
    if w is None or atom.text in seen: return ["bad-ref"]
    seen = seen | {atom.text}
    if w.type == "entity":
        ent = ["entity", w.value("head")]
        a = w.get("attr"); ent.append(_line_filler(a, by_id, legend, seen) if a is not None else None)
        det = w.value("det"); ent.append(None if det is None or (legend is not None and det in legend.reader["definite_dets"]) else det)
        return ent
    if w.type == "not":
        s = w.get("scope")
        if s is not None and len(s.atoms) == 1 and s.atoms[0].kind == "word": return ["negword", s.atoms[0].text]
        return ["not", _line_filler(s, by_id, legend, seen) if s is not None else None]
    return ["wire", w.type]


def _line_filler(line, by_id, legend, seen=frozenset()):
    fs = [_atom_filler(a, by_id, legend, seen) for a in line.atoms]
    if len(fs) == 1: return fs[0]
    return ["set" if line.sep == "&" else "alt"] + sorted(fs, key=json.dumps)


def frame_of(wire, by_id, legend):
    """canonical frame of a top-level wire: {speakers, neg, event, roles, shape}; shape 'ok' when the v1 decoder can read the structure
    (says* -> [not] -> event), else the name of what it cannot read"""
    f = {"speakers": [], "neg": False, "event": None, "roles": {}, "shape": "ok", "lemma": None, "scope": None}
    w, seen = wire, set()
    while w is not None and w.type == "says" and w.id not in seen:
        seen.add(w.id); f["speakers"].append(w.value("speaker")); s = w.get("scope")
        w = by_id.get(s.atoms[0].text) if s is not None and s.atoms and s.atoms[0].kind == "ref" else None
    if w is not None and w.type == "not":
        s = w.get("scope")
        if s is not None and s.atoms and s.atoms[0].kind == "ref": f["neg"] = True; w = by_id.get(s.atoms[0].text)
    if w is None: f["shape"] = "broken"; return f
    events = legend.events if legend is not None else None
    if (w.type not in events) if events is not None else (w.type in ("entity", "unknown", "verdict") or w.type in OPERATORS):
        f["shape"] = w.type; f["event"] = w.type if w.type == "unknown" else None; return f
    f["event"] = w.type; f["lemma"] = w.value("_verb"); f["scope"] = w.value("_scope")
    for l in w.lines:
        if l.key.startswith("_"): continue
        f["roles"][l.key] = _line_filler(l, by_id, legend)
    f["event_wire"] = w.id
    return f


def reduce_filler(fl, drop=()):
    """a filler with the parts outside the legend's questions removed (legend reader.full_excludes: 'attr', 'det'); sets sorted"""
    if not fl or not drop: return fl
    if fl[0] in ("set", "alt"): return [fl[0]] + sorted((reduce_filler(x, drop) for x in fl[1:]), key=json.dumps)
    if fl[0] != "entity": return fl
    out = list(fl)
    if "attr" in drop: out[2] = None
    if "det" in drop: out[3] = "quantified" if out[3] else None
    return out


def canon(f, with_pred=True, drop=()):
    """the comparison key of a frame. drop: parts outside the legend's questions (legend reader.full_excludes, e.g. 'det': the quantifier
    world v2 asks WHICH phrase is wide, not which determiner word was used — the determiner identity is its own axis, outside FULL; the
    presence of a quantifier on a slot stays inside FULL: author Q2, 78d36e8)"""
    roles = f["roles"]
    def red(fl):
        if not fl: return fl
        if fl[0] in ("set", "alt"): return [fl[0]] + sorted((red(x) for x in fl[1:]), key=json.dumps)
        if fl[0] != "entity": return fl
        out = list(fl)
        if "attr" in drop: out[2] = None
        if "det" in drop: out[3] = "quantified" if out[3] else None
        return out
    if drop: roles = {r: red(fl) for r, fl in roles.items()}
    keep = {"speakers": f["speakers"], "neg": f["neg"], "event": f["event"], "roles": roles, "shape": f["shape"], "scope": f.get("scope")}
    if with_pred: keep["lemma"] = f.get("lemma")
    return json.dumps(keep, sort_keys=True)


def filler_heads(fl):
    """the heads of a filler, in order: a word, an entity's head, a set's members' heads (set roles: the member identities)"""
    if fl is None: return []
    if fl[0] in ("word", "negword", "entity"): return [fl[1]]
    if fl[0] in ("set", "alt"): return [h for x in fl[1:] for h in filler_heads(x)]
    return []


def filler_words(fl):
    """all words a filler mentions (for 'placed' bookkeeping)"""
    if fl is None: return []
    if fl[0] in ("word", "negword"): return [fl[1]]
    if fl[0] == "entity": return [x for x in [fl[1], fl[3]] if x] + filler_words(fl[2])
    if fl[0] in ("set", "alt"): return [w for x in fl[1:] for w in filler_words(x)]
    if fl[0] == "not": return filler_words(fl[1])
    return []


# ---------------------------------------------------------------- target states
class Compiler:
    def __init__(self, legend, stars):
        self.L, self.st = legend, stars; self.marker = legend.reader["occupancy_marker"]; self.pred = legend.reader["predicate_role"]
        self.missing_lemma = 0

    def lemma_for(self, ev_type, tokens, used):
        verbs = set(self.L.events.get(ev_type, []))
        for i, t in enumerate(tokens):
            if i in used: continue
            if self.L.lemma_of(t) in verbs: used.add(i); return self.L.lemma_of(t)
        return None

    def scope_terms(self, line, by_id):
        """quantifier world v2 (C2c): an entity filler whose det is a wide-scope marker adds R_wide (x) R_role (x) f_head, a plain quantifier
        adds R_q (x) R_role (x) f_head (category-local, unconditional terms; the event sums them / sqrt(n)); other determiners add nothing"""
        st = self.L.reader.get("scope_terms")
        if not st or len(line.atoms) != 1 or line.atoms[0].kind != "ref": return []
        en = by_id.get(line.atoms[0].text)
        if en is None or en.type != "entity" or en.value("head") is None: return []
        det = en.value("det"); s = self.st
        if det in st.get("wide", []): return [s.op("wide") * s.role(line.key) * s.id(en.value("head"))]
        if det in st.get("q", []): return [s.op("q") * s.role(line.key) * s.id(en.value("head"))]
        return []

    def value(self, line, by_id, tokens, used, depth=0):
        vs = [self.atom(a, by_id, tokens, used, depth, line.key) for a in line.atoms]; vs = [v for v in vs if v is not None]
        if not vs: return None
        s = sum(vs)
        return s / np.sqrt(len(vs)) if line.sep == "|" else s

    def atom(self, a, by_id, tokens, used, depth, key=None):
        if a.kind == "word": return self.st.val(a.text, key, self.L.reader)
        if a.kind == "unknown": return self.st.marker()
        if a.kind == "asked": return None
        w = by_id.get(a.text)
        if w is None or depth > 32: return None
        return self.wire(w, by_id, tokens, used, depth + 1)

    def wire(self, w, by_id, tokens, used=None, depth=0):
        used = set() if used is None else used; st = self.st; S = np.zeros(st.D, dtype=complex)
        V = lambda key: (lambda l: None if l is None else self.value(l, by_id, tokens, used, depth))(w.get(key))
        if w.type in self.L.events:
            if self.pred:
                lem = w.value("_verb") or self.lemma_for(w.type, tokens or [], used)
                if lem is not None: S = S + st.op("pred") * st.id(lem)
                else: self.missing_lemma += 1
            terms = []
            for l in w.lines:
                if l.key.startswith("_"): continue
                v = self.value(l, by_id, tokens, used, depth)
                if v is None: continue
                S = S + st.role(l.key) * (v + (st.marker() if self.marker else 0))
                terms += self.scope_terms(l, by_id)
            if terms: S = S + sum(terms) / np.sqrt(len(terms))
        elif w.type == "entity":
            h = w.value("head")
            if h is not None: S = S + st.val(h, "head", self.L.reader)
            a = V("attr")
            if a is not None: S = S + st.op("attr") * a
            det = w.value("det")
            if det is not None and det not in self.L.reader["definite_dets"]: S = S + st.op("det") * st.id(det)
            q = w.value("quantity")
            if q is not None and self.L.reader.get("value_terms"): S = S + st.op("quantity") * st.val(q, "quantity", self.L.reader)
        elif w.type in ("not", "must", "may"):
            v = V("scope")
            if v is not None: S = S + st.op(w.type) * v
        elif w.type == "if":
            c, q = V("condition"), V("consequence")
            if c is not None: S = S + st.op("if.condition") * c
            if q is not None: S = S + st.op("if.consequence") * q
        elif w.type == "says":
            sp = w.get("speaker")
            if sp is not None:
                v = self.value(sp, by_id, tokens, used, depth)
                if v is not None: S = S + st.op("says.speaker") * (v + (st.marker() if self.marker else 0))
            v = V("scope")
            if v is not None: S = S + st.op("says.scope") * v
        elif w.type == "and":
            it = w.get("items")
            if it is not None:
                for i, a in enumerate(it.atoms):
                    v = self.atom(a, by_id, tokens, used, depth)
                    if v is not None: S = S + st.op("and.next") ** i * v
        return S
