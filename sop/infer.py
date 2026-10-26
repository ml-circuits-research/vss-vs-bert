"""
infer.py — the reader at inference, numpy only (spec 4-5): a `model/` folder -> SOP wires for sentences.

  1. tokens -> kinds (the lexicon in model/catalog.json; a token with no kind cannot be placed: it gets no displacement and is listed in
     `_unplaced`, never silently dropped);
  2. the chart S = sum_k address_k (x) f_k, address_k = the bundle of the learned codes (path + soft chainL + bigram [+ beacon-relative]),
     exactly the computation of experiment/v8_gates.V8.states, re-implemented in numpy from model/routebook.npz;
  3. readout with the WORLD's role vectors (rule 1), regenerated from the seed: structure hypotheses (reporting depth d, event negation)
     scored by the mean best cleanup cosine of the always-present roles; per role: presence (occupancy marker or best cosine, as a ratio
     to the structure score, against a threshold set on CALIBRATION rows = training sentences, model/config.json), identity by cleanup
     over the catalog words of the role's kinds, margin = top1 - top2 cosine;
  4. `_residual` = ||S - a * rec|| / ||S|| with rec = the state of the decoded frame (world.Compiler) and a the best real scale; with
     config residual = "explained", rec = the reader's own chart of the explained tokens (residual_explained; 0 when all are explained);
     `_status`: ok | abstain (some margin < tau) | unknown-event | no-sense (more than `no_sense_share` of the tokens unplaced).
What the v1 decoder reads: says* -> [not] -> event, with fillers word | negated word | entity(head, attr [negated], det). Other wire
types (if, must, may, and, `|` alternatives, `~`) are written by annotators and compiled into training targets, but not decoded (stated).
"""
import json, os
import numpy as np
from . import soplang, world
from .legend import Legend, tokenize

UNKNOWN_KIND = "unknown"


def sigmoid(x): return 1 / (1 + np.exp(-x))
E = lambda ph: np.exp(1j * ph)


class ModelError(Exception): pass


class Routebook:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cats = [str(c) for c in z["cats"]]; self.cat_id = {c: i for i, c in enumerate(self.cats)}
        self.codes = [str(c) for c in z["codes"]]
        self.path, self.bg0, self.bg = z["path"].astype(float), z["bg0"].astype(float), z["bg"].astype(float)
        self.A, self.L, self.logits = z["A"].astype(float), z["L"].astype(float), z["logits"].astype(float)
        self.rel = z["rel"].astype(float) if "rel" in z.files else None
        self.pos = z["pos"].astype(float) if "pos" in z.files else None; self.pc = z["pc"].astype(float) if "pc" in z.files else None
        self.pos_max = int(z["pos_max"]) if "pos_max" in z.files else None
        self.rel_mask = z["rel_mask"].astype(float) if "rel_mask" in z.files else None
        self.taker_mask = z["taker_mask"].astype(float) if "taker_mask" in z.files else None
        self.vo = z["vo"].astype(float) if "vo" in z.files else None                  # the verb-order route (2, K, K, nmax, D)
        self.verb_mask = z["verb_mask"].astype(float) if "verb_mask" in z.files else None
        self.vo_nmax = int(z["vo_nmax"]) if "vo_nmax" in z.files else None
        self.cue_mask = z["cue_mask"].astype(float) if "cue_mask" in z.files else (np.zeros(len(self.cats)) if self.vo is not None else None)
        self.D = self.path.shape[1]

    def vorder_index(self, c):
        """(side, vcat, ordinal, valid) per token, as V8.vorder_index: side 1 = after the nearest verb to the left (counted from it), side 0 =
        before any verb (keyed by the nearest verb to the right, counted from the sentence start); verbs and sentences without a verb omitted"""
        T = len(c); side = np.zeros(T, int); vcat = np.zeros(T, int); ordinal = np.zeros(T, int); valid = np.zeros(T, bool)
        isv = self.verb_mask[c] > 0; verbs = [k for k in range(T) if isv[k]]
        cued = [k > 0 and self.cue_mask[c[k - 1]] > 0 for k in range(T)]          # variant b: introduced by a role cue -> left to the chain
        bare = lambda i: c[i] == c[k] and not cued[i]
        for k in range(T):
            if isv[k] or cued[k]: continue
            left = [j for j in verbs if j < k]; right = [j for j in verbs if j > k]
            if left:
                j = left[-1]; side[k] = 1; vcat[k] = c[j]; ordinal[k] = sum(1 for i in range(j + 1, k) if bare(i))
            elif right:
                j = right[0]; side[k] = 0; vcat[k] = c[j]; ordinal[k] = sum(1 for i in range(k) if bare(i))
            else: continue
            ordinal[k] = min(ordinal[k], self.vo_nmax - 1); valid[k] = True
        return side, vcat, ordinal, valid

    def addresses(self, c):
        """c: category indices (T,) -> addresses (T, D): the sum of the codes' terms before the identity, as V8.states"""
        c = np.asarray(c, dtype=int); T, D = len(c), self.D; terms = np.zeros((T, D), dtype=complex)
        if "path" in self.codes: terms += E(np.cumsum(self.path[c], 0))
        if "bigram" in self.codes: terms += E(np.concatenate([self.bg0[c[:1]], self.bg[c[:-1], c[1:]]], 0))
        if "pos" in self.codes and self.pos is not None:     # B7: the position code, omitted beyond the trained length (pos_max)
            posidx = np.minimum(np.arange(T), 39); pt = E(self.pos[posidx] + self.pc[c])
            if self.pos_max is not None: pt = pt * (posidx < self.pos_max)[:, None]
            terms += pt
        if "chainL" in self.codes or "chain" in self.codes:
            g = sigmoid(self.logits)[c]; u = np.log1p(-g + 1e-9); Sx = np.concatenate([[0.0], np.cumsum(u)])
            EA = E(self.A[c[:, None], c[None, :]])                                # EA[q, k] = A[c_q, c_k]
            addr = np.zeros((T, D), dtype=complex)
            for k in range(T):
                p_root = np.exp(Sx[k])
                if k == 0: addr[0] = p_root; continue
                p = g[:k] * np.exp(Sx[k] - Sx[1:k + 1])
                addr[k] = p_root + (p[:, None] * addr[:k] * EA[:k, k]).sum(0)
            if "chainL" in self.codes:
                EL = E(self.L[c[None, :], c[:, None]])                            # EL[k, q] = L[c_q, c_k]
                addrL = np.zeros_like(addr)
                for k in range(T):
                    acc = np.exp(Sx[T] - Sx[k + 1]) * addr[k]
                    if k < T - 1:
                        pL = g[k + 1:] * np.exp(Sx[k + 1:T] - Sx[k + 1])
                        acc = acc + (pL[:, None] * addr[k + 1:] * EL[k, k + 1:]).sum(0)
                    addrL[k] = acc
                addr = g[:, None] * addr + (1 - g[:, None]) * addrL
            terms += addr
        if self.vo is not None:
            side, vcat, ordinal, valid = self.vorder_index(c)
            terms += E(self.vo[side, vcat, c, ordinal]) * valid[:, None]
        if self.rel is not None and T > 1:
            sel = np.zeros(T); sel[1:] = self.rel_mask[c[:-1]] * self.taker_mask[c[1:]]
            relph = np.zeros((T, D)); relph[1:] = self.rel[c[:-1]]
            terms = terms + sel[:, None] * terms * E(relph)
        return terms


class Reader:
    def __init__(self, model_dir, legend_path=None, force=False, sentence_candidates=False):
        """sentence_candidates (default off: the existing behaviour): the candidates of a role are the catalog words of the role's kinds AND the
        sentence's own tokens whose kind is one of them (new labels and numbers read without retraining; mirrored by app/src/vss.js)"""
        self.dir = model_dir; self.sentence_candidates = bool(sentence_candidates); self._sent_tokens = ()
        need = ["config.json", "legend.json", "catalog.json", "routebook.npz"]
        for n in need:
            if not os.path.exists(os.path.join(model_dir, n)): raise ModelError(f"model incomplete: {n} missing in {model_dir}")
        with open(os.path.join(model_dir, "config.json")) as f: self.cfg = json.load(f)
        with open(os.path.join(model_dir, "catalog.json")) as f: self.cat = json.load(f)
        self.legend = Legend(os.path.join(model_dir, "legend.json"))
        if self.legend.sha != self.cfg["legend_sha256"] and not force:
            raise ModelError(f"model/legend.json hash {self.legend.sha} differs from config.json {self.cfg['legend_sha256']}")
        if legend_path is not None:
            other = Legend(legend_path)
            if other.sha != self.legend.sha and not force:
                raise ModelError(f"legend mismatch: {legend_path} is {other.sha}, the model was trained with {self.legend.sha} (--force to inspect)")
        self.rb = Routebook(os.path.join(model_dir, "routebook.npz")); self.D = self.rb.D
        self.st = world.Stars(self.cfg["seed"], self.D, self.cat.get("star_seed", world.GLOBAL_STAR_SEED), self.cat.get("identity_table"),
                              self.cfg.get("star_version", 1), self.cat.get("star_table"))
        self.comp = world.Compiler(self.legend, self.st)
        self.words = self.cat["words"]                      # word -> {"kind", "lemma"?}
        self.schema = self.cat["schema"]; self.th = self.cfg.get("thresholds", {})
        self.tau = self.cfg["thresholds"].get("tau", 0.05) if "thresholds" in self.cfg else 0.05
        self._cands = {}
        bg = os.path.join(model_dir, "background.npz")
        self.background = dict(np.load(bg)) if os.path.exists(bg) else None
        self.content_kinds = set(self.schema.get("content_kinds", []))
        rr = self.legend.reader; self.closed = rr.get("closed_values") or {}; self.table_roles = set(rr.get("table_roles") or []); self.set_roles = set(rr.get("set_roles") or [])
        self._table = None

    # ---------------- words
    def kind(self, tok):
        """the kind of a token: the IN-CONTEXT kind given for this sentence (formalize(..., token_kinds=...), e.g. a typer's; R88) when
        there is one, else the lexicon's (the catalog word's kind, or the legend's patterns)"""
        ck = getattr(self, "_ctx_kinds", None)
        if ck and tok in ck: return ck[tok]
        return self.lex_kind(tok)
    def lex_kind(self, tok):
        e = self.words.get(tok) or self.words.get(tok.lower())
        if e is not None: return e["kind"]
        return self.legend.kind(tok)
    def lemma(self, tok):
        e = self.words.get(tok) or self.words.get(tok.lower()); return (e or {}).get("lemma", tok)
    def ident(self, tok):
        """the word whose star stands for a token: the token, or its lexicon lemma when the legend says reader.identity = "lemma"
        (a given bilingual lexicon: the surface forms of several languages share the canonical value's identity)"""
        return self.lemma(tok) if self.legend.reader.get("identity") == "lemma" else tok

    def cands(self, kinds, filt=None):
        ck = getattr(self, "_ctx_kinds", None)
        if ck and filt is not None:       # R88: in-context kinds given for this sentence: the candidates are the sentence's words of the admitted kinds BY THEIR CONTEXT KIND
            ks = set(kinds); ws = sorted({w for w in filt if self.kind(w) in ks})
            if self.legend.reader.get("identity") == "lemma": ws = list(dict.fromkeys(self.ident(w) for w in ws))
            return (ws, np.array([self.st.id(w) for w in ws]) / np.sqrt(self.D) if ws else np.zeros((0, self.D), complex))
        extra = ()
        if self.sentence_candidates and self._sent_tokens:
            ks = set(kinds)
            extra = tuple(sorted({t for t in self._sent_tokens if t not in self.words and t.lower() not in self.words and self.kind(t) in ks}))
        key = (tuple(sorted(kinds)), None if filt is None else frozenset(filt), extra)
        if key not in self._cands:
            ws = sorted(w for w, e in self.words.items() if e["kind"] in kinds and (filt is None or w in filt)) + [t for t in extra if filt is None or t in filt]
            if self.legend.reader.get("identity") == "lemma": ws = list(dict.fromkeys(self.ident(w) for w in ws))     # one candidate per canonical value
            M = np.array([self.st.id(w) for w in ws]) / np.sqrt(self.D) if ws else np.zeros((0, self.D), complex)
            self._cands[key] = (ws, M)
        return self._cands[key]
    def role_cands(self, r, filt=None):
        """the candidates of a role: a legend closed value set (symbols), the sentence's TABLE (its columns, for table roles: given with the
        sentence, `_table` in examples, after a TAB in formalize's input), else the catalog words of the role's kinds"""
        if r in self.closed:
            key = ("__closed", r)
            if key not in self._cands:
                ws = list(self.closed[r]); M = np.array([self.st.val(w, r, self.legend.reader) for w in ws]); M = M / np.linalg.norm(M, axis=1, keepdims=True)
                self._cands[key] = (ws, M)
            return self._cands[key]
        if r in self.table_roles:
            cols = tuple(self._table or ()); key = ("__table", cols)
            if key not in self._cands:
                if len(self._cands) > 20000: self._cands = {k: v for k, v in self._cands.items() if k[0] != "__table"}
                ws = list(cols); vs = [self.st.val(c, "head", self.legend.reader) for c in ws]
                M = np.array([v / max(1e-12, np.linalg.norm(v)) for v in vs]) if ws else np.zeros((0, self.D), complex)
                self._cands[key] = (ws, M)
            return self._cands[key]
        rs = self.schema["roles"][r]
        return self.cands(rs["head_kinds"] if rs["main"] == "entity" else rs["kinds"], filt)

    def verb_cands(self):
        if "__verbs" not in self._cands:
            ws = self.legend.all_verbs(); self._cands["__verbs"] = (ws, np.array([self.st.id(w) for w in ws]) / np.sqrt(self.D))
        return self._cands["__verbs"]

    # ---------------- chart
    def encode(self, tokens):
        kept, cidx, unknown = [], [], []
        for i, t in enumerate(tokens):
            k = self.lex_kind(t)          # the chart uses the kinds the routes were trained with (the lexicon's); context kinds act in the decoder (R88)
            if k is None or k == UNKNOWN_KIND or k not in self.rb.cat_id:
                unknown.append(i)
                if UNKNOWN_KIND not in self.rb.cat_id: continue          # spec 6.3: a word with no kind gets the kind `unknown` (placed, never a filler)
                k = UNKNOWN_KIND
            kept.append(i); cidx.append(self.rb.cat_id[k])
        return kept, cidx, unknown
    def chart(self, tokens):
        kept, cidx, unknown = self.encode(tokens)
        if not kept: return np.zeros(self.D, complex), kept, unknown, np.zeros((0, self.D), complex)
        addr = self.rb.addresses(cidx); F = np.array([self.st.id(self.ident(tokens[i])) for i in kept])
        return (addr * F).sum(0), kept, unknown, addr

    # ---------------- readout
    @staticmethod
    def best(x, cand):
        ws, M = cand
        if not ws: return None, 0.0, 0.0
        sc = np.real(M.conj() @ x)
        if len(ws) == 1: return ws[0], float(sc[0]), float(sc[0])
        j = np.argsort(-sc)[:2]; return ws[j[0]], float(sc[j[0]] - sc[j[1]]), float(sc[j[0]])

    def _filler_top(self, Sn, key, r, filt):
        if r in self.closed or r in self.table_roles: return self.best(Sn * key, self.role_cands(r, filt))
        rs = self.schema["roles"][r]
        if rs["main"] == "entity": return self.best(Sn * key, self.cands(rs["head_kinds"], filt))
        return self.best(Sn * key, self.cands(rs["kinds"], filt))

    def structure_score(self, Sn, K, filt=None):
        """the anchor every presence ratio is divided by: the mean top of the core roles (present in >= 95% of the training rows) and of
        the predicate; a world with neither (R79) falls back to the mean over every role, which is weak when most roles are absent"""
        core = self.schema["core_roles"] or ([] if self.legend.reader["predicate_role"] else self.schema["role_order"])
        tops = [self._filler_top(Sn, K * np.conj(self.st.role(r)), r, filt)[2] for r in core]
        if self.legend.reader["predicate_role"]: tops.append(self.best(Sn * K * np.conj(self.st.op("pred")), self.verb_cands())[2])
        return float(np.mean(tops)) if tops else 0.0

    def read_set_role(self, Sn, key, r, core, presence):
        """a SET role (legend reader.set_roles, e.g. WikiSQL's conditions): every candidate of the sentence's table is a member when its ratio
        (cosine after unbinding the role / the structure score) reaches the member threshold set on calibration rows (thresholds.set_member,
        by the calibration rule); the margin = the gap between the weakest member and the strongest non-member (cosine); the presence margin =
        the smallest |ratio - threshold| over the candidates"""
        ws, M = self.role_cands(r); out = {"margins": {}}
        if not ws: out.update(presence=0.0, present=False); return out
        sc = np.real(M.conj() @ (Sn * key)); ratio = sc / core if core > 1e-9 else sc * 0
        thr = self.th.get("set_member", {}).get(r)
        mem = [i for i in range(len(ws)) if (ratio[i] >= thr if thr is not None else i == int(np.argmax(sc)))]
        if presence == "all" and not mem: mem = [int(np.argmax(sc))]
        out["presence"] = float(ratio.max()); out["member_ratios"] = {w: float(x) for w, x in zip(ws, ratio)}; out["present"] = bool(mem)
        if thr is not None and presence != "all": out["presence_margin"] = float(np.min(np.abs(ratio - thr)))
        if not mem: return out
        non = [i for i in range(len(ws)) if i not in mem]
        out["margins"][r] = float(sc[mem].min() - (sc[non].max() if non else 0.0))
        ents = sorted((["entity", ws[i], None, None] for i in mem), key=json.dumps)
        out["filler"] = ents[0] if len(ents) == 1 else ["set"] + ents
        return out

    def read_role(self, Sn, K, r, core, filt=None, presence="decided"):
        key = K * np.conj(self.st.role(r))
        if r in self.set_roles: return self.read_set_role(Sn, key, r, core, presence)
        if r in self.closed or r in self.table_roles:
            w, mg, top = self.best(Sn * key, self.role_cands(r, filt)); out = {"margins": {}, "presence": top / core if core > 1e-9 else 0.0, "present": w is not None}
            if w is not None: out["filler"] = ["word", w]; out["margins"][r] = mg
            return out
        rs = self.schema["roles"][r]; out = {"margins": {}}
        if self.legend.reader["occupancy_marker"]:
            mk = self.st.marker() / np.sqrt(self.D); pres = float(np.real(np.sum(Sn * key * np.conj(mk))))
        else: pres = self._filler_top(Sn, key, r, filt)[2]
        ratio = pres / core if core > 1e-9 else 0.0; out["presence"] = ratio
        thr = self.th.get("presence", {}).get(r)
        present = (ratio >= thr) if thr is not None else rs["rate"] >= 0.5
        if presence == "all": present = True
        out["present"] = present
        if thr is not None and presence != "all": out["presence_margin"] = abs(ratio - thr)    # the PRESENCE margin (judgment 752853c §2)
        if self.cfg.get("presence_by_kind") and presence != "all" and filt is not None:
            # R88 (C): presence by kinds where a kind names one question. The kinds given (in context when token_kinds were passed) decide:
            # a word of a kind that only this question admits -> present; no word of any admitted kind -> absent; words only of shared
            # kinds -> the threshold as above. The decision is recorded in out["presence_rule"].
            admitted = set(rs.get("head_kinds") or rs.get("kinds") or [])
            others = {k for q, qs in self.schema["roles"].items() if q != r for k in (qs.get("head_kinds") or qs.get("kinds") or [])}
            sk = {self.kind(w) for w in filt}
            if admitted & sk & (admitted - others): out["present"], out["presence_rule"] = True, "unique kind present"; out.pop("presence_margin", None)
            elif not (admitted & sk): out["present"], out["presence_rule"] = False, "no word of an admitted kind"; out.pop("presence_margin", None)
            else: out["presence_rule"] = "threshold (shared kinds)"
            present = out["present"]
        if not present: return out
        if rs["main"] == "entity":
            h, mg, top = self.best(Sn * key, self.cands(rs["head_kinds"], filt)); out["margins"][r] = mg
            if h is None: out["filler"] = None; out["no_candidate"] = True; out["margins"].pop(r, None); return out     # same rule for an entity head (D5)
            attr = None
            if rs.get("attr_kinds"):
                ka = key * np.conj(self.st.op("attr")); ca = self.cands(rs["attr_kinds"], filt)
                a1 = self.best(Sn * ka, ca); neg = False
                if rs.get("attr_neg"):
                    a2 = self.best(Sn * ka * np.conj(self.st.op("not")), ca)
                    if a2[2] > a1[2]: a1, neg = a2, True
                aratio = a1[2] / top if top > 1e-9 else 0.0; athr = self.th.get("attr_presence", {}).get(r); out["attr_ratio"] = aratio
                # vss-demo Q1 / D1: an attribute (or determiner) is given only when a candidate exists; with sentence candidates an
                # adjective outside the catalog leaves no candidate, and the word is then reported as unplaced (the query-wire case)
                if a1[0] is not None and ((aratio >= athr) if athr is not None else rs.get("attr_rate", 0) >= 0.5):
                    attr = ["negword" if neg else "word", a1[0]]; out["margins"][r + ".attr"] = a1[1]
            det = None
            if rs.get("det_kinds"):
                d1 = self.best(Sn * key * np.conj(self.st.op("det")), self.cands(rs["det_kinds"], filt))
                dratio = d1[2] / top if top > 1e-9 else 0.0; dthr = self.th.get("det_presence", {}).get(r); out["det_ratio"] = dratio; out["det_best"] = (d1[0], d1[1])
                if d1[0] is not None and ((dratio >= dthr) if dthr is not None else rs.get("det_rate", 0) >= 0.5):
                    det = d1[0]; out["margins"][r + ".det"] = d1[1]
            out["filler"] = ["entity", h, attr, det]
        else:
            c = self.cands(rs["kinds"], filt); w1 = self.best(Sn * key, c); neg = False
            if rs.get("negword"):
                w2 = self.best(Sn * key * np.conj(self.st.op("not")), c)
                if w2[2] > w1[2]: w1, neg = w2, True
            if w1[0] is None: out["filler"] = None; out["no_candidate"] = True      # present by the threshold, no admissible word in the sentence: not given (vss-demo Q4 / D5)
            else: out["filler"] = ["negword" if neg else "word", w1[0]]; out["margins"][r] = w1[1]
        return out

    def depth_from_gates(self, tokens):
        """the PRIMARY depth estimate (judgment e9274a9 Q6): the reader's own count of open level gates passed on the token sequence,
        deterministic from the learned gate values. A level is opened by a token of a level-opener kind (legend reader.level_openers, e.g.
        the reporting verbs 'says' / 'claims'); the gate that opens it is the opener's own or, when that one is closed, the gate of the token
        right after it (the complementizer 'that', open in every P0 run). Returns None when the legend names no level openers."""
        ops = set(self.legend.reader.get("level_openers") or [])
        if not ops: return None
        g = sigmoid(self.rb.logits); ks = [self.kind(t) for t in tokens]
        gate = lambda k: float(g[self.rb.cat_id[k]]) if k in self.rb.cat_id else 0.0
        return sum(1 for i, k in enumerate(ks) if k in ops and (gate(k) >= 0.5 or (i + 1 < len(ks) and gate(ks[i + 1]) >= 0.5)))

    def depth_from_kinds(self, tokens):
        """the depth as the number of tokens of a level-opener KIND (legend reader.level_openers), deterministic from the kinds on the token
        sequence, independent of the learned gate values (round 3, run E). None when the legend names no level openers."""
        ops = set(self.legend.reader.get("level_openers") or [])
        return sum(1 for t in tokens if self.kind(t) in ops) if ops else None

    def read(self, S, tokens, filt=None, force=None, force_depth=None, estimator="gates", presence="decided", table=None):
        """S -> decoded frame (same layout as world.frame_of) with margins and presence scores.
        force = (depth, negation) GIVEN; force_depth = depth GIVEN, negation estimated; otherwise the depth is ESTIMATED — by the open level
        gates passed (estimator 'gates', the primary one, when the legend names level openers) or by the fit (estimator 'fit': the depth whose
        role readout is best, frame.py style; kept as a check) or by the count of level-opener kinds (estimator 'kinds', gate-independent).
        The negation is always chosen by the fit at the chosen depth. The depth estimates are kept on the frame (depth_gates, depth_kinds,
        depth_fit) and written on the top-level wire. presence='all' reads every role of the schema without the presence decision (as the
        repository's v16 readout, which always reads C); the default 'decided' applies the calibrated presence thresholds."""
        nS = np.linalg.norm(S); self._sent_tokens = tuple(tokens) if tokens is not None else ()
        if table is not None: self._table = list(table)
        f = {"speakers": [], "neg": False, "event": None, "roles": {}, "shape": "ok", "lemma": None, "margins": {}, "presence": {}}
        if nS < 1e-9: f["shape"] = "unknown"; return f
        Sn = S / nS; sch = self.schema; rep = np.conj(self.st.op("says.scope")); nott = np.conj(self.st.op("not"))
        negs = [False, True] if sch["neg_event"] else [False]
        dmax = sch["max_depth"] + 1 if "says" in self.legend.operators else 0      # no reporting operator in the legend: depth 0 only
        hyps = [(d, n) for d in range(dmax + 1) for n in negs]
        scores = {h: self.structure_score(Sn, rep ** h[0] * (nott if h[1] else 1), filt) for h in hyps}
        order = sorted(hyps, key=lambda h: -scores[h])
        f["depth_fit"] = order[0][0]; f["depth_gates"] = dg = self.depth_from_gates(tokens) if tokens is not None else None
        f["depth_kinds"] = dk = self.depth_from_kinds(tokens) if tokens is not None else None
        if force is not None: h = force
        else:
            fd = force_depth if force_depth is not None else (dg if estimator == "gates" else dk if estimator == "kinds" else None)
            if fd is not None:
                for n in negs:
                    if (fd, n) not in scores: scores[(fd, n)] = self.structure_score(Sn, rep ** fd * (nott if n else 1), filt)
                h = max([(fd, n) for n in negs], key=lambda x: scores[x])
            else: h = order[0]
        if h not in scores: scores[h] = self.structure_score(Sn, rep ** h[0] * (nott if h[1] else 1), filt)
        f["margins"]["structure"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
        d, neg = h; K = rep ** d * (nott if neg else 1); core = scores[h]; f["neg"] = neg; f["depth"] = d; f["core"] = core
        for k in range(d):
            sp, mg, _ = self.best(Sn * rep ** k * np.conj(self.st.op("says.speaker")), self.cands(sch["speaker_kinds"], filt))
            f["speakers"].append(sp); f["margins"][f"speaker.{k}"] = mg
        if self.legend.reader["predicate_role"]:
            vc = self.verb_cands()
            if self.cfg.get("candidates") == "sentence":          # D14 item 4: the predicate's candidates are the sentence's verbs (the contract: the answer is in the sentence)
                inv = {w: i for i, w in enumerate(vc[0])}; js = sorted({inv[self.lemma(t)] for t in tokens if self.lemma(t) in inv})
                if js: vc = ([vc[0][j] for j in js], vc[1][js])
            lem, mg, _ = self.best(Sn * K * np.conj(self.st.op("pred")), vc)
            f["lemma"] = lem; f["event"] = self.legend.event_of_lemma(lem); f["margins"]["event"] = mg
        else:
            f["event"] = sch.get("single_event") or next((self.legend.event_of_lemma(self.lemma(t)) for t in tokens if self.legend.event_of_lemma(self.lemma(t))), None)
        taken = self.legend.event_roles.get(f.get("event")) if f.get("event") else None     # Q12 / D19: the roles the decided event takes
        for r in sch["role_order"]:
            if taken is not None and r not in taken: f["presence"][r] = 0.0; continue            # not a role of this event: not read, no candidate
            o = self.read_role(Sn, K, r, core, filt, presence); f["presence"][r] = o["presence"]
            for part in ("attr", "det"):
                if part + "_ratio" in o: f["presence"][r + "." + part] = o[part + "_ratio"]
            if "member_ratios" in o: f["presence"][r + ".members"] = o["member_ratios"]
            if "presence_rule" in o: f.setdefault("presence_rules", {})[r] = o["presence_rule"]
            if "presence_margin" in o:
                given = o.get("present") and o.get("filler") is not None
                if given or not self.cfg.get("presence_query"): f["margins"][r + ".presence"] = o["presence_margin"]
                elif o["presence_margin"] < self.tau or o.get("present"): f.setdefault("presence_queries", {})[r] = o["presence_margin"]   # D14 item 1 / Q38: a role not given (absent near its threshold, or present with no admissible word) is a presence query, not an abstention of the frame
            if o.get("present") and o.get("filler") is not None:
                f["roles"][r] = o["filler"]; f["margins"].update(o["margins"])
                if "presence_margin" in o:
                    f.setdefault("identity_margins", {})[r] = o["margins"].get(r)
                    f["margins"][r] = min(o["margins"].get(r, 1.0), o["presence_margin"])   # the role's margin = min(identity, presence)
                if "det_best" in o: f.setdefault("det_best", {})[r] = o["det_best"]
        if filt is not None and self.cfg.get("candidates") == "sentence": self.one_to_one(Sn, K, core, f, filt, presence)
        if self.cfg.get("presence_query") and f.get("event") in self.legend.event_required:        # Q12 / D19: a required role not given is asked for
            for r in self.legend.event_required[f["event"]]:
                if f["roles"].get(r) is None: f.setdefault("presence_queries", {})[r] = f["margins"].pop(r + ".presence", 0.0)
        st = self.legend.reader.get("scope_terms")
        if st: self.read_scope(Sn, K, core, f)
        if f["event"] is None: f["shape"] = "unknown"
        return f

    @staticmethod
    def head_of(fl): return fl[1] if fl and fl[0] in ("entity", "word", "negword") else None

    def one_to_one(self, Sn, K, core, f, filt, presence):
        """the one-to-one rule of the sentence-candidates decoder (vss-vs-bert R10 / R46 / R64): a word fills at most as many questions as it
        has occurrences in the sentence, over EVERY question of the frame, the speakers of its levels included (a speaker is not also the
        giver unless the sentence names the same word twice). When more questions read the same head than it has occurrences, the
        questions with the larger margins keep it and the others are read again without that word; a question left with no candidate is
        'not given' (a speaker left without one makes the frame incomplete). The words a question lost accumulate across rounds; a question
        still colliding after the last round is left unanswered."""
        toks = list(self._sent_tokens); occ = lambda h: max(1, sum(1 for t in toks if t == h or t.lower() == str(h).lower()))
        rep = np.conj(self.st.op("says.scope")); spk = np.conj(self.st.op("says.speaker"))
        excl = {}                                    # per question, the words it lost: accumulated across rounds (vss-vs-bert Q31)
        def clear(kind, r):
            if kind == "role": f["roles"].pop(r, None); f["margins"].pop(r, None); f.setdefault("one_to_one", {})[r] = "not given"
            else: f["speakers"][r] = None; f["margins"][f"speaker.{r}"] = 0.0; f.setdefault("one_to_one", {})[f"speaker.{r}"] = "not given"
        for rnd in range(5):
            heads = {}
            for r, fl in f["roles"].items():
                h = self.head_of(fl)
                if h is not None: heads.setdefault(h, []).append(("role", r, f["margins"].get(r, 0.0)))
            for k, sp in enumerate(f["speakers"]):
                if sp is not None: heads.setdefault(sp, []).append(("speaker", k, f["margins"].get(f"speaker.{k}", 0.0)))
            dup = {h: sl for h, sl in heads.items() if len(sl) > occ(h)}
            if not dup: return
            last = rnd == 4
            for h, sl in dup.items():
                sl = sorted(sl, key=lambda x: -x[2])
                for kind, r, _m in sl[occ(h):]:
                    ex = excl.setdefault((kind, r), set()); ex.add(str(h).lower())
                    if last: clear(kind, r); continue            # still colliding after the rounds: the question stays unanswered
                    sub = {w for w in filt if w.lower() not in ex}
                    if kind == "role":
                        o = self.read_role(Sn, K, r, core, sub, presence)
                        if o.get("present") and o.get("filler") is not None and self.head_of(o["filler"]) is not None:
                            f["roles"][r] = o["filler"]; f["margins"].update(o["margins"]); f.setdefault("one_to_one", {})[r] = "re-read"
                        else: clear(kind, r)
                    else:
                        w, mg, _ = self.best(Sn * rep ** r * spk, self.cands(self.schema["speaker_kinds"], sub))
                        f["speakers"][r] = w; f["margins"][f"speaker.{r}"] = mg if w is not None else 0.0
                        f.setdefault("one_to_one", {})[f"speaker.{r}"] = "re-read" if w is not None else "not given"

    def read_scope(self, Sn, K, core, f):
        """the quantifier world v2 (C2c, RESULTS 9c-sexies): category-local terms R_wide (x) R_role (x) f_head (wide-scope marker) and
        R_q (x) R_role (x) f_head (plain quantifier) in the event. Readout rule (the legend's `scope_rule`): an R_wide term present -> that slot
        is wide; one R_q term alone -> the only quantifier; two R_q and no R_wide -> both. `_scope` is written only with two or more quantified
        slots (spec 3.3: absent with zero or one quantifier). Term presence = ratio to the structure score >= the threshold set on calibration
        rows (model/config.json thresholds.scope_term)."""
        thr = self.th.get("scope_term"); sw, sq = {}, {}
        for r, fl in f["roles"].items():
            if fl[0] != "entity" or fl[1] is None: continue
            fh = np.conj(self.st.id(fl[1])) / np.sqrt(self.D); base = Sn * K * np.conj(self.st.role(r)) * fh
            sw[r] = float(np.real(np.sum(base * np.conj(self.st.op("wide"))))) / max(core, 1e-9)
            sq[r] = float(np.real(np.sum(base * np.conj(self.st.op("q"))))) / max(core, 1e-9)
            f["presence"][r + ".wide"], f["presence"][r + ".q"] = sw[r], sq[r]
        if thr is None: return
        W = [r for r in sw if sw[r] >= thr]; Q = [r for r in sq if sq[r] >= thr and r not in W]
        # the PRESENCE of a quantifier on a slot is read from its category-local term (as v14's C2c decoder); the determiner WORD from the
        # determiner channel (argmax), reported as its own axis
        for r in sw:
            fl = f["roles"][r]; best = f.get("det_best", {}).get(r)
            if r in W or r in Q:
                if fl[3] is None and best: fl[3] = best[0]; f["margins"][r + ".det"] = best[1]
            elif fl[3] is not None: fl[3] = None; f["margins"].pop(r + ".det", None)
        if len(set(W) | set(Q)) < 2: f["scope"] = None; dec = [abs(v - thr) for v in list(sw.values()) + list(sq.values())]
        elif W: f["scope"] = max(W, key=lambda r: sw[r]); dec = [sw[f["scope"]] - thr] + [abs(sw[r] - thr) for r in sw if r != f["scope"]]
        else: f["scope"] = "both"; dec = [sq[r] - thr for r in Q]
        f["margins"]["scope"] = float(min(dec) * core) if dec else 1.0

    # ---------------- frame -> wires
    def frame_wires(self, f, ids, text=None, sentence=None, extra=None):
        """the decoded frame as SOP wires (top-level first). ids: an iterator of fresh wire ids"""
        wires = []; mg = f.get("margins", {})
        def new(t): w = soplang.Wire(next(ids), t); wires.append(w); return w
        if f["shape"] != "ok":
            w = new("unknown")
            if text is not None: w.set("_text", text)
            if sentence is not None: w.set("_sentence", str(sentence))
            return wires
        top = None; chain = []
        for k, sp in enumerate(f["speakers"]):
            w = new("says"); chain.append(w); top = top or w; w.set("speaker", sp)
        notw = None
        if f["neg"]:
            notw = new("not"); chain.append(notw); top = top or notw
            if "structure" in mg: notw.set("_margin.structure", fmt(mg["structure"]))
        ev = new(f["event"]); top = top or ev
        for a, b in zip(chain, chain[1:] + [ev]): a.set("scope", ("ref", b.id))
        for k, w in enumerate(chain):
            if w.type == "says": w.set("_margin.speaker", fmt(mg.get(f"speaker.{k}", 0)))
            w.set("_depth", str(k))
        later = []
        for r in self.legend.roles:
            if r not in f["roles"]: continue
            fl = f["roles"][r]
            if fl[0] == "word": ev.set(r, fl[1])
            elif fl[0] == "negword":
                nw = soplang.Wire(next(ids), "not"); nw.set("scope", fl[1]); later.append(nw); ev.set(r, ("ref", nw.id))
                if r in mg: nw.set("_margin.scope", fmt(mg[r]))
            elif fl[0] == "set":            # a SET value: one entity wire per member, the role line their & (spec 3.2)
                refs = []
                for it in fl[1:]:
                    en = soplang.Wire(next(ids), "entity"); en.set("head", it[1]); later.append(en); refs.append(("ref", en.id))
                ev.set(r, ("&", refs))
            elif fl[0] == "entity":
                en = soplang.Wire(next(ids), "entity"); later.append(en); ev.set(r, ("ref", en.id)); en.set("head", fl[1])
                if fl[2] is not None:
                    if fl[2][0] == "negword":
                        nw = soplang.Wire(next(ids), "not"); nw.set("scope", fl[2][1]); later.append(nw); en.set("attr", ("ref", nw.id))
                        if f"{r}.attr" in mg: nw.set("_margin.scope", fmt(mg[f"{r}.attr"]))
                    else: en.set("attr", fl[2][1])
                if fl[3] is not None: en.set("det", fl[3])
                if r in mg: en.set("_margin.head", fmt(mg[r]))
                for part in ("attr", "det"):
                    if f"{r}.{part}" in mg: en.set(f"_margin.{part}", fmt(mg[f"{r}.{part}"]))
        if f.get("scope"): ev.set("_scope", f["scope"])
        if f.get("lemma"): ev.set("_verb", f["lemma"])
        ev.set("_mood", self.mood(text))
        for r in self.legend.roles:
            if r in mg: ev.set(f"_margin.{r}", fmt(mg[r]))
            if r + ".presence" in mg: ev.set(f"_presence_margin.{r}", fmt(mg[r + ".presence"]))
            if r in f.get("presence_queries", {}): ev.set(f"_presence_query.{r}", fmt(f["presence_queries"][r]))
            if r in f.get("offers", {}): ev.set(f"_offer.{r}", f["offers"][r])          # the unplaced word offered to this empty role (D27)   # absent, near threshold: "is there a <r>?" (query wire kind undecided, presence)
        if "event" in mg: ev.set("_margin.event", fmt(mg["event"]))
        if "scope" in mg: ev.set("_margin.scope", fmt(mg["scope"]))
        evm = [mg[k] for k in mg if k in self.legend.roles or k == "event" or (k.endswith(".presence") and k[:-9] in self.legend.roles)]
        if evm: ev.set("_margin", fmt(min(evm)))
        wires += later
        # top-level lines first: _text, _sentence
        if text is not None or sentence is not None:
            head = []
            if text is not None: head.append(soplang.Line("_text", text.replace("\n", " ")))
            if sentence is not None: head.append(soplang.Line("_sentence", str(sentence)))
            top.lines = head + top.lines
        return wires

    def no_fit_residual(self):
        """dispatcher only (section 9): a residual at or above this value means the reader explains almost nothing of the chart. A STATED
        constant (config thresholds.no_fit_residual, default 0.9), not fitted: fitting it needs out-of-world sentences, which a world folder
        does not have; unseen depths of a world reach 0.6-0.8 (P0), random text 0.99."""
        return float(self.cfg.get("thresholds", {}).get("no_fit_residual", 0.9))

    def wire_lines(self, wires, status):
        """spec 5: every wire written by formalize has _margin, _status, _empty, _depth. _depth = distance from the top-level wire;
        _empty = the keys of the wire's type not filled ('-' when none); _margin = the minimum of the wire's own _margin.* lines (the
        top-level wire: over the whole sentence)."""
        by = {w.id: w for w in wires}; depth = {wires[0].id: 0}; q = [wires[0].id]
        while q:
            x = q.pop(0)
            for r in by[x].refs():
                if r in by and r not in depth: depth[r] = depth[x] + 1; q.append(r)
        for w in wires:
            keys = self.legend.keys_for(w.type) if w.type != "unknown" else set()
            order = [k for k in (self.legend.roles if w.type in self.legend.events else sorted(keys)) if k in keys]
            empty = [k for k in order if w.get(k) is None]
            if w.type in self.legend.events: empty = [k for k in self.schema["role_order"] if w.get(k) is None] + [k for k in empty if k not in self.schema["role_order"]]
            w.set("_empty", ",".join(empty) if empty else "-")
            if w.get("_margin") is None:
                ms = [float(l.text) for l in w.lines if l.key.startswith("_margin.")]
                w.set("_margin", fmt(min(ms)) if ms else "none")
            if w.get("_status") is None: w.set("_status", status)
            w.set("_depth", str(depth.get(w.id, 0)))

    def mood(self, text):
        """_mood comes from the normalizer, not the reader (stated): the legend's only mood, or '?' -> question, else the first mood"""
        ms = self.legend.moods
        if len(ms) == 1: return ms[0]
        if text and text.strip().endswith("?") and "question" in ms: return "question"
        return ms[0]

    def residual(self, S, wires, tokens):
        by = {w.id: w for w in wires}; rec = self.comp.wire(wires[0], by, tokens)
        nS, nr = np.linalg.norm(S), np.linalg.norm(rec)
        if nS < 1e-9 or nr < 1e-9: return 1.0
        Sn, rn = S / nS, rec / nr; a = max(0.0, float(np.real(np.vdot(rn, Sn))))
        return float(np.linalg.norm(Sn - a * rn))

    def frame_incomplete(self, f):
        """a frame that does not answer every always-present (core) question of the legend, or a speaker of its depth, explains nothing as a
        frame: the explained residual is then 1.0 (vss-vs-bert R42 / Q30; an optional role left 'not given' does not count)"""
        req = self.legend.event_required.get(f.get("event"), self.schema.get("core_roles", [])) if f.get("event") else self.schema.get("core_roles", [])
        if any(f["roles"].get(r) is None or self.head_of(f["roles"].get(r)) is None for r in req): return True
        return any(sp is None for sp in f.get("speakers", []))

    def residual_explained(self, S, addr, kept, tokens, placed):
        """the explained-tokens residual (addendum q-formalisation 9; vss-vs-bert R41): rec' = the reader's OWN chart of the explained tokens
        (fillers of decoded roles, speakers, the event lemma, and every token whose kind is not a content kind: cue words, openers,
        punctuation); residual' = ||S_n - a rec'_n||. When every token is explained rec' = S and the residual is 0 by construction; an
        unplaced content word, or a word of the kind `unknown`, contributes its share of the state. The compiled residual (`residual`)
        compares S with the compiler's state of the frame, which lacks the routes' deposits and therefore grows with the sentence length."""
        if not kept: return 1.0
        ex = [j for j, i in enumerate(kept) if self.kind(tokens[i]) not in self.content_kinds and self.kind(tokens[i]) != UNKNOWN_KIND
              or tokens[i] in placed or self.lemma(tokens[i]) in placed]
        if not ex: return 1.0
        F = np.array([self.st.id(self.ident(tokens[kept[j]])) for j in ex]); rec = (addr[ex] * F).sum(0)
        nS, nr = np.linalg.norm(S), np.linalg.norm(rec)
        if nS < 1e-9 or nr < 1e-9: return 1.0
        Sn, rn = S / nS, rec / nr; a = max(0.0, float(np.real(np.vdot(rn, Sn))))
        r = float(np.linalg.norm(Sn - a * rn)); return 0.0 if r < 1e-4 else r     # numerical tolerance, declared (vss-vs-bert Q32)

    def undecided_adjacent(self, f, tokens, unplaced):
        """the adjacency rule (addendum q-formalisation 7; vss-vs-bert R54): an identity left in no role, next to the filler of a role whose
        kinds admit it, turns that role into an 'undecided' query with both as candidates ("p3 p5 gives o1 to p2": who gives, p3 or p5?).
        Returns {role: (filler word, the other word)}. Config `undecided_adjacent: true`; off by default (the paper's decoder)."""
        out = {}
        pos = {t: i for i, t in enumerate(tokens)}
        for r, fl in f["roles"].items():
            ws = world.filler_words(fl)
            if not ws: continue
            rs = self.schema["roles"].get(r) or {}; kinds = set(rs.get("head_kinds") or rs.get("kinds") or [])
            head = ws[0]; i = pos.get(head)
            if i is None: continue
            run = self.legend.reader.get("spans") == "kind_run"
            for j in (i - 1, i + 1):
                if j in unplaced and self.kind(tokens[j]) in kinds and tokens[j] != head:
                    if run and self.kind(tokens[j]) == self.kind(head): continue          # the same run (a multi-word span), not a competitor (Q38)
                    out[r] = (head, tokens[j]); break
        return out

    def placed_words(self, f):
        ws = set(f["speakers"]) | ({f["lemma"]} if f.get("lemma") else set())
        for fl in f["roles"].values(): ws |= set(world.filler_words(fl))
        if self.legend.reader.get("value_identity") == "bundle_alnum": ws |= {t for w in list(ws) for t in world.alnum(w)}
        return ws

    def formalize(self, text, sentence, ids, filt=None, force=None, force_depth=None, estimator=None, presence=None, table=None, token_kinds=None):
        """one sentence -> (wires, frame, info). force / force_depth: see read() (evaluation only; the CLI estimates both).
        token_kinds (R88): the IN-CONTEXT kind of every token of this sentence (a list aligned with the tokens, or {word: kind}), e.g. a
        typer's; used by the decoder (candidates, admission, residual, runs, adjacency), not by the chart (the routes' kinds)"""
        tokens = tokenize(text)
        if token_kinds is not None:
            self._ctx_kinds = dict(zip(tokens, token_kinds)) if isinstance(token_kinds, (list, tuple)) else dict(token_kinds)
        else: self._ctx_kinds = None
        S, kept, unknown, addr = self.chart(tokens)
        if filt is None and self.cfg.get("candidates") == "sentence":     # the sentence-candidates decoder (R46): a role's filler is a word of the sentence
            filt = set(tokens) | {t.lower() for t in tokens}
        # a model without the key predates round 3: its primary estimator was the open-gate count
        estimator = estimator or self.cfg.get("depth_estimator", "gates"); presence = presence or self.cfg.get("presence_readout", "decided")
        f = self.read(S, tokens, filt, force, force_depth, estimator, presence, table) if kept else {"shape": "unknown", "roles": {}, "speakers": [], "margins": {}, "neg": False, "lemma": None}
        placed = self.placed_words(f) if f["shape"] == "ok" else set()
        if f["shape"] == "ok" and self.legend.reader.get("spans") == "kind_run":      # Q38 / R87: multi-word fillers: a filler is the run of its kind around the head
            kinds_seq = [self.kind(t) for t in tokens]
            for fl in f["roles"].values():
                for w in world.filler_words(fl):
                    for i, t in enumerate(tokens):
                        if t == w and str(kinds_seq[i]).startswith("S:"):
                            a = b = i
                            while a > 0 and kinds_seq[a - 1] == kinds_seq[i]: a -= 1
                            while b + 1 < len(tokens) and kinds_seq[b + 1] == kinds_seq[i]: b += 1
                            placed |= set(tokens[a:b + 1])
        if placed and f.get("lemma") is None and f.get("event"):         # no predicate role: the event is explained by its FIRST verb only;
            ev = next((t for t in tokens if self.legend.event_of_lemma(self.lemma(t)) == f["event"]), None)   # a second verb stays content
            if ev is not None: placed = placed | {ev}                   # the frame does not explain (vss-vs-bert Q32)
        unplaced = [i for i in unknown]
        for i in kept:
            t = tokens[i]; k = self.kind(t)
            if k in self.content_kinds and t not in placed and self.lemma(t) not in placed: unplaced.append(i)
        unplaced = sorted(set(unplaced))
        # vss-demo Q16 / D27: an unplaced word that exactly one EMPTY role of the decided event admits is offered to that role as a
        # presence query ("is 'irina' the new name?") before the adjacency rule can make its neighbour undecided
        if f["shape"] == "ok" and self.cfg.get("presence_query") and unplaced:
            taken = self.legend.event_roles.get(f.get("event")) if f.get("event") else None
            empty = [r for r in self.schema["role_order"] if f["roles"].get(r) is None and (taken is None or r in taken)]
            for i in list(unplaced):
                k = self.kind(tokens[i]); fits = [r for r in empty if k in set(self.schema["roles"][r].get("head_kinds") or self.schema["roles"][r].get("kinds") or [])]
                if len(fits) == 1 and fits[0] not in f.get("offers", {}):
                    r = fits[0]; f.setdefault("offers", {})[r] = tokens[i]; f.setdefault("presence_queries", {})[r] = f["margins"].pop(r + ".presence", 0.0)
                    unplaced = [j for j in unplaced if j != i]
        undecided = self.undecided_adjacent(f, tokens, unplaced) if f["shape"] == "ok" and self.cfg.get("undecided_adjacent") else {}
        for r, (a, b) in undecided.items():                              # the role becomes a query: both candidates, no margin
            f["margins"][r] = 0.0; unplaced = [i for i in unplaced if tokens[i] != b]
        status = "ok"
        has_verb = any(self.legend.event_of_lemma(self.lemma(t)) for t in tokens) or not self.legend.reader["predicate_role"]
        nss = self.legend.reader.get("no_sense_share") or self.cfg.get("thresholds", {}).get("no_sense_share", 0.5)
        if tokens and len(unplaced) > nss * len(tokens): status = "no-sense"
        elif f["shape"] != "ok" or not has_verb: status = "unknown-event"
        if status != "ok": f = dict(f, shape="unknown")
        wires = self.frame_wires(f, ids, text, sentence)
        top = wires[0]
        mins = [v for v in f.get("margins", {}).values()]
        if f["shape"] == "ok":
            top.set("_margin.structure", fmt(f["margins"]["structure"]))
            top.set("_margin", fmt(min(mins)) if mins else "0")
            if min(mins) < self.tau: status = "abstain"
        else:
            top.set("_margin", "0")
        top.set("_status", status)
        top.lines.insert(next((i + 1 for i, l in enumerate(top.lines) if l.key == "_sentence"), 0), soplang.line("_world", self.cfg.get("world", "?")))
        mode = self.cfg.get("residual", "compiled")                      # "compiled" (the paper's) | "explained" (R41)
        res = ((1.0 if self.frame_incomplete(f) else self.residual_explained(S, addr, kept, tokens, placed)) if mode == "explained" else self.residual(S, wires, tokens)) if f["shape"] == "ok" else 1.0
        top.set("_residual", fmt(res))
        if unplaced: top.set("_unplaced", ",".join(tokens[i] for i in unplaced))
        for r, (a, b) in undecided.items(): top.set(f"_undecided.{r}", f"{a}|{b}")
        if f.get("depth_gates") is not None: top.set("_depth.gates", str(f["depth_gates"]))
        if f.get("depth_kinds") is not None: top.set("_depth.kinds", str(f["depth_kinds"]))
        if f.get("depth_fit") is not None: top.set("_depth.fit", str(f["depth_fit"]))
        self.wire_lines(wires, status)
        info = {"tokens": tokens, "S": S, "kept": kept, "unplaced": unplaced, "addr": addr, "residual": res, "status": status}
        return wires, f, info


def fmt(x): return f"{x:.3f}"


def idgen(prefix="w", start=1):
    i = start
    while True: yield f"{prefix}{i}"; i += 1


def formalize_text(reader, lines, filt=None):
    """sentences -> Doc. `reader` is one Reader, or a list of Readers: the DISPATCHER BY FIT (spec v2 section 0) — every reader reads the
    sentence, the formalization with the smallest _residual wins (a status other than no-sense / unknown-event first), and its _world names
    the winner. Directives: one `!legend <world> sha256:...` per world (check.py)."""
    readers = reader if isinstance(reader, list) else [reader]
    doc = soplang.Doc(); doc.directives = ([("world", readers[0].cfg.get("world", "?"), None)] if len(readers) == 1 else [("dispatch", "by-fit (smallest _residual)", None)])
    doc.directives += [("legend", f"{r.cfg.get('world', '?')} {r.legend.sha}", None) for r in readers]
    doc.directives.append(("generated", "sop 0.1, " + ", ".join(f"model {r.cfg.get('world', '?')}/model ({r.cfg.get('trained', '?')[:10]})" for r in readers), None))
    k = 1; frames = []
    for n, text in enumerate(lines):
        cands = []; table = None
        if "\t" in text: text, tcols = text.split("\t", 1); table = [c.strip() for c in tcols.split("|") if c.strip()]
        for r in readers:
            wires, f, info = r.formalize(text, n, idgen(start=k), filt, table=table)
            rel = info["residual"] / max(1e-9, r.no_fit_residual())          # relative residual: >= 1.0 = this reader explains almost nothing
            mm = [v for v in f.get("margins", {}).values()]; mean_margin = float(np.mean(mm)) if mm else 0.0
            cands.append(((info["status"] in ("no-sense", "unknown-event"), rel, -mean_margin), r, wires, f, info))
        cands.sort(key=lambda c: c[0]); (key, r, wires, f, info) = cands[0]
        if len(readers) > 1:
            top = wires[0]
            if key[0] or key[1] >= 1.0:     # the best reader cannot place it or explains almost nothing of it: no known world
                w = soplang.Wire(f"w{k}", "unknown"); w.set("_text", text); w.set("_sentence", str(n)); w.set("_world", "none")
                w.set("_unplaced", ",".join(info["tokens"])) if info["tokens"] else None
                w.set("_margin", "0"); w.set("_status", "no-sense"); w.set("_empty", "-"); w.set("_depth", "0"); wires = [w]
            else:
                gap = (cands[1][0][1] - key[1]) if len(cands) > 1 else 1.0
                i = next(j for j, l in enumerate(top.lines) if l.key == "_world"); top.lines.insert(i + 1, soplang.line("_world_margin", fmt(gap)))
        doc.wires += wires; frames.append((f, info))
        k = max([k] + [int(w.id[1:]) + 1 for w in wires if w.id[1:].isdigit()])
    return doc, frames


def readers_of_root(root, force=False):
    """every <root>/<world>/model that loads (section 9 / spec 5 `--worlds`)"""
    out = []
    for name in sorted(os.listdir(root)):
        md = os.path.join(root, name, "model")
        if os.path.exists(os.path.join(md, "config.json")): out.append(Reader(md, None, force))
    return out
