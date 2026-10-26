"""
typer.py: the KINDS of the words of a world learned from its annotated examples, for worlds whose lexicon is not given (real data).

The reader takes a word's kind as an input (rule 20: kinds are given). On a constructed world the lexicon gives them; on real
utterances (SNIPS, MASSIVE) the kinds have to come from somewhere. The earlier recipe (skip-gram clusters of the training text) puts most
slot values of an unseen name into a cluster unrelated to its role. This module learns the kinds from the training annotation instead:

  - the --n_freq most frequent training tokens are their own kind (`w:<word>`), as before (function words, verbs, frequent cues);
  - a token annotated inside a slot span of type T in training has kind `S:<T>` (the token's annotated type, per occurrence);
  - every other token has kind `O`;
  - a small context classifier (multinomial logistic regression over hashed features: character 2-4-grams of the word, its shape,
    the previous and next two words) is trained on the training tokens that are not their own kind, and predicts the kind of a token
    in context. A word's lexicon kind is the majority of its predicted kinds over the sentences it occurs in (the reader's lexicon is
    per word). No gold label of an evaluation sentence is used: only its text.

The classifier is a tagger of entity types, not of roles: it answers "what kind of thing is this word" (a playlist name, a city, a
time); the reader still answers "which question of the legend does it answer", the intent, the margins, the residual and the queries.
Its size is reported with the reader's.

spans: `kind_run(tokens, j, kind)`: the maximal run of consecutive tokens of the same S: kind containing position j; the reader's answer
to a slot question is then that run (its head = the run's last token), not the single word.

numpy only. Deterministic for a given seed.
"""
import collections, zlib
import numpy as np

SHAPES = (("X", str.isupper), ("x", str.islower), ("d", str.isdigit))


def shape(w):
    out = []
    for ch in w:
        c = next((k for k, f in SHAPES if f(ch)), ch if not ch.isalnum() else "o")
        if not out or out[-1] != c: out.append(c)
    return "".join(out)[:6]


def features(tokens, j):
    w = tokens[j].lower(); b = f"<{w}>"
    f = [f"w={w}", f"sh={shape(tokens[j])}"]
    f += [f"c{n}={b[i:i + n]}" for n in (2, 3, 4) for i in range(max(0, len(b) - n + 1))]
    for d in (-2, -1, 1, 2):
        k = j + d; f.append(f"p{d}={tokens[k].lower() if 0 <= k < len(tokens) else '<s>' if k < 0 else '</s>'}")
    f.append(f"p-1p1={tokens[j - 1].lower() if j > 0 else '<s>'}|{tokens[j + 1].lower() if j + 1 < len(tokens) else '</s>'}")
    return f


def hidx(feat, bits):
    return zlib.crc32(feat.encode("utf-8")) & ((1 << bits) - 1)


class Typer:
    def __init__(self, n_freq=40, bits=16, epochs=8, lr=0.5, l2=1e-6, seed=0):
        self.n_freq, self.bits, self.epochs, self.lr, self.l2, self.seed = n_freq, bits, epochs, lr, l2, seed

    # ---------------- training
    def labels(self, tokens, spans):
        """per-token kind of an annotated sentence: spans = [(type, start, end), ...]"""
        lab = ["O"] * len(tokens)
        for t, s, e in spans:
            for k in range(s, e): lab[k] = "S:" + t
        return lab

    def fit(self, sentences):
        """sentences: [(tokens, spans)]"""
        freq = collections.Counter(w for toks, _ in sentences for w in toks)
        self.singles = {w for w, _ in freq.most_common(self.n_freq)}
        X, Y, kinds = [], [], {}
        for toks, spans in sentences:
            lab = self.labels(toks, spans)
            for j, w in enumerate(toks):
                y = lab[j] if lab[j] != "O" or w not in self.singles else "W"      # a frequent word outside a span: its own kind (state W)
                X.append([hidx(f, self.bits) for f in features(toks, j)]); Y.append(kinds.setdefault(y, len(kinds)))
        self.kinds = sorted(kinds, key=kinds.get); K = len(self.kinds); H = 1 << self.bits
        self.W = np.zeros((H, K), dtype=np.float32); self.b = np.zeros(K, dtype=np.float32)
        rng = np.random.default_rng(self.seed); Y = np.array(Y); n = len(X)
        Wsum = np.zeros_like(self.W); bsum = np.zeros_like(self.b); steps = 0
        for ep in range(self.epochs):
            lr = self.lr / (1 + ep)
            for i in rng.permutation(n):
                idx = X[i]; z = self.W[idx].sum(0) + self.b; z -= z.max(); p = np.exp(z); p /= p.sum()
                p[Y[i]] -= 1.0
                self.W[idx] -= lr * p[None, :]; self.b -= lr * p
            Wsum += self.W; bsum += self.b; steps += 1
        self.W, self.b = (Wsum / steps).astype(np.float32), (bsum / steps).astype(np.float32)   # averaged over epochs
        self.train_acc = float(np.mean([self._argmax(X[i]) == Y[i] for i in range(min(n, 5000))]))
        # kind transitions over the training sentences (singles collapsed to one state 'W'), for a Viterbi decoding of a sentence
        st = {k: i for i, k in enumerate(self.kinds)}; self.states = st; T = np.ones((len(st) + 1, len(st)))   # +1: start
        for toks, spans in sentences:
            lab = self.labels(toks, spans); prev = len(st)
            for j, w in enumerate(toks):
                cur = st[lab[j] if lab[j] != "O" or w not in self.singles else "W"]; T[prev, cur] += 1; prev = cur
        self.logT = np.log(T / T.sum(1, keepdims=True))
        return self

    def _argmax(self, idx):
        return int(np.argmax(self.W[idx].sum(0) + self.b))

    # ---------------- prediction
    def kind_in_context(self, tokens, j):
        k = self.kinds[self._argmax([hidx(f, self.bits) for f in features(tokens, j)])]
        return "w:" + tokens[j] if k == "W" and tokens[j] in self.singles else ("O" if k == "W" else k)

    def sentence_kinds(self, tokens, viterbi=True):
        if not viterbi or not hasattr(self, "logT"): return [self.kind_in_context(tokens, j) for j in range(len(tokens))]
        st = self.states; nS = len(st); W = st.get("W", -1); n = len(tokens)
        em = np.full((n, nS), -1e9)
        for j, w in enumerate(tokens):
            z = self.W[[hidx(f, self.bits) for f in features(tokens, j)]].sum(0) + self.b; z = z - z.max(); em[j] = z - np.log(np.exp(z).sum())
            if w not in self.singles and W >= 0: em[j, W] = -1e9                     # only a frequent word can be its own kind
        d = self.logT[nS] + em[0]; back = []
        for j in range(1, n):
            sc = d[:, None] + self.logT[:nS] + em[j][None, :]; back.append(sc.argmax(0)); d = sc.max(0)
        path = [int(d.argmax())]
        for bp in reversed(back): path.append(int(bp[path[-1]]))
        path.reverse(); inv = {i: k for k, i in st.items()}
        return ["w:" + tokens[j] if path[j] == W else inv[path[j]] for j in range(n)]


    def lexicon(self, sentences):
        """{word: kind}: per word, the majority of its in-context kinds over the given sentences (text only)"""
        votes = collections.defaultdict(collections.Counter)
        for toks in sentences:
            for j, k in enumerate(self.sentence_kinds(toks)): votes[toks[j]][k] += 1
        return {w: c.most_common(1)[0][0] for w, c in votes.items()}

    def n_params(self):
        return int(self.W.size + self.b.size)

    # ---------------- export (R91): the table the browser needs, pruned and quantised
    def export(self, out_dir):
        """typer.json + typer.bin in out_dir. Only the hashed rows that any training feature touched are kept (the others are zero and
        act as zero in the browser: a bucket absent from the index contributes nothing); each kept row is stored as int8 with one
        float32 scale per row (symmetric: w = q * scale). The JSON carries the kinds, the frequent words (their own kind), the start
        and transition log-probabilities of the Viterbi, bits and the feature recipe; a JS mirror reproduces features() and hidx()
        (CRC-32 of the UTF-8 feature string, masked to bits) exactly. Returns the sizes."""
        import json, os
        os.makedirs(out_dir, exist_ok=True)
        rows = np.where(np.abs(self.W).sum(1) > 0)[0].astype(np.int32)
        Wk = self.W[rows]; scale = (np.maximum(np.abs(Wk).max(1), 1e-12) / 127.0).astype(np.float32)
        q = np.clip(np.round(Wk / scale[:, None]), -127, 127).astype(np.int8)
        with open(os.path.join(out_dir, "typer.bin"), "wb") as f:
            f.write(rows.tobytes()); f.write(scale.tobytes()); f.write(q.tobytes())
        meta = {"format": "sop typer 1", "bits": self.bits, "n_freq": self.n_freq, "kinds": list(self.kinds), "singles": sorted(self.singles),
                "bias": [float(x) for x in self.b], "logT": self.logT.tolist(), "states": self.states, "rows": int(len(rows)), "K": int(len(self.kinds)),
                "layout": "typer.bin = int32 rows[n] (bucket index of each kept row, ascending) | float32 scale[n] | int8 q[n][K] (row-major); w[bucket] = q * scale; absent buckets are zero",
                "features": "w=<lower word>; sh=<shape: runs of X (upper) x (lower) d (digit) o (other alnum) or the char, first 6>; c2/c3/c4=<char n-grams of '<'+word+'>'>; "
                            "p-2,p-1,p1,p2=<lower neighbour or <s> / </s>>; p-1p1=<prev>|<next>; bucket = crc32(utf8(feature)) & ((1 << bits) - 1)",
                "decode": "kind_in_context: argmax(sum_f w[bucket_f] + bias), state W -> 'w:' + word if word in singles else 'O'; sentence_kinds: Viterbi over logT with "
                          "emissions log_softmax(z), state W forbidden for a word not in singles",
                "quantisation": "int8 symmetric per row, scale = max |w| of the row / 127, w = q * scale"}
        with open(os.path.join(out_dir, "typer.json"), "w") as f: json.dump(meta, f)
        return {"rows_kept": int(len(rows)), "rows_total": int(self.W.shape[0]), "bytes_bin": int(len(rows) * (4 + 4 + len(self.kinds))),
                "bytes_json": os.path.getsize(os.path.join(out_dir, "typer.json"))}

    @classmethod
    def load_export(cls, out_dir):
        """the exported typer read back (the reference for the parity test of the JS mirror): a Typer whose W holds the dequantised rows"""
        import json, os
        meta = json.load(open(os.path.join(out_dir, "typer.json"))); n, K = meta["rows"], meta["K"]
        raw = open(os.path.join(out_dir, "typer.bin"), "rb").read()
        rows = np.frombuffer(raw[:4 * n], dtype=np.int32); scale = np.frombuffer(raw[4 * n:8 * n], dtype=np.float32); q = np.frombuffer(raw[8 * n:], dtype=np.int8).reshape(n, K)
        t = cls(n_freq=meta["n_freq"], bits=meta["bits"]); t.kinds = meta["kinds"]; t.singles = set(meta["singles"]); t.b = np.array(meta["bias"], dtype=np.float32)
        t.logT = np.array(meta["logT"]); t.states = meta["states"]
        t.W = np.zeros((1 << meta["bits"], K), dtype=np.float32); t.W[rows] = q.astype(np.float32) * scale[:, None]
        return t


def kind_run(kinds, j):
    """the maximal run of consecutive positions with kind == kinds[j] (an S: kind) containing j: (start, end)"""
    k = kinds[j]; s = e = j
    if not str(k).startswith("S:"): return j, j + 1
    while s > 0 and kinds[s - 1] == k: s -= 1
    while e + 1 < len(kinds) and kinds[e + 1] == k: e += 1
    return s, e + 1
