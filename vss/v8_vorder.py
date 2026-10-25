"""v8_vorder.py (responses R56, 2026-10-25): the verb-order route ('vorder') as a SUBCLASS of the main repository's v8_gates.V8.
Reference implementation: vss-demo commit 434fa40 (experiment/v8_gates.py, a 41-line diff reproduced in responses R56); it lives here as a
transcription because third_party/vsa-semantic-space is pinned at e531283 and read-only. Nothing else of V8 changes: the same three routes
(path, chainL, bigram), the gates learned from the closed start, 1a's lr / N / batch.
- set_vorder(verb_cats, nmax=4): verb_mask[K] = 1 for the verb categories; vo = Parameter (2, K, K, nmax, D), uniform(-pi, pi) from
  np.random.default_rng(int(self.rng.integers(1 << 30))) (one draw of the model's rng, as the reference: the training sentences that
  follow are therefore not 1a's); vo is added to the optimiser's parameters (train() below is V8.train with that one change).
- vorder_index(cats, mask) -> (side, vcat, ordinal, valid), all B x T, as R56 specifies.
- states(): V8.states plus E(vo[side, vcat, cats, ordinal]) * valid at every token, deposited like every term (x F[ids] x mask); the
  term is added after the chain term in the reference and the relative-emitter term is not used here (rel is None), so adding it to the
  state after V8.states is the same sum.
Variant b (responses R58, reference sop/v8_gates_vorder_ref.py from vss-demo 561cbcb): set_vorder(..., cue_cats=['to', 'by', 'from']):
a word introduced by a role cue is CUED (no order term, not counted in the ordinals); cue_cats=() is variant a.
Without set_vorder() the class is V8 exactly (tests: tests/test_vorder.py, EQUAL to 1a on seed 0)."""
import numpy as np, torch
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import v8_gates

NONVERB = {"P", "O", "COP", "from", ",", "what", "it", "to", "who", "by", ".", "that"}   # every other category of the basic world is a verb
                                                                                         # kind, a participle, or the opener says / claims
def verb_categories(cats): return [c for c in cats if c not in NONVERB]

class V8Vorder(v8_gates.V8):
    vo = None
    def set_vorder(self, verb_cats, nmax=4, cue_cats=()):
        K = len(self.cats); D = self.theta.shape[1]; self.nmax = nmax
        # variant b (R58; sop/v8_gates_vorder_ref.py): a word whose PREVIOUS word's category is in cue_cats is CUED: drawn by no order term
        # and not counted in the ordinals; cue_cats=() is variant a (R56) exactly
        self.cue_mask = torch.zeros(K, dtype=self.dtype, device=self.device)
        for c in cue_cats:
            if c in self.cat_id: self.cue_mask[self.cat_id[c]] = 1
        self.verb_mask = torch.zeros(K, dtype=self.dtype, device=self.device)
        for c in verb_cats:
            if c in self.cat_id: self.verb_mask[self.cat_id[c]] = 1
        r = np.random.default_rng(int(self.rng.integers(1 << 30)))
        self.vo = torch.nn.Parameter(torch.tensor(r.uniform(-np.pi, np.pi, (2, K, K, nmax, D)), dtype=self.dtype, device=self.device))
        return [c for c in verb_cats if c in self.cat_id]

    def vorder_index(self, cats, mask):
        B, T = cats.shape; K = len(self.cats); dev = cats.device
        isv = (self.verb_mask[cats] * mask) > 0; idx = torch.arange(T, device=dev).expand(B, T)
        cm = torch.cummax(torch.where(isv, idx, torch.full_like(idx, -1)), 1).values
        left_excl = torch.cat([torch.full((B, 1), -1, device=dev, dtype=cm.dtype), cm[:, :-1]], 1)
        rc = torch.flip(torch.cummin(torch.flip(torch.where(isv, idx, torch.full_like(idx, T + 1)), [1]), 1).values, [1])
        right_excl = torch.cat([rc[:, 1:], torch.full((B, 1), T + 1, device=dev, dtype=rc.dtype)], 1)
        has_l = left_excl >= 0; has_r = right_excl <= T - 1
        side = has_l.long(); vpos = torch.where(has_l, left_excl, right_excl).clamp(0, T - 1)
        vcat = torch.gather(cats, 1, vpos)
        cued = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), (self.cue_mask[cats][:, :-1] * mask[:, :-1]) > 0], 1)
        cum = torch.cumsum(torch.nn.functional.one_hot(cats, K).to(torch.long) * ((mask > 0) & ~cued).long().unsqueeze(-1), 1)   # bare words, inclusive
        before_k = torch.gather(cum, 2, cats.unsqueeze(-1)).squeeze(-1) - 1
        at_v = torch.gather(torch.gather(cum, 1, vpos.unsqueeze(-1).expand(B, T, K)), 2, cats.unsqueeze(-1)).squeeze(-1)
        ordinal = torch.where(has_l, before_k - at_v, before_k).clamp(0, self.nmax - 1)
        valid = (has_l | has_r) & ~isv & ~cued & (mask > 0)
        return side, vcat, ordinal, valid

    def states(self, cats, ids, mask):
        S = super().states(cats, ids, mask)
        if self.vo is None: return S
        side, vcat, ordinal, valid = self.vorder_index(cats, mask)
        ph = self.vo[side, vcat, cats, ordinal]                                                  # B x T x D
        term = torch.polar(torch.ones_like(ph), ph) * valid.to(ph.dtype).unsqueeze(-1)
        return S + (term * self.F[ids] * mask.unsqueeze(-1)).sum(1)

    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None, batch=1, depth_train=2):
        """V8.train (v8_gates.py lines 104-117 at e531283) with vo in the optimiser's parameters"""
        w = self.w; sents, targets = [], []
        for n in range(n_sent):
            d = int(self.rng.integers(depth_train + 1)); ev = w.event(persons, objects, depth=d)
            sents.append(w.render(ev, self.rng.choice(list(templates)))); targets.append(world.state(ev))
        cats, ids, mask = self.encode(sents)
        tgt_all = torch.tensor(np.array(targets), dtype=self.F.dtype, device=self.device)
        params = [self.theta] + ([self.logits] if self.logits.requires_grad else []) + ([self.vo] if self.vo is not None else [])
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        for i in range(0, n_sent, batch):
            sl = slice(i, i + batch); S = self.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            opt.zero_grad(); (-cos.mean()).backward(); opt.step()
            with torch.no_grad(): self.theta[0] = 0
