"""
v8_gates: §5.2 — GATES LEARNED BY GRADIENT, on the reference (compositional) world, prefixed reporting forms.

Question. In rounds 2-6 the gates (which categories EMIT expectations) were set by hand (everything except P and O) or found by discrete
search on the small Romanian world (gates.py). Do they emerge from the single generative objective by gradient?
Criterion (judgment 2026-10-15): learned gates converge to the same set as by search — identities (P, O) closed, verbs / prepositions /
connectives open — on >= 3 worlds/seeds, with no loss in the round-6c table.

Soft chain (differentiable in the gates; everything in torch, no Python structure per step):
  g_c = sigmoid(s_c) is the probability that a word of category c emits expectations. For word k with category c_k:
    - right address (which expectation it takes): the most recent OPEN emitter before it, scanning backwards:
        p(q, k) = g_q * prod_{q<i<k} (1 - g_i)   for q < k;    p_root(k) = prod_{i<k} (1 - g_i)
        addr_k = p_root(k) * 1 + sum_{q<k} p(q, k) * addr_q (x) exp(i A[c_q, c_k])          (a recursion over positions)
    - left re-addressing (v6_left): a NON-emitter is re-addressed by the first open emitter after it:
        pL(k, q) = g_q * prod_{k<i<q} (1 - g_i)   for q > k;    r(k) = prod_{i>k} (1 - g_i)
        addrL_k = r(k) * addr_k + sum_{q>k} pL(k, q) * addr_q (x) exp(i L[c_q, c_k])
        final_k = g_k * addr_k + (1 - g_k) * addrL_k
    - term_k = final_k (x) f_{w_k}.   With hard gates (g in {0, 1}) this is the v6 chainL WITHOUT the "consumed expectation" bookkeeping
      of v2 (an expectation may be taken by several words); the variant 'hand gates (fixed)' measures what that simplification costs.
  path and bigram codes as in v4 (computed directly from category indices). Bundle: path + soft chainL + bigram.
Gates: logits s_c, one per category, trained with the phases by the same Adam. Variants: hand gates fixed (+-6, no gradient) = reference;
learned from an all-closed start (s = -2, g = 0.12); learned from an all-open start (s = +2).
Evaluation = v7_nested (compositional world, unbinding conj(R_rep)^k), per role at depths 2 and 3; gates printed per category per seed.
Run: python3 v8_gates.py [--seeds 5] [--N 30000] [--D 1024] [--batch 8] [--outer prefixed]
"""
import guard  # process-level memory/time protection (see guard.py; the machine hung on 2026-10-15)
import argparse, time
import numpy as np, torch
import v2, v5_scale as v5, v6_left as v6, v7_nested as v7

class V8(v6.V6):
    def __init__(self, world, D, rng, codes, device, gate_init="closed", hand=None):
        super().__init__(world, D, rng, v2.HAND_GATES(sorted(set(world.cat.values()))), codes, device)
        m = self.np_model; self.cats = m.cats; self.cat_id = {c: i for i, c in enumerate(self.cats)}; K = len(self.cats)
        self.ipath = torch.tensor([m.pid[f"m:{c}"] for c in self.cats], device=device)
        self.ibg0 = torch.tensor([m.pid[f"bg:^:{c}"] for c in self.cats], device=device)
        self.ibg = torch.tensor([[m.pid[f"bg:{c1}:{c2}"] for c2 in self.cats] for c1 in self.cats], device=device)
        self.iA = torch.tensor([[m.pid[f"A:{c1}:{c2}"] for c2 in self.cats] for c1 in self.cats], device=device)   # [c_emitter, c_taker]
        self.iL = torch.tensor([[m.pid[f"L:{c1}:{c2}"] for c2 in self.cats] for c1 in self.cats], device=device)
        if "pos" in codes:   # B7: the round-1 POSITION code brought into V8 — term_i = exp(i (pos[i] + pc[c_i])) (x) f_i: one displacement per
            # absolute position (shared by all categories) plus one per category; positions 16-39 added to the parameter table (V4 had 16)
            extra = [f"pos:{i}" for i in range(16, 40)]; m.names += extra; m.pid = {n: i for i, n in enumerate(m.names)}
            th = torch.cat([self.theta.detach(), torch.tensor(np.random.default_rng(int(rng.integers(1 << 30))).uniform(-np.pi, np.pi, (len(extra), D)), dtype=self.dtype, device=device)])
            self.theta = torch.nn.Parameter(th); self.P = len(m.names); self.root_mask = torch.ones(self.P, 1, dtype=self.dtype, device=device); self.root_mask[0] = 0
            self.ipos = torch.tensor([m.pid[f"pos:{i}"] for i in range(40)], device=device); self.ipc = torch.tensor([m.pid[f"pc:{c}"] for c in self.cats], device=device)
        if gate_init == "hand":
            s = torch.tensor([6.0 if (hand or v2.HAND_GATES(self.cats))[c] else -6.0 for c in self.cats], dtype=self.dtype, device=device)
            self.logits = torch.nn.Parameter(s, requires_grad=False)
        else:
            self.logits = torch.nn.Parameter(torch.full((K,), -2.0 if gate_init == "closed" else 2.0, dtype=self.dtype, device=device))

    def encode(self, sentences):
        """sentences -> (cats B x T, ids B x T, mask B x T); the structure is computed in torch from the categories"""
        w = self.w; B = len(sentences); T = max(len(s) for s in sentences)
        cats = torch.zeros(B, T, dtype=torch.long); ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, toks in enumerate(sentences):
            cats[b, :len(toks)] = torch.tensor([self.cat_id[w.cat[t]] for t in toks]); ids[b, :len(toks)] = torch.tensor([w.idx[t] for t in toks]); mask[b, :len(toks)] = 1
        return cats.to(self.device), ids.to(self.device), mask.to(self.device)

    def states(self, cats, ids, mask):
        th = self.theta * self.root_mask; B, T = cats.shape; one = torch.ones((), dtype=self.dtype, device=self.device)
        E = lambda ph: torch.polar(torch.ones_like(ph), ph)
        terms = 0
        if "path" in self.codes:
            terms = terms + E(torch.cumsum(th[self.ipath[cats]] * mask.unsqueeze(-1), 1))
        if "bigram" in self.codes:
            ph = th[self.ibg[cats[:, :-1], cats[:, 1:]]]; ph = torch.cat([th[self.ibg0[cats[:, :1]]], ph], 1); terms = terms + E(ph)
        if "pos" in self.codes:                                                       # B7: position code (round 1), see __init__
            posidx = torch.arange(T, device=self.device).clamp(max=39); pterm = E(th[self.ipos[posidx]].unsqueeze(0) + th[self.ipc[cats]])
            if getattr(self, "pos_max", None) is not None: pterm = pterm * (posidx < self.pos_max).to(pterm.dtype).unsqueeze(0).unsqueeze(-1)   # B7 (b): the position term is OMITTED beyond the trained range, not drawn at random
            terms = terms + pterm
        if "chainL" in self.codes or "chain" in self.codes:
            g = torch.sigmoid(self.logits)[cats] * mask                                   # B x T
            u = torch.log1p(-g + 1e-9); Sx = torch.cat([torch.zeros(B, 1, dtype=self.dtype, device=self.device), torch.cumsum(u, 1)], 1)   # Sx[i] = sum_{j<i} u_j
            EA = E(th[self.iA[cats.unsqueeze(2), cats.unsqueeze(1)]])                   # B x T(q) x T(k) x D : A[c_q, c_k]
            addrs = []                                                                 # lists, not in-place writes (autograd)
            for k in range(T):
                p_root = torch.exp(Sx[:, k]).unsqueeze(-1) * one
                if k == 0: addrs.append(p_root.to(EA.dtype).expand(B, th.shape[1])); continue
                prev = torch.stack(addrs, 1)                                              # B x k x D
                p = g[:, :k] * torch.exp(Sx[:, k].unsqueeze(1) - Sx[:, 1:k + 1])          # B x k : g_q prod_{q<i<k}(1-g_i)
                addrs.append(p_root.to(EA.dtype) + torch.einsum("bq,bqd->bd", p.to(EA.dtype), prev * EA[:, :k, k]))
            addr = torch.stack(addrs, 1)                                                  # B x T x D
            if "chainL" in self.codes:
                EL = E(th[self.iL[cats.unsqueeze(1), cats.unsqueeze(2)]])               # B x T(k) x T(q) x D : L[c_q, c_k]
                addrLs = []
                for k in range(T):
                    r = torch.exp(Sx[:, T] - Sx[:, k + 1])                                # prod_{i>k} (1-g_i)
                    acc = r.unsqueeze(-1).to(EA.dtype) * addr[:, k]
                    if k < T - 1:
                        pL = g[:, k + 1:] * torch.exp(Sx[:, k + 1:T] - Sx[:, k + 1].unsqueeze(1))   # g_q prod_{k<i<q}(1-g_i)
                        acc = acc + torch.einsum("bq,bqd->bd", pL.to(EA.dtype), addr[:, k + 1:] * EL[:, k, k + 1:])
                    addrLs.append(acc)
                addrL = torch.stack(addrLs, 1)
                gk = g.unsqueeze(-1).to(EA.dtype); addr = gk * addr + (1 - gk) * addrL
            terms = terms + addr
        if getattr(self, "vo", None) is not None:                                        # the verb-order route (set_vorder)
            side, vcat, ordinal, valid = self.vorder_index(cats, mask)
            ph = self.vo[side, vcat, cats, ordinal]                                       # B x T x D
            terms = terms + E(ph) * valid.to(self.dtype).unsqueeze(-1)
        if getattr(self, "rel", None) is not None:                                       # C2b: second drawing of the taker, relative to its own address
            B, T = cats.shape; z1 = torch.zeros(B, 1, dtype=self.dtype, device=self.device)
            sel = torch.cat([z1, (self.rel_mask[cats] * mask)[:, :-1] * self.taker_mask[cats][:, 1:]], 1)      # word k takes if k-1 is an emitter and k a taker
            relph = torch.cat([torch.zeros(B, 1, self.rel.shape[1], dtype=self.dtype, device=self.device), self.rel[cats[:, :-1]]], 1)
            terms = terms + sel.unsqueeze(-1).to(terms.dtype) * terms * E(relph)
        return (terms * self.F[ids] * mask.unsqueeze(-1)).sum(1)

    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None, batch=1, depth_train=2):
        w = self.w; sents, targets = [], []
        for n in range(n_sent):
            d = int(self.rng.integers(depth_train + 1)); ev = w.event(persons, objects, depth=d)
            sents.append(w.render(ev, self.rng.choice(list(templates)))); targets.append(world.state(ev))
        t0 = time.time(); cats, ids, mask = self.encode(sents); self.t_encode = time.time() - t0
        tgt_all = torch.tensor(np.array(targets), dtype=self.F.dtype, device=self.device)
        params = [self.theta] + ([self.logits] if self.logits.requires_grad else [])
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        for i in range(0, n_sent, batch):
            sl = slice(i, i + batch); S = self.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            opt.zero_grad(); (-cos.mean()).backward(); opt.step()
            with torch.no_grad(): self.theta[0] = 0

    def set_vorder(self, verb_cats, nmax=4, cue_cats=()):
        """the VERB-ORDER route (vss-demo 2026-10-25, vss-vs-bert R47; addendum q-formalisation 11): every non-verb word is drawn a second
        time at a learned displacement keyed by (side, the category of the nearest verb, the word's own category, its ordinal among the words
        of its category since that verb): side 1 = after the nearest verb to the left, counted from that verb; side 0 = no verb to the left,
        keyed by the nearest verb to the right and counted from the sentence start. Independent of the cue chain, so "gives p4 o3" and
        "gives o3 to p4" draw p4 and o3 at the same verb-order addresses. One phase vector per key, learned; omitted for verbs and when the
        sentence has no verb. Returns the verb categories found."""
        K = len(self.cats); self.vo_nmax = int(nmax)
        self.verb_mask = torch.zeros(K, dtype=self.dtype, device=self.device)
        found = [c for c in verb_cats if c in self.cat_id]
        for c in found: self.verb_mask[self.cat_id[c]] = 1
        # variant b (vss-vs-bert R58): a word introduced by a role cue (the previous word's category in cue_cats) is CUED: it is left to the
        # cue chain, drawn by no order term and not counted in the ordinals; the order route addresses the BARE fillers only
        self.cue_mask = torch.zeros(K, dtype=self.dtype, device=self.device)
        for c in cue_cats:
            if c in self.cat_id: self.cue_mask[self.cat_id[c]] = 1
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        self.vo = torch.nn.Parameter(torch.tensor(rng.uniform(-np.pi, np.pi, (2, K, K, self.vo_nmax, self.theta.shape[1])), dtype=self.dtype, device=self.device))
        return found

    def vorder_index(self, cats, mask):
        """(side B x T, vcat B x T, ord B x T, valid B x T) of set_vorder, from the category indices alone (torch, no Python per word)"""
        B, T = cats.shape; dev = cats.device; K = len(self.cats)
        isv = (self.verb_mask[cats] * mask) > 0                                                       # B x T
        idx = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        left = torch.where(isv, idx, torch.full_like(idx, -1)); left = torch.cummax(left, 1).values    # nearest verb at or left of k
        left_excl = torch.cat([torch.full((B, 1), -1, dtype=left.dtype, device=dev), left[:, :-1]], 1)  # strictly left of k
        big = T + 1
        right = torch.where(isv, idx, torch.full_like(idx, big)); right = torch.flip(torch.cummin(torch.flip(right, [1]), 1).values, [1])
        right_excl = torch.cat([right[:, 1:], torch.full((B, 1), big, dtype=right.dtype, device=dev)], 1)   # strictly right of k
        has_l = left_excl >= 0; has_r = right_excl < big
        side = has_l.long(); vpos = torch.where(has_l, left_excl, right_excl.clamp(max=T - 1))
        vcat = torch.gather(cats, 1, vpos)
        cued = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), (self.cue_mask[cats][:, :-1] * mask[:, :-1]) > 0], 1)   # B x T
        onehot = torch.nn.functional.one_hot(cats, K).to(torch.long) * (mask.unsqueeze(-1).long() * (~cued).long().unsqueeze(-1))   # bare words only
        cum = torch.cumsum(onehot, 1)                                                                 # inclusive counts
        own = torch.gather(cum, 2, cats.unsqueeze(-1)).squeeze(-1)                                    # count of own category up to k inclusive
        before_k = own - 1                                                                            # strictly before k
        at_v = torch.gather(cum, 1, vpos.unsqueeze(-1).expand(B, T, K)); at_v = torch.gather(at_v, 2, cats.unsqueeze(-1)).squeeze(-1)
        ordinal = torch.where(has_l, before_k - at_v, before_k).clamp(min=0, max=self.vo_nmax - 1)
        valid = ((has_l | has_r) & ~isv & ~cued) & (mask > 0)
        return side, vcat, ordinal, valid

    def set_rel_emitters(self, emitters, takers):
        """C2b (judgment 3b4f837 E2): a BEACON-RELATIVE emitter. A word of an emitter category draws the NEXT word (if its category is a taker)
        a second time at (the taker's own address) (x) exp(i REL[c_emitter]) (x) f_taker — one learned displacement per emitter category,
        relative to the taker's slot; the probe's cumulative position is untouched (the emitter's gate is whatever it is). Not a new term type:
        a word star at an address. Returns the emitter categories found."""
        K = len(self.cats); self.rel = torch.nn.Parameter(torch.zeros(K, self.theta.shape[1], dtype=self.dtype, device=self.device))
        self.rel_mask = torch.zeros(K, dtype=self.dtype, device=self.device); self.taker_mask = torch.zeros(K, dtype=self.dtype, device=self.device)
        for c in emitters:
            if c in self.cat_id: self.rel_mask[self.cat_id[c]] = 1
        for c in takers:
            if c in self.cat_id: self.taker_mask[self.cat_id[c]] = 1
        return [c for c in emitters if c in self.cat_id]

    def fix_gates(self, cats, open_):
        """rule 20: gates GIVEN for these categories (not learned): logit +8 (open) or -8 (closed), gradient masked. Returns the categories fixed."""
        idx = [self.cat_id[c] for c in cats if c in self.cat_id]
        with torch.no_grad(): self.logits[idx] = 8.0 if open_ else -8.0
        mask = torch.ones_like(self.logits); mask[idx] = 0
        self.logits.register_hook(lambda g: g * mask)
        return [self.cats[i] for i in idx]
    def gates(self): return {c: float(torch.sigmoid(self.logits[i])) for i, c in enumerate(self.cats)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--lang", default="en")
    ap.add_argument("--outer", default="prefixed"); ap.add_argument("--variants", default="all")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang)
    if a.outer == "prefixed": v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v8_gates: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    print(f"reference world: compositional (v7_nested), {v5.LANG}, outer forms {v5.OUTER}; training formulations {v5.TRAIN_T}, depths 0-{a.depth_train}, test depth 3")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0)
    cues_ok, cues_bad = v5.split_templates_rolecues(w0)
    print(f"  => rebuilt axis: new formulations with ALL ROLE CUES seen (same role): {cues_ok}; with unseen role cues: {cues_bad}")
    cats0 = sorted(set(w0.cat.values())); print(f"categories ({len(cats0)}): {cats0}")
    print(f"hand gates (reference set): open = everything except P, O. chance: seen ids 1/12 = {1/12:.3f}; new ids 1/6 = {1/6:.3f}\n")
    variants = [("hand gates (fixed)", "hand", ["path", "chainL", "bigram"]), ("learned, start closed", "closed", ["path", "chainL", "bigram"]),
                ("learned, start open", "open", ["path", "chainL", "bigram"]),
                ("chainL ALONE, hand", "hand", ["chainL"]), ("chainL ALONE, learned closed", "closed", ["chainL"]), ("chainL ALONE, learned open", "open", ["chainL"])]
    if a.variants == "learned": variants = [v for v in variants if v[1] != "hand"]
    if a.variants == "bundle": variants = variants[:3]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "new form., ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]
    results = {n: [] for n, _, _ in variants}; roles3 = {n: [] for n, _, _ in variants}; gates_all = {n: [] for n, _, _ in variants}
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed))
        trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, init, codes in variants:
            m = V8(w, D, np.random.default_rng(200 + seed), codes, a.device, gate_init=init); m.set_F(F)
            assert m.F_np is world.F
            t0 = time.time(); m.train(world, N, templates=v5.TRAIN_T, persons=trP, objects=trO, batch=a.batch, lr=a.lr, depth_train=a.depth_train)
            torch.cuda.synchronize(); t_train = time.time() - t0
            d2, pr2 = v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 2, n=15, per_role=True)
            d3, pr3 = v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 3, n=15, per_role=True)
            r = [v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 0, n=20), v7.decode_nested(m, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
                 v7.decode_nested(m, world, w, held_seen, trP, trO, 0, n=30), v7.decode_nested(m, world, w, held_new, trP, trO, 0, n=30),
                 v7.decode_nested(m, world, w, cues_ok, trP, trO, 0, n=30) if cues_ok else (float("nan"), float("nan")),
                 v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 1, n=15), d2, d3]
            results[name].append(r); roles3[name].append([pr3[k] for k in ("A", "B", "C", "X1", "X2", "X3")]); gt = m.gates(); gates_all[name].append(gt)
            print(f"  seed {seed} {name:28s} ({t_train:5.1f}s train) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r)
                  + "  || depth 3 per role: " + " ".join(f"{k} {pr3[k]:.2f}" for k in ("A", "B", "C", "X1", "X2", "X3")), flush=True)
            if init != "hand":
                op = sorted([c for c, v in gt.items() if v > 0.5]); cl = sorted([c for c, v in gt.items() if v <= 0.5])
                print(f"           gates: OPEN {op}\n                  CLOSED {cl}", flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    for name, _, _ in variants:
        r = np.array(results[name]); print(f"  {name:28s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3")
    for name, _, _ in variants:
        r = np.array(roles3[name]); print(f"  {name:28s} " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(6)))
    print("== learned gates, mean sigmoid over seeds (OPEN if > 0.5); hand reference: P, O closed, all else open")
    for name, init, _ in variants:
        if init == "hand": continue
        g = {c: np.mean([gt[c] for gt in gates_all[name]]) for c in cats0}
        agree = sum((g[c] > 0.5) == (c not in ("P", "O")) for c in cats0)
        print(f"  {name:28s} agreement with hand set {agree}/{len(cats0)}: " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])))
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control = chance [x] same regime (13) [x] no new role at new depth (14) [x] per role (15)")
