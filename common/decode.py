"""decode.py: decoding with the WORLD's addresses on the frozen splits, for every model that has the main repository's interface
(encode(sentences) -> (x, ids, mask); states(x, ids, mask) -> complex S; .F the identity table; .device): the reader (v8_gates.V8), the 7b
transformer (v9_transformer.TModel) and the 7b-bis transformer (v22_transformer_categories.TModelCat). It is decode_nested of
v7_nested.py applied to STORED sentences instead of freshly drawn ones, with the same unbinding (conj(R_role) * conj(R_rep)^level) and
the same restricted candidate lists, returning one prediction per sentence in the scorer's format (eval/metrics.py)."""
import numpy as np, torch
from . import basic_world as bwm

def decode_frozen(m, world, split, axis, batch=256):
    items = split["axes"][axis]; preds = []
    if not items: return preds
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); conj_rep = torch.conj(R["rep"]); w = world.w
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; enc = m.encode([it["tokens"] for it in chunk])
        with torch.no_grad(): S = m.states(*enc)
        Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
        depths = sorted(set(it["depth"] for it in chunk)); out = [{"frame": {}, "margins": {}} for _ in chunk]
        for d in depths:
            rows = [j for j, it in enumerate(chunk) if it["depth"] == d]
            for r in bwm.roles_at(d):
                level = int(r[1:]) - 1 if r.startswith("X") else d
                key = R["X"] if r.startswith("X") else R[r]; unbind = torch.conj(key) * conj_rep ** level
                cands = bwm.candidates_for(split, axis, r); cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
                sc = torch.real((Sn[rows] * unbind) @ torch.conj(Fn[cid]).T).cpu().numpy()
                top = np.argsort(-sc, axis=1)
                for k, j in enumerate(rows):
                    out[j]["frame"][r] = cands[top[k, 0]]; out[j]["margins"][r] = float(sc[k, top[k, 0]] - sc[k, top[k, 1]])
                    out[j].setdefault("cand_scores", {})[r] = (list(cands), sc[k].tolist())   # R10 item 1: for the one-to-one rule
            for j in rows:
                for r in bwm.ROLES:
                    if r not in out[j]["frame"]: out[j]["frame"][r] = None
        preds.extend(out)
    return preds

# R17 (responses R17 / R18, 2026-10-25): depth estimated from the level openers, the sop tool's residual, nonce words as placeholders
REPORTING = ("says", "claims")

def depth_from_openers(tokens, max_depth=3):
    """the depth of a sentence from its level openers: the count of 'that' right after a reporting verb (the sop tool's default; a
    property of the token sequence, the same for every model), clipped to max_depth"""
    return min(max_depth, sum(1 for i in range(1, len(tokens)) if tokens[i] == "that" and tokens[i - 1] in REPORTING))

def residuals(m, world, items, preds, batch=256):
    """the sop tool's residual (tool/sop/infer.py: ||S/|S| - a rec/|rec| || with a = max(0, Re <rec, S>) on the unit vectors), rec = the
    world's state of the decoded frame at the item's depth; 1.0 when a role of that depth is not decoded"""
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]; enc = m.encode([it["tokens"] for it in chunk])
        with torch.no_grad(): S = m.states(*enc).cpu().numpy()
        for k, (it, p) in enumerate(zip(chunk, preds[i:i + batch])):
            roles = bwm.roles_at(it["depth"]); f = p["frame"]
            if any(f.get(r) is None for r in roles): out.append(1.0); continue
            ev = (None, f["A"], f["B"], f["C"]) + tuple(f[f"X{j+1}"] for j in range(it["depth"]))
            rec = world.state(ev); s = S[k]; ns, nr = np.linalg.norm(s), np.linalg.norm(rec)
            if ns < 1e-9 or nr < 1e-9: out.append(1.0); continue
            sn, rn = s / ns, rec / nr; a = max(0.0, float(np.real(np.vdot(rn, sn)))); out.append(float(np.linalg.norm(sn - a * rn)))
    return out

class Placeholders:
    """R18 (Q14), the row 'nonce as placeholder': every word outside the lexicon is read as ONE fixed random vector of category 'O' per
    seed (rng 80_000 + seed), so that the reader must decide from its state. Used as a context manager around decoding; it extends the
    model's identity table and the world's lexicon of THIS process only (nothing in the main repository is modified) and restores them."""
    TOKEN = "__nonce__"
    def __init__(self, m, w, seed): self.m, self.w, self.seed = m, w, seed
    def __enter__(self):
        m, w = self.m, self.w; D = m.F.shape[1]; self.F0, self.n0 = m.F, len(w.idx)
        v = np.exp(1j * np.random.default_rng(80_000 + self.seed).uniform(-np.pi, np.pi, D))
        m.F = torch.cat([m.F, torch.tensor(v, dtype=m.F.dtype, device=m.F.device).unsqueeze(0)]); w.idx[self.TOKEN] = self.n0; w.cat[self.TOKEN] = "O"
        return self
    def map(self, tokens): return [t if t in self.w.idx else self.TOKEN for t in tokens]
    def __exit__(self, *exc):
        self.m.F = self.F0; del self.w.idx[self.TOKEN]; del self.w.cat[self.TOKEN]
