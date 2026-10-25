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
