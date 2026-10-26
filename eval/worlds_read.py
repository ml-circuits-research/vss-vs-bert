"""worlds_read.py: the per-intent worlds' reading shared by the CPU rows on saved models (R84 items 1-2): load a version of sop/ by
directory, load the saved worlds of a condition and seed, read an utterance in every world (frame, residual, anchor score, structure
score, margins), and the per-world calibration distributions (the first 200 of the world's own training utterances, R79 / R80) that the
percentiles are taken in. The quantities and the anchor-percentile dispatch are those of eval/r80_dispatch.py (R80), unchanged."""
import importlib, importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
import numpy as np


def load_sop(sop_dir):
    """the 'sop' package under sop_dir as the module 'sop_repo' (so vss/run_reader_real.load_sop_repo returns it); -> (module, SOURCE)"""
    if "sop_repo" not in sys.modules:
        spec = importlib.util.spec_from_file_location("sop_repo", os.path.join(sop_dir, "sop", "__init__.py"), submodule_search_locations=[os.path.join(sop_dir, "sop")])
        m = importlib.util.module_from_spec(spec); sys.modules["sop_repo"] = m; spec.loader.exec_module(m); importlib.import_module("sop_repo.infer")
    return sys.modules["sop_repo"], open(os.path.join(sop_dir, "sop", "SOURCE")).read().strip()[:7]


def ctx_admission(R, rinv, cond, I, ty, share=0.02):
    """R88 (B): per role of world I, the typer's IN-CONTEXT kinds of the gold head tokens of its fillers on the world's positive training
    utterances, kept when they carry >= share of the role's fillers; written into R.schema (roles[r].kinds and head_kinds) at load (the saved
    schema files are not rewritten; no retraining). Returns {question: (stored kinds, new kinds)}."""
    items = cond["items"]; cnt = {}; rmap = {v: k for k, v in rinv.items()}
    for i in cond["train"]:
        it = items[i]
        if it["intent"] != I: continue
        ks = ty.sentence_kinds(it["tokens"]); seen = set()
        for (t, s_, e_, _) in sorted(it["slots"], key=lambda x: x[1]):
            if t in seen or t not in rmap: continue
            seen.add(t); cnt.setdefault(t, {}); k = ks[e_ - 1]; cnt[t][k] = cnt[t].get(k, 0) + 1
    out = {}
    for q, c in cnt.items():
        r = rmap[q]; n = sum(c.values()); new = sorted(k for k, v in c.items() if v >= share * n); rs = R.schema["roles"][r]
        old = sorted(set(rs.get("kinds") or []) | set(rs.get("head_kinds") or [])); rs["kinds"] = list(new)
        if rs.get("head_kinds"): rs["head_kinds"] = list(new)
        out[q] = (old, new)
    return out


def pct(cal, x):
    return float(np.mean(cal < x)) if len(cal) else 0.0


class Worlds:
    def __init__(self, SR, cond, condition, seed, models, kinds=None, decoder="paper", spans=None, ty=None, admission="stored"):
        self.ty = ty   # R88: when given, every reading passes the typer's in-context kinds (token_kinds)
        self.SR = SR; items = cond["items"]; self.items = items
        self.intents = sorted({items[i]["intent"] for i in cond["train"]}); self.W = {}
        for I in self.intents:
            wd = os.path.join(mainrepo.ROOT, "logs", "worlds", f"real_snips_{condition}_intent_{I}_seed{seed}_{models}")
            if kinds is not None:
                saved = {}
                for line in open(os.path.join(wd, "kinds.tsv")):
                    if line.startswith("#") or not line.strip(): continue
                    w, k = line.rstrip("\n").split("\t"); saved[w] = k
                diff = sum(1 for w in set(saved) | set(kinds) if saved.get(w) != kinds.get(w))
                if diff: raise SystemExit(f"world {I}: the re-fitted typer's lexicon differs from the saved kinds.tsv in {diff} words")
            leg = json.load(open(os.path.join(wd, "legend.json"))); rinv = {r["id"]: r["question"][5:] for r in leg["roles"] if r["question"].startswith("slot ")}
            R = SR.infer.Reader(os.path.join(wd, "model_seed")); R.cfg["residual"] = "explained"
            if decoder == "contract": R.cfg.update(importlib.import_module("sop_repo.activate").DECODERS["contract"])   # R84 item 3 / R87
            elif decoder == "presence_query": R.cfg["presence_query"] = True
            if spans: R.legend.reader["spans"] = spans
            if admission == "ctx": ctx_admission(R, rinv, cond, I, ty)
            self.W[I] = {"R": R, "rinv": rinv, "dir": wd, "train": [i for i in cond["train"] if items[i]["intent"] == I][:200]}
        self.cal = {}
        for I, w in self.W.items():
            rows = [self.read_one(I, items[i]["tokens"]) for i in w["train"]]
            self.cal[I] = {q: np.array([r[q] for r in rows]) for q in ("residual", "anchor", "structure", "min_slot_margin")}
            self.cal[I]["thr"] = float(np.quantile(self.cal[I]["residual"], 0.98))

    def read_one(self, I, toks):
        w = self.W[I]; R = w["R"]; rinv = w["rinv"]; SR = self.SR
        tk = {"token_kinds": dict(zip(toks, self.ty.sentence_kinds(toks)))} if self.ty is not None else {}
        wires, f, info = R.formalize(" ".join(toks), 0, SR.infer.idgen("e"), filt=set(toks), **tk); S = info["S"]; nS = np.linalg.norm(S)
        anchor = structure = 0.0
        if nS > 1e-9 and info["kept"]:
            Sn = S / nS; d = f.get("depth", 0) or 0
            rep = np.conj(R.st.op("says.scope")) if "says" in R.legend.operators else 1.0
            K = rep ** d * (np.conj(R.st.op("not")) if f.get("neg") else 1.0)
            anchor = R.best(Sn * K * np.conj(R.st.role("intent")), R.role_cands("intent", set(toks)))[2]
            structure = float(f.get("core", R.structure_score(Sn, K, set(toks))))
        fr = {}
        for r in R.schema["role_order"]:
            fl = f["roles"].get(r)
            if r != "intent" and fl is not None and fl[1] is not None: fr[rinv.get(r, r)] = fl[1]
        sm = [float(f["margins"][r]) for r in R.schema["role_order"] if r != "intent" and r in f.get("margins", {})]
        mm = [float(v) for v in f.get("margins", {}).values()]
        pq = [l.key for l in wires[0].lines if str(getattr(l, "key", "")).startswith("_presence_query.")]
        return {"frame": fr, "status": info["status"], "residual": float(info["residual"]), "anchor": float(anchor), "structure": structure,
                "min_margin": min(mm) if mm else 0.0, "min_slot_margin": min(sm) if sm else 0.0, "unplaced": [info["tokens"][j] for j in info["unplaced"]], "presence_queries": pq}

    def read_all(self, toks):
        outs = {}
        for I in self.W:
            o = self.read_one(I, toks); c = self.cal[I]
            o.update(p_res=pct(c["residual"], o["residual"]), p_anc=pct(c["anchor"], o["anchor"])); outs[I] = o
        return outs

    @staticmethod
    def dispatch_anchor(outs):
        """R80's anchor-percentile rule: max (pct anchor, anchor raw)"""
        return sorted(outs, key=lambda I: (-outs[I]["p_anc"], -outs[I]["anchor"]))[0]
