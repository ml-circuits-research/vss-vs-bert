"""rule_ceiling.py (responses R30 item 1, 2026-10-25): what the basic world's axes measure, by three models that need no GPU, 5 seeds,
the frozen splits, every axis, depth given (the common report) and depth NOT given (six roles, exact match).
  rule:       a hand-written rule over the GIVEN kinds (P, O) and the words says / claims / that / to / by / from / is and the verbs'
              direction: speakers = the person before each reporting verb, outermost first; inside the last reported clause B = the
              object, C = the person after 'to', A = the person after 'by' or 'from'; a person left over is A, except with a receiving verb
              (receives, buys, gets, borrows, sees, accepts) or a passive subject (a person followed by 'is' + participle), where it is C;
              two persons left over in an active clause (the double object) are A then C. Written from the legend's frames (all 16
              forms of the world are listed in its definition): a ceiling of knowing the grammar, not a learned model.
  positional: per role, the training frequency of (sentence length, position) for that role; the candidate at the most frequent
              position wins (training sentences of the seed only).
  window LR:  a per-token logistic regression on (previous word, the word's kind, next word), trained on the training sentences,
              decoded over the world's candidates as the taggers are (a local-context baseline, not a bag of words over the sentence).
Writes logs/results_ceiling_<model>_seed<s>.json (a prefix eval/compare.py reads into its own table).
Run: ./run.sh eval/rule_ceiling.py [--seeds 5]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode, report
from eval import metrics
from models.bert_basic import make_train

REPORTING = ("says", "claims"); RECEIVING = ("receives", "buys", "gets", "borrows", "sees", "accepts")
PARTICIPLES = ("given", "sold", "sent", "lent", "shown", "offered")

def rule(tokens, cat):
    f = {r: None for r in bwm.ROLES}; sp = [tokens[i - 1] for i in range(1, len(tokens)) if tokens[i] in REPORTING]
    for k, x in enumerate(sp[:3]): f[f"X{k+1}"] = x
    last = max([i for i in range(1, len(tokens)) if tokens[i] == "that" and tokens[i - 1] in REPORTING], default=-1)
    inner = tokens[last + 1:]; P = [i for i, t in enumerate(inner) if cat.get(t) == "P"]; O = [i for i, t in enumerate(inner) if cat.get(t) == "O"]
    if O: f["B"] = inner[O[0]]
    used = set()
    for i in P:
        if i > 0 and inner[i - 1] == "to": f["C"] = inner[i]; used.add(i)
        elif i > 0 and inner[i - 1] in ("by", "from"): f["A"] = inner[i]; used.add(i)
    left = [i for i in P if i not in used]
    recv = any(t in RECEIVING for t in inner)
    passive_subj = lambda i: i + 2 < len(inner) and inner[i + 1] == "is" and inner[i + 2] in PARTICIPLES
    if len(left) == 2 and f["A"] is None and f["C"] is None: f["A"], f["C"] = inner[left[0]], inner[left[1]]
    else:
        for i in left:
            if (recv or passive_subj(i)) and f["C"] is None: f["C"] = inner[i]
            elif f["A"] is None: f["A"] = inner[i]
            elif f["C"] is None: f["C"] = inner[i]
    return f

def as_pred(f, depth):
    return {"frame": {r: (f[r] if r in bwm.roles_at(depth) else None) for r in bwm.ROLES}, "margins": {r: 1.0 for r in bwm.roles_at(depth)}}

class Positional:
    def __init__(self, data):
        self.c = collections.Counter()
        for toks, tags in data:
            for i, t in enumerate(tags):
                if t != "O": self.c[(len(toks), i, t)] += 1
    def predict(self, it, split, axis):
        out = {"frame": {r: None for r in bwm.ROLES}, "margins": {}}
        for r in bwm.roles_at(it["depth"]):
            cands = set(bwm.candidates_for(split, axis, r)); sc = sorted(((self.c[(len(it["tokens"]), i, r)], -i, t) for i, t in enumerate(it["tokens"]) if t in cands), reverse=True)
            if sc: out["frame"][r] = sc[0][2]; tot = sum(x[0] for x in sc) or 1; out["margins"][r] = (sc[0][0] - (sc[1][0] if len(sc) > 1 else 0)) / tot
            else: out["margins"][r] = 0.0
        return out

class WindowLR:
    LAB = ["O"] + bwm.ROLES
    def __init__(self, data, split, epochs=5):
        self.V = {w_: i + 1 for i, w_ in enumerate(sorted(split["vocab"]))}; self.K = {c: i + 1 for i, c in enumerate(sorted(set(split["cat"].values())))}; self.cat = split["cat"]
        X, Y = [], []
        for toks, tags in data:
            for i, t in enumerate(tags): X.append(self.feat(toks, i)); Y.append(self.LAB.index(t))
        X, Y = torch.tensor(X), torch.tensor(Y); n = len(self.V) + 1; self.n = n
        torch.manual_seed(0); self.W = torch.nn.Linear(2 * n + len(self.K) + 1, len(self.LAB)); opt = torch.optim.Adam(self.W.parameters(), lr=0.05)
        for _ in range(epochs):
            for i in range(0, len(X), 4096):
                loss = torch.nn.functional.cross_entropy(self.W(self.onehot(X[i:i + 4096])), Y[i:i + 4096]); opt.zero_grad(); loss.backward(); opt.step()
    def feat(self, toks, i):
        return [self.V.get(toks[i - 1], 0) if i > 0 else 0, self.K.get(self.cat.get(toks[i]), 0), self.V.get(toks[i + 1], 0) if i + 1 < len(toks) else 0]
    def onehot(self, X):
        n = self.n; Z = torch.zeros(len(X), 2 * n + len(self.K) + 1); r = torch.arange(len(X))
        Z[r, X[:, 0]] = 1; Z[r, n + X[:, 2]] = 1; Z[r, 2 * n + X[:, 1]] = 1; return Z
    @torch.no_grad()
    def predict(self, it, split, axis):
        toks = it["tokens"]; p = torch.softmax(self.W(self.onehot(torch.tensor([self.feat(toks, i) for i in range(len(toks))]))), -1).numpy()
        out = {"frame": {r: None for r in bwm.ROLES}, "margins": {}}
        for r in bwm.roles_at(it["depth"]):
            cands = set(bwm.candidates_for(split, axis, r)); pos = [i for i, t in enumerate(toks) if t in cands]
            sc = sorted(((p[i, self.LAB.index(r)], toks[i]) for i in pos), reverse=True)
            if sc: out["frame"][r] = sc[0][1]; out["margins"][r] = float(sc[0][0] - (sc[1][0] if len(sc) > 1 else 0.0))
            else: out["margins"][r] = 0.0
        return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); a = ap.parse_args()
    print(f"rule_ceiling: main repository {mainrepo.check_pinned()}; seeds {a.seeds}; models: rule (given kinds + function words), positional, window LR; CPU")
    print("regime (rule 24): basic world, frozen splits; the rule is written from the legend's 16 forms (a ceiling of knowing the grammar); positional and window LR trained on the seed's 30k training sentences")
    rows = collections.defaultdict(list)
    for seed in range(a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, split["D"]); data = make_train(bw, seed, a.N); cat = split["cat"]
        pos, lr = Positional(data), WindowLR(data, split)
        models = {"rule": lambda it, ax: as_pred(rule(it["tokens"], cat), it["depth"]), "positional": pos.predict, "window_lr": lr.predict}
        for name, fn in models.items():
            preds = {ax: [fn(it, ax) for it in split["axes"][ax]] for ax in report.AXES + ["calibration"]}
            row = report.evaluate(f"ceiling: {name}", seed, preds, split, f"ceiling_{name}")
            if name == "rule":                                       # depth NOT given: the rule's own speaker count, all six roles
                g3 = split["axes"]["depth3_unseen"]; ng = [rule(it["tokens"], cat) for it in g3]
                row["depth_not_given_exact_match"] = {k: float(np.mean([all(f[r] == it["frame"][r] for r in roles) for f, it in zip(ng, g3)])) for k, roles in (("six", bwm.ROLES), ("trained", ["A", "B", "C", "X1", "X2"]), ("inner", ["A", "B", "C"]))}
                row["exact_match_by_axis_depth_not_given"] = {ax: float(np.mean([all(rule(it["tokens"], cat)[r] == it["frame"][r] for r in bwm.ROLES) for it in split["axes"][ax]])) for ax in report.AXES if split["axes"][ax]}
                print(f"  seed {seed} rule, depth NOT given: exact match per axis {row['exact_match_by_axis_depth_not_given']}")
            with open(os.path.join(mainrepo.ROOT, "logs", f"results_ceiling_{name}_seed{seed}.json"), "w") as f: json.dump(row, f)
            rows[name].append(row)
    for name, rs in rows.items(): report.summarize(rs, f"ceiling: {name}")
    ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f} [{' '.join(f'{x:.3f}' for x in xs)}]"
    for ax in rows["rule"][0]["exact_match_by_axis_depth_not_given"]: print(f"  rule, depth NOT given, exact match (six roles) {ax}: {ms([r['exact_match_by_axis_depth_not_given'][ax] for r in rows['rule']])}")
    metrics.checklist(a.seeds, "[x] no GPU [x] the rule written from the legend's forms, never tuned on report rows")
