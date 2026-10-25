"""depth3_saved_reader.py: the two depth-3 numbers of R6 (FULL over the trained labels A B C X1 X2, FULL of the inner frame A B C) for
run 1a, computed from the saved seed-0 route book (logs/models/reader_basic_seed0.pt, --save_model of run 1a) on the frozen seed-0 split,
because run 1a's JSON stores scores, not per-item predictions. Nothing is trained. The six-role FULL is recomputed as a check: it must
equal run 1a's seed-0 value. Run: ./run.sh eval/depth3_saved_reader.py [--device cuda]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode, report
from eval import metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=os.path.join(mainrepo.ROOT, "logs", "models", "reader_basic_seed0.pt")); a = ap.parse_args()
    ck = torch.load(a.model); seed, D = ck["seed"], ck["D"]
    split = bwm.load(seed); bw = bwm.build(seed, D)
    m = v8_gates.V8(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(bw["F"])
    with torch.no_grad(): m.theta.copy_(ck["theta"].to(m.theta.device)); m.logits.copy_(ck["logits"].to(m.logits.device))
    items = split["axes"]["depth3_unseen"]; preds = decode.decode_frozen(m, bw["world"], split, "depth3_unseen")
    full6 = metrics.score_axis(items, preds, bwm.roles_at(3))["full"]
    stored = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_reader_basic_seed{seed}.json")))["axes"]["depth3_unseen"]["full"]
    print(f"saved reader seed {seed}, D {D}, {len(items)} depth-3 sentences")
    print(f"  FULL six roles {full6:.4f} (run 1a stored {stored:.4f}): {'EQUAL' if abs(full6 - stored) < 1e-9 else 'DIFFERENT'}")
    print(f"  FULL trained labels (A B C X1 X2) {metrics.full_over(items, preds, report.TRAINED_D3):.4f}")
    print(f"  FULL inner frame (A B C) {metrics.full_over(items, preds, report.INNER):.4f}")
    print("checklist: [x] same frozen split [x] no training [x] scorer of eval/metrics [x] seed 0 only (the only saved model)")
