"""r10_saved.py: the R10 additions (responses R10 items 1 and 2a, 2026-10-25) for runs whose models were saved before R10 existed, with
no retraining: the one-to-one decoding rule and the depth-3 split by filler position, on the frozen seed-0 split.
  --reader logs/models/reader_basic_seed0.pt          (run 1a; the route book, decoded by the world's addresses)
  --tagger logs/models/bert_<...>_seed0               (run 2a and later; tagger.pt + tokenizer + config.json)
The per-role rule is recomputed as a check: its numbers must equal the stored results of the run (EQUAL is printed). Writes
logs/results_r10saved_<name>_seed0.json (a prefix eval/compare.py does not read as a run row).
Run: ./run.sh eval/r10_saved.py --reader ... | --tagger ... [--device cuda]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode, report
from eval import metrics

def check(row, stored_path):
    st = json.load(open(stored_path)); ok = all(abs(row["axes"][ax]["mean_role"] - st["axes"][ax]["mean_role"]) < 1e-9 for ax in report.AXES if st["axes"][ax])
    ok &= abs(row["axes"]["depth3_unseen"]["full"] - st["axes"]["depth3_unseen"]["full"]) < 1e-9
    print(f"  per-role rule recomputed vs stored {os.path.basename(stored_path)}: {'EQUAL' if ok else 'DIFFERENT'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reader"); ap.add_argument("--tagger"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    AX = report.AXES + ["calibration"]
    if a.reader:
        import v8_gates
        ck = torch.load(a.reader); seed, D = ck["seed"], ck["D"]; split = bwm.load(seed); bw = bwm.build(seed, D)
        m = v8_gates.V8(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(bw["F"])
        with torch.no_grad(): m.theta.copy_(ck["theta"].to(m.theta.device)); m.logits.copy_(ck["logits"].to(m.logits.device))
        preds = {ax: decode.decode_frozen(m, bw["world"], split, ax) for ax in AX}
        row = report.evaluate("VSS reader (reference), saved seed 0, R10 rules", seed, preds, split, "r10saved_reader_basic")
        check(row, os.path.join(mainrepo.ROOT, "logs", f"results_reader_basic_seed{seed}.json")); report.summarize([row], "VSS reader, saved seed 0 (R10)")
    if a.tagger:
        from transformers import AutoTokenizer
        from models import bert_basic as bb
        cfg = json.load(open(os.path.join(a.tagger, "config.json"))); seed = 0; split = bwm.load(seed); bw = bwm.build(seed, split["D"])
        tok = AutoTokenizer.from_pretrained(a.tagger); model = bb.Tagger(cfg["model"], cfg["scratch"], cfg["n_tokens"], cue_tok=tok if cfg.get("cues") == "given" else None).to(a.device)
        model.load_state_dict(torch.load(os.path.join(a.tagger, "tagger.pt"), map_location=a.device))
        preds = {ax: bb.predict(model, tok, split, ax, a.device) for ax in AX}
        pos = bb.subword_positions(tok, split["axes"]["depth3_unseen"]); tmax = bb.train_max_subwords(tok, bb.make_train(bw, seed, 30000))
        name = os.path.basename(a.tagger.rstrip("/")); stored = name[:-len("_seed0")]
        row = report.evaluate(f"{name}, saved seed 0, R10 rules", seed, preds, split, f"r10saved_{stored}", positions=pos, train_max=tmax, unit="sub-token")
        check(row, os.path.join(mainrepo.ROOT, "logs", f"results_{stored}_seed{seed}.json")); report.summarize([row], f"{name} (R10)")
    metrics.checklist(1, "[x] no training [x] seed 0 only (the saved model) [x] per-role rule checked against the stored run")
