"""determinism.py (run 7): the same trained model on the same inputs gives the same frames, twice. For the reader (the seed-0 route book
saved by `vss/run_reader_basic.py --save_model`) and for a tagger (saved by `models/bert_basic.py --save_model`): 500 sentences of the
basic world's frozen splits (the first 250 of 'seen' and of 'depth3_unseen', seed 0) are read in two independent passes, each from a
model loaded FRESH from disk, and on the device and on the CPU; frames must be identical, and the largest margin difference is reported.
Run: python3 eval/determinism.py --reader logs/models/reader_basic_seed0.pt --tagger logs/models/<tag>_seed0 [--device cuda]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode

def items500(split): return split["axes"]["seen"][:250] + split["axes"]["depth3_unseen"][:250]

def reader_pass(path, split, device):
    st = torch.load(path, map_location="cpu"); D = st["D"]; bw = bwm.build(0, D)
    m = v8_gates.V8(bw["w"], D, np.random.default_rng(200), ["path", "chainL", "bigram"], device, gate_init="closed"); m.set_F(bw["F"])
    with torch.no_grad(): m.theta.copy_(st["theta"].to(device)); m.logits.copy_(st["logits"].to(device))
    sp = {"axes": {"x": items500(split)}, "candidates": split["candidates"]}; return decode.decode_frozen(m, bw["world"], sp, "x")

def tagger_pass(md, split, device):
    from transformers import AutoTokenizer
    from models.bert_basic import Tagger, predict
    cfg = json.load(open(os.path.join(md, "config.json"))); tok = AutoTokenizer.from_pretrained(md)
    model = Tagger(cfg["model"], True, cfg["n_tokens"]).to(device); model.load_state_dict(torch.load(os.path.join(md, "tagger.pt"), map_location=device))
    return predict(model, tok, {"axes": {"x": items500(split)}, "candidates": split["candidates"]}, "x", device)

def compare(name, runs):
    a = runs[0]; ok = True; dm = 0.0
    for k, b in enumerate(runs[1:], 1):
        same = all(p["frame"] == q["frame"] for p, q in zip(a, b)); ok &= same
        dm = max(dm, max(abs(p["margins"].get(r, 0) - q["margins"].get(r, 0)) for p, q in zip(a, b) for r in p["margins"]))
        print(f"  {name}: pass 0 vs pass {k}: frames identical on {sum(p['frame'] == q['frame'] for p, q in zip(a, b))} / {len(a)}")
    print(f"  {name}: {'DETERMINISTIC (frames identical in every pass)' if ok else 'NOT identical'}; largest margin difference {dm:.2e}")
    return ok, dm

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reader"); ap.add_argument("--tagger"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); split = bwm.load(0); out = {}
    devs = [a.device, a.device] + (["cpu"] if a.device != "cpu" else [])
    print(f"determinism: main repository {mainrepo.check_pinned()}; 500 sentences (seen 250 + depth3_unseen 250, seed 0); passes on {devs}, each from a fresh load")
    if a.reader: out["reader"] = compare("VSS reader", [reader_pass(a.reader, split, d) for d in devs])
    if a.tagger: out["tagger"] = compare(f"tagger {os.path.basename(a.tagger)}", [tagger_pass(a.tagger, split, d) for d in devs])
    json.dump({k: {"identical": v[0], "max_margin_diff": v[1]} for k, v in out.items()}, open(os.path.join(mainrepo.ROOT, "logs", "results_determinism.json"), "w"))
