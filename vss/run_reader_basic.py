"""run_reader_basic.py (run 1, part 1): the VSS reader (reference: path + chainL + bigram, gates learned from closed, v8_gates.V8 of the
main repository, UNMODIFIED) trained on the basic world exactly as RESULTS 6c / 7b (N 30000, depths 0-2, D 1024, batch 8, lr 0.02) and
decoded on the FROZEN splits of data/basic_seed<s>.json. Exit check of run 1: within seed variance of the main repository's numbers
(RESULTS 6c: new ids 1.000, depth 3 1.000 on the basic world).
Run: python3 vss/run_reader_basic.py [--seeds 5] [--N 30000] [--D 1024] [--device cuda]
R14 (responses R14, 2026-10-25): --gates_given cues fixes the gates of the role-cue categories {to, by, from, '.'} and the level openers
{says, that} OPEN (rule 20, v8_gates.fix_gates: logit +8, gradient masked, as v16's --rel_gates open); every other gate is learned from
the closed start as before. Default 'none' = the tagged code. Row 'gates given', results reader_basic_cues_seed<s>.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard                                   # the main repository's process guard (memory, wall clock); installs itself on import
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode, report
from eval import metrics

GIVEN_OPEN = ["to", "by", "from", ".", "says", "that"]   # R14: the same set as models/bert_basic.CUE_WORDS

def reader_cost(m, bw, split, a, D, seed):
    """the reader's size and CPU cost, the same fields as the BERT rows: parameters (learned phases), bytes of the route book at 32 and
    4 bits (the paper's storage), and CPU milliseconds per sentence for encode + states + world decoding, one sentence at a time"""
    n = int(m.theta.numel() + m.logits.numel())
    mc = v8_gates.V8(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cpu", gate_init="closed"); mc.set_F(bw["F"])
    with torch.no_grad(): mc.theta.copy_(m.theta.detach().cpu()); mc.logits.copy_(m.logits.detach().cpu())
    items = split["axes"]["seen"][:a.cost_n]; t0 = time.time()
    for it in items: decode.decode_frozen(mc, bw["world"], {"axes": {"x": [it]}, "candidates": split["candidates"]}, "x")
    return {"params": n, "bytes": n * 4, "bytes_float32": n * 4, "bytes_4bit": n // 2, "cpu_ms_per_sentence": 1000 * (time.time() - t0) / max(1, len(items))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda")
    ap.add_argument("--cost_n", type=int, default=200); ap.add_argument("--save_model", action="store_true", help="save the seed-0 route book to logs/models/ for eval/determinism.py")
    ap.add_argument("--gates_given", default="none", choices=["none", "cues"], help="R14: 'cues' gives the GIVEN_OPEN gates open (row 'gates given')")
    a = ap.parse_args(); D = a.D
    name, tag = ("VSS reader (reference)", "reader_basic") if a.gates_given == "none" else ("VSS reader, gates given (R14)", "reader_basic_cues")
    print(f"run_reader_basic: main repository {mainrepo.check_pinned()}; device {a.device}; D {D}; N {a.N}; batch {a.batch}; lr {a.lr}; seeds {a.seeds}")
    print("regime (rule 24): English basic world, prefixed reporting forms, training depths 0-2 (templates TRAIN_T, 12 persons / 12 objects), tested at depth 3 UNSEEN, 6 held-out persons / objects; frozen splits")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167")
    rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); assert split["D"] == D, f"splits were made with D={split['D']}; rerun data/make_splits.py --D {D}"
        bw = bwm.build(seed, D); w, world, F = bw["w"], bw["world"], bw["F"]
        m = v8_gates.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
        assert m.F_np is world.F
        if a.gates_given == "cues": print(f"  seed {seed}: gates GIVEN open (rule 20, R14): {m.fix_gates(GIVEN_OPEN, True)}")
        t0 = time.time(); m.train(world, a.N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=a.batch, lr=a.lr, depth_train=bwm.DEPTH_TRAIN)
        if a.device == "cuda": torch.cuda.synchronize()
        print(f"  seed {seed}: trained in {time.time() - t0:.0f}s; gates OPEN {sorted(c for c, v in m.gates().items() if v > 0.5)}")
        preds = {axis: decode.decode_frozen(m, world, split, axis) for axis in report.AXES + ["calibration"]}
        row = report.evaluate(name, seed, preds, split, tag)
        row["cost"] = reader_cost(m, bw, split, a, D, seed); c = row["cost"]
        print(f"  seed {seed} cost: {c['params']/1e6:.2f}M parameters (phases), route book {c['bytes_float32']/1e6:.1f} MB at 32 bits ({c['bytes_4bit']/1e6:.1f} MB at 4 bits), {c['cpu_ms_per_sentence']:.2f} ms/sentence on CPU")
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        if a.save_model and seed == 0:
            os.makedirs(os.path.join(mainrepo.ROOT, "logs", "models"), exist_ok=True)
            torch.save({"theta": m.theta.detach().cpu(), "logits": m.logits.detach().cpu(), "D": D, "seed": seed}, os.path.join(mainrepo.ROOT, "logs", "models", "{tag}_seed0.pt"))
        rows.append(row)
    report.summarize(rows, name)
    print("reference (main repository, RESULTS 6c / 7b, fresh test draws): new ids 1.000, depth 3 1.000 on all six roles")
    metrics.checklist(a.seeds)
