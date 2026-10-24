"""run_reader_basic.py (run 1, part 1): the VSS reader (reference: path + chainL + bigram, gates learned from closed, v8_gates.V8 of the
main repository, UNMODIFIED) trained on the basic world exactly as RESULTS 6c / 7b (N 30000, depths 0-2, D 1024, batch 8, lr 0.02) and
decoded on the FROZEN splits of data/basic_seed<s>.json. Exit check of run 1: within seed variance of the main repository's numbers
(RESULTS 6c: new ids 1.000, depth 3 1.000 on the basic world).
Run: python3 vss/run_reader_basic.py [--seeds 5] [--N 30000] [--D 1024] [--device cuda]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard                                   # the main repository's process guard (memory, wall clock); installs itself on import
import numpy as np, torch
import v8_gates
from common import basic_world as bwm, decode, report
from eval import metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); D = a.D
    print(f"run_reader_basic: main repository {mainrepo.check_pinned()}; device {a.device}; D {D}; N {a.N}; batch {a.batch}; lr {a.lr}; seeds {a.seeds}")
    print("regime (rule 24): English basic world, prefixed reporting forms, training depths 0-2 (templates TRAIN_T, 12 persons / 12 objects), tested at depth 3 UNSEEN, 6 held-out persons / objects; frozen splits")
    print("chance: seen ids 1/12 = 0.083; new ids 1/6 = 0.167")
    rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); assert split["D"] == D, f"splits were made with D={split['D']}; rerun data/make_splits.py --D {D}"
        bw = bwm.build(seed, D); w, world, F = bw["w"], bw["world"], bw["F"]
        m = v8_gates.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
        assert m.F_np is world.F
        t0 = time.time(); m.train(world, a.N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=a.batch, lr=a.lr, depth_train=bwm.DEPTH_TRAIN)
        if a.device == "cuda": torch.cuda.synchronize()
        print(f"  seed {seed}: trained in {time.time() - t0:.0f}s; gates OPEN {sorted(c for c, v in m.gates().items() if v > 0.5)}")
        preds = {axis: decode.decode_frozen(m, world, split, axis) for axis in report.AXES + ["calibration"]}
        rows.append(report.evaluate("VSS reader (reference)", seed, preds, split, "reader_basic"))
    report.summarize(rows, "VSS reader (reference)")
    print("reference (main repository, RESULTS 6c / 7b, fresh test draws): new ids 1.000, depth 3 1.000 on all six roles")
    metrics.checklist(a.seeds)
