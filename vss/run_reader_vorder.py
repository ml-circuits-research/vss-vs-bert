"""run_reader_vorder.py (responses R56, declared in RESULTS 'Deviation R56'): the reader 1a + the verb-order route (vss/v8_vorder.py, a
transcription of vss-demo 434fa40), retrained with 1a's seeds, generators, N 30,000, batch 8, lr 0.02, gates learned from the closed
start; the frozen splits; design on seeds 0-4, the claim on seeds 5-9 (tag reader_vorder / reader_vorder_fresh, never pooled).
--check: before any vorder training, the subclass WITHOUT set_vorder is trained on the first seed exactly as 1a and must reproduce the
stored 1a row EQUAL (the run stops otherwise). Reports the per-role rule and the one-to-one rule (report.evaluate) with the original
decoder, the per-cell error counts of the new-formulation rows (design set only), and saves every seed's model (theta, logits, vo) for
eval/sentence_candidates.py (the R46 decoder, residuals, decision categories).
Run: ./run.sh vss/run_reader_vorder.py --seed_start 0|5 --seeds 5 [--check]"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode, report
from eval import metrics
from vss.v8_vorder import V8Vorder, verb_categories

CELLS = {"T4:C": [(4, "C")], "T7:A": [(7, "A")], "T10:A+C": [(10, "A"), (10, "C")], "T4:B+T11:B": [(4, "B"), (11, "B")]}

def build(bw, seed, D, device, vorder):
    m = V8Vorder(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], device, gate_init="closed"); m.set_F(bw["F"])
    vc = m.set_vorder(verb_categories(m.cats)) if vorder else None
    return m, vc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed_start", type=int, default=0); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--N", type=int, default=30000); a = ap.parse_args(); D = 1024
    fresh = a.seed_start >= 5; tag = "reader_vorder_fresh" if fresh else "reader_vorder"; name = "VSS reader 1a + verb-order route (R56)" + (" [fresh splits, seeds 5-9]" if fresh else "")
    print(f"run_reader_vorder (R56): main repository {mainrepo.check_pinned()}; reference implementation vss-demo 434fa40 (transcribed in vss/v8_vorder.py); seeds {a.seed_start}-{a.seed_start + a.seeds - 1} ({'CLAIM SET' if fresh else 'DESIGN SET (inspected)'}); 1a's N / batch / lr")
    if a.check:
        s0 = a.seed_start; split = bwm.load(s0); bw = bwm.build(s0, D); m, _ = build(bw, s0, D, a.device, False)
        m.train(bw["world"], a.N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=8, lr=0.02, depth_train=bwm.DEPTH_TRAIN)
        P = {ax: decode.decode_frozen(m, bw["world"], split, ax) for ax in report.AXES + ["calibration"]}; row = report.evaluate("check: V8Vorder without the route", s0, P, split, "vordercheck")
        st = json.load(open(os.path.join(mainrepo.ROOT, "logs", f"results_{'reader_basic_fresh' if fresh else 'reader_basic'}_seed{s0}.json")))
        diff = max(abs(row["axes"][ax]["mean_role"] - st["axes"][ax]["mean_role"]) for ax in report.AXES if st["axes"][ax])
        print(f"  check (seed {s0}): the subclass without set_vorder vs the stored 1a row: {'EQUAL' if diff < 1e-9 else f'largest difference {diff:.4f}'}", flush=True)
        if diff >= 1e-9: sys.exit("check failed: the subclass does not reproduce 1a; no vorder row is run")
        del m
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        split = bwm.load(seed); bw = bwm.build(seed, D); m, vc = build(bw, seed, D, a.device, True)
        print(f"  seed {seed}: verb categories {vc}; vo {tuple(m.vo.shape)} = {m.vo.numel() / 1e6:.1f} M phases", flush=True)
        t0 = time.time(); m.train(bw["world"], a.N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=8, lr=0.02, depth_train=bwm.DEPTH_TRAIN)
        if a.device == "cuda": torch.cuda.synchronize()
        print(f"  seed {seed}: trained in {time.time() - t0:.0f}s; gates OPEN {sorted(c for c, v in m.gates().items() if v > 0.5)}", flush=True)
        P = {ax: decode.decode_frozen(m, bw["world"], split, ax) for ax in report.AXES + ["calibration"]}
        row = report.evaluate(name, seed, P, split, tag); row["params_vo"] = int(m.vo.numel())
        if not fresh:
            c = collections.Counter()
            for ax in ("new_form_seen_trans", "new_form_unseen_trans", "new_form_role_cues_seen"):
                for it, p in zip(split["axes"][ax], P[ax]):
                    for r in bwm.roles_at(it["depth"]):
                        if p["frame"][r] != it["frame"][r]:
                            c["wrong"] += 1; g = [k for k, v in CELLS.items() if (it["template"], r) in v]; c[g[0] if g else "rest"] += 1
            row["cells_wrong"] = dict(c); print(f"  seed {seed} wrong role decisions on the new-formulation rows (per-role rule): {dict(c)}", flush=True)
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json"), "w") as f: json.dump(row, f)
        os.makedirs(os.path.join(mainrepo.ROOT, "logs", "models"), exist_ok=True)
        torch.save({"theta": m.theta.detach().cpu(), "logits": m.logits.detach().cpu(), "vo": m.vo.detach().cpu(), "D": D, "seed": seed}, os.path.join(mainrepo.ROOT, "logs", "models", f"{tag}_seed{seed}.pt"))
        rows.append(row); del m; torch.cuda.empty_cache() if a.device == "cuda" else None
    report.summarize(rows, name)
    metrics.checklist(len(rows), "[x] the subclass checked EQUAL to 1a without the route [x] 1a's configuration, no selection [x] expected effects written before the run (R56)")
