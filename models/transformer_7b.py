"""transformer_7b.py (run 1, part 2): the paper's control reproduced from the main repository's own classes, UNMODIFIED
(v9_transformer.TModel = RESULTS 7b; v22_transformer_categories.TModelCat = RESULTS 7b-bis, the transformer also given the word categories),
same tuning grid (lr {1e-3, 3e-4} x dropout {0.0, 0.1}, selected on seed 0 by training-distribution validation cosine), 5 seeds, decoded on
the FROZEN splits. Exit check of run 1: within seed variance of RESULTS 7b-bis (new ids 0.253±0.034, depth 3 0.393±0.023) and 7b
(0.215, 0.390); otherwise stop and report before any BERT run.
Run: python3 models/transformer_7b.py [--variant cat|plain] [--seeds 5] [--N 30000] [--epochs 12] [--D 1024] [--device cuda]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from v9_transformer import TModel
from v22_transformer_categories import TModelCat
from common import basic_world as bwm, decode, report
from eval import metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cat", choices=["cat", "plain"]); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lrs", default="1e-3,3e-4"); ap.add_argument("--dropouts", default="0.0,0.1")
    ap.add_argument("--save_model", action="store_true", help="R11 (2026-10-25): save every seed's network to logs/models/ (for the R10 one-to-one rule)")
    ap.add_argument("--tag_suffix", default="", help="appended to the results file name and the model name (R11: the re-run of 1b / 1c with the pre-registered configuration)")
    a = ap.parse_args(); D = a.D; name = ("transformer 7b-bis (categories given)" if a.variant == "cat" else "transformer 7b") + (f" ({a.tag_suffix.strip('_').replace('_', ' ')})" if a.tag_suffix else "")
    print(f"transformer_7b: {name}; main repository {mainrepo.check_pinned()}; device {a.device}; D {D}; N {a.N}; epochs {a.epochs}; seeds {a.seeds}")
    print("regime (rule 24): as RESULTS 7b / 7b-bis: same world, same fixed identities F at the input, objective -cos(S, S_world), decoded with the world's addresses on the frozen splits")
    grid = [(float(lr), float(dp)) for lr in a.lrs.split(",") for dp in a.dropouts.split(",")]; chosen = None; rows = []
    for seed in range(a.seeds):
        split = bwm.load(seed); assert split["D"] == D
        bw = bwm.build(seed, D); w, world, F = bw["w"], bw["world"], bw["F"]; best = None
        for lr, dp in (grid if seed == 0 else [chosen]):
            M = TModelCat if a.variant == "cat" else TModel
            m = M(w, D, a.device, seed, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, d_ff=a.d_ff, dropout=dp); m.set_F(F); assert m.F_np is world.F
            data = m.make_data(world, np.random.default_rng(200 + seed), a.N, v5.TRAIN_T, bw["trP"], bw["trO"], bwm.DEPTH_TRAIN)
            val = m.make_data(world, np.random.default_rng(900 + seed), 2000, v5.TRAIN_T, bw["trP"], bw["trO"], bwm.DEPTH_TRAIN)
            t0 = time.time(); m.train(data, lr, a.epochs, a.batch)
            if a.device == "cuda": torch.cuda.synchronize()
            vc = m.val_cos(val); print(f"  seed {seed} lr={lr:g} dropout={dp:g} ({time.time() - t0:.0f}s, {m.n_params/1e6:.2f}M params) validation cos {vc:.3f}", flush=True)
            if best is None or vc > best[0]: best = (vc, lr, dp, m)
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, dropout={chosen[1]:g} (by validation cos, seed 0)")
        m = best[3]; preds = {axis: decode.decode_frozen(m, world, split, axis) for axis in report.AXES + ["calibration"]}
        rows.append(report.evaluate(name, seed, preds, split, f"transformer7b_{a.variant}{a.tag_suffix}"))
        if a.save_model:
            md = os.path.join(mainrepo.ROOT, "logs", "models"); os.makedirs(md, exist_ok=True)
            torch.save({"net": m.net.state_dict(), "variant": a.variant, "seed": seed, "D": D, "lr": best[1], "dropout": best[2],
                        "kw": dict(d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, d_ff=a.d_ff, dropout=best[2])}, os.path.join(md, f"transformer7b_{a.variant}{a.tag_suffix}_seed{seed}.pt"))
    report.summarize(rows, name)
    print("reference (main repository): 7b new ids 0.215±0.012, depth 3 0.390±0.024; 7b-bis new ids 0.253±0.034, depth 3 0.393±0.023 (fresh test draws)")
    metrics.checklist(a.seeds)
