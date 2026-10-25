"""reader_models.py (R38 / R41, 2026-10-25): the reader of a declared row for one seed, loaded from logs/models/<tag>_seed<s>.pt when it
exists, otherwise retrained exactly as its row (vss/run_reader_basic.py: the same generators, N, batch, lr, gate start) and saved there.
Either way its per-role rule on the frozen split is compared with the stored row (results_<tag>_seed<s>.json): EQUAL or the largest
difference is returned, so that a dump or a check is known to describe the reported model. Rows: 1a 'reader_basic', R14
'reader_basic_cues' (six gates fixed open), R37 'reader_basic_cuesinit' (six gates initialised open, trainable)."""
import os, time
import numpy as np, torch
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import v8_gates
from common import basic_world as bwm, decode, report
from vss.run_reader_basic import GIVEN_OPEN

ROWS = {"1a": "reader_basic", "r14": "reader_basic_cues", "r37": "reader_basic_cuesinit"}

def get_reader(row, seed, device="cuda", D=1024, N=30000, batch=8, lr=0.02):
    tag = ROWS[row]; bw = bwm.build(seed, D); path = os.path.join(mainrepo.ROOT, "logs", "models", f"{tag}_seed{seed}.pt")
    m = v8_gates.V8(bw["w"], D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], device, gate_init="closed"); m.set_F(bw["F"])
    t0 = time.time()
    if os.path.exists(path):
        ck = torch.load(path, map_location=device)
        with torch.no_grad(): m.theta.copy_(ck["theta"].to(device)); m.logits.copy_(ck["logits"].to(device))
        how = f"loaded {os.path.relpath(path, mainrepo.ROOT)}"
    else:
        if row == "r14": m.fix_gates(GIVEN_OPEN, True)
        if row == "r37":
            l1a = torch.load(os.path.join(mainrepo.ROOT, "logs", "models", "reader_basic_seed0.pt"))["logits"].float()
            init = float(torch.median(l1a[torch.sigmoid(l1a) > 0.5])); idx = [m.cats.index(c) for c in GIVEN_OPEN if c in m.cats]
            with torch.no_grad(): m.logits[idx] = init
        m.train(bw["world"], N, templates=v5.TRAIN_T, persons=bw["trP"], objects=bw["trO"], batch=batch, lr=lr, depth_train=bwm.DEPTH_TRAIN)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"theta": m.theta.detach().cpu(), "logits": m.logits.detach().cpu(), "D": D, "seed": seed}, path)
        how = f"retrained in {time.time() - t0:.0f}s and saved to {os.path.relpath(path, mainrepo.ROOT)}"
    split = bwm.load(seed); preds = {ax: decode.decode_frozen(m, bw["world"], split, ax) for ax in report.AXES + ["calibration"]}
    st = __import__("json").load(open(os.path.join(mainrepo.ROOT, "logs", f"results_{tag}_seed{seed}.json")))
    chk = report.evaluate("check", seed, preds, split, f"dumpcheck_{tag}")
    diff = max(abs(chk["axes"][ax]["mean_role"] - st["axes"][ax]["mean_role"]) for ax in report.AXES if st["axes"][ax])
    return m, bw, split, preds, how, ("EQUAL" if diff < 1e-9 else f"largest difference {diff:.4f}")
