"""throughput.py (responses R23, 2026-10-25): the concurrency check of run.sh's slots. Trains BERT-MRC (models/bert_mrc.py's model and
data) for a fixed number of optimizer steps and prints the training samples per second; run once alone and then N copies at once, and
compare the summed throughput (keep N slots if the total is >= 1.3x the single run, else halve N and measure again). No result of any
axis is produced; the model is discarded.
Run: ./run.sh eval/throughput.py [--model bert-base-uncased] [--steps 300] [--batch 32]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import torch, torch.nn as nn
from common import basic_world as bwm
from models.bert_basic import make_train
from models import bert_mrc as bm
from transformers import AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="bert-base-uncased"); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=32); a = ap.parse_args()
    split = bwm.load(0); bw = bwm.build(0, split["D"]); tok = AutoTokenizer.from_pretrained(a.model)
    ex = bm.examples(make_train(bw, 0, 3000), lambda d: ["A", "B", "C", "X1", "X2"]); torch.manual_seed(0); m = bm.MRC(a.model).to("cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5); ce = nn.CrossEntropyLoss(); m.train(); t0 = None
    for step in range(a.steps + 20):
        if step == 20: torch.cuda.synchronize(); t0 = time.time()
        chunk = [ex[(step * a.batch + j) % len(ex)] for j in range(a.batch)]; pairs = [(q, s) for q, s, _ in chunk]; ids, att, tt, spans = bm.encode(tok, pairs, "cuda")
        ys = torch.tensor([0 if k is None else spans[b][k][0] for b, (_, _, k) in enumerate(chunk)], device="cuda"); ye = torch.tensor([0 if k is None else spans[b][k][1] for b, (_, _, k) in enumerate(chunk)], device="cuda")
        s, e = bm.fwd(m, ids, att, tt, spans, pairs); loss = ce(s, ys) + ce(e, ye); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"throughput: {a.model} MRC, batch {a.batch}, {a.steps} steps in {dt:.1f} s: {a.steps * a.batch / dt:.1f} samples/s; peak cuda reserved {torch.cuda.max_memory_reserved()/1e9:.2f} GB; MemAvailable {guard.mem_available_gb() if hasattr(guard, 'mem_available_gb') else 'n/a'} GB")
