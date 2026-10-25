"""cost_symmetric.py (deviation R27 item 4, responses R27, 2026-10-25): the inference cost measured the same way for every saved model, on
the same 500 sentences (the first 500 of seed 0's depth3_unseen axis, depth given as for every model), in fp32:
  CPU at 1 thread and at 20 threads, batch 1 and batch 32: per-sentence latency, median and p95 (batch 32: the batch time / 32 per batch);
  CUDA at batch 32: per-sentence latency, the run's GPU energy coming from the sidecar (eval/power_sampler.sh) over this segment.
Models: --reader logs/models/reader_basic_seed0.pt (the reader, decoded by the world's addresses), --tagger logs/models/<tag>_seed0 (a
saved tagger), --mrc logs/models/<tag>_seed0 (a saved BERT-MRC; the questions of the sentence's depth, and also all six). One model per
flag, several flags allowed. Writes logs/results_cost_<name>.json.
Run: ./run.sh eval/cost_symmetric.py [--reader ...] [--tagger ...] [--mrc ...]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mainrepo
v5, v7 = mainrepo.import_experiment()
import guard
import numpy as np, torch
from common import basic_world as bwm, decode

N = int(os.environ.get("COST_N", 500))

def cells(fn, items, devices=("cpu",)):
    """fn(batch_items, device) runs one batch; returns {cell: {median_ms, p95_ms}}"""
    out = {}
    for dev in devices:
        for threads in ((1, 20) if dev == "cpu" else (None,)):
            if threads: torch.set_num_threads(threads)
            for B in ((1, 32) if dev == "cpu" else (32,)):
                fn(items[:B], dev); ts = []
                for i in range(0, len(items), B):
                    t0 = time.perf_counter(); fn(items[i:i + B], dev)
                    if dev == "cuda": torch.cuda.synchronize()
                    ts.append(1000 * (time.perf_counter() - t0) / len(items[i:i + B]))
                out[f"{dev}{'' if threads is None else f'_{threads}t'}_b{B}"] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.quantile(ts, 0.95)), "t_start": None}
    torch.set_num_threads(20); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reader"); ap.add_argument("--tagger"); ap.add_argument("--mrc"); a = ap.parse_args()
    split = bwm.load(0); bw = bwm.build(0, split["D"]); items = split["axes"]["depth3_unseen"][:N]
    print(f"cost_symmetric: main repository {mainrepo.check_pinned()}; {len(items)} sentences (seed 0, depth3_unseen), fp32; CPU threads 1 / 20, batch 1 / 32; CUDA batch 32")
    res = {}
    if a.reader:
        import v8_gates
        ck = torch.load(a.reader); models = {}
        def mk(dev):
            if dev not in models:
                m = v8_gates.V8(bw["w"], ck["D"], np.random.default_rng(200), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.set_F(bw["F"])
                with torch.no_grad(): m.theta.copy_(ck["theta"].to(m.theta.device)); m.logits.copy_(ck["logits"].to(m.logits.device))
                models[dev] = m
            return models[dev]
        res["reader"] = cells(lambda b, dev: decode.decode_frozen(mk(dev), bw["world"], {"axes": {"x": b}, "candidates": split["candidates"]}, "x"), items, ("cpu", "cuda"))
    if a.tagger:
        from transformers import AutoTokenizer
        from models import bert_basic as bb
        cfg = json.load(open(os.path.join(a.tagger, "config.json"))); tok = AutoTokenizer.from_pretrained(a.tagger); models = {}
        def mk(dev):
            if dev not in models:
                m = bb.Tagger(cfg["model"], cfg["scratch"], cfg["n_tokens"]).to(dev); m.load_state_dict(torch.load(os.path.join(a.tagger, "tagger.pt"), map_location=dev)); models[dev] = m
            return models[dev]
        res[os.path.basename(a.tagger.rstrip("/"))] = cells(lambda b, dev: bb.predict(mk(dev), tok, {"axes": {"x": b}, "candidates": split["candidates"]}, "x", dev, batch=len(b)), items, ("cpu", "cuda"))
    if a.mrc:
        from transformers import AutoTokenizer
        from models import bert_mrc as bm
        cfg = json.load(open(os.path.join(a.mrc, "config.json"))); tok = AutoTokenizer.from_pretrained(a.mrc); models = {}
        if cfg.get("kinds"): bm.KIND_OF = {w_: cfg["kinds"].index(c) + 1 for w_, c in split["cat"].items()}
        def mk(dev):
            if dev not in models:
                m = bm.MRC(cfg["model"], cfg.get("kinds"), cfg.get("scratch", False)).to(dev); m.load_state_dict(torch.load(os.path.join(a.mrc, "mrc.pt"), map_location=dev)); models[dev] = m
            return models[dev]
        name = os.path.basename(a.mrc.rstrip("/"))
        res[name + "_depth_given"] = cells(lambda b, dev: bm.read(mk(dev), tok, split, "depth3_unseen", b, dev), items, ("cpu", "cuda"))
        res[name + "_all_six"] = cells(lambda b, dev: bm.read(mk(dev), tok, split, "depth3_unseen", b, dev, given=False), items, ("cpu", "cuda"))
    for name, c in res.items():
        print(f"  {name}: " + "; ".join(f"{k} median {v['median_ms']:.2f} ms p95 {v['p95_ms']:.2f} ms" for k, v in c.items()))
        with open(os.path.join(mainrepo.ROOT, "logs", f"results_cost_{name}.json"), "w") as f: json.dump({"model": name, "n": len(items), "cells": c}, f)
    print("checklist: [x] the same 500 sentences for every model [x] fp32 [x] four CPU cells and CUDA batch 32 [x] energy from the sidecar over the run (this machine, GB10)")
