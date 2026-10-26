"""export_demo.py (responses R84 item 4, R91): the SNIPS demo export of the per-intent configuration. For the chosen seed: every base world's
saved R81 model through `sop export-web` (this repository's sop/, the tree's version, run as a subprocess: python3 -m sop export-web
--model <model_seed> --out <out>/worlds/<intent> --bits 8); the typer once (Typer.export: typer.json + typer.bin), re-fitted exactly as
R81 fitted it (seed 0, n_freq 40, the base condition's training annotation; its lexicon checked equal to every world's kinds.tsv); and
reading.json, the demo rule measured by R90 item 1: dispatch = the anchor percentile (each world's 200 calibration anchors are written);
frame = line C (decoder keys presence_query + presence_by_kind, legend reader.spans kind_run, token_kinds = the typer's in-context kinds,
admission sets recomputed from the in-context kinds >= 2%, written per world and question); refusal = line A's explained residual of the
chosen world (presence_query, kind_run, in-context kinds, admission as stored in the model) above the dev-calibrated 2% threshold (the
pooled threshold R90 measured; per-world 98th percentiles over the dev_cal utterances dispatched to each world are written as a second,
unmeasured field). The saved models are not modified. Run: ./run.sh vss/export_demo.py --seed S --sop_dir DIR --out DIR"""
import argparse, json, os, subprocess, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import worlds_read as wr
from common import mainrepo, real_data as rd
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--sop_dir", required=True, help="the sop version of the reading (R90: 040d44e)")
    ap.add_argument("--out", required=True); ap.add_argument("--bits", type=int, default=8); ap.add_argument("--n_freq", type=int, default=40); a = ap.parse_args()
    SR, src = wr.load_sop(a.sop_dir); tree_src = open(os.path.join(mainrepo.ROOT, "sop", "SOURCE")).read().strip()[:7]
    from vss import run_reader_real as rr
    cond = rd.load_condition("snips", "base"); items = cond["items"]
    kinds, ty, T = rr.build_kinds_typer(cond, types.SimpleNamespace(n_freq=a.n_freq))
    WA = wr.Worlds(SR, cond, "base", a.seed, "r81", kinds, "presence_query", "kind_run", ty, "stored")
    WC = wr.Worlds(SR, cond, "base", a.seed, "r81", kinds, "presence_by_kind", "kind_run", ty, "ctx")
    os.makedirs(os.path.join(a.out, "worlds"), exist_ok=True); os.makedirs(os.path.join(a.out, "typer"), exist_ok=True)
    print(f"export_demo: seed {a.seed}; reading sop @ {src}; export-web and typer export by the tree's sop @ {tree_src}; out {a.out}")
    sizes = {}
    for I in WA.intents:
        md = os.path.join(WA.W[I]["dir"], "model_seed"); od = os.path.join(a.out, "worlds", I)
        r = subprocess.run([sys.executable, "-m", "sop", "export-web", "--model", md, "--out", od, "--bits", str(a.bits)], cwd=mainrepo.ROOT, capture_output=True, text=True)
        print(r.stdout.strip()[-600:]);
        if r.returncode: raise SystemExit(f"export-web failed for {I}: {r.stderr[-2000:]}")
        sizes[I] = {f: os.path.getsize(os.path.join(od, f)) for f in sorted(os.listdir(od))}
        print(f"  world {I}: " + ", ".join(f"{f} {b / 1e6:.2f} MB" for f, b in sizes[I].items()), flush=True)
    tsz = ty.export(os.path.join(a.out, "typer")); meta = json.load(open(os.path.join(a.out, "typer", "typer.json")))
    tbytes = {f: os.path.getsize(os.path.join(a.out, "typer", f)) for f in ("typer.json", "typer.bin")}
    T2 = T.Typer.load_export(os.path.join(a.out, "typer")); chk = [i for i in cond["eval"]["dev_rep"]][:200]
    agree = float(np.mean([ty.sentence_kinds(items[i]["tokens"]) == T2.sentence_kinds(items[i]["tokens"]) for i in chk]))
    print(f"  typer: kept rows {meta['rows']} of {2 ** meta['bits']}; typer.json {tbytes['typer.json'] / 1e3:.1f} kB, typer.bin {tbytes['typer.bin'] / 1e3:.1f} kB; "
          f"re-loaded export reproduces the in-memory typer's sentence kinds on {agree:.3f} of 200 dev_rep utterances", flush=True)
    # the refusal threshold, as R90 measured it (pooled over the dev calibration half), and per world (second field)
    cal = []
    for i in cond["dev_cal"]:
        oa = WA.read_all(items[i]["tokens"]); I = WA.dispatch_anchor(oa); cal.append((I, oa[I]["residual"]))
    thr = float(np.quantile([r_ for _, r_ in cal], 0.98))
    per_world = {I: (float(np.quantile([r_ for J, r_ in cal if J == I], 0.98)) if any(J == I for J, _ in cal) else None) for I in WA.intents}
    adm = {}
    for I in WC.intents:
        R = WC.W[I]["R"]; adm[I] = {r: sorted(R.schema["roles"][r].get("kinds") or []) for r in R.schema["role_order"] if r != "intent"}
    reading = {"source": "vss-vs-bert responses R84 item 4, R90 item 1, R91; RESULTS 'R90 item 1: measured'", "seed": a.seed, "reading_sop": src, "export_sop": tree_src,
               "typer": {"files": ["typer/typer.json", "typer/typer.bin"], "kinds_in_context": "Typer.sentence_kinds(tokens)", "rows_kept": meta["rows"], "bits": meta["bits"]},
               "worlds": {I: f"worlds/{I}" for I in WA.intents},
               "dispatch": {"rule": "anchor percentile: the world whose intent-role cleanup score (the anchor) has the highest percentile among that world's 200 calibration anchors; ties by the raw anchor",
                            "calibration_anchors": {I: [float(x) for x in WA.cal[I]["anchor"]] for I in WA.intents}},
               "frame": {"line": "R88 C", "decoder": {"residual": "explained", "presence_query": True, "presence_by_kind": True}, "legend_reader": {"spans": "kind_run"},
                         "token_kinds": "typer in-context", "admission_kinds": adm, "admission_rule": "per question, the typer's in-context kinds of the gold head tokens of its training fillers, kinds with >= 2% of the fillers"},
               "refusal": {"line": "R88 A", "decoder": {"residual": "explained", "presence_query": True}, "legend_reader": {"spans": "kind_run"}, "token_kinds": "typer in-context",
                           "admission_kinds": "as stored in each world's model (schema)", "rule": "refuse iff the chosen world's explained residual > threshold",
                           "threshold": thr, "threshold_rule": "the 98th percentile of the chosen world's line-A residual over the dev calibration half (350 utterances), pooled over worlds: the measured row (R90)",
                           "per_world_thresholds_unmeasured": per_world},
               "measured": {"dispatch_intent_test": "0.966 +- 0.003 (seeds 0-2)", "span_f1_test": "0.892 +- 0.001", "held_out_refusal_ood_intents": "0.460 +- 0.051 at 0.016 in-domain false rejection"},
               "sizes_bytes": {"worlds": sizes, "typer": tbytes}}
    json.dump(reading, open(os.path.join(a.out, "reading.json"), "w"), indent=1)
    print(f"  reading.json: refusal threshold {thr:.4f} (pooled; per world {', '.join(f'{I} {v:.3f}' for I, v in per_world.items() if v is not None)}); total "
          f"{(sum(sum(v.values()) for v in sizes.values()) + sum(tbytes.values())) / 1e6:.1f} MB in {a.out}")
