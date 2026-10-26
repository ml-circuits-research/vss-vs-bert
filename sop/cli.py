"""
cli.py — the `sop` command (spec 5).

  sop formalize --model <dir> [--model <dir2> ...] [file|-] [-o out.sop] [--tau 0.05] [--json] [--legend legend.json] [--force] [--paragraphs]
                (several --model: the dispatcher by fit — every reader reads, the smallest _residual wins, and fills _world)
  sop check     --legend <[name=]legend.json> [--legend ...] [--legends <dir>] <files...> [--json] [--force]
  sop diff      <a.sop> <b.sop> [--tau 0.05] [--model <dir>] [-o out.sop] [--force] [--json]
  sop stats     --model <dir> [--json]
  sop profile   <world> [--seed 0] [--json]   (the nature of the world: need signals of the routes, diagnostics; activate.py)
  sop train     <world> [...]            (torch; see train.py)
  sop export    [--out dir] [--n N] ...  (the repository's three constructed worlds as world folders; tool/export_worlds.py)
  sop export-web --model <dir> --out <dir> [--bits 8|4|32]   (the files app/src/vss.js loads; export_web.py)
Exit codes: 0 ok; 1 input / validation error; 2 model missing or legend mismatch. Output is deterministic for the same model and input.
"""
import argparse, json, os, re, sys

from . import soplang


def _out(text, path):
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: f.write(text)
        os.replace(tmp, path)
    else: sys.stdout.write(text)


def wires_json(doc):
    lines = [json.dumps({"directives": {n: v for n, v, _ in doc.directives}})]
    for w in doc.wires:
        o = {"id": w.id, "type": w.type}
        for l in w.lines:
            o[l.key] = l.text if (len(l.atoms) <= 1 and l.sep is None) else {"sep": l.sep, "items": [soplang.format_atom(a) for a in l.atoms]}
            if len(l.atoms) == 1 and l.atoms[0].kind == "ref": o[l.key] = {"ref": l.atoms[0].text}
        lines.append(json.dumps(o, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def split_sentences(text, paragraphs=False):
    lines = [l.strip() for l in text.split("\n")]
    if not paragraphs: return [l for l in lines if l]
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", para) if s.strip()]
    return out


def cmd_formalize(a):
    from .infer import Reader, ModelError, formalize_text
    rds = []
    if a.worlds:
        from .infer import readers_of_root
        try: rds = readers_of_root(a.worlds, a.force)
        except ModelError as e: print(f"sop formalize: {e}", file=sys.stderr); return 2
        if not rds: print(f"sop formalize: no world with a model under {a.worlds}", file=sys.stderr); return 2
    for mdir in (a.model or []):
        try: rds.append(Reader(mdir, a.legend if len(a.model) == 1 and not a.worlds else None, a.force))
        except ModelError as e: print(f"sop formalize: {e}", file=sys.stderr); return 2
        except (OSError, ValueError, KeyError) as e: print(f"sop formalize: model unreadable: {e}", file=sys.stderr); return 2
    for rd in rds:
        if a.tau is not None: rd.tau = a.tau
        if a.presence: rd.cfg["presence_readout"] = a.presence
        if a.depth_estimator: rd.cfg["depth_estimator"] = a.depth_estimator
    try: text = sys.stdin.read() if a.input in (None, "-") else open(a.input, encoding="utf-8").read()
    except (OSError, UnicodeDecodeError) as e: print(f"sop formalize: {e}", file=sys.stderr); return 1
    doc, _ = formalize_text(rds if len(rds) > 1 else rds[0], split_sentences(text, a.paragraphs))
    _out(wires_json(doc) if a.json else soplang.write(doc), a.o); return 0


def cmd_check(a):
    from .legend import Legend
    from .check import check_file, legend_name
    specs = []
    for s in a.legend or []:
        name, path = s.split("=", 1) if "=" in s and not os.path.exists(s) else (None, s)
        specs.append((name or legend_name(path), path))
    if a.legends:
        for n in sorted(os.listdir(a.legends)):
            p = os.path.join(a.legends, n, "legend.json")
            if os.path.exists(p) and n not in {x for x, _ in specs}: specs.append((n, p))
    if not specs: print("sop check: give --legend <legend.json> (repeatable, or name=path) or --legends <dir>", file=sys.stderr); return 1
    legends = {}
    for name, path in specs:
        try: lg = Legend(path)
        except (OSError, ValueError) as e: print(f"sop check: cannot read legend {path}: {e}", file=sys.stderr); return 1
        if lg.errors: print(f"legend errors in {path}:\n  " + "\n  ".join(lg.errors), file=sys.stderr); return 1
        legends[name] = lg
    reps = [check_file(p, legends, require_text=not a.no_text, force=a.force)[0] for p in a.files]
    names = ", ".join(f"{n} {lg.sha[:19]}…" for n, lg in legends.items())
    if a.json: print(json.dumps({"legends": {n: lg.sha for n, lg in legends.items()}, "ok": all(r.ok for r in reps), "files": [r.as_json() for r in reps]}, indent=1))
    else:
        for r in reps: print("\n".join(r.lines()))
        print(f"{sum(r.ok for r in reps)}/{len(reps)} files OK against legend(s) {names}")
    return 0 if all(r.ok for r in reps) else 1


def cmd_diff(a):
    from .diff import diff_docs
    docs = []
    for p in (a.a, a.b):
        try: docs.append(soplang.load(p))
        except soplang.SopError as e: print(f"sop diff: {p}:{e.lineno}: {e.msg}", file=sys.stderr); return 1
        except (OSError, UnicodeDecodeError) as e: print(f"sop diff: {e}", file=sys.stderr); return 1
    from .check import legend_hashes
    ha, hb = legend_hashes(docs[0]), legend_hashes(docs[1])
    for k in set(ha) & set(hb) - {"__bad__"}:
        if ha[k][0] != hb[k][0] and not a.force:
            print(f"sop diff: legend mismatch for world {k or '(default)'} ({ha[k][0]} vs {hb[k][0]}); --force to compare anyway", file=sys.stderr); return 2
    tau, legend = a.tau, None
    if a.model:
        try:
            cfg = json.load(open(os.path.join(a.model, "config.json"))); tau = cfg["thresholds"]["tau"] if a.tau is None else a.tau
            from .legend import Legend; legend = Legend(os.path.join(a.model, "legend.json"))
        except (OSError, ValueError, KeyError) as e: print(f"sop diff: model unreadable: {e}", file=sys.stderr); return 2
    tau = 0.05 if tau is None else tau
    out, counts = diff_docs(docs[0], docs[1], a.a, a.b, tau, legend)
    if a.json: _out(wires_json(out), a.o)
    else: _out(soplang.write(out), a.o)
    print("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), file=sys.stderr)
    return 0


def cmd_stats(a):
    p = os.path.join(a.model, "metrics.json")
    if not os.path.exists(p): print(f"sop stats: no metrics.json in {a.model}", file=sys.stderr); return 2
    m = json.load(open(p))
    if a.json: print(json.dumps(m["mean_over_seeds"], indent=1)); return 0
    print(f"world {m['world']}  legend {m['legend'][:23]}…  version {m['legend_version']}  seeds {m['seeds']}  label {m['label']}")
    print("splits: " + ", ".join(f"{k} {v}" for k, v in sorted(m["splits"].items())))
    print("chance: " + ", ".join(f"{k} {v:.3f}" for k, v in m["chance"].items() if v))
    for s, cs in m["mean_over_seeds"].items():
        for c, axes in cs.items():
            print(f"\n{s} — {c}")
            for k, v in axes.items(): print(f"  {k:44s} {v['mean']:.3f}" + (f" ± {v['std']:.3f}" if len(m['seeds']) > 1 else ""))
        cal = m["per_seed"][0][s].get("catalog candidates", {}).get("calibration", {}).get("roles (pooled)")
        if cal: print(f"  calibration roles (pooled), tau {cal['tau']}: coverage {cal['coverage']:.2f}, confident-wrong {cal['confident_wrong']:.3f}, "
                      f"margin mean {cal['margin_mean']:.3f} p10 {cal['margin_p10']:.3f}")
    return 0


def cmd_profile(a):
    """sop profile <world>: activate.profile on the training split (the automatic splits of --seed), no torch needed"""
    from . import activate
    from .legend import Legend
    from .train import load_rows, make_splits, build_schema
    wd = os.path.abspath(a.world); legend = Legend(os.path.join(wd, "legend.json"))
    if legend.errors: print("legend errors:\n  " + "\n  ".join(legend.errors), file=sys.stderr); return 1
    rows, errors, _ = load_rows(wd, legend)
    if errors: print("examples rejected by sop check:\n" + "\n".join(errors[:20]), file=sys.stderr); return 1
    hp = os.path.join(wd, "split_hints.json"); hints = json.load(open(hp)) if os.path.exists(hp) else {}
    make_splits(rows, legend, hints, a.seed, cal_forms=True)
    train_rows = [r for r in rows if r["split"] == "train"]; schema = build_schema(train_rows, legend)
    prof = activate.profile(rows, legend, schema); needs = activate.need_signals(prof); diags = activate.diagnostics(prof, [])
    if a.json: print(json.dumps({"profile": {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) and "rows" in v else v) for k, v in prof.items()},
                                 "need_signals": {c: {"fires": f, "reason": r} for c, (f, r) in needs.items()}, "diagnostics": diags}, indent=1))
    else: sys.stdout.write(activate.profile_markdown(prof, needs, diags))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sop"); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("formalize"); p.add_argument("--model", action="append", help="repeat for the dispatcher by fit (smallest relative _residual wins)")
    p.add_argument("--worlds", help="DATA_ROOT: every <world>/model under it reads each sentence; the dispatcher picks by fit (spec section 9)"); p.add_argument("input", nargs="?"); p.add_argument("-o")
    p.add_argument("--tau", type=float); p.add_argument("--json", action="store_true"); p.add_argument("--legend"); p.add_argument("--force", action="store_true")
    p.add_argument("--paragraphs", action="store_true", help="split paragraphs into sentences (default: one sentence per line)")
    p.add_argument("--presence", choices=["decided", "all"], help="'all' = the v16 PROTOCOL (every role read, no presence decision): a comparison, never the product readout — a role missing from the text must read 'not given'; default: decided")
    p.add_argument("--depth_estimator", choices=["gates", "kinds", "fit"], help="override the model's primary depth estimator")
    p = sub.add_parser("check"); p.add_argument("--legend", action="append", help="legend.json, or name=legend.json; repeatable (one per world)")
    p.add_argument("--legends", help="a folder of worlds: <dir>/<name>/legend.json"); p.add_argument("files", nargs="*"); p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true"); p.add_argument("--no-text", dest="no_text", action="store_true", help="do not require _text on top-level wires")
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--tau", type=float); p.add_argument("--model"); p.add_argument("-o")
    p.add_argument("--force", action="store_true"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("stats"); p.add_argument("--model", required=True); p.add_argument("--json", action="store_true")
    p = sub.add_parser("profile", help="the nature of a world from its training rows: need signals of the candidate routes, diagnostics (no training)")
    p.add_argument("world"); p.add_argument("--seed", type=int, default=0); p.add_argument("--json", action="store_true")
    sub.add_parser("train", add_help=False)
    if argv is None: argv = sys.argv[1:]
    if argv and argv[0] == "train":
        from .train import main as tmain; return tmain(argv[1:])
    if argv and argv[0] == "export-web":        # the model for the browser interpreter app/src/vss.js (VSS Diagram Editor spec section 6)
        from .export_web import main as emain; return emain(argv[1:])
    if argv and argv[0] == "export":            # the repository's constructed worlds as world folders (tool/export_worlds.py)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import export_worlds; return export_worlds.main(argv[1:])
    a = ap.parse_args(argv)
    if a.cmd == "formalize" and not (a.model or a.worlds): ap.error("formalize needs --model <dir> (repeatable) or --worlds <DATA_ROOT>")
    try:
        rc = {"formalize": cmd_formalize, "check": cmd_check, "diff": cmd_diff, "stats": cmd_stats, "profile": cmd_profile}[a.cmd](a); sys.stdout.flush(); return rc
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno()); return 0


if __name__ == "__main__":
    sys.exit(main())
